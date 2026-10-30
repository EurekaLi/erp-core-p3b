"""固定受试者划分上的三模型对照；验证组早停，smoke完全不预测外层测试组。"""

from __future__ import annotations

import argparse
import gc
import os
import random
import re
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from erp_core_p3b.cbramod_audit import model_provenance
from erp_core_p3b.classification_io import (
    canonical_hash,
    new_artifact_run,
    resolve_path,
    verify_artifact,
    write_json,
)
from erp_core_p3b.deep_learning_data import load_dl_dataset
from erp_core_p3b.deep_learning_models import build_model
from erp_core_p3b.logistic_baseline import binary_metrics, subject_metrics, summarize_subjects

MODEL_NAMES = ("eegnet", "cbramod_linear", "cbramod_finetune")


class SignalDataset(Dataset):
    """逐trial复制只读memmap切片；不原地缩放缓存，不建立全数据GPU副本。

    EEGNet使用原生V/100µV，CBraMod缓存已完成相同单位缩放。这里没有按trial标准化。
    """

    def __init__(
        self, arrays: dict[str, np.ndarray], indices: np.ndarray, name: str, scale_v: float
    ):
        self.signals = arrays["native" if name == "eegnet" else "cbramod"]
        self.labels, self.indices, self.name, self.scale_v = (
            arrays["labels"],
            indices,
            name,
            scale_v,
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.indices[index]
        signal = np.array(self.signals[row], dtype=np.float32, copy=True)
        if self.name == "eegnet":
            signal /= self.scale_v
            signal = signal[None, ...]
        return torch.from_numpy(signal), torch.tensor(int(self.labels[row]), dtype=torch.long)


def seed_process(seed: int, deterministic: bool, threads: int) -> None:
    """固定本进程随机性；不承诺跨PyTorch版本/GPU硬件逐位一致。"""
    if deterministic:
        # 必须在首次CUDA上下文/矩阵运算前设置；不改变系统CUDA安装或Conda环境。
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def class_weights(labels: np.ndarray) -> torch.Tensor:
    """N/(2*N_k)只由该折训练标签计算；不再叠加加权采样器。"""
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("训练标签必须同时包含Standard和Target")
    counts = np.bincount(labels, minlength=2)
    return torch.tensor(len(labels) / (2 * counts), dtype=torch.float32)


def make_optimizer(model: nn.Module, name: str, config: dict[str, Any]):
    if name == "eegnet":
        options = config["eegnet"]
        return torch.optim.AdamW(
            model.parameters(), lr=options["learning_rate"], weight_decay=options["weight_decay"]
        )
    options = config["cbramod"]
    groups = [{"params": model.head.parameters(), "lr": options["head_learning_rate"]}]
    if name == "cbramod_finetune":
        groups.append(
            {
                "params": [p for p in model.encoder.parameters() if p.requires_grad],
                "lr": options["encoder_learning_rate"],
            }
        )
    return torch.optim.AdamW(groups, weight_decay=options["weight_decay"])


def _loader(arrays, indices, name, scale, batch_size, *, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    # Windows下不启动子进程复制memmap；每个batch只复制当前需要的trial。
    return DataLoader(
        SignalDataset(arrays, indices, name, scale),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        generator=generator,
    )


def _predict(model, loader, device, weights):
    model.eval()
    scores, numerator, denominator = [], 0.0, 0.0
    with torch.inference_mode():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            if not torch.isfinite(logits).all():
                raise RuntimeError("验证/测试logits含NaN/Inf")
            numerator += float(
                nn.functional.cross_entropy(logits, labels, weight=weights, reduction="sum")
            )
            denominator += float(weights[labels].sum())
            scores.append(logits.softmax(dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(scores), numerator / denominator


def _prediction_frame(trials, indices, scores, name, fold):
    frame = trials.iloc[indices][["trial_id", "subject_id", "label"]].copy()
    frame["score"], frame["experiment"], frame["fold"] = scores, name, fold
    return frame.reset_index(drop=True)


def _fit_fold(model, name, fold, arrays, trials, indices, config, device, output, seed):
    train_options = config["training"]
    batch_size = int(config["eegnet" if name == "eegnet" else "cbramod"]["batch_size"])
    scale = config["model_scale_v"]
    train_loader = _loader(
        arrays, indices["train"], name, scale, batch_size, shuffle=True, seed=seed
    )
    validation_loader = _loader(
        arrays, indices["validation"], name, scale, batch_size, shuffle=False, seed=seed
    )
    weights = class_weights(arrays["labels"][indices["train"]]).to(device)
    model.to(device)
    optimizer = make_optimizer(model, name, config)
    best, best_epoch, stale, history = -float("inf"), 0, 0, []
    path = output / f"{name}_fold-{fold:02d}"
    path.mkdir()
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, int(train_options["effective_epochs"]) + 1):
        model.train()
        numerator, denominator = 0.0, 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = nn.functional.cross_entropy(logits, labels, weight=weights)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise RuntimeError("训练loss/logits含NaN/Inf")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(train_options["gradient_clip_norm"]),
                error_if_nonfinite=True,
            )
            optimizer.step()
            if name == "eegnet":
                model.constrain_weights()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise RuntimeError("参数更新后含NaN/Inf")
            batch_weight = float(weights[labels].sum())
            numerator += float(loss.detach()) * batch_weight
            denominator += batch_weight
        scores, validation_loss = _predict(model, validation_loader, device, weights)
        validation = _prediction_frame(trials, indices["validation"], scores, name, fold)
        subject_ba = subject_metrics(validation, train_options["prediction_threshold"])[
            "balanced_accuracy"
        ]
        if not np.isfinite(subject_ba).all():
            raise ValueError("某名验证受试者BA未定义，不能跳过该人后早停")
        macro = float(subject_ba.mean())
        if not np.isfinite(macro):
            raise ValueError("验证受试者BA未定义，不能可靠早停")
        history.append(
            {
                "epoch": epoch,
                "train_weighted_loss": numerator / denominator,
                "validation_weighted_loss": validation_loss,
                "validation_subject_macro_ba": macro,
            }
        )
        pd.DataFrame(history).to_csv(path / "history.csv", index=False)
        print(
            f"{name} fold={fold} epoch={epoch}: loss={numerator / denominator:.4f} "
            f"val_macro_BA={macro:.4f}",
            flush=True,
        )
        if macro > best + train_options["improvement_tolerance"]:
            best, best_epoch, stale = macro, epoch, 0
            # CPU独立副本避免state_dict引用随下一次optimizer.step改变；不保存不可信对象。
            torch.save(
                {
                    "state_dict": {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    },
                    "epoch": epoch,
                    "validation_subject_macro_ba": macro,
                    "seed": seed,
                    "experiment": name,
                    "fold": fold,
                },
                path / "best_model.pt",
            )
        else:
            stale += 1
        if stale >= train_options["effective_patience"]:
            break
    best_state = torch.load(path / "best_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(best_state["state_dict"], strict=True)
    scores, _ = _predict(model, validation_loader, device, weights)
    validation = _prediction_frame(trials, indices["validation"], scores, name, fold)
    validation.to_csv(path / "validation_predictions.csv", index=False)
    audit = {
        "experiment": name,
        "fold": fold,
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_validation_subject_macro_ba": best,
        "class_weights_standard_target": weights.cpu().tolist(),
        "train_trials": len(indices["train"]),
        "validation_trials": len(indices["validation"]),
        "batch_size": batch_size,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "elapsed_seconds": time.perf_counter() - start,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device)
        if device.type == "cuda"
        else None,
    }
    write_json(path / "fit_audit.json", audit)
    return validation, audit, weights


def _plots(output, audits):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for audit in audits:
        name, fold = audit["experiment"], audit["fold"]
        history = pd.read_csv(output / f"{name}_fold-{fold:02d}/history.csv")
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
        axes[0].plot(history["epoch"], history["train_weighted_loss"], label="Train")
        axes[0].plot(history["epoch"], history["validation_weighted_loss"], label="Validation")
        axes[0].set(xlabel="Epoch", ylabel="Weighted cross entropy")
        axes[0].legend()
        axes[1].plot(history["epoch"], history["validation_subject_macro_ba"])
        axes[1].axvline(audit["best_epoch"], color="black", linestyle="--")
        axes[1].set(xlabel="Epoch", ylabel="Validation subject-macro BA", ylim=(0, 1))
        fig.savefig(output / f"{name}_fold-{fold:02d}/learning_curve.png", dpi=130)
        plt.close(fig)


def train_deep_learning(
    config_path: Path,
    dataset_dir: Path,
    root: Path,
    *,
    mode: str = "smoke",
    models: Sequence[str] | None = None,
    seed: int | None = None,
    device: str = "cuda",
    run_name: str | None = None,
    output: Path | None = None,
    adaptation_review_note: str | None = None,
    allow_nondeterministic: bool = False,
) -> Path:
    """仅消费12的完整缓存；覆盖已有结果被禁止，失败运行保留用于诊断。

    full外层OOF是一次正式评估。不能看测试分数后用同一测试组反复选择参数/种子。
    """
    if mode not in ("smoke", "full"):
        raise ValueError("mode须为smoke或full")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    names = list(config["models"] if models is None else models)
    if not names or len(set(names)) != len(names) or any(name not in MODEL_NAMES for name in names):
        raise ValueError("models必须是非重复的已支持模型列表")
    trials, manifest, splits, arrays, cache_summary = load_dl_dataset(dataset_dir)
    if not cache_summary["short_window_limitations_acknowledged"]:
        raise ValueError("尚未接受单patch短窗限制")
    if cache_summary["adaptation_review_trials"] and (
        not adaptation_review_note or len(adaptation_review_note.strip()) < 10
    ):
        raise ValueError(
            "适配有振幅提示：先看复核图，再提供至少10字符的--adaptation-review-note；不自动删trial"
        )
    if device not in ("cuda", "cpu"):
        raise ValueError("device须为cuda或cpu")
    options = config["training"]
    seed = int(options["seed"] if seed is None else seed)
    deterministic = bool(options["deterministic"]) and not allow_nondeterministic
    seed_process(seed, deterministic, int(options["cpu_threads"]))
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用；不会静默退回CPU，请检查环境")
    device_object = torch.device(device)
    config["model_scale_v"] = manifest["input_protocol"]["model_scale_v"]
    options.update(
        effective_epochs=int(options["smoke_epochs" if mode == "smoke" else "max_epochs"]),
        effective_patience=int(options["smoke_patience" if mode == "smoke" else "patience"]),
    )
    if (
        min(options["effective_epochs"], options["effective_patience"], int(options["cpu_threads"]))
        < 1
    ):
        raise ValueError("epochs/patience/threads必须为正数")
    if (
        not 0 <= options["prediction_threshold"] <= 1
        or options["improvement_tolerance"] < 0
        or options["gradient_clip_norm"] <= 0
    ):
        raise ValueError("阈值/早停容差/梯度裁剪无效")
    provenance = None
    if any(name.startswith("cbramod") for name in names):
        provenance = model_provenance(root, manifest["model_config"])
        expected = manifest["model_resources"]
        for key in ("source_commit", "checkpoint_sha256", "source_files_sha256"):
            if provenance.get(key) != expected.get(key):
                raise ValueError(f"CBraMod资源改变：{key}，需要重新审计及导出")
    lr_reference = None
    reference_path = resolve_path(root, config["lr_reference_directory"])
    if reference_path.exists():
        reference = verify_artifact(reference_path, "logistic_baseline")
        if (
            reference["dataset_hash"] != manifest["dataset_hash"]
            or reference["split_hash"] != manifest["split_hash"]
        ):
            raise ValueError("LR对照不属于同一数据与划分版本")
        lr_reference = str(reference_path)
    run_name = run_name or f"{mode}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-seed-{seed}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_name):
        raise ValueError("run-name只能包含字母、数字、下划线、连字符")
    output = resolve_path(root, output or Path(config["outputs"]["directory"]) / run_name)
    with new_artifact_run(root, output, "deep_learning_training") as summary:
        effective = {
            "config": config,
            "mode": mode,
            "models": names,
            "seed": seed,
            "device": device,
            "deterministic": deterministic,
            "adaptation_review_note": adaptation_review_note,
            "dataset_hash": manifest["dataset_hash"],
            "split_hash": manifest["split_hash"],
            "input_protocol_hash": manifest["input_protocol_hash"],
        }
        write_json(output / "config_snapshot.json", effective)
        write_json(output / "subject_splits.json", splits)
        write_json(output / "input_protocol.json", manifest["input_protocol"])
        write_json(output / "model_resources.json", provenance)
        validation_frames, test_frames, audits = [], [], []
        folds = splits["folds"][:1] if mode == "smoke" else splits["folds"]
        trial_lookup = pd.Series(np.arange(len(trials)), index=trials["trial_id"])
        for name in names:
            for fold in folds:
                number = int(fold["fold"])
                fold_seed = seed + number
                seed_process(fold_seed, deterministic, int(options["cpu_threads"]))
                indices = {
                    part: trial_lookup.loc[fold["trial_ids"][part]].to_numpy(dtype=int)
                    for part in ("train", "validation", "test")
                }
                model = build_model(
                    name,
                    len(manifest["input_protocol"]["eeg_channel_names"]),
                    len(manifest["input_protocol"]["native_times_s"]),
                    config,
                    provenance,
                )
                validation, audit, weights = _fit_fold(
                    model,
                    name,
                    number,
                    arrays,
                    trials,
                    indices,
                    config,
                    device_object,
                    output,
                    fold_seed,
                )
                validation_frames.append(validation)
                audits.append(audit)
                if mode == "full":
                    # 测试loader仅在正式模式、验证早停完成并恢复best checkpoint后创建。
                    test_loader = _loader(
                        arrays,
                        indices["test"],
                        name,
                        config["model_scale_v"],
                        audit["batch_size"],
                        shuffle=False,
                        seed=fold_seed,
                    )
                    scores, _ = _predict(model, test_loader, device_object, weights)
                    test_frames.append(
                        _prediction_frame(trials, indices["test"], scores, name, number)
                    )
                del model, weights
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
        validation = pd.concat(validation_frames, ignore_index=True)
        validation.to_csv(output / "validation_predictions.csv", index=False)
        subject_metrics(validation, options["prediction_threshold"]).to_csv(
            output / "validation_subject_metrics.csv", index=False
        )
        pd.DataFrame(audits).to_csv(output / "fit_audit.csv", index=False)
        report = [
            "# 深度学习实验",
            "",
            f"模式：{mode}；模型：{', '.join(names)}；种子：{seed}。",
            "",
            "早停依据：验证受试者等权balanced accuracy。阈值固定0.5（默认），非测试组优化。",
            "原生EEGNet与适配CBraMod不是完全相同输入；差异同时包含输入适配和架构。",
            "CBraMod只有一个时间patch，不证明跨patch时间建模优势。",
            "全部预处理为离线场景；不是严格在线或完全未见受试者的预处理协议。",
        ]
        if mode == "full":
            predictions = pd.concat(test_frames, ignore_index=True)
            for name, group in predictions.groupby("experiment"):
                if group["trial_id"].duplicated().any() or set(group["trial_id"]) != set(
                    trials["trial_id"]
                ):
                    raise ValueError(f"{name}: OOF没有恰好覆盖每个trial一次")
            predictions.to_csv(output / "oof_predictions.csv", index=False)
            metrics = subject_metrics(predictions, options["prediction_threshold"])
            metrics.to_csv(output / "subject_metrics.csv", index=False)
            macro, paired = summarize_subjects(metrics, config["statistics"])
            macro.to_csv(output / "subject_macro_summary.csv", index=False)
            paired.to_csv(output / "paired_model_comparisons.csv", index=False)
            pooled = [
                {
                    "experiment": name,
                    **binary_metrics(
                        group["label"].to_numpy(),
                        group["score"].to_numpy(),
                        options["prediction_threshold"],
                    ),
                }
                for name, group in predictions.groupby("experiment")
            ]
            pd.DataFrame(pooled).to_csv(output / "pooled_metrics.csv", index=False)
            report += [
                "",
                "主结果是39人等权指标，pooled仅辅助；CI按受试者bootstrap，不包含重新拟合不确定性。",
                "模型间配对比较为描述性；不得观察OOF后挑参数、挑种子并仍把它称为未见测试集。",
            ]
            for row in macro[macro["metric"] == "balanced_accuracy"].itertuples():
                report.append(
                    f"{row.experiment}: macro BA={row.mean:.4f}, "
                    f"{100 * config['statistics']['confidence_level']:g}% CI "
                    f"[{row.ci_low:.4f}, {row.ci_high:.4f}]。"
                )
        else:
            report += ["", "smoke只训练第一折并评价验证组；没有外层测试预测或正式性能结论。"]
        _plots(output, audits)
        (output / "training_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        summary.update(
            mode=mode,
            models=names,
            seed=seed,
            device=device,
            deterministic=deterministic,
            dataset_hash=manifest["dataset_hash"],
            split_hash=manifest["split_hash"],
            input_protocol_hash=manifest["input_protocol_hash"],
            effective_config_hash=canonical_hash(effective),
            fit_count=len(audits),
            test_evaluated=mode == "full",
            lr_reference_directory=lr_reference,
            adaptation_review_trials=cache_summary["adaptation_review_trials"],
            adaptation_review_note=adaptation_review_note,
            num_workers=0,
            amp=False,
            torch_version=str(torch.__version__),
            torch_cuda_version=torch.version.cuda,
            gpu_name=torch.cuda.get_device_name(device_object) if device == "cuda" else None,
            input_dataset_directory=str(dataset_dir.resolve()),
        )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path, default=Path("configs/deep_learning_training.yaml"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/processed/deep_learning/v1"))
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--adaptation-review-note")
    parser.add_argument("--allow-nondeterministic", action="store_true")
    args = parser.parse_args(argv)
    try:
        output = train_deep_learning(
            resolve_path(args.project_root, args.config),
            resolve_path(args.project_root, args.dataset_dir),
            args.project_root,
            mode=args.mode,
            models=args.models,
            seed=args.seed,
            device=args.device,
            run_name=args.run_name,
            output=args.output_dir,
            adaptation_review_note=args.adaptation_review_note,
            allow_nondeterministic=args.allow_nondeterministic,
        )
    except Exception as error:
        print(f"DL training failed: {error}", file=sys.stderr)
        return 1
    print(f"Deep learning run: {output}")
    return 0
