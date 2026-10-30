"""审计裁决版短EEG到CBraMod候选输入的身份、单位、时间网格和工程兼容性。

使用刺激零点锚定的多相重采样，绝不把MNE伏特直接当成预训练模型尺度。
GPU测试只更新临时模型并随进程丢弃，不拟合分类实验、不保存模型、不改原FIF。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from fractions import Fraction
from importlib import metadata
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
import psutil
import yaml
from mne._fiff.constants import FIFF
from scipy.signal import resample_poly
from threadpoolctl import threadpool_limits

EVENT_ID = {"Target": 1, "Standard": 2}


def sha256_file(path: Path) -> str:
    """分块校验，避免为了记录来源将整份FIF额外载入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_epoch_header(
    epochs: mne.BaseEpochs, subject: str, config: dict[str, Any]
) -> tuple[list[int], pd.DataFrame]:
    """验证已有保留trial身份和SI单位；不能在适配期间重新筛trial或重排通道。"""
    if epochs.event_id != EVENT_ID:
        raise ValueError(f"{subject}: Target/Standard事件映射不符合冻结协议")
    if not np.isin(epochs.events[:, 2], list(EVENT_ID.values())).all():
        raise ValueError(f"{subject}: 存在未知事件")
    picks = list(mne.pick_types(epochs.info, eeg=True, exclude=[]))
    eog = mne.pick_types(epochs.info, eog=True, exclude=[])
    if len(picks) != config["expected_eeg_channels"] or len(eog) != config["expected_eog_channels"]:
        raise ValueError(f"{subject}: EEG/EOG通道数不符合协议")
    if epochs.info["bads"]:
        raise ValueError(f"{subject}: 仍有未裁决坏道，不能偷偷忽略这些通道")
    for index in picks:
        channel = epochs.info["chs"][index]
        if channel["unit"] != FIFF.FIFF_UNIT_V or channel["unit_mul"] != 0:
            raise ValueError(f"{subject}: EEG通道单位不符合MNE物理伏特约定")
    for key, expected in (
        ("sfreq", "expected_sfreq_hz"),
        ("highpass", "expected_highpass_hz"),
        ("lowpass", "expected_lowpass_hz"),
    ):
        if not np.isclose(epochs.info[key], config[expected], rtol=0, atol=1e-6):
            raise ValueError(f"{subject}: {key}不符合裁决版ERP参数")
    if len(epochs.times) != config["expected_time_points"]:
        raise ValueError(f"{subject}: 时间点数改变")
    if epochs.baseline is None or not np.allclose(
        epochs.baseline, config["expected_baseline_s"], rtol=0, atol=1e-6
    ):
        raise ValueError(f"{subject}: 既有电压baseline不符合协议")
    frame = epochs.metadata
    required = {"subject", "trial_index", "condition_label", "event_sample"}
    if frame is None or not required.issubset(frame.columns) or len(frame) != len(epochs):
        raise ValueError(f"{subject}: 缺少可靠trial元数据")
    if not frame["subject"].eq(subject).all():
        raise ValueError(f"{subject}: 元数据受试者身份错配")
    trials = pd.to_numeric(frame["trial_index"], errors="raise").to_numpy(dtype=float)
    if (
        not np.isfinite(trials).all()
        or (trials < 1).any()
        or not np.equal(trials, np.floor(trials)).all()
    ):
        raise ValueError(f"{subject}: trial_index必须是正整数")
    if frame["trial_index"].duplicated().any():
        raise ValueError(f"{subject}: 重复trial身份")
    expected_labels = np.where(epochs.events[:, 2] == EVENT_ID["Target"], "Target", "Standard")
    if not np.array_equal(frame["condition_label"].to_numpy(), expected_labels):
        raise ValueError(f"{subject}: 条件标签与真实事件不一致")
    if not np.array_equal(frame["event_sample"].to_numpy(), epochs.events[:, 0]):
        raise ValueError(f"{subject}: 事件sample与元数据不一致")
    if "accuracy" in frame and not frame["accuracy"].eq("correct").all():
        raise ValueError(f"{subject}: 非正确反应trial进入冻结名单")
    if (np.diff(epochs.events[:, 0]) <= 0).any():
        raise ValueError(f"{subject}: 事件顺序/唯一性异常")
    labels = (epochs.events[:, 2] == EVENT_ID["Target"]).astype(np.int64)
    if len(np.unique(labels)) != 2:
        raise ValueError(f"{subject}: 某个条件为空")
    identities = pd.DataFrame(
        {
            "trial_id": [f"{subject}:trial-{int(value):04d}" for value in trials],
            "subject_id": subject,
            "trial_index": trials.astype(int),
            "epoch_index": np.arange(len(epochs)),
            "event_sample_256hz": epochs.events[:, 0],
            "event_code": epochs.events[:, 2],
            "condition": expected_labels,
            "label": labels,
        }
    )
    return picks, identities


def adapt_signal(
    data_v: np.ndarray, native_times: np.ndarray, native_sfreq: float, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """返回float64伏特及零点对齐时间轴；反射只服务重采样，不冒充真实长窗。

    原起点-51/256不能直接变成200Hz整数网格。先在左侧反射13点使零点索引64
    能被降采样分母32整除，再做25/32重采样并选取原记录范围内的目标采样点。
    方法依据：https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html
    """
    data = np.asarray(data_v, dtype=np.float64)
    times = np.asarray(native_times, dtype=float)
    if data.ndim != 3 or data.shape[-1] != len(times) or len(times) < 3:
        raise ValueError("信号必须是trial×EEG通道×时间，且与时间轴一致")
    if not np.isfinite(data).all() or not np.isfinite(times).all():
        raise ValueError("原EEG/时间轴包含NaN或Inf")
    if native_sfreq <= 0 or not np.allclose(np.diff(times), 1 / native_sfreq, atol=1e-10):
        raise ValueError("原时间轴不是声明采样率的均匀网格")
    target_sfreq = float(config["target_sfreq_hz"])
    if target_sfreq <= 0:
        raise ValueError("目标采样率必须为正")
    ratio = Fraction(target_sfreq / native_sfreq).limit_denominator(10000)
    if not np.isclose(float(ratio), target_sfreq / native_sfreq, rtol=0, atol=1e-12):
        raise ValueError("无法建立精确的有理重采样比例")
    zero_index = int(round(-times[0] * native_sfreq))
    if zero_index < 0 or zero_index >= len(times) or abs(times[zero_index]) > 1e-10:
        raise ValueError("原epoch必须包含准确的刺激零点，禁止凭数组位置猜测")
    points = int(config["points_per_patch"])
    start = float(config["start_time_s"])
    if points != 200 or target_sfreq != 200:
        raise ValueError("本审计仅支持作者预训练的200Hz、200点patch")
    target_ticks = np.arange(points) + int(round(start * target_sfreq))
    if abs(start * target_sfreq - round(start * target_sfreq)) > 1e-10:
        raise ValueError("候选起点不在以刺激零点为锚的目标网格")
    target_times = target_ticks / target_sfreq
    if target_times[0] < times[0] - 1e-10 or target_times[-1] > times[-1] + 1e-10:
        raise ValueError("候选点超出真实短epoch，禁止外推为额外信号")
    if not (target_ticks == 0).any():
        raise ValueError("候选网格必须包含刺激零点")

    left_pad = (-zero_index) % ratio.denominator
    padded = np.pad(data, ((0, 0), (0, 0), (left_pad, 0)), mode="reflect")
    resampled = resample_poly(
        padded,
        ratio.numerator,
        ratio.denominator,
        axis=-1,
        window=("kaiser", float(config["kaiser_beta"])),
        padtype=config["resample_padtype"],
    )
    resampled_zero = (zero_index + left_pad) * ratio.numerator // ratio.denominator
    selected_indices = resampled_zero + target_ticks
    if selected_indices.min() < 0 or selected_indices.max() >= resampled.shape[-1]:
        raise ValueError("目标采样点无法在重采样结果中定位")
    adapted = resampled[:, :, selected_indices]
    if not np.isfinite(adapted).all():
        raise ValueError("重采样产生NaN/Inf")
    return (
        adapted,
        target_times,
        {
            "up": ratio.numerator,
            "down": ratio.denominator,
            "left_reflection_samples": int(left_pad),
            "padtype": config["resample_padtype"],
            "native_zero_index": zero_index,
            "target_zero_index": int(np.flatnonzero(target_ticks == 0)[0]),
            "native_start_s": float(times[0]),
            "native_end_s": float(times[-1]),
            "target_start_s": float(target_times[0]),
            "target_end_s": float(target_times[-1]),
            "target_points": points,
            "stimulus_alignment_error_s": 0.0,
            "edge_support": "reflected_short_epoch_not_observed_long_context",
        },
    )


def model_scaled_input(data_v: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, float]:
    """MNE物理伏特除以100µV后转float32；不重新baseline、不按trial做z-score。"""
    scale = float(config["model_scale_v"])
    if not np.isfinite(scale) or not np.isclose(scale, 1e-4, rtol=0, atol=1e-12):
        raise ValueError("本审计的模型幅值尺度必须是作者采用的100µV（0.0001V）")
    scaled = np.asarray(data_v / scale, dtype=np.float32)[:, :, np.newaxis, :]
    error_uv = float(np.max(np.abs(scaled[:, :, 0].astype(float) * scale - data_v)) * 1e6)
    if not np.isfinite(scaled).all() or error_uv > config["float32_max_error_uv"]:
        raise ValueError("模型尺度转换产生非有限值或float32精度误差超限")
    return scaled, error_uv


def _inventory(root: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    dataset = config["dataset"]
    input_root = root / dataset["epochs_root"]
    manifest = json.loads((root / dataset["phase5_manifest"]).read_text(encoding="utf-8"))
    files = sorted(input_root.glob("sub-*_task-P3_epo.fif"))
    subjects = [path.name.split("_", 1)[0] for path in files]
    if len(files) != dataset["expected_subject_count"] or set(subjects) != set(
        manifest["subjects"]
    ):
        raise ValueError("当前FIF名单与Phase 5主分析不一致")
    if set(subjects) & set(dataset["excluded_subjects"]):
        raise ValueError("被排除受试者进入主分析目录")
    expected = {row["subject"]: row for row in manifest["input_files"]}
    records = []
    canonical_channels = None
    for subject, path in zip(subjects, files, strict=True):
        epochs = mne.read_epochs(path, preload=False, verbose="ERROR")
        picks, identities = validate_epoch_header(epochs, subject, dataset)
        channels = [epochs.ch_names[index] for index in picks]
        if canonical_channels is None:
            canonical_channels = channels
        if channels != canonical_channels:
            raise ValueError(f"{subject}: EEG顺序与其他受试者不一致")
        counts = identities["label"].value_counts().to_dict()
        if (
            counts[1] != expected[subject]["target_trials"]
            or counts[0] != expected[subject]["standard_trials"]
        ):
            raise ValueError(f"{subject}: 保留trial数与Phase 5不一致")
        records.append(
            {
                "subject": subject,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "trials": len(identities),
                "target": int(counts[1]),
                "standard": int(counts[0]),
                "eeg_channel_order": channels,
                "size_bytes": path.stat().st_size,
            }
        )
    for field, key in (
        ("trials", "expected_trial_count"),
        ("target", "expected_target_count"),
        ("standard", "expected_standard_count"),
    ):
        if sum(record[field] for record in records) != dataset[key]:
            raise ValueError(f"全体{field}数量不符合冻结名单")
    # 数据身份不包含机器路径，复制到另一台机器不会凭空产生不同的科学数据版本。
    identities = [{key: value for key, value in row.items() if key != "path"} for row in records]
    digest = hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()
    return records, digest


def model_provenance(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """记录已下载资源；safe.directory只作用于已明确授权的本次git只读调用。"""
    source = (root / config["source_root"]).resolve()
    checkpoint = (root / config["checkpoint"]).resolve()
    if not (source / "models" / "cbramod.py").is_file() or not checkpoint.is_file():
        raise FileNotFoundError("缺少作者代码或权重；请先完成下载，不会自动联网补齐")
    prefix = ["git", "-c", f"safe.directory={source.as_posix()}", "-C", str(source)]
    revision = subprocess.check_output([*prefix, "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        [*prefix, "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if revision != config["source_commit"] or dirty:
        raise ValueError("作者代码版本改变或有本地修改，需先单独复核并更新资源配置")
    weight_hash = sha256_file(checkpoint)
    if weight_hash != config["checkpoint_sha256"]:
        raise ValueError("权重SHA-256不符合固定资源版本")
    download_record = (
        checkpoint.parent / ".cache/huggingface/download/pretrained_weights.pth.metadata"
    )
    recorded_revision = None
    if download_record.is_file():
        recorded_revision = download_record.read_text(encoding="utf-8").splitlines()[0]
        if recorded_revision != config["weight_revision"]:
            raise ValueError("本地HF下载元数据的revision不符合资源配置")
    return {
        "source_root": str(source),
        "source_commit": revision,
        "tracked_changes": dirty,
        "source_files_sha256": {
            str(path.relative_to(source)): sha256_file(path)
            for path in sorted((source / "models").rglob("*.py"))
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": weight_hash,
        "expected_weight_revision": config["weight_revision"],
        "locally_recorded_weight_revision": recorded_revision,
        "license_note": "upstream code MIT; weight hub metadata Apache-2.0; record separately",
    }


class ModelProbe:
    """临时分类头分别验证冻结编码器/全量梯度；权重不落盘、不输出准确率。"""

    def __init__(self, provenance: dict[str, Any], config: dict[str, Any], device_name: str):
        import torch

        self.torch = torch
        if config["probe_batch_size"] < 2 or config["probe_batch_size"] % 2:
            raise ValueError("工程检查batch size须为至少2的偶数，以覆盖两类")
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用；禁止悄悄回退CPU并宣称GPU通过")
        self.device = torch.device(device_name)
        source = Path(provenance["source_root"])
        # 作者包名models较通用；拒绝重用来自其他项目的同名模块。
        existing = sys.modules.get("models")
        if existing is not None:
            paths = list(getattr(existing, "__path__", []))
            if str(source / "models") not in paths:
                raise RuntimeError("进程已有其他models包，请在独立进程运行审计")
        sys.path.insert(0, str(source))
        module = importlib.import_module("models.cbramod")
        if Path(module.__file__).resolve() != source / "models/cbramod.py":
            raise RuntimeError("导入的CBraMod不是配置中的作者实现")
        torch.manual_seed(int(config["random_state"]))
        self.backbone = module.CBraMod()
        state = torch.load(provenance["checkpoint"], map_location="cpu", weights_only=True)
        self.backbone.load_state_dict(state, strict=True)
        self.backbone.proj_out = torch.nn.Identity()
        self.backbone.to(self.device)
        self.config = config
        self.environment = {
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "device": device_name,
            "strict_checkpoint_loading": True,
            "gpu": torch.cuda.get_device_name(self.device) if device_name == "cuda" else None,
        }

    def run(self, inputs: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
        torch = self.torch
        indices = np.concatenate(
            [
                np.flatnonzero(labels == label)[: max(1, self.config["probe_batch_size"] // 2)]
                for label in (0, 1)
            ]
        )
        x = torch.from_numpy(np.ascontiguousarray(inputs[indices])).to(self.device)
        y = torch.from_numpy(labels[indices].astype(np.int64)).to(self.device)
        head = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(int(np.prod(inputs.shape[1:])), 2),
        ).to(self.device)
        results: dict[str, Any] = {
            "probe_epoch_indices": indices.tolist(),
            "input_shape": list(x.shape),
        }
        for frozen in (True, False):
            for name, parameter in self.backbone.named_parameters():
                parameter.requires_grad_(not frozen and not name.endswith("mask_encoding"))
            self.backbone.eval() if frozen else self.backbone.train()
            parameters = [
                p
                for p in list(self.backbone.parameters()) + list(head.parameters())
                if p.requires_grad
            ]
            optimizer = torch.optim.AdamW(parameters, lr=self.config["probe_learning_rate"])
            self.backbone.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
                torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            features = self.backbone(x)
            if features.shape != x.shape:
                raise ValueError(f"真实EEG编码器输出维度异常: {tuple(features.shape)}")
            logits = head(features)
            loss = torch.nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(features).all() or not torch.isfinite(loss):
                raise ValueError("真实EEG前向产生非有限特征或损失")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                raise ValueError("反向传播缺少梯度或产生NaN/Inf")
            if frozen and any(p.grad is not None for p in self.backbone.parameters()):
                raise ValueError("冻结阶段编码器仍收到梯度")
            if not frozen and not any(
                p.grad is not None and torch.count_nonzero(p.grad).item() > 0
                for p in self.backbone.parameters()
            ):
                raise ValueError("全量阶段编码器没有非零梯度")
            before = head[-1].weight.detach().clone()
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in parameters):
                raise ValueError("优化器更新后出现NaN/Inf参数")
            if torch.equal(before, head[-1].weight.detach()):
                raise ValueError("分类头权重没有更新")
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            results["frozen" if frozen else "full_gradient"] = {
                "loss_engineering_only": float(loss.detach().cpu()),
                "feature_shape": list(features.shape),
                "logits_shape": list(logits.shape),
                "finite_gradients": True,
                "head_updated": True,
                "elapsed_s": time.perf_counter() - started,
                "peak_allocated_mib": torch.cuda.max_memory_allocated(self.device) / 2**20
                if self.device.type == "cuda"
                else None,
                "peak_reserved_mib": torch.cuda.max_memory_reserved(self.device) / 2**20
                if self.device.type == "cuda"
                else None,
            }
        self.backbone.zero_grad(set_to_none=True)
        return results


def _figures(
    output: Path,
    subject: str,
    names: list[str],
    native: np.ndarray,
    adapted: np.ndarray,
    times: np.ndarray,
    target_times: np.ndarray,
    identities: pd.DataFrame,
    plot_channel: str,
) -> None:
    # 审计从任何工作目录启动时，字体/配置缓存都留在本次可写输出内。
    cache = output.parent / ".matplotlib-cache"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache.resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    channel = names.index(plot_channel)
    labels = identities["label"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for axis, label in zip(axes, (1, 0), strict=True):
        selected = labels == label
        axis.plot(times, native[selected, channel].mean(axis=0) * 1e6, label="Native 256 Hz")
        axis.plot(
            target_times,
            adapted[selected, channel].mean(axis=0) * 1e6,
            "--",
            label="Adapted 200 Hz",
        )
        axis.axvline(0, color="black", linestyle=":")
        axis.axvspan(0.3, 0.6, color="grey", alpha=0.15)
        condition = "Target" if label else "Standard"
        axis.set(
            title=f"{subject} {plot_channel} {condition} n={selected.sum()}",
            xlabel="Time from stimulus (s)",
            ylabel="Potential (uV)",
        )
        axis.legend()
    fig.savefig(output / f"{subject}_erp_overlay.png", dpi=140)
    plt.close(fig)
    strongest = int(np.ptp(native, axis=-1).max(axis=1).argmax())
    chosen = list(
        dict.fromkeys(
            [
                int(np.flatnonzero(labels == 1)[0]),
                int(np.flatnonzero(labels == 0)[0]),
                strongest,
            ]
        )
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, index in zip(axes.flat, chosen, strict=False):
        worst_channel = int(np.ptp(native[index], axis=-1).argmax())
        axis.plot(times, native[index, worst_channel] * 1e6, label="Native")
        axis.plot(target_times, adapted[index, worst_channel] * 1e6, "--", label="Adapted")
        axis.axvline(0, color="black", linestyle=":")
        axis.set(
            title=f"{identities.iloc[index]['trial_id']} | {names[worst_channel]}",
            xlabel="Time (s)",
            ylabel="uV",
        )
        axis.legend()
    for axis in list(axes.flat)[len(chosen) : 3]:
        axis.set_visible(False)
    axis = axes.flat[-1]
    axis.hist(np.ptp(native, axis=-1).max(axis=1) * 1e6, bins=20, alpha=0.5, label="Native")
    axis.hist(np.ptp(adapted, axis=-1).max(axis=1) * 1e6, bins=20, alpha=0.5, label="Adapted")
    axis.set(title="Per-trial maximum EEG peak-to-peak", xlabel="uV", ylabel="Trials")
    axis.legend()
    fig.savefig(output / f"{subject}_trials_and_amplitude.png", dpi=140)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(11, 7), constrained_layout=True)
    image = axis.imshow(
        adapted[strongest] * 1e6,
        aspect="auto",
        origin="lower",
        # imshow的extent是像素外缘；半个采样间隔保证零点线落在正确采样中心。
        extent=[target_times[0] - 0.0025, target_times[-1] + 0.0025, -0.5, len(names) - 0.5],
        cmap="RdBu_r",
    )
    axis.set_yticks(np.arange(len(names)), names, fontsize=8)
    axis.axvline(0, color="black", linestyle=":")
    axis.set(title=f"{subject}: largest native PTP trial; channel order audit", xlabel="Time (s)")
    fig.colorbar(image, ax=axis, label="Potential (uV)")
    fig.savefig(output / f"{subject}_channel_heatmap.png", dpi=140)
    plt.close(fig)


def _subject_audit(
    record: dict[str, Any],
    config: dict[str, Any],
    output: Path,
    probe: ModelProbe | None,
    save_inputs: bool,
) -> dict[str, Any]:
    path = Path(record["path"])
    epochs = mne.read_epochs(path, preload=True, verbose="ERROR")
    picks, identities = validate_epoch_header(epochs, record["subject"], config["dataset"])
    native = epochs.get_data(picks=picks, copy=True)
    reference_error = float(np.abs(native.mean(axis=1)).max() * 1e6)
    if reference_error > config["adaptation"]["mean_reference_tolerance_uv"]:
        raise ValueError("最终平均参考残差超限；不要在审计中偷偷重参考")
    adapted, target_times, alignment = adapt_signal(
        native, epochs.times, epochs.info["sfreq"], config["adaptation"]
    )
    inputs, cast_error = model_scaled_input(adapted, config["adaptation"])
    names = [epochs.ch_names[index] for index in picks]
    identities["source_file"] = str(path)
    identities["native_max_ptp_uv"] = np.ptp(native, axis=-1).max(axis=1) * 1e6
    identities["adapted_max_ptp_uv"] = np.ptp(adapted, axis=-1).max(axis=1) * 1e6
    identities["adapted_max_abs_uv"] = np.abs(adapted).max(axis=(1, 2)) * 1e6
    # 复用200µV仅定位变化，不重新执行trial剔除；目标网格本身也是候选。
    identities["adaptation_review_flag"] = (
        identities["adapted_max_ptp_uv"] > config["adaptation"]["review_peak_to_peak_uv"]
    )
    channel_audit = pd.DataFrame(
        {
            "channel_index": np.arange(len(names)),
            "channel": names,
            "native_rms_uv": np.sqrt(np.mean(native**2, axis=(0, 2))) * 1e6,
            "adapted_rms_uv": np.sqrt(np.mean(adapted**2, axis=(0, 2))) * 1e6,
            "native_ptp_median_uv": np.median(np.ptp(native, axis=-1), axis=0) * 1e6,
            "adapted_ptp_median_uv": np.median(np.ptp(adapted, axis=-1), axis=0) * 1e6,
            "native_exactly_constant_trials": (np.ptp(native, axis=-1) == 0).sum(axis=0),
        }
    )
    channel_audit.to_csv(output / f"{record['subject']}_channel_audit.csv", index=False)
    identities.to_csv(output / f"{record['subject']}_trial_audit.csv", index=False)
    _figures(
        output / "figures",
        record["subject"],
        names,
        native,
        adapted,
        epochs.times,
        target_times,
        identities,
        config["adaptation"]["plot_channel"],
    )
    model_result = None if probe is None else probe.run(inputs, identities["label"].to_numpy())
    if save_inputs:
        # 只保存审计缓存，metadata中的反应/事件码不混入特征张量。
        np.savez_compressed(
            output / f"{record['subject']}_audit_inputs.npz",
            native_eeg_v=native.astype(np.float32),
            model_input=inputs,
            native_times_s=epochs.times,
            model_times_s=target_times,
            labels=identities["label"].to_numpy(),
            trial_ids=identities["trial_id"].to_numpy(dtype=str),
            eeg_channel_names=np.asarray(names),
        )
    if sha256_file(path) != record["sha256"]:
        raise ValueError("运行期间原FIF改变，结果不得与冻结名单混用")
    return {
        "subject": record["subject"],
        "trials": len(epochs),
        "target_trials": int(identities["label"].sum()),
        "standard_trials": int((identities["label"] == 0).sum()),
        "native_shape": list(native.shape),
        "model_shape": list(inputs.shape),
        "eeg_channel_order": names,
        "source_sha256": record["sha256"],
        "stimulus_alignment": alignment,
        "model_scale_v": config["adaptation"]["model_scale_v"],
        "float32_roundtrip_max_error_uv": cast_error,
        "mean_reference_max_error_uv": reference_error,
        "native_ptp_max_uv": float(identities["native_max_ptp_uv"].max()),
        "adapted_ptp_max_uv": float(identities["adapted_max_ptp_uv"].max()),
        "adaptation_review_trials": int(identities["adaptation_review_flag"].sum()),
        "native_constant_channel_trials": int(
            channel_audit["native_exactly_constant_trials"].sum()
        ),
        "adapted_above_pretraining_scale_trials": int(
            (identities["adapted_max_abs_uv"] > config["adaptation"]["model_scale_v"] * 1e6).sum()
        ),
        "model_abs_p99": float(np.quantile(np.abs(inputs), 0.99)),
        "model_abs_max": float(np.abs(inputs).max()),
        "native_float32_mib": native.size * 4 / 2**20,
        "model_float32_mib": inputs.nbytes / 2**20,
        "rss_at_subject_end_mib_not_peak": psutil.Process().memory_info().rss / 2**20,
        "model_probe": model_result,
        "audit_inputs_saved": save_inputs,
    }


def _report(output: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# CBraMod真实EEG适配审计",
        "",
        f"技术状态：{summary['technical_status']}",
        f"执行模式：{summary['mode']}；计算设备：{summary['requested_device']}。",
        "",
        "本报告不是模型性能评估。图形仍需人工复核；没有保存训练权重、自动删trial或修改FIF。",
        "",
        "## 输入与边界",
        "",
        "Standard=0、Target=1；只有30个EEG通道进入模型，事件/反应字段仅留作审计。",
        "候选网格为200Hz、200点、[-0.195,0.800]秒，刺激0位于索引39。",
        "幅值为物理伏特除以100µV；保留原来的0.1–30Hz和ERP电压baseline，不做trial z-score。",
        "短窗重采样使用反射边界，不等同于新增真实长窗。一个patch没有跨patch时间关系。",
        "上游对每名受试者全记录进行ICA/QC；这是离线受试者级清洗，不是严格在线盲测流程。",
        "",
        "## 逐人结果",
        "",
        "|受试者|Target/Standard|适配PTP最大µV|适配提示trial|模型最大绝对值|",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["subjects"]:
        subject = row["subject"]
        lines.append(
            f"|{subject}|{row['target_trials']}/{row['standard_trials']}|"
            f"{row['adapted_ptp_max_uv']:.3f}|{row['adaptation_review_trials']}|"
            f"{row['model_abs_max']:.3f}|"
        )
    lines.extend(
        [
            "",
            "## 怎么看图",
            "",
            "1. ERP overlay：看Pz两条件适配前后的形状、幅值及零点是否一致。",
            "2. Trials and amplitude：查看首个两类trial和原最大PTP trial；直方图比较PTP变化。",
            "3. Channel heatmap：核对30通道顺序与异常是否集中；不是头皮地形图。",
            "4. 提示仅用于人工查看，不产生新的排除名单；通过技术检查后再决定是否冻结输入。",
            "",
        ]
    )
    for row in summary["subjects"]:
        subject = row["subject"]
        for suffix in ("erp_overlay", "trials_and_amplitude", "channel_heatmap"):
            lines.append(f"![{subject} {suffix}](figures/{subject}_{suffix}.png)")
    lines.extend(
        [
            "",
            "## 后续工作",
            "",
            "人工复核完成后另行确认候选网格，再实现固定分组划分、全体缓存和正式训练。",
            "不要从本轮临时损失、pilot受试者或全体ERP显著图挑超参数或挑电极。",
            "资源、软件版本和完整结果见同目录run_summary.json、input_inventory.json。",
        ]
    )
    (output / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(
    config_path: Path,
    *,
    project_root: Path,
    signal_only: bool = False,
    device: str = "cuda",
    run_name: str | None = None,
    save_inputs: bool = False,
) -> Path:
    """默认只处理三人；每次独立目录，失败保存部分诊断而不覆盖上次输出。"""
    config_path = config_path.resolve()
    root = project_root.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    name = run_name or datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("run-name仅允许字母、数字、下划线和连字符")
    output = root / config["outputs"]["results_root"] / name
    output.mkdir(parents=True, exist_ok=False)
    (output / "figures").mkdir()
    summary: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "technical_status": "in_progress",
        "scientific_status": "candidate_requires_visual_review_and_downstream_evaluation",
        "mode": "signal_only" if signal_only else "signal_and_model",
        "requested_device": "not_used" if signal_only else device,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "subjects": [],
        "preprocessing_refit": False,
        "automatic_trial_removal": False,
        "classification_results": False,
        "cross_patch_temporal_attention_degenerate": True,
        "software": {
            "python": platform.python_version(),
            **{
                package: metadata.version(package)
                for package in ("mne", "numpy", "scipy", "pandas", "matplotlib", "PyYAML")
            },
        },
    }
    _write_json(output / "run_summary.json", summary)
    _write_json(output / "audit_config.json", config)
    try:
        inventory, inventory_hash = _inventory(root, config)
        summary["all_subjects_inventory_sha256"] = inventory_hash
        summary["all_subjects_count"] = len(inventory)
        summary["all_subjects_trial_count"] = sum(row["trials"] for row in inventory)
        _write_json(output / "input_inventory.json", inventory)
        selected = config["dataset"]["pilot_subjects"]
        if (
            not selected
            or len(set(selected)) != len(selected)
            or not set(selected).issubset({row["subject"] for row in inventory})
        ):
            raise ValueError("pilot名单重复或不在正式数据中")
        probe = None
        if not signal_only:
            provenance = model_provenance(root, config["model"])
            summary["model_resources"] = provenance
            probe = ModelProbe(provenance, config["model"], device)
            summary["model_environment"] = probe.environment
        with threadpool_limits(limits=1):
            for record in inventory:
                if record["subject"] not in selected:
                    continue
                summary["active_subject"] = record["subject"]
                _write_json(output / "run_summary.json", summary)
                print(f"Auditing {record['subject']} ...", flush=True)
                summary["subjects"].append(
                    _subject_audit(record, config, output, probe, save_inputs)
                )
                _write_json(output / "run_summary.json", summary)
                gc.collect()
        summary["technical_status"] = "passed"
        summary.pop("active_subject", None)
        summary["gpu_probe_complete"] = not signal_only and device == "cuda"
        summary["visual_review_status"] = "pending"
        summary["eligible_for_formal_training"] = False
        summary["warnings"] = [
            "single time patch cannot test cross-patch temporal modeling",
            "ERP filter and baseline differ from the foundation model pretraining distribution",
            "reflection padding is numerical edge support, not observed neighboring EEG",
            "engineering loss is not a validation/test performance metric",
        ]
        _write_json(output / "run_summary.json", summary)
        _report(output, summary)
    except Exception as error:
        summary["technical_status"] = "failed"
        summary["error"] = f"{type(error).__name__}: {error}"
        _write_json(output / "run_summary.json", summary)
        raise
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path, default=Path("configs/cbramod_audit.yaml"))
    parser.add_argument("--signal-only", action="store_true", help="不导入torch、不读取模型权重")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--run-name", help="独立结果目录名；已存在时拒绝覆盖")
    parser.add_argument(
        "--save-inputs", action="store_true", help="保存三人审计NPZ，不是正式全体数据集"
    )
    args = parser.parse_args(argv)
    config = args.config if args.config.is_absolute() else args.project_root / args.config
    try:
        output = run_audit(
            config,
            project_root=args.project_root,
            signal_only=args.signal_only,
            device=args.device,
            run_name=args.run_name,
            save_inputs=args.save_inputs,
        )
    except Exception as error:
        print(f"Audit failed: {error}", file=sys.stderr)
        return 1
    print(f"Audit output: {output.resolve()}")
    print("Technical checks passed; visual review and formal evaluation are still pending.")
    return 0
