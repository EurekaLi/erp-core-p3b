"""导出有来源/身份校验的内存映射EEG缓存；人工接受与数值检查分别记录。"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from erp_core_p3b.cbramod_audit import (
    adapt_signal,
    model_provenance,
    model_scaled_input,
    sha256_file,
    validate_epoch_header,
)
from erp_core_p3b.classification_data import load_classification_dataset, validate_splits
from erp_core_p3b.classification_io import (
    canonical_hash,
    new_artifact_run,
    resolve_path,
    verify_artifact,
    write_json,
)


def verify_prior_audit(
    root: Path, config: dict[str, Any], dataset: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """要求三人真实GPU技术检查通过及来源一致；不把pending看图状态伪装成已批准。"""
    audit_dir = resolve_path(root, config["audit_directory"])
    audit_path = resolve_path(root, config["audit_config"])
    audit_config = yaml.safe_load(audit_path.read_text(encoding="utf-8"))
    report = json.loads((audit_dir / "run_summary.json").read_text(encoding="utf-8"))
    if report.get("technical_status") != "passed" or not report.get("gpu_probe_complete"):
        raise ValueError("三人真实GPU审计尚未通过")
    if report["config_sha256"] != sha256_file(audit_path):
        raise ValueError("审计配置已改变，需重新运行08而不是混用旧审计")
    if [row["subject"] for row in report["subjects"]] != audit_config["dataset"]["pilot_subjects"]:
        raise ValueError("GPU审计的pilot名单与约定不一致")
    inventory = json.loads((audit_dir / "input_inventory.json").read_text(encoding="utf-8"))
    source_hashes = {row["subject"]: row["sha256"] for row in dataset["sources"]}
    if {row["subject"]: row["sha256"] for row in inventory} != source_hashes:
        raise ValueError("GPU审计与共享清单使用不同FIF版本")
    provenance = model_provenance(root, audit_config["model"])
    old = report["model_resources"]
    if any(old[key] != provenance[key] for key in ("source_commit", "checkpoint_sha256")):
        raise ValueError("模型资源与三人GPU审计不一致")
    return audit_config, {
        "audit_report_sha256": sha256_file(audit_dir / "run_summary.json"),
        "prior_visual_review_status": report.get("visual_review_status"),
        "model_resources": provenance,
    }


def _review_figures(
    output: Path,
    subject: str,
    native: np.ndarray,
    adapted: np.ndarray,
    times: np.ndarray,
    target_times: np.ndarray,
    names: list[str],
    frame: pd.DataFrame,
) -> None:
    flagged = np.flatnonzero(frame["adaptation_review_flag"].to_numpy())
    if not len(flagged):
        return
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = output / "review_figures"
    directory.mkdir(exist_ok=True)
    for start in range(0, len(flagged), 6):
        fig, axes = plt.subplots(3, 2, figsize=(13, 9), constrained_layout=True)
        for axis, index in zip(axes.flat, flagged[start : start + 6], strict=False):
            channel = int(np.ptp(adapted[index], axis=-1).argmax())
            axis.plot(times, native[index, channel] * 1e6, label="Native")
            axis.plot(target_times, adapted[index, channel] * 1e6, "--", label="Adapted")
            axis.axvline(0, color="black", linestyle=":")
            axis.set(
                title=f"{frame.iloc[index]['trial_id']} {names[channel]}",
                xlabel="Time (s)",
                ylabel="uV",
            )
            axis.legend()
        for axis in list(axes.flat)[len(flagged[start : start + 6]) :]:
            axis.set_visible(False)
        fig.savefig(directory / f"{subject}_page-{start // 6 + 1:02d}.png", dpi=130)
        plt.close(fig)


def export_dataset(
    config_path: Path,
    dataset_dir: Path,
    root: Path,
    *,
    stage: str = "full",
    accept_short_window: bool = False,
    output: Path | None = None,
) -> Path:
    """float32原生V及无量纲CBraMod数组逐人写盘；新增提示不删trial、也不自动放行训练。"""
    if stage not in ("pilot", "full"):
        raise ValueError("stage须为pilot或full")
    if stage == "full" and not accept_short_window:
        raise ValueError(
            "先查看三人图形，再显式传入--accept-short-window-limitations确认离线单patch限制"
        )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    trials, dataset, splits = load_classification_dataset(dataset_dir)
    audit, prior = verify_prior_audit(root, config, dataset)
    subjects = (
        set(audit["dataset"]["pilot_subjects"]) if stage == "pilot" else set(trials["subject_id"])
    )
    selected = trials[trials["subject_id"].isin(subjects)].reset_index(drop=True)
    output = resolve_path(root, output or config["outputs"][f"{stage}_directory"])
    shape_native = (len(selected), len(dataset["eeg_channel_names"]), len(dataset["times_s"]))
    shape_model = (len(selected), len(dataset["eeg_channel_names"]), 1, 200)
    with new_artifact_run(root, output, "deep_learning_dataset") as summary:
        write_json(output / "config_snapshot.json", {"export": config, "audit": audit})
        write_json(output / "subject_splits.json", splits)
        selected.to_csv(output / "trial_manifest.csv", index=False)
        native_file = np.lib.format.open_memmap(
            output / "native_eeg_v.npy", mode="w+", dtype=np.float32, shape=shape_native
        )
        model_file = np.lib.format.open_memmap(
            output / "cbramod_input.npy", mode="w+", dtype=np.float32, shape=shape_model
        )
        np.save(
            output / "labels.npy", selected["label"].to_numpy(dtype=np.int64), allow_pickle=False
        )
        reviews, signals, target_times = [], [], None
        try:
            with threadpool_limits(limits=1):
                for source in dataset["sources"]:
                    subject = source["subject"]
                    if subject not in subjects:
                        continue
                    print(f"Exporting {subject} ...", flush=True)
                    path = resolve_path(root, source["source_file"])
                    if sha256_file(path) != source["sha256"]:
                        raise ValueError(f"{subject}: FIF在清单建立后改变")
                    epochs = mne.read_epochs(path, preload=True, verbose="ERROR")
                    picks, identities = validate_epoch_header(
                        epochs, subject, dataset["header_protocol"]
                    )
                    expected = selected[selected["subject_id"] == subject]
                    for column in ("trial_id", "label", "epoch_index", "event_sample_256hz"):
                        if not np.array_equal(identities[column], expected[column]):
                            raise ValueError(f"{subject}: {column}与共同清单不同")
                    names = [epochs.ch_names[index] for index in picks]
                    if names != dataset["eeg_channel_names"] or not np.array_equal(
                        epochs.times, dataset["times_s"]
                    ):
                        raise ValueError("原生通道/时间轴不符合数据版本")
                    native = epochs.get_data(picks=picks, copy=True)
                    residual = float(np.max(np.abs(native.mean(axis=1))) * 1e6)
                    if residual > audit["adaptation"]["mean_reference_tolerance_uv"]:
                        raise ValueError(f"{subject}: 平均参考残差超限")
                    adapted, times, alignment = adapt_signal(
                        native, epochs.times, epochs.info["sfreq"], audit["adaptation"]
                    )
                    scaled, cast_error = model_scaled_input(adapted, audit["adaptation"])
                    if target_times is None:
                        target_times = times
                    if not np.array_equal(target_times, times):
                        raise ValueError("适配后时间网格在受试者间改变")
                    original32 = native.astype(np.float32)
                    native_error = float(np.max(np.abs(original32.astype(float) - native)) * 1e6)
                    if native_error > audit["adaptation"]["float32_max_error_uv"]:
                        raise ValueError("原生float32转换误差超限")
                    native_file[expected.index] = original32
                    model_file[expected.index] = scaled
                    identities["native_max_ptp_uv"] = np.ptp(native, axis=-1).max(axis=1) * 1e6
                    identities["adapted_max_ptp_uv"] = np.ptp(adapted, axis=-1).max(axis=1) * 1e6
                    identities["adaptation_review_flag"] = (
                        identities["adapted_max_ptp_uv"]
                        > audit["adaptation"]["review_peak_to_peak_uv"]
                    )
                    reviews.append(identities)
                    _review_figures(
                        output, subject, native, adapted, epochs.times, times, names, identities
                    )
                    signals.append(
                        {
                            "subject": subject,
                            "trials": len(epochs),
                            "alignment": alignment,
                            "native_float32_error_uv": native_error,
                            "model_float32_error_uv": cast_error,
                            "mean_reference_residual_uv": residual,
                        }
                    )
                    if sha256_file(path) != source["sha256"]:
                        raise ValueError("导出期间源FIF改变")
                    del epochs, native, adapted, scaled, original32
                    gc.collect()
            native_file.flush()
            model_file.flush()
        finally:
            # Windows下主动释放写映射，摘要计算hash前保证数组写入已结束。
            del native_file, model_file
        review = pd.concat(reviews, ignore_index=True)
        if review["trial_id"].tolist() != selected["trial_id"].tolist():
            raise ValueError("导出身份顺序改变")
        review.to_csv(output / "adaptation_review.csv", index=False)
        protocol = {
            "native_times_s": dataset["times_s"],
            "cbramod_times_s": target_times.tolist(),
            "eeg_channel_names": dataset["eeg_channel_names"],
            "adaptation": audit["adaptation"],
            "model_scale_v": audit["adaptation"]["model_scale_v"],
            "single_patch_limitation": True,
            "native_preprocessing": dataset["header_protocol"],
        }
        manifest = {
            "schema_version": 1,
            "dataset_hash": dataset["dataset_hash"],
            "split_hash": canonical_hash(splits),
            "input_protocol": protocol,
            "input_protocol_hash": canonical_hash(protocol),
            "trial_count": len(selected),
            "native_shape": list(shape_native),
            "cbramod_shape": list(shape_model),
            "complete_dataset": stage == "full",
            "model_resources": prior["model_resources"],
            "model_config": audit["model"],
        }
        write_json(output / "export_manifest.json", manifest)
        write_json(output / "subject_export_audit.json", signals)
        flags = int(review["adaptation_review_flag"].sum())
        summary.update(
            dataset_hash=dataset["dataset_hash"],
            split_hash=canonical_hash(splits),
            input_protocol_hash=manifest["input_protocol_hash"],
            complete_dataset=stage == "full",
            stage_selection=stage,
            trial_count=len(selected),
            adaptation_review_trials=flags,
            short_window_limitations_acknowledged=accept_short_window,
            eligible_for_training=stage == "full" and accept_short_window and flags == 0,
            prior_audit=prior,
            raw_fif_modified=False,
            trials_removed=0,
            config_sha256=sha256_file(config_path),
        )
        report = [
            "# 深度学习输入导出",
            "",
            f"阶段：{stage}；trial数：{len(selected)}；新适配提示：{flags}。",
            "",
            "原生数组单位V，CBraMod数组为V/100µV；没有重新滤波、ICA或删trial。",
            "共用09的trial身份与划分，没有重新划分受试者。",
            "单patch和短窗反射支持限制的命令行接受单独记录，不修改08的pending状态。",
            "若有新提示，先看adaptation_review.csv及review_figures，再以明确复核说明运行13。",
            "这不是模型性能结果。数组形状、时间轴和资源记录见export_manifest.json。",
        ]
        (output / "export_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return output


def load_dl_dataset(
    directory: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    summary = verify_artifact(directory, "deep_learning_dataset")
    manifest = json.loads((directory / "export_manifest.json").read_text(encoding="utf-8"))
    splits = json.loads((directory / "subject_splits.json").read_text(encoding="utf-8"))
    trials = pd.read_csv(directory / "trial_manifest.csv")
    if not summary["complete_dataset"] or not manifest["complete_dataset"]:
        raise ValueError("pilot缓存不允许用于共同划分的训练")
    validate_splits(trials, splits)
    if (
        manifest["dataset_hash"] != splits["dataset_hash"]
        or summary["dataset_hash"] != manifest["dataset_hash"]
    ):
        raise ValueError("导出与划分数据版本不一致")
    if (
        canonical_hash(splits) != manifest["split_hash"]
        or manifest["split_hash"] != summary["split_hash"]
    ):
        raise ValueError("固定划分hash不一致")
    if (
        canonical_hash(manifest["input_protocol"]) != manifest["input_protocol_hash"]
        or manifest["input_protocol_hash"] != summary["input_protocol_hash"]
    ):
        raise ValueError("输入协议hash不一致")
    arrays = {
        "native": np.load(directory / "native_eeg_v.npy", mmap_mode="r", allow_pickle=False),
        "cbramod": np.load(directory / "cbramod_input.npy", mmap_mode="r", allow_pickle=False),
        "labels": np.load(directory / "labels.npy", allow_pickle=False),
    }
    if (
        list(arrays["native"].shape) != manifest["native_shape"]
        or list(arrays["cbramod"].shape) != manifest["cbramod_shape"]
    ):
        raise ValueError("缓存数组形状不符合协议")
    if arrays["native"].dtype != np.float32 or arrays["cbramod"].dtype != np.float32:
        raise ValueError("信号缓存必须是float32")
    if len(trials) != manifest["trial_count"] or not np.array_equal(
        arrays["labels"], trials["label"].to_numpy()
    ):
        raise ValueError("缓存标签/行数与trial身份不一致")
    # 分块检查保持memmap的内存优势，避免为全体缓存分配同尺寸的布尔临时数组。
    for name in ("native", "cbramod"):
        for start in range(0, len(trials), 256):
            if not np.isfinite(arrays[name][start : start + 256]).all():
                raise ValueError("信号缓存含NaN/Inf")
    return trials, manifest, splits, arrays, summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path, default=Path("configs/deep_learning_dataset.yaml"))
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("data/processed/classification/v1")
    )
    parser.add_argument("--stage", choices=("pilot", "full"), default="full")
    parser.add_argument("--accept-short-window-limitations", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        output = export_dataset(
            resolve_path(args.project_root, args.config),
            resolve_path(args.project_root, args.dataset_dir),
            args.project_root,
            stage=args.stage,
            accept_short_window=args.accept_short_window_limitations,
            output=args.output_dir,
        )
    except Exception as error:
        print(f"DL export failed: {error}", file=sys.stderr)
        return 1
    print(f"Deep learning dataset: {output}")
    return 0
