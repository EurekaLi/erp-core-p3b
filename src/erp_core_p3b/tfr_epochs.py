"""重建Phase 6.2长窗，并与裁决版ERP复现对照，不重新拟合ICA或自动剔除trial。

完整长窗、主分析卷积支持、正式展示窗分别审计。仅最外侧异常不能自动等同于
主要TFR无效；保护支持中的异常必须先复核。依据见docs/09_tfr_epoch_audit.md。
"""

# ruff: noqa: E402 -- pyplot导入前固定Matplotlib缓存，适配本机受限写入环境。

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import threading
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_MATPLOTLIB_CACHE = Path("results/.matplotlib-cache").resolve()
_MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CACHE))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import psutil
import yaml
from threadpoolctl import threadpool_limits

from erp_core_p3b.preprocessing import EVENT_ID, _filter_branch, _prepare_raw
from erp_core_p3b.raw_qc import _read_tsv, discover_p3_recordings, pair_behavior_events
from erp_core_p3b.tfr_audit import map_retained_trials, morlet_support_table


class RSSMonitor:
    """采样本进程绝对RSS；采样峰值不是操作系统精确峰值，也不是TFR全流程峰值。"""

    def __init__(self, interval_s: float) -> None:
        self.process = psutil.Process()
        self.interval_s = interval_s
        self.initial_bytes = self.process.memory_info().rss
        self.peak_bytes = self.initial_bytes
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop_event.wait(self.interval_s):
            self.mark()

    def mark(self) -> None:
        current = self.process.memory_info().rss
        with self.lock:
            self.peak_bytes = max(self.peak_bytes, current)

    def __enter__(self) -> RSSMonitor:
        self.thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.mark()
        self.stop_event.set()
        self.thread.join()


def check_ica_compatibility(
    raw: mne.io.BaseRaw,
    ica: mne.preprocessing.ICA,
    *,
    excluded: Sequence[int],
) -> None:
    """要求拟合/应用通道顺序、参考、采样率及冻结排除集合一致，错配时直接失败。"""
    good_eeg = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude="bads")]
    if good_eeg != list(ica.ch_names):
        raise ValueError("ICA拟合通道与应用分支的有效EEG通道/顺序不一致")
    if raw.info["sfreq"] != ica.info["sfreq"]:
        raise ValueError("ICA拟合与应用采样率不同")
    if raw.info["custom_ref_applied"] != ica.info["custom_ref_applied"]:
        raise ValueError("ICA拟合与应用参考标志不同")
    if raw.info["projs"] or ica.info["projs"]:
        raise ValueError("本项目冻结ICA不应含额外投影；须显式复核后适配")
    if sorted(excluded) != sorted(ica.exclude):
        raise ValueError("ICA文件排除集合与Phase 4汇总不一致")
    if any(index < 0 or index >= ica.n_components_ for index in excluded):
        raise ValueError("冻结ICA排除编号越界")


def apply_frozen_ica(
    raw: mne.io.BaseRaw, ica: mne.preprocessing.ICA, bad_channels: Sequence[str]
) -> None:
    """保持Phase 4的处理顺序；坏道不进入ICA，插值后才进入最终平均参考。"""
    ica.apply(raw, exclude=list(ica.exclude), verbose="ERROR")
    if bad_channels:
        raw.interpolate_bads(reset_bads=True, mode="accurate", verbose="ERROR")
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")


def make_retained_epochs(
    raw: mne.io.BaseRaw,
    old: mne.BaseEpochs,
    window: Sequence[float],
    baseline: Sequence[float] | None,
) -> mne.Epochs:
    """仅使用旧保留事件；任何边界/注释导致的丢失都须显式处理，不能悄悄改样本。"""
    result = mne.Epochs(
        raw,
        old.events.copy(),
        event_id=EVENT_ID,
        tmin=float(window[0]),
        tmax=float(window[1]),
        baseline=None if baseline is None else tuple(baseline),
        picks=mne.pick_types(raw.info, eeg=True, eog=True, exclude=[]),
        metadata=old.metadata.copy(),
        preload=True,
        reject=None,
        flat=None,
        reject_by_annotation=True,
        event_repeated="error",
        verbose="ERROR",
    )
    if not np.array_equal(result.events, old.events):
        raise ValueError("重建时出现事件丢失/变化，禁止自动改变裁决版名单")
    if not np.array_equal(result.metadata["trial_index"], old.metadata["trial_index"]):
        raise ValueError("重建trial顺序与裁决名单不同")
    return result


def compare_erp_reproduction(
    reconstructed: mne.BaseEpochs, saved: mne.BaseEpochs, tolerance_uv: float
) -> dict[str, float]:
    """在相同ERP参数下验证重建；FIF的float32保存误差允许到预先固定的容差。"""
    if reconstructed.ch_names != saved.ch_names:
        raise ValueError("ERP复现通道顺序不同")
    if not np.array_equal(reconstructed.times, saved.times):
        raise ValueError("ERP复现时间轴不同")
    delta = reconstructed.get_data(copy=False) - saved.get_data(copy=False)
    max_error = float(np.max(np.abs(delta)) * 1e6)
    if not np.isfinite(delta).all() or max_error > tolerance_uv:
        raise ValueError(f"ERP同参数复现误差超限: {max_error:.6g} µV")
    return {
        "erp_reproduction_max_error_uv": max_error,
        "erp_reproduction_rms_error_uv": float(np.sqrt(np.mean(delta**2)) * 1e6),
    }


def artifact_metrics(
    data_v: np.ndarray,
    times: np.ndarray,
    support: pd.DataFrame,
    threshold_uv: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    """分别报告全窗和主分析真实输入支持；200µV仅作提示，不返回自动剔除集合。"""
    if not np.isfinite(data_v).all():
        raise ValueError("长窗含NaN/Inf，不能进入TFR")
    baseline_left = support["baseline_support_start_s"].min()
    baseline_right = support["baseline_support_end_s"].max()
    display_left = support["display_support_start_s"].min()
    display_right = support["display_support_end_s"].max()
    masks = {
        "full": np.ones(len(times), dtype=bool),
        "display": (times >= -0.2) & (times <= 0.8),
        "baseline_support": (times >= baseline_left) & (times <= baseline_right),
        "display_support": (times >= display_left) & (times <= display_right),
        "protected": (times >= min(baseline_left, display_left))
        & (times <= max(baseline_right, display_right)),
    }
    result: dict[str, Any] = {}
    for name, mask in masks.items():
        if not mask.any():
            raise ValueError(f"{name}保护时间窗为空")
        channel_ptp = np.ptp(data_v[:, :, mask], axis=-1) * 1e6
        result[f"{name}_max_ptp_uv"] = channel_ptp.max(axis=1)
        result[f"{name}_channels_above_review"] = (channel_ptp > threshold_uv).sum(axis=1)
    result["full_review_flag"] = result["full_max_ptp_uv"] > threshold_uv
    result["protected_review_flag"] = result["protected_max_ptp_uv"] > threshold_uv
    result["outer_context_only_review_flag"] = (
        result["full_review_flag"] & ~result["protected_review_flag"]
    )
    # 记录最大瞬时偏离的位置，便于判断异常是否远离真正使用的支持区域。
    centered = data_v - np.median(data_v, axis=-1, keepdims=True)
    max_channel_time = np.abs(centered).reshape(len(data_v), -1).argmax(axis=1)
    result["largest_deviation_time_s"] = times[max_channel_time % len(times)]
    result["largest_deviation_channel_index"] = max_channel_time // len(times)
    return pd.DataFrame(result), np.ptp(data_v, axis=-1) * 1e6


def smoke_test_morlet(
    epochs: mne.BaseEpochs, feature: dict[str, Any], metrics: pd.DataFrame
) -> dict[str, Any]:
    """只测试小批卷积与基线数值，不生成条件结果；包含两条件和最严重保护窗trial。"""
    labels = epochs.metadata["condition_label"].to_numpy()
    selected = [int(metrics["protected_max_ptp_uv"].idxmax())]
    for label in ("Target", "Standard"):
        selected.extend(np.flatnonzero(labels == label)[:3].tolist())
    selected = list(dict.fromkeys(selected))
    remaining = [i for i in range(len(epochs)) if i not in selected]
    selected.extend(remaining[: max(0, feature["batch_trials"] - len(selected))])
    selected = selected[: feature["batch_trials"]]
    frequencies = np.arange(
        feature["fmin_hz"], feature["fmax_hz"] + 0.01, feature["frequency_step_hz"]
    )
    rule = feature["morlet"]
    cycles = np.clip(frequencies / rule["cycle_divisor_hz"], rule["min_cycles"], rule["max_cycles"])
    coefficients = mne.time_frequency.tfr_array_morlet(
        epochs[selected].get_data(picks="eeg", copy=False),
        sfreq=epochs.info["sfreq"],
        freqs=frequencies,
        n_cycles=cycles,
        zero_mean=rule["zero_mean"],
        use_fft=True,
        output="complex",
        n_jobs=1,
        verbose="ERROR",
    )
    power = np.abs(coefficients) ** 2
    baseline = (epochs.times >= feature["baseline_s"][0]) & (
        epochs.times <= feature["baseline_s"][1]
    )
    denominator = power[..., baseline].mean(axis=-1, keepdims=True)
    if not np.isfinite(power).all() or not (denominator > 0).all():
        raise ValueError("试运行功率非有限或基线分母不为正，禁止用任意epsilon掩盖")
    db = 10 * np.log10(power / denominator)
    display = (epochs.times >= feature["display_window_s"][0]) & (
        epochs.times <= feature["display_window_s"][1]
    )
    if not np.isfinite(db[..., display]).all():
        raise ValueError("试运行正式展示区间存在非有限dB")
    return {
        "smoke_trial_indices": epochs.metadata.iloc[selected]["trial_index"].astype(int).tolist(),
        "smoke_trial_count": len(selected),
        "smoke_shape": list(coefficients.shape),
        "baseline_power_min_v2": float(denominator.min()),
        "baseline_power_median_v2": float(np.median(denominator)),
        "baseline_power_max_v2": float(denominator.max()),
        "smoke_finite": True,
    }


def plot_long_qc(
    epochs: mne.BaseEpochs,
    old: mne.BaseEpochs,
    reproduced: mne.BaseEpochs,
    metrics: pd.DataFrame,
    channel_ptp: np.ndarray,
    support: pd.DataFrame,
    path: Path,
) -> None:
    """将保护支持与完整窗区别可视化，最严重trial只用于QC、不作条件统计。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    subject = epochs.metadata.iloc[0]["subject"]
    fig.suptitle(f"{subject} | long-epoch reconstruction QC")
    for name, color in (("full", "tab:blue"), ("protected", "tab:orange")):
        axes[0, 0].hist(metrics[f"{name}_max_ptp_uv"], bins=30, alpha=0.5, label=name, color=color)
    axes[0, 0].axvline(200, color="red", linestyle="--", label="Review only: 200 uV")
    axes[0, 0].set(title="Full vs protected peak-to-peak", xlabel="uV", ylabel="Trials")
    axes[0, 0].legend()
    worst = int(metrics["protected_max_ptp_uv"].idxmax())
    picks = mne.pick_types(epochs.info, eeg=True, exclude=[])
    values = epochs.get_data(picks=picks, copy=False)[worst] * 1e6
    protected_mask = (epochs.times >= support["baseline_support_start_s"].min()) & (
        epochs.times <= support["display_support_end_s"].max()
    )
    # 最严重保护窗的通道按该窗排序，避免全窗外围异常掩盖真正触发复核的通道。
    protected_ptp = np.ptp(values[:, protected_mask], axis=-1)
    for channel in np.argsort(protected_ptp)[-5:]:
        axes[0, 1].plot(
            epochs.times, values[channel], label=epochs.ch_names[picks[channel]], linewidth=0.8
        )
    left = min(support["baseline_support_start_s"].min(), support["display_support_start_s"].min())
    right = max(support["baseline_support_end_s"].max(), support["display_support_end_s"].max())
    axes[0, 1].axvspan(left, right, alpha=0.1, color="green", label="Protected support")
    axes[0, 1].axvline(0, color="black", linestyle="--")
    trial_index = epochs.metadata.iloc[worst]["trial_index"]
    axes[0, 1].set(title=f"Worst protected trial: {trial_index}", xlabel="s", ylabel="uV")
    axes[0, 1].legend(fontsize=8, ncol=2)
    mesh = axes[1, 0].imshow(channel_ptp.T, aspect="auto", origin="lower", cmap="magma")
    axes[1, 0].set(title="Full-window channel PTP", xlabel="Retained trial order", ylabel="EEG")
    axes[1, 0].set_yticks(range(len(picks)), [epochs.ch_names[p] for p in picks], fontsize=7)
    fig.colorbar(mesh, ax=axes[1, 0], label="uV")
    for source, label, style in ((old, "Saved ERP", "-"), (reproduced, "Reproduced ERP", "--")):
        evoked = source["Target"].average(picks=["Pz"])
        axes[1, 1].plot(evoked.times, evoked.data[0] * 1e6, style, label=label)
    axes[1, 1].set(title="Same-parameter ERP reproduction: Target/Pz", xlabel="s", ylabel="uV")
    axes[1, 1].legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def process_long_subject(
    subject: str,
    set_path: Path,
    row: pd.Series,
    config: dict[str, Any],
    feature: dict[str, Any],
    frozen_preprocessing: dict[str, Any],
    output_epochs: Path,
    output_results: Path,
    output_figures: Path,
    verify_existing: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    with RSSMonitor(config["resources"]["rss_sampling_interval_s"]) as monitor:
        old = mne.read_epochs(
            Path(config["dataset"]["erp_epochs_root"]) / f"{subject}_task-P3_epo.fif",
            preload=True,
            verbose="ERROR",
        )
        trials, _ = pair_behavior_events(
            _read_tsv(set_path.with_name(set_path.name.replace("_eeg.set", "_events.tsv"))),
            subject=subject,
            min_reaction_time_s=0.15,
            max_response_window_s=1.6,
        )
        map_retained_trials(
            old.metadata,
            trials,
            sfreq_hz=feature["target_sfreq_hz"],
            event_samples=old.events[:, 0],
            event_codes=old.events[:, 2],
        )
        bad_channels = (
            []
            if pd.isna(row["manual_bad_channels"])
            else (str(row["manual_bad_channels"]).split(";"))
        )
        excluded = (
            []
            if pd.isna(row["ica_excluded_components"])
            else ([int(v) for v in str(row["ica_excluded_components"]).split(";")])
        )
        ica = mne.preprocessing.read_ica(row["ica_path"], verbose="ERROR")
        raw = _prepare_raw(set_path, bad_channels)
        monitor.mark()
        erp_config = frozen_preprocessing["signal"]
        # 对照分支完全复用Phase 4参数，只用于证明ICA/参考/事件路径没有改变。
        erp = _filter_branch(
            raw,
            highpass_hz=erp_config["analysis_highpass_hz"],
            lowpass_hz=erp_config["analysis_lowpass_hz"],
            target_sfreq_hz=feature["target_sfreq_hz"],
        )
        check_ica_compatibility(erp, ica, excluded=excluded)
        apply_frozen_ica(erp, ica, bad_channels)
        ep = frozen_preprocessing["epochs"]
        reproduced = make_retained_epochs(erp, old, [ep["tmin_s"], ep["tmax_s"]], ep["baseline_s"])
        reproduction = compare_erp_reproduction(
            reproduced, old, config["quality_control"]["erp_reproduction_max_abs_error_uv"]
        )
        del erp
        gc.collect()
        branch = _filter_branch(
            raw,
            highpass_hz=config["signal"]["highpass_hz"],
            lowpass_hz=config["signal"]["lowpass_hz"],
            target_sfreq_hz=feature["target_sfreq_hz"],
        )
        del raw
        check_ica_compatibility(branch, ica, excluded=excluded)
        apply_frozen_ica(branch, ica, bad_channels)
        long = make_retained_epochs(branch, old, feature["epoch_window_s"], baseline=None)
        del branch
        frequencies = np.arange(
            feature["fmin_hz"], feature["fmax_hz"] + 0.01, feature["frequency_step_hz"]
        )
        support = morlet_support_table(
            sfreq_hz=feature["target_sfreq_hz"],
            frequencies_hz=frequencies,
            config=feature["morlet"],
            epoch_window_s=feature["epoch_window_s"],
            baseline_window_s=feature["baseline_s"],
            display_window_s=feature["display_window_s"],
        )
        if (
            not support["baseline_edge_safe"].all()
            or not support["display_edge_safe"].all()
            or support["baseline_reaches_current_stimulus"].any()
        ):
            raise ValueError("冻结TFR参数的完整支持检查失败")
        data = long.get_data(picks="eeg", copy=False)
        metrics, channel_ptp = artifact_metrics(
            data, long.times, support, config["quality_control"]["review_peak_to_peak_uv"]
        )
        if not np.isfinite(long.get_data(copy=False)).all():
            raise ValueError("EOG或EEG出现非有限值")
        reference_residual = float(np.abs(data.mean(axis=1)).max() * 1e6)
        if reference_residual > 0.001:
            raise ValueError("最终EEG通道平均不为零，请检查参考/坏道处理")
        smoke = smoke_test_morlet(long, feature, metrics)
        long.metadata = pd.concat([long.metadata.reset_index(drop=True), metrics], axis=1)
        subject_dir = output_results / subject
        subject_dir.mkdir(parents=True, exist_ok=True)
        long.metadata.to_csv(subject_dir / "trial_qc.csv", index=False, encoding="utf-8-sig")
        # 独立分支不覆盖ERP；double降低数据载荷量化误差，但FIF校准头仍有有限精度。
        output_epochs.mkdir(parents=True, exist_ok=True)
        epochs_path = output_epochs / f"{subject}_task-P3_epo.fif"
        if epochs_path.exists():
            if not verify_existing:
                raise FileExistsError(f"拒绝覆盖已有长epochs: {epochs_path}")
            # FIF校准头也参与缩放，故double载荷仍不能要求逐位相等；沿用固定µV容差。
            stored = mne.read_epochs(epochs_path, preload=True, verbose="ERROR")
            np.testing.assert_array_equal(stored.events, long.events)
            np.testing.assert_array_equal(stored.times, long.times)
            compare_erp_reproduction(
                stored, long, config["quality_control"]["erp_reproduction_max_abs_error_uv"]
            )
            np.testing.assert_array_equal(
                stored.metadata["trial_index"], long.metadata["trial_index"]
            )
            np.testing.assert_array_equal(
                stored.metadata["condition_label"], long.metadata["condition_label"]
            )
            if stored.baseline is not None:
                raise ValueError("既有长窗不应包含电压baseline")
            del stored
        else:
            long.save(epochs_path, fmt="double", overwrite=False, verbose="ERROR")
        plot_long_qc(
            long,
            old,
            reproduced,
            metrics,
            channel_ptp,
            support,
            output_figures / f"{subject}_long_epoch_qc.png",
        )
        counts = long.metadata["condition_label"].value_counts()
        if counts.get("Target", 0) != row["target_retained"] or (
            counts.get("Standard", 0) != row["standard_retained"]
        ):
            raise ValueError("最终条件计数与Phase 4不同")
        summary = {
            "subject": subject,
            "trial_count": len(long),
            "target_trials": int(counts.get("Target", 0)),
            "standard_trials": int(counts.get("Standard", 0)),
            "n_channels": len(long.ch_names),
            "n_times": len(long.times),
            "epoch_tmin_s": long.tmin,
            "epoch_tmax_s": long.tmax,
            "sfreq_hz": long.info["sfreq"],
            "ica_components": int(ica.n_components_),
            "ica_excluded": excluded,
            "interpolated_channels": bad_channels,
            "ica_compatibility_passed": True,
            "erp_reproduction_passed": True,
            "trial_identity_passed": True,
            "reference_residual_max_uv": reference_residual,
            **reproduction,
            **smoke,
            "full_review_trials": int(metrics["full_review_flag"].sum()),
            "protected_review_trials": int(metrics["protected_review_flag"].sum()),
            "outer_context_only_review_trials": int(
                metrics["outer_context_only_review_flag"].sum()
            ),
            "protected_ptp_max_uv": float(metrics["protected_max_ptp_uv"].max()),
            "full_ptp_max_uv": float(metrics["full_max_ptp_uv"].max()),
            "epochs_path": epochs_path.as_posix(),
            "epochs_file_mib": epochs_path.stat().st_size / 2**20,
        }
        del data, long, old, reproduced, ica
        gc.collect()
    summary.update(
        elapsed_s=time.perf_counter() - started,
        initial_rss_gib=monitor.initial_bytes / 2**30,
        sampled_peak_rss_gib=monitor.peak_bytes / 2**30,
        sampled_rss_increase_gib=(monitor.peak_bytes - monitor.initial_bytes) / 2**30,
    )
    (subject_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def load_and_validate_inputs(
    config_path: Path,
    feature_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    """冻结来源一致才允许重建；避免新的配置静默改变旧ICA/trial的解释。"""
    paths = {
        "builder": config_path,
        "features": feature_path,
        "preprocessing": Path("configs/preprocessing.yaml"),
    }
    texts = {key: path.read_text(encoding="utf-8") for key, path in paths.items()}
    config = yaml.safe_load(texts["builder"])
    feature = yaml.safe_load(texts["features"])["time_frequency"]
    frozen = json.loads(Path(config["dataset"]["preprocessing_run"]).read_text(encoding="utf-8"))
    if yaml.safe_load(texts["preprocessing"]) != frozen["config"]:
        raise ValueError("当前预处理配置已偏离裁决版配置，必须先审计差异")
    audit = json.loads(Path(feature["audit_summary"]).read_text(encoding="utf-8"))
    if not feature["parameters_frozen"] or not audit["is_full_cohort"]:
        raise ValueError("必须存在完整长窗审计并冻结参数")
    hashes = {key: hashlib.sha256(text.encode("utf-8")).hexdigest() for key, text in texts.items()}
    hashes["preprocessing_run"] = hashlib.sha256(
        Path(config["dataset"]["preprocessing_run"]).read_bytes()
    ).hexdigest()
    return config, feature, frozen["config"], hashes


def run_builder(
    *, config_path: Path, feature_path: Path, stage: str, verify_existing: bool = False
) -> dict[str, Any]:
    config, feature, frozen, hashes = load_and_validate_inputs(config_path, feature_path)
    paths = config["outputs"]
    results_root = Path(paths["results_root"])
    pilot_summary_path = results_root / "pilot" / "run_summary.json"
    if stage == "final":
        pilot = json.loads(pilot_summary_path.read_text(encoding="utf-8"))
        if not pilot["eligible_for_full_run"] or pilot["config_hashes"] != hashes:
            raise ValueError("三人pilot尚未通过门控或配置已改变，不能直接运行全体")
    desired = config["dataset"]["pilot_subjects"] if stage == "pilot" else None
    recordings = dict(discover_p3_recordings(Path(config["dataset"]["raw_root"])))
    inventory = pd.read_csv(config["dataset"]["preprocessing_summary"]).set_index("subject")
    if not inventory.index.is_unique:
        raise ValueError("裁决版受试者索引重复")
    subjects = desired if desired is not None else sorted(inventory.index)
    if not verify_existing:
        existing = [
            Path(paths["epochs_root"]) / stage / f"{subject}_task-P3_epo.fif"
            for subject in subjects
        ]
        if any(path.exists() for path in existing):
            raise FileExistsError("已有长epochs；仅可使用--verify-existing校验，不默认覆盖")
    if stage == "final" and len(subjects) != config["dataset"]["expected_subject_count"]:
        raise ValueError("全体受试者数与协议不同")
    if set(subjects) & set(config["dataset"]["excluded_subjects"]):
        raise ValueError("主分析中出现已排除受试者")
    result_dir = results_root / stage
    result_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, subject in enumerate(subjects, 1):
        print(f"[{index:02d}/{len(subjects):02d}] build {subject}", flush=True)
        with threadpool_limits(limits=config["resources"]["thread_limit"]):
            summary = process_long_subject(
                subject,
                recordings[subject],
                inventory.loc[subject],
                config,
                feature,
                frozen,
                Path(paths["epochs_root"]) / stage,
                result_dir,
                Path(paths["figures_root"]) / stage,
                verify_existing=verify_existing,
            )
        summaries.append(summary)
        pd.DataFrame(summaries).to_csv(
            result_dir / "subject_summary.csv", index=False, encoding="utf-8-sig"
        )
        print(
            f"  trials={summary['trial_count']}"
            f" | protected_review={summary['protected_review_trials']}"
            f" | peak_RSS={summary['sampled_peak_rss_gib']:.3f} GiB"
            f" | ERP_error={summary['erp_reproduction_max_error_uv']:.3g} uV",
            flush=True,
        )
    total = sum(row["trial_count"] for row in summaries)
    if stage == "final" and total != config["dataset"]["expected_trial_count"]:
        raise ValueError("全体trial数偏离协议")
    review_count = sum(row["protected_review_trials"] for row in summaries)
    budget_ok = (
        max(row["sampled_peak_rss_gib"] for row in summaries) < feature["planning_budget_gib"]
    )
    require_review_free = config["quality_control"][
        "require_no_protected_review_flags_for_full_run"
    ]
    eligible = budget_ok and (not require_review_free or review_count == 0)
    result = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "stage": stage,
        "subjects_processed": len(summaries),
        "trials": total,
        "target_trials": sum(row["target_trials"] for row in summaries),
        "standard_trials": sum(row["standard_trials"] for row in summaries),
        "technical_checks_passed": True,
        "protected_review_trials": review_count,
        "outer_context_only_review_trials": sum(
            row["outer_context_only_review_trials"] for row in summaries
        ),
        "sampled_peak_rss_gib": max(row["sampled_peak_rss_gib"] for row in summaries),
        "within_planning_budget": budget_ok,
        "eligible_for_full_run": eligible,
        "formal_tfr_ready_without_further_artifact_review": review_count == 0,
        "config_hashes": hashes,
        "mne_version": mne.__version__,
        "numpy_version": np.__version__,
        "psutil_version": psutil.__version__,
        "ica_refit": False,
        "automatic_trial_rejections": 0,
        "rss_measurement": "process RSS sampled every 20ms plus explicit checkpoints",
        "smoke_only_not_formal_tfr": True,
        "subjects": summaries,
    }
    (result_dir / "run_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def review_saved_pilot(*, config_path: Path, feature_path: Path) -> dict[str, Any]:
    """从已有pilot信号生成全部候选的波形分页和清单；不修改原epochs或复核决定。"""
    config, feature, _, hashes = load_and_validate_inputs(config_path, feature_path)
    root = Path(config["outputs"]["results_root"]) / "pilot"
    summary = json.loads((root / "run_summary.json").read_text(encoding="utf-8"))
    if summary["config_hashes"] != hashes:
        raise ValueError("已有pilot的配置与当前配置不一致，禁止混合复核")
    figure_root = Path(config["outputs"]["figures_root"]) / "pilot" / "review"
    figure_root.mkdir(parents=True, exist_ok=True)
    frequencies = np.arange(
        feature["fmin_hz"], feature["fmax_hz"] + 0.01, feature["frequency_step_hz"]
    )
    support = morlet_support_table(
        sfreq_hz=feature["target_sfreq_hz"],
        frequencies_hz=frequencies,
        config=feature["morlet"],
        epoch_window_s=feature["epoch_window_s"],
        baseline_window_s=feature["baseline_s"],
        display_window_s=feature["display_window_s"],
    )
    left = min(support["baseline_support_start_s"].min(), support["display_support_start_s"].min())
    right = max(support["baseline_support_end_s"].max(), support["display_support_end_s"].max())
    candidate_rows: list[dict[str, Any]] = []
    pages: list[str] = []
    for record in summary["subjects"]:
        epochs = mne.read_epochs(record["epochs_path"], preload=True, verbose="ERROR")
        subject = record["subject"]
        old = mne.read_epochs(
            Path(config["dataset"]["erp_epochs_root"]) / f"{subject}_task-P3_epo.fif",
            preload=True,
            verbose="ERROR",
        )
        np.testing.assert_array_equal(epochs.events, old.events)
        data = epochs.get_data(picks="eeg", copy=False) * 1e6
        eeg_names = [epochs.ch_names[p] for p in mne.pick_types(epochs.info, eeg=True, exclude=[])]
        protected = (epochs.times >= left) & (epochs.times <= right)
        protected_ptp = np.ptp(data[:, :, protected], axis=-1)
        old_ptp = np.ptp(old.get_data(picks="eeg", copy=False), axis=-1).max(axis=1) * 1e6
        candidates = np.flatnonzero(epochs.metadata["protected_review_flag"].to_numpy())
        # 依严重程度展示，但清单保留trial_index和绝对记录时间，避免把重叠窗重复当独立异常。
        candidates = candidates[np.argsort(protected_ptp[candidates].max(axis=1))[::-1]]
        for index in candidates:
            row = epochs.metadata.iloc[index]
            channel = int(protected_ptp[index].argmax())
            signal = data[index, channel, protected]
            protected_times = epochs.times[protected]
            t_min = float(protected_times[signal.argmin()])
            t_max = float(protected_times[signal.argmax()])
            candidate_rows.append(
                {
                    "subject": subject,
                    "trial_index": int(row["trial_index"]),
                    "condition": row["condition_label"],
                    "stimulus_onset_s": row["stimulus_onset_s"],
                    "strongest_protected_channel": eeg_names[channel],
                    "protected_max_ptp_uv": float(protected_ptp[index, channel]),
                    "display_max_ptp_uv": row["display_max_ptp_uv"],
                    "baseline_support_max_ptp_uv": row["baseline_support_max_ptp_uv"],
                    "display_support_max_ptp_uv": row["display_support_max_ptp_uv"],
                    "full_max_ptp_uv": row["full_max_ptp_uv"],
                    "saved_erp_max_ptp_uv": float(old_ptp[index]),
                    "protected_min_time_s": t_min,
                    "protected_max_time_s": t_max,
                    "recording_min_time_s": float(row["stimulus_onset_s"] + t_min),
                    "recording_max_time_s": float(row["stimulus_onset_s"] + t_max),
                    "manual_decision": "pending",
                    "reason": "",
                }
            )
        for page_start in range(0, len(candidates), 6):
            fig, axes = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)
            fig.suptitle(f"{subject} | protected-support candidates | review only")
            for axis, index in zip(
                axes.flat, candidates[page_start : page_start + 6], strict=False
            ):
                row = epochs.metadata.iloc[index]
                for channel in np.argsort(protected_ptp[index])[-2:]:
                    axis.plot(
                        epochs.times, data[index, channel], label=eeg_names[channel], linewidth=0.9
                    )
                axis.axvspan(left, right, color="green", alpha=0.1, label="Protected support")
                axis.axvline(0, color="black", linestyle="--", linewidth=0.8)
                axis.axvline(-0.2, color="0.4", linestyle=":")
                axis.axvline(0.8, color="0.4", linestyle=":")
                axis.set(
                    title=f"trial {row['trial_index']} | {row['condition_label']} | "
                    f"protected={row['protected_max_ptp_uv']:.1f} uV, "
                    f"old ERP={old_ptp[index]:.1f} uV",
                    xlabel="s",
                    ylabel="uV",
                )
                axis.legend(fontsize=7, ncol=2)
            for axis in list(axes.flat)[len(candidates[page_start : page_start + 6]) :]:
                axis.set_visible(False)
            name = f"{subject}_protected_review_{page_start // 6 + 1:02d}.png"
            fig.savefig(figure_root / name, dpi=140)
            plt.close(fig)
            pages.append(name)
        del epochs, old, data
        gc.collect()
    frame = pd.DataFrame(candidate_rows)
    frame.to_csv(root / "protected_review_candidates.csv", index=False, encoding="utf-8-sig")
    grouped = frame.groupby(["subject", "condition"]).size().reset_index(name="candidate_trials")
    grouped.to_csv(
        root / "protected_review_condition_counts.csv", index=False, encoding="utf-8-sig"
    )
    review = {
        "candidate_trials": len(frame),
        "waveform_pages": len(pages),
        "pages": pages,
        "automatic_decisions": 0,
        "protected_support_window_s": [float(left), float(right)],
    }
    (root / "review_summary.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    content = "\n\n".join(f"## {name}\n\n![候选波形]({name})" for name in pages)
    (figure_root / "README.md").write_text(
        "# 三人长窗复核波形\n\n绿色为主分析输入支持；点线为旧ERP范围。"
        "所有标记仅待复核，未自动删trial。\n\n" + content + "\n",
        encoding="utf-8",
    )
    return review


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="复用冻结ICA重建长TFR epochs，含三人门控。")
    parser.add_argument("--config", type=Path, default=Path("configs/tfr_epochs.yaml"))
    parser.add_argument("--features", type=Path, default=Path("configs/features.yaml"))
    parser.add_argument("--stage", choices=("pilot", "final"), default="pilot")
    parser.add_argument("--review-only", action="store_true", help="仅从已有pilot生成候选复核图")
    parser.add_argument(
        "--verify-existing", action="store_true", help="重建并校验已有长窗，绝不覆盖"
    )
    args = parser.parse_args(argv)
    if args.review_only:
        if args.stage != "pilot":
            parser.error("--review-only只适用于pilot")
        print(review_saved_pilot(config_path=args.config, feature_path=args.features))
        return 0
    result = run_builder(
        config_path=args.config,
        feature_path=args.features,
        stage=args.stage,
        verify_existing=args.verify_existing,
    )
    print(
        f"Long epochs complete | stage={args.stage} | trials={result['trials']} | "
        f"eligible_for_full_run={result['eligible_for_full_run']}"
    )
    return 0
