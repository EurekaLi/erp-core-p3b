"""ERP CORE P3 原始行为与连续 EEG 的只读质量控制。"""

# ruff: noqa: E402 -- Matplotlib 导入前必须把缓存重定向到沙箱可写目录。

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Codex 沙箱中的默认 Matplotlib 用户缓存不可写，项目内缓存也能避免每次重建字体索引。
_MATPLOTLIB_CACHE = Path("results/.matplotlib-cache").resolve()
_MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CACHE))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import yaml
from scipy.signal import welch

from erp_core_p3b.validation import classify_p3_event


@dataclass
class SignalQCResult:
    """单名受试者的通道指标、汇总指标及绘图所需数组。"""

    channel_metrics: list[dict[str, Any]]
    summary: dict[str, Any]
    plot_data: dict[str, Any]


def discover_p3_recordings(bids_root: Path, task: str = "P3") -> list[tuple[str, Path]]:
    """发现有 session 或无 session 两种 BIDS 布局中的 EEGLAB 记录。"""

    recordings: list[tuple[str, Path]] = []
    for subject_dir in sorted(path for path in bids_root.glob("sub-*") if path.is_dir()):
        pattern = f"*_task-{task}_eeg.set"
        candidates = list(subject_dir.glob(f"eeg/{pattern}"))
        candidates.extend(subject_dir.glob(f"ses-*/eeg/{pattern}"))
        recordings.extend((subject_dir.name, path) for path in sorted(set(candidates)))
    return recordings


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def pair_behavior_events(
    rows: Sequence[dict[str, Any]],
    *,
    subject: str,
    min_reaction_time_s: float,
    max_response_window_s: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """把每个刺激与其后、下一个刺激前的反应配对。

    最后一个 trial 没有“下一刺激”作为边界，因此还使用固定最大反应窗；这也防止任务
    结束后的无关按键被错误计入最后一个 trial。
    """

    parsed: list[dict[str, Any]] = []
    for order, row in enumerate(rows):
        try:
            onset = float(row["onset"])
            code = int(row["value"])
        except (KeyError, TypeError, ValueError):
            continue
        parsed.append(
            {
                "order": order,
                "onset": onset,
                "code": code,
                "category": classify_p3_event(code),
            }
        )
    parsed.sort(key=lambda event: (event["onset"], event["order"]))

    stimuli = [event for event in parsed if event["category"] in {"target", "standard"}]
    responses = [
        event
        for event in parsed
        if event["category"] in {"correct_response", "error_response"}
    ]
    response_used: set[int] = set()
    trial_rows: list[dict[str, Any]] = []

    for index, stimulus in enumerate(stimuli):
        next_onset = stimuli[index + 1]["onset"] if index + 1 < len(stimuli) else np.inf
        window_end = min(next_onset, stimulus["onset"] + max_response_window_s)
        matches = [
            (response_index, response)
            for response_index, response in enumerate(responses)
            if stimulus["onset"] <= response["onset"] < window_end
        ]
        response_used.update(response_index for response_index, _ in matches)
        primary = matches[0][1] if matches else None
        reaction_time = primary["onset"] - stimulus["onset"] if primary else np.nan
        response_code = primary["code"] if primary else ""
        accuracy = (
            "correct"
            if response_code == 201
            else "incorrect"
            if response_code == 202
            else "missing"
        )
        flags: list[str] = []
        if not matches:
            flags.append("missing_response")
        if len(matches) > 1:
            flags.append("multiple_responses")
        if primary and reaction_time < min_reaction_time_s:
            flags.append("anticipatory")

        trial_rows.append(
            {
                "subject": subject,
                "trial_index": index + 1,
                "condition": stimulus["category"],
                "stimulus_code": stimulus["code"],
                "stimulus_onset_s": stimulus["onset"],
                "response_code": response_code,
                "response_onset_s": primary["onset"] if primary else np.nan,
                "reaction_time_s": reaction_time,
                "response_count": len(matches),
                "accuracy": accuracy,
                "behavior_qc_flag": ";".join(flags),
                "valid_behavior_trial": accuracy == "correct" and not flags,
            }
        )

    unmatched_responses = len(responses) - len(response_used)
    accuracy_counts = Counter(row["accuracy"] for row in trial_rows)
    condition_counts = Counter(row["condition"] for row in trial_rows)
    correct_rts = [
        float(row["reaction_time_s"])
        for row in trial_rows
        if row["accuracy"] == "correct" and np.isfinite(row["reaction_time_s"])
    ]
    summary = {
        "subject": subject,
        "stimulus_trials": len(trial_rows),
        "target_trials": condition_counts["target"],
        "standard_trials": condition_counts["standard"],
        "correct_trials": accuracy_counts["correct"],
        "incorrect_trials": accuracy_counts["incorrect"],
        "missing_response_trials": accuracy_counts["missing"],
        "multiple_response_trials": sum(
            "multiple_responses" in row["behavior_qc_flag"] for row in trial_rows
        ),
        "anticipatory_trials": sum(
            "anticipatory" in row["behavior_qc_flag"] for row in trial_rows
        ),
        "unmatched_responses": unmatched_responses,
        "valid_behavior_trials": sum(row["valid_behavior_trial"] for row in trial_rows),
        "correct_rt_median_s": float(np.median(correct_rts)) if correct_rts else np.nan,
        "correct_rt_p05_s": float(np.percentile(correct_rts, 5)) if correct_rts else np.nan,
        "correct_rt_p95_s": float(np.percentile(correct_rts, 95)) if correct_rts else np.nan,
    }
    return trial_rows, summary


def robust_zscore(values: np.ndarray) -> np.ndarray:
    """以 median/MAD 计算稳健 z 分数；常量数组返回0而不是无穷值。"""

    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    if not np.isfinite(mad) or mad == 0:
        return np.zeros_like(values)
    return (values - median) / (1.4826 * mad)


def longest_flat_duration_s(data_uv: np.ndarray, sfreq: float, tolerance_uv: float) -> float:
    """返回相邻采样点变化不超过容差的最长连续时间。"""

    if data_uv.size < 2:
        return 0.0
    flat = np.abs(np.diff(data_uv)) <= tolerance_uv
    padded = np.concatenate(([False], flat, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    run_lengths = changes[1::2] - changes[::2]
    return float(run_lengths.max() + 1) / sfreq if run_lengths.size else 0.0


def _band_power(psd: np.ndarray, frequencies: np.ndarray, low: float, high: float) -> np.ndarray:
    mask = (frequencies >= low) & (frequencies < high)
    if mask.sum() < 2:
        return np.full(psd.shape[0], np.nan)
    return np.trapezoid(psd[:, mask], frequencies[mask], axis=1)


def _safe_ratio_db(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    tiny = np.finfo(float).tiny
    return 10 * np.log10(np.maximum(numerator, tiny) / np.maximum(denominator, tiny))


def _load_channel_types(raw: mne.io.BaseRaw, channels_path: Path) -> None:
    if not channels_path.is_file():
        return
    type_map: dict[str, str] = {}
    for row in _read_tsv(channels_path):
        name = row.get("name", "")
        channel_type = row.get("type", "").lower()
        if name in raw.ch_names and channel_type in {"eeg", "eog"}:
            type_map[name] = channel_type
    if type_map:
        raw.set_channel_types(type_map, verbose="ERROR")


def compute_signal_qc(
    set_path: Path,
    *,
    subject: str,
    signal_config: dict[str, Any],
    thresholds: dict[str, Any],
) -> SignalQCResult:
    """逐人加载连续数据，计算未滤波通道指标并释放内存。"""

    raw = mne.io.read_raw_eeglab(set_path, preload=False, verbose="ERROR")
    prefix = set_path.name.removesuffix("_eeg.set")
    _load_channel_types(raw, set_path.with_name(f"{prefix}_channels.tsv"))
    sfreq = float(raw.info["sfreq"])
    data_v = raw.get_data()
    data_uv = data_v * 1e6
    channel_types = raw.get_channel_types()
    eeg_indices = np.array([index for index, kind in enumerate(channel_types) if kind == "eeg"])
    eog_indices = np.array([index for index, kind in enumerate(channel_types) if kind == "eog"])

    finite_mask = np.isfinite(data_uv)
    cleaned_uv = np.where(finite_mask, data_uv, np.nan)
    channel_medians = np.nanmedian(cleaned_uv, axis=1, keepdims=True)
    psd_input_v = np.where(finite_mask, data_v, channel_medians / 1e6)
    window_samples = max(256, round(float(signal_config["welch_window_s"]) * sfreq))
    overlap_samples = round(window_samples * float(signal_config["welch_overlap_fraction"]))
    frequencies, psd_v2 = welch(
        psd_input_v,
        fs=sfreq,
        window="hann",
        nperseg=window_samples,
        noverlap=overlap_samples,
        detrend="constant",
        axis=1,
        average="median",
    )
    frequency_mask = frequencies <= float(signal_config["psd_fmax_hz"])
    frequencies = frequencies[frequency_mask]
    psd_uv2 = psd_v2[:, frequency_mask] * 1e12

    slow_power = _band_power(psd_uv2, frequencies, 0.1, 1.0)
    brain_power = _band_power(psd_uv2, frequencies, 1.0, 30.0)
    high_power = _band_power(psd_uv2, frequencies, 30.0, 45.0)
    line_power = _band_power(psd_uv2, frequencies, 58.0, 62.0)
    line_neighbors = _band_power(psd_uv2, frequencies, 54.0, 58.0) + _band_power(
        psd_uv2, frequencies, 62.0, 66.0
    )
    slow_ratio_db = _safe_ratio_db(slow_power, brain_power)
    high_ratio_db = _safe_ratio_db(high_power, brain_power)
    line_ratio_db = _safe_ratio_db(line_power, line_neighbors)

    correlation_step = max(1, round(sfreq / float(signal_config["correlation_target_sfreq_hz"])))
    eeg_for_correlation = cleaned_uv[eeg_indices, ::correlation_step]
    eeg_for_correlation -= np.nanmedian(eeg_for_correlation, axis=1, keepdims=True)
    correlation = np.corrcoef(np.nan_to_num(eeg_for_correlation))
    np.fill_diagonal(correlation, np.nan)
    median_abs_correlation = np.nanmedian(np.abs(correlation), axis=1)

    metrics: list[dict[str, Any]] = []
    for index, channel_name in enumerate(raw.ch_names):
        finite_data = cleaned_uv[index, finite_mask[index]]
        metrics.append(
            {
                "subject": subject,
                "channel": channel_name,
                "channel_type": channel_types[index].upper(),
                "nonfinite_fraction": 1.0 - float(finite_mask[index].mean()),
                "std_uv": float(np.std(finite_data)) if finite_data.size else np.nan,
                "variance_uv2": float(np.var(finite_data)) if finite_data.size else np.nan,
                "peak_to_peak_uv": float(np.ptp(finite_data)) if finite_data.size else np.nan,
                "robust_peak_to_peak_uv": (
                    float(np.percentile(finite_data, 99.9) - np.percentile(finite_data, 0.1))
                    if finite_data.size
                    else np.nan
                ),
                "median_abs_diff_uv": (
                    float(np.median(np.abs(np.diff(finite_data))))
                    if finite_data.size > 1
                    else np.nan
                ),
                "flat_longest_s": longest_flat_duration_s(
                    finite_data,
                    sfreq,
                    float(signal_config["flat_diff_uv"]),
                ),
                "slow_power_ratio_db": float(slow_ratio_db[index]),
                "high_frequency_ratio_db": float(high_ratio_db[index]),
                "line_noise_ratio_db": float(line_ratio_db[index]),
                "median_abs_correlation": (
                    float(median_abs_correlation[np.flatnonzero(eeg_indices == index)[0]])
                    if index in eeg_indices
                    else np.nan
                ),
            }
        )

    eeg_rows = [row for row in metrics if row["channel_type"] == "EEG"]
    log_variance = np.log10([max(row["variance_uv2"], np.finfo(float).tiny) for row in eeg_rows])
    variance_z = robust_zscore(log_variance)
    high_z = robust_zscore(np.array([row["high_frequency_ratio_db"] for row in eeg_rows]))
    line_z = robust_zscore(np.array([row["line_noise_ratio_db"] for row in eeg_rows]))
    for row, variance_score, high_score, line_score in zip(
        eeg_rows, variance_z, high_z, line_z, strict=True
    ):
        row["log_variance_robust_z"] = float(variance_score)
        row["high_frequency_robust_z"] = float(high_score)
        row["line_noise_robust_z"] = float(line_score)

    for row in metrics:
        row.setdefault("log_variance_robust_z", np.nan)
        row.setdefault("high_frequency_robust_z", np.nan)
        row.setdefault("line_noise_robust_z", np.nan)
        flags: list[str] = []
        if row["nonfinite_fraction"] > 0:
            flags.append("nonfinite")
        if thresholds.get("frozen") and row["channel_type"] == "EEG":
            if row["flat_longest_s"] >= float(thresholds["flat_duration_s"]):
                flags.append("flat")
            z_limit = float(thresholds["robust_z_abs"])
            if abs(row["log_variance_robust_z"]) >= z_limit:
                flags.append("variance_outlier")
            if row["high_frequency_robust_z"] >= z_limit:
                flags.append("high_frequency_outlier")
            if row["line_noise_robust_z"] >= z_limit:
                flags.append("line_noise_outlier")
            if row["median_abs_correlation"] < float(
                thresholds["median_abs_correlation_min"]
            ):
                flags.append("low_correlation")
        statistical_flags = [
            flag
            for flag in flags
            if flag
            in {
                "variance_outlier",
                "high_frequency_outlier",
                "line_noise_outlier",
                "low_correlation",
            }
        ]
        critical = any(flag in {"nonfinite", "flat"} for flag in flags)
        row["automatic_flags"] = ";".join(flags)
        row["bad_channel_candidate"] = row["channel_type"] == "EEG" and (
            critical
            or len(statistical_flags)
            >= int(thresholds.get("candidate_min_statistical_flags", 2))
        )

    bad_channels = [row["channel"] for row in eeg_rows if row["bad_channel_candidate"]]
    max_bad_fraction = float(thresholds.get("max_bad_channel_fraction", 0.2))
    summary = {
        "subject": subject,
        "recording_duration_s": raw.times[-1] if raw.n_times else 0.0,
        "sampling_frequency_hz": sfreq,
        "eeg_channels": len(eeg_indices),
        "eog_channels": len(eog_indices),
        "bad_channel_candidates": len(bad_channels),
        "bad_channel_names": ";".join(bad_channels),
        "bad_channel_fraction": len(bad_channels) / len(eeg_indices) if len(eeg_indices) else 1.0,
        "automatic_exclusion_candidate": (
            len(bad_channels) / len(eeg_indices) > max_bad_fraction if len(eeg_indices) else True
        ),
    }

    trace_samples = min(raw.n_times, round(float(signal_config["trace_window_s"]) * sfreq))
    trace_start = max(0, raw.n_times // 2 - trace_samples // 2)
    plot_data = {
        "frequencies": frequencies,
        "psd_uv2": psd_uv2,
        "data_uv": data_uv[:, trace_start : trace_start + trace_samples],
        "trace_times": np.arange(trace_samples) / sfreq,
        "channel_names": raw.ch_names,
        "eeg_indices": eeg_indices,
        "eog_indices": eog_indices,
    }
    return SignalQCResult(metrics, summary, plot_data)


def _plot_subject_qc(
    *,
    subject: str,
    signal_result: SignalQCResult,
    behavior_trials: list[dict[str, Any]],
    behavior_summary: dict[str, Any],
    output_path: Path,
) -> None:
    metrics = pd.DataFrame(signal_result.channel_metrics)
    eeg_metrics = metrics[metrics["channel_type"] == "EEG"].reset_index(drop=True)
    plot_data = signal_result.plot_data
    eeg_indices = plot_data["eeg_indices"]
    eog_indices = plot_data["eog_indices"]
    data_uv = plot_data["data_uv"]
    # BioSemi 连续记录允许较大的通道直流偏置；显示时居中，但QC计算仍使用未改动数据。
    display_data_uv = data_uv - np.nanmedian(data_uv, axis=1, keepdims=True)

    fig, axes = plt.subplots(2, 4, figsize=(22, 11), constrained_layout=True)
    fig.suptitle(f"{subject} | Raw P3 quality control (unfiltered)", fontsize=16)

    eeg_window = display_data_uv[eeg_indices]
    clip = max(50.0, min(250.0, float(np.nanpercentile(np.abs(eeg_window), 99))))
    image = axes[0, 0].imshow(
        np.clip(eeg_window, -clip, clip),
        aspect="auto",
        origin="upper",
        extent=[0, plot_data["trace_times"][-1], len(eeg_indices), 0],
        cmap="RdBu_r",
        vmin=-clip,
        vmax=clip,
    )
    axes[0, 0].set(title=f"Middle {plot_data['trace_times'][-1]:.1f} s EEG (µV)", xlabel="s")
    axes[0, 0].set_yticks(np.arange(len(eeg_indices)) + 0.5)
    axes[0, 0].set_yticklabels([plot_data["channel_names"][i] for i in eeg_indices], fontsize=6)
    fig.colorbar(image, ax=axes[0, 0], shrink=0.75)

    for index in eog_indices:
        axes[0, 1].plot(
            plot_data["trace_times"],
            display_data_uv[index],
            label=plot_data["channel_names"][index],
            linewidth=0.8,
        )
    axes[0, 1].set(title="EOG in the same window", xlabel="s", ylabel="µV")
    axes[0, 1].legend(fontsize=7)

    frequency_mask = (plot_data["frequencies"] >= 0.5) & (plot_data["frequencies"] <= 100)
    eeg_psd = plot_data["psd_uv2"][eeg_indices][:, frequency_mask]
    frequencies = plot_data["frequencies"][frequency_mask]
    psd_db = 10 * np.log10(np.maximum(eeg_psd, np.finfo(float).tiny))
    axes[0, 2].plot(frequencies, psd_db.T, color="0.75", linewidth=0.45, alpha=0.55)
    axes[0, 2].plot(frequencies, np.median(psd_db, axis=0), color="black", linewidth=1.5)
    axes[0, 2].axvline(60, color="red", linestyle="--", linewidth=1)
    axes[0, 2].set(title="Welch PSD (no notch filter)", xlabel="Hz", ylabel="dB µV²/Hz")

    info_lines = [
        f"Duration: {signal_result.summary['recording_duration_s']:.1f} s",
        f"Sampling: {signal_result.summary['sampling_frequency_hz']:.0f} Hz",
        f"EEG/EOG: {signal_result.summary['eeg_channels']}/{signal_result.summary['eog_channels']}",
        (
            f"Target/Standard: {behavior_summary['target_trials']}/"
            f"{behavior_summary['standard_trials']}"
        ),
        (
            f"Correct/Incorrect: {behavior_summary['correct_trials']}/"
            f"{behavior_summary['incorrect_trials']}"
        ),
        (
            f"Missing/Multiple: {behavior_summary['missing_response_trials']}/"
            f"{behavior_summary['multiple_response_trials']}"
        ),
        f"Unmatched responses: {behavior_summary['unmatched_responses']}",
        f"Median correct RT: {behavior_summary['correct_rt_median_s']:.3f} s",
        f"Bad-channel candidates: {signal_result.summary['bad_channel_candidates']}",
    ]
    axes[0, 3].axis("off")
    axes[0, 3].text(0.02, 0.98, "\n".join(info_lines), va="top", family="monospace")

    colors = np.where(eeg_metrics["bad_channel_candidate"], "tab:red", "tab:blue")
    axes[1, 0].bar(eeg_metrics["channel"], eeg_metrics["std_uv"], color=colors)
    axes[1, 0].set(title="Channel standard deviation", ylabel="µV")
    axes[1, 0].tick_params(axis="x", rotation=90, labelsize=6)

    axes[1, 1].bar(
        eeg_metrics["channel"], eeg_metrics["high_frequency_ratio_db"], color=colors
    )
    axes[1, 1].set(title="30–45 Hz / 1–30 Hz", ylabel="dB")
    axes[1, 1].tick_params(axis="x", rotation=90, labelsize=6)

    axes[1, 2].bar(
        eeg_metrics["channel"], eeg_metrics["median_abs_correlation"], color=colors
    )
    axes[1, 2].set(title="Median absolute channel correlation", ylim=(0, 1))
    axes[1, 2].tick_params(axis="x", rotation=90, labelsize=6)

    trial_frame = pd.DataFrame(behavior_trials)
    for condition, color in (("target", "tab:orange"), ("standard", "tab:green")):
        values = trial_frame.loc[
            (trial_frame["condition"] == condition)
            & np.isfinite(trial_frame["reaction_time_s"]),
            "reaction_time_s",
        ]
        axes[1, 3].hist(
            values,
            bins=np.linspace(0, 1.6, 33),
            alpha=0.55,
            label=condition,
            color=color,
        )
    axes[1, 3].axvline(0.15, color="red", linestyle="--", linewidth=1)
    axes[1, 3].set(title="Reaction-time distribution", xlabel="s", ylabel="Trials")
    axes[1, 3].legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _write_frame(frame: pd.DataFrame, path: Path, *, separator: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", sep=separator)


def _preserve_manual_review(
    frame: pd.DataFrame,
    path: Path,
    *,
    key_columns: list[str],
    manual_columns: list[str],
) -> pd.DataFrame:
    """重新运行自动QC时保留人工结论，避免覆盖不可重建的复核记录。"""

    for column in manual_columns:
        frame[column] = ""
    if not path.is_file():
        return frame
    existing = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    if not set(key_columns + manual_columns) <= set(existing.columns):
        return frame
    lookup = existing.set_index(key_columns)
    for index, row in frame.iterrows():
        key = tuple(str(row[column]) for column in key_columns)
        lookup_key: Any = key[0] if len(key) == 1 else key
        if lookup_key in lookup.index:
            for column in manual_columns:
                frame.at[index, column] = lookup.loc[lookup_key, column]
    return frame


def run_raw_qc(
    *,
    bids_root: Path,
    config_path: Path,
    output_dir: Path,
    figures_dir: Path,
    subjects: set[str] | None = None,
    make_figures: bool = True,
) -> dict[str, Any]:
    """串行处理受试者并写出trial、通道、受试者三级QC结果。"""

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    recordings = discover_p3_recordings(bids_root)
    if subjects:
        recordings = [(subject, path) for subject, path in recordings if subject in subjects]
    if not recordings:
        raise FileNotFoundError("没有发现符合条件的 P3 .set 记录")

    all_trials: list[dict[str, Any]] = []
    behavior_summaries: list[dict[str, Any]] = []
    channel_metrics: list[dict[str, Any]] = []
    subject_summaries: list[dict[str, Any]] = []

    for position, (subject, set_path) in enumerate(recordings, start=1):
        print(f"[{position:02d}/{len(recordings):02d}] {subject}", flush=True)
        prefix = set_path.name.removesuffix("_eeg.set")
        event_rows = _read_tsv(set_path.with_name(f"{prefix}_events.tsv"))
        trials, behavior_summary = pair_behavior_events(
            event_rows,
            subject=subject,
            min_reaction_time_s=float(config["behavior"]["min_reaction_time_s"]),
            max_response_window_s=float(config["behavior"]["max_response_window_s"]),
        )
        signal_result = compute_signal_qc(
            set_path,
            subject=subject,
            signal_config=config["signal"],
            thresholds=config["thresholds"],
        )
        combined_summary = {**signal_result.summary, **behavior_summary}
        all_trials.extend(trials)
        behavior_summaries.append(behavior_summary)
        channel_metrics.extend(signal_result.channel_metrics)
        subject_summaries.append(combined_summary)
        if make_figures:
            _plot_subject_qc(
                subject=subject,
                signal_result=signal_result,
                behavior_trials=trials,
                behavior_summary=behavior_summary,
                output_path=figures_dir / f"{subject}_raw_qc.png",
            )

    trials_frame = pd.DataFrame(all_trials)
    behavior_frame = pd.DataFrame(behavior_summaries)
    channel_frame = pd.DataFrame(channel_metrics)
    subject_frame = pd.DataFrame(subject_summaries)
    candidate_frame = channel_frame[channel_frame["bad_channel_candidate"]].copy()
    exclusion_frame = subject_frame[
        [
            "subject",
            "automatic_exclusion_candidate",
            "bad_channel_candidates",
            "bad_channel_fraction",
            "bad_channel_names",
        ]
    ].copy()
    exclusion_frame["automatic_reason"] = np.where(
        exclusion_frame["automatic_exclusion_candidate"],
        "bad_channel_fraction_exceeds_configured_maximum",
        "",
    )

    channel_review_path = output_dir / "bad_channel_review.tsv"
    channel_review = candidate_frame[["subject", "channel", "automatic_flags"]].copy()
    channel_review = _preserve_manual_review(
        channel_review,
        channel_review_path,
        key_columns=["subject", "channel"],
        manual_columns=["manual_decision", "manual_reason"],
    )
    exclusion_path = output_dir / "exclusion_log.tsv"
    exclusion_frame = _preserve_manual_review(
        exclusion_frame,
        exclusion_path,
        key_columns=["subject"],
        manual_columns=["manual_decision", "manual_reason"],
    )

    _write_frame(trials_frame, output_dir / "behavior_trials.csv")
    _write_frame(behavior_frame, output_dir / "behavior_summary.csv")
    _write_frame(channel_frame, output_dir / "channel_metrics.csv")
    _write_frame(subject_frame, output_dir / "subject_qc.csv")
    _write_frame(candidate_frame, output_dir / "bad_channel_candidates.csv")
    _write_frame(channel_review, channel_review_path, separator="\t")
    _write_frame(exclusion_frame, exclusion_path, separator="\t")

    summary = {
        "bids_root": str(bids_root.resolve()),
        "subjects_processed": len(recordings),
        "figures_created": len(recordings) if make_figures else 0,
        "thresholds_frozen": bool(config["thresholds"].get("frozen")),
        "bad_channel_candidates": int(candidate_frame.shape[0]),
        "subject_exclusion_candidates": int(
            subject_frame["automatic_exclusion_candidate"].sum()
        ),
        "behavior": {
            "stimulus_trials": int(behavior_frame["stimulus_trials"].sum()),
            "correct_trials": int(behavior_frame["correct_trials"].sum()),
            "incorrect_trials": int(behavior_frame["incorrect_trials"].sum()),
            "missing_response_trials": int(behavior_frame["missing_response_trials"].sum()),
            "multiple_response_trials": int(behavior_frame["multiple_response_trials"].sum()),
            "anticipatory_trials": int(behavior_frame["anticipatory_trials"].sum()),
            "unmatched_responses": int(behavior_frame["unmatched_responses"].sum()),
        },
        "thresholds": config["thresholds"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ERP CORE P3 原始行为与连续信号质量控制。")
    parser.add_argument("--bids-root", type=Path, default=Path("data/raw/erp-core-p3"))
    parser.add_argument("--config", type=Path, default=Path("configs/qc.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/raw_qc/final"))
    parser.add_argument(
        "--figures-dir", type=Path, default=Path("reports/figures/raw_qc/final")
    )
    parser.add_argument("--subjects", nargs="*", help="可选的 sub-XXX 列表；默认全部。")
    parser.add_argument("--no-figures", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    summary = run_raw_qc(
        bids_root=args.bids_root,
        config_path=args.config,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        subjects=set(args.subjects) if args.subjects else None,
        make_figures=not args.no_figures,
    )
    print(f"QC subjects: {summary['subjects_processed']}")
    print(f"Thresholds frozen: {summary['thresholds_frozen']}")
    print(f"Bad-channel candidates: {summary['bad_channel_candidates']}")
    print(f"Subject exclusion candidates: {summary['subject_exclusion_candidates']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

