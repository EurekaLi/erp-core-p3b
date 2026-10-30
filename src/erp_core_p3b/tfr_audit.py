"""审计TFR可行性，复用裁决版trial身份而不读取或修改EEG信号数组。

记录可用、卷积边缘完整、邻近刺激不进入支持范围是三个独立条件。这里用本机MNE
实际生成的小波核计算支持长度，避免把n_cycles/f误当作完整核跨度。方法依据见
https://mne.tools/stable/generated/mne.time_frequency.morlet.html 。
"""

# ruff: noqa: E402 -- pyplot导入前需把Matplotlib缓存设到项目可写目录。

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
import yaml

from erp_core_p3b.raw_qc import (
    _read_tsv,
    discover_p3_recordings,
    pair_behavior_events,
)
from erp_core_p3b.validation import classify_p3_event


def sampled_times(window_s: Sequence[float], sfreq_hz: float) -> np.ndarray:
    """按MNE epoch端点的就近取整约定，返回包含两端的相对时间轴。"""
    start, end = map(float, window_s)
    if start >= end or sfreq_hz <= 0:
        raise ValueError("时间窗必须递增，且采样率必须为正")
    return np.arange(round(start * sfreq_hz), round(end * sfreq_hz) + 1) / sfreq_hz


def recording_window_validity(
    event_samples: np.ndarray,
    *,
    n_times: int,
    sfreq_hz: float,
    window_s: Sequence[float],
) -> np.ndarray:
    """在未来降采样后的整数采样网格检查两端，正确处理最后一个可用采样点。"""
    offsets = np.rint(np.asarray(window_s) * sfreq_hz).astype(np.int64)
    return (event_samples + offsets[0] >= 0) & (event_samples + offsets[1] < n_times)


def annotation_overlap(
    onsets_s: np.ndarray,
    window_s: Sequence[float],
    annotations: Sequence[tuple[float, float, str]],
) -> np.ndarray:
    """标记与BAD区间或EEGLAB边界点相交的窗口，普通刺激/按键注释不参与。"""
    overlap = np.zeros(len(onsets_s), dtype=bool)
    left = onsets_s + window_s[0]
    right = onsets_s + window_s[1]
    for onset, duration, description in annotations:
        if not (description.lower().startswith("bad") or "boundary" in description.lower()):
            continue
        if duration == 0:
            # 数据断点没有持续时间，但跨过断点的卷积仍不具有连续信号含义。
            overlap |= (left <= onset) & (onset <= right)
        else:
            overlap |= (left < onset + duration) & (right > onset)
    return overlap


def map_retained_trials(
    metadata_frame: pd.DataFrame,
    trials: Sequence[dict[str, Any]],
    *,
    sfreq_hz: float,
    event_samples: np.ndarray,
    event_codes: np.ndarray,
) -> pd.DataFrame:
    """按原始trial_index匹配保留trial；相邻事件从全部刺激计算，包含被QC剔除的trial。"""
    retained = metadata_frame.copy().reset_index(drop=True)
    required = {
        "trial_index",
        "subject",
        "stimulus_onset_s",
        "stimulus_code",
        "condition_label",
        "event_sample",
    }
    if not required.issubset(retained.columns):
        raise ValueError(f"epoch元数据缺少字段: {sorted(required - set(retained.columns))}")
    if retained["trial_index"].duplicated().any():
        raise ValueError("保留trial的trial_index不唯一")
    original = pd.DataFrame(trials).set_index("trial_index")
    if not original.index.is_unique:
        raise ValueError("原始事件trial_index不唯一")
    missing = set(retained["trial_index"]) - set(original.index)
    if missing:
        raise ValueError(f"保留trial在原始事件中不存在: {sorted(missing)}")
    aligned = original.loc[retained["trial_index"]].reset_index()
    if not np.allclose(
        retained["stimulus_onset_s"].to_numpy(),
        aligned["stimulus_onset_s"].to_numpy(),
        atol=1e-7,
        rtol=0,
    ):
        raise ValueError("保留trial刺激时间与原始事件不一致")
    if not np.array_equal(retained["stimulus_code"], aligned["stimulus_code"]):
        raise ValueError("保留trial刺激码与原始事件不一致")
    expected_labels = np.where(aligned["condition"] == "target", "Target", "Standard")
    if not np.array_equal(retained["condition_label"], expected_labels):
        raise ValueError("保留trial条件与原始刺激条件不一致")
    expected_samples = np.rint(aligned["stimulus_onset_s"].to_numpy() * sfreq_hz).astype(int)
    if not np.array_equal(event_samples, expected_samples) or not np.array_equal(
        retained["event_sample"].to_numpy(), expected_samples
    ):
        raise ValueError("保留trial采样点与Phase 4降采样事件约定不一致")
    if not np.array_equal(event_codes, np.where(expected_labels == "Target", 1, 2)):
        raise ValueError("epoch事件码与元数据条件不一致")
    if not aligned["valid_behavior_trial"].all():
        raise ValueError("裁决版中出现行为未通过的trial")

    all_onsets = original["stimulus_onset_s"].to_numpy(dtype=float)
    if np.any(np.diff(all_onsets) <= 0):
        raise ValueError("全部原始刺激时间必须严格递增")
    previous = np.r_[np.nan, np.diff(all_onsets)]
    following = np.r_[np.diff(all_onsets), np.nan]
    gaps = (
        pd.DataFrame({"previous_gap_s": previous, "next_gap_s": following}, index=original.index)
        .loc[retained["trial_index"]]
        .reset_index(drop=True)
    )
    return pd.concat([retained, gaps], axis=1)


def morlet_support_table(
    *,
    sfreq_hz: float,
    frequencies_hz: np.ndarray,
    config: dict[str, Any],
    epoch_window_s: Sequence[float],
    baseline_window_s: Sequence[float],
    display_window_s: Sequence[float],
) -> pd.DataFrame:
    """计算完整核支持和FWHM，并分别验证基线、展示窗和当前刺激的时间交叠。"""
    cycles = np.clip(
        frequencies_hz / float(config["cycle_divisor_hz"]),
        float(config["min_cycles"]),
        float(config["max_cycles"]),
    )
    wavelets = mne.time_frequency.morlet(
        sfreq_hz, frequencies_hz, n_cycles=cycles, zero_mean=bool(config["zero_mean"])
    )
    epoch_times = sampled_times(epoch_window_s, sfreq_hz)
    baseline_times = epoch_times[
        (epoch_times >= baseline_window_s[0]) & (epoch_times <= baseline_window_s[1])
    ]
    # 展示裁剪沿用MNE就近端点；基线则按数值区间选点，二者不能混用。
    display_times = sampled_times(display_window_s, sfreq_hz)
    if baseline_times.size == 0:
        raise ValueError("基线必须与epoch时间轴相交")
    if display_times[0] < epoch_times[0] or display_times[-1] > epoch_times[-1]:
        raise ValueError("展示窗必须完整位于epoch时间轴内")
    rows: list[dict[str, Any]] = []
    for frequency, n_cycles, kernel in zip(frequencies_hz, cycles, wavelets, strict=True):
        half_samples = (len(kernel) - 1) // 2
        half_s = half_samples / sfreq_hz
        sigma_s = n_cycles / (2 * np.pi * frequency)
        rows.append(
            {
                "frequency_hz": float(frequency),
                "n_cycles": float(n_cycles),
                "kernel_samples": len(kernel),
                "kernel_span_s": 2 * half_s,
                "half_support_s": half_s,
                "envelope_fwhm_s": float(2 * np.sqrt(2 * np.log(2)) * sigma_s),
                "kernel_fits_epoch": len(kernel) <= len(epoch_times),
                "baseline_support_start_s": float(baseline_times[0] - half_s),
                "baseline_support_end_s": float(baseline_times[-1] + half_s),
                "baseline_edge_safe": bool(
                    baseline_times[0] - half_s >= epoch_times[0]
                    and baseline_times[-1] + half_s <= epoch_times[-1]
                ),
                "baseline_reaches_current_stimulus": bool(baseline_times[-1] + half_s >= 0),
                "display_support_start_s": float(display_times[0] - half_s),
                "display_support_end_s": float(display_times[-1] + half_s),
                "display_edge_safe": bool(
                    display_times[0] - half_s >= epoch_times[0]
                    and display_times[-1] + half_s <= epoch_times[-1]
                ),
            }
        )
    return pd.DataFrame(rows)


def neighbor_overlap_flags(
    trials: pd.DataFrame,
    *,
    left_s: float,
    right_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """检查支持窗是否触及前一刺激呈现区间或下一刺激起点，不把重叠自动视作剔除。"""
    previous_onsets = -trials["previous_gap_s"].to_numpy(dtype=float)
    previous_ends = previous_onsets + trials["previous_stimulus_duration_s"].to_numpy()
    next_onsets = trials["next_gap_s"].to_numpy(dtype=float)
    previous_overlap = (previous_onsets <= right_s) & (previous_ends >= left_s)
    next_overlap = (next_onsets >= left_s) & (next_onsets <= right_s)
    return previous_overlap, next_overlap


def estimate_memory_bytes(
    *,
    n_trials: int,
    n_channels: int,
    n_times: int,
    n_frequencies: int,
    raw_n_times: int,
    raw_n_channels: int,
    raw_copies: int,
    batch_trials: int,
) -> dict[str, int]:
    """估算主要数组载荷；FFT、Python对象和底层库工作区不在该模型内。"""
    epoch_elements = n_trials * n_channels * n_times
    power_elements = epoch_elements * n_frequencies
    batch_elements = min(batch_trials, n_trials) * n_channels * n_times * n_frequencies
    raw_bytes = raw_n_times * raw_n_channels * 8
    return {
        "epoch_float32_bytes": epoch_elements * 4,
        "epoch_float64_bytes": epoch_elements * 8,
        "full_power_float64_bytes": power_elements * 8,
        "full_complex128_bytes": power_elements * 16,
        "batch_power_float64_bytes": batch_elements * 8,
        "batch_complex128_bytes": batch_elements * 16,
        "raw_float64_bytes": raw_bytes,
        "modeled_working_bytes": (
            raw_bytes * raw_copies + epoch_elements * 8 + batch_elements * 24
        ),
    }


def _save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _plot_audit(
    *,
    trials: pd.DataFrame,
    intervals: pd.DataFrame,
    candidates: pd.DataFrame,
    memory: pd.DataFrame,
    support: pd.DataFrame,
    figures_root: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    within_block = intervals.loc[~intervals["is_long_gap"], "interval_s"]
    axes[0, 0].hist(within_block, bins=40, color="tab:blue", edgecolor="white")
    axes[0, 0].set(
        title="All stimulus intervals (short gaps)", xlabel="Interval (s)", ylabel="Count"
    )
    for values, label, color in (
        (trials["previous_gap_s"], "Previous", "tab:blue"),
        (trials["next_gap_s"], "Next", "tab:orange"),
    ):
        finite_values = values[np.isfinite(values) & (values < 5)]
        axes[0, 1].hist(finite_values, bins=35, histtype="step", label=label, color=color)
    axes[0, 1].set(title="Neighbors of retained trials", xlabel="Gap (s)", ylabel="Count")
    axes[0, 1].legend()
    axes[1, 0].bar(candidates["candidate"], candidates["retained_fraction"] * 100)
    axes[1, 0].set(
        title="Recording / annotation availability", ylabel="Retained (%)", ylim=(0, 105)
    )
    axes[1, 0].tick_params(axis="x", rotation=25, labelsize=8)
    grouped = memory.groupby("candidate")[["full_power_float64_gib", "modeled_working_gib"]].max()
    grouped.plot.bar(ax=axes[1, 1])
    axes[1, 1].set(title="Largest subject: full power vs batch workload", ylabel="GiB")
    axes[1, 1].tick_params(axis="x", rotation=25, labelsize=8)
    axes[1, 1].legend(["Full trial power", "Modeled batch workload"], fontsize=8)
    fig.savefig(figures_root / "epoch_feasibility.png", dpi=160)
    plt.close(fig)

    subset = support[support["candidate"] == "padded_long_2hz"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].plot(subset["frequency_hz"], subset["kernel_span_s"], label="Full kernel support")
    axes[0].plot(subset["frequency_hz"], subset["envelope_fwhm_s"], label="Envelope FWHM")
    axes[0].axhline(1, color="0.5", linestyle="--", label="Previous ERP epoch (1 s)")
    axes[0].set(title="Measured MNE Morlet kernels", xlabel="Frequency (Hz)", ylabel="Duration (s)")
    axes[0].legend()
    for candidate, color in (("padded_long_2hz", "tab:orange"), ("padded_long_6hz", "tab:blue")):
        part = support[support["candidate"] == candidate]
        axes[1].plot(
            part["frequency_hz"], part["baseline_support_end_s"], label=candidate, color=color
        )
    axes[1].axhline(0, color="black", linestyle="--", label="Current stimulus onset")
    axes[1].set(
        title="Latest data used by baseline convolution",
        xlabel="Frequency (Hz)",
        ylabel="Relative time (s)",
    )
    axes[1].legend()
    fig.savefig(figures_root / "morlet_support.png", dpi=160)
    plt.close(fig)


def run_tfr_audit(*, config_path: Path, subjects: Sequence[str] | None = None) -> dict[str, Any]:
    """读取头信息、事件和裁决元数据，生成可用率、支持范围、资源成本和复现清单。"""
    config_text = config_path.read_text(encoding="utf-8")
    config = yaml.safe_load(config_text)
    dataset = config["dataset"]
    sfreq = float(config["sampling"]["target_sfreq_hz"])
    files = sorted(Path(dataset["epochs_root"]).glob("sub-*_task-P3_epo.fif"))
    if len(files) != int(dataset["expected_subject_count"]):
        raise ValueError("裁决版文件数量与主分析人数不一致")
    if subjects:
        wanted = set(subjects)
        files = [path for path in files if path.name.split("_")[0] in wanted]
        if len(files) != len(wanted):
            raise ValueError("请求的受试者不全部存在于裁决版目录")
    recordings_list = discover_p3_recordings(Path(dataset["raw_root"]))
    recordings = dict(recordings_list)
    if len(recordings) != len(recordings_list):
        raise ValueError("同一受试者存在多份P3记录，无法唯一匹配")
    preprocessing = pd.read_csv(dataset["preprocessing_summary"]).set_index("subject")
    if not preprocessing.index.is_unique:
        raise ValueError("Phase 4汇总表受试者不唯一")
    support_parts: list[pd.DataFrame] = []
    for name, candidate in config["candidates"].items():
        frequencies = np.arange(
            candidate["fmin_hz"],
            config["morlet"]["fmax_hz"] + 0.01,
            config["morlet"]["frequency_step_hz"],
        )
        part = morlet_support_table(
            sfreq_hz=sfreq,
            frequencies_hz=frequencies,
            config=config["morlet"],
            epoch_window_s=candidate["epoch_window_s"],
            baseline_window_s=candidate["baseline_window_s"],
            display_window_s=config["sampling"]["display_window_s"],
        )
        part.insert(0, "candidate", name)
        support_parts.append(part)
    support = pd.concat(support_parts, ignore_index=True)

    trial_parts: list[pd.DataFrame] = []
    interval_parts: list[pd.DataFrame] = []
    candidate_trials: list[pd.DataFrame] = []
    memory_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    for position, epoch_path in enumerate(files, start=1):
        subject = epoch_path.name.split("_")[0]
        print(f"[{position:02d}/{len(files):02d}] audit {subject}", flush=True)
        if subject in dataset["excluded_subjects"]:
            raise ValueError(f"主分析中出现排除受试者: {subject}")
        set_path = recordings[subject]
        event_path = set_path.with_name(set_path.name.replace("_eeg.set", "_events.tsv"))
        event_rows = _read_tsv(event_path)
        all_trials, _ = pair_behavior_events(
            event_rows, subject=subject, min_reaction_time_s=0.15, max_response_window_s=1.6
        )
        epochs = mne.read_epochs(epoch_path, preload=False, verbose="ERROR")
        if epochs.metadata is None or float(epochs.info["sfreq"]) != sfreq:
            raise ValueError(f"{subject} 缺少元数据或采样率不符")
        if set(epochs.metadata["subject"]) != {subject}:
            raise ValueError(f"{subject} 文件与元数据受试者不一致")
        retained = map_retained_trials(
            epochs.metadata,
            all_trials,
            sfreq_hz=sfreq,
            event_samples=epochs.events[:, 0],
            event_codes=epochs.events[:, 2],
        )
        summary_row = preprocessing.loc[subject]
        for label, summary_column in (
            ("Target", "target_retained"),
            ("Standard", "standard_retained"),
        ):
            if (retained["condition_label"] == label).sum() != int(summary_row[summary_column]):
                raise ValueError(f"{subject} trial计数与Phase 4审计表不符")
        ica_path = Path(summary_row["ica_path"])
        if not ica_path.is_file():
            raise FileNotFoundError(f"{subject} 的冻结ICA模型缺失: {ica_path}")

        raw = mne.io.read_raw_eeglab(set_path, preload=False, verbose="ERROR")
        if raw.first_samp != 0:
            raise ValueError("原始EEGLAB首采样点非零，需要显式适配事件时间原点")
        annotations = list(
            zip(
                raw.annotations.onset,
                raw.annotations.duration,
                raw.annotations.description,
                strict=True,
            )
        )
        raw_sfreq = float(raw.info["sfreq"])
        resampled_n_times = int(round(raw.n_times * sfreq / raw_sfreq))
        original_onsets = np.array([trial["stimulus_onset_s"] for trial in all_trials])
        durations_by_onset = {
            float(row["onset"]): float(row["duration"])
            for row in event_rows
            if classify_p3_event(row["value"]) in {"target", "standard"}
        }
        original_durations = np.array([durations_by_onset[onset] for onset in original_onsets])
        prior_durations = np.r_[np.nan, original_durations[:-1]]
        retained["previous_stimulus_duration_s"] = prior_durations[
            retained["trial_index"].to_numpy(dtype=int) - 1
        ]
        onset_s = retained["stimulus_onset_s"].to_numpy(dtype=float)
        retained["recording_start_margin_s"] = onset_s
        retained["recording_end_margin_s"] = (raw.n_times - 1) / raw_sfreq - onset_s
        retained["matching_passed"] = True
        trial_parts.append(retained)
        intervals = np.diff(original_onsets)
        interval_parts.append(
            pd.DataFrame(
                {
                    "subject": subject,
                    "previous_trial_index": np.arange(1, len(all_trials)),
                    "next_trial_index": np.arange(2, len(all_trials) + 1),
                    "interval_s": intervals,
                    "is_long_gap": intervals >= float(config["memory"]["break_gap_s"]),
                }
            )
        )
        eeg_count = int(len(mne.pick_types(epochs.info, eeg=True, eog=False, exclude=[])))
        subject_rows.append(
            {
                "subject": subject,
                "raw_file": set_path.as_posix(),
                "epoch_file": epoch_path.as_posix(),
                "ica_file": ica_path.as_posix(),
                "raw_sfreq_hz": raw_sfreq,
                "raw_samples": raw.n_times,
                "recording_duration_s": raw.n_times / raw_sfreq,
                "resampled_samples": resampled_n_times,
                "eeg_channels": eeg_count,
                "retained_trials": len(retained),
                "target_trials": int((retained["condition_label"] == "Target").sum()),
                "standard_trials": int((retained["condition_label"] == "Standard").sum()),
                "raw_preloaded_by_reader": bool(raw.preload),
                "metadata_alignment_passed": True,
                "raw_first_samp": int(raw.first_samp),
                "discontinuity_or_bad_annotations": sum(
                    d.lower().startswith("bad") or "boundary" in d.lower()
                    for _, _, d in annotations
                ),
                "isi_min_s": float(intervals.min()),
                "isi_median_s": float(np.median(intervals)),
                "isi_max_s": float(intervals.max()),
            }
        )
        for name, candidate in config["candidates"].items():
            actual_times = sampled_times(candidate["epoch_window_s"], sfreq)
            actual_window = [float(actual_times[0]), float(actual_times[-1])]
            boundary_ok = recording_window_validity(
                retained["event_sample"].to_numpy(dtype=int),
                n_times=resampled_n_times,
                sfreq_hz=sfreq,
                window_s=candidate["epoch_window_s"],
            )
            annotation_bad = annotation_overlap(onset_s, actual_window, annotations)
            previous_overlap, next_overlap = neighbor_overlap_flags(
                retained, left_s=actual_window[0], right_s=actual_window[1]
            )
            part = retained[["subject", "trial_index", "condition_label"]].copy()
            part["candidate"] = name
            part["recording_boundary_ok"] = boundary_ok
            part["bad_annotation_overlap"] = annotation_bad
            part["available_by_geometry"] = boundary_ok & ~annotation_bad
            part["epoch_contains_previous_stimulus"] = previous_overlap
            part["epoch_contains_next_stimulus"] = next_overlap
            candidate_trials.append(part)
            n_frequencies = int((support["candidate"] == name).sum())
            memory = estimate_memory_bytes(
                n_trials=int((boundary_ok & ~annotation_bad).sum()),
                n_channels=eeg_count,
                n_times=len(actual_times),
                n_frequencies=n_frequencies,
                raw_n_times=raw.n_times,
                raw_n_channels=len(raw.ch_names),
                raw_copies=int(config["memory"]["raw_float64_copies"]),
                batch_trials=int(config["memory"]["batch_trials"]),
            )
            memory_rows.append(
                {
                    "subject": subject,
                    "candidate": name,
                    "samples_per_epoch": len(actual_times),
                    "n_frequencies": n_frequencies,
                    **{
                        key.replace("_bytes", "_gib"): value / 2**30
                        for key, value in memory.items()
                    },
                }
            )
        del raw, epochs

    trials = pd.concat(trial_parts, ignore_index=True)
    if trials.duplicated(["subject", "trial_index"]).any():
        raise ValueError("全体trial主键重复")
    if subjects is None and len(trials) != int(dataset["expected_trial_count"]):
        raise ValueError("裁决版trial总数与冻结的Phase 5计数不一致")
    interval_frame = pd.concat(interval_parts, ignore_index=True)
    trial_candidates = pd.concat(candidate_trials, ignore_index=True)
    memory_frame = pd.DataFrame(memory_rows)
    for index, row in support.iterrows():
        for field in ("baseline", "display"):
            previous, following = neighbor_overlap_flags(
                trials,
                left_s=row[f"{field}_support_start_s"],
                right_s=row[f"{field}_support_end_s"],
            )
            support.loc[index, f"{field}_previous_stimulus_overlap_fraction"] = previous.mean()
            support.loc[index, f"{field}_next_stimulus_overlap_fraction"] = following.mean()
        measurement = config["sampling"]["measurement_window_s"]
        previous, following = neighbor_overlap_flags(
            trials,
            left_s=measurement[0] - row["half_support_s"],
            right_s=measurement[1] + row["half_support_s"],
        )
        support.loc[index, "measurement_previous_stimulus_overlap_fraction"] = previous.mean()
        support.loc[index, "measurement_next_stimulus_overlap_fraction"] = following.mean()

    candidate_rows: list[dict[str, Any]] = []
    for name, candidate in config["candidates"].items():
        part = trial_candidates[trial_candidates["candidate"] == name]
        supported = support[support["candidate"] == name]
        available = part["available_by_geometry"]
        memory = memory_frame[memory_frame["candidate"] == name]
        candidate_rows.append(
            {
                "candidate": name,
                "epoch_tmin_s": candidate["epoch_window_s"][0],
                "epoch_tmax_s": candidate["epoch_window_s"][1],
                "baseline_tmin_s": candidate["baseline_window_s"][0],
                "baseline_tmax_s": candidate["baseline_window_s"][1],
                "fmin_hz": candidate["fmin_hz"],
                "input_trials": len(part),
                "available_trials": int(available.sum()),
                "retained_fraction": float(available.mean()),
                "target_available": int((available & (part["condition_label"] == "Target")).sum()),
                "standard_available": int(
                    (available & (part["condition_label"] == "Standard")).sum()
                ),
                "boundary_unavailable": int((~part["recording_boundary_ok"]).sum()),
                "annotation_unavailable": int(part["bad_annotation_overlap"].sum()),
                "all_kernels_fit_epoch": bool(supported["kernel_fits_epoch"].all()),
                "all_baseline_samples_edge_safe": bool(supported["baseline_edge_safe"].all()),
                "all_display_samples_edge_safe": bool(supported["display_edge_safe"].all()),
                "baseline_reaches_current_stimulus_any_frequency": bool(
                    supported["baseline_reaches_current_stimulus"].any()
                ),
                "display_next_stimulus_overlap_max_fraction": float(
                    supported["display_next_stimulus_overlap_fraction"].max()
                ),
                "all_subjects_epoch_f64_gib": float(memory["epoch_float64_gib"].sum()),
                "all_subjects_full_power_f64_gib": float(memory["full_power_float64_gib"].sum()),
                "largest_subject_modeled_working_gib": float(memory["modeled_working_gib"].max()),
            }
        )
    candidates_frame = pd.DataFrame(candidate_rows)
    full_run = subjects is None
    summary = {
        "audit_complete": True,
        "is_full_cohort": full_run,
        "subject_count": len(files),
        "retained_trials": len(trials),
        "target_trials": int((trials["condition_label"] == "Target").sum()),
        "standard_trials": int((trials["condition_label"] == "Standard").sum()),
        "metadata_alignment_passed": True,
        "interval_min_s": float(interval_frame["interval_s"].min()),
        "interval_median_s": float(interval_frame["interval_s"].median()),
        "interval_max_s": float(interval_frame["interval_s"].max()),
        "short_interval_quantiles_s": {
            str(q): float(
                interval_frame.loc[~interval_frame["is_long_gap"], "interval_s"].quantile(q)
            )
            for q in (0.0, 0.05, 0.5, 0.95, 1.0)
        },
        "long_gap_count": int(interval_frame["is_long_gap"].sum()),
        "memory_estimates_are_measured_peak": False,
        "signal_amplitude_in_extended_flanks_audited": False,
        "limitations": [
            "几何可用率不等同于长窗信号质量通过率，扩展时段的伪迹须在重建时检查。",
            "相邻事件不进入小波支持范围，并不证明此前ERP或按键相关活动完全消失。",
            "内存是主要数组模型，未包含FFT、Python及底层库全部开销。",
            "本程序未生成长epochs，也未计算条件TFR或重新拟合ICA。",
        ],
        "candidates": candidates_frame.to_dict("records"),
    }
    results_root = Path(config["outputs"]["results_root"])
    figures_root = Path(config["outputs"]["figures_root"])
    if subjects:
        # 局部试运行不能覆盖全体审计及冻结参数所依赖的正式结果。
        subset_name = "subset_" + "_".join(sorted(subjects))
        results_root /= subset_name
        figures_root /= subset_name
    results_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)
    for frame, name in (
        (pd.DataFrame(subject_rows), "subject_inventory"),
        (trials, "retained_trial_audit"),
        (interval_frame, "stimulus_intervals"),
        (trial_candidates, "trial_candidate_availability"),
        (candidates_frame, "candidate_summary"),
        (support, "morlet_support"),
        (memory_frame, "memory_estimates"),
    ):
        _save_csv(frame, results_root / f"{name}.csv")
    (results_root / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_path": config_path.as_posix(),
        "config_sha256": hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
        "mne_version": mne.__version__,
        "numpy_version": np.__version__,
        "subjects": [path.name.split("_")[0] for path in files],
        "source_paths": {
            "raw_root": dataset["raw_root"],
            "epochs_root": dataset["epochs_root"],
            "preprocessing_summary": dataset["preprocessing_summary"],
        },
        "all_inputs_read_only": True,
        "raw_signal_preloaded_by_any_reader": any(
            row["raw_preloaded_by_reader"] for row in subject_rows
        ),
    }
    (results_root / "audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _plot_audit(
        trials=trials,
        intervals=interval_frame,
        candidates=candidates_frame,
        memory=memory_frame,
        support=support,
        figures_root=figures_root,
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="审计长TFR epoch的时间支持和资源成本。")
    parser.add_argument("--config", type=Path, default=Path("configs/tfr_audit.yaml"))
    parser.add_argument("--subjects", nargs="+", help="指定试运行受试者；正式冻结使用完整39人。")
    args = parser.parse_args(argv)
    summary = run_tfr_audit(config_path=args.config, subjects=args.subjects)
    print(
        f"TFR audit complete | subjects={summary['subject_count']} | "
        f"trials={summary['retained_trials']} | min_ISI={summary['interval_min_s']:.4f} s"
    )
    return 0
