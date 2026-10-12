"""ERP CORE P3 的滤波、ICA、坏道插值和分段流水线。"""

# ruff: noqa: E402 -- Matplotlib 导入前必须把缓存重定向到项目内可写目录。

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
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
from scipy.signal import welch

from erp_core_p3b.raw_qc import (
    _load_channel_types,
    _read_tsv,
    discover_p3_recordings,
    pair_behavior_events,
)

EVENT_ID = {"Target": 1, "Standard": 2}


def load_manual_qc(path: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    """读取人工坏道和受试者排除结论，并标准化为便于查询的字典。"""

    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    bad_channels = {
        subject: list(entry.get("channels", []))
        for subject, entry in config.get("bad_channels", {}).items()
    }
    excluded_subjects = {
        subject: str(reason)
        for subject, reason in config.get("excluded_subjects", {}).items()
    }
    return bad_channels, excluded_subjects


def load_manual_ica_exclusions(path: Path) -> dict[str, list[int]]:
    """读取经地形和EOG相关共同确认的人工ICA成分编号。"""

    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {
        subject: [int(component) for component in components]
        for subject, components in config.get("ica_exclusions", {}).items()
    }


def make_epoch_events(
    trials: Sequence[dict[str, Any]],
    *,
    sfreq: float,
    first_samp: int,
    require_correct_response: bool,
    exclude_behavior_flags: bool,
) -> tuple[np.ndarray, pd.DataFrame]:
    """把行为trial转换为MNE事件矩阵，同时保留逐trial元数据。"""

    events: list[list[int]] = []
    metadata_rows: list[dict[str, Any]] = []
    for trial in trials:
        if require_correct_response and trial["accuracy"] != "correct":
            continue
        if exclude_behavior_flags and trial["behavior_qc_flag"]:
            continue
        label = "Target" if trial["condition"] == "target" else "Standard"
        sample = first_samp + int(round(float(trial["stimulus_onset_s"]) * sfreq))
        events.append([sample, 0, EVENT_ID[label]])
        metadata_rows.append(
            {
                **trial,
                "condition_label": label,
                "event_sample": sample,
            }
        )
    return np.asarray(events, dtype=int), pd.DataFrame(metadata_rows)


def choose_eog_components(
    component_scores: dict[str, np.ndarray],
    *,
    threshold: float,
    maximum: int,
) -> tuple[list[int], np.ndarray]:
    """按所有EOG通道中的最大绝对相关排序，并限制自动删除数量。"""

    if not component_scores:
        return [], np.array([], dtype=float)
    score_matrix = np.vstack(
        [np.asarray(scores, dtype=float) for scores in component_scores.values()]
    )
    maximum_scores = np.max(np.abs(score_matrix), axis=0)
    candidates = np.flatnonzero(maximum_scores >= threshold)
    ordered = candidates[np.argsort(maximum_scores[candidates])[::-1]]
    return ordered[:maximum].astype(int).tolist(), maximum_scores


def _prepare_raw(set_path: Path, bad_channels: list[str]) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")
    prefix = set_path.name.removesuffix("_eeg.set")
    _load_channel_types(raw, set_path.with_name(f"{prefix}_channels.tsv"))
    montage = mne.channels.make_standard_montage("colin27_1020")
    raw.set_montage(montage, match_case=False, on_missing="ignore", verbose="ERROR")
    missing = sorted(set(bad_channels) - set(raw.ch_names))
    if missing:
        raise ValueError(f"人工坏道不在数据中: {missing}")
    raw.info["bads"] = list(bad_channels)
    return raw


def _filter_branch(
    raw: mne.io.BaseRaw,
    *,
    highpass_hz: float,
    lowpass_hz: float,
    target_sfreq_hz: float,
) -> mne.io.BaseRaw:
    branch = raw.copy()
    branch.filter(
        l_freq=highpass_hz,
        h_freq=lowpass_hz,
        picks="data",
        phase="zero",
        fir_design="firwin",
        verbose="ERROR",
    )
    branch.resample(target_sfreq_hz, npad="auto", verbose="ERROR")
    branch.set_eeg_reference("average", projection=False, verbose="ERROR")
    return branch


def _fit_ica(
    raw: mne.io.BaseRaw,
    config: dict[str, Any],
) -> tuple[mne.preprocessing.ICA, list[int], dict[str, np.ndarray], np.ndarray]:
    ica = mne.preprocessing.ICA(
        n_components=float(config["n_components_variance"]),
        method=str(config["method"]),
        rng=int(config["random_state"]),
        max_iter=int(config["max_iter"]),
    )
    ica.fit(
        raw,
        picks="eeg",
        decim=int(config["fit_decim"]),
        reject_by_annotation=True,
        verbose="ERROR",
    )
    eog_scores: dict[str, np.ndarray] = {}
    for channel in mne.pick_info(raw.info, mne.pick_types(raw.info, eog=True)).ch_names:
        _, scores = ica.find_bads_eog(
            raw,
            ch_name=channel,
            threshold=float(config["eog_correlation_threshold"]),
            measure="correlation",
            l_freq=1.0,
            h_freq=10.0,
            verbose="ERROR",
        )
        eog_scores[channel] = np.asarray(scores, dtype=float)
    excluded, maximum_scores = choose_eog_components(
        eog_scores,
        threshold=float(config["eog_correlation_threshold"]),
        maximum=int(config["max_eog_components"]),
    )
    ica.exclude = excluded
    return ica, excluded, eog_scores, maximum_scores


def _epoch_peak_to_peak_uv(epochs: mne.Epochs) -> np.ndarray:
    eeg_data = epochs.get_data(picks="eeg", copy=False)
    return np.ptp(eeg_data, axis=-1).max(axis=1) * 1e6


def _drop_large_epochs(
    epochs: mne.Epochs,
    peak_to_peak_uv: np.ndarray,
    threshold_uv: float | None,
) -> np.ndarray:
    rejected = np.zeros(len(epochs), dtype=bool)
    if threshold_uv is None:
        return rejected
    rejected = peak_to_peak_uv > threshold_uv
    if rejected.any():
        epochs.drop(np.flatnonzero(rejected), reason="EEG_PTP", verbose="ERROR")
    return rejected


def _psd_curve(raw: mne.io.BaseRaw, channel: str) -> tuple[np.ndarray, np.ndarray]:
    data_uv = raw.get_data(picks=[channel])[0] * 1e6
    frequencies, psd = welch(
        data_uv,
        fs=float(raw.info["sfreq"]),
        nperseg=min(len(data_uv), round(4 * float(raw.info["sfreq"]))),
        noverlap=round(2 * float(raw.info["sfreq"])),
    )
    return frequencies, 10 * np.log10(np.maximum(psd, np.finfo(float).tiny))


def _plot_preprocessing_qc(
    *,
    subject: str,
    raw_psd: tuple[np.ndarray, np.ndarray],
    processed_psd: tuple[np.ndarray, np.ndarray],
    maximum_eog_scores: np.ndarray,
    excluded_components: list[int],
    peak_to_peak_uv: np.ndarray,
    rejection_threshold_uv: float | None,
    epochs: mne.Epochs,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle(f"{subject} | preprocessing QC", fontsize=15)

    for (frequencies, psd), label in ((raw_psd, "raw"), (processed_psd, "processed")):
        mask = (frequencies >= 0.1) & (frequencies <= 80)
        axes[0, 0].plot(frequencies[mask], psd[mask], label=label, linewidth=1.1)
    axes[0, 0].axvline(30, color="0.5", linestyle="--", linewidth=1)
    axes[0, 0].set(title="Pz PSD before/after", xlabel="Hz", ylabel="dB µV²/Hz")
    axes[0, 0].legend()

    component_indices = np.arange(len(maximum_eog_scores))
    colors = [
        "tab:red" if index in excluded_components else "tab:blue"
        for index in component_indices
    ]
    axes[0, 1].bar(component_indices, maximum_eog_scores, color=colors)
    axes[0, 1].set(title="Maximum |ICA–EOG correlation|", xlabel="ICA component")

    axes[1, 0].hist(peak_to_peak_uv, bins=30, color="tab:blue", alpha=0.75)
    if rejection_threshold_uv is not None:
        axes[1, 0].axvline(rejection_threshold_uv, color="red", linestyle="--")
    axes[1, 0].set(title="Behavior-valid epoch peak-to-peak", xlabel="maximum EEG µV")

    if "Pz" in epochs.ch_names and len(epochs):
        for condition, color in (("Target", "tab:orange"), ("Standard", "tab:green")):
            if condition not in epochs.event_id or len(epochs[condition]) == 0:
                continue
            evoked = epochs[condition].average(picks=["Pz"])
            axes[1, 1].plot(evoked.times, evoked.data[0] * 1e6, label=condition, color=color)
    axes[1, 1].axvline(0, color="black", linewidth=0.8)
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set(title="Pz condition averages", xlabel="s", ylabel="µV")
    axes[1, 1].legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_excluded_component_topographies(
    ica: mne.preprocessing.ICA,
    excluded_components: list[int],
    output_path: Path,
) -> None:
    """保存自动删除成分的空间分布，供人工确认其是否呈眼动模式。"""

    if not excluded_components:
        return
    figures = ica.plot_components(
        picks=excluded_components,
        ch_type="eeg",
        title="Automatically excluded EOG-related ICA components",
        show=False,
    )
    figure_list = figures if isinstance(figures, list) else [figures]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for page, figure in enumerate(figure_list, start=1):
        suffix = "" if len(figure_list) == 1 else f"_page-{page:02d}"
        figure.savefig(output_path.with_name(f"{output_path.stem}{suffix}.png"), dpi=150)
        plt.close(figure)


def process_subject(
    *,
    subject: str,
    set_path: Path,
    config: dict[str, Any],
    bad_channels: list[str],
    manual_ica_components: list[int] | None,
    interim_dir: Path,
    processed_dir: Path,
    figures_dir: Path,
    make_figure: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """处理一名受试者并返回受试者、epoch和ICA三级审计记录。"""

    signal_config = config["signal"]
    epoch_config = config["epochs"]
    qc_config = config["quality_control"]
    raw = _prepare_raw(set_path, bad_channels)
    if "Pz" not in raw.ch_names:
        raise ValueError(f"{subject} 缺少用于QC的Pz通道")
    raw_psd = _psd_curve(raw, "Pz")

    erp_raw = _filter_branch(
        raw,
        highpass_hz=float(signal_config["analysis_highpass_hz"]),
        lowpass_hz=float(signal_config["analysis_lowpass_hz"]),
        target_sfreq_hz=float(signal_config["target_sfreq_hz"]),
    )
    ica_raw = _filter_branch(
        raw,
        highpass_hz=float(signal_config["ica_highpass_hz"]),
        lowpass_hz=float(signal_config["ica_lowpass_hz"]),
        target_sfreq_hz=float(signal_config["target_sfreq_hz"]),
    )
    del raw

    ica, automatic_excluded_components, eog_scores, maximum_eog_scores = _fit_ica(
        ica_raw, config["ica"]
    )
    manual_ica_components = manual_ica_components or []
    invalid_manual_components = [
        component
        for component in manual_ica_components
        if component < 0 or component >= int(ica.n_components_)
    ]
    if invalid_manual_components:
        raise ValueError(f"{subject}的人工ICA成分编号越界: {invalid_manual_components}")
    excluded_components = sorted(
        set(automatic_excluded_components) | set(manual_ica_components)
    )
    ica.exclude = excluded_components
    ica.apply(erp_raw, exclude=excluded_components, verbose="ERROR")
    if bad_channels:
        erp_raw.interpolate_bads(reset_bads=True, mode="accurate", verbose="ERROR")
        # 插值通道原先未参与平均参考，插值后重算才能让最终所有EEG通道处于同一参考系。
        erp_raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    processed_psd = _psd_curve(erp_raw, "Pz")

    prefix = set_path.name.removesuffix("_eeg.set")
    trials, behavior_summary = pair_behavior_events(
        _read_tsv(set_path.with_name(f"{prefix}_events.tsv")),
        subject=subject,
        min_reaction_time_s=0.15,
        max_response_window_s=1.60,
    )
    events, metadata = make_epoch_events(
        trials,
        sfreq=float(erp_raw.info["sfreq"]),
        first_samp=erp_raw.first_samp,
        require_correct_response=bool(epoch_config["require_correct_response"]),
        exclude_behavior_flags=bool(epoch_config["exclude_behavior_flags"]),
    )
    epochs = mne.Epochs(
        erp_raw,
        events,
        event_id=EVENT_ID,
        tmin=float(epoch_config["tmin_s"]),
        tmax=float(epoch_config["tmax_s"]),
        baseline=tuple(float(value) for value in epoch_config["baseline_s"]),
        picks=[*mne.pick_types(erp_raw.info, eeg=True, eog=True)],
        metadata=metadata,
        preload=True,
        reject_by_annotation=True,
        event_repeated="drop",
        verbose="ERROR",
    )
    peak_to_peak_uv = _epoch_peak_to_peak_uv(epochs)
    epoch_metadata_before_rejection = epochs.metadata.copy().reset_index(drop=True)
    threshold_value = qc_config.get("peak_to_peak_reject_uv")
    threshold_uv = float(threshold_value) if threshold_value is not None else None
    rejected = _drop_large_epochs(epochs, peak_to_peak_uv, threshold_uv)

    epoch_rows: list[dict[str, Any]] = []
    for index, row in epoch_metadata_before_rejection.iterrows():
        epoch_rows.append(
            {
                "subject": subject,
                "trial_index": int(row["trial_index"]),
                "condition": row["condition_label"],
                "reaction_time_s": float(row["reaction_time_s"]),
                "max_eeg_peak_to_peak_uv": float(peak_to_peak_uv[index]),
                "rejected_eeg_ptp": bool(rejected[index]),
            }
        )

    subject_interim_dir = interim_dir / subject
    subject_interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    ica_path = subject_interim_dir / f"{subject}_task-P3_ica.fif"
    epochs_path = processed_dir / f"{subject}_task-P3_epo.fif"
    ica.save(ica_path, overwrite=True, verbose="ERROR")
    epochs.save(epochs_path, overwrite=True, verbose="ERROR")

    ica_rows: list[dict[str, Any]] = []
    for component, maximum_score in enumerate(maximum_eog_scores):
        row: dict[str, Any] = {
            "subject": subject,
            "component": component,
            "maximum_absolute_eog_correlation": float(maximum_score),
            "excluded": component in excluded_components,
            "automatic_exclusion": component in automatic_excluded_components,
            "manual_exclusion": component in manual_ica_components,
        }
        for channel, scores in eog_scores.items():
            row[f"correlation_{channel}"] = float(scores[component])
        ica_rows.append(row)

    if make_figure:
        _plot_preprocessing_qc(
            subject=subject,
            raw_psd=raw_psd,
            processed_psd=processed_psd,
            maximum_eog_scores=maximum_eog_scores,
            excluded_components=excluded_components,
            peak_to_peak_uv=peak_to_peak_uv,
            rejection_threshold_uv=threshold_uv,
            epochs=epochs,
            output_path=figures_dir / f"{subject}_preprocessing_qc.png",
        )
        _save_excluded_component_topographies(
            ica,
            excluded_components,
            figures_dir / f"{subject}_excluded_ica_topographies.png",
        )

    condition_before = metadata["condition_label"].value_counts()
    condition_after = epochs.metadata["condition_label"].value_counts()
    summary = {
        "subject": subject,
        "manual_bad_channels": ";".join(bad_channels),
        "ica_components": int(ica.n_components_),
        "ica_iterations": int(ica.n_iter_),
        "ica_converged": int(ica.n_iter_) < int(config["ica"]["max_iter"]),
        "ica_excluded_components": ";".join(map(str, excluded_components)),
        "ica_excluded_count": len(excluded_components),
        "ica_automatic_excluded_components": ";".join(
            map(str, automatic_excluded_components)
        ),
        "ica_manual_excluded_components": ";".join(map(str, manual_ica_components)),
        "behavior_stimulus_trials": behavior_summary["stimulus_trials"],
        "behavior_valid_trials": int(len(metadata)),
        "epochs_before_amplitude_rejection": int(len(peak_to_peak_uv)),
        "epochs_rejected_amplitude": int(rejected.sum()),
        "epochs_retained": int(len(epochs)),
        "epoch_rejection_fraction": float(rejected.mean()),
        "target_before": int(condition_before.get("Target", 0)),
        "standard_before": int(condition_before.get("Standard", 0)),
        "target_retained": int(condition_after.get("Target", 0)),
        "standard_retained": int(condition_after.get("Standard", 0)),
        "epoch_ptp_median_uv": float(np.median(peak_to_peak_uv)),
        "epoch_ptp_p95_uv": float(np.percentile(peak_to_peak_uv, 95)),
        "epoch_ptp_p99_uv": float(np.percentile(peak_to_peak_uv, 99)),
        "ica_path": str(ica_path.resolve()),
        "epochs_path": str(epochs_path.resolve()),
    }
    del ica_raw, erp_raw, epochs, ica
    return summary, epoch_rows, ica_rows


def run_preprocessing(
    *,
    bids_root: Path,
    config_path: Path,
    manual_qc_path: Path,
    output_dir: Path,
    interim_dir: Path,
    processed_dir: Path,
    figures_dir: Path,
    subjects: set[str] | None,
    make_figures: bool,
) -> dict[str, Any]:
    """串行运行预处理，避免同时把多名受试者的连续信号载入内存。"""

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    bad_channels, excluded_subjects = load_manual_qc(manual_qc_path)
    manual_ica_exclusions = load_manual_ica_exclusions(manual_qc_path)
    recordings = discover_p3_recordings(bids_root, task=str(config["dataset"]["task"]))
    if subjects:
        recordings = [(subject, path) for subject, path in recordings if subject in subjects]
    recordings = [item for item in recordings if item[0] not in excluded_subjects]
    if not recordings:
        raise FileNotFoundError("没有发现可处理的P3记录")

    subject_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    ica_rows: list[dict[str, Any]] = []
    for position, (subject, set_path) in enumerate(recordings, start=1):
        print(f"[{position:02d}/{len(recordings):02d}] {subject}", flush=True)
        summary, subject_epochs, subject_ica = process_subject(
            subject=subject,
            set_path=set_path,
            config=config,
            bad_channels=bad_channels.get(subject, []),
            manual_ica_components=manual_ica_exclusions.get(subject, []),
            interim_dir=interim_dir,
            processed_dir=processed_dir,
            figures_dir=figures_dir,
            make_figure=make_figures,
        )
        subject_rows.append(summary)
        epoch_rows.extend(subject_epochs)
        ica_rows.extend(subject_ica)

    output_dir.mkdir(parents=True, exist_ok=True)
    subject_frame = pd.DataFrame(subject_rows)
    epoch_frame = pd.DataFrame(epoch_rows)
    ica_frame = pd.DataFrame(ica_rows)
    review_reasons: list[str] = []
    for _, row in subject_frame.iterrows():
        reasons: list[str] = []
        if row["epoch_rejection_fraction"] > float(
            config["quality_control"]["subject_review_max_epoch_rejection_fraction"]
        ):
            reasons.append("epoch_rejection_fraction_above_limit")
        if row["ica_components"] < int(
            config["quality_control"]["subject_review_min_ica_components"]
        ):
            reasons.append("unusually_low_ica_dimension")
        if not row["ica_converged"]:
            reasons.append("ica_not_converged")
        review_reasons.append(";".join(reasons))
    subject_frame["automatic_review_reason"] = review_reasons
    subject_frame["subject_review_candidate"] = subject_frame[
        "automatic_review_reason"
    ].astype(bool)
    review_frame = subject_frame.loc[
        subject_frame["subject_review_candidate"],
        [
            "subject",
            "automatic_review_reason",
            "epoch_rejection_fraction",
            "ica_components",
            "ica_iterations",
            "epochs_retained",
            "target_retained",
            "standard_retained",
        ],
    ]
    subject_frame.to_csv(
        output_dir / "subject_preprocessing.csv", index=False, encoding="utf-8-sig"
    )
    epoch_frame.to_csv(output_dir / "epoch_qc.csv", index=False, encoding="utf-8-sig")
    ica_frame.to_csv(output_dir / "ica_components.csv", index=False, encoding="utf-8-sig")
    review_frame.to_csv(
        output_dir / "subject_review_candidates.csv", index=False, encoding="utf-8-sig"
    )

    run_summary = {
        "subjects_processed": len(subject_rows),
        "subjects_excluded_before_processing": excluded_subjects,
        "epoch_threshold_frozen": bool(config["quality_control"]["epoch_threshold_frozen"]),
        "peak_to_peak_reject_uv": config["quality_control"]["peak_to_peak_reject_uv"],
        "epochs_before_amplitude_rejection": int(
            subject_frame["epochs_before_amplitude_rejection"].sum()
        ),
        "epochs_rejected_amplitude": int(subject_frame["epochs_rejected_amplitude"].sum()),
        "epochs_retained": int(subject_frame["epochs_retained"].sum()),
        "ica_components_excluded": int(subject_frame["ica_excluded_count"].sum()),
        "ica_nonconverged_subjects": subject_frame.loc[
            ~subject_frame["ica_converged"], "subject"
        ].tolist(),
        "subject_review_candidates": review_frame["subject"].tolist(),
        "manual_bad_channels": bad_channels,
        "config": config,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return run_summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ERP CORE P3 滤波、ICA、插值和分段。")
    parser.add_argument("--bids-root", type=Path, default=Path("data/raw/erp-core-p3"))
    parser.add_argument("--config", type=Path, default=Path("configs/preprocessing.yaml"))
    parser.add_argument("--manual-qc", type=Path, default=Path("configs/manual_qc.yaml"))
    parser.add_argument(
        "--stage", choices=("pilot", "final", "reviewed", "adjudicated"), default="final"
    )
    parser.add_argument("--subjects", nargs="*", help="可选的 sub-XXX 列表；默认处理全部。")
    parser.add_argument("--no-figures", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    summary = run_preprocessing(
        bids_root=args.bids_root,
        config_path=args.config,
        manual_qc_path=args.manual_qc,
        output_dir=Path("results/preprocessing") / args.stage,
        interim_dir=Path("data/interim/preprocessing") / args.stage,
        processed_dir=Path("data/processed/epochs") / args.stage,
        figures_dir=Path("reports/figures/preprocessing") / args.stage,
        subjects=set(args.subjects) if args.subjects else None,
        make_figures=not args.no_figures,
    )
    print(f"Processed subjects: {summary['subjects_processed']}")
    print(f"Retained epochs: {summary['epochs_retained']}")
    print(f"Excluded ICA components: {summary['ica_components_excluded']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
