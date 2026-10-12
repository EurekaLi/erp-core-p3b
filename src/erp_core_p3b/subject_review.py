"""为高剔除率受试者生成预处理复核指标和多面板图。"""

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

from erp_core_p3b.preprocessing import (
    EVENT_ID,
    _filter_branch,
    _prepare_raw,
    load_manual_qc,
    make_epoch_events,
)
from erp_core_p3b.raw_qc import _read_tsv, discover_p3_recordings, pair_behavior_events


def channel_epoch_peak_to_peak_uv(epochs: mne.Epochs) -> tuple[np.ndarray, list[str]]:
    """返回 epoch×EEG通道 的峰峰值矩阵，单位为微伏。"""

    picks = mne.pick_types(epochs.info, eeg=True, eog=False, exclude=[])
    data_uv = epochs.get_data(picks=picks, copy=False) * 1e6
    return np.ptp(data_uv, axis=-1), [epochs.ch_names[index] for index in picks]


def temporal_rejection_profile(
    onset_s: np.ndarray,
    rejected: np.ndarray,
    *,
    bins: int,
) -> pd.DataFrame:
    """按记录时间等宽分箱，区分局部坏时段和贯穿全程的噪声。"""

    edges = np.linspace(float(onset_s.min()), float(onset_s.max()), bins + 1)
    # 最后一个事件等于最右边界，clip保证它仍被分到最后一箱。
    indices = np.clip(np.digitize(onset_s, edges[1:-1]), 0, bins - 1)
    rows: list[dict[str, Any]] = []
    for index in range(bins):
        mask = indices == index
        count = int(mask.sum())
        rejected_count = int(rejected[mask].sum())
        rows.append(
            {
                "time_bin": index + 1,
                "start_s": float(edges[index]),
                "end_s": float(edges[index + 1]),
                "epoch_count": count,
                "rejected_count": rejected_count,
                "rejection_fraction": rejected_count / count if count else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _reconstruct_epochs_before_amplitude_rejection(
    *,
    subject: str,
    set_path: Path,
    config: dict[str, Any],
    bad_channels: list[str],
    ica_path: Path,
) -> tuple[mne.Epochs, mne.preprocessing.ICA]:
    """复用正式参数重建epoch，但停在200 µV剔除之前。"""

    raw = _prepare_raw(set_path, bad_channels)
    signal_config = config["signal"]
    erp_raw = _filter_branch(
        raw,
        highpass_hz=float(signal_config["analysis_highpass_hz"]),
        lowpass_hz=float(signal_config["analysis_lowpass_hz"]),
        target_sfreq_hz=float(signal_config["target_sfreq_hz"]),
    )
    del raw
    ica = mne.preprocessing.read_ica(ica_path, verbose="ERROR")
    ica.apply(erp_raw, exclude=ica.exclude, verbose="ERROR")
    if bad_channels:
        erp_raw.interpolate_bads(reset_bads=True, mode="accurate", verbose="ERROR")
        erp_raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    prefix = set_path.name.removesuffix("_eeg.set")
    trials, _ = pair_behavior_events(
        _read_tsv(set_path.with_name(f"{prefix}_events.tsv")),
        subject=subject,
        min_reaction_time_s=0.15,
        max_response_window_s=1.60,
    )
    epoch_config = config["epochs"]
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
    return epochs, ica


def _review_summary(
    *,
    subject: str,
    epochs: mne.Epochs,
    channel_ptp_uv: np.ndarray,
    channel_names: list[str],
    threshold_uv: float,
    ica: mne.preprocessing.ICA,
    retained_epochs: mne.Epochs,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    abnormal = channel_ptp_uv > threshold_uv
    rejected = abnormal.any(axis=1)
    abnormal_channel_count = abnormal.sum(axis=1)
    channel_counts = abnormal.sum(axis=0)
    channel_frame = pd.DataFrame(
        {
            "subject": subject,
            "channel": channel_names,
            "abnormal_epoch_count": channel_counts,
            "abnormal_epoch_fraction": channel_counts / len(epochs),
            "maximum_peak_to_peak_uv": channel_ptp_uv.max(axis=0),
        }
    ).sort_values("abnormal_epoch_count", ascending=False)
    onset_s = epochs.metadata["stimulus_onset_s"].to_numpy(dtype=float)
    temporal_frame = temporal_rejection_profile(onset_s, rejected, bins=10)
    top_channels = channel_frame.head(5)
    rejected_count = int(rejected.sum())
    rejected_with_multiple = int(((abnormal_channel_count >= 3) & rejected).sum())
    target_before = int((epochs.metadata["condition_label"] == "Target").sum())
    standard_before = int((epochs.metadata["condition_label"] == "Standard").sum())
    summary = {
        "subject": subject,
        "epochs_before": len(epochs),
        "epochs_rejected": rejected_count,
        "rejection_fraction": rejected_count / len(epochs),
        "rejected_with_3plus_channels": rejected_with_multiple,
        "diffuse_rejection_fraction": (
            rejected_with_multiple / rejected_count if rejected_count else 0.0
        ),
        "time_bins_with_rejection": int((temporal_frame["rejected_count"] > 0).sum()),
        "maximum_time_bin_rejection_fraction": float(
            temporal_frame["rejection_fraction"].max()
        ),
        "top_abnormal_channels": ";".join(
            f"{row.channel}:{int(row.abnormal_epoch_count)}"
            for row in top_channels.itertuples()
        ),
        "worst_epoch_index": int(np.argmax(channel_ptp_uv.max(axis=1))),
        "worst_epoch_peak_to_peak_uv": float(channel_ptp_uv.max()),
        "ica_components": int(ica.n_components_),
        "ica_iterations": int(ica.n_iter_),
        "ica_excluded_components": ";".join(map(str, ica.exclude)),
        "target_before": target_before,
        "standard_before": standard_before,
        "target_retained": len(retained_epochs["Target"]),
        "standard_retained": len(retained_epochs["Standard"]),
    }
    return summary, channel_frame, temporal_frame


def _plot_review_dashboard(
    *,
    subject: str,
    epochs: mne.Epochs,
    retained_epochs: mne.Epochs,
    channel_ptp_uv: np.ndarray,
    channel_names: list[str],
    threshold_uv: float,
    ica_components: pd.DataFrame,
    channel_frame: pd.DataFrame,
    temporal_frame: pd.DataFrame,
    output_path: Path,
) -> None:
    abnormal = channel_ptp_uv > threshold_uv
    rejected = abnormal.any(axis=1)
    per_epoch_ptp = channel_ptp_uv.max(axis=1)
    onset_min = epochs.metadata["stimulus_onset_s"].to_numpy(dtype=float) / 60.0
    worst_epoch = int(np.argmax(per_epoch_ptp))
    worst_channel = int(np.argmax(channel_ptp_uv[worst_epoch]))
    eeg_picks = mne.pick_types(epochs.info, eeg=True, eog=False, exclude=[])
    worst_data_uv = epochs.get_data(picks=eeg_picks)[worst_epoch] * 1e6

    fig, axes = plt.subplots(3, 3, figsize=(21, 17), constrained_layout=True)
    fig.suptitle(f"{subject} | four-subject review dashboard", fontsize=17)

    ordered_channels = channel_frame.sort_values("abnormal_epoch_count", ascending=False)
    axes[0, 0].bar(
        ordered_channels["channel"], ordered_channels["abnormal_epoch_count"], color="tab:red"
    )
    axes[0, 0].set(title="Epochs >200 µV by channel", ylabel="epoch count")
    axes[0, 0].tick_params(axis="x", rotation=90, labelsize=7)

    count_values = abnormal.sum(axis=1)[rejected]
    if count_values.size:
        bins = np.arange(0.5, max(3.5, float(count_values.max()) + 1.5), 1)
        axes[0, 1].hist(count_values, bins=bins, color="tab:purple", rwidth=0.85)
    else:
        axes[0, 1].text(0.5, 0.5, "No epochs above threshold", ha="center", va="center")
    axes[0, 1].set(
        title="Spatial spread of rejected epochs",
        xlabel="channels above threshold in one epoch",
        ylabel="epoch count",
    )

    axes[0, 2].scatter(onset_min[~rejected], per_epoch_ptp[~rejected], s=12, alpha=0.6)
    axes[0, 2].scatter(
        onset_min[rejected], per_epoch_ptp[rejected], s=18, color="tab:red", alpha=0.75
    )
    axes[0, 2].axhline(threshold_uv, color="black", linestyle="--")
    axes[0, 2].set_yscale("log")
    axes[0, 2].set(title="Anomalies across recording", xlabel="stimulus onset (min)", ylabel="µV")

    axes[1, 0].bar(
        temporal_frame["time_bin"], temporal_frame["rejection_fraction"], color="tab:orange"
    )
    axes[1, 0].axhline(0.2, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set(
        title="Rejection rate in 10 time bins",
        xlabel="recording time bin",
        ylabel="fraction",
        ylim=(0, 1),
    )

    display_limit = max(50.0, float(np.percentile(np.abs(worst_data_uv), 98)))
    image = axes[1, 1].imshow(
        np.clip(worst_data_uv, -display_limit, display_limit),
        aspect="auto",
        origin="upper",
        extent=[epochs.times[0], epochs.times[-1], len(channel_names), 0],
        cmap="RdBu_r",
        vmin=-display_limit,
        vmax=display_limit,
    )
    axes[1, 1].axvline(0, color="black", linewidth=0.8)
    axes[1, 1].set(title=f"Worst epoch #{worst_epoch}: all EEG", xlabel="s", ylabel="channel")
    axes[1, 1].set_yticks(np.arange(len(channel_names)) + 0.5)
    axes[1, 1].set_yticklabels(channel_names, fontsize=6)
    fig.colorbar(image, ax=axes[1, 1], shrink=0.75, label="µV (display clipped)")

    axes[1, 2].plot(epochs.times, worst_data_uv[worst_channel], color="tab:red")
    axes[1, 2].axvline(0, color="black", linewidth=0.8)
    axes[1, 2].axhline(0, color="black", linewidth=0.8)
    axes[1, 2].set(
        title=(
            f"Worst channel: {channel_names[worst_channel]} | "
            f"PTP={channel_ptp_uv[worst_epoch, worst_channel]:.1f} µV"
        ),
        xlabel="s",
        ylabel="µV",
    )

    component_index = ica_components["component"].to_numpy(dtype=int)
    component_score = ica_components["maximum_absolute_eog_correlation"].to_numpy(dtype=float)
    component_excluded = ica_components["excluded"].astype(str).str.lower().eq("true")
    colors = np.where(component_excluded, "tab:red", "tab:blue")
    axes[2, 0].bar(component_index, component_score, color=colors)
    axes[2, 0].axhline(0.7, color="black", linestyle="--")
    axes[2, 0].set(title="ICA–EOG evidence", xlabel="component", ylabel="maximum |r|")

    trial_counts = [
        int((epochs.metadata["condition_label"] == "Target").sum()),
        int((epochs.metadata["condition_label"] == "Standard").sum()),
        len(retained_epochs["Target"]),
        len(retained_epochs["Standard"]),
    ]
    axes[2, 1].bar(
        ["Target\nbefore", "Standard\nbefore", "Target\nretained", "Standard\nretained"],
        trial_counts,
        color=["tab:orange", "tab:green", "tab:orange", "tab:green"],
    )
    axes[2, 1].set(title="Condition trial counts", ylabel="epochs")

    for condition, color in (("Target", "tab:orange"), ("Standard", "tab:green")):
        evoked = retained_epochs[condition].average(picks=["Pz"])
        axes[2, 2].plot(evoked.times, evoked.data[0] * 1e6, label=condition, color=color)
    axes[2, 2].axvline(0, color="black", linewidth=0.8)
    axes[2, 2].axhline(0, color="black", linewidth=0.8)
    axes[2, 2].axvspan(0.3, 0.6, color="0.8", alpha=0.25)
    axes[2, 2].set(title="Retained Pz ERP", xlabel="s", ylabel="µV")
    axes[2, 2].legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_top_eog_component_topographies(
    *,
    subject: str,
    ica: mne.preprocessing.ICA,
    ica_components: pd.DataFrame,
    output_path: Path,
) -> None:
    """显示EOG相关最高的成分，辅助区分眼动地形和神经源地形。"""

    ranked = ica_components.sort_values(
        "maximum_absolute_eog_correlation", ascending=False
    ).head(3)
    picks = ranked["component"].astype(int).tolist()
    scores = ranked["maximum_absolute_eog_correlation"].astype(float).tolist()
    title = f"{subject} | top ICA–EOG components | " + ", ".join(
        f"ICA{component:03d}: |r|={score:.2f}"
        for component, score in zip(picks, scores, strict=True)
    )
    figures = ica.plot_components(
        picks=picks,
        ch_type="eeg",
        title=title,
        show=False,
    )
    figure_list = figures if isinstance(figures, list) else [figures]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for page, figure in enumerate(figure_list, start=1):
        suffix = "" if len(figure_list) == 1 else f"_page-{page:02d}"
        figure.savefig(output_path.with_name(f"{output_path.stem}{suffix}.png"), dpi=150)
        plt.close(figure)


def run_subject_review(
    *,
    bids_root: Path,
    config_path: Path,
    manual_qc_path: Path,
    preprocessing_results_dir: Path,
    interim_dir: Path,
    processed_dir: Path,
    output_dir: Path,
    figures_dir: Path,
    subjects: Sequence[str],
) -> dict[str, Any]:
    """重建振幅剔除前数据并生成四类审计表和逐人复核图。"""

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    bad_channels, _ = load_manual_qc(manual_qc_path)
    recordings = dict(discover_p3_recordings(bids_root, task=str(config["dataset"]["task"])))
    ica_frame = pd.read_csv(preprocessing_results_dir / "ica_components.csv")
    threshold_uv = float(config["quality_control"]["peak_to_peak_reject_uv"])
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    all_channel_rows: list[pd.DataFrame] = []
    all_temporal_rows: list[pd.DataFrame] = []
    for position, subject in enumerate(subjects, start=1):
        print(f"[{position:02d}/{len(subjects):02d}] {subject}", flush=True)
        if subject not in recordings:
            raise FileNotFoundError(f"找不到{subject}的P3记录")
        epochs, ica = _reconstruct_epochs_before_amplitude_rejection(
            subject=subject,
            set_path=recordings[subject],
            config=config,
            bad_channels=bad_channels.get(subject, []),
            ica_path=interim_dir / subject / f"{subject}_task-P3_ica.fif",
        )
        retained_epochs = mne.read_epochs(
            processed_dir / f"{subject}_task-P3_epo.fif", preload=True, verbose="ERROR"
        )
        channel_ptp_uv, channel_names = channel_epoch_peak_to_peak_uv(epochs)
        summary, channel_frame, temporal_frame = _review_summary(
            subject=subject,
            epochs=epochs,
            channel_ptp_uv=channel_ptp_uv,
            channel_names=channel_names,
            threshold_uv=threshold_uv,
            ica=ica,
            retained_epochs=retained_epochs,
        )
        summaries.append(summary)
        all_channel_rows.append(channel_frame)
        temporal_frame.insert(0, "subject", subject)
        all_temporal_rows.append(temporal_frame)
        _plot_review_dashboard(
            subject=subject,
            epochs=epochs,
            retained_epochs=retained_epochs,
            channel_ptp_uv=channel_ptp_uv,
            channel_names=channel_names,
            threshold_uv=threshold_uv,
            ica_components=ica_frame[ica_frame["subject"] == subject],
            channel_frame=channel_frame,
            temporal_frame=temporal_frame,
            output_path=figures_dir / f"{subject}_review_dashboard.png",
        )
        _save_top_eog_component_topographies(
            subject=subject,
            ica=ica,
            ica_components=ica_frame[ica_frame["subject"] == subject],
            output_path=figures_dir / f"{subject}_top_ica_eog_topographies.png",
        )

    summary_frame = pd.DataFrame(summaries)
    channel_output = pd.concat(all_channel_rows, ignore_index=True)
    temporal_output = pd.concat(all_temporal_rows, ignore_index=True)
    summary_frame.to_csv(output_dir / "review_summary.csv", index=False, encoding="utf-8-sig")
    channel_output.to_csv(output_dir / "channel_anomalies.csv", index=False, encoding="utf-8-sig")
    temporal_output.to_csv(output_dir / "temporal_anomalies.csv", index=False, encoding="utf-8-sig")
    run_summary = {
        "subjects": list(subjects),
        "threshold_uv": threshold_uv,
        "dashboards": len(subjects),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return run_summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成高剔除率受试者的详细复核报告。")
    parser.add_argument(
        "--subjects", nargs="+", default=["sub-009", "sub-025", "sub-030", "sub-039"]
    )
    parser.add_argument("--stage", choices=("final", "reviewed"), default="final")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    manual_qc_path = (
        Path("configs/manual_qc.yaml")
        if args.stage == "final"
        else Path("configs/review_interventions.yaml")
    )
    summary = run_subject_review(
        bids_root=Path("data/raw/erp-core-p3"),
        config_path=Path("configs/preprocessing.yaml"),
        manual_qc_path=manual_qc_path,
        preprocessing_results_dir=Path("results/preprocessing") / args.stage,
        interim_dir=Path("data/interim/preprocessing") / args.stage,
        processed_dir=Path("data/processed/epochs") / args.stage,
        output_dir=(
            Path("results/subject_review/four_candidates")
            if args.stage == "final"
            else Path("results/subject_review/four_candidates_after_intervention")
        ),
        figures_dir=(
            Path("reports/figures/subject_review/four_candidates")
            if args.stage == "final"
            else Path("reports/figures/subject_review/four_candidates_after_intervention")
        ),
        subjects=args.subjects,
    )
    print(f"Review dashboards: {summary['dashboards']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
