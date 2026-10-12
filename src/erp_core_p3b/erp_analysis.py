"""运行受试者级 P3b 测量、群体统计、可视化和探索性时空检验。"""

# ruff: noqa: E402 -- Matplotlib 导入前必须把缓存重定向到项目内可写目录。

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib import metadata
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
from scipy import stats


def time_window_mask(times_s: np.ndarray, window_s: Sequence[float]) -> np.ndarray:
    """返回闭区间时间窗掩码，并拒绝与采样点不相交的配置。"""

    start_s, end_s = map(float, window_s)
    if start_s >= end_s:
        raise ValueError(f"时间窗起点必须小于终点，收到 {window_s}")
    mask = (times_s >= start_s) & (times_s <= end_s)
    if not mask.any():
        raise ValueError(f"时间窗 {window_s} 与数据时间范围不相交")
    return mask


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    iterations: int,
    confidence_level: float,
    random_state: int,
    batch_size: int = 250,
) -> tuple[np.ndarray, np.ndarray]:
    """按受试者成对重采样，返回均值的百分位 bootstrap 置信区间。"""

    data = np.asarray(values, dtype=float)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    if data.ndim != 2 or data.shape[0] < 2:
        raise ValueError("bootstrap 输入必须是至少两名受试者的二维矩阵")
    if iterations < 1:
        raise ValueError("bootstrap_iterations 必须为正整数")

    rng = np.random.default_rng(random_state)
    boot_means = np.empty((iterations, data.shape[1]), dtype=float)
    # 分批构造索引，避免波形 bootstrap 临时生成数百 MB 的三维数组。
    for start in range(0, iterations, batch_size):
        stop = min(start + batch_size, iterations)
        indices = rng.integers(0, data.shape[0], size=(stop - start, data.shape[0]))
        boot_means[start:stop] = data[indices].mean(axis=1)
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(boot_means, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0)
    return lower.squeeze(), upper.squeeze()


def compute_subject_measurements(
    *,
    subject: str,
    times_s: np.ndarray,
    channel_names: Sequence[str],
    target_eeg_uv: np.ndarray,
    standard_eeg_uv: np.ndarray,
    target_trials: int,
    standard_trials: int,
    primary_channel: str,
    posterior_roi_channels: Sequence[str],
    window_s: Sequence[float],
) -> dict[str, Any]:
    """从两条条件平均波形提取一个受试者的一行 ERP 测量。"""

    names = list(channel_names)
    required = [primary_channel, *posterior_roi_channels]
    missing = sorted(set(required) - set(names))
    if missing:
        raise ValueError(f"{subject} 缺少配置通道: {', '.join(missing)}")
    if target_eeg_uv.shape != standard_eeg_uv.shape:
        raise ValueError(f"{subject} 的两个条件波形形状不一致")
    if target_eeg_uv.shape != (len(names), len(times_s)):
        raise ValueError(f"{subject} 的波形维度与通道或时间轴不一致")

    mask = time_window_mask(times_s, window_s)
    pz_index = names.index(primary_channel)
    roi_indices = [names.index(channel) for channel in posterior_roi_channels]
    target_pz = target_eeg_uv[pz_index]
    standard_pz = standard_eeg_uv[pz_index]
    difference_pz = target_pz - standard_pz
    window_times = times_s[mask]

    def peak(values: np.ndarray) -> tuple[float, float]:
        window_values = values[mask]
        index = int(np.argmax(window_values))
        return float(window_values[index]), float(window_times[index] * 1000.0)

    target_peak_uv, target_peak_latency_ms = peak(target_pz)
    standard_peak_uv, standard_peak_latency_ms = peak(standard_pz)
    difference_peak_uv, difference_peak_latency_ms = peak(difference_pz)
    target_roi = target_eeg_uv[roi_indices][:, mask]
    standard_roi = standard_eeg_uv[roi_indices][:, mask]

    return {
        "subject": subject,
        "target_trials": int(target_trials),
        "standard_trials": int(standard_trials),
        "target_pz_mean_uv": float(target_pz[mask].mean()),
        "standard_pz_mean_uv": float(standard_pz[mask].mean()),
        "difference_pz_mean_uv": float(difference_pz[mask].mean()),
        "target_pz_peak_uv": target_peak_uv,
        "target_pz_peak_latency_ms": target_peak_latency_ms,
        "standard_pz_peak_uv": standard_peak_uv,
        "standard_pz_peak_latency_ms": standard_peak_latency_ms,
        "difference_pz_peak_uv": difference_peak_uv,
        "difference_pz_peak_latency_ms": difference_peak_latency_ms,
        "target_pz_signed_area_uv_s": float(np.trapezoid(target_pz[mask], window_times)),
        "standard_pz_signed_area_uv_s": float(
            np.trapezoid(standard_pz[mask], window_times)
        ),
        "difference_pz_signed_area_uv_s": float(
            np.trapezoid(difference_pz[mask], window_times)
        ),
        "target_posterior_roi_mean_uv": float(target_roi.mean()),
        "standard_posterior_roi_mean_uv": float(standard_roi.mean()),
        "difference_posterior_roi_mean_uv": float((target_roi - standard_roi).mean()),
    }


def matched_rank_biserial(differences: np.ndarray) -> float:
    """计算配对差值的秩二列相关，正值表示 Target 大于 Standard。"""

    nonzero = np.asarray(differences, dtype=float)
    nonzero = nonzero[nonzero != 0]
    if nonzero.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def compute_paired_statistics(
    target_uv: np.ndarray,
    standard_uv: np.ndarray,
    *,
    bootstrap_iterations: int,
    confidence_level: float,
    random_state: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """计算预先定义的配对 t 检验及非参数稳健性检验。"""

    target = np.asarray(target_uv, dtype=float)
    standard = np.asarray(standard_uv, dtype=float)
    if target.ndim != 1 or standard.ndim != 1 or target.shape != standard.shape:
        raise ValueError("配对统计要求两个等长的一维数组")
    if target.size < 2 or not (np.isfinite(target).all() and np.isfinite(standard).all()):
        raise ValueError("配对统计要求至少两个有限的受试者测量")

    differences = target - standard
    difference_sd = float(differences.std(ddof=1))
    cohens_dz = float(differences.mean() / difference_sd) if difference_sd else np.nan
    ci_low, ci_high = bootstrap_mean_ci(
        differences,
        iterations=bootstrap_iterations,
        confidence_level=confidence_level,
        random_state=random_state,
    )
    t_result = stats.ttest_rel(target, standard, alternative="two-sided")
    shapiro_result = stats.shapiro(differences)
    common = {
        "n_subjects": int(target.size),
        "target_mean_uv": float(target.mean()),
        "target_sd_uv": float(target.std(ddof=1)),
        "standard_mean_uv": float(standard.mean()),
        "standard_sd_uv": float(standard.std(ddof=1)),
        "mean_difference_uv": float(differences.mean()),
        "median_difference_uv": float(np.median(differences)),
        "sd_difference_uv": difference_sd,
        "ci_level": confidence_level,
        "ci_low_uv": float(ci_low),
        "ci_high_uv": float(ci_high),
        "shapiro_w": float(shapiro_result.statistic),
        "shapiro_p_value": float(shapiro_result.pvalue),
    }
    primary = {
        "analysis": "primary_paired_t_test",
        **common,
        "statistic_name": "t",
        "statistic": float(t_result.statistic),
        "degrees_of_freedom": int(target.size - 1),
        "p_value": float(t_result.pvalue),
        "effect_size_name": "cohens_dz",
        "effect_size": cohens_dz,
    }

    if np.allclose(differences, 0.0):
        wilcoxon_statistic, wilcoxon_p = 0.0, 1.0
    else:
        wilcoxon_result = stats.wilcoxon(
            target,
            standard,
            alternative="two-sided",
            zero_method="wilcox",
        )
        wilcoxon_statistic = float(wilcoxon_result.statistic)
        wilcoxon_p = float(wilcoxon_result.pvalue)
    secondary = {
        "analysis": "secondary_wilcoxon_signed_rank",
        **common,
        "statistic_name": "W",
        "statistic": wilcoxon_statistic,
        "degrees_of_freedom": np.nan,
        "p_value": wilcoxon_p,
        "effect_size_name": "matched_rank_biserial",
        "effect_size": matched_rank_biserial(differences),
    }
    return primary, secondary


def group_statistics_row(
    result: dict[str, Any],
    *,
    role: str,
    metric: str,
    unit: str,
    p_adjustment: str,
) -> dict[str, Any]:
    """把内部配对统计结果转换为不依赖具体物理单位的长表记录。"""

    return {
        "analysis_role": role,
        "metric": metric,
        "unit": unit,
        "test": result["analysis"],
        "n_subjects": result["n_subjects"],
        "target_mean": result["target_mean_uv"],
        "target_sd": result["target_sd_uv"],
        "standard_mean": result["standard_mean_uv"],
        "standard_sd": result["standard_sd_uv"],
        "mean_difference": result["mean_difference_uv"],
        "median_difference": result["median_difference_uv"],
        "sd_difference": result["sd_difference_uv"],
        "ci_level": result["ci_level"],
        "ci_low": result["ci_low_uv"],
        "ci_high": result["ci_high_uv"],
        "statistic_name": result["statistic_name"],
        "statistic": result["statistic"],
        "degrees_of_freedom": result["degrees_of_freedom"],
        "p_value": result["p_value"],
        "p_adjustment": p_adjustment,
        "effect_size_name": result["effect_size_name"],
        "effect_size": result["effect_size"],
        "difference_shapiro_w": result["shapiro_w"],
        "difference_shapiro_p_value": result["shapiro_p_value"],
    }


def _discover_epoch_files(root: Path) -> list[Path]:
    files = sorted(root.glob("sub-*_task-P3_epo.fif"))
    if not files:
        raise FileNotFoundError(f"在 {root} 下找不到 P3 epoch 文件")
    return files


def _subject_from_path(path: Path) -> str:
    subject = path.name.split("_task-", maxsplit=1)[0]
    if not subject.startswith("sub-"):
        raise ValueError(f"无法从文件名解析受试者编号: {path.name}")
    return subject


def _load_subject_averages(
    *,
    path: Path,
    target_label: str,
    standard_label: str,
    required_channels: Sequence[str],
) -> tuple[dict[str, Any], mne.Info]:
    """逐人读取 epochs，只返回条件平均，防止群体分析同时驻留全部 trial。"""

    subject = _subject_from_path(path)
    epochs = mne.read_epochs(path, preload=True, verbose="ERROR")
    missing_events = sorted({target_label, standard_label} - set(epochs.event_id))
    if missing_events:
        raise ValueError(f"{subject} 缺少事件条件: {', '.join(missing_events)}")
    missing_channels = sorted(set(required_channels) - set(epochs.ch_names))
    if missing_channels:
        raise ValueError(f"{subject} 缺少分析通道: {', '.join(missing_channels)}")

    eeg_picks = mne.pick_types(epochs.info, eeg=True, eog=False, exclude=[])
    eeg_names = [epochs.ch_names[index] for index in eeg_picks]
    eeg_info = mne.pick_info(epochs.info, eeg_picks, copy=True)
    target_epochs = epochs[target_label]
    standard_epochs = epochs[standard_label]
    if len(target_epochs) == 0 or len(standard_epochs) == 0:
        raise ValueError(f"{subject} 至少有一个条件没有保留 trial")

    record = {
        "subject": subject,
        "file": path.as_posix(),
        "sfreq_hz": float(epochs.info["sfreq"]),
        "times_s": epochs.times.copy(),
        "eeg_channel_names": eeg_names,
        "target_trials": len(target_epochs),
        "standard_trials": len(standard_epochs),
        "target_eeg_uv": target_epochs.get_data(picks=eeg_picks).mean(axis=0) * 1e6,
        "standard_eeg_uv": standard_epochs.get_data(picks=eeg_picks).mean(axis=0) * 1e6,
    }
    del epochs, target_epochs, standard_epochs
    return record, eeg_info


def _validate_common_axes(records: Sequence[dict[str, Any]]) -> None:
    reference = records[0]
    for record in records[1:]:
        subject = record["subject"]
        if record["eeg_channel_names"] != reference["eeg_channel_names"]:
            raise ValueError(f"{subject} 的 EEG 通道顺序与首名受试者不一致")
        if not np.array_equal(record["times_s"], reference["times_s"]):
            raise ValueError(f"{subject} 的 epoch 时间轴与首名受试者不一致")
        if record["sfreq_hz"] != reference["sfreq_hz"]:
            raise ValueError(f"{subject} 的采样率与首名受试者不一致")


def _plot_group_waveforms(
    *,
    times_s: np.ndarray,
    target_uv: np.ndarray,
    standard_uv: np.ndarray,
    window_s: Sequence[float],
    iterations: int,
    confidence_level: float,
    random_state: int,
    output_path: Path,
) -> pd.DataFrame:
    difference_uv = target_uv - standard_uv
    target_ci = bootstrap_mean_ci(
        target_uv,
        iterations=iterations,
        confidence_level=confidence_level,
        random_state=random_state,
    )
    standard_ci = bootstrap_mean_ci(
        standard_uv,
        iterations=iterations,
        confidence_level=confidence_level,
        random_state=random_state + 1,
    )
    difference_ci = bootstrap_mean_ci(
        difference_uv,
        iterations=iterations,
        confidence_level=confidence_level,
        random_state=random_state + 2,
    )
    times_ms = times_s * 1000.0
    window_ms = np.asarray(window_s, dtype=float) * 1000.0

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    for values, ci, label, color in (
        (target_uv, target_ci, "Target", "tab:orange"),
        (standard_uv, standard_ci, "Standard", "tab:green"),
    ):
        mean = values.mean(axis=0)
        axes[0].plot(times_ms, mean, label=label, color=color, linewidth=2)
        axes[0].fill_between(times_ms, ci[0], ci[1], color=color, alpha=0.2)
    axes[0].set(title="Pz group ERP", ylabel="Amplitude (µV)")
    axes[0].legend()

    axes[1].plot(times_ms, difference_uv.mean(axis=0), color="tab:purple", linewidth=2)
    axes[1].fill_between(
        times_ms, difference_ci[0], difference_ci[1], color="tab:purple", alpha=0.2
    )
    axes[1].set(
        title="Pz difference wave (Target - Standard)",
        xlabel="Time from stimulus (ms)",
        ylabel="Amplitude (µV)",
    )
    for axis in axes:
        axis.axvline(0, color="black", linewidth=0.8)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.axvspan(window_ms[0], window_ms[1], color="0.75", alpha=0.2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return pd.DataFrame(
        {
            "time_s": times_s,
            "target_mean_uv": target_uv.mean(axis=0),
            "target_ci_low_uv": target_ci[0],
            "target_ci_high_uv": target_ci[1],
            "standard_mean_uv": standard_uv.mean(axis=0),
            "standard_ci_low_uv": standard_ci[0],
            "standard_ci_high_uv": standard_ci[1],
            "difference_mean_uv": difference_uv.mean(axis=0),
            "difference_ci_low_uv": difference_ci[0],
            "difference_ci_high_uv": difference_ci[1],
        }
    )


def _plot_paired_amplitudes(measurements: pd.DataFrame, output_path: Path) -> None:
    target = measurements["target_pz_mean_uv"].to_numpy()
    standard = measurements["standard_pz_mean_uv"].to_numpy()
    fig, axis = plt.subplots(figsize=(7.5, 7), constrained_layout=True)
    for target_value, standard_value in zip(target, standard, strict=True):
        axis.plot([0, 1], [standard_value, target_value], color="0.75", linewidth=0.8)
    axis.scatter(np.zeros_like(standard), standard, color="tab:green", alpha=0.8, label="Standard")
    axis.scatter(np.ones_like(target), target, color="tab:orange", alpha=0.8, label="Target")
    axis.scatter(
        [0, 1],
        [standard.mean(), target.mean()],
        color="black",
        marker="D",
        s=70,
        label="Group mean",
        zorder=4,
    )
    axis.set(
        xticks=[0, 1],
        xticklabels=["Standard", "Target"],
        ylabel="Pz mean amplitude, 300–600 ms (µV)",
        title="Subject-level paired amplitudes",
    )
    axis.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_difference_distribution(measurements: pd.DataFrame, output_path: Path) -> None:
    differences = measurements["difference_pz_mean_uv"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].hist(differences, bins="auto", color="tab:purple", alpha=0.75, edgecolor="white")
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].axvline(differences.mean(), color="tab:red", linestyle="--", label="Mean")
    axes[0].set(
        title="Subject difference distribution",
        xlabel="Target - Standard (µV)",
        ylabel="Subject count",
    )
    axes[0].legend()
    stats.probplot(differences, dist="norm", plot=axes[1])
    axes[1].set_title("Normal Q-Q plot of paired differences")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_subject_heatmap(
    *,
    subjects: Sequence[str],
    times_s: np.ndarray,
    difference_uv: np.ndarray,
    window_s: Sequence[float],
    output_path: Path,
) -> None:
    window = time_window_mask(times_s, window_s)
    order = np.argsort(difference_uv[:, window].mean(axis=1))
    sorted_data = difference_uv[order]
    limit = float(np.percentile(np.abs(sorted_data), 98))
    fig, axis = plt.subplots(figsize=(12, 9), constrained_layout=True)
    image = axis.imshow(
        sorted_data,
        aspect="auto",
        origin="lower",
        extent=[times_s[0] * 1000.0, times_s[-1] * 1000.0, 0, len(subjects)],
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    axis.axvline(0, color="black", linewidth=0.8)
    axis.axvspan(window_s[0] * 1000.0, window_s[1] * 1000.0, color="black", alpha=0.08)
    axis.set(
        title="Pz subject difference waves (sorted by window mean)",
        xlabel="Time from stimulus (ms)",
        ylabel="Subject",
        yticks=np.arange(len(subjects)) + 0.5,
        yticklabels=np.asarray(subjects)[order],
    )
    axis.tick_params(axis="y", labelsize=7)
    fig.colorbar(image, ax=axis, label="Target - Standard (µV)")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_topographies(
    *,
    eeg_info: mne.Info,
    target_uv: np.ndarray,
    standard_uv: np.ndarray,
    times_s: np.ndarray,
    window_s: Sequence[float],
    output_path: Path,
) -> pd.DataFrame:
    mask = time_window_mask(times_s, window_s)
    target_topography = target_uv[:, :, mask].mean(axis=(0, 2))
    standard_topography = standard_uv[:, :, mask].mean(axis=(0, 2))
    difference_topography = target_topography - standard_topography
    topographies = [target_topography, standard_topography, difference_topography]
    limit = float(max(np.max(np.abs(values)) for values in topographies))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), constrained_layout=True)
    image = None
    for axis, values, title in zip(
        axes,
        topographies,
        ("Target", "Standard", "Target - Standard"),
        strict=True,
    ):
        image, _ = mne.viz.plot_topomap(
            values,
            eeg_info,
            axes=axis,
            show=False,
            cmap="RdBu_r",
            vlim=(-limit, limit),
            contours=6,
        )
        axis.set_title(title)
    if image is not None:
        fig.colorbar(image, ax=axes, label="Mean amplitude (µV)", shrink=0.8)
    fig.suptitle("300–600 ms scalp topographies")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return pd.DataFrame(
        {
            "channel": eeg_info.ch_names,
            "target_mean_uv": target_topography,
            "standard_mean_uv": standard_topography,
            "difference_mean_uv": difference_topography,
        }
    )


def _run_cluster_permutation(
    *,
    difference_uv: np.ndarray,
    times_s: np.ndarray,
    eeg_info: mne.Info,
    config: dict[str, Any],
    output_npz: Path,
    output_csv: Path,
    output_figure: Path,
) -> dict[str, Any]:
    window = time_window_mask(times_s, config["time_window_s"])
    cluster_data = difference_uv[:, :, window].transpose(0, 2, 1)
    cluster_times = times_s[window]
    adjacency, adjacency_names = mne.channels.find_ch_adjacency(eeg_info, ch_type="eeg")
    if list(adjacency_names) != list(eeg_info.ch_names):
        raise ValueError("cluster 邻接矩阵的通道顺序与输入数据不一致")

    alpha = float(config["cluster_forming_alpha"])
    threshold = float(stats.t.ppf(1.0 - alpha / 2.0, df=cluster_data.shape[0] - 1))
    t_observed, clusters, p_values, null_distribution = (
        mne.stats.spatio_temporal_cluster_1samp_test(
            cluster_data,
            threshold=threshold,
            n_permutations=int(config["permutations"]),
            tail=0,
            adjacency=adjacency,
            out_type="mask",
            rng=np.random.default_rng(int(config["random_state"])),
            n_jobs=1,
            verbose="ERROR",
        )
    )
    if clusters:
        cluster_masks = np.stack([np.asarray(cluster, dtype=bool) for cluster in clusters])
    else:
        cluster_masks = np.empty((0, *t_observed.shape), dtype=bool)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        t_observed=t_observed,
        cluster_masks=cluster_masks,
        cluster_p_values=p_values,
        null_distribution=null_distribution,
        times_s=cluster_times,
        channel_names=np.asarray(eeg_info.ch_names),
        threshold=threshold,
    )

    rows: list[dict[str, Any]] = []
    for index, (mask, p_value) in enumerate(zip(cluster_masks, p_values, strict=True), start=1):
        time_indices, channel_indices = np.where(mask)
        cluster_t_values = t_observed[mask]
        peak_position = int(np.argmax(np.abs(cluster_t_values)))
        peak_time_index, peak_channel_index = np.argwhere(mask)[peak_position]
        rows.append(
            {
                "cluster": index,
                "p_value": float(p_value),
                "significant_at_0_05": bool(p_value < 0.05),
                "polarity": "positive" if cluster_t_values.mean() > 0 else "negative",
                "mean_t": float(cluster_t_values.mean()),
                "peak_t": float(cluster_t_values[peak_position]),
                "peak_time_s": float(cluster_times[peak_time_index]),
                "peak_channel": eeg_info.ch_names[peak_channel_index],
                "start_s": float(cluster_times[time_indices.min()]),
                "end_s": float(cluster_times[time_indices.max()]),
                "channel_count": int(np.unique(channel_indices).size),
                "channels": ";".join(
                    np.asarray(eeg_info.ch_names)[np.unique(channel_indices)].tolist()
                ),
                "sample_count": int(mask.sum()),
            }
        )
    summary = pd.DataFrame(
        rows,
        columns=[
            "cluster",
            "p_value",
            "significant_at_0_05",
            "polarity",
            "mean_t",
            "peak_t",
            "peak_time_s",
            "peak_channel",
            "start_s",
            "end_s",
            "channel_count",
            "channels",
            "sample_count",
        ],
    )
    summary.to_csv(output_csv, index=False, encoding="utf-8-sig")

    significant = cluster_masks[p_values < 0.05].any(axis=0) if cluster_masks.size else None
    limit = float(np.max(np.abs(t_observed)))
    fig, axis = plt.subplots(figsize=(13, 8), constrained_layout=True)
    image = axis.imshow(
        t_observed.T,
        aspect="auto",
        origin="lower",
        extent=[cluster_times[0] * 1000.0, cluster_times[-1] * 1000.0, 0, len(eeg_info.ch_names)],
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    if significant is not None and significant.any():
        axis.contour(
            cluster_times * 1000.0,
            np.arange(len(eeg_info.ch_names)) + 0.5,
            significant.T.astype(int),
            levels=[0.5],
            colors="black",
            linewidths=1.2,
        )
    axis.set(
        title="Exploratory spatiotemporal cluster test (black = cluster p < .05)",
        xlabel="Time from stimulus (ms)",
        ylabel="EEG channel",
        yticks=np.arange(len(eeg_info.ch_names)) + 0.5,
        yticklabels=eeg_info.ch_names,
    )
    axis.tick_params(axis="y", labelsize=7)
    fig.colorbar(image, ax=axis, label="One-sample t statistic")
    output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_figure, dpi=180)
    plt.close(fig)
    return {
        "cluster_count": int(len(cluster_masks)),
        "significant_cluster_count": int((p_values < 0.05).sum()),
        "minimum_cluster_p_value": float(p_values.min()) if p_values.size else None,
        "permutations": int(config["permutations"]),
        "cluster_forming_threshold_t": threshold,
    }


def _sensitivity_row(
    *,
    label: str,
    measurements: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    primary, secondary = compute_paired_statistics(
        measurements["target_pz_mean_uv"].to_numpy(),
        measurements["standard_pz_mean_uv"].to_numpy(),
        bootstrap_iterations=int(config["bootstrap_iterations"]),
        confidence_level=float(config["confidence_level"]),
        random_state=int(config["random_state"]),
    )
    return {
        "dataset": label,
        "n_subjects": primary["n_subjects"],
        "mean_difference_uv": primary["mean_difference_uv"],
        "ci_low_uv": primary["ci_low_uv"],
        "ci_high_uv": primary["ci_high_uv"],
        "t_statistic": primary["statistic"],
        "degrees_of_freedom": primary["degrees_of_freedom"],
        "p_value": primary["p_value"],
        "cohens_dz": primary["effect_size"],
        "wilcoxon_p_value": secondary["p_value"],
        "matched_rank_biserial": secondary["effect_size"],
    }


def _write_results_summary(
    *,
    primary: dict[str, Any],
    secondary: dict[str, Any],
    group_statistics: pd.DataFrame,
    sensitivity: pd.DataFrame,
    cluster_summary: dict[str, Any],
    output_path: Path,
) -> None:
    primary_p = primary["p_value"]
    cluster_minimum = cluster_summary.get("minimum_cluster_p_value")
    cluster_text = (
        "未运行"
        if not cluster_summary.get("enabled", True)
        else (
            f"发现 {cluster_summary['significant_cluster_count']} 个 p < .05 的时空簇，"
            f"最小簇 p={cluster_minimum:.4g}。"
            if cluster_minimum is not None
            else "没有形成可检验的时空簇。"
        )
    )
    sensitivity_lines = [
        "| 数据集 | N | 平均差 (µV) | 95% CI (µV) | t | p | Cohen's dz |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sensitivity.itertuples(index=False):
        sensitivity_lines.append(
            f"| {row.dataset} | {row.n_subjects} | {row.mean_difference_uv:.3f} | "
            f"[{row.ci_low_uv:.3f}, {row.ci_high_uv:.3f}] | {row.t_statistic:.3f} | "
            f"{row.p_value:.4g} | {row.cohens_dz:.3f} |"
        )
    secondary_lines = [
        "| 指标 | 单位 | Target均值 | Standard均值 | 差值 | t | p（未校正） | dz |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    secondary_frame = group_statistics[group_statistics["analysis_role"] == "secondary"]
    for row in secondary_frame.itertuples(index=False):
        secondary_lines.append(
            f"| {row.metric} | {row.unit} | {row.target_mean:.3f} | "
            f"{row.standard_mean:.3f} | {row.mean_difference:.3f} | "
            f"{row.statistic:.3f} | {row.p_value:.4g} | {row.effect_size:.3f} |"
        )
    lines = [
        "# Phase 5 P3b 结果摘要",
        "",
        "## 主要分析",
        "",
        (
            f"39人裁决版数据中，Pz 300--600 ms 的 Target - Standard 平均差为 "
            f"{primary['mean_difference_uv']:.3f} µV，"
            f"95% bootstrap CI [{primary['ci_low_uv']:.3f}, {primary['ci_high_uv']:.3f}]，"
            f"t({primary['degrees_of_freedom']})={primary['statistic']:.3f}，"
            f"p={primary_p:.4g}，Cohen's dz={primary['effect_size']:.3f}。"
        ),
        "",
        "## 稳健性检查",
        "",
        (
            f"Wilcoxon符号秩检验：W={secondary['statistic']:.3f}，"
            f"p={secondary['p_value']:.4g}，"
            f"配对秩二列相关={secondary['effect_size']:.3f}。"
        ),
        "",
        "## 次要指标",
        "",
        *secondary_lines,
        "",
        "这些指标没有进行多重比较校正，只用于描述和稳健性理解。",
        "",
        "## QC决策敏感性分析",
        "",
        *sensitivity_lines,
        "",
        "## 探索性时空检验",
        "",
        cluster_text,
        "",
        "主分析、敏感性分析和探索性检验的角色不同；不得用后两者替代冻结的主要检验。",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_erp_analysis(
    *,
    config_path: Path,
    skip_cluster: bool = False,
    permutations_override: int | None = None,
) -> dict[str, Any]:
    """执行 Phase 5 的全部正式输出并返回关键结果。"""

    config_text = config_path.read_text(encoding="utf-8")
    config = yaml.safe_load(config_text)
    dataset_config = config["dataset"]
    primary_config = config["primary_analysis"]
    secondary_config = config["secondary_analysis"]
    cluster_config = dict(config["exploratory_analysis"]["cluster_permutation"])
    if permutations_override is not None:
        if permutations_override < 1:
            raise ValueError("permutations 必须为正整数")
        cluster_config["permutations"] = permutations_override

    epoch_root = Path(dataset_config["primary_epochs_root"])
    epoch_files = _discover_epoch_files(epoch_root)
    expected_subjects = int(dataset_config["primary_subject_count"])
    if len(epoch_files) != expected_subjects:
        raise ValueError(f"主分析要求 {expected_subjects} 人，实际找到 {len(epoch_files)} 个文件")

    required_channels = [
        str(primary_config["channel"]),
        *map(str, secondary_config["posterior_roi_channels"]),
    ]
    records: list[dict[str, Any]] = []
    eeg_info: mne.Info | None = None
    for position, path in enumerate(epoch_files, start=1):
        subject = _subject_from_path(path)
        print(f"[{position:02d}/{len(epoch_files):02d}] 读取并平均 {subject}", flush=True)
        record, subject_info = _load_subject_averages(
            path=path,
            target_label=str(config["conditions"]["target"]),
            standard_label=str(config["conditions"]["standard"]),
            required_channels=required_channels,
        )
        records.append(record)
        if eeg_info is None:
            eeg_info = subject_info
    if eeg_info is None:
        raise RuntimeError("没有可用于群体分析的 EEG 信息")
    _validate_common_axes(records)

    subjects = [record["subject"] for record in records]
    if set(subjects) & set(dataset_config["excluded_from_primary"]):
        raise ValueError("裁决版目录仍包含配置中排除的主分析受试者")
    times_s = records[0]["times_s"]
    channel_names = records[0]["eeg_channel_names"]
    target_eeg_uv = np.stack([record["target_eeg_uv"] for record in records])
    standard_eeg_uv = np.stack([record["standard_eeg_uv"] for record in records])
    difference_eeg_uv = target_eeg_uv - standard_eeg_uv

    measurement_rows = [
        compute_subject_measurements(
            subject=record["subject"],
            times_s=times_s,
            channel_names=channel_names,
            target_eeg_uv=record["target_eeg_uv"],
            standard_eeg_uv=record["standard_eeg_uv"],
            target_trials=record["target_trials"],
            standard_trials=record["standard_trials"],
            primary_channel=str(primary_config["channel"]),
            posterior_roi_channels=secondary_config["posterior_roi_channels"],
            window_s=primary_config["time_window_s"],
        )
        for record in records
    ]
    measurements = pd.DataFrame(measurement_rows).sort_values("subject").reset_index(drop=True)
    output_config = config["outputs"]
    results_root = Path(output_config["results_root"])
    figures_root = Path(output_config["figures_root"])
    results_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)
    measurements.to_csv(output_config["subject_measurements"], index=False, encoding="utf-8-sig")

    primary, secondary = compute_paired_statistics(
        measurements["target_pz_mean_uv"].to_numpy(),
        measurements["standard_pz_mean_uv"].to_numpy(),
        bootstrap_iterations=int(primary_config["bootstrap_iterations"]),
        confidence_level=float(primary_config["confidence_level"]),
        random_state=int(primary_config["random_state"]),
    )
    statistics_rows = [
        group_statistics_row(
            primary,
            role="primary",
            metric="Pz 300-600 ms mean amplitude",
            unit="µV",
            p_adjustment="not_applicable_single_primary_test",
        ),
        group_statistics_row(
            secondary,
            role="robustness",
            metric="Pz 300-600 ms mean amplitude",
            unit="µV",
            p_adjustment="not_applicable_robustness_check",
        ),
    ]
    secondary_metrics = (
        ("Pz peak amplitude", "µV", "target_pz_peak_uv", "standard_pz_peak_uv"),
        (
            "Pz peak latency",
            "ms",
            "target_pz_peak_latency_ms",
            "standard_pz_peak_latency_ms",
        ),
        (
            "Pz signed area",
            "µV*s",
            "target_pz_signed_area_uv_s",
            "standard_pz_signed_area_uv_s",
        ),
        (
            "posterior ROI mean amplitude",
            "µV",
            "target_posterior_roi_mean_uv",
            "standard_posterior_roi_mean_uv",
        ),
    )
    for offset, (metric, unit, target_column, standard_column) in enumerate(
        secondary_metrics, start=1
    ):
        metric_result, _ = compute_paired_statistics(
            measurements[target_column].to_numpy(),
            measurements[standard_column].to_numpy(),
            bootstrap_iterations=int(primary_config["bootstrap_iterations"]),
            confidence_level=float(primary_config["confidence_level"]),
            random_state=int(primary_config["random_state"]) + offset,
        )
        statistics_rows.append(
            group_statistics_row(
                metric_result,
                role="secondary",
                metric=metric,
                unit=unit,
                p_adjustment="none_secondary_descriptive",
            )
        )
    statistics_frame = pd.DataFrame(statistics_rows)
    statistics_frame.to_csv(output_config["group_statistics"], index=False, encoding="utf-8-sig")

    primary_channel_index = channel_names.index(str(primary_config["channel"]))
    target_pz_uv = target_eeg_uv[:, primary_channel_index, :]
    standard_pz_uv = standard_eeg_uv[:, primary_channel_index, :]
    waveforms = _plot_group_waveforms(
        times_s=times_s,
        target_uv=target_pz_uv,
        standard_uv=standard_pz_uv,
        window_s=primary_config["time_window_s"],
        iterations=int(primary_config["bootstrap_iterations"]),
        confidence_level=float(primary_config["confidence_level"]),
        random_state=int(primary_config["random_state"]),
        output_path=figures_root / "group_waveforms.png",
    )
    waveforms.to_csv(output_config["group_waveforms"], index=False, encoding="utf-8-sig")
    _plot_paired_amplitudes(measurements, figures_root / "paired_amplitudes.png")
    _plot_difference_distribution(measurements, figures_root / "difference_distribution.png")
    _plot_subject_heatmap(
        subjects=subjects,
        times_s=times_s,
        difference_uv=target_pz_uv - standard_pz_uv,
        window_s=primary_config["time_window_s"],
        output_path=figures_root / "subject_difference_heatmap.png",
    )
    topographies = _plot_topographies(
        eeg_info=eeg_info,
        target_uv=target_eeg_uv,
        standard_uv=standard_eeg_uv,
        times_s=times_s,
        window_s=config["exploratory_analysis"]["topography_window_s"],
        output_path=figures_root / "p3b_topographies.png",
    )
    topographies.to_csv(output_config["group_topographies"], index=False, encoding="utf-8-sig")

    conservative_exclusions = set(
        config["sensitivity_analysis"]["conservative_38"][
            "additionally_excluded_subjects"
        ]
    )
    sensitivity_rows = [
        _sensitivity_row(label="primary_39", measurements=measurements, config=primary_config),
        _sensitivity_row(
            label="conservative_38",
            measurements=measurements[~measurements["subject"].isin(conservative_exclusions)],
            config=primary_config,
        ),
    ]
    inclusive_path = Path(
        config["sensitivity_analysis"]["inclusive_40"]["added_epoch_file"]
    )
    inclusive_record, _ = _load_subject_averages(
        path=inclusive_path,
        target_label=str(config["conditions"]["target"]),
        standard_label=str(config["conditions"]["standard"]),
        required_channels=required_channels,
    )
    if inclusive_record["subject"] in set(measurements["subject"]):
        raise ValueError("inclusive_40 追加的受试者已经存在于主分析数据中")
    if inclusive_record["eeg_channel_names"] != channel_names or not np.array_equal(
        inclusive_record["times_s"], times_s
    ):
        raise ValueError("inclusive_40 追加文件与主分析的通道或时间轴不一致")
    inclusive_measurement = compute_subject_measurements(
        subject=inclusive_record["subject"],
        times_s=times_s,
        channel_names=channel_names,
        target_eeg_uv=inclusive_record["target_eeg_uv"],
        standard_eeg_uv=inclusive_record["standard_eeg_uv"],
        target_trials=inclusive_record["target_trials"],
        standard_trials=inclusive_record["standard_trials"],
        primary_channel=str(primary_config["channel"]),
        posterior_roi_channels=secondary_config["posterior_roi_channels"],
        window_s=primary_config["time_window_s"],
    )
    inclusive_measurements = pd.concat(
        [measurements, pd.DataFrame([inclusive_measurement])], ignore_index=True
    )
    sensitivity_rows.append(
        _sensitivity_row(
            label="inclusive_40", measurements=inclusive_measurements, config=primary_config
        )
    )
    minimum_target_trials = int(
        config["sensitivity_analysis"]["posthoc_low_trial_check"][
            "minimum_target_trials"
        ]
    )
    sensitivity_rows.append(
        _sensitivity_row(
            label=f"posthoc_target_trials_ge_{minimum_target_trials}",
            measurements=measurements[
                measurements["target_trials"] >= minimum_target_trials
            ],
            config=primary_config,
        )
    )
    sensitivity_frame = pd.DataFrame(sensitivity_rows)
    sensitivity_frame.to_csv(
        output_config["sensitivity_results"], index=False, encoding="utf-8-sig"
    )

    if skip_cluster or not bool(cluster_config["enabled"]):
        cluster_summary: dict[str, Any] = {"enabled": False}
    else:
        print(
            f"运行探索性时空置换检验：{cluster_config['permutations']} permutations",
            flush=True,
        )
        cluster_summary = {
            "enabled": True,
            **_run_cluster_permutation(
                difference_uv=difference_eeg_uv,
                times_s=times_s,
                eeg_info=eeg_info,
                config=cluster_config,
                output_npz=Path(output_config["cluster_results"]),
                output_csv=Path(output_config["cluster_summary"]),
                output_figure=figures_root / "cluster_permutation.png",
            ),
        }

    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "phase": 5,
        "config_path": config_path.as_posix(),
        "config_sha256": hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
        "subject_count": len(records),
        "subjects": subjects,
        "input_files": [
            {
                "subject": record["subject"],
                "file": record["file"],
                "target_trials": record["target_trials"],
                "standard_trials": record["standard_trials"],
                "sfreq_hz": record["sfreq_hz"],
            }
            for record in records
        ],
        "primary_analysis": primary_config,
        "cluster_analysis": cluster_summary,
        "software": {
            "python": platform.python_version(),
            "mne": metadata.version("mne"),
            "numpy": metadata.version("numpy"),
            "pandas": metadata.version("pandas"),
            "scipy": metadata.version("scipy"),
        },
    }
    Path(output_config["analysis_manifest"]).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_results_summary(
        primary=primary,
        secondary=secondary,
        group_statistics=statistics_frame,
        sensitivity=sensitivity_frame,
        cluster_summary=cluster_summary,
        output_path=Path(output_config["results_summary"]),
    )
    return {
        "subjects": len(records),
        "primary": primary,
        "secondary": secondary,
        "sensitivity": sensitivity_rows,
        "cluster": cluster_summary,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行 Phase 5 经典 P3b 群体分析。")
    parser.add_argument("--config", type=Path, default=Path("configs/erp_analysis.yaml"))
    parser.add_argument(
        "--skip-cluster",
        action="store_true",
        help="跳过耗时的探索性时空置换；正式结果不应使用该选项。",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=None,
        help="临时覆盖置换次数，主要用于开发检查。",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = run_erp_analysis(
        config_path=args.config,
        skip_cluster=args.skip_cluster,
        permutations_override=args.permutations,
    )
    primary = result["primary"]
    print(
        "Phase 5 complete | "
        f"N={result['subjects']} | "
        # Windows 的默认 GBK 控制台不能编码 µ；文件和图中仍使用正式单位符号。
        f"Target-Standard={primary['mean_difference_uv']:.3f} uV | "
        f"t({primary['degrees_of_freedom']})={primary['statistic']:.3f} | "
        f"p={primary['p_value']:.4g} | dz={primary['effect_size']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

