"""从每条已清洗原生EEG独立提取固定公式特征；不fit跨样本统计量。"""

from __future__ import annotations

import argparse
import gc
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
import pywt
import yaml
from scipy.signal import welch
from threadpoolctl import threadpool_limits

from erp_core_p3b.cbramod_audit import sha256_file, validate_epoch_header
from erp_core_p3b.classification_data import load_classification_dataset
from erp_core_p3b.classification_io import (
    canonical_hash,
    new_artifact_run,
    resolve_path,
    write_json,
)


def window_mask(times: np.ndarray, start: float, end: float) -> np.ndarray:
    """使用真实时间轴的闭区间；不把0.80078125秒错误算作<=0.800秒。"""
    if not np.isfinite([start, end]).all() or start >= end:
        raise ValueError("特征窗口必须是有限且递增的区间")
    mask = (times >= start) & (times <= end)
    if not mask.any():
        raise ValueError("特征窗口不包含采样点")
    return mask


def compute_compact_features(
    data_v: np.ndarray,
    times_s: np.ndarray,
    sfreq: float,
    names: Sequence[str],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    """输入单位V，输出40列固定特征；字典记录实际窗/频率bin及DWT边界。"""
    data = np.asarray(data_v, dtype=np.float64)
    times = np.asarray(times_s, dtype=float)
    if data.ndim != 3 or data.shape[1:] != (len(names), len(times)) or not np.isfinite(data).all():
        raise ValueError("EEG维度错误或含NaN/Inf")
    if not np.allclose(np.diff(times), 1 / sfreq, rtol=0, atol=1e-10):
        raise ValueError("特征时间轴与声明采样率不同")
    channels = config["channels"]
    if len(channels) != len(set(channels)) or not set(channels).issubset(names):
        raise ValueError("ROI通道缺失/重复")
    uv = data[:, [list(names).index(channel) for channel in channels], :] * 1e6
    values, dictionary = {}, []

    for window in config["erp_windows"]:
        mask = window_mask(times, window["start_s"], window["end_s"])
        amplitudes = uv[:, :, mask].mean(axis=-1)
        for index, channel in enumerate(channels):
            name = f"erp_{channel}_{window['name']}_mean_uv"
            values[name] = amplitudes[:, index]
            dictionary.append(
                {
                    "name": name,
                    "family": "erp",
                    "channel": channel,
                    "unit": "uV",
                    "formula": "mean(single_trial_eeg_v * 1e6)",
                    "requested_window_s": [window["start_s"], window["end_s"]],
                    "actual_window_s": [float(times[mask][0]), float(times[mask][-1])],
                    "sample_count": int(mask.sum()),
                    "boundary": "closed_interval",
                }
            )

    power = config["power"]
    mask = window_mask(times, *power["window_s"])
    parameters = power["welch"]
    nperseg, noverlap, nfft = (int(parameters[key]) for key in ("nperseg", "noverlap", "nfft"))
    if nperseg > mask.sum() or not 0 <= noverlap < nperseg or nfft < nperseg:
        raise ValueError("Welch窗口/重叠参数不适合短epoch，禁止自动缩短参数")
    if power["log_floor_uv2"] <= 0:
        raise ValueError("功率log下限必须为正µV²")
    frequencies, psd = welch(
        uv[:, :, mask],
        fs=sfreq,
        window=parameters["window"],
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=parameters["detrend"],
        scaling="density",
        average="mean",
        axis=-1,
    )
    delta = float(frequencies[1] - frequencies[0])
    for band in power["bands"]:
        if not 0 <= band["left_hz"] < band["right_hz"] <= sfreq / 2:
            raise ValueError("频带边界无效")
        right = (
            frequencies <= band["right_hz"]
            if band["include_right"]
            else frequencies < band["right_hz"]
        )
        selected = (frequencies >= band["left_hz"]) & right
        if not selected.any():
            raise ValueError("频带不包含任何实际Welch bin")
        # 固定矩形bin积分，不声称等同于理想连续频带或精细时频分析。
        log_power = np.log10(
            np.maximum(psd[:, :, selected].sum(axis=-1) * delta, power["log_floor_uv2"])
        )
        for index, channel in enumerate(channels):
            name = f"power_{channel}_{band['name']}_log10_uv2"
            values[name] = log_power[:, index]
            dictionary.append(
                {
                    "name": name,
                    "family": "power",
                    "channel": channel,
                    "unit": "log10(power / 1 uV^2)",
                    "formula": "log10(max(sum(selected_psd_bins)*delta_f, floor_uv2))",
                    "band_hz": [band["left_hz"], band["right_hz"]],
                    "include_right": band["include_right"],
                    "frequency_bins_hz": frequencies[selected].tolist(),
                    "delta_f_hz": delta,
                    "welch": parameters,
                    "requested_window_s": power["window_s"],
                    "actual_window_s": [float(times[mask][0]), float(times[mask][-1])],
                    "sample_count": int(mask.sum()),
                    "log_floor_uv2": power["log_floor_uv2"],
                }
            )

    wavelet = config["wavelet"]
    family, level = pywt.Wavelet(wavelet["family"]), int(wavelet["level"])
    maximum = pywt.dwt_max_level(len(times), family.dec_len)
    if not 1 <= level <= maximum or not wavelet["entropy_normalized"]:
        raise ValueError("DWT层数超出短序列可靠上限或熵定义改变")
    if len(set(wavelet["detail_levels"])) != len(wavelet["detail_levels"]):
        raise ValueError("重复的小波detail层")
    coefficients = pywt.wavedec(uv, family, mode=wavelet["mode"], level=level, axis=-1)
    energies = np.stack([np.square(block).sum(axis=-1) for block in coefficients], axis=-1)
    total = energies.sum(axis=-1, keepdims=True)
    proportions = np.divide(energies, total, out=np.zeros_like(energies), where=total > 0)
    log_p = np.zeros_like(proportions)
    np.log(proportions, out=log_p, where=proportions > 0)
    entropy = -(proportions * log_p).sum(axis=-1) / np.log(len(coefficients))
    for detail in wavelet["detail_levels"]:
        if not 1 <= detail <= level:
            raise ValueError("请求的小波detail不在分解层数内")
        for index, channel in enumerate(channels):
            name = f"wavelet_{channel}_D{detail}_relative_energy"
            values[name] = proportions[:, index, level - detail + 1]
            dictionary.append(
                {
                    "name": name,
                    "family": "wavelet",
                    "channel": channel,
                    "unit": "ratio",
                    "formula": "sum(Dj^2) / sum(all_A_and_D_blocks_squared)",
                    "nominal_band_hz": [sfreq / 2 ** (detail + 1), sfreq / 2**detail],
                    "wavelet": wavelet,
                    "sample_count": len(times),
                    "boundary_note": "symmetric extension; not ideal band power",
                }
            )
    for index, channel in enumerate(channels):
        name = f"wavelet_{channel}_block_energy_entropy"
        values[name] = entropy[:, index]
        dictionary.append(
            {
                "name": name,
                "family": "wavelet",
                "channel": channel,
                "unit": "ratio",
                "formula": "-sum(p_block*ln(p_block)) / ln(number_of_blocks)",
                "zero_energy_convention": 0.0,
                "wavelet": wavelet,
                "sample_count": len(times),
            }
        )
    frame = pd.DataFrame(values)
    if frame.shape[1] != len(dictionary) or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("特征重名或非有限值")
    return (
        frame,
        dictionary,
        {
            "welch_segments": 1 + (int(mask.sum()) - nperseg) // (nperseg - noverlap),
            "dwt_max_level": maximum,
            "zero_energy_channel_trials": int((total == 0).sum()),
        },
    )


def extract_features(
    config_path: Path,
    dataset_dir: Path,
    root: Path,
    *,
    stage: str = "full",
    output: Path | None = None,
) -> Path:
    if stage not in ("pilot", "full"):
        raise ValueError("stage必须是pilot或full")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    trials, dataset, _ = load_classification_dataset(dataset_dir)
    output = resolve_path(root, output or config["outputs"][f"{stage}_directory"])
    with new_artifact_run(root, output, "compact_features") as summary:
        write_json(output / "config_snapshot.json", config)
        selected = set(config["pilot_subjects"]) if stage == "pilot" else set(trials["subject_id"])
        if not selected or not selected.issubset(trials["subject_id"]):
            raise ValueError("pilot受试者不在数据清单中")
        frames, subject_stats, dictionary = [], [], None
        with threadpool_limits(limits=1):
            for source in dataset["sources"]:
                subject = source["subject"]
                if subject not in selected:
                    continue
                print(f"Extracting {subject} ...", flush=True)
                path = resolve_path(root, source["source_file"])
                if sha256_file(path) != source["sha256"]:
                    raise ValueError(f"{subject}: 原始FIF已改变")
                epochs = mne.read_epochs(path, preload=True, verbose="ERROR")
                picks, identities = validate_epoch_header(
                    epochs, subject, dataset["header_protocol"]
                )
                expected = trials[trials["subject_id"] == subject].reset_index(drop=True)
                for column in ("trial_id", "label", "epoch_index", "event_sample_256hz"):
                    if not np.array_equal(identities[column], expected[column]):
                        raise ValueError(f"{subject}: {column}与冻结清单不同")
                names = [epochs.ch_names[index] for index in picks]
                if names != dataset["eeg_channel_names"] or not np.array_equal(
                    epochs.times, dataset["times_s"]
                ):
                    raise ValueError("通道或时间轴与数据版本不同")
                values, definitions, stats = compute_compact_features(
                    epochs.get_data(picks=picks, copy=False),
                    epochs.times,
                    epochs.info["sfreq"],
                    names,
                    config,
                )
                if dictionary is None:
                    dictionary = definitions
                if definitions != dictionary:
                    raise ValueError("受试者之间特征定义改变")
                frames.append(
                    pd.concat([identities[["trial_id", "subject_id", "label"]], values], axis=1)
                )
                subject_stats.append({"subject_id": subject, "trials": len(epochs), **stats})
                if sha256_file(path) != source["sha256"]:
                    raise ValueError("提取期间FIF改变")
                del epochs, values
                gc.collect()
        features = pd.concat(frames, ignore_index=True)
        expected_ids = trials[trials["subject_id"].isin(selected)]["trial_id"].tolist()
        if features["trial_id"].tolist() != expected_ids:
            raise ValueError("提取结果顺序/行数与清单不同")
        features.to_csv(output / "features.csv", index=False)
        write_json(output / "feature_dictionary.json", dictionary)
        pd.DataFrame(subject_stats).to_csv(output / "subject_feature_audit.csv", index=False)
        features[[item["name"] for item in dictionary]].describe().to_csv(
            output / "feature_descriptive_statistics.csv"
        )
        summary.update(
            dataset_hash=dataset["dataset_hash"],
            dataset_dir=str(dataset_dir.resolve()),
            config_sha256=sha256_file(config_path),
            feature_definition_hash=canonical_hash(dictionary),
            extraction_stage=stage,
            complete_dataset=stage == "full",
            trial_count=len(features),
            feature_count=len(dictionary),
            feature_families={
                family: sum(item["family"] == family for item in dictionary)
                for family in ("erp", "power", "wavelet")
            },
            learned_transforms_fitted=False,
        )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path, default=Path("configs/compact_features.yaml"))
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("data/processed/classification/v1")
    )
    parser.add_argument("--stage", choices=("pilot", "full"), default="full")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        output = extract_features(
            resolve_path(args.project_root, args.config),
            resolve_path(args.project_root, args.dataset_dir),
            args.project_root,
            stage=args.stage,
            output=args.output_dir,
        )
    except Exception as error:
        print(f"Feature extraction failed: {error}", file=sys.stderr)
        return 1
    print(f"Compact features: {output}")
    return 0
