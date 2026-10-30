from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from erp_core_p3b.tfr_audit import (
    annotation_overlap,
    estimate_memory_bytes,
    map_retained_trials,
    morlet_support_table,
    neighbor_overlap_flags,
    recording_window_validity,
)


def test_recording_window_includes_last_sample_but_not_after_recording() -> None:
    result = recording_window_validity(
        np.array([10, 11, 89, 90]), n_times=100, sfreq_hz=10, window_s=[-1, 1]
    )
    assert result.tolist() == [True, True, True, False]


def test_boundary_annotations_are_detected_even_when_duration_is_zero() -> None:
    result = annotation_overlap(
        np.array([1.0, 5.0, 9.0]),
        [-0.5, 0.5],
        [(1.0, 0.0, "11"), (5.0, 0.0, "boundary"), (9.25, 1.0, "BAD_noise")],
    )
    assert result.tolist() == [False, True, True]


def test_morlet_actual_kernel_and_baseline_support_expose_2hz_problem() -> None:
    config = {
        "cycle_divisor_hz": 2.0,
        "min_cycles": 3.0,
        "max_cycles": 10.0,
        "zero_mean": True,
    }
    result = morlet_support_table(
        sfreq_hz=256,
        frequencies_hz=np.array([2.0, 6.0]),
        config=config,
        epoch_window_s=[-1.5, 2.0],
        baseline_window_s=[-0.8, -0.2],
        display_window_s=[-0.2, 0.8],
    )
    low = result.iloc[0]
    assert low["kernel_samples"] == 611
    assert low["half_support_s"] == 305 / 256
    assert low["display_support_end_s"] == 205 / 256 + 305 / 256
    assert not low["baseline_edge_safe"]
    assert low["baseline_reaches_current_stimulus"]
    high = morlet_support_table(
        sfreq_hz=256,
        frequencies_hz=np.array([6.0]),
        config=config,
        epoch_window_s=[-1.1, 1.3],
        baseline_window_s=[-0.6, -0.4],
        display_window_s=[-0.2, 0.8],
    ).iloc[0]
    assert high["baseline_edge_safe"] and high["display_edge_safe"]
    assert not high["baseline_reaches_current_stimulus"]


def test_trial_alignment_counts_discarded_stimulus_as_neighbor() -> None:
    original = [
        {
            "trial_index": index,
            "stimulus_onset_s": onset,
            "stimulus_code": 11,
            "condition": "target",
            "valid_behavior_trial": True,
        }
        for index, onset in enumerate([2.0, 3.5, 5.0], start=1)
    ]
    retained = pd.DataFrame(
        {
            "subject": ["sub-001", "sub-001"],
            "trial_index": [1, 3],
            "stimulus_onset_s": [2.0, 5.0],
            "stimulus_code": [11, 11],
            "condition_label": ["Target", "Target"],
            "event_sample": [512, 1280],
        }
    )
    result = map_retained_trials(
        retained,
        original,
        sfreq_hz=256,
        event_samples=np.array([512, 1280]),
        event_codes=np.array([1, 1]),
    )
    assert result.loc[0, "next_gap_s"] == 1.5
    assert result.loc[1, "previous_gap_s"] == 1.5
    assert np.isnan(result.loc[0, "previous_gap_s"])
    altered = retained.copy()
    altered.loc[1, "stimulus_onset_s"] = 5.1
    with pytest.raises(ValueError, match="刺激时间"):
        map_retained_trials(
            altered,
            original,
            sfreq_hz=256,
            event_samples=np.array([512, 1280]),
            event_codes=np.array([1, 1]),
        )


def test_neighbor_presentations_have_duration_and_undefined_neighbors_are_safe() -> None:
    trials = pd.DataFrame(
        {
            "previous_gap_s": [1.4, np.nan, 1.6],
            "next_gap_s": [1.4, 1.5, np.nan],
            "previous_stimulus_duration_s": [0.2, np.nan, 0.2],
        }
    )
    previous, following = neighbor_overlap_flags(trials, left_s=-1.3, right_s=1.4)
    assert previous.tolist() == [True, False, False]
    assert following.tolist() == [True, False, False]


def test_memory_model_distinguishes_float64_power_and_complex128() -> None:
    result = estimate_memory_bytes(
        n_trials=100,
        n_channels=30,
        n_times=1000,
        n_frequencies=29,
        raw_n_times=100_000,
        raw_n_channels=33,
        raw_copies=3,
        batch_trials=8,
    )
    assert result["full_power_float64_bytes"] == 100 * 30 * 1000 * 29 * 8
    assert result["full_complex128_bytes"] == 2 * result["full_power_float64_bytes"]
    assert result["batch_complex128_bytes"] < result["full_complex128_bytes"]


def test_frozen_main_parameters_protect_baseline_and_display_support() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((project_root / "configs/features.yaml").read_text(encoding="utf-8"))
    main = config["time_frequency"]
    frequencies = np.arange(main["fmin_hz"], main["fmax_hz"] + 0.01, main["frequency_step_hz"])
    support = morlet_support_table(
        sfreq_hz=main["target_sfreq_hz"],
        frequencies_hz=frequencies,
        config=main["morlet"],
        epoch_window_s=main["epoch_window_s"],
        baseline_window_s=main["baseline_s"],
        display_window_s=main["display_window_s"],
    )
    assert main["parameters_frozen"]
    assert support["kernel_fits_epoch"].all()
    assert support["baseline_edge_safe"].all() and support["display_edge_safe"].all()
    assert not support["baseline_reaches_current_stimulus"].any()
    low = main["exploratory_low_frequency"]
    assert low["baseline_s"] is None and not low["include_in_primary_ml_features"]
