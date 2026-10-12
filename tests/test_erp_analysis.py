from __future__ import annotations

import numpy as np

from erp_core_p3b.erp_analysis import (
    bootstrap_mean_ci,
    compute_paired_statistics,
    compute_subject_measurements,
    matched_rank_biserial,
    time_window_mask,
)


def test_time_window_mask_uses_closed_interval() -> None:
    times = np.array([-0.1, 0.0, 0.3, 0.4, 0.6, 0.7])

    result = time_window_mask(times, [0.3, 0.6])

    assert result.tolist() == [False, False, True, True, True, False]


def test_subject_measurement_preserves_microvolt_units_and_pairing() -> None:
    times = np.array([0.0, 0.3, 0.4, 0.5, 0.6, 0.7])
    channels = ["Pz", "CPz", "PO3", "PO4"]
    target = np.array(
        [
            [0.0, 4.0, 6.0, 8.0, 6.0, 0.0],
            [0.0, 2.0, 2.0, 2.0, 2.0, 0.0],
            [0.0, 3.0, 3.0, 3.0, 3.0, 0.0],
            [0.0, 4.0, 4.0, 4.0, 4.0, 0.0],
        ]
    )
    standard = np.ones_like(target)

    result = compute_subject_measurements(
        subject="sub-001",
        times_s=times,
        channel_names=channels,
        target_eeg_uv=target,
        standard_eeg_uv=standard,
        target_trials=30,
        standard_trials=150,
        primary_channel="Pz",
        posterior_roi_channels=channels,
        window_s=[0.3, 0.6],
    )

    assert result["target_pz_mean_uv"] == 6.0
    assert result["standard_pz_mean_uv"] == 1.0
    assert result["difference_pz_mean_uv"] == 5.0
    assert result["target_pz_peak_uv"] == 8.0
    assert result["target_pz_peak_latency_ms"] == 500.0
    assert result["target_trials"] == 30
    assert result["standard_trials"] == 150


def test_paired_statistics_use_subject_differences() -> None:
    standard = np.array([1.0, 2.0, 3.0, 4.0])
    target = standard + np.array([1.0, 2.0, 3.0, 4.0])

    primary, secondary = compute_paired_statistics(
        target,
        standard,
        bootstrap_iterations=200,
        confidence_level=0.95,
        random_state=7,
    )

    expected_dz = np.mean([1.0, 2.0, 3.0, 4.0]) / np.std([1.0, 2.0, 3.0, 4.0], ddof=1)
    assert primary["n_subjects"] == 4
    assert primary["mean_difference_uv"] == 2.5
    assert np.isclose(primary["effect_size"], expected_dz)
    assert primary["degrees_of_freedom"] == 3
    assert secondary["effect_size"] == 1.0


def test_bootstrap_mean_ci_is_reproducible() -> None:
    values = np.arange(12, dtype=float).reshape(4, 3)

    first = bootstrap_mean_ci(
        values, iterations=100, confidence_level=0.95, random_state=42
    )
    second = bootstrap_mean_ci(
        values, iterations=100, confidence_level=0.95, random_state=42
    )

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_matched_rank_biserial_keeps_difference_direction() -> None:
    assert matched_rank_biserial(np.array([1.0, 2.0, 3.0])) == 1.0
    assert matched_rank_biserial(np.array([-1.0, -2.0, -3.0])) == -1.0

