from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import mne
import numpy as np
import pandas as pd
import pytest

from erp_core_p3b.tfr_epochs import (
    artifact_metrics,
    check_ica_compatibility,
    compare_erp_reproduction,
    make_retained_epochs,
    run_builder,
)


def _raw_and_ica():
    info = mne.create_info(["Pz", "Cz", "EOG"], 256, ["eeg", "eeg", "eog"])
    raw = mne.io.RawArray(np.zeros((3, 2000)), info, verbose="ERROR")
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    ica = SimpleNamespace(ch_names=["Pz", "Cz"], info=raw.info.copy(), exclude=[0], n_components_=2)
    return raw, ica


def test_ica_requires_frozen_channel_order_and_exclusions() -> None:
    raw, ica = _raw_and_ica()
    check_ica_compatibility(raw, ica, excluded=[0])
    with pytest.raises(ValueError, match="排除集合"):
        check_ica_compatibility(raw, ica, excluded=[1])
    ica.ch_names.reverse()
    with pytest.raises(ValueError, match="通道/顺序"):
        check_ica_compatibility(raw, ica, excluded=[0])


def test_long_epoch_keeps_trial_identity_without_voltage_baseline() -> None:
    raw, _ = _raw_and_ica()
    metadata = pd.DataFrame({"subject": ["sub-001", "sub-001"], "trial_index": [2, 9]})
    old = mne.Epochs(
        raw,
        np.array([[600, 0, 1], [1200, 0, 2]]),
        event_id={"Target": 1, "Standard": 2},
        tmin=-0.2,
        tmax=0.8,
        metadata=metadata,
        preload=True,
        verbose="ERROR",
    )
    long = make_retained_epochs(raw, old, [-2.1, 2.1], None)
    assert long.baseline is None
    assert len(long.times) == 1077
    np.testing.assert_array_equal(long.events, old.events)
    assert long.metadata["trial_index"].tolist() == [2, 9]


def test_long_epoch_cannot_silently_drop_boundary_trial() -> None:
    raw, _ = _raw_and_ica()
    old = mne.Epochs(
        raw,
        np.array([[300, 0, 1], [1200, 0, 2]]),
        event_id={"Target": 1, "Standard": 2},
        tmin=-0.2,
        tmax=0.8,
        metadata=pd.DataFrame({"trial_index": [1, 2]}),
        preload=True,
        verbose="ERROR",
    )
    with pytest.raises(ValueError, match="事件丢失"):
        make_retained_epochs(raw, old, [-2.1, 2.1], None)


def test_artifact_review_distinguishes_outer_context_without_rejecting() -> None:
    times = np.linspace(-2, 2, 401)
    data = np.zeros((3, 2, len(times)))
    data[0, 0, 10] = 300e-6
    data[1, 0, 200] = 300e-6
    support = pd.DataFrame(
        {
            "baseline_support_start_s": [-1.0],
            "baseline_support_end_s": [-0.01],
            "display_support_start_s": [-0.6],
            "display_support_end_s": [1.2],
        }
    )
    metrics, channels = artifact_metrics(data, times, support, 200)
    assert metrics["full_review_flag"].tolist() == [True, True, False]
    assert metrics["protected_review_flag"].tolist() == [False, True, False]
    assert metrics["outer_context_only_review_flag"].tolist() == [True, False, False]
    assert channels.shape == (3, 2)
    data[2, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        artifact_metrics(data, times, support, 200)


def test_erp_reproduction_rejects_signal_change() -> None:
    info = mne.create_info(["Pz"], 256, "eeg")
    original = mne.EpochsArray(np.zeros((1, 1, 257)), info, tmin=-0.2, verbose="ERROR")
    matching = original.copy()
    assert compare_erp_reproduction(matching, original, 0.001)["erp_reproduction_max_error_uv"] == 0
    changed = mne.EpochsArray(np.ones((1, 1, 257)) * 1e-6, info, tmin=-0.2, verbose="ERROR")
    with pytest.raises(ValueError, match="复现误差"):
        compare_erp_reproduction(changed, original, 0.001)


def test_final_run_is_blocked_before_loading_signals_when_pilot_needs_review(tmp_path, monkeypatch):
    config = {"outputs": {"results_root": str(tmp_path)}}
    (tmp_path / "pilot").mkdir()
    (tmp_path / "pilot/run_summary.json").write_text(
        json.dumps({"eligible_for_full_run": False, "config_hashes": {}}), encoding="utf-8"
    )
    monkeypatch.setattr(
        "erp_core_p3b.tfr_epochs.load_and_validate_inputs", lambda *_: (config, {}, {}, {})
    )
    with pytest.raises(ValueError, match="三人pilot尚未通过"):
        run_builder(config_path=Path("unused"), feature_path=Path("unused"), stage="final")


def test_double_fif_roundtrip_uses_physical_tolerance_not_bitwise_equality(tmp_path):
    info = mne.create_info(["Pz"], 256, "eeg")
    # FIF通道校准头有独立的有限精度；double数据载荷不能保证读取后逐位相等。
    info["chs"][0]["cal"] = 1e-6
    original = mne.EpochsArray(np.full((1, 1, 257), 100e-6), info, tmin=-0.2, verbose="ERROR")
    path = tmp_path / "roundtrip_epo.fif"
    original.save(path, fmt="double", verbose="ERROR")
    restored = mne.read_epochs(path, preload=True, verbose="ERROR")
    result = compare_erp_reproduction(restored, original, 0.001)
    assert 0 < result["erp_reproduction_max_error_uv"] < 0.001
