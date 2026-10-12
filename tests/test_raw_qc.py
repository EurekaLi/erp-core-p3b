from __future__ import annotations

import numpy as np
import pandas as pd

from erp_core_p3b.raw_qc import (
    _preserve_manual_review,
    longest_flat_duration_s,
    pair_behavior_events,
    robust_zscore,
)


def test_pair_behavior_events_tracks_missing_multiple_and_unmatched() -> None:
    rows = [
        {"onset": "0.1", "value": "201"},
        {"onset": "1.0", "value": "11"},
        {"onset": "1.4", "value": "201"},
        {"onset": "1.5", "value": "202"},
        {"onset": "2.5", "value": "12"},
        {"onset": "4.0", "value": "22"},
        {"onset": "4.05", "value": "201"},
    ]

    trials, summary = pair_behavior_events(
        rows,
        subject="sub-test",
        min_reaction_time_s=0.15,
        max_response_window_s=1.6,
    )

    assert len(trials) == 3
    assert trials[0]["accuracy"] == "correct"
    assert trials[0]["response_count"] == 2
    assert trials[1]["accuracy"] == "missing"
    assert "anticipatory" in trials[2]["behavior_qc_flag"]
    assert summary["unmatched_responses"] == 1
    assert summary["multiple_response_trials"] == 1


def test_longest_flat_duration() -> None:
    data = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 2.0])

    duration = longest_flat_duration_s(data, sfreq=2.0, tolerance_uv=0.0)

    assert duration == 1.5


def test_robust_zscore_constant_array_is_zero() -> None:
    result = robust_zscore(np.array([3.0, 3.0, 3.0]))

    np.testing.assert_array_equal(result, np.zeros(3))


def test_preserve_manual_review_after_rerun(tmp_path) -> None:
    path = tmp_path / "review.tsv"
    pd.DataFrame(
        [
            {
                "subject": "sub-001",
                "channel": "Pz",
                "manual_decision": "bad",
                "manual_reason": "visual confirmation",
            }
        ]
    ).to_csv(path, sep="\t", index=False)
    current = pd.DataFrame(
        [{"subject": "sub-001", "channel": "Pz", "automatic_flags": "low_correlation"}]
    )

    result = _preserve_manual_review(
        current,
        path,
        key_columns=["subject", "channel"],
        manual_columns=["manual_decision", "manual_reason"],
    )

    assert result.loc[0, "manual_decision"] == "bad"
    assert result.loc[0, "manual_reason"] == "visual confirmation"

