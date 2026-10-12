from __future__ import annotations

import numpy as np

from erp_core_p3b.preprocessing import (
    choose_eog_components,
    load_manual_ica_exclusions,
    make_epoch_events,
)


def test_make_epoch_events_filters_behavior_and_maps_conditions() -> None:
    trials = [
        {
            "trial_index": 1,
            "condition": "target",
            "stimulus_onset_s": 1.25,
            "accuracy": "correct",
            "behavior_qc_flag": "",
        },
        {
            "trial_index": 2,
            "condition": "standard",
            "stimulus_onset_s": 2.0,
            "accuracy": "incorrect",
            "behavior_qc_flag": "",
        },
        {
            "trial_index": 3,
            "condition": "standard",
            "stimulus_onset_s": 3.0,
            "accuracy": "correct",
            "behavior_qc_flag": "anticipatory",
        },
    ]

    events, metadata = make_epoch_events(
        trials,
        sfreq=256.0,
        first_samp=10,
        require_correct_response=True,
        exclude_behavior_flags=True,
    )

    np.testing.assert_array_equal(events, np.array([[330, 0, 1]]))
    assert metadata.loc[0, "condition_label"] == "Target"


def test_choose_eog_components_sorts_by_largest_absolute_score_and_caps_count() -> None:
    scores = {
        "HEOG_left": np.array([0.1, -0.8, 0.35, 0.2]),
        "VEOG_lower": np.array([0.2, 0.4, -0.7, 0.1]),
    }

    excluded, maximum_scores = choose_eog_components(scores, threshold=0.3, maximum=2)

    assert excluded == [1, 2]
    np.testing.assert_allclose(maximum_scores, [0.2, 0.8, 0.7, 0.2])


def test_load_manual_ica_exclusions_normalizes_component_numbers(tmp_path) -> None:
    path = tmp_path / "manual_qc.yaml"
    path.write_text("ica_exclusions:\n  sub-001: [0, '2']\n", encoding="utf-8")

    result = load_manual_ica_exclusions(path)

    assert result == {"sub-001": [0, 2]}
