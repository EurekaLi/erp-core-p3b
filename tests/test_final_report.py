"""报告统计契约：拒绝错折/漏行OOF，保持同人配对，并保留trial错误定位。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erp_core_p3b.final_report import compare_subjects, trial_errors, validate_oof
from erp_core_p3b.logistic_baseline import subject_metrics


@pytest.fixture
def oof_case():
    trials = pd.DataFrame(
        {
            "trial_id": ["a1", "a2", "b1", "b2"],
            "subject_id": ["a", "a", "b", "b"],
            "label": [0, 1, 0, 1],
            "epoch_index": [0, 1, 0, 1],
            "trial_index": [1, 3, 2, 4],
        }
    )
    splits = {
        "folds": [
            {"fold": 1, "trial_ids": {"test": ["a1", "a2"]}},
            {"fold": 2, "trial_ids": {"test": ["b1", "b2"]}},
        ]
    }
    predictions = trials[["trial_id", "subject_id", "label"]].assign(
        experiment="eegnet",
        fold=[1, 1, 2, 2],
        score=[0.1, 0.7, 0.6, 0.8],
    )
    return trials, splits, predictions


def test_oof_identity_join_is_order_independent(oof_case):
    trials, splits, predictions = oof_case
    aligned = validate_oof(
        predictions.sample(frac=1, random_state=1), trials, splits, ["eegnet"], 0.5
    )
    assert aligned["trial_id"].tolist() == trials["trial_id"].tolist()
    assert aligned["prediction"].tolist() == [0, 1, 1, 1]


@pytest.mark.parametrize(
    "failure", ["duplicate", "missing", "fold", "label", "subject", "score", "prediction"]
)
def test_oof_rejects_corruption_or_validation_in_test(failure, oof_case):
    trials, splits, predictions = oof_case
    bad = predictions.copy()
    if failure == "duplicate":
        bad = pd.concat([bad, bad.iloc[[0]]], ignore_index=True)
    elif failure == "missing":
        bad = bad.iloc[:-1]
    elif failure == "fold":
        bad.loc[0, "fold"] = 2
    elif failure == "label":
        bad.loc[0, "label"] = 1
    elif failure == "subject":
        bad.loc[0, "subject_id"] = "b"
    elif failure == "score":
        bad.loc[0, "score"] = np.inf
    else:
        bad["prediction"] = 0
    with pytest.raises(ValueError):
        validate_oof(bad, trials, splits, ["eegnet"], 0.5)


def test_paired_difference_uses_same_person_and_direction(oof_case):
    _, _, predictions = oof_case
    better = predictions.assign(experiment="improved", score=[0.1, 0.9, 0.1, 0.9])
    metrics = subject_metrics(pd.concat([predictions, better], ignore_index=True), 0.5)
    config = {"bootstrap_iterations": 100, "confidence_level": 0.95, "random_state": 42}
    macro, paired = compare_subjects(
        metrics, ["eegnet", "improved"], [["eegnet", "improved"]], config
    )
    row = paired[paired["metric"] == "balanced_accuracy"].iloc[0]
    assert row["mean_candidate_minus_reference"] == 0.25
    assert row["candidate_higher_subjects"] == 1 and row["tied_subjects"] == 1
    assert (
        macro.query("experiment == 'improved' and metric == 'balanced_accuracy'")["mean"].item()
        == 1
    )
    with pytest.raises(ValueError, match="漏人"):
        compare_subjects(metrics.iloc[:-1], ["eegnet", "improved"], [], config)


def test_error_join_preserves_original_trial_and_common_error_count(oof_case):
    trials, _, predictions = oof_case
    frames = [
        predictions.assign(experiment=name)
        for name in ("eegnet", "cbramod_linear", "cbramod_finetune")
    ]
    errors = trial_errors(
        pd.concat(frames, ignore_index=True),
        trials,
        ["eegnet", "cbramod_linear", "cbramod_finetune"],
        0.5,
    )
    common = errors[errors["nn_error_models"] == 3]
    assert common["trial_id"].tolist() == ["b1"]
    assert common["label"].item() == 0
    assert common["trial_index"].item() == 2 and common["epoch_index"].item() == 0
    assert not errors["dl_prediction_disagreement"].any()
