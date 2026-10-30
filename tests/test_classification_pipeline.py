"""合成受试者端到端CPU测试；不读取用户真实FIF，不运行真实LR实验。"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import mne
import numpy as np
import pandas as pd
import pytest
import yaml

from erp_core_p3b.classification_data import (
    build_manifest,
    load_classification_dataset,
    make_subject_splits,
    validate_splits,
)
from erp_core_p3b.classification_io import new_artifact_run, verify_artifact, write_json
from erp_core_p3b.compact_features import compute_compact_features, extract_features, window_mask
from erp_core_p3b.logistic_baseline import (
    binary_metrics,
    bootstrap_mean,
    fit_pipeline,
    train_baseline,
)

PROJECT = Path(__file__).resolve().parents[1]


def _config(name):
    return yaml.safe_load((PROJECT / "configs" / name).read_text(encoding="utf-8"))


def _trials():
    rows = []
    for subject in range(10):
        for trial in range(8):
            rows.append(
                {
                    "trial_id": f"sub-{subject + 1:03d}:trial-{trial + 1:04d}",
                    "subject_id": f"sub-{subject + 1:03d}",
                    "label": int(trial < 2),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_project(tmp_path):
    header = _config("cbramod_audit.yaml")
    header["dataset"].update(
        expected_subject_count=10,
        expected_trial_count=80,
        expected_target_count=20,
        expected_standard_count=60,
    )
    (tmp_path / "configs").mkdir()
    header_path = tmp_path / "configs/cbramod_audit.yaml"
    header_path.write_text(yaml.safe_dump(header), encoding="utf-8")
    dataset_config = _config("classification_dataset.yaml")
    dataset_config["evidence"] = {}
    config_path = tmp_path / "configs/classification_dataset.yaml"
    config_path.write_text(yaml.safe_dump(dataset_config), encoding="utf-8")
    feature_config = _config("compact_features.yaml")
    feature_config["pilot_subjects"] = ["sub-001", "sub-002", "sub-003"]
    feature_path = tmp_path / "configs/compact_features.yaml"
    feature_path.write_text(yaml.safe_dump(feature_config), encoding="utf-8")
    model_config = _config("modeling.yaml")
    model_config["statistics"]["bootstrap_iterations"] = 30
    model_path = tmp_path / "configs/modeling.yaml"
    model_path.write_text(yaml.safe_dump(model_config), encoding="utf-8")
    names = ["CPz", "Pz", "PO3", "PO4", *[f"EEG{i}" for i in range(26)], "EOG1", "EOG2", "EOG3"]
    info = mne.create_info(names, 256, ["eeg"] * 30 + ["eog"] * 3)
    with info._unlock():
        info["highpass"], info["lowpass"] = 0.1, 30.0
    directory = tmp_path / header["dataset"]["epochs_root"]
    directory.mkdir(parents=True)
    times = np.arange(-51, 206) / 256
    inputs, subjects = [], []
    for number in range(1, 11):
        subject = f"sub-{number:03d}"
        subjects.append(subject)
        rng = np.random.default_rng(number)
        data = rng.normal(scale=0.4e-6, size=(8, 33, 257))
        data[:2, :4, :] += 4e-6 * np.exp(-(((times - 0.4) / 0.1) ** 2))
        data[:, :30, :] -= data[:, :30, :].mean(axis=1, keepdims=True)
        events = np.array([[500 + trial * 400, 0, 1 if trial < 2 else 2] for trial in range(8)])
        frame = pd.DataFrame(
            {
                "subject": [subject] * 8,
                "trial_index": np.arange(1, 9),
                "condition_label": ["Target"] * 2 + ["Standard"] * 6,
                "event_sample": events[:, 0],
            }
        )
        epochs = mne.EpochsArray(
            data,
            info,
            events=events,
            event_id={"Target": 1, "Standard": 2},
            tmin=-0.2,
            baseline=(-0.2, 0),
            metadata=frame,
            verbose="ERROR",
        )
        epochs.save(directory / f"{subject}_task-P3_epo.fif", fmt="double", verbose="ERROR")
        inputs.append({"subject": subject, "target_trials": 2, "standard_trials": 6})
    phase5 = tmp_path / header["dataset"]["phase5_manifest"]
    phase5.parent.mkdir(parents=True)
    write_json(phase5, {"subjects": subjects, "input_files": inputs})
    return tmp_path, config_path, feature_path, model_path


def test_splits_reproducible_and_subject_disjoint():
    trials, config = _trials(), _config("classification_dataset.yaml")["split"]
    first = make_subject_splits(trials, config, "synthetic")
    assert first == make_subject_splits(trials, config, "synthetic")
    validate_splits(trials, first)
    for fold in first["folds"]:
        assert not set(fold["subjects"]["train"]) & set(fold["subjects"]["test"])
        assert not set(fold["subjects"]["validation"]) & set(fold["subjects"]["test"])


def test_split_overlap_is_rejected():
    trials, config = _trials(), _config("classification_dataset.yaml")["split"]
    splits = make_subject_splits(trials, config, "synthetic")
    splits["folds"][0]["subjects"]["validation"].append(splits["folds"][0]["subjects"]["train"][0])
    with pytest.raises(ValueError):
        validate_splits(trials, splits)


def test_missing_test_coverage_is_rejected():
    trials, config = _trials(), _config("classification_dataset.yaml")["split"]
    splits = make_subject_splits(trials, config, "synthetic")
    splits["folds"] = splits["folds"][:-1]
    with pytest.raises(ValueError):
        validate_splits(trials, splits)


def test_feature_units_frequency_bins_and_count():
    times = np.arange(-51, 206) / 256
    data = np.broadcast_to((10e-6 * np.sin(2 * np.pi * 10 * times))[None, None, :], (2, 4, 257))
    config = _config("compact_features.yaml")
    frame, dictionary, audit = compute_compact_features(
        data, times, 256, config["channels"], config
    )
    assert frame.shape == (2, 40)
    assert audit["welch_segments"] == 2 and audit["dwt_max_level"] == 5
    assert pd.Series([item["family"] for item in dictionary]).value_counts().to_dict() == {
        "wavelet": 16,
        "erp": 12,
        "power": 12,
    }
    np.testing.assert_allclose(frame["power_Pz_alpha_log10_uv2"], np.log10(50), atol=1e-8)
    expected = data[0, 1, (times >= 0.3) & (times <= 0.6)].mean() * 1e6
    np.testing.assert_allclose(frame["erp_Pz_w300_600_mean_uv"], expected)
    alpha = next(item for item in dictionary if item["name"] == "power_Pz_alpha_log10_uv2")
    assert alpha["frequency_bins_hz"] == [8.0, 10.0, 12.0]
    assert all(0 <= value <= 1 for value in frame.filter(like="entropy").to_numpy().ravel())


def test_zero_signal_has_finite_features_and_explicit_convention():
    config = _config("compact_features.yaml")
    frame, _, audit = compute_compact_features(
        np.zeros((1, 4, 257)), np.arange(-51, 206) / 256, 256, config["channels"], config
    )
    assert np.isfinite(frame.to_numpy()).all()
    assert (frame.filter(like="power_") == -12).all().all()
    assert (frame.filter(like="wavelet_") == 0).all().all()
    assert audit["zero_energy_channel_trials"] == 4


@pytest.mark.parametrize("case", ["nan", "missing_channel", "wavelet_level", "welch"])
def test_bad_feature_input_refused(case):
    config = _config("compact_features.yaml")
    data = np.zeros((1, 4, 257))
    names = config["channels"].copy()
    if case == "nan":
        data[0, 0, 10] = np.nan
    elif case == "missing_channel":
        names[0] = "wrong"
    elif case == "wavelet_level":
        config["wavelet"]["level"] = 6
    else:
        config["power"]["welch"]["nperseg"] = 256
    with pytest.raises(ValueError):
        compute_compact_features(data, np.arange(-51, 206) / 256, 256, names, config)


def test_closed_window_respects_native_endpoint():
    times = np.arange(-51, 206) / 256
    mask = window_mask(times, 0, 0.8)
    assert mask.sum() == 205
    assert times[mask][-1] == 0.796875
    assert not mask[-1]


def test_train_scaler_uses_only_training_statistics():
    x = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=float)
    pipeline = fit_pipeline(x, np.array([0, 0, 1, 1]), _config("modeling.yaml")["model"])
    np.testing.assert_allclose(pipeline.named_steps["scaler"].mean_, x.mean(axis=0))
    original = pipeline.named_steps["scaler"].mean_.copy()
    pipeline.predict_proba(np.ones((3, 2)) * 100000)
    np.testing.assert_array_equal(pipeline.named_steps["scaler"].mean_, original)


def test_lr_nonconvergence_is_not_silently_reported():
    config = _config("modeling.yaml")["model"]
    config["max_iter"] = 1
    x = np.random.default_rng(12).normal(size=(100, 5))
    labels = (x[:, 0] > 0).astype(int)
    with pytest.raises(RuntimeError, match="未收敛"):
        fit_pipeline(x, labels, config)


def test_imbalanced_standard_only_reference():
    scores = binary_metrics(np.array([0] * 8 + [1] * 2), np.zeros(10))
    assert scores["accuracy"] == 0.8
    assert scores["balanced_accuracy"] == 0.5
    assert scores["average_precision"] == 0.2
    assert scores["target_recall"] == 0


def test_single_class_metrics_are_missing():
    metrics = binary_metrics(np.zeros(4, dtype=int), np.zeros(4))
    assert np.isnan(metrics["roc_auc"]) and np.isnan(metrics["average_precision"])
    assert np.isnan(metrics["balanced_accuracy"])


def test_subject_bootstrap_reproducible():
    config = _config("modeling.yaml")["statistics"]
    assert bootstrap_mean(np.array([0.4, 0.6, 0.8]), config) == bootstrap_mean(
        np.array([0.4, 0.6, 0.8]), config
    )
    assert bootstrap_mean(np.ones(5), config) == (1.0, 1.0, 1.0)


def test_end_to_end_smoke_and_full_on_synthetic_project(synthetic_project):
    root, dataset_config, feature_config, model_config = synthetic_project
    dataset_dir = build_manifest(dataset_config, root)
    trials, _, splits = load_classification_dataset(dataset_dir)
    assert len(trials) == 80 and len(splits["folds"]) == 5
    assert len(pd.read_csv(dataset_dir / "split_audit.csv")) == 15
    feature_dir = extract_features(feature_config, dataset_dir, root)
    features = pd.read_csv(feature_dir / "features.csv")
    assert features.shape == (80, 43)
    smoke = train_baseline(
        model_config, dataset_dir, feature_dir, root, mode="smoke", run_name="synthetic-smoke"
    )
    assert not (smoke / "test_predictions.csv").exists()
    validation = pd.read_csv(smoke / "validation_predictions.csv")
    assert set(validation["subject_id"]) == set(splits["folds"][0]["subjects"]["validation"])
    assert not set(validation["subject_id"]) & set(splits["folds"][0]["subjects"]["test"])
    pipeline = joblib.load(smoke / "models/erp_fold-1.joblib")
    columns = features.filter(like="erp_").columns
    train = features[features["subject_id"].isin(splits["folds"][0]["subjects"]["train"])]
    np.testing.assert_allclose(pipeline.named_steps["scaler"].mean_, train[columns].mean(axis=0))
    full = train_baseline(
        model_config, dataset_dir, feature_dir, root, mode="full", run_name="synthetic-full"
    )
    summary = verify_artifact(full, "logistic_baseline")
    assert summary["fitted_models"] == 15 and summary["test_predictions"] == 240
    predictions = pd.read_csv(full / "test_predictions.csv")
    for _, group in predictions.groupby("experiment"):
        assert not group["trial_id"].duplicated().any()
        assert set(group["trial_id"]) == set(trials["trial_id"])
    assert len(pd.read_csv(full / "test_subject_metrics.csv")) == 30
    assert (full / "figures/subject_balanced_accuracy.png").is_file()
    with pytest.raises(FileExistsError):
        build_manifest(dataset_config, root)


def test_pilot_features_cannot_train_formal_model(synthetic_project):
    root, dataset_config, feature_config, model_config = synthetic_project
    dataset_dir = build_manifest(dataset_config, root)
    feature_dir = extract_features(feature_config, dataset_dir, root, stage="pilot")
    with pytest.raises(ValueError, match="pilot"):
        train_baseline(model_config, dataset_dir, feature_dir, root)


def test_tampered_artifact_refused(synthetic_project):
    root, dataset_config, _, _ = synthetic_project
    directory = build_manifest(dataset_config, root)
    path = directory / "trial_manifest.csv"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="改动"):
        load_classification_dataset(directory)


def test_changed_source_refused_before_extraction(synthetic_project):
    root, dataset_config, feature_config, _ = synthetic_project
    directory = build_manifest(dataset_config, root)
    _, manifest, _ = load_classification_dataset(directory)
    source = root / manifest["sources"][0]["source_file"]
    with source.open("ab") as stream:
        stream.write(b"synthetic corruption")
    with pytest.raises(ValueError, match="FIF已改变"):
        extract_features(feature_config, directory, root)


def test_failed_run_keeps_diagnostic_and_cannot_be_reused(tmp_path):
    with pytest.raises(RuntimeError):
        with new_artifact_run(tmp_path, tmp_path / "output", "synthetic"):
            raise RuntimeError("test failure")
    summary = json.loads((tmp_path / "output/run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    with pytest.raises(ValueError, match="未完成"):
        verify_artifact(tmp_path / "output", "synthetic")


def test_output_cannot_escape_project(tmp_path):
    with pytest.raises(ValueError, match="项目内"):
        with new_artifact_run(tmp_path, tmp_path.parent / "outside", "synthetic"):
            pass
