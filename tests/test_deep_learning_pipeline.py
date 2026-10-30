"""合成FIF及小型CPU网络验证DL契约；不启动真实GPU训练，不产生真实项目成绩。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from test_classification_pipeline import synthetic_project as synthetic_project
from torch import nn

from erp_core_p3b import deep_learning_data as export_module
from erp_core_p3b import deep_learning_training as training_module
from erp_core_p3b.cbramod_audit import sha256_file
from erp_core_p3b.classification_data import build_manifest, load_classification_dataset
from erp_core_p3b.classification_io import verify_artifact, write_json
from erp_core_p3b.deep_learning_data import export_dataset, load_dl_dataset, verify_prior_audit
from erp_core_p3b.deep_learning_models import CBraClassifier, EEGNet
from erp_core_p3b.deep_learning_training import SignalDataset, class_weights, train_deep_learning

PROJECT = Path(__file__).resolve().parents[1]


def _config(name):
    return yaml.safe_load((PROJECT / "configs" / name).read_text(encoding="utf-8"))


@pytest.fixture
def dl_project(synthetic_project, monkeypatch):
    root, dataset_config, _, _ = synthetic_project
    dataset_dir = build_manifest(dataset_config, root)
    _, manifest, _ = load_classification_dataset(dataset_dir)
    audit_path = root / "configs/cbramod_audit.yaml"
    audit = yaml.safe_load(audit_path.read_text(encoding="utf-8"))
    audit["dataset"]["pilot_subjects"] = ["sub-001", "sub-002", "sub-003"]
    audit_path.write_text(yaml.safe_dump(audit), encoding="utf-8")
    provenance = {
        "source_commit": "synthetic-code",
        "checkpoint_sha256": "synthetic-weights",
        "source_files_sha256": {"toy.py": "synthetic-source"},
    }
    # 假GPU摘要仅作为测试fixture，真实运行必须验证用户已有的真实pilot-gpu产物。
    audit_dir = root / "results/cbramod_audit/pilot-gpu"
    audit_dir.mkdir(parents=True)
    write_json(
        audit_dir / "run_summary.json",
        {
            "technical_status": "passed",
            "gpu_probe_complete": True,
            "config_sha256": sha256_file(audit_path),
            "visual_review_status": "pending",
            "subjects": [{"subject": s} for s in audit["dataset"]["pilot_subjects"]],
            "model_resources": provenance,
        },
    )
    write_json(audit_dir / "input_inventory.json", manifest["sources"])
    monkeypatch.setattr(export_module, "model_provenance", lambda *_: provenance)
    monkeypatch.setattr(training_module, "model_provenance", lambda *_: provenance)
    export_config = root / "configs/deep_learning_dataset.yaml"
    export_config.write_text(
        yaml.safe_dump(_config("deep_learning_dataset.yaml")), encoding="utf-8"
    )
    train = _config("deep_learning_training.yaml")
    train["eegnet"].update(F1=2, D=1, F2=2, temporal_kernel=16, separable_kernel=8, batch_size=16)
    train["training"].update(
        max_epochs=2, smoke_epochs=2, patience=1, smoke_patience=1, cpu_threads=1
    )
    train["statistics"]["bootstrap_iterations"] = 20
    train_path = root / "configs/deep_learning_training.yaml"
    train_path.write_text(yaml.safe_dump(train), encoding="utf-8")
    return root, dataset_dir, export_config, train_path


def _full_cache(project):
    root, dataset_dir, export_config, _ = project
    return export_dataset(export_config, dataset_dir, root, accept_short_window=True)


def test_export_full_shapes_units_alignment_and_identity(dl_project):
    cache = _full_cache(dl_project)
    trials, manifest, splits, arrays, summary = load_dl_dataset(cache)
    assert arrays["native"].shape == (80, 30, 257)
    assert arrays["cbramod"].shape == (80, 30, 1, 200)
    assert arrays["labels"].sum() == 20
    assert len(trials) == 80 and manifest["dataset_hash"] == splits["dataset_hash"]
    assert summary["eligible_for_training"]
    assert summary["prior_audit"]["prior_visual_review_status"] == "pending"
    protocol = manifest["input_protocol"]
    assert protocol["native_times_s"][51] == 0 and protocol["cbramod_times_s"][39] == 0
    original = arrays["native"][0].copy()
    sample, label = SignalDataset(arrays, np.array([0]), "eegnet", 1e-4)[0]
    assert sample.shape == (1, 30, 257) and label.item() == 1
    np.testing.assert_allclose(sample.numpy()[0], original / 1e-4)
    sample.zero_()
    np.testing.assert_array_equal(arrays["native"][0], original)


def test_export_requires_explicit_acceptance_and_pilot_is_not_trainable(dl_project):
    root, dataset_dir, config, _ = dl_project
    with pytest.raises(ValueError, match="accept-short"):
        export_dataset(config, dataset_dir, root)
    pilot = export_dataset(config, dataset_dir, root, stage="pilot")
    with pytest.raises(ValueError, match="pilot"):
        load_dl_dataset(pilot)


def test_prior_audit_failure_and_config_changes_rejected(dl_project):
    root, dataset_dir, config, _ = dl_project
    _, manifest, _ = load_classification_dataset(dataset_dir)
    options = yaml.safe_load(config.read_text(encoding="utf-8"))
    report_path = root / options["audit_directory"] / "run_summary.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["gpu_probe_complete"] = False
    write_json(report_path, report)
    with pytest.raises(ValueError, match="GPU"):
        verify_prior_audit(root, options, manifest)
    report["gpu_probe_complete"] = True
    report["config_sha256"] = "changed"
    write_json(report_path, report)
    with pytest.raises(ValueError, match="配置"):
        verify_prior_audit(root, options, manifest)


def test_cache_corruption_and_repeat_export_rejected(dl_project):
    cache = _full_cache(dl_project)
    with pytest.raises(FileExistsError):
        _full_cache(dl_project)
    (cache / "labels.npy").write_bytes(b"corrupt synthetic file")
    with pytest.raises(ValueError, match="labels.npy"):
        load_dl_dataset(cache)


def test_eegnet_forward_backward_and_max_norm():
    model = EEGNet(30, 257, _config("deep_learning_training.yaml")["eegnet"])
    logits = model(torch.randn(4, 1, 30, 257))
    assert logits.shape == (4, 2)
    logits.sum().backward()
    assert model.spatial.weight.grad is not None
    model.constrain_weights()
    assert torch.all(torch.linalg.vector_norm(model.spatial.weight, dim=(1, 2, 3)) <= 1.00001)
    assert torch.all(model.classifier.weight.norm(dim=1) <= 0.25001)
    with pytest.raises(ValueError):
        model(torch.randn(4, 30, 257))


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(200, 200)
        self.dropout = nn.Dropout(0.5)
        self.mask_encoding = nn.Parameter(torch.zeros(200), requires_grad=False)

    def forward(self, signal):
        return self.dropout(self.projection(signal))


@pytest.mark.parametrize("frozen", [True, False])
def test_cbramod_freezing_dropout_and_gradients(frozen):
    model = CBraClassifier(ToyEncoder(), 30, frozen=frozen).train()
    assert model.encoder.training is not frozen
    inputs = torch.randn(2, 30, 1, 200)
    model(inputs).sum().backward()
    assert (model.encoder.projection.weight.grad is None) == frozen
    assert model.head[1].weight.grad is not None
    assert not model.encoder.mask_encoding.requires_grad
    if frozen:
        torch.testing.assert_close(model(inputs), model(inputs))


def test_class_weights_use_only_passed_labels():
    torch.testing.assert_close(class_weights(np.array([1, 0, 0, 0])), torch.tensor([2 / 3, 2.0]))
    with pytest.raises(ValueError):
        class_weights(np.zeros(3, dtype=int))


def test_smoke_no_test_predictions_repeatability_and_checkpoint(dl_project):
    root, _, _, config = dl_project
    cache = _full_cache(dl_project)
    outputs = [
        train_deep_learning(config, cache, root, models=["eegnet"], device="cpu", run_name=name)
        for name in ("smoke-a", "smoke-b")
    ]
    first = verify_artifact(outputs[0], "deep_learning_training")
    assert first["fit_count"] == 1 and not first["test_evaluated"]
    assert not (outputs[0] / "oof_predictions.csv").exists()
    a = pd.read_csv(outputs[0] / "validation_predictions.csv")
    b = pd.read_csv(outputs[1] / "validation_predictions.csv")
    pd.testing.assert_frame_equal(a, b)
    _, _, splits, _, _ = load_dl_dataset(cache)
    assert set(a["trial_id"]) == set(splits["folds"][0]["trial_ids"]["validation"])
    state = torch.load(next(outputs[0].glob("*/best_model.pt")), weights_only=True)
    assert state["epoch"] >= 1 and state["state_dict"]


def test_full_three_models_oof_coverage_with_toy_cbra(dl_project, monkeypatch):
    root, _, _, config = dl_project
    cache = _full_cache(dl_project)
    original_builder = training_module.build_model

    def toy_builder(name, channels, samples, options, provenance):
        if name == "eegnet":
            return original_builder(name, channels, samples, options, provenance)
        return CBraClassifier(ToyEncoder(), channels, frozen=name == "cbramod_linear")

    monkeypatch.setattr(training_module, "build_model", toy_builder)
    output = train_deep_learning(
        config, cache, root, mode="full", device="cpu", run_name="full-toy"
    )
    predictions = pd.read_csv(output / "oof_predictions.csv")
    assert len(predictions) == 80 * 3
    assert predictions.groupby("experiment")["trial_id"].nunique().eq(80).all()
    assert not predictions.duplicated(["experiment", "trial_id"]).any()
    assert len(pd.read_csv(output / "subject_metrics.csv")) == 30
    assert verify_artifact(output, "deep_learning_training")["fit_count"] == 15


def test_review_flags_gate_training_and_make_figures(dl_project, monkeypatch):
    root, _, _, config = dl_project
    original_review = export_module.verify_prior_audit

    def low_threshold(*args):
        audit, prior = original_review(*args)
        audit["adaptation"]["review_peak_to_peak_uv"] = 0.01
        return audit, prior

    monkeypatch.setattr(export_module, "verify_prior_audit", low_threshold)
    cache = _full_cache(dl_project)
    assert list((cache / "review_figures").glob("*.png"))
    assert not verify_artifact(cache, "deep_learning_dataset")["eligible_for_training"]
    with pytest.raises(ValueError, match="adaptation-review-note"):
        train_deep_learning(config, cache, root, models=["eegnet"], device="cpu")
    output = train_deep_learning(
        config,
        cache,
        root,
        models=["eegnet"],
        device="cpu",
        adaptation_review_note="仅为合成测试：已确认该测试提示，保留全部trial",
        run_name="reviewed",
    )
    assert verify_artifact(output, "deep_learning_training")["adaptation_review_trials"] == 80


def test_no_silent_cpu_fallback(dl_project, monkeypatch):
    root, _, _, config = dl_project
    cache = _full_cache(dl_project)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA不可用"):
        train_deep_learning(config, cache, root, models=["eegnet"], device="cuda")
