"""合成信号、临时FIF和CPU小网络测试；不运行真实数据或GPU训练。"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import ModuleType

import mne
import numpy as np
import pandas as pd
import pytest
import yaml

from erp_core_p3b.cbramod_audit import (
    ModelProbe,
    adapt_signal,
    main,
    model_provenance,
    model_scaled_input,
    run_audit,
    sha256_file,
    validate_epoch_header,
)

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return yaml.safe_load((PROJECT / "configs/cbramod_audit.yaml").read_text(encoding="utf-8"))


def _epochs():
    info = mne.create_info(["Pz", "Cz", "EOG"], 256, ["eeg", "eeg", "eog"])
    # 合成头信息模拟已有滤波结果；测试不重复运行真实滤波/ICA。
    with info._unlock():
        info["highpass"] = 0.1
        info["lowpass"] = 30.0
    rng = np.random.default_rng(42)
    signal = rng.normal(scale=10e-6, size=(4, 257))
    data = np.stack([signal, -signal, np.zeros_like(signal)], axis=1)
    events = np.array([[1000 + 500 * index, 0, 1 if index % 2 == 0 else 2] for index in range(4)])
    frame = pd.DataFrame(
        {
            "subject": ["sub-001"] * 4,
            "trial_index": [1, 3, 5, 7],
            "condition_label": ["Target", "Standard", "Target", "Standard"],
            "event_sample": events[:, 0],
        }
    )
    return mne.EpochsArray(
        data,
        info,
        events=events,
        event_id={"Target": 1, "Standard": 2},
        metadata=frame,
        tmin=-0.2,
        baseline=(-0.2, 0),
        verbose="ERROR",
    )


def _small_dataset_config(config):
    result = copy.deepcopy(config)
    result["dataset"].update(expected_eeg_channels=2, expected_eog_channels=1)
    return result


def test_polyphase_grid_keeps_stimulus_zero_and_identity(config):
    times = np.arange(-51, 206) / 256
    signal = np.zeros((2, 3, 257))
    signal[:, :, 51] = 10e-6
    result, target_times, alignment = adapt_signal(signal, times, 256, config["adaptation"])
    assert result.shape == (2, 3, 200)
    assert target_times[39] == 0
    assert target_times[0] == -0.195
    assert target_times[-1] == 0.8
    assert result[0, 0].argmax() == 39
    assert alignment["left_reflection_samples"] == 13
    assert alignment["up"] == 25 and alignment["down"] == 32
    np.testing.assert_array_equal(result[0], result[1])


def test_resampling_constant_amplitude_and_model_units(config):
    signal = np.full((2, 3, 257), 10e-6)
    adapted, _, _ = adapt_signal(signal, np.arange(-51, 206) / 256, 256, config["adaptation"])
    # 多相滤波在常量信号上有很小的相位纹波，不假设逐位恒等。
    np.testing.assert_allclose(adapted, 10e-6, rtol=0.001)
    scaled, error = model_scaled_input(adapted, config["adaptation"])
    assert scaled.shape == (2, 3, 1, 200)
    assert scaled.dtype == np.float32
    np.testing.assert_allclose(scaled, 0.1, rtol=0.001)
    assert error < 0.001


def test_resampling_preserves_low_frequency_peak(config):
    times = np.arange(-51, 206) / 256
    signal = (20e-6 * np.exp(-(((times - 0.4) / 0.08) ** 2)))[None, None, :]
    adapted, target_times, _ = adapt_signal(signal, times, 256, config["adaptation"])
    assert abs(target_times[adapted[0, 0].argmax()] - 0.4) <= 0.005
    assert abs(adapted.max() / signal.max() - 1) < 0.01


@pytest.mark.parametrize("scale", [1.0, 1e-6, 0.0, np.nan])
def test_wrong_model_scale_is_rejected(config, scale):
    parameters = copy.deepcopy(config["adaptation"])
    parameters["model_scale_v"] = scale
    with pytest.raises(ValueError, match="100µV"):
        model_scaled_input(np.ones((1, 2, 200)) * 10e-6, parameters)


@pytest.mark.parametrize("case", ["nonfinite", "shifted_zero", "outside", "wrong_patch"])
def test_invalid_adaptation_is_rejected(config, case):
    parameters = copy.deepcopy(config["adaptation"])
    signal = np.zeros((1, 2, 257))
    times = np.arange(-51, 206) / 256
    if case == "nonfinite":
        signal[0, 0, 20] = np.nan
    elif case == "shifted_zero":
        times = times + 0.001
    elif case == "outside":
        parameters["start_time_s"] = -0.2
    else:
        parameters["points_per_patch"] = 256
    with pytest.raises(ValueError):
        adapt_signal(signal, times, 256, parameters)


def test_header_uses_event_labels_and_retained_trial_ids(config):
    epochs = _epochs()
    small = _small_dataset_config(config)
    picks, identities = validate_epoch_header(epochs, "sub-001", small["dataset"])
    assert picks == [0, 1]
    assert identities["label"].tolist() == [1, 0, 1, 0]
    assert identities["trial_id"].tolist() == [
        "sub-001:trial-0001",
        "sub-001:trial-0003",
        "sub-001:trial-0005",
        "sub-001:trial-0007",
    ]


@pytest.mark.parametrize("case", ["label", "subject", "event", "duplicate", "unit", "bad"])
def test_header_refuses_silent_mismatch(config, case):
    epochs = _epochs()
    small = _small_dataset_config(config)
    if case == "label":
        epochs.metadata.loc[0, "condition_label"] = "Standard"
    elif case == "subject":
        epochs.metadata.loc[0, "subject"] = "sub-006"
    elif case == "event":
        epochs.metadata.loc[0, "event_sample"] += 1
    elif case == "duplicate":
        epochs.metadata.loc[1, "trial_index"] = 1
    elif case == "unit":
        epochs.info["chs"][0]["unit"] = -1
    else:
        epochs.info["bads"] = ["Pz"]
    with pytest.raises(ValueError):
        validate_epoch_header(epochs, "sub-001", small["dataset"])


def test_signal_only_run_outputs_report_without_torch_or_model(config, tmp_path, monkeypatch):
    small = _small_dataset_config(config)
    small["dataset"].update(
        expected_subject_count=1,
        expected_trial_count=4,
        expected_target_count=2,
        expected_standard_count=2,
        pilot_subjects=["sub-001"],
    )
    input_root = tmp_path / small["dataset"]["epochs_root"]
    input_root.mkdir(parents=True)
    input_file = input_root / "sub-001_task-P3_epo.fif"
    _epochs().save(input_file, fmt="double", verbose="ERROR")
    before = sha256_file(input_file)
    manifest = tmp_path / small["dataset"]["phase5_manifest"]
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "subjects": ["sub-001"],
                "input_files": [
                    {"subject": "sub-001", "target_trials": 2, "standard_trials": 2},
                ],
            }
        ),
        encoding="utf-8",
    )
    configuration = tmp_path / "config.yaml"
    configuration.write_text(yaml.safe_dump(small), encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("signal-only不得构造模型")

    monkeypatch.setattr("erp_core_p3b.cbramod_audit.ModelProbe", forbidden)
    output = run_audit(
        configuration,
        project_root=tmp_path,
        signal_only=True,
        run_name="synthetic",
        save_inputs=True,
    )
    summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["technical_status"] == "passed"
    assert not summary["eligible_for_formal_training"]
    assert not summary["gpu_probe_complete"]
    assert summary["subjects"][0]["trials"] == 4
    assert summary["subjects"][0]["model_shape"] == [4, 2, 1, 200]
    assert (output / "audit_report.md").is_file()
    assert len(list((output / "figures").glob("*.png"))) == 3
    with np.load(output / "sub-001_audit_inputs.npz", allow_pickle=False) as cache:
        assert cache["labels"].tolist() == [1, 0, 1, 0]
    assert sha256_file(input_file) == before
    with pytest.raises(FileExistsError):
        run_audit(configuration, project_root=tmp_path, signal_only=True, run_name="synthetic")


def test_failed_inventory_keeps_diagnostic_json(config, tmp_path):
    configuration = tmp_path / "config.yaml"
    configuration.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        run_audit(configuration, project_root=tmp_path, signal_only=True, run_name="failure")
    summary = json.loads((tmp_path / "results/cbramod_audit/failure/run_summary.json").read_text())
    assert summary["technical_status"] == "failed"
    assert "error" in summary


def test_run_name_cannot_escape_output_folder(config, tmp_path):
    configuration = tmp_path / "config.yaml"
    configuration.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="run-name"):
        run_audit(configuration, project_root=tmp_path, signal_only=True, run_name="../outside")


def test_changed_weight_hash_is_rejected(config, tmp_path, monkeypatch):
    source = tmp_path / config["model"]["source_root"] / "models"
    source.mkdir(parents=True)
    (source / "cbramod.py").write_text("# synthetic fixture\n", encoding="utf-8")
    weight = tmp_path / config["model"]["checkpoint"]
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"not the downloaded checkpoint")
    replies = iter([config["model"]["source_commit"], ""])
    monkeypatch.setattr("subprocess.check_output", lambda *args, **kwargs: next(replies))
    with pytest.raises(ValueError, match="SHA-256"):
        model_provenance(tmp_path, config["model"])


def test_cpu_toy_model_checks_frozen_and_full_gradients(config, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    source = tmp_path / "toy_source"
    source.mkdir()
    parent = ModuleType("models")
    parent.__path__ = [str(source / "models")]
    module = ModuleType("models.cbramod")
    module.__file__ = str(source / "models/cbramod.py")

    class ToyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(200, 200)
            self.mask_encoding = torch.nn.Parameter(torch.zeros(200), requires_grad=False)
            self.proj_out = torch.nn.Linear(200, 200)

        def forward(self, x):
            return self.proj_out(self.encoder(x))

    module.CBraMod = ToyBackbone
    monkeypatch.setitem(sys.modules, "models", parent)
    monkeypatch.setitem(sys.modules, "models.cbramod", module)
    checkpoint = tmp_path / "toy.pth"
    torch.save(ToyBackbone().state_dict(), checkpoint)
    probe = ModelProbe(
        {"source_root": str(source), "checkpoint": str(checkpoint)}, config["model"], "cpu"
    )
    inputs = np.random.default_rng(10).normal(size=(4, 2, 1, 200)).astype(np.float32)
    result = probe.run(inputs, np.array([1, 0, 1, 0]))
    assert result["frozen"]["finite_gradients"]
    assert result["full_gradient"]["head_updated"]
    assert probe.environment["strict_checkpoint_loading"]


def test_cuda_does_not_silently_fallback(config, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="禁止"):
        ModelProbe({}, config["model"], "cuda")


def test_cli_failure_has_nonzero_exit(tmp_path):
    assert main(["--project-root", str(tmp_path), "--signal-only"]) == 1
