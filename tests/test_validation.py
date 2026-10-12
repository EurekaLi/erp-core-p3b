import csv
import json
from pathlib import Path

from erp_core_p3b.validation import classify_p3_event, validate_dataset, write_validation_reports


def _write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _make_minimal_dataset(root: Path, *, with_session: bool = True) -> None:
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "ERP CORE P3"}), encoding="utf-8"
    )
    (root / "task-P3_events.json").write_text("{}", encoding="utf-8")
    _write_tsv(root / "participants.tsv", ["participant_id"], [{"participant_id": "sub-001"}])

    eeg_dir = root / "sub-001" / "eeg"
    if with_session:
        eeg_dir = root / "sub-001" / "ses-P3" / "eeg"
    eeg_dir.mkdir(parents=True)
    prefix = eeg_dir / "sub-001_ses-P3_task-P3"
    prefix.with_name(prefix.name + "_eeg.set").write_bytes(b"set")
    prefix.with_name(prefix.name + "_eeg.fdt").write_bytes(b"fdt")
    prefix.with_name(prefix.name + "_eeg.json").write_text(
        json.dumps(
            {
                "SamplingFrequency": 1024,
                "EEGReference": "CMS",
                "PowerLineFrequency": 60,
                "RecordingDuration": 1,
            }
        ),
        encoding="utf-8",
    )
    channel_rows = [
        {"name": f"EEG{index}", "type": "EEG", "units": "microV"}
        for index in range(30)
    ] + [
        {"name": f"EOG{index}", "type": "EOG", "units": "microV"}
        for index in range(3)
    ]
    _write_tsv(
        prefix.with_name(prefix.name + "_channels.tsv"),
        ["name", "type", "units"],
        channel_rows,
    )
    _write_tsv(
        prefix.with_name(prefix.name + "_events.tsv"),
        ["onset", "sample", "trial_type", "value"],
        [
            {"onset": 0, "sample": 0, "trial_type": "stimulus", "value": 11},
            {"onset": 0.5, "sample": 512, "trial_type": "response", "value": 201},
            {"onset": 1, "sample": 1024, "trial_type": "stimulus", "value": 12},
            {"onset": 2, "sample": 2048, "trial_type": "stimulus", "value": 13},
            {"onset": 3, "sample": 3072, "trial_type": "stimulus", "value": 14},
            {"onset": 4, "sample": 4096, "trial_type": "stimulus", "value": 15},
        ],
    )


def test_classify_p3_event_codes() -> None:
    assert classify_p3_event(11) == "target"
    assert classify_p3_event("55") == "target"
    assert classify_p3_event(12) == "standard"
    assert classify_p3_event(201) == "correct_response"
    assert classify_p3_event(202) == "error_response"
    assert classify_p3_event("bad") == "unknown"


def test_validate_complete_minimal_dataset(tmp_path: Path) -> None:
    _make_minimal_dataset(tmp_path)

    result = validate_dataset(tmp_path, expected_subjects=1)

    assert result.summary["subject_directories_found"] == 1
    assert result.summary["task_recordings_found"] == 1
    assert result.event_counts[0]["target"] == 1
    assert result.event_counts[0]["standard"] == 4
    assert result.summary["status"] == "PASS_WITH_WARNINGS"
    assert {issue.code for issue in result.issues} == {"ELECTRODE_COORDINATES_MISSING"}


def test_reports_are_written(tmp_path: Path) -> None:
    bids_root = tmp_path / "bids"
    bids_root.mkdir()
    _make_minimal_dataset(bids_root)
    result = validate_dataset(bids_root, expected_subjects=1)

    paths = write_validation_reports(result, tmp_path / "reports")

    assert len(paths) == 4
    assert all(path.is_file() for path in paths)
    summary = json.loads(paths[0].read_text(encoding="utf-8"))
    assert summary["event_totals"]["target"] == 1


def test_validate_sessionless_bids_layout(tmp_path: Path) -> None:
    _make_minimal_dataset(tmp_path, with_session=False)

    result = validate_dataset(tmp_path, expected_subjects=1)

    assert result.summary["task_recordings_found"] == 1
    assert result.event_counts[0]["stimulus_total"] == 5
