"""构建共享trial身份及受试者独立划分；划分不读取特征或模型成绩。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from erp_core_p3b.cbramod_audit import sha256_file, validate_epoch_header
from erp_core_p3b.classification_io import (
    canonical_hash,
    new_artifact_run,
    resolve_path,
    verify_artifact,
    write_json,
)


def validate_splits(trials: pd.DataFrame, splits: dict[str, Any]) -> None:
    """验证每折三集合覆盖全体，受试者不交叉，且每条trial恰好被外层测试一次。"""
    if (
        trials["trial_id"].duplicated().any()
        or trials[["trial_id", "subject_id", "label"]].isna().any().any()
    ):
        raise ValueError("trial身份重复/缺失")
    if set(trials["label"]) != {0, 1}:
        raise ValueError("标签必须包含且仅包含Standard=0、Target=1")
    all_subjects, all_ids = set(trials["subject_id"]), set(trials["trial_id"])
    coverage: Counter = Counter()
    fold_ids = []
    for fold in splits["folds"]:
        fold_ids.append(fold["fold"])
        used_subjects, used_ids = set(), set()
        for role in ("train", "validation", "test"):
            subjects = fold["subjects"][role]
            ids = fold["trial_ids"][role]
            if not subjects or len(subjects) != len(set(subjects)) or len(ids) != len(set(ids)):
                raise ValueError("划分集合为空或有重复身份")
            if used_subjects & set(subjects) or used_ids & set(ids):
                raise ValueError("同一受试者/trial跨划分集合")
            actual = trials[trials["subject_id"].isin(subjects)]
            if set(actual["trial_id"]) != set(ids) or set(actual["label"]) != {0, 1}:
                raise ValueError("划分trial清单与受试者名单/两类覆盖不一致")
            used_subjects.update(subjects)
            used_ids.update(ids)
        if used_subjects != all_subjects or used_ids != all_ids:
            raise ValueError("某折没有覆盖全部样本")
        coverage.update(fold["trial_ids"]["test"])
    if len(fold_ids) < 2 or len(set(fold_ids)) != len(fold_ids):
        raise ValueError("外层折数不足或编号重复")
    if fold_ids != list(range(1, int(splits["protocol"]["n_splits"]) + 1)):
        raise ValueError("外层折数/编号与协议不一致")
    if set(coverage) != all_ids or any(count != 1 for count in coverage.values()):
        raise ValueError("外层测试未恰好覆盖每条trial一次")


def make_subject_splits(
    trials: pd.DataFrame, config: dict[str, Any], dataset_hash: str
) -> dict[str, Any]:
    """分层仅近似平衡标签；验证集按组抽取，不按测试效果反复挑种子。"""
    if config["outer_strategy"] != "stratified_group_k_fold" or not config["shuffle"]:
        raise ValueError("本版本固定外层分组分层且shuffle=True")
    if config["validation_strategy"] != "group_shuffle_split":
        raise ValueError("本版本固定内层受试者组随机留出")
    fraction = float(config["validation_subject_fraction"])
    if not 0 < fraction < 1:
        raise ValueError("验证受试者比例须处于(0,1)")
    seed = int(config["random_state"])
    splitter = StratifiedGroupKFold(
        n_splits=int(config["n_splits"]), shuffle=True, random_state=seed
    )
    labels, groups = trials["label"].to_numpy(), trials["subject_id"].to_numpy()
    result = {"schema_version": 1, "dataset_hash": dataset_hash, "protocol": config, "folds": []}
    for number, (outer_train, test) in enumerate(
        splitter.split(np.zeros((len(trials), 1)), labels, groups), 1
    ):
        inner = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed + number)
        inner_train, validation = next(
            inner.split(outer_train, labels[outer_train], groups[outer_train])
        )
        indices = {
            "train": outer_train[inner_train],
            "validation": outer_train[validation],
            "test": test,
        }
        result["folds"].append(
            {
                "fold": number,
                "subjects": {role: sorted(set(groups[index])) for role, index in indices.items()},
                "trial_ids": {
                    role: trials.iloc[index]["trial_id"].tolist() for role, index in indices.items()
                },
                "validation_seed": seed + number,
            }
        )
    validate_splits(trials, result)
    return result


def load_classification_dataset(
    directory: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    summary = verify_artifact(directory, "classification_dataset")
    trials = pd.read_csv(directory / "trial_manifest.csv")
    manifest = json.loads((directory / "dataset_manifest.json").read_text(encoding="utf-8"))
    splits = json.loads((directory / "subject_splits.json").read_text(encoding="utf-8"))
    if (
        manifest["dataset_hash"] != splits["dataset_hash"]
        or manifest["dataset_hash"] != summary["dataset_hash"]
    ):
        raise ValueError("清单与划分的数据版本不一致")
    if canonical_hash(splits) != summary["split_hash"]:
        raise ValueError("固定划分内容hash不一致")
    validate_splits(trials, splits)
    if len(trials) != manifest["trial_count"]:
        raise ValueError("trial清单长度与数据版本不一致")
    return trials, manifest, splits


def build_manifest(config_path: Path, root: Path, output: Path | None = None) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    header_path = resolve_path(root, config["header_config"])
    header = yaml.safe_load(header_path.read_text(encoding="utf-8"))["dataset"]
    output = resolve_path(root, output or config["outputs"]["directory"])
    with new_artifact_run(root, output, "classification_dataset") as summary:
        write_json(
            output / "config_snapshot.json", {"classification": config, "header_protocol": header}
        )
        phase5_path = resolve_path(root, header["phase5_manifest"])
        phase5 = json.loads(phase5_path.read_text(encoding="utf-8"))
        expected = {item["subject"]: item for item in phase5["input_files"]}
        files = sorted(resolve_path(root, header["epochs_root"]).glob("sub-*_task-P3_epo.fif"))
        subjects = [file.name.split("_", 1)[0] for file in files]
        if len(subjects) != header["expected_subject_count"] or set(subjects) != set(
            phase5["subjects"]
        ):
            raise ValueError("实际受试者名单与Phase 5不一致")
        if set(subjects) & set(header["excluded_subjects"]):
            raise ValueError("排除受试者进入主分析")
        rows, sources = [], []
        channels, times = None, None
        for subject, path in zip(subjects, files, strict=True):
            print(f"Indexing {subject} ...", flush=True)
            epochs = mne.read_epochs(path, preload=False, verbose="ERROR")
            picks, identities = validate_epoch_header(epochs, subject, header)
            names = [epochs.ch_names[index] for index in picks]
            if channels is None:
                channels, times = names, epochs.times.copy()
            if names != channels or not np.array_equal(times, epochs.times):
                raise ValueError("受试者之间通道顺序或时间网格不同")
            counts = identities["label"].value_counts()
            if (
                counts[1] != expected[subject]["target_trials"]
                or counts[0] != expected[subject]["standard_trials"]
            ):
                raise ValueError(f"{subject}: Phase 5计数变化")
            relative = path.resolve().relative_to(root.resolve()).as_posix()
            identities["source_file"] = relative
            rows.append(identities)
            sources.append(
                {
                    "subject": subject,
                    "source_file": relative,
                    "sha256": sha256_file(path),
                    "trials": len(identities),
                    "target": int(counts[1]),
                    "standard": int(counts[0]),
                }
            )
        trials = pd.concat(rows, ignore_index=True)
        if (
            len(trials) != header["expected_trial_count"]
            or int(trials["label"].sum()) != header["expected_target_count"]
        ):
            raise ValueError("全体trial/Target数量不符合冻结协议")
        if int((trials["label"] == 0).sum()) != header["expected_standard_count"]:
            raise ValueError("全体Standard数量不符合冻结协议")
        dataset_hash = canonical_hash(
            {
                "sources": sources,
                "trial_ids": trials["trial_id"].tolist(),
                "labels": trials["label"].tolist(),
                "channels": channels,
                "times_s": times.tolist(),
                "header_protocol": header,
            }
        )
        manifest = {
            "schema_version": 1,
            "dataset_hash": dataset_hash,
            "subject_count": len(subjects),
            "trial_count": len(trials),
            "sources": sources,
            "eeg_channel_names": channels,
            "times_s": times.tolist(),
            "header_protocol": header,
            "labels": {"Standard": 0, "Target": 1},
        }
        trials.to_csv(output / "trial_manifest.csv", index=False)
        write_json(output / "dataset_manifest.json", manifest)
        splits = make_subject_splits(trials, config["split"], dataset_hash)
        write_json(output / "subject_splits.json", splits)
        audit_rows = []
        for fold in splits["folds"]:
            for role, ids in fold["trial_ids"].items():
                group = trials[trials["trial_id"].isin(ids)]
                audit_rows.append(
                    {
                        "fold": fold["fold"],
                        "role": role,
                        "subjects": group["subject_id"].nunique(),
                        "trials": len(group),
                        "target": int(group["label"].sum()),
                        "standard": int((group["label"] == 0).sum()),
                        "target_fraction": float(group["label"].mean()),
                    }
                )
        pd.DataFrame(audit_rows).to_csv(output / "split_audit.csv", index=False)
        overview = (
            trials.groupby("subject_id")
            .agg(trials=("label", "size"), target=("label", "sum"))
            .reset_index()
        )
        overview["standard"] = overview["trials"] - overview["target"]
        evidence = []
        for name, relative in config.get("evidence", {}).items():
            path = resolve_path(root, relative)
            if not path.is_file():
                raise FileNotFoundError(f"缺少已有证据表: {path}")
            frame = pd.read_csv(path).rename(columns={"subject": "subject_id"})
            if frame["subject_id"].duplicated().any() or not set(subjects).issubset(
                frame["subject_id"]
            ):
                raise ValueError(f"{name}: 证据表身份重复/缺失")
            frame = frame.rename(
                columns={column: f"{name}__{column}" for column in frame if column != "subject_id"}
            )
            overview = overview.merge(frame, on="subject_id", how="left", validate="one_to_one")
            evidence.append({"name": name, "path": str(path), "sha256": sha256_file(path)})
        overview.to_csv(output / "subject_evidence_summary.csv", index=False)
        summary.update(
            dataset_hash=dataset_hash,
            split_hash=canonical_hash(splits),
            trial_count=len(trials),
            subject_count=len(subjects),
            config_sha256=sha256_file(config_path),
            phase5_sha256=sha256_file(phase5_path),
            evidence=evidence,
            n_splits=len(splits["folds"]),
            test_coverage="each_trial_once",
        )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path, default=Path("configs/classification_dataset.yaml"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        output = build_manifest(
            resolve_path(args.project_root, args.config), args.project_root, args.output_dir
        )
    except Exception as error:
        print(f"Dataset build failed: {error}", file=sys.stderr)
        return 1
    print(f"Classification dataset: {output}")
    return 0
