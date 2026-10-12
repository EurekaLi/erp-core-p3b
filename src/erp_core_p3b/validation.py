"""ERP CORE P3 数据集的只读结构与元数据验证。

本模块故意不执行滤波、插值或 ICA。数据验证应该先回答“输入是什么、是否完整、
事件怎样编码”，否则后续信号处理可能在错误数据上产生看似合理的结果。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Issue:
    """一条可写入机器可读报告的验证问题。"""

    level: str
    code: str
    subject: str
    file: str
    message: str
    explanation: str


@dataclass
class ValidationResult:
    """验证结果及用于后续审计的明细表。"""

    summary: dict[str, Any]
    recordings: list[dict[str, Any]]
    event_counts: list[dict[str, Any]]
    issues: list[Issue]


EVENT_LABELS = (
    "target",
    "standard",
    "correct_response",
    "error_response",
    "unknown",
)


def classify_p3_event(value: int | str) -> str:
    """把 ERP CORE P3 的数值事件码映射为便于分析的类别。

    刺激码的十位表示当前 block 的目标字母，个位表示本 trial 实际出现的字母。
    两位相同表示目标刺激，例如 22；两位不同表示标准刺激，例如 25。
    """

    try:
        code = int(value)
    except (TypeError, ValueError):
        return "unknown"

    if code == 201:
        return "correct_response"
    if code == 202:
        return "error_response"

    target_letter, presented_letter = divmod(code, 10)
    if 1 <= target_letter <= 5 and 1 <= presented_letter <= 5:
        return "target" if target_letter == presented_letter else "standard"
    return "unknown"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig") as file:
        return json.load(file)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _add_issue(
    issues: list[Issue],
    *,
    level: str,
    code: str,
    message: str,
    explanation: str,
    subject: str = "",
    file: str = "",
) -> None:
    issues.append(
        Issue(
            level=level,
            code=code,
            subject=subject,
            file=file,
            message=message,
            explanation=explanation,
        )
    )


def _coerce_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _discover_recordings(bids_root: Path, task: str) -> list[tuple[str, Path]]:
    recordings: list[tuple[str, Path]] = []
    for subject_dir in sorted(path for path in bids_root.glob("sub-*") if path.is_dir()):
        pattern = f"*_task-{task}_eeg.set"
        # ERP CORE 的不同发布版本分别采用有 session 和无 session 的合法 BIDS 布局。
        candidates = list(subject_dir.glob(f"eeg/{pattern}"))
        candidates.extend(subject_dir.glob(f"ses-*/eeg/{pattern}"))
        for set_path in sorted(set(candidates)):
            recordings.append((subject_dir.name, set_path))
    return recordings


def _validate_channels(
    rows: list[dict[str, str]],
    *,
    expected_eeg: int,
    expected_eog: int,
    subject: str,
    file: str,
    issues: list[Issue],
) -> tuple[int, int, str]:
    type_counts = Counter(row.get("type", "").strip().upper() for row in rows)
    eeg_count = type_counts["EEG"]
    eog_count = type_counts["EOG"]

    if eeg_count != expected_eeg:
        _add_issue(
            issues,
            level="ERROR",
            code="EEG_CHANNEL_COUNT",
            subject=subject,
            file=file,
            message=f"EEG 通道数为 {eeg_count}，期望 {expected_eeg}。",
            explanation="通道数量变化会影响重参考、空间分布图和跨受试者特征对齐。",
        )
    if eog_count != expected_eog:
        _add_issue(
            issues,
            level="ERROR",
            code="EOG_CHANNEL_COUNT",
            subject=subject,
            file=file,
            message=f"EOG 通道数为 {eog_count}，期望 {expected_eog}。",
            explanation="EOG 用于识别眨眼和眼动伪迹，缺失会影响后续 ICA 质量控制。",
        )

    signal_rows = [
        row for row in rows if row.get("type", "").strip().upper() in {"EEG", "EOG"}
    ]
    units = sorted({row.get("units", "").strip() for row in signal_rows})
    normalized_units = {unit.lower().replace("μ", "u").replace("µ", "u") for unit in units}
    if not normalized_units or not normalized_units <= {"uv", "microv"}:
        _add_issue(
            issues,
            level="WARNING",
            code="SIGNAL_UNIT",
            subject=subject,
            file=file,
            message=f"EEG/EOG 单位记录为 {units or ['缺失']}，预期 microV/uV。",
            explanation=(
                "单位错误会造成幅值相差数量级。MNE 读取后通常统一换算为伏特，"
                "因此原始元数据必须先确认。"
            ),
        )
    return eeg_count, eog_count, ";".join(units)


def _validate_events(
    rows: list[dict[str, str]],
    *,
    sampling_frequency: float | None,
    subject: str,
    file: str,
    issues: list[Issue],
) -> dict[str, Any]:
    counts = Counter({label: 0 for label in EVENT_LABELS})
    previous_onset: float | None = None
    onset_order_error = False
    sample_mismatch_count = 0

    for row in rows:
        counts[classify_p3_event(row.get("value", ""))] += 1
        onset = _coerce_float(row.get("onset"))
        sample = _coerce_float(row.get("sample"))
        if onset is not None and previous_onset is not None and onset < previous_onset:
            onset_order_error = True
        if onset is not None:
            previous_onset = onset

        # BIDS 的 onset 使用秒，sample 使用采样点；舍入可能产生约 1 个采样点误差。
        if (
            onset is not None
            and sample is not None
            and sampling_frequency is not None
            and abs(sample - onset * sampling_frequency) > 1.1
        ):
            sample_mismatch_count += 1

    if onset_order_error:
        _add_issue(
            issues,
            level="ERROR",
            code="EVENT_ONSET_ORDER",
            subject=subject,
            file=file,
            message="事件 onset 不是单调递增。",
            explanation="乱序事件会导致刺激与 EEG 时间点错误对应。",
        )
    if sample_mismatch_count:
        _add_issue(
            issues,
            level="WARNING",
            code="EVENT_SAMPLE_MISMATCH",
            subject=subject,
            file=file,
            message=f"有 {sample_mismatch_count} 个事件的 onset 与 sample 不一致。",
            explanation="onset（秒）乘以采样率应约等于 sample（采样点）。",
        )
    if counts["unknown"]:
        _add_issue(
            issues,
            level="WARNING",
            code="UNKNOWN_EVENT_CODE",
            subject=subject,
            file=file,
            message=f"发现 {counts['unknown']} 个未识别事件码。",
            explanation="未知事件不能直接归入 Target/Standard，必须先核对事件字典。",
        )

    stimuli = counts["target"] + counts["standard"]
    target_ratio = counts["target"] / stimuli if stimuli else None
    if not stimuli:
        _add_issue(
            issues,
            level="ERROR",
            code="NO_STIMULUS_EVENTS",
            subject=subject,
            file=file,
            message="没有识别到 P3 刺激事件。",
            explanation="没有刺激事件就无法切分 Target/Standard 的 ERP epoch。",
        )
    elif target_ratio is not None and not 0.10 <= target_ratio <= 0.30:
        _add_issue(
            issues,
            level="WARNING",
            code="TARGET_RATIO",
            subject=subject,
            file=file,
            message=f"Target 占刺激 trial 的 {target_ratio:.1%}，不在 10%–30% 的检查范围内。",
            explanation=(
                "P3 oddball 通常要求目标刺激明显少于标准刺激；"
                "异常比例可能表示事件映射错误。"
            ),
        )

    return {
        "subject": subject,
        **{label: counts[label] for label in EVENT_LABELS},
        "stimulus_total": stimuli,
        "target_ratio": target_ratio,
        "event_total": len(rows),
    }


def _check_signal_readability(
    set_paths: Iterable[tuple[str, Path]],
    *,
    bids_root: Path,
    limit: int | None,
    issues: list[Issue],
) -> int:
    try:
        import mne
        import numpy as np
    except ImportError as exc:
        _add_issue(
            issues,
            level="ERROR",
            code="SIGNAL_CHECK_DEPENDENCY",
            message=f"无法执行信号读取检查：{exc}",
            explanation="请先按 environment.yml 创建环境并安装本项目。",
        )
        return 0

    checked = 0
    for subject, set_path in set_paths:
        if limit is not None and checked >= limit:
            break
        try:
            raw = mne.io.read_raw_eeglab(set_path, preload=False, verbose="ERROR")
            edge_samples = min(round(raw.info["sfreq"]), raw.n_times)
            start_data = raw.get_data(start=0, stop=edge_samples)
            end_data = raw.get_data(start=max(0, raw.n_times - edge_samples), stop=raw.n_times)
            if not np.isfinite(start_data).all() or not np.isfinite(end_data).all():
                raise ValueError("信号边界样本包含 NaN 或无穷值")
            checked += 1
        except Exception as exc:  # 第三方读取器可能抛出多种格式相关异常。
            _add_issue(
                issues,
                level="ERROR",
                code="SIGNAL_READ_FAILED",
                subject=subject,
                file=_relative(set_path, bids_root),
                message=f"MNE 无法完整读取记录的开头和结尾：{exc}",
                explanation="这通常表示 .set/.fdt 配对损坏、路径错误或格式不兼容。",
            )
    return checked


def validate_dataset(
    bids_root: Path,
    *,
    task: str = "P3",
    expected_subjects: int = 40,
    expected_sampling_frequency: float = 1024.0,
    expected_eeg_channels: int = 30,
    expected_eog_channels: int = 3,
    check_signals: bool = False,
    signal_limit: int | None = None,
) -> ValidationResult:
    """验证一个 ERP CORE BIDS 根目录，全程不修改原始数据。"""

    bids_root = Path(bids_root).resolve()
    issues: list[Issue] = []
    recording_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    if not bids_root.is_dir():
        _add_issue(
            issues,
            level="ERROR",
            code="BIDS_ROOT_MISSING",
            file=str(bids_root),
            message="BIDS 根目录不存在。",
            explanation="程序需要指向包含 dataset_description.json 和 sub-* 的目录。",
        )
        return ValidationResult(
            summary={"bids_root": str(bids_root), "status": "FAIL"},
            recordings=[],
            event_counts=[],
            issues=issues,
        )

    required_root_files = [
        bids_root / "dataset_description.json",
        bids_root / "participants.tsv",
        bids_root / f"task-{task}_events.json",
    ]
    for path in required_root_files:
        if not path.is_file():
            _add_issue(
                issues,
                level="ERROR",
                code="ROOT_FILE_MISSING",
                file=_relative(path, bids_root),
                message=f"缺少根目录文件 {path.name}。",
                explanation="这些文件分别描述数据集、参与者和任务事件语义。",
            )

    dataset_name = ""
    description_path = bids_root / "dataset_description.json"
    if description_path.is_file():
        try:
            dataset_name = str(_read_json(description_path).get("Name", ""))
        except (OSError, json.JSONDecodeError) as exc:
            _add_issue(
                issues,
                level="ERROR",
                code="INVALID_DATASET_DESCRIPTION",
                file=_relative(description_path, bids_root),
                message=f"dataset_description.json 无法解析：{exc}",
                explanation="JSON 格式错误会使 BIDS 工具无法正确识别数据集。",
            )
    known_task_names = ("P3", "N400", "N170", "N2PC", "MMN", "LRP", "ERN", "FLANKERS")
    named_tasks = {
        known_task for known_task in known_task_names if known_task.lower() in dataset_name.lower()
    }
    if named_tasks and task.upper() not in named_tasks:
        _add_issue(
            issues,
            level="WARNING",
            code="DATASET_NAME_TASK_MISMATCH",
            file="dataset_description.json",
            message=f"数据集名称为“{dataset_name}”，当前却在验证任务 {task}。",
            explanation="这可能只是发布包沿用了其他任务名称，但应记录为来源元数据的不一致。",
        )

    participants_path = bids_root / "participants.tsv"
    listed_subjects: set[str] = set()
    if participants_path.is_file():
        try:
            participant_rows = _read_tsv(participants_path)
            listed_subjects = {
                row.get("participant_id", "").strip()
                for row in participant_rows
                if row.get("participant_id", "").strip()
            }
        except OSError as exc:
            _add_issue(
                issues,
                level="ERROR",
                code="PARTICIPANTS_READ_FAILED",
                file="participants.tsv",
                message=f"participants.tsv 无法读取：{exc}",
                explanation="无法读取参与者清单时，不能确认数据是否完整。",
            )

    actual_subjects = {
        path.name for path in bids_root.glob("sub-*") if path.is_dir()
    }
    if len(actual_subjects) != expected_subjects:
        _add_issue(
            issues,
            level="ERROR",
            code="SUBJECT_COUNT",
            message=f"磁盘上发现 {len(actual_subjects)} 名受试者，期望 {expected_subjects} 名。",
            explanation="参与者表中的名字不等于实际数据；正式群体分析前必须补齐受试者目录。",
        )

    missing_directories = sorted(listed_subjects - actual_subjects)
    unlisted_directories = sorted(actual_subjects - listed_subjects)
    if missing_directories:
        _add_issue(
            issues,
            level="ERROR",
            code="PARTICIPANT_DATA_MISSING",
            message=f"participants.tsv 中有 {len(missing_directories)} 人没有对应 sub-* 目录。",
            explanation="清单完整并不表示信号文件已经下载；这些受试者当前不可用于分析。",
        )
    if unlisted_directories:
        _add_issue(
            issues,
            level="WARNING",
            code="UNLISTED_SUBJECT_DIRECTORY",
            message=f"有 {len(unlisted_directories)} 个 sub-* 目录未出现在 participants.tsv。",
            explanation="这会造成参与者属性与实际信号无法可靠对应。",
        )

    recordings = _discover_recordings(bids_root, task)
    recording_subjects = {subject for subject, _ in recordings}
    subjects_without_task = sorted(actual_subjects - recording_subjects)
    if subjects_without_task:
        _add_issue(
            issues,
            level="ERROR",
            code="TASK_RECORDING_MISSING",
            message=f"有 {len(subjects_without_task)} 名实际受试者缺少 {task} EEG 记录。",
            explanation="只有包含指定任务记录的受试者才能进入本项目分析。",
        )

    for subject, set_path in recordings:
        prefix = set_path.name.removesuffix("_eeg.set")
        companion_paths = {
            "fdt": set_path.with_name(f"{prefix}_eeg.fdt"),
            "eeg_json": set_path.with_name(f"{prefix}_eeg.json"),
            "channels": set_path.with_name(f"{prefix}_channels.tsv"),
            "events": set_path.with_name(f"{prefix}_events.tsv"),
        }
        for kind, path in companion_paths.items():
            if not path.is_file() or path.stat().st_size == 0:
                _add_issue(
                    issues,
                    level="ERROR",
                    code="RECORDING_FILE_MISSING",
                    subject=subject,
                    file=_relative(path, bids_root),
                    message=f"记录缺少或文件为空：{kind}。",
                    explanation="EEGLAB 的 .set 头文件必须与外部 .fdt 信号文件及 BIDS 表格配套。",
                )

        metadata: dict[str, Any] = {}
        metadata_path = companion_paths["eeg_json"]
        if metadata_path.is_file():
            try:
                metadata = _read_json(metadata_path)
            except (OSError, json.JSONDecodeError) as exc:
                _add_issue(
                    issues,
                    level="ERROR",
                    code="INVALID_EEG_JSON",
                    subject=subject,
                    file=_relative(metadata_path, bids_root),
                    message=f"EEG JSON 无法解析：{exc}",
                    explanation="采样率、参考方式和通道数等关键信息来自该文件。",
                )

        sfreq = _coerce_float(metadata.get("SamplingFrequency"))
        if sfreq != expected_sampling_frequency:
            _add_issue(
                issues,
                level="ERROR",
                code="SAMPLING_FREQUENCY",
                subject=subject,
                file=_relative(metadata_path, bids_root),
                message=f"采样率为 {sfreq} Hz，期望 {expected_sampling_frequency:g} Hz。",
                explanation="采样率决定每秒包含多少数据点，并影响滤波参数和事件时间换算。",
            )

        eeg_count = eog_count = 0
        units = ""
        channels_path = companion_paths["channels"]
        if channels_path.is_file():
            try:
                channel_rows = _read_tsv(channels_path)
                eeg_count, eog_count, units = _validate_channels(
                    channel_rows,
                    expected_eeg=expected_eeg_channels,
                    expected_eog=expected_eog_channels,
                    subject=subject,
                    file=_relative(channels_path, bids_root),
                    issues=issues,
                )
            except OSError as exc:
                _add_issue(
                    issues,
                    level="ERROR",
                    code="CHANNELS_READ_FAILED",
                    subject=subject,
                    file=_relative(channels_path, bids_root),
                    message=f"channels.tsv 无法读取：{exc}",
                    explanation="无法读取通道表时，不能确认通道类型和原始单位。",
                )

        events_path = companion_paths["events"]
        if events_path.is_file():
            try:
                event_rows.append(
                    _validate_events(
                        _read_tsv(events_path),
                        sampling_frequency=sfreq,
                        subject=subject,
                        file=_relative(events_path, bids_root),
                        issues=issues,
                    )
                )
            except OSError as exc:
                _add_issue(
                    issues,
                    level="ERROR",
                    code="EVENTS_READ_FAILED",
                    subject=subject,
                    file=_relative(events_path, bids_root),
                    message=f"events.tsv 无法读取：{exc}",
                    explanation="无法读取事件表就不能把 EEG 分成 Target 与 Standard trial。",
                )

        eeg_dir = set_path.parent
        coordinate_files = list(eeg_dir.glob("*_electrodes.tsv")) + list(
            eeg_dir.glob("*_coordsystem.json")
        )
        if not coordinate_files:
            _add_issue(
                issues,
                level="WARNING",
                code="ELECTRODE_COORDINATES_MISSING",
                subject=subject,
                file=_relative(eeg_dir, bids_root),
                message="没有找到 electrodes.tsv 或 coordsystem.json。",
                explanation=(
                    "后续可按标准 10-20 通道名称使用模板坐标，但模板坐标不是该受试者实际测量坐标，"
                    "报告中必须明确这一限制。"
                ),
            )

        recording_rows.append(
            {
                "subject": subject,
                "task": task,
                "set_file": _relative(set_path, bids_root),
                "set_size_bytes": set_path.stat().st_size,
                "fdt_size_bytes": (
                    companion_paths["fdt"].stat().st_size
                    if companion_paths["fdt"].is_file()
                    else 0
                ),
                "sampling_frequency_hz": sfreq,
                "eeg_channels": eeg_count,
                "eog_channels": eog_count,
                "units": units,
                "reference": metadata.get("EEGReference", ""),
                "line_frequency_hz": metadata.get("PowerLineFrequency", ""),
                "duration_s": metadata.get("RecordingDuration", ""),
                "coordinate_files_present": bool(coordinate_files),
            }
        )

    signal_checked = 0
    if check_signals:
        signal_checked = _check_signal_readability(
            recordings,
            bids_root=bids_root,
            limit=signal_limit,
            issues=issues,
        )

    issue_counts = Counter(issue.level for issue in issues)
    status = "FAIL" if issue_counts["ERROR"] else "PASS_WITH_WARNINGS" if issues else "PASS"
    aggregate_events = Counter()
    for row in event_rows:
        for label in EVENT_LABELS:
            aggregate_events[label] += int(row[label])

    summary = {
        "bids_root": str(bids_root),
        "dataset_name": dataset_name,
        "task": task,
        "status": status,
        "expected_subjects": expected_subjects,
        "participants_listed": len(listed_subjects),
        "subject_directories_found": len(actual_subjects),
        "task_recordings_found": len(recordings),
        "subjects_missing_from_disk": missing_directories,
        "subjects_without_task_recording": subjects_without_task,
        "issues": dict(sorted(issue_counts.items())),
        "event_totals": dict(aggregate_events),
        "signal_files_checked": signal_checked,
        "notes": {
            "unit": "channels.tsv 的 microV 是原始记录单位；MNE 内部通常统一使用 V。",
            "event_mapping": (
                "11/22/33/44/55=Target；其余 1–5 组成的两位刺激码=Standard；"
                "201=正确反应；202=错误反应。"
            ),
            "coordinate_limit": "缺失个体电极坐标时可使用标准 montage，但不能声称使用了实测坐标。",
        },
    }
    return ValidationResult(summary, recording_rows, event_rows, issues)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    # utf-8-sig 让 Windows Excel 能直接正确识别中文，不改变字段实际内容。
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_validation_reports(result: ValidationResult, output_dir: Path) -> list[Path]:
    """将验证结果写为 JSON 和 CSV；不会写入 BIDS 原始目录。"""

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "validation_summary.json"
    summary_path.write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    recordings_path = output_dir / "subject_task_inventory.csv"
    _write_csv(
        recordings_path,
        result.recordings,
        (
            "subject",
            "task",
            "set_file",
            "set_size_bytes",
            "fdt_size_bytes",
            "sampling_frequency_hz",
            "eeg_channels",
            "eog_channels",
            "units",
            "reference",
            "line_frequency_hz",
            "duration_s",
            "coordinate_files_present",
        ),
    )

    events_path = output_dir / "event_counts.csv"
    _write_csv(
        events_path,
        result.event_counts,
        (
            "subject",
            "target",
            "standard",
            "correct_response",
            "error_response",
            "unknown",
            "stimulus_total",
            "target_ratio",
            "event_total",
        ),
    )

    issues_path = output_dir / "issues.csv"
    _write_csv(
        issues_path,
        [asdict(issue) for issue in result.issues],
        ("level", "code", "subject", "file", "message", "explanation"),
    )
    return [summary_path, recordings_path, events_path, issues_path]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="只读验证 ERP CORE P3 BIDS 数据，并输出 JSON/CSV 审计报告。"
    )
    parser.add_argument(
        "--bids-root",
        type=Path,
        default=Path("data/raw/erp-core-p3"),
        help="包含 dataset_description.json 与 sub-* 的 BIDS 根目录。",
    )
    parser.add_argument("--task", default="P3", help="要检查的任务名，默认 P3。")
    parser.add_argument("--expected-subjects", type=int, default=40)
    parser.add_argument("--expected-sfreq", type=float, default=1024.0)
    parser.add_argument("--expected-eeg-channels", type=int, default=30)
    parser.add_argument("--expected-eog-channels", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/data_validation"),
    )
    parser.add_argument(
        "--check-signals",
        action="store_true",
        help="使用 MNE 读取每份 .set/.fdt 的开头和结尾；比纯元数据检查慢。",
    )
    parser.add_argument(
        "--signal-limit",
        type=int,
        default=None,
        help="最多读取多少份信号；不指定表示全部。",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="发现 ERROR 时以退出码 1 结束，适合自动化流水线。",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = validate_dataset(
        args.bids_root,
        task=args.task,
        expected_subjects=args.expected_subjects,
        expected_sampling_frequency=args.expected_sfreq,
        expected_eeg_channels=args.expected_eeg_channels,
        expected_eog_channels=args.expected_eog_channels,
        check_signals=args.check_signals,
        signal_limit=args.signal_limit,
    )
    paths = write_validation_reports(result, args.output_dir)

    # 保持终端摘要为 ASCII，兼容部分 Windows Conda 启动器的 GBK 输出捕获问题。
    print(f"Validation status: {result.summary['status']}")
    print(
        "Subjects: "
        f"listed={result.summary.get('participants_listed', 0)}, "
        f"directories={result.summary.get('subject_directories_found', 0)}, "
        f"{args.task}_recordings={result.summary.get('task_recordings_found', 0)}"
    )
    print(f"Issues: {result.summary.get('issues', {})}")
    print("Report files:")
    for path in paths:
        print(f"  - {path}")
    return 1 if args.strict and result.summary["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
