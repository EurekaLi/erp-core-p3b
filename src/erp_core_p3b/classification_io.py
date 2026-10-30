"""分类流水线的版本化产物约定：新目录、完成状态、内容哈希和防混用检查。"""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from erp_core_p3b.cbramod_audit import sha256_file


def write_json(path: Path, value: Any) -> None:
    """禁止JSON里的NaN，未定义统计量应在调用方明确使用null/CSV缺失值。"""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        **{
            package: metadata.version(package)
            for package in (
                "mne",
                "numpy",
                "scipy",
                "pandas",
                "scikit-learn",
                "PyWavelets",
                "joblib",
            )
        },
    }


def resolve_path(root: Path, path: Path | str) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (root / value).resolve()


@contextmanager
def new_artifact_run(root: Path, output: Path, stage: str) -> Iterator[dict[str, Any]]:
    """只在项目内新建独立产物目录；失败保留诊断，不覆盖旧版本或静默复用。"""
    root, output = root.resolve(), output.resolve()
    if not output.is_relative_to(root) or output == root:
        raise ValueError("输出目录必须是项目内的独立子目录")
    output.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "stage": stage,
        "status": "in_progress",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "software": software_versions(),
        "implementation_sha256": {
            path.name: sha256_file(path) for path in sorted(Path(__file__).parent.glob("*.py"))
        },
    }
    write_json(output / "run_summary.json", summary)
    try:
        yield summary
        # 摘要不参与自身校验，避免自引用；各下游检查其他全部产物的完整性。
        summary["artifact_sha256"] = {
            path.relative_to(output).as_posix(): sha256_file(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "run_summary.json"
        }
        summary["status"] = "completed"
        write_json(output / "run_summary.json", summary)
    except Exception as error:
        summary["status"] = "failed"
        summary["error"] = f"{type(error).__name__}: {error}"
        write_json(output / "run_summary.json", summary)
        raise


def verify_artifact(directory: Path, stage: str) -> dict[str, Any]:
    """拒绝未完成或被改动的输入；哈希用于防误混用，不是数字签名认证。"""
    directory = directory.resolve()
    summary = json.loads((directory / "run_summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "completed" or summary.get("stage") != stage:
        raise ValueError(f"{directory}: 上游产物未完成或阶段不符")
    if not summary.get("artifact_sha256"):
        raise ValueError("缺少上游产物校验清单")
    for relative, expected in summary["artifact_sha256"].items():
        path = (directory / relative).resolve()
        if (
            not path.is_relative_to(directory)
            or not path.is_file()
            or sha256_file(path) != expected
        ):
            raise ValueError(f"上游产物文件缺失/改动: {relative}")
    return summary
