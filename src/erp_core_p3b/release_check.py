"""仅用标准库核验Git公开候选与精选产物，既不暂存文件，也不访问网络。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

FORBIDDEN_SUFFIXES = {
    ".fif",
    ".set",
    ".fdt",
    ".edf",
    ".bdf",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".joblib",
    ".pkl",
    ".zip",
    ".pem",
    ".key",
}
PRIVATE_PREFIXES = ("data/raw/", "data/interim/", "data/processed/", "third_party/", "models/")
ERP_RESULTS = {
    "analysis_manifest.json",
    "cluster_summary.csv",
    "group_statistics.csv",
    "results_summary.md",
    "sensitivity_analysis.csv",
}
SECRET = re.compile(
    r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----|"
    r"\bgh[pousr]_[A-Za-z0-9]{30,}\b|\bhf_[A-Za-z0-9]{30,}\b"
)


def candidate_issue(relative: str, size: int, max_bytes: int) -> str | None:
    """即使误用git add -f，也拒绝公开大数据目录、凭据文件和本地面试材料。"""
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts:
        return "非法候选路径"
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        return "数据、权重、归档或凭据文件"
    if any(
        part.startswith((".env", ".pytest-"))
        or part
        in {
            "__pycache__",
            ".venv",
            ".conda",
            ".codex_tmp",
            ".matplotlib-cache",
        }
        for part in path.parts
    ):
        return "本地环境或临时目录"
    if relative == "docs/17_interview_questions_signals_audio_video.md":
        return "个人面试题库不属于公开研究材料"
    if relative.startswith(PRIVATE_PREFIXES) and path.name != ".gitkeep":
        return "本地数据/第三方资源目录"
    if relative.startswith("results/") and relative != "results/README.md":
        if path.parent.as_posix() != "results/erp_analysis" or path.name not in ERP_RESULTS:
            return "未选择公开的运行产物"
    if size > max_bytes:
        return "超过公开文件大小上限"
    return None


def check_assets(root: Path) -> list[str]:
    """只核验发布副本；新克隆不需要已忽略的原始results目录。"""
    problems = []
    manifest = json.loads((root / "reports/release/artifact_manifest.json").read_text("utf-8"))
    for row in manifest["artifacts"]:
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            problems.append(f"缺失/非法精选产物：{row['path']}")
            continue
        if (
            path.stat().st_size != row["bytes"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]
        ):
            problems.append(f"精选产物大小/hash不一致：{row['path']}")
    return problems


def check_links(root: Path, document: str, candidates: set[str]) -> list[str]:
    """首页和发布导航的相对链接必须指向实际会公开的文件，而非仅本机存在的缓存。"""
    path = root / document
    problems = []
    for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", path.read_text("utf-8")):
        if re.match(r"(?:https?://|mailto:|#)", target):
            continue
        linked = (path.parent / target.split("#", 1)[0]).resolve()
        if not linked.is_relative_to(root.resolve()):
            problems.append(f"跨仓库链接：{document} -> {target}")
        elif linked.relative_to(root.resolve()).as_posix() not in candidates:
            problems.append(f"链接不在公开候选：{document} -> {target}")
    return problems


def audit(root: Path, max_bytes: int = 20 * 1024**2) -> dict:
    root = root.resolve()
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    candidates = {name for name in result.stdout.decode("utf-8").split("\0") if name}
    problems, total = [], 0
    for relative in sorted(candidates):
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            problems.append(f"缺失/非法公开文件：{relative}")
            continue
        size = path.stat().st_size
        total += size
        issue = candidate_issue(relative, size, max_bytes)
        if issue:
            problems.append(f"{relative}：{issue}")
            continue
        if path.suffix.lower() in {".md", ".py", ".yaml", ".yml", ".toml", ".json", ".txt"}:
            # 只报告命中的文件，不将潜在凭据打印到终端或报告中。
            if SECRET.search(path.read_text("utf-8")):
                problems.append(f"疑似凭据：{relative}")
    problems.extend(check_assets(root))
    for document in ("README.md", "REPRODUCING.md", "NOTICE.md", "reports/release/README.md"):
        if document not in candidates:
            problems.append(f"缺少发布入口：{document}")
        else:
            problems.extend(check_links(root, document, candidates))
    return {
        "status": "passed" if not problems else "failed",
        "candidate_files": len(candidates),
        "candidate_bytes": total,
        "max_file_bytes": max_bytes,
        "problems": problems,
        "scope": "working_tree_candidates_not_git_history_or_comprehensive_secret_audit",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    try:
        result = audit(args.project_root)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"Release check failed: {error}")
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1
