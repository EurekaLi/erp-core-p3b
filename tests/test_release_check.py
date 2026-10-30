"""发布边界测试：即使被强行跟踪，也拒绝大数据/权重；导航不得依赖本地缓存。"""

import json

import pytest

from erp_core_p3b.release_check import candidate_issue, check_assets, check_links


@pytest.mark.parametrize(
    "path",
    [
        "data/raw/sub-001/events.tsv",
        "models/best.pt",
        "third_party/CBraMod/README.md",
        "results/final_report/v1/trial_error_audit.csv",
        ".env.local",
        ".pytest-release-tmp/test/a.py",
        "docs/17_interview_questions_signals_audio_video.md",
    ],
)
def test_reject_private_candidates(path):
    assert candidate_issue(path, 100, 1024)


def test_public_assets_size_and_tracked_erp_exceptions():
    assert candidate_issue("reports/release/tables/model_overview.csv", 100, 1024) is None
    assert candidate_issue("results/erp_analysis/group_statistics.csv", 100, 1024) is None
    assert candidate_issue("data/raw/.gitkeep", 0, 1024) is None
    assert candidate_issue("reports/large.png", 1025, 1024)


def test_link_must_be_in_git_candidate_not_just_on_disk(tmp_path):
    (tmp_path / "README.md").write_text("[local](cache.csv)", encoding="utf-8")
    (tmp_path / "cache.csv").write_text("local", encoding="utf-8")
    assert check_links(tmp_path, "README.md", {"README.md"})
    assert not check_links(tmp_path, "README.md", {"README.md", "cache.csv"})


def test_asset_hash_mismatch_without_original_data(tmp_path):
    release = tmp_path / "reports/release"
    release.mkdir(parents=True)
    (release / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "artifacts": [
                    {"path": "reports/release/a.csv", "bytes": 3, "sha256": "invalid"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (release / "a.csv").write_text("abc", encoding="utf-8")
    assert check_assets(tmp_path)
