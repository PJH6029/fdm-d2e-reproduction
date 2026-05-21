from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _script(path: str) -> str:
    return (ROOT / path).read_text()


def test_g003_integrated_run_builds_split_statistics_before_evidence() -> None:
    text = _script("scripts/run_g003_d2e_full_idm_parallel.sh")

    train_idx = text.index("scripts/train_idm_streaming.py")
    split_idx = text.index("scripts/build_split_statistical_comparisons.py")
    evidence_idx = text.index("g003_d2e_full_idm_run_evidence.v1")

    assert "SPLIT_STATS_CONFIG=\"${SPLIT_STATS_CONFIG:-configs/eval/g003_split_statistics.yaml}\"" in text
    assert "SPLIT_STATS_SUMMARY=\"${SPLIT_STATS_SUMMARY:-artifacts/eval/g003_split_statistical_comparisons_summary.json}\"" in text
    assert train_idx < split_idx < evidence_idx
    assert '"split_stats_summary_exists": split_stats_summary_path.exists()' in text


def test_standalone_g003_and_g004_wrappers_fail_closed_on_split_statistics() -> None:
    wrappers = {
        "scripts/run_g003_idm_training_4xh200.sh": "configs/eval/g003_split_statistics.yaml",
        "scripts/run_g004_d2e_full_fdm_4xh200.sh": "configs/eval/g004_split_statistics.yaml",
    }

    for script, config in wrappers.items():
        text = _script(script)
        assert f'SPLIT_STATS_CONFIG="${{SPLIT_STATS_CONFIG:-{config}}}"' in text
        assert 'BUILD_SPLIT_STATS="${BUILD_SPLIT_STATS:-1}"' in text
        assert "set -euo pipefail" in text
        assert "scripts/build_split_statistical_comparisons.py --config \"$SPLIT_STATS_CONFIG\"" in text
        assert "split_stats_summary_exists" in text
        assert "split_stats_status" in text
