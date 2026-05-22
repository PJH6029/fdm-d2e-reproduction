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
    assert "IDM_SUMMARY=\"${IDM_SUMMARY:-artifacts/idm/idm_streaming_d2e_full_compact_summary.json}\"" in text
    assert "export BUILD_SPLIT_STATS SPLIT_STATS_CONFIG SPLIT_STATS_SUMMARY IDM_SUMMARY" in text
    assert "scripts/precompute_streaming_idm_stats.py" in text
    assert text.index("scripts/precompute_streaming_idm_stats.py") < text.index("torchrun")
    assert train_idx < split_idx < evidence_idx
    assert '"split_stats_summary_exists": split_stats_summary_path.exists()' in text


def test_g003_accel64_training_resume_skips_extraction_and_preserves_evidence_paths() -> None:
    text = _script("scripts/run_g003_accel64_training_resume.sh")
    assert "scripts/extract_d2e_full_corpus.py" not in text
    assert "scripts/merge_d2e_full_corpus_shards.py" not in text
    assert "artifacts/sources/d2e_full_corpus_decode_summary_accel64.json" in text
    assert '"${DATA_OUTPUT_DIR}/train_core.jsonl"' in text
    assert "configs/model/idm_streaming_d2e_full_compact_accel64.yaml" in text
    assert "scripts/precompute_streaming_idm_stats.py" in text
    assert text.index("scripts/precompute_streaming_idm_stats.py") < text.index("torchrun")
    assert "training_only_after_successful_accel64_merge" in text


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


def test_g004_wrapper_exposes_parent_pid_for_postrun_watcher() -> None:
    text = _script("scripts/run_g004_d2e_full_fdm_4xh200.sh")
    assert 'PID_FILE="${PID_FILE:-outputs/cluster/g004_d2e_full_fdm_4xh200.pid}"' in text
    assert 'echo "$$" >"$PID_FILE"' in text
    assert "cleanup_pid_file" in text
    assert '"pid_file": "$PID_FILE"' in text


def test_runbooks_do_not_overwrite_self_written_watcher_pids() -> None:
    doc_text = "\n".join(
        path.read_text()
        for directory in ("docs", "notes")
        for path in sorted((ROOT / directory).glob("*.md"))
    )
    self_written_pid_files = [
        "outputs/cluster/g003_postrun_watcher.pid",
        "outputs/cluster/g003_to_g004_chain_watcher.pid",
        "outputs/cluster/g004_postrun_watcher.pid",
        "outputs/cluster/g004_to_g005_readiness_chain.pid",
        "outputs/cluster/g005_aux_materialization_watcher.pid",
        "outputs/cluster/g005_postrun_watcher.pid",
    ]

    for pid_file in self_written_pid_files:
        assert f"echo $! > {pid_file}" not in doc_text

    # Parent/background job PID files that are not self-written watchers remain valid.
    assert "echo $! > outputs/cluster/g005_aux_materialization.pid" in doc_text
