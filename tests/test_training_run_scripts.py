from __future__ import annotations

import json
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


def test_g005_exactset_history_uses_precompute_then_fail_closed_training() -> None:
    precompute = _script("scripts/run_g005_idm_exactset_history_precompute.sh")
    exactset = _script("scripts/run_g005_idm_exactset_history_4xh200.sh")
    surface = _script("scripts/run_g005_idm_surface_paper_target_4xh200.sh")
    recovery = _script("scripts/recover_g005_idm_exactset_history_from_checkpoint.sh")

    stats_idx = precompute.index("scripts/precompute_streaming_idm_stats.py")
    cache_idx = precompute.index("scripts/precompute_streaming_idm_training_cache.py")
    validate_idx = precompute.rindex("--validate-only")
    assert stats_idx < cache_idx < validate_idx
    assert 'ALLOW_CACHE_BUILD="${ALLOW_CACHE_BUILD:-0}"' in exactset
    assert 'REQUIRE_PRECOMPUTED_CACHE="${REQUIRE_PRECOMPUTED_CACHE:-1}"' in exactset
    assert "--validate-only" in surface
    assert surface.index("validating precomputed streaming IDM stats/cache") < surface.index("uv run torchrun")
    assert 'PREDICTION_WORKERS="${PREDICTION_WORKERS:-64}"' in recovery
    assert "--prediction-workers \"$PREDICTION_WORKERS\"" in recovery
    assert "scripts/build_split_statistical_comparisons.py --config \"$SPLIT_STATS_CONFIG\"" in recovery
    assert "scripts/build_g005_idm_paper_metrics.py --config \"$PAPER_TARGET_CONFIG\"" in recovery
    assert "scripts/validate_g005_idm_paper_target.py --config \"$PAPER_TARGET_CONFIG\"" in recovery
    assert "initial_integrated_process_interrupted_after_checkpoint" in recovery


def test_g005_state_luma_pair_materializes_state_corpus_and_logs_wandb() -> None:
    text = _script("scripts/run_g005_idm_state_luma_pair_4xh200.sh")

    materialize_idx = text.index("scripts/materialize_d2e_state_corpus.py")
    stats_idx = text.index("scripts/synthesize_state_streaming_stats.py")
    cache_idx = text.index("scripts/precompute_streaming_idm_training_cache.py")
    stats_artifact_idx = text.index("$MODEL_SLUG-stats-synthesis")
    cache_artifact_idx = text.index("$MODEL_SLUG-cache-precompute")
    wandb_idx = text.index("uv run --with wandb python scripts/watch_wandb_training.py")
    train_idx = text.index("scripts/run_g005_idm_surface_paper_target_4xh200.sh")
    assert materialize_idx < stats_idx < stats_artifact_idx < cache_idx < cache_artifact_idx < wandb_idx < train_idx
    assert 'ENABLE_WANDB_SIDECAR="${ENABLE_WANDB_SIDECAR:-1}"' in text
    assert 'WANDB_TAGS="${WANDB_TAGS:-g005,idm,d2e,state-corpus,pipeline}"' in text
    assert "--env-file \"$WANDB_ENV_FILE\"" in text
    assert "$MODEL_SLUG-stats-synthesis" in text
    assert "$MODEL_SLUG-cache-precompute" in text
    assert "STATE_STATS_SYNTHESIS_WANDB_STATUS" in text
    assert "PRECOMPUTE_CACHE_WANDB_STATUS" in text
    assert "--workers \"${STATE_MATERIALIZE_WORKERS:-16}\"" in text
    assert "outputs/data/d2e_state_corpus_shards_accel64" in text
    assert "ALLOW_CACHE_BUILD=0" in text
    assert "REQUIRE_PRECOMPUTED_CACHE=1" in text


def test_g005_state_sequence_prior_uses_distinct_wandb_artifact_status_paths() -> None:
    text = _script("scripts/run_g005_idm_state_sequence_prior_4xh200.sh")
    config = json.loads((ROOT / "configs/model/idm_streaming_d2e_full_state_sequence_prior_paper_target.yaml").read_text())

    assert "g005_idm_state_sequence_prior_precompute_wandb_status.json" in text
    assert "g005_idm_state_sequence_prior_stats_synthesis_wandb_status.json" in text
    assert 'WANDB_TAGS="${WANDB_TAGS:-g005,idm,d2e,state-sequence-prior,pipeline}"' in text
    assert "scripts/run_g005_idm_state_luma_pair_4xh200.sh" in text
    assert config["action_history_seed_state_mode"] == "empty"


def test_g005_state_sequence_stack5_prior_uses_future_context_without_seed_synthesis() -> None:
    text = _script("scripts/run_g005_idm_state_sequence_stack5_prior_4xh200.sh")
    config = json.loads(
        (ROOT / "configs/model/idm_streaming_d2e_full_state_sequence_stack5_prior_paper_target.yaml").read_text()
    )
    paper = json.loads((ROOT / "configs/eval/g005_idm_state_sequence_stack5_prior_paper_target.yaml").read_text())

    assert "g005_idm_state_sequence_stack5_prior_precompute_wandb_status.json" in text
    assert 'SKIP_STATE_STATS_SYNTHESIS="${SKIP_STATE_STATS_SYNTHESIS:-1}"' in text
    assert "scripts/run_g005_idm_state_sequence_prior_4xh200.sh" in text
    assert config["feature_mode"] == "summary_luma16_stack5_time"
    assert config["model_arch"] == "luma_action_sequence_prior"
    assert config["visual_stack_frames"] == 5
    assert config["action_history_seed_state_mode"] == "empty"
    assert paper["paths"]["run_summary"] == "artifacts/idm/g005_idm_state_sequence_stack5_prior_4xh200_run.json"


def test_g005_video_stack_offset_candidate_separates_precompute_training_and_recovery() -> None:
    precompute = _script("scripts/run_g005_idm_video_stack_luma96_offsets012_precompute.sh")
    training = _script("scripts/run_g005_idm_video_stack_luma96_offsets012_4xh200.sh")
    recovery = _script("scripts/recover_g005_idm_video_stack_luma96_offsets012_from_checkpoint.sh")

    assert "scripts/precompute_video_idm_cache.py --config \"$CONFIG\"" in precompute
    assert 'PRECOMPUTE_SPLITS="${PRECOMPUTE_SPLITS:-}"' in precompute
    assert 'INSTALL_FFMPEG_IF_MISSING="${INSTALL_FFMPEG_IF_MISSING:-1}"' in precompute
    assert "apt-get install -y --no-install-recommends ffmpeg" in precompute
    assert "CMD+=(--splits \"$PRECOMPUTE_SPLITS\")" in precompute
    assert 'SKIP_PREDICTION="${SKIP_PREDICTION:-1}"' in training
    assert 'BUILD_SPLIT_STATS="${BUILD_SPLIT_STATS:-0}"' in training
    assert 'BUILD_PAPER_METRICS="${BUILD_PAPER_METRICS:-0}"' in training
    assert "scripts/run_g005_idm_video_pair_raw112_4xh200.sh" in training
    assert 'PREDICTION_WORKERS="${PREDICTION_WORKERS:-4}"' in recovery
    assert "scripts/recover_idm_video_outputs.py" in recovery
    assert "--prediction-workers \"$PREDICTION_WORKERS\"" in recovery
    assert "scripts/build_split_statistical_comparisons.py --config \"$SPLIT_STATS_CONFIG\"" in recovery
    assert "scripts/build_g005_idm_paper_metrics.py --config \"$PAPER_TARGET_CONFIG\"" in recovery
    assert "scripts/validate_g005_idm_paper_target.py --config \"$PAPER_TARGET_CONFIG\"" in recovery


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
