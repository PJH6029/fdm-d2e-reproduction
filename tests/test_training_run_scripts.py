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


def test_g005_raw112_offset2_candidate_uses_nonleaky_nep100_video_paths() -> None:
    precompute = _script("scripts/run_g005_idm_video_pair_raw112_offset2_precompute.sh")
    training = _script("scripts/run_g005_idm_video_pair_raw112_offset2_4xh200.sh")
    recovery = _script("scripts/recover_g005_idm_video_pair_raw112_offset2_from_checkpoint.sh")
    prefix_probe = _script("scripts/run_g005_idm_video_pair_raw112_offset2_prefix_probe.sh")
    config = json.loads(
        (ROOT / "configs/model/idm_video_pair_d2e_full_raw112_offset2_keysoftmax_paper_target.yaml").read_text()
    )
    paper = json.loads((ROOT / "configs/eval/g005_idm_video_pair_raw112_offset2_keysoftmax_paper_target.yaml").read_text())
    split = json.loads((ROOT / "configs/eval/g005_idm_video_pair_raw112_offset2_keysoftmax_split_statistics.yaml").read_text())

    assert "scripts/run_g005_idm_video_stack_luma96_offsets012_precompute.sh" in precompute
    assert "scripts/run_g005_idm_video_pair_raw112_4xh200.sh" in training
    assert "scripts/recover_g005_idm_video_stack_luma96_offsets012_from_checkpoint.sh" in recovery
    assert "Prefix diagnostic only; not full-corpus G005 completion evidence" in prefix_probe
    assert 'PREFIX_ROWS="${PREFIX_ROWS:-320000}"' in prefix_probe
    assert 'PREFIX_TRAIN_SHARDS="${PREFIX_TRAIN_SHARDS:-2}"' in prefix_probe
    assert "existing_completed_cache_manifests" in prefix_probe
    assert '--finish-manifests "$PREFIX_TRAIN_SHARDS"' in prefix_probe
    assert "scripts/run_g005_idm_video_pair_raw112_offset2_precompute.sh" in prefix_probe
    assert "scripts/run_g005_idm_video_pair_raw112_offset2_4xh200.sh" in prefix_probe
    assert "scripts/predict_idm_video.py" in prefix_probe
    assert "scripts/build_g005_idm_paper_metrics.py" in prefix_probe
    assert 'SKIP_PREDICTION="${SKIP_PREDICTION:-1}"' in training
    assert 'BUILD_SPLIT_STATS="${BUILD_SPLIT_STATS:-0}"' in training
    assert 'PREDICTION_WORKERS="${PREDICTION_WORKERS:-4}"' in recovery
    assert config["source_namespace"] == "d2e_full_corpus"
    assert config["video_image_size"] == 112
    assert config["video_input_mode"] == "pair_delta_abs"
    assert config["next_frame_offset"] == 2
    assert config["video_frame_offsets"] == [0, 2]
    assert config["train_records_glob"] == "outputs/data/d2e_full_corpus_shards_accel64/shard_*/train_core.jsonl"
    assert paper["paper_metrics"]["target_path"] == "outputs/data/d2e_full_corpus_accel64/target_all_eval.jsonl"
    assert paper["paper_metrics"]["predictions_path"] == (
        "outputs/idm_video_pair_d2e_full_raw112_offset2_keysoftmax_paper_target/predictions.jsonl"
    )
    assert split["ground_truth_path"] == "outputs/data/d2e_full_corpus_accel64/target_all_eval.jsonl"
    assert split["train_stats_path"] == (
        "outputs/idm_video_pair_d2e_full_raw112_offset2_keysoftmax_paper_target/video_idm_stats.json"
    )


def test_g005_realvideo_frozen_embedding_prefix_uses_video_decode_and_cache() -> None:
    text = _script("scripts/run_g005_idm_frozen_frame_embedding_realvideo_prefix16k.sh")
    base = _script("scripts/run_g005_idm_frozen_frame_embedding_prefix.sh")
    config = json.loads(
        (ROOT / "configs/model/idm_streaming_d2e_full_frozen_frame_embedding_realvideo_prefix16k.yaml").read_text()
    )
    paper = json.loads(
        (ROOT / "configs/eval/g005_idm_frozen_frame_embedding_realvideo_prefix16k_paper_metrics.yaml").read_text()
    )

    assert "scripts/run_g005_idm_frozen_frame_embedding_prefix.sh" in text
    assert 'EMBED_FRAME_SOURCE="${EMBED_FRAME_SOURCE:-video}"' in text
    assert 'EMBED_BACKEND="${EMBED_BACKEND:-dinov2-torchhub}"' in text
    assert 'EMBED_FEATURE_CACHE="${EMBED_FEATURE_CACHE:-1}"' in text
    assert 'EMBED_THIN_OUTPUT="${EMBED_THIN_OUTPUT:-1}"' in text
    assert 'MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-16000}"' in text
    assert config["source_namespace"] == "d2e_frozen_frame_embedding_realvideo_prefix16k"
    assert config["train_records"] == "outputs/data/d2e_frozen_frame_embedding_realvideo_prefix16k/train_core.jsonl"
    assert config["target_records"] == "outputs/data/d2e_frozen_frame_embedding_realvideo_prefix16k/target_all_eval.jsonl"
    assert "cv2/ffmpeg" in config["claim_boundary"]
    assert paper["max_rows"] == 16000
    assert paper["target_paths"] == ["outputs/data/d2e_frozen_frame_embedding_realvideo_prefix16k/target_all_eval.jsonl"]
    assert 'PAPER_METRICS_JSON="${PAPER_METRICS_JSON:-}"' in base
    assert '"paper_metrics_path": "$PAPER_METRICS_JSON"' in base
    assert 'g005_idm_frozen_frame_embedding_prefix320k_paper_metrics.json' not in base


def test_g005_realvideo_raw96_axisclass_prefix_uses_balanced_real_video() -> None:
    text = _script("scripts/run_g005_idm_temporal_raw96_patch_axisclass_realvideo_prefix32k.sh")
    config = json.loads(
        (
            ROOT
            / "configs/model/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_prefix32k.yaml"
        ).read_text()
    )

    assert "scripts/materialize_balanced_prefix.py" in text
    assert "--balance-key recording_id" in text
    assert "--balance-key eval_split_tags" in text
    assert "--group-value heldout_recording" in text
    assert "scripts/run_g005_idm_temporal_raw96_family_presence_prefix.sh" in text
    assert 'SUMMARY_PATH="${SUMMARY_PATH:-artifacts/idm/${MODEL_SLUG}_summary.json}"' in text
    assert config["video_feature_source"] == "raw_frames"
    assert config["require_precomputed_video_cache"] is False
    assert config["train_records"] == "outputs/data/d2e_event_state_duration_realvideo_balanced_prefix32k/train_core.jsonl"
    assert config["target_records"] == "outputs/data/d2e_event_state_duration_realvideo_balanced_prefix32k/target_all_eval.jsonl"
    assert config["max_train_rows"] == 32000
    assert config["max_target_rows"] == 24000
    assert config["raw_video_frame_offsets"] == [0, 1, 2]
    assert "actual D2E video" in config["claim_boundary"]


def test_g005_realvideo_raw96_train320k_uses_distributed_feature_cache() -> None:
    text = _script("scripts/run_g005_idm_temporal_raw96_patch_axisclass_realvideo_train320k_target24k.sh")
    config = json.loads(
        (
            ROOT
            / "configs/model/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_train320k_target24k.yaml"
        ).read_text()
    )

    assert "scripts/materialize_balanced_prefix.py" in text
    assert 'MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-320000}"' in text
    assert 'TRAIN_MAX_PER_RECORDING="${TRAIN_MAX_PER_RECORDING:-20000}"' in text
    assert 'SUMMARY_PATH="${SUMMARY_PATH:-artifacts/idm/${MODEL_SLUG}_summary.json}"' in text
    assert "distributed-feature-cache" in text
    assert config["video_feature_source"] == "raw_frames"
    assert config["distributed_feature_cache_dir"].endswith("/distributed_feature_cache")
    assert config["train_records"] == "outputs/data/d2e_event_state_duration_realvideo_balanced_train320k_target24k/train_core.jsonl"
    assert config["target_records"] == "outputs/data/d2e_event_state_duration_realvideo_balanced_train320k_target24k/target_all_eval.jsonl"
    assert config["max_train_rows"] == 320000
    assert config["max_target_rows"] == 24000
    assert config["hidden_dim"] == 512
    assert config["transformer_layers"] == 6
    assert "distributed raw-frame feature cache" in config["claim_boundary"]


def test_g005_statectx_candidate_reranker_is_prediction_only_and_split_safe() -> None:
    text = _script("scripts/run_g005_idm_temporal_raw96_statectx_reranker_predict24k.sh")
    config = json.loads(
        (
            ROOT
            / "configs/model/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_statectx_reranker_predict24k.yaml"
        ).read_text()
    )

    assert "scripts/predict_idm_temporal_masked_diffusion.py" in text
    assert "scripts/log_wandb_artifacts.py" in text
    assert "statectx_prefix32k" in text
    assert 'CHECKPOINT="${CHECKPOINT:-$BASE_OUTPUT_DIR/checkpoint.pt}"' in text
    assert "prediction-reranker" in text
    assert "paper_target_pass" in text
    assert "not G005 completion evidence" in text
    assert "torchrun" not in text
    assert config["source_checkpoint"] == (
        "outputs/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_statectx_prefix32k/checkpoint.pt"
    )
    assert config["target_records"] == (
        "outputs/data/d2e_event_state_duration_realvideo_balanced_mouseagg_prefix32k/target_all_eval.jsonl"
    )
    assert config["candidate_score_reranker_enabled"] is True
    assert config["candidate_score_reranker_blend"] == 1.0
    assert config["candidate_score_reranker_epochs"] == 300
    assert config["candidate_score_reranker_max_examples_per_family"] == 240000
    assert "candidate_score_reranker" in config["fdm1_recipe_alignment"]
    assert "target-label calibration" in config["fdm1_recipe_alignment"]["candidate_score_reranker"]["claim_boundary"]


def test_g005_statectx_train320k_scales_best_prefix_with_state_features() -> None:
    text = _script("scripts/run_g005_idm_temporal_raw96_patch_axisclass_realvideo_statectx_train320k_target24k.sh")
    config = json.loads(
        (
            ROOT
            / "configs/model/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_statectx_train320k_target24k.yaml"
        ).read_text()
    )

    materialize_idx = text.index("scripts/materialize_balanced_prefix.py")
    train_idx = text.index("scripts/run_g005_idm_temporal_raw96_family_presence_prefix.sh")
    assert materialize_idx < train_idx
    assert 'MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-320000}"' in text
    assert 'NPROC_PER_NODE="${NPROC_PER_NODE:-4}"' in text
    assert "state_duration_prior_action_features" in text
    assert "state-context,balanced-prefix,train320k" in text
    assert "outputs/data/d2e_event_state_duration_realvideo_balanced_train320k_target24k" in text
    assert "g005_idm_temporal_masked_diffusion_raw96_patch_axisclass_realvideo_train320k_target24k_source_train_balanced_summary.json" in text
    assert "g005_idm_temporal_masked_diffusion_raw96_patch_axisclass_realvideo_train320k_target24k_source_target_balanced_summary.json" in text
    assert config["raw_video_aux_feature_paths"] == ["state_duration_prior_action_features"]
    assert config["raw_video_aux_feature_dim"] == 156
    assert config["video_feature_dim"] == 27804
    assert config["max_train_rows"] == 320000
    assert config["max_target_rows"] == 24000
    assert config["hidden_dim"] == 512
    assert config["transformer_layers"] == 6
    assert config["epochs"] == 6
    assert config["distributed_feature_cache_dir"].endswith("/distributed_statectx_feature_cache")
    assert "state_context_320k_scaleup" in config["fdm1_recipe_alignment"]
    assert "not fdm-1 parity" in config["claim_boundary"].lower()


def test_g005_statectx_public49_train320k_keeps_recipe_bins_and_train_calibration() -> None:
    text = _script("scripts/run_g005_idm_temporal_raw96_patch_axisclass_realvideo_statectx_public49_train320k_stratcal_target24k.sh")
    config = json.loads(
        (
            ROOT
            / "configs/model/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_statectx_public49_train320k_stratcal_target24k.yaml"
        ).read_text()
    )

    assert "public49" in text
    assert "statectx_public49_train320k_stratcal_target24k" in text
    assert "scripts/run_g005_idm_temporal_raw96_family_presence_prefix.sh" in text
    assert 'NPROC_PER_NODE="${NPROC_PER_NODE:-4}"' in text
    assert "outputs/data/d2e_event_state_duration_realvideo_balanced_train320k_target24k" in text
    assert config["action_mouse_tokenization"] == "fdm1_49_aggregate"
    assert config["fdm1_recipe"]["action_tokenization"]["mouse_delta_bins_per_axis"] == 49
    assert config["fdm1_recipe"]["action_tokenization"]["metric_conversion_for_eval_only"] is True
    assert config["temporal_calibration_strategy"] == "stratified_action"
    assert config["temporal_calibration_max_rows"] == 8000
    assert config["temporal_calibration_family_quotas"] == {
        "keyboard": 2000,
        "mouse_button": 2000,
        "mouse_move": 2000,
        "noop": 2000,
    }
    assert config["adaptive_family_budget_to_unlabeled_target"] is False
    assert config["family_non_noop_budget_mouse_button_max_no_button_fpr"] == 0.10
    assert config["distributed_feature_cache_dir"].endswith("/distributed_statectx_feature_cache")
    assert "public_49_mouse_bins_state_context" in config["fdm1_recipe_alignment"]
    assert "no target-label calibration" in config["claim_boundary"]
    assert "Not FDM-1 parity" in config["claim_boundary"]


def test_g005_statectx_public49_cache_train320k_reuses_feature_cache() -> None:
    text = _script("scripts/run_g005_idm_temporal_raw96_patch_axisclass_realvideo_statectx_public49_cache_train320k_target24k.sh")
    config = json.loads(
        (
            ROOT
            / "configs/model/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_statectx_public49_cache_train320k_target24k.yaml"
        ).read_text()
    )

    assert "cache-reuse" in text
    assert "statectx_public49_cache_train320k_target24k" in text
    assert config["action_mouse_tokenization"] == "fdm1_49_aggregate"
    assert config["fdm1_recipe"]["action_tokenization"]["mouse_delta_bins_per_axis"] == 49
    assert config["temporal_calibration_strategy"] == "tail"
    assert config["temporal_calibration_max_rows"] == 2000
    assert "temporal_calibration_family_quotas" not in config
    assert config["adaptive_family_budget_to_unlabeled_target"] is False
    assert config["distributed_feature_cache_dir"].endswith("/distributed_statectx_feature_cache")
    assert "cache_compatible_public49_training" in config["fdm1_recipe_alignment"]
    assert "Stratified" not in config["claim_boundary"]
    assert "feature cache" in config["claim_boundary"]

def test_g005_statectx_train320k_stratified_calibration_is_prediction_only() -> None:
    text = _script("scripts/run_g005_idm_temporal_raw96_statectx_train320k_predict24k.sh")
    noadapt = json.loads(
        (
            ROOT
            / "configs/model/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_statectx_train320k_stratcal_noadapt_predict24k.yaml"
        ).read_text()
    )
    adapt = json.loads(
        (
            ROOT
            / "configs/model/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_statectx_train320k_stratcal_adapt_relaxed_predict24k.yaml"
        ).read_text()
    )

    assert "scripts/predict_idm_temporal_masked_diffusion.py" in text
    assert "scripts/log_wandb_artifacts.py" in text
    assert "prediction-calibration" in text
    assert "paper_target_pass" in text
    assert "torchrun" not in text
    assert "no target-label calibration" in text
    for config in [noadapt, adapt]:
        assert config["source_checkpoint"] == (
            "outputs/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_statectx_train320k_target24k/checkpoint.pt"
        )
        assert config["train_records"] == "outputs/data/d2e_event_state_duration_realvideo_balanced_train320k_target24k/train_core.jsonl"
        assert config["target_records"] == "outputs/data/d2e_event_state_duration_realvideo_balanced_train320k_target24k/target_all_eval.jsonl"
        assert config["max_train_rows"] == 320000
        assert config["max_target_rows"] == 24000
        assert config["temporal_calibration_strategy"] == "stratified_action"
        assert config["temporal_calibration_family_quotas"] == {
            "keyboard": 600,
            "mouse_button": 600,
            "mouse_move": 600,
            "noop": 600,
        }
        assert config["candidate_score_reranker_enabled"] is False
        assert config["family_non_noop_budget_mouse_button_max_no_button_fpr"] == 0.10
        assert "calibration_sweep" in config["fdm1_recipe_alignment"]
        assert "target-label calibration" in config["claim_boundary"]
    assert noadapt["adaptive_family_budget_to_unlabeled_target"] is False
    assert adapt["adaptive_family_budget_to_unlabeled_target"] is True
    assert adapt["adaptive_family_budget_only_raise_threshold"] is False


def test_g005_statectx_mouseprior_keyadapt_predict24k_is_split_safe_prediction_only() -> None:
    text = _script("scripts/run_g005_idm_temporal_raw96_statectx_mouseprior_keyadapt_predict24k.sh")
    config = json.loads(
        (
            ROOT
            / "configs/model/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_statectx_train320k_mouseprior_keyadapt_predict24k.yaml"
        ).read_text()
    )

    assert "scripts/run_g005_idm_temporal_raw96_statectx_train320k_predict24k.sh" in text
    assert "train-fit-mouse-prior" in text
    assert "torchrun" not in text
    assert config["source_checkpoint"] == (
        "outputs/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_statectx_train320k_target24k/checkpoint.pt"
    )
    assert config["candidate_token_prior_correction"] is True
    assert config["candidate_token_prior_families"] == ["mouse_move"]
    assert config["direct_auxiliary_candidate_apply_token_prior"] is True
    assert config["adaptive_family_budget_to_unlabeled_target"] is True
    assert config["adaptive_family_budget_families"] == ["keyboard", "mouse_move"]
    assert config["adaptive_family_budget_only_raise_threshold"] is False
    assert config["family_non_noop_budget_keyboard_max_tokens_per_row"] == 3
    assert "No target labels are used for calibration" in config["claim_boundary"]
    assert "not completion evidence" in config["claim_boundary"]


def test_g005_statectx_mouseprior_noadapt_predict24k_keeps_train_only_calibration() -> None:
    text = _script("scripts/run_g005_idm_temporal_raw96_statectx_mouseprior_noadapt_predict24k.sh")
    config = json.loads(
        (
            ROOT
            / "configs/model/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_statectx_train320k_mouseprior_noadapt_predict24k.yaml"
        ).read_text()
    )

    assert "scripts/run_g005_idm_temporal_raw96_statectx_train320k_predict24k.sh" in text
    assert "no-target-adapt" in text
    assert "single-process-cache-write" in text
    assert "torchrun" not in text
    assert config["candidate_token_prior_correction"] is True
    assert config["candidate_token_prior_families"] == ["mouse_move"]
    assert config["candidate_token_prior_strength"] == 0.35
    assert config["direct_auxiliary_candidate_apply_token_prior"] is False
    assert config["adaptive_family_budget_to_unlabeled_target"] is False
    assert config["adaptive_family_budget_only_raise_threshold"] is True
    assert config["write_single_process_feature_cache"] is True
    assert "train_fit_mouse_move_prior_no_target_adapt" in config["fdm1_recipe_alignment"]
    assert "target labels" in config["claim_boundary"]
    assert "unlabeled-target rate adaptation" in config["claim_boundary"]


def test_g005_statectx_keyrerank_blend_predict24k_is_train_heldout_only() -> None:
    text = _script("scripts/run_g005_idm_temporal_raw96_statectx_keyrerank_blend025_predict24k.sh")
    config = json.loads(
        (
            ROOT
            / "configs/model/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_statectx_train320k_keyrerank_blend025_predict24k.yaml"
        ).read_text()
    )

    assert "scripts/run_g005_idm_temporal_raw96_statectx_train320k_predict24k.sh" in text
    assert "keyboard-reranker" in text
    assert "no-target-adapt" in text
    assert "torchrun" not in text
    assert config["source_checkpoint"] == (
        "outputs/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_statectx_train320k_target24k/checkpoint.pt"
    )
    assert config["candidate_score_reranker_enabled"] is True
    assert config["candidate_score_reranker_families"] == ["keyboard"]
    assert config["candidate_score_reranker_blend"] == 0.25
    assert config["adaptive_family_budget_to_unlabeled_target"] is False
    assert config["candidate_token_prior_correction"] is False
    assert config["write_single_process_feature_cache_splits"] == ["calibration", "target"]
    assert "train_heldout_keyboard_reranker" in config["fdm1_recipe_alignment"]
    assert "held-out train rows" in config["claim_boundary"]
    assert "no target-label calibration" in config["claim_boundary"]


def test_g005_statectx_keyrerank_fast256_preserves_recipe_and_adds_progress() -> None:
    text = _script("scripts/run_g005_idm_temporal_raw96_statectx_keyrerank_blend025_fast256_predict24k.sh")
    config = json.loads(
        (
            ROOT
            / "configs/model/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_statectx_train320k_keyrerank_blend025_fast256_predict24k.yaml"
        ).read_text()
    )

    assert "scripts/run_g005_idm_temporal_raw96_statectx_train320k_predict24k.sh" in text
    assert "keyboard-reranker" in text
    assert "fast256" in text
    assert config["candidate_score_reranker_enabled"] is True
    assert config["candidate_score_reranker_families"] == ["keyboard"]
    assert config["adaptive_family_budget_to_unlabeled_target"] is False
    assert config["prediction_batch_size"] == 256
    assert config["prediction_autocast_dtype"] == "bfloat16"
    assert config["vectorized_iterative_unmasking"] is True
    assert config["precompute_video_cache_features_as_tensor"] is True
    assert config["prediction_progress_every_batches"] == 1
    assert "held-out train rows" in config["claim_boundary"]
    assert "no target-label calibration" in config["claim_boundary"]


def test_g005_statectx_source_m101_fast256_uses_noncausal_offset_candidates() -> None:
    text = _script("scripts/run_g005_idm_temporal_raw96_statectx_source_m101_fast256_predict24k.sh")
    config = json.loads(
        (
            ROOT
            / "configs/model/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_statectx_train320k_source_m101_fast256_predict24k.yaml"
        ).read_text()
    )

    assert "scripts/run_g005_idm_temporal_raw96_statectx_train320k_predict24k.sh" in text
    assert "temporal-source-offsets" in text
    assert config["temporal_candidate_source_offsets"] == [-1, 0, 1]
    assert config["temporal_candidate_source_offset_weights"] == {"-1": 0.85, "0": 1.0, "1": 0.85}
    assert config["adaptive_family_budget_to_unlabeled_target"] is False
    assert config["candidate_token_prior_correction"] is True
    assert config["prediction_batch_size"] == 256
    assert config["prediction_autocast_dtype"] == "bfloat16"
    assert config["vectorized_iterative_unmasking"] is True
    assert config["precompute_video_cache_features_as_tensor"] is True
    assert "temporal_source_offset_candidate_aggregation" in config["fdm1_recipe_alignment"]
    assert "target labels" in config["claim_boundary"]


def test_g005_statectx_teacher_motiondistill_runs_train_teacher_before_masked_student() -> None:
    text = _script("scripts/run_g005_idm_temporal_raw96_statectx_teacher_motiondistill_train320k.sh")
    config = json.loads(
        (
            ROOT
            / "configs/model/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_statectx_teacher_motiondistill_train320k_target24k.yaml"
        ).read_text()
    )
    teacher = json.loads(
        (
            ROOT
            / "configs/model/idm_streaming_d2e_full_event_state_duration_context_teacher_train320k_predict.yaml"
        ).read_text()
    )

    materialize_idx = text.index("scripts/materialize_balanced_prefix.py")
    teacher_idx = text.index("scripts/predict_idm_streaming.py")
    train_idx = text.index("scripts/run_g005_idm_temporal_raw96_family_presence_prefix.sh")
    assert materialize_idx < teacher_idx < train_idx
    assert "teacher-motion-distill" in text
    assert config["teacher_distillation_enabled"] is True
    assert config["teacher_distillation_families"] == ["mouse_move"]
    assert config["teacher_distillation_aux_weight"] == 0.4
    assert config["teacher_prediction_paths"] == [
        "outputs/idm_streaming_d2e_full_event_state_duration_context_teacher_train320k_predictions/predictions.jsonl"
    ]
    assert (
        config["distributed_feature_cache_dir"]
        == "outputs/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_statectx_train320k_target24k/distributed_statectx_feature_cache"
    )
    assert config["teacher_distillation_reuses_statectx_feature_cache"] is True
    assert "train_teacher_motion_distillation" in config["fdm1_recipe_alignment"]
    assert "statectx_feature_cache_reuse" in config["fdm1_recipe_alignment"]
    assert "Target labels are never used" in config["claim_boundary"]
    assert teacher["records_path"] == "outputs/data/d2e_event_state_duration_realvideo_balanced_train320k_target24k/train_core.jsonl"
    assert teacher["checkpoint_path"] == "outputs/idm_streaming_d2e_full_event_state_duration_context_paper_target/checkpoint.pt"
    assert teacher["claim_boundary"].startswith("Prediction-only train-row teacher artifact")


def test_g005_statectx_warm_teacher_motiondistill_warm_starts_best_checkpoint() -> None:
    text = _script("scripts/run_g005_idm_temporal_raw96_statectx_warm_teacher_motiondistill_train320k.sh")
    config = json.loads(
        (
            ROOT
            / "configs/model/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_statectx_warm_teacher_motiondistill_train320k_target24k.yaml"
        ).read_text()
    )

    assert "run_g005_idm_temporal_raw96_statectx_teacher_motiondistill_train320k.sh" in text
    assert "warm-start" in text
    assert config["source_checkpoint"] == (
        "outputs/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_statectx_train320k_target24k/checkpoint.pt"
    )
    assert config["source_checkpoint_strict"] is True
    assert config["source_checkpoint_skip_video_pretrain"] is True
    assert config["teacher_distillation_enabled"] is True
    assert config["teacher_distillation_families"] == ["mouse_move"]
    assert config["teacher_distillation_aux_weight"] == 0.08
    assert config["teacher_distillation_extend_vocab"] is False
    assert config["epochs"] == 2
    assert config["lr"] == 5e-5
    assert "source_checkpoint_warm_start" in config["fdm1_recipe_alignment"]
    assert "prediction shortcut" in config["claim_boundary"]
    assert "target-label calibration" in config["claim_boundary"]


def test_g005_compact_luma_window5_materializes_nep_context_before_training() -> None:
    text = _script("scripts/run_g005_idm_compact_luma_window5_4xh200.sh")
    config = json.loads((ROOT / "configs/model/idm_streaming_d2e_full_compact_luma_window5_paper_target.yaml").read_text())
    paper = json.loads((ROOT / "configs/eval/g005_idm_compact_luma_window5_paper_target.yaml").read_text())

    materialize_idx = text.index("scripts/materialize_d2e_luma_window_corpus.py")
    cache_idx = text.index("scripts/precompute_streaming_idm_training_cache.py")
    train_idx = text.index("scripts/run_g005_idm_surface_paper_target_4xh200.sh")
    assert materialize_idx < cache_idx < train_idx
    assert "--offsets=\"${WINDOW_OFFSETS:--2,-1,0,1,2}\"" in text
    assert "outputs/data/d2e_luma_window5_corpus_shards_accel64" in text
    assert 'WANDB_TAGS="${WANDB_TAGS:-g005,idm,d2e,compact-luma-window5,pipeline}"' in text
    assert config["feature_mode"] == "summary_compact_luma16_window5_time"
    assert config["model_arch"] == "luma_temporal_conv"
    assert paper["paper_metrics"]["empty_bins_as_correct"] is False
    assert paper["paper_metrics"]["target_path"] == "outputs/data/d2e_full_corpus_shards_accel64/shard_*/target_all_eval.jsonl"


def test_g005_compact_luma_window5_residual_reuses_materialization_and_targets_original_records() -> None:
    text = _script("scripts/run_g005_idm_compact_luma_window5_residual_4xh200.sh")
    config = json.loads((ROOT / "configs/model/idm_streaming_d2e_full_compact_luma_window5_residual_paper_target.yaml").read_text())
    paper = json.loads((ROOT / "configs/eval/g005_idm_compact_luma_window5_residual_paper_target.yaml").read_text())
    split = json.loads((ROOT / "configs/eval/g005_idm_compact_luma_window5_residual_split_statistics.yaml").read_text())

    assert "scripts/run_g005_idm_compact_luma_window5_4xh200.sh" in text
    assert "g005_idm_compact_luma_window5_residual_precomputed_cache_validation.json" in text
    assert "compact-luma-window5,residual" in text
    assert config["model_arch"] == "luma_action_sequence_prior"
    assert config["mouse_target_mode"] == "residual_last_seen"
    assert config["mouse_head_mode"] == "axis_softmax"
    assert config["action_history_len"] == 4
    assert config["action_history_parallel_by_path"] is True
    assert paper["paper_metrics"]["target_path"] == "outputs/data/d2e_full_corpus_shards_accel64/shard_*/target_all_eval.jsonl"
    assert split["ground_truth_glob"] == "outputs/data/d2e_full_corpus_shards_accel64/shard_*/target_all_eval.jsonl"
    assert split["train_stats_path"] == "outputs/idm_streaming_d2e_full_compact_luma_window5_residual_paper_target/streaming_stats.json"


def test_g005_event_state_duration_context_uses_distinct_context_and_per_axis_gain() -> None:
    text = _script("scripts/run_g005_idm_event_state_duration_context_4xh200.sh")
    config = json.loads((ROOT / "configs/model/idm_streaming_d2e_full_event_state_duration_context_paper_target.yaml").read_text())
    paper = json.loads((ROOT / "configs/eval/g005_idm_event_state_duration_context_paper_target.yaml").read_text())

    assert "scripts/materialize_d2e_event_state_context_corpus.py" in text
    assert "outputs/data/d2e_event_state_duration_context_shards_accel64" in text
    assert "event-state-duration-context" in text
    assert config["feature_mode"] == "summary_compact_luma16_pair_shift_time_state_duration_prior_action"
    assert config["mouse_output_gain_mode"] == "train_abs_ratio_per_axis"
    assert config["state_duration_feature_dim"] == 80
    assert paper["paper_metrics"]["empty_bins_as_correct"] is False
    assert paper["paper_metrics"]["target_path"] == "outputs/data/d2e_event_state_duration_context_shards_accel64/shard_*/target_all_eval.jsonl"


def test_g005_context_dropout_prefix_logs_gpu_and_wandb_sidecar() -> None:
    text = _script("scripts/run_g005_idm_event_state_duration_context_dropout035_closed_loop_prefix.sh")
    config = json.loads(
        (
            ROOT
            / "configs/model/idm_streaming_d2e_full_event_state_duration_context_dropout035_closed_loop_prefix320k.yaml"
        ).read_text()
    )

    materialize_idx = text.index("scripts/materialize_state_context_dropout_train.py")
    wandb_idx = text.index("uv run --with wandb python scripts/watch_wandb_training.py")
    train_idx = text.index("scripts/train_idm_streaming.py")
    metrics_idx = text.index("scripts/build_g005_idm_paper_metrics.py")
    assert materialize_idx < wandb_idx < train_idx < metrics_idx
    assert 'ENABLE_WANDB_SIDECAR="${ENABLE_WANDB_SIDECAR:-1}"' in text
    assert "g005_idm_event_state_duration_context_dropout035_closed_loop_prefix320k_gpu_monitor.csv" in text
    assert "g005_context_dropout_prefix_wandb_sidecar.pid" in text
    assert "MLXP_RESERVATION_END_AT" in text
    assert config["closed_loop_state_context"] is True
    assert config["state_context_source"] == "predicted_closed_loop"
    assert config["max_train_examples"] == 320000
    assert config["max_target_examples"] == 320000


def test_g005_hierarchical_prefix_uses_exactset_heads_and_wandb_sidecar() -> None:
    text = _script("scripts/run_g005_idm_event_state_duration_hierarchical_prefix.sh")
    config = json.loads(
        (
            ROOT
            / "configs/model/idm_streaming_d2e_full_event_state_duration_hierarchical_prefix320k.yaml"
        ).read_text()
    )
    paper = json.loads(
        (
            ROOT
            / "configs/eval/g005_idm_event_state_duration_hierarchical_prefix320k_paper_metrics.yaml"
        ).read_text()
    )

    train_materialize_idx = text.index("g005_event_state_duration_hierarchical_train_prefix320k")
    target_materialize_idx = text.index("g005_event_state_duration_hierarchical_target_prefix320k")
    wandb_idx = text.index("uv run --with wandb python scripts/watch_wandb_training.py")
    train_idx = text.index("scripts/train_idm_streaming.py")
    metrics_idx = text.index("scripts/build_g005_idm_paper_metrics.py")
    assert train_materialize_idx < target_materialize_idx < wandb_idx < train_idx < metrics_idx
    assert 'ENABLE_WANDB_SIDECAR="${ENABLE_WANDB_SIDECAR:-1}"' in text
    assert "g005_idm_event_state_duration_hierarchical_prefix320k_gpu_monitor.csv" in text
    assert "g005_hierarchical_prefix_wandb_sidecar.pid" in text
    assert config["keyboard_head_mode"] == "hierarchical_softmax"
    assert config["button_head_mode"] == "hierarchical_softmax"
    assert config["feature_mode"] == "summary_compact_luma16_pair_shift_time_state_duration_prior_action"
    assert config["training_cache_shard_by_path"] is False
    assert paper["empty_bins_as_correct"] is False
    assert paper["target_paths"] == ["outputs/data/d2e_event_state_duration_hierarchical_prefix320k/target_all_eval.jsonl"]


def test_g005_sequence_prior_prefix_uses_causal_action_history_and_wandb_sidecar() -> None:
    text = _script("scripts/run_g005_idm_event_state_duration_sequence_prior_prefix.sh")
    config = json.loads(
        (
            ROOT
            / "configs/model/idm_streaming_d2e_full_event_state_duration_sequence_prior_prefix320k.yaml"
        ).read_text()
    )
    paper = json.loads(
        (
            ROOT
            / "configs/eval/g005_idm_event_state_duration_sequence_prior_prefix320k_paper_metrics.yaml"
        ).read_text()
    )

    train_materialize_idx = text.index("g005_event_state_duration_sequence_prior_train_prefix320k")
    target_materialize_idx = text.index("g005_event_state_duration_sequence_prior_target_prefix320k")
    wandb_idx = text.index("uv run --with wandb python scripts/watch_wandb_training.py")
    train_idx = text.index("scripts/train_idm_streaming.py")
    metrics_idx = text.index("scripts/build_g005_idm_paper_metrics.py")
    assert train_materialize_idx < target_materialize_idx < wandb_idx < train_idx < metrics_idx
    assert 'ENABLE_WANDB_SIDECAR="${ENABLE_WANDB_SIDECAR:-1}"' in text
    assert "g005_idm_event_state_duration_sequence_prior_prefix320k_gpu_monitor.csv" in text
    assert "g005_sequence_prior_prefix_wandb_sidecar.pid" in text
    assert "MLXP_RESERVATION_END_AT" in text
    assert "TRAIN_PREFIX_REUSE_SUMMARY" in text
    assert "TARGET_PREFIX_REUSE_SUMMARY" in text
    assert config["model_arch"] == "luma_action_sequence_prior"
    assert config["action_history_len"] == 8
    assert config["keyboard_head_mode"] == "multilabel"
    assert config["button_head_mode"] == "multilabel"
    assert config["feature_mode"] == "summary_compact_luma16_pair_shift_time_state_duration_prior_action"
    assert config["training_cache_shard_by_path"] is False
    assert paper["empty_bins_as_correct"] is False
    assert paper["target_paths"] == ["outputs/data/d2e_event_state_duration_hierarchical_prefix320k/target_all_eval.jsonl"]


def test_g005_event_state_duration_luma_window5_combines_nep_window_and_state_context() -> None:
    text = _script("scripts/run_g005_idm_event_state_duration_luma_window5_4xh200.sh")
    config = json.loads(
        (ROOT / "configs/model/idm_streaming_d2e_full_event_state_duration_luma_window5_paper_target.yaml").read_text()
    )
    paper = json.loads((ROOT / "configs/eval/g005_idm_event_state_duration_luma_window5_paper_target.yaml").read_text())
    split = json.loads((ROOT / "configs/eval/g005_idm_event_state_duration_luma_window5_split_statistics.yaml").read_text())

    context_idx = text.index("scripts/materialize_d2e_event_state_context_corpus.py")
    window_idx = text.index("scripts/materialize_d2e_luma_window_corpus.py")
    cache_idx = text.index("scripts/precompute_streaming_idm_training_cache.py")
    train_idx = text.index("scripts/run_g005_idm_surface_paper_target_4xh200.sh")
    assert context_idx < window_idx < cache_idx < train_idx
    assert "--offsets=\"${WINDOW_OFFSETS:-0,1,2,3,4}\"" in text
    assert "outputs/data/d2e_event_state_duration_context_shards_accel64" in text
    assert "outputs/data/d2e_event_state_duration_luma_window5_shards_accel64" in text
    assert "event-state-duration-luma-window5" in text
    assert config["feature_mode"] == "summary_compact_luma16_window5_time_state_duration_prior_action"
    assert config["visual_stack_frames"] == 5
    assert config["state_duration_feature_dim"] == 80
    assert config["previous_event_feature_dim"] == 38
    assert paper["paper_metrics"]["empty_bins_as_correct"] is False
    assert paper["paper_metrics"]["target_path"] == "outputs/data/d2e_event_state_duration_luma_window5_shards_accel64/shard_*/target_all_eval.jsonl"
    assert split["ground_truth_glob"] == "outputs/data/d2e_event_state_duration_luma_window5_shards_accel64/shard_*/target_all_eval.jsonl"



def test_g005_chrono_closed_loop_prefix_materializes_before_prediction() -> None:
    text = _script("scripts/run_g005_idm_event_state_duration_context_chrono_closed_loop_prefix.sh")
    config = json.loads((ROOT / "configs/model/idm_streaming_d2e_full_event_state_duration_context_chrono_closed_loop_prefix320k_predict.yaml").read_text())
    paper = json.loads((ROOT / "configs/eval/g005_idm_event_state_duration_context_chrono_closed_loop_prefix320k_paper_metrics.yaml").read_text())

    materialize_idx = text.index("scripts/materialize_chronological_prefix.py")
    predict_idx = text.index("scripts/predict_idm_streaming.py")
    metrics_idx = text.index("scripts/build_g005_idm_paper_metrics.py")
    assert materialize_idx < predict_idx < metrics_idx
    assert "uv run --extra train python scripts/predict_idm_streaming.py" in text
    assert "uv run --extra train --with wandb python scripts/log_wandb_artifacts.py" in text
    assert '--json "$CHRONO_SUMMARY"' in text
    assert '--json "artifacts/idm/g005_idm_event_state_duration_context_chrono_closed_loop_prefix320k_paper_metrics.json"' in text
    assert "d2e_event_state_duration_context_chrono_prefix320k" in text
    assert config["closed_loop_state_context"] is True
    assert config["closed_loop_state_context_seed_from_train"] is False
    assert config["records_path"] == "outputs/data/d2e_event_state_duration_context_chrono_prefix320k/target_all_eval.jsonl"
    assert paper["target_paths"] == ["outputs/data/d2e_event_state_duration_context_chrono_prefix320k/target_all_eval.jsonl"]
    assert paper["output_path"] == "artifacts/idm/g005_idm_event_state_duration_context_chrono_closed_loop_prefix320k_paper_metrics.json"


def test_g005_endpoint_mixture_prefix_wrapper_logs_json_artifacts() -> None:
    text = _script("scripts/run_g005_idm_endpoint_mixture_prefix.sh")
    ensemble = json.loads((ROOT / "configs/eval/g005_idm_endpoint_mixture_state_luma_gate_context_prefix320k.yaml").read_text())
    paper = json.loads((ROOT / "configs/eval/g005_idm_endpoint_mixture_state_luma_gate_context_prefix320k_paper_metrics.yaml").read_text())

    ensemble_idx = text.index("scripts/ensemble_idm_predictions.py")
    metrics_idx = text.index("scripts/build_g005_idm_paper_metrics.py")
    assert ensemble_idx < metrics_idx
    assert '--json "artifacts/idm/g005_idm_endpoint_mixture_state_luma_gate_context_prefix320k_summary.json"' in text
    assert '--json "artifacts/idm/g005_idm_endpoint_mixture_state_luma_gate_context_prefix320k_paper_metrics.json"' in text
    assert ensemble["policies"]["button"]["mode"] == "source_with_endpoint_gate"
    assert ensemble["policies"]["button"]["source"] == "state_luma_pair"
    assert ensemble["policies"]["button"]["gate_sources"] == ["event_state_duration_context"]
    assert paper["max_rows"] == 320000
    assert paper["empty_bins_as_correct"] is False


def test_g005_luma_window_nep100_prefix_wrapper_materializes_future_offsets_and_train_extra() -> None:
    text = _script("scripts/run_g005_idm_event_state_duration_luma_window_nep100_prefix.sh")
    config = json.loads(
        (
            ROOT
            / "configs/model/idm_streaming_d2e_full_event_state_duration_luma_window_nep100_prefix320k.yaml"
        ).read_text()
    )
    paper = json.loads(
        (
            ROOT
            / "configs/eval/g005_idm_event_state_duration_luma_window_nep100_prefix320k_paper_metrics.yaml"
        ).read_text()
    )

    materialize_idx = text.index("scripts/materialize_luma_window_prefix.py")
    wandb_idx = text.index("uv run --with wandb python scripts/watch_wandb_training.py")
    train_idx = text.index("uv run --extra train torchrun")
    metrics_idx = text.index("uv run --extra train python scripts/build_g005_idm_paper_metrics.py")
    assert materialize_idx < wandb_idx < train_idx < metrics_idx
    assert 'WINDOW_OFFSETS="${WINDOW_OFFSETS:-0,2,4,6,8}"' in text
    assert "--offsets \"$WINDOW_OFFSETS\"" in text
    assert "outputs/data/d2e_event_state_duration_luma_window_nep100_prefix320k" in text
    assert "g005-luma-window-nep100-prefix320k" in text
    assert config["feature_mode"] == "summary_compact_luma16_window5_time_state_duration_prior_action"
    assert config["model_arch"] == "luma_temporal_conv"
    assert config["max_train_examples"] == 320000
    assert config["max_target_examples"] == 320000
    assert config["prediction_cuda_devices"] == [0]
    assert paper["max_rows"] == 320000
    assert paper["empty_bins_as_correct"] is False
    assert paper["target_paths"] == ["outputs/data/d2e_event_state_duration_luma_window_nep100_prefix320k/target_all_eval.jsonl"]


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
