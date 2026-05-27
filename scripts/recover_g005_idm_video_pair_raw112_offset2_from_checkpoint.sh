#!/usr/bin/env bash
set -euo pipefail

export CONFIG="${CONFIG:-configs/model/idm_video_pair_d2e_full_raw112_offset2_keysoftmax_paper_target.yaml}"
export MODEL_SLUG="${MODEL_SLUG:-g005_idm_video_pair_raw112_offset2_keysoftmax}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/idm_video_pair_d2e_full_raw112_offset2_keysoftmax_paper_target}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-$OUTPUT_DIR/checkpoint.pt}"
export PREDICTION_WORKERS="${PREDICTION_WORKERS:-4}"
export PREDICTION_CUDA_DEVICES="${PREDICTION_CUDA_DEVICES:-0,1,2,3}"
export PREDICTION_PARTS_DIR="${PREDICTION_PARTS_DIR:-$OUTPUT_DIR/prediction_parts}"
export PREDICTION_SUMMARY="${PREDICTION_SUMMARY:-artifacts/idm/g005_idm_video_pair_raw112_offset2_keysoftmax_prediction_summary.json}"
export RECOVERY_SUMMARY="${RECOVERY_SUMMARY:-artifacts/idm/g005_idm_video_pair_raw112_offset2_keysoftmax_checkpoint_recovery_run.json}"
export LOG_PATH="${LOG_PATH:-artifacts/idm/g005_idm_video_pair_raw112_offset2_keysoftmax_checkpoint_recovery.log}"
export PID_FILE="${PID_FILE:-outputs/cluster/g005_idm_video_pair_raw112_offset2_keysoftmax_checkpoint_recovery.pid}"
export PREDICTIONS_PATH="${PREDICTIONS_PATH:-$OUTPUT_DIR/predictions.jsonl}"
export PSEUDOLABELS_PATH="${PSEUDOLABELS_PATH:-$OUTPUT_DIR/pseudolabels.jsonl}"
export SPLIT_STATS_CONFIG="${SPLIT_STATS_CONFIG:-configs/eval/g005_idm_video_pair_raw112_offset2_keysoftmax_split_statistics.yaml}"
export PAPER_TARGET_CONFIG="${PAPER_TARGET_CONFIG:-configs/eval/g005_idm_video_pair_raw112_offset2_keysoftmax_paper_target.yaml}"
export PAPER_METRICS_PATH="${PAPER_METRICS_PATH:-artifacts/idm/g005_idm_video_pair_raw112_offset2_keysoftmax_paper_metrics.json}"
export PAPER_TARGET_AUDIT="${PAPER_TARGET_AUDIT:-artifacts/idm/g005_idm_video_pair_raw112_offset2_keysoftmax_paper_target_audit.json}"
export WANDB_PREDICTION_LOG="${WANDB_PREDICTION_LOG:-artifacts/idm/g005_idm_video_pair_raw112_offset2_keysoftmax_prediction_wandb_sidecar.log}"
export WANDB_PREDICTION_STATUS="${WANDB_PREDICTION_STATUS:-artifacts/idm/g005_idm_video_pair_raw112_offset2_keysoftmax_prediction_wandb_sidecar_status.json}"
export WANDB_PREDICTION_PID_FILE="${WANDB_PREDICTION_PID_FILE:-outputs/cluster/g005_idm_video_pair_raw112_offset2_keysoftmax_prediction_wandb_sidecar.pid}"
exec scripts/recover_g005_idm_video_stack_luma96_offsets012_from_checkpoint.sh
