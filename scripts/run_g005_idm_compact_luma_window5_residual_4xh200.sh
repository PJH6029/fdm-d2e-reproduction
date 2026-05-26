#!/usr/bin/env bash
set -euo pipefail

export CONFIG="${CONFIG:-configs/model/idm_streaming_d2e_full_compact_luma_window5_residual_paper_target.yaml}"
export MODEL_SLUG="${MODEL_SLUG:-g005_idm_compact_luma_window5_residual}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/idm_streaming_d2e_full_compact_luma_window5_residual_paper_target}"
export PRECOMPUTE_CACHE_VALIDATION="${PRECOMPUTE_CACHE_VALIDATION:-artifacts/idm/g005_idm_compact_luma_window5_residual_precomputed_cache_validation.json}"
export PRECOMPUTE_CACHE_PROGRESS="${PRECOMPUTE_CACHE_PROGRESS:-artifacts/idm/g005_idm_compact_luma_window5_residual_precompute_progress.json}"
export PRECOMPUTE_CACHE_LOG="${PRECOMPUTE_CACHE_LOG:-artifacts/idm/g005_idm_compact_luma_window5_residual_precompute.log}"
export PRECOMPUTE_CACHE_WANDB_STATUS="${PRECOMPUTE_CACHE_WANDB_STATUS:-artifacts/idm/g005_idm_compact_luma_window5_residual_precompute_wandb_status.json}"
export SPLIT_STATS_CONFIG="${SPLIT_STATS_CONFIG:-configs/eval/g005_idm_compact_luma_window5_residual_split_statistics.yaml}"
export PAPER_TARGET_CONFIG="${PAPER_TARGET_CONFIG:-configs/eval/g005_idm_compact_luma_window5_residual_paper_target.yaml}"
export LOG_PATH="${LOG_PATH:-artifacts/idm/g005_idm_compact_luma_window5_residual_4xh200.log}"
export RUN_SUMMARY="${RUN_SUMMARY:-artifacts/idm/g005_idm_compact_luma_window5_residual_4xh200_run.json}"
export GPU_MONITOR_LOG="${GPU_MONITOR_LOG:-artifacts/idm/g005_idm_compact_luma_window5_residual_4xh200_gpu_monitor.csv}"
export PID_FILE="${PID_FILE:-outputs/cluster/g005_idm_compact_luma_window5_residual_4xh200.pid}"
export GPU_SMOKE_REPORT="${GPU_SMOKE_REPORT:-outputs/cluster/g005_idm_compact_luma_window5_residual_gpu_smoke.json}"
export WANDB_SIDECAR_STATUS="${WANDB_SIDECAR_STATUS:-artifacts/idm/g005_idm_compact_luma_window5_residual_wandb_sidecar_status.json}"
export WANDB_SIDECAR_LOG="${WANDB_SIDECAR_LOG:-artifacts/idm/g005_idm_compact_luma_window5_residual_wandb_sidecar.log}"
export WANDB_SIDECAR_PID_FILE="${WANDB_SIDECAR_PID_FILE:-outputs/cluster/g005_idm_compact_luma_window5_residual_wandb_sidecar.pid}"
export WANDB_TAGS="${WANDB_TAGS:-g005,idm,d2e,compact-luma-window5,residual,pipeline}"
export WANDB_SIDECAR_TAGS="${WANDB_SIDECAR_TAGS:-g005,idm,d2e,compact-luma-window5,residual,4xh200,sidecar}"
export WANDB_PROCESS_PATTERN="${WANDB_PROCESS_PATTERN:-train_idm_streaming|torchrun|run_g005_idm_compact_luma_window5_residual|run_g005_idm_compact_luma_window5}"

exec scripts/run_g005_idm_compact_luma_window5_4xh200.sh
