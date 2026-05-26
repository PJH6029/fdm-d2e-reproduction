#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/model/idm_streaming_d2e_full_transition_hazard_paper_target.yaml}" \
MODEL_SLUG="${MODEL_SLUG:-g005_idm_transition_hazard}" \
OUTPUT_DIR="${OUTPUT_DIR:-outputs/idm_streaming_d2e_full_transition_hazard_paper_target}" \
PRECOMPUTE_CACHE_VALIDATION="${PRECOMPUTE_CACHE_VALIDATION:-artifacts/idm/g005_idm_transition_hazard_precomputed_cache_validation.json}" \
PRECOMPUTE_CACHE_PROGRESS="${PRECOMPUTE_CACHE_PROGRESS:-artifacts/idm/g005_idm_transition_hazard_precompute_progress.json}" \
PRECOMPUTE_CACHE_LOG="${PRECOMPUTE_CACHE_LOG:-artifacts/idm/g005_idm_transition_hazard_precompute.log}" \
PRECOMPUTE_CACHE_WANDB_STATUS="${PRECOMPUTE_CACHE_WANDB_STATUS:-artifacts/idm/g005_idm_transition_hazard_precompute_wandb_status.json}" \
SPLIT_STATS_CONFIG="${SPLIT_STATS_CONFIG:-configs/eval/g005_idm_transition_hazard_split_statistics.yaml}" \
PAPER_TARGET_CONFIG="${PAPER_TARGET_CONFIG:-configs/eval/g005_idm_transition_hazard_paper_target.yaml}" \
LOG_PATH="${LOG_PATH:-artifacts/idm/g005_idm_transition_hazard_4xh200.log}" \
RUN_SUMMARY="${RUN_SUMMARY:-artifacts/idm/g005_idm_transition_hazard_4xh200_run.json}" \
GPU_MONITOR_LOG="${GPU_MONITOR_LOG:-artifacts/idm/g005_idm_transition_hazard_4xh200_gpu_monitor.csv}" \
PID_FILE="${PID_FILE:-outputs/cluster/g005_idm_transition_hazard_4xh200.pid}" \
GPU_SMOKE_REPORT="${GPU_SMOKE_REPORT:-outputs/cluster/g005_idm_transition_hazard_gpu_smoke.json}" \
WANDB_SIDECAR_STATUS="${WANDB_SIDECAR_STATUS:-artifacts/idm/g005_idm_transition_hazard_wandb_sidecar_status.json}" \
WANDB_SIDECAR_LOG="${WANDB_SIDECAR_LOG:-artifacts/idm/g005_idm_transition_hazard_wandb_sidecar.log}" \
WANDB_SIDECAR_PID_FILE="${WANDB_SIDECAR_PID_FILE:-outputs/cluster/g005_idm_transition_hazard_wandb_sidecar.pid}" \
WANDB_TAGS="${WANDB_TAGS:-g005,idm,d2e,transition-hazard,pipeline}" \
WANDB_SIDECAR_TAGS="${WANDB_SIDECAR_TAGS:-g005,idm,d2e,transition-hazard,4xh200,sidecar}" \
WANDB_PROCESS_PATTERN="${WANDB_PROCESS_PATTERN:-train_idm_streaming|torchrun|run_g005_idm_transition_hazard|run_g005_idm_event_state_context}" \
scripts/run_g005_idm_event_state_context_4xh200.sh
