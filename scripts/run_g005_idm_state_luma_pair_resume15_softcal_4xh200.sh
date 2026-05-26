#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/model/idm_streaming_d2e_full_state_luma_pair_resume15_softcal_paper_target.yaml}" \
MODEL_SLUG="${MODEL_SLUG:-g005_idm_state_luma_pair_resume15_softcal}" \
OUTPUT_DIR="${OUTPUT_DIR:-outputs/idm_streaming_d2e_full_state_luma_pair_resume15_softcal_paper_target}" \
STATE_STATS_SYNTHESIS_SUMMARY="${STATE_STATS_SYNTHESIS_SUMMARY:-artifacts/idm/g005_idm_state_luma_pair_resume15_softcal_stats_synthesis_summary.json}" \
STATE_STATS_SYNTHESIS_LOG="${STATE_STATS_SYNTHESIS_LOG:-artifacts/idm/g005_idm_state_luma_pair_resume15_softcal_stats_synthesis.log}" \
STATE_STATS_SYNTHESIS_WANDB_STATUS="${STATE_STATS_SYNTHESIS_WANDB_STATUS:-artifacts/idm/g005_idm_state_luma_pair_resume15_softcal_stats_synthesis_wandb_status.json}" \
PRECOMPUTE_CACHE_VALIDATION="${PRECOMPUTE_CACHE_VALIDATION:-artifacts/idm/g005_idm_state_luma_pair_precomputed_cache_validation.json}" \
PRECOMPUTE_CACHE_PROGRESS="${PRECOMPUTE_CACHE_PROGRESS:-artifacts/idm/g005_idm_state_luma_pair_resume15_softcal_precompute_progress.json}" \
PRECOMPUTE_CACHE_LOG="${PRECOMPUTE_CACHE_LOG:-artifacts/idm/g005_idm_state_luma_pair_resume15_softcal_precompute.log}" \
PRECOMPUTE_CACHE_WANDB_STATUS="${PRECOMPUTE_CACHE_WANDB_STATUS:-artifacts/idm/g005_idm_state_luma_pair_resume15_softcal_precompute_wandb_status.json}" \
SPLIT_STATS_CONFIG="${SPLIT_STATS_CONFIG:-configs/eval/g005_idm_state_luma_pair_resume15_softcal_split_statistics.yaml}" \
PAPER_TARGET_CONFIG="${PAPER_TARGET_CONFIG:-configs/eval/g005_idm_state_luma_pair_resume15_softcal_paper_target.yaml}" \
LOG_PATH="${LOG_PATH:-artifacts/idm/g005_idm_state_luma_pair_resume15_softcal_4xh200.log}" \
RUN_SUMMARY="${RUN_SUMMARY:-artifacts/idm/g005_idm_state_luma_pair_resume15_softcal_4xh200_run.json}" \
GPU_MONITOR_LOG="${GPU_MONITOR_LOG:-artifacts/idm/g005_idm_state_luma_pair_resume15_softcal_4xh200_gpu_monitor.csv}" \
PID_FILE="${PID_FILE:-outputs/cluster/g005_idm_state_luma_pair_resume15_softcal_4xh200.pid}" \
GPU_SMOKE_REPORT="${GPU_SMOKE_REPORT:-outputs/cluster/g005_idm_state_luma_pair_resume15_softcal_gpu_smoke.json}" \
WANDB_SIDECAR_STATUS="${WANDB_SIDECAR_STATUS:-artifacts/idm/g005_idm_state_luma_pair_resume15_softcal_wandb_sidecar_status.json}" \
WANDB_SIDECAR_LOG="${WANDB_SIDECAR_LOG:-artifacts/idm/g005_idm_state_luma_pair_resume15_softcal_wandb_sidecar.log}" \
WANDB_SIDECAR_PID_FILE="${WANDB_SIDECAR_PID_FILE:-outputs/cluster/g005_idm_state_luma_pair_resume15_softcal_wandb_sidecar.pid}" \
WANDB_TAGS="${WANDB_TAGS:-g005,idm,d2e,state-corpus,resume,softcal,pipeline}" \
WANDB_SIDECAR_TAGS="${WANDB_SIDECAR_TAGS:-g005,idm,d2e,state-corpus,resume,softcal,4xh200,sidecar}" \
WANDB_PROCESS_PATTERN="${WANDB_PROCESS_PATTERN:-train_idm_streaming|torchrun|run_g005_idm_state_luma_pair_resume15_softcal|run_g005_idm_state_luma_pair}" \
scripts/run_g005_idm_state_luma_pair_4xh200.sh
