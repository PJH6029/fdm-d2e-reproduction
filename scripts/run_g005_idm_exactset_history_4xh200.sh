#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/model/idm_streaming_d2e_full_luma_pair_exactset_history_paper_target.yaml}" \
MODEL_SLUG="${MODEL_SLUG:-g005_idm_exactset_history}" \
LOG_PATH="${LOG_PATH:-artifacts/idm/g005_idm_exactset_history_4xh200.log}" \
RUN_SUMMARY="${RUN_SUMMARY:-artifacts/idm/g005_idm_exactset_history_4xh200_run.json}" \
GPU_MONITOR_LOG="${GPU_MONITOR_LOG:-artifacts/idm/g005_idm_exactset_history_4xh200_gpu_monitor.csv}" \
PID_FILE="${PID_FILE:-outputs/cluster/g005_idm_exactset_history_4xh200.pid}" \
GPU_SMOKE_REPORT="${GPU_SMOKE_REPORT:-outputs/cluster/g005_idm_exactset_history_gpu_smoke.json}" \
RUN_CONFIG_RECORD="${RUN_CONFIG_RECORD:-outputs/cluster/g005_idm_exactset_history_runtime_config_path.txt}" \
RUNTIME_NO_CACHE_CONFIG="${RUNTIME_NO_CACHE_CONFIG:-outputs/cluster/g005_idm_exactset_history_runtime_no_cache.yaml}" \
SPLIT_STATS_CONFIG="${SPLIT_STATS_CONFIG:-configs/eval/g005_idm_exactset_history_split_statistics.yaml}" \
PAPER_TARGET_CONFIG="${PAPER_TARGET_CONFIG:-configs/eval/g005_idm_exactset_history_paper_target.yaml}" \
STATS_SEED_PATH="${STATS_SEED_PATH:-}" \
OUTPUT_DIR="${OUTPUT_DIR:-outputs/idm_streaming_d2e_full_luma_pair_exactset_history_paper_target}" \
PRESEED_STATS="${PRESEED_STATS:-0}" \
ALLOW_CACHE_BUILD="${ALLOW_CACHE_BUILD:-1}" \
BUILD_SPLIT_STATS="${BUILD_SPLIT_STATS:-1}" \
BUILD_PAPER_METRICS="${BUILD_PAPER_METRICS:-1}" \
VALIDATE_G005="${VALIDATE_G005:-1}" \
scripts/run_g005_idm_surface_paper_target_4xh200.sh
