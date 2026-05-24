#!/usr/bin/env bash
set -euo pipefail

export CONFIG="${CONFIG:-configs/model/idm_video_stack_d2e_full_luma96_offsets012_keysoftmax_paper_target.yaml}"
export MODEL_SLUG="${MODEL_SLUG:-g005_idm_video_stack_luma96_offsets012_keysoftmax}"
export LOG_PATH="${LOG_PATH:-artifacts/idm/g005_idm_video_stack_luma96_offsets012_keysoftmax_4xh200.log}"
export RUN_SUMMARY="${RUN_SUMMARY:-artifacts/idm/g005_idm_video_stack_luma96_offsets012_keysoftmax_4xh200_run.json}"
export GPU_MONITOR_LOG="${GPU_MONITOR_LOG:-artifacts/idm/g005_idm_video_stack_luma96_offsets012_keysoftmax_4xh200_gpu_monitor.csv}"
export PID_FILE="${PID_FILE:-outputs/cluster/g005_idm_video_stack_luma96_offsets012_keysoftmax_4xh200.pid}"
export GPU_SMOKE_REPORT="${GPU_SMOKE_REPORT:-outputs/cluster/g005_idm_video_stack_luma96_offsets012_keysoftmax_gpu_smoke.json}"
export SPLIT_STATS_CONFIG="${SPLIT_STATS_CONFIG:-configs/eval/g005_idm_video_stack_luma96_offsets012_keysoftmax_split_statistics.yaml}"
export PAPER_TARGET_CONFIG="${PAPER_TARGET_CONFIG:-configs/eval/g005_idm_video_stack_luma96_offsets012_keysoftmax_paper_target.yaml}"
export SKIP_PREDICTION="${SKIP_PREDICTION:-1}"
export BUILD_SPLIT_STATS="${BUILD_SPLIT_STATS:-0}"
export BUILD_PAPER_METRICS="${BUILD_PAPER_METRICS:-0}"
export VALIDATE_G005="${VALIDATE_G005:-0}"

exec bash scripts/run_g005_idm_video_pair_raw112_4xh200.sh
