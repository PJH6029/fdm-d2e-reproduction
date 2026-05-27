#!/usr/bin/env bash
set -euo pipefail

export CONFIG="${CONFIG:-configs/model/idm_video_pair_d2e_full_raw112_offset2_keysoftmax_paper_target.yaml}"
export MODEL_SLUG="${MODEL_SLUG:-g005_idm_video_pair_raw112_offset2_keysoftmax}"
export LOG_PATH="${LOG_PATH:-artifacts/idm/g005_idm_video_pair_raw112_offset2_keysoftmax_precompute.log}"
export PID_FILE="${PID_FILE:-outputs/cluster/g005_idm_video_pair_raw112_offset2_keysoftmax_precompute.pid}"
export RUN_SUMMARY="${RUN_SUMMARY:-artifacts/idm/g005_idm_video_pair_raw112_offset2_keysoftmax_precompute_run.json}"
exec scripts/run_g005_idm_video_stack_luma96_offsets012_precompute.sh
