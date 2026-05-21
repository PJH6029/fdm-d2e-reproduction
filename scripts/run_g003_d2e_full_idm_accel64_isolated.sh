#!/usr/bin/env bash
set -euo pipefail

# Isolated 64-shard full-D2E G003 fallback path. This intentionally avoids
# canonical G003 output/log/cache paths until an operator promotes the result.
export OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-accel64}"
export NUM_SHARDS="${NUM_SHARDS:-64}"
export SHARD_ROOT="${SHARD_ROOT:-outputs/data/d2e_full_corpus_shards_accel64}"
export LOG_DIR="${LOG_DIR:-artifacts/sources/g003_accel64}"
export DATA_OUTPUT_DIR="${DATA_OUTPUT_DIR:-outputs/data/d2e_full_corpus_accel64}"
export DECODE_SUMMARY="${DECODE_SUMMARY:-artifacts/sources/d2e_full_corpus_decode_summary_accel64.json}"
export IDM_CONFIG="${IDM_CONFIG:-configs/model/idm_streaming_d2e_full_compact_accel64.yaml}"
export IDM_SUMMARY="${IDM_SUMMARY:-artifacts/idm/idm_streaming_d2e_full_compact_accel64_summary.json}"
export CACHE_DIR="${CACHE_DIR:-/root/work/data/d2e/cache_accel64}"
export SPLIT_STATS_CONFIG="${SPLIT_STATS_CONFIG:-configs/eval/g003_split_statistics_accel64.yaml}"
export SPLIT_STATS_SUMMARY="${SPLIT_STATS_SUMMARY:-artifacts/eval/g003_split_statistical_comparisons_accel64_summary.json}"

exec bash scripts/run_g003_d2e_full_idm_parallel.sh
