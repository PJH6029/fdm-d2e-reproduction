#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/model/idm_streaming_d2e_full_state_luma_pair_paper_target.yaml}"
MODEL_SLUG="${MODEL_SLUG:-g005_idm_state_luma_pair}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/idm_streaming_d2e_full_state_luma_pair_paper_target}"
STATE_INPUT_ROOT="${STATE_INPUT_ROOT:-outputs/data/d2e_full_corpus_shards_accel64}"
STATE_OUTPUT_ROOT="${STATE_OUTPUT_ROOT:-outputs/data/d2e_state_corpus_shards_accel64}"
STATE_SUMMARY="${STATE_SUMMARY:-artifacts/idm/g005_idm_state_corpus_materialization_summary.json}"
STATE_PROGRESS="${STATE_PROGRESS:-artifacts/idm/g005_idm_state_corpus_materialization_progress.json}"
STATE_LOG="${STATE_LOG:-artifacts/idm/g005_idm_state_corpus_materialization.log}"
PRECOMPUTE_CACHE_VALIDATION="${PRECOMPUTE_CACHE_VALIDATION:-artifacts/idm/g005_idm_state_luma_pair_precomputed_cache_validation.json}"
PRECOMPUTE_CACHE_PROGRESS="${PRECOMPUTE_CACHE_PROGRESS:-artifacts/idm/g005_idm_state_luma_pair_precompute_progress.json}"
PRECOMPUTE_CACHE_LOG="${PRECOMPUTE_CACHE_LOG:-artifacts/idm/g005_idm_state_luma_pair_precompute.log}"
SPLIT_STATS_CONFIG="${SPLIT_STATS_CONFIG:-configs/eval/g005_idm_state_luma_pair_split_statistics.yaml}"
PAPER_TARGET_CONFIG="${PAPER_TARGET_CONFIG:-configs/eval/g005_idm_state_luma_pair_paper_target.yaml}"
LOG_PATH="${LOG_PATH:-artifacts/idm/g005_idm_state_luma_pair_4xh200.log}"
RUN_SUMMARY="${RUN_SUMMARY:-artifacts/idm/g005_idm_state_luma_pair_4xh200_run.json}"
GPU_MONITOR_LOG="${GPU_MONITOR_LOG:-artifacts/idm/g005_idm_state_luma_pair_4xh200_gpu_monitor.csv}"
PID_FILE="${PID_FILE:-outputs/cluster/g005_idm_state_luma_pair_4xh200.pid}"
GPU_SMOKE_REPORT="${GPU_SMOKE_REPORT:-outputs/cluster/g005_idm_state_luma_pair_gpu_smoke.json}"
WANDB_SIDECAR_STATUS="${WANDB_SIDECAR_STATUS:-artifacts/idm/g005_idm_state_luma_pair_wandb_sidecar_status.json}"
WANDB_SIDECAR_LOG="${WANDB_SIDECAR_LOG:-artifacts/idm/g005_idm_state_luma_pair_wandb_sidecar.log}"
WANDB_SIDECAR_PID_FILE="${WANDB_SIDECAR_PID_FILE:-outputs/cluster/g005_idm_state_luma_pair_wandb_sidecar.pid}"
ENABLE_WANDB_SIDECAR="${ENABLE_WANDB_SIDECAR:-1}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-.env}"

mkdir -p artifacts/idm artifacts/eval outputs/cluster "$OUTPUT_DIR"

needs_state_materialization=1
if [[ -s "$STATE_SUMMARY" ]]; then
  if uv run python - "$STATE_SUMMARY" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
ok = payload.get("status") == "pass" and int(payload.get("train_rows") or 0) >= 19211006 and int(payload.get("target_rows") or 0) >= 16698646
raise SystemExit(0 if ok else 1)
PY
  then
    needs_state_materialization=0
  fi
fi
if [[ "${FORCE_STATE_MATERIALIZE:-0}" != "0" ]]; then
  needs_state_materialization=1
fi

if [[ "$needs_state_materialization" != "0" ]]; then
  {
    echo "state_materialization_started_at=$(date -Iseconds)"
    uv run python scripts/materialize_d2e_state_corpus.py \
      --train-input "$STATE_INPUT_ROOT/shard_*/train_core.jsonl" \
      --target-input "$STATE_INPUT_ROOT/shard_*/target_all_eval.jsonl" \
      --input-root "$STATE_INPUT_ROOT" \
      --output-root "$STATE_OUTPUT_ROOT" \
      --summary "$STATE_SUMMARY" \
      --progress-output "$STATE_PROGRESS" \
      --mouse-emit-mode decompose \
      --mouse-max-tokens-per-axis 32 \
      --workers "${STATE_MATERIALIZE_WORKERS:-16}"
    echo "state_materialization_finished_at=$(date -Iseconds)"
  } 2>&1 | tee "$STATE_LOG"
fi

needs_cache_precompute=1
if [[ -s "$PRECOMPUTE_CACHE_VALIDATION" ]]; then
  if uv run python - "$PRECOMPUTE_CACHE_VALIDATION" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
ok = payload.get("status") == "pass" and int(payload.get("rows") or 0) >= 19211006
raise SystemExit(0 if ok else 1)
PY
  then
    needs_cache_precompute=0
  fi
fi
if [[ "${FORCE_STATE_CACHE_PRECOMPUTE:-0}" != "0" ]]; then
  needs_cache_precompute=1
fi

if [[ "$needs_cache_precompute" != "0" ]]; then
  {
    echo "cache_precompute_started_at=$(date -Iseconds)"
    uv run python scripts/precompute_streaming_idm_training_cache.py \
      --config "$CONFIG" \
      --workers "${STATE_CACHE_WORKERS:-32}" \
      --output "$PRECOMPUTE_CACHE_VALIDATION" \
      --progress-output "$PRECOMPUTE_CACHE_PROGRESS"
    echo "cache_precompute_finished_at=$(date -Iseconds)"
  } 2>&1 | tee "$PRECOMPUTE_CACHE_LOG"
fi

SIDECAR_PID=""
if [[ "$ENABLE_WANDB_SIDECAR" != "0" ]]; then
  uv run --with wandb python scripts/watch_wandb_training.py \
    --env-file "$WANDB_ENV_FILE" \
    --train-history "$OUTPUT_DIR/train_history.json" \
    --rank-progress-dir "$OUTPUT_DIR/rank_progress" \
    --gpu-monitor "$GPU_MONITOR_LOG" \
    --run-summary "$RUN_SUMMARY" \
    --checkpoint "$OUTPUT_DIR/checkpoint.pt" \
    --metadata "$OUTPUT_DIR/checkpoint_metadata.json" \
    --output "$WANDB_SIDECAR_STATUS" \
    --pid-file "$WANDB_SIDECAR_PID_FILE" \
    --run-name "$MODEL_SLUG-4xh200" \
    --group "g005-idm-paper-target" \
    --job-type "train-sidecar" \
    --tags "g005,idm,d2e,state-corpus,4xh200,sidecar" \
    --poll-seconds 60 \
    --process-pattern "train_idm_streaming|torchrun|run_g005_idm_state_luma_pair" \
    --finish-on-run-summary >"$WANDB_SIDECAR_LOG" 2>&1 &
  SIDECAR_PID="$!"
fi

cleanup_sidecar() {
  if [[ -n "${SIDECAR_PID:-}" ]]; then
    wait "$SIDECAR_PID" >/dev/null 2>&1 || true
    SIDECAR_PID=""
  fi
}
trap cleanup_sidecar EXIT

CONFIG="$CONFIG" \
MODEL_SLUG="$MODEL_SLUG" \
OUTPUT_DIR="$OUTPUT_DIR" \
LOG_PATH="$LOG_PATH" \
RUN_SUMMARY="$RUN_SUMMARY" \
GPU_MONITOR_LOG="$GPU_MONITOR_LOG" \
PID_FILE="$PID_FILE" \
GPU_SMOKE_REPORT="$GPU_SMOKE_REPORT" \
SPLIT_STATS_CONFIG="$SPLIT_STATS_CONFIG" \
PAPER_TARGET_CONFIG="$PAPER_TARGET_CONFIG" \
PRESEED_STATS=0 \
ALLOW_CACHE_BUILD=0 \
REQUIRE_PRECOMPUTED_CACHE=1 \
PRECOMPUTE_CACHE_VALIDATION="$PRECOMPUTE_CACHE_VALIDATION" \
scripts/run_g005_idm_surface_paper_target_4xh200.sh
