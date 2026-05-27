#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
CONFIG="${CONFIG:-configs/model/idm_streaming_d2e_full_event_state_duration_context_dropout035_closed_loop_prefix320k.yaml}"
DROPOUT_ROOT="${DROPOUT_ROOT:-outputs/data/d2e_event_state_duration_context_dropout035_shards_accel64}"
DROPOUT_SUMMARY="${DROPOUT_SUMMARY:-artifacts/idm/g005_idm_event_state_duration_context_dropout035_materialization_summary.json}"
DROPOUT_PROGRESS="${DROPOUT_PROGRESS:-artifacts/idm/g005_idm_event_state_duration_context_dropout035_materialization_progress.json}"
PAPER_CONFIG="${PAPER_CONFIG:-configs/eval/g005_idm_event_state_duration_context_dropout035_closed_loop_prefix320k_paper_metrics.yaml}"
RUN_SUMMARY="${RUN_SUMMARY:-artifacts/idm/g005_idm_event_state_duration_context_dropout035_closed_loop_prefix320k_run_summary.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/idm_streaming_d2e_full_event_state_duration_context_dropout035_closed_loop_prefix320k}"
GPU_MONITOR_LOG="${GPU_MONITOR_LOG:-artifacts/idm/g005_idm_event_state_duration_context_dropout035_closed_loop_prefix320k_gpu_monitor.csv}"
GPU_MONITOR_PID_FILE="${GPU_MONITOR_PID_FILE:-outputs/cluster/g005_context_dropout_prefix_gpu_monitor.pid}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-.env}"
WANDB_SIDECAR_STATUS="${WANDB_SIDECAR_STATUS:-artifacts/idm/g005_idm_event_state_duration_context_dropout035_closed_loop_prefix320k_wandb_sidecar_status.json}"
WANDB_SIDECAR_LOG="${WANDB_SIDECAR_LOG:-artifacts/idm/g005_idm_event_state_duration_context_dropout035_closed_loop_prefix320k_wandb_sidecar.log}"
WANDB_SIDECAR_PID_FILE="${WANDB_SIDECAR_PID_FILE:-outputs/cluster/g005_context_dropout_prefix_wandb_sidecar.pid}"
WANDB_SIDECAR_TAGS="${WANDB_SIDECAR_TAGS:-g005,idm,d2e,context-dropout,closed-loop,prefix,sidecar}"
WANDB_PROCESS_PATTERN="${WANDB_PROCESS_PATTERN:-train_idm_streaming|torchrun|run_g005_idm_event_state_duration_context_dropout035}"
ENABLE_WANDB_SIDECAR="${ENABLE_WANDB_SIDECAR:-1}"
MAX_ROWS_PER_FILE="${MAX_ROWS_PER_FILE:-20000}"
NPROC="${NPROC:-2}"

mkdir -p artifacts/idm outputs/cluster "$OUTPUT_DIR" "$(dirname "$GPU_MONITOR_LOG")"
MONITOR_PID=""
SIDECAR_PID=""

cleanup_background() {
  if [[ -n "${MONITOR_PID:-}" ]]; then
    kill "$MONITOR_PID" >/dev/null 2>&1 || true
    wait "$MONITOR_PID" >/dev/null 2>&1 || true
    MONITOR_PID=""
  fi
  if [[ -n "${SIDECAR_PID:-}" ]]; then
    for _ in {1..6}; do
      if ! kill -0 "$SIDECAR_PID" >/dev/null 2>&1; then
        wait "$SIDECAR_PID" >/dev/null 2>&1 || true
        SIDECAR_PID=""
        break
      fi
      [[ -s "$RUN_SUMMARY" ]] && sleep 5 || break
    done
    if [[ -n "${SIDECAR_PID:-}" ]]; then
      kill "$SIDECAR_PID" >/dev/null 2>&1 || true
      wait "$SIDECAR_PID" >/dev/null 2>&1 || true
      SIDECAR_PID=""
    fi
  fi
}
trap cleanup_background EXIT

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
    --format=csv \
    -l 30 >"$GPU_MONITOR_LOG" 2>&1 &
  MONITOR_PID="$!"
  echo "$MONITOR_PID" >"$GPU_MONITOR_PID_FILE"
fi

STARTED_AT="$(date -Iseconds)"
uv run python scripts/materialize_state_context_dropout_train.py \
  --input 'outputs/data/d2e_event_state_duration_context_shards_accel64/shard_*/train_core.jsonl' \
  --output-root "$DROPOUT_ROOT" \
  --dropout-rate 0.35 \
  --seed 20260528 \
  --summary-out "$DROPOUT_SUMMARY" \
  --progress-out "$DROPOUT_PROGRESS" \
  --max-rows-per-file "$MAX_ROWS_PER_FILE"
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
    --run-name "g005-context-dropout-closed-loop-prefix320k" \
    --group "g005-idm-paper-target" \
    --job-type "train-sidecar" \
    --tags "$WANDB_SIDECAR_TAGS" \
    --poll-seconds 30 \
    --process-pattern "$WANDB_PROCESS_PATTERN" \
    --finish-on-run-summary >"$WANDB_SIDECAR_LOG" 2>&1 &
  SIDECAR_PID="$!"
fi
uv run torchrun --standalone --nproc-per-node="$NPROC" scripts/train_idm_streaming.py --config "$CONFIG" --require-torch
uv run python scripts/build_g005_idm_paper_metrics.py --config "$PAPER_CONFIG"
python3 - <<PY
import json, pathlib, time
summary={
  'schema':'g005_dropout_closed_loop_prefix_run_summary.v1',
  'status':'pass',
  'started_at':'$STARTED_AT',
  'finished_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
  'config':'$CONFIG',
  'dropout_summary':'$DROPOUT_SUMMARY',
  'paper_metrics':json.load(open('artifacts/idm/g005_idm_event_state_duration_context_dropout035_closed_loop_prefix320k_paper_metrics.json')),
  'gpu_monitor_log':'$GPU_MONITOR_LOG',
  'wandb_sidecar_status':'$WANDB_SIDECAR_STATUS',
  'reservation': {
    'id': "${MLXP_RESERVATION_ID:-}" or None,
    'start_at': "${MLXP_RESERVATION_START_AT:-}" or None,
    'end_at': "${MLXP_RESERVATION_END_AT:-}" or None,
    'node_id': "${MLXP_RESERVATION_NODE_ID:-}" or None,
    'gpu_indices': "${MLXP_RESERVATION_GPU_INDICES:-}" or None,
    'pod_name': "${MLXP_RESERVATION_POD_NAME:-}" or None,
    'checked_at': "${MLXP_RESERVATION_CHECKED_AT:-}" or None,
  },
  'claim_boundary':'Prefix diagnostic only; not G005 completion evidence.'
}
pathlib.Path('$RUN_SUMMARY').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
PY
