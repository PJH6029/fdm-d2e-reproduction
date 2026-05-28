#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
CONFIG="${CONFIG:-configs/model/idm_streaming_d2e_full_event_state_duration_counted_hierarchical_prefix320k.yaml}"
PAPER_CONFIG="${PAPER_CONFIG:-configs/eval/g005_idm_event_state_duration_counted_hierarchical_prefix320k_paper_metrics.yaml}"
INPUT_ROOT="${INPUT_ROOT:-outputs/data/d2e_event_state_duration_context_shards_accel64}"
PREFIX_ROOT="${PREFIX_ROOT:-outputs/data/d2e_event_state_duration_counted_hierarchical_prefix320k}"
TRAIN_PREFIX_SUMMARY="${TRAIN_PREFIX_SUMMARY:-artifacts/idm/g005_idm_event_state_duration_counted_hierarchical_prefix320k_train_materialization_summary.json}"
TARGET_PREFIX_SUMMARY="${TARGET_PREFIX_SUMMARY:-artifacts/idm/g005_idm_event_state_duration_counted_hierarchical_prefix320k_target_materialization_summary.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/idm_streaming_d2e_full_event_state_duration_counted_hierarchical_prefix320k}"
RUN_SUMMARY="${RUN_SUMMARY:-artifacts/idm/g005_idm_event_state_duration_counted_hierarchical_prefix320k_run_summary.json}"
GPU_MONITOR_LOG="${GPU_MONITOR_LOG:-artifacts/idm/g005_idm_event_state_duration_counted_hierarchical_prefix320k_gpu_monitor.csv}"
GPU_MONITOR_PID_FILE="${GPU_MONITOR_PID_FILE:-outputs/cluster/g005_counted_hierarchical_prefix_gpu_monitor.pid}"
WANDB_SIDECAR_STATUS="${WANDB_SIDECAR_STATUS:-artifacts/idm/g005_idm_event_state_duration_counted_hierarchical_prefix320k_wandb_sidecar_status.json}"
WANDB_SIDECAR_LOG="${WANDB_SIDECAR_LOG:-artifacts/idm/g005_idm_event_state_duration_counted_hierarchical_prefix320k_wandb_sidecar.log}"
WANDB_SIDECAR_PID_FILE="${WANDB_SIDECAR_PID_FILE:-outputs/cluster/g005_counted_hierarchical_prefix_wandb_sidecar.pid}"
ENABLE_WANDB_SIDECAR="${ENABLE_WANDB_SIDECAR:-1}"
NPROC="${NPROC:-1}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-320000}"
MAX_TARGET_ROWS="${MAX_TARGET_ROWS:-320000}"
mkdir -p artifacts/idm outputs/cluster "$OUTPUT_DIR" "$PREFIX_ROOT" "$(dirname "$GPU_MONITOR_LOG")"
MONITOR_PID=""
SIDECAR_PID=""
cleanup_background() {
  if [[ -n "${MONITOR_PID:-}" ]]; then kill "$MONITOR_PID" >/dev/null 2>&1 || true; wait "$MONITOR_PID" >/dev/null 2>&1 || true; MONITOR_PID=""; fi
  if [[ -n "${SIDECAR_PID:-}" ]]; then wait "$SIDECAR_PID" >/dev/null 2>&1 || true; SIDECAR_PID=""; fi
}
trap cleanup_background EXIT
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv -l 30 >"$GPU_MONITOR_LOG" 2>&1 &
  MONITOR_PID="$!"
  echo "$MONITOR_PID" >"$GPU_MONITOR_PID_FILE"
fi
STARTED_AT="$(date -Iseconds)"
uv run python scripts/materialize_chronological_prefix.py \
  --input "$INPUT_ROOT/shard_*/train_core.jsonl" \
  --output "$PREFIX_ROOT/train_core.jsonl" \
  --summary-out "$TRAIN_PREFIX_SUMMARY" \
  --max-rows "$MAX_TRAIN_ROWS" \
  --source-label "g005_event_state_duration_counted_hierarchical_train_prefix320k"
uv run python scripts/materialize_chronological_prefix.py \
  --input "$INPUT_ROOT/shard_*/target_all_eval.jsonl" \
  --output "$PREFIX_ROOT/target_all_eval.jsonl" \
  --summary-out "$TARGET_PREFIX_SUMMARY" \
  --max-rows "$MAX_TARGET_ROWS" \
  --source-label "g005_event_state_duration_counted_hierarchical_target_prefix320k"
if [[ "$ENABLE_WANDB_SIDECAR" != "0" ]]; then
  uv run --with wandb python scripts/watch_wandb_training.py \
    --env-file .env \
    --train-history "$OUTPUT_DIR/train_history.json" \
    --rank-progress-dir "$OUTPUT_DIR/rank_progress" \
    --gpu-monitor "$GPU_MONITOR_LOG" \
    --run-summary "$RUN_SUMMARY" \
    --checkpoint "$OUTPUT_DIR/checkpoint.pt" \
    --metadata "$OUTPUT_DIR/checkpoint_metadata.json" \
    --output "$WANDB_SIDECAR_STATUS" \
    --pid-file "$WANDB_SIDECAR_PID_FILE" \
    --run-name "g005-counted-hierarchical-exactset-prefix320k" \
    --group "g005-idm-paper-target" \
    --job-type "train-sidecar" \
    --tags "g005,idm,d2e,counted-hierarchical-exactset,prefix" \
    --poll-seconds 30 \
    --process-pattern "train_idm_streaming|torchrun|run_g005_idm_event_state_duration_counted_hierarchical_prefix" \
    --finish-on-run-summary >"$WANDB_SIDECAR_LOG" 2>&1 &
  SIDECAR_PID="$!"
fi
uv run --extra train torchrun --standalone --nproc-per-node="$NPROC" scripts/train_idm_streaming.py --config "$CONFIG" --require-torch
uv run --extra train python scripts/build_g005_idm_paper_metrics.py --config "$PAPER_CONFIG"
python3 - <<PY
import json, pathlib, time
summary={
  'schema':'g005_counted_hierarchical_exactset_prefix_run_summary.v1',
  'status':'pass',
  'started_at':'$STARTED_AT',
  'finished_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
  'config':'$CONFIG',
  'train_materialization_summary':'$TRAIN_PREFIX_SUMMARY',
  'target_materialization_summary':'$TARGET_PREFIX_SUMMARY',
  'paper_metrics':json.load(open('artifacts/idm/g005_idm_event_state_duration_counted_hierarchical_prefix320k_paper_metrics.json')),
  'gpu_monitor_log':'$GPU_MONITOR_LOG',
  'wandb_sidecar_status':'$WANDB_SIDECAR_STATUS',
  'claim_boundary':'Count-preserving hierarchical exact-set prefix diagnostic only; not G005 completion evidence.'
}
pathlib.Path('$RUN_SUMMARY').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
PY
