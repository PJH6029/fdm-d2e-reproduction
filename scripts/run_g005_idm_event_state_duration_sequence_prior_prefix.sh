#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
CONFIG="${CONFIG:-configs/model/idm_streaming_d2e_full_event_state_duration_sequence_prior_prefix320k.yaml}"
PAPER_CONFIG="${PAPER_CONFIG:-configs/eval/g005_idm_event_state_duration_sequence_prior_prefix320k_paper_metrics.yaml}"
INPUT_ROOT="${INPUT_ROOT:-outputs/data/d2e_event_state_duration_context_shards_accel64}"
PREFIX_ROOT="${PREFIX_ROOT:-outputs/data/d2e_event_state_duration_hierarchical_prefix320k}"
TRAIN_PREFIX_SUMMARY="${TRAIN_PREFIX_SUMMARY:-artifacts/idm/g005_idm_event_state_duration_sequence_prior_prefix320k_train_materialization_summary.json}"
TARGET_PREFIX_SUMMARY="${TARGET_PREFIX_SUMMARY:-artifacts/idm/g005_idm_event_state_duration_sequence_prior_prefix320k_target_materialization_summary.json}"
TRAIN_PREFIX_REUSE_SUMMARY="${TRAIN_PREFIX_REUSE_SUMMARY:-artifacts/idm/g005_idm_event_state_duration_hierarchical_prefix320k_train_materialization_summary.json}"
TARGET_PREFIX_REUSE_SUMMARY="${TARGET_PREFIX_REUSE_SUMMARY:-artifacts/idm/g005_idm_event_state_duration_hierarchical_prefix320k_target_materialization_summary.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/idm_streaming_d2e_full_event_state_duration_sequence_prior_prefix320k}"
RUN_SUMMARY="${RUN_SUMMARY:-artifacts/idm/g005_idm_event_state_duration_sequence_prior_prefix320k_run_summary.json}"
GPU_MONITOR_LOG="${GPU_MONITOR_LOG:-artifacts/idm/g005_idm_event_state_duration_sequence_prior_prefix320k_gpu_monitor.csv}"
GPU_MONITOR_PID_FILE="${GPU_MONITOR_PID_FILE:-outputs/cluster/g005_sequence_prior_prefix_gpu_monitor.pid}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-.env}"
WANDB_SIDECAR_STATUS="${WANDB_SIDECAR_STATUS:-artifacts/idm/g005_idm_event_state_duration_sequence_prior_prefix320k_wandb_sidecar_status.json}"
WANDB_SIDECAR_LOG="${WANDB_SIDECAR_LOG:-artifacts/idm/g005_idm_event_state_duration_sequence_prior_prefix320k_wandb_sidecar.log}"
WANDB_SIDECAR_PID_FILE="${WANDB_SIDECAR_PID_FILE:-outputs/cluster/g005_sequence_prior_prefix_wandb_sidecar.pid}"
WANDB_SIDECAR_TAGS="${WANDB_SIDECAR_TAGS:-g005,idm,d2e,event-state-duration,sequence-prior,prefix,sidecar}"
WANDB_PROCESS_PATTERN="${WANDB_PROCESS_PATTERN:-train_idm_streaming|torchrun|run_g005_idm_event_state_duration_sequence_prior_prefix}"
ENABLE_WANDB_SIDECAR="${ENABLE_WANDB_SIDECAR:-1}"
NPROC="${NPROC:-1}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-320000}"
MAX_TARGET_ROWS="${MAX_TARGET_ROWS:-320000}"
mkdir -p artifacts/idm outputs/cluster "$OUTPUT_DIR" "$PREFIX_ROOT" "$(dirname "$GPU_MONITOR_LOG")"
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
materialize_prefix_if_needed() {
  local input_glob="$1"
  local output_path="$2"
  local summary_path="$3"
  local max_rows="$4"
  local label="$5"
  local reuse_summary="${6:-}"
  if [[ -s "$output_path" ]]; then
    for candidate_summary in "$summary_path" "$reuse_summary"; do
      [[ -n "$candidate_summary" && -s "$candidate_summary" ]] || continue
      if uv run python - "$candidate_summary" "$summary_path" "$max_rows" "$label" <<'PY'
import json, pathlib, sys, time
source = pathlib.Path(sys.argv[1])
dest = pathlib.Path(sys.argv[2])
required = int(sys.argv[3])
label = sys.argv[4]
payload = json.loads(source.read_text())
rows = int(payload.get('rows') or payload.get('written_rows') or payload.get('output_rows') or 0)
if payload.get('status') != 'pass' or rows < required:
    raise SystemExit(1)
if source != dest:
    reused = dict(payload)
    reused['source_label'] = label
    reused['reused_from_summary'] = str(source)
    reused['reused_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    reused['claim_boundary'] = 'Reused existing chronological prefix rows for sequence-prior diagnostic; not G005 completion evidence.'
    dest.write_text(json.dumps(reused, indent=2, sort_keys=True) + '\n')
raise SystemExit(0)
PY
      then
        return 0
      fi
    done
  fi
  uv run python scripts/materialize_chronological_prefix.py \
    --input "$input_glob" \
    --output "$output_path" \
    --summary-out "$summary_path" \
    --max-rows "$max_rows" \
    --source-label "$label"
}
materialize_prefix_if_needed \
  "$INPUT_ROOT/shard_*/train_core.jsonl" \
  "$PREFIX_ROOT/train_core.jsonl" \
  "$TRAIN_PREFIX_SUMMARY" \
  "$MAX_TRAIN_ROWS" \
  "g005_event_state_duration_sequence_prior_train_prefix320k" \
  "$TRAIN_PREFIX_REUSE_SUMMARY"
materialize_prefix_if_needed \
  "$INPUT_ROOT/shard_*/target_all_eval.jsonl" \
  "$PREFIX_ROOT/target_all_eval.jsonl" \
  "$TARGET_PREFIX_SUMMARY" \
  "$MAX_TARGET_ROWS" \
  "g005_event_state_duration_sequence_prior_target_prefix320k" \
  "$TARGET_PREFIX_REUSE_SUMMARY"

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
    --run-name "g005-event-state-duration-sequence-prior-prefix320k" \
    --group "g005-idm-paper-target" \
    --job-type "train-sidecar" \
    --tags "$WANDB_SIDECAR_TAGS" \
    --poll-seconds 30 \
    --process-pattern "$WANDB_PROCESS_PATTERN" \
    --finish-on-run-summary >"$WANDB_SIDECAR_LOG" 2>&1 &
  SIDECAR_PID="$!"
fi
uv run --extra train torchrun --standalone --nproc-per-node="$NPROC" scripts/train_idm_streaming.py --config "$CONFIG" --require-torch
uv run --extra train python scripts/build_g005_idm_paper_metrics.py --config "$PAPER_CONFIG"
python3 - <<PY
import json, pathlib, time
summary={
  'schema':'g005_sequence_prior_prefix_run_summary.v1',
  'status':'pass',
  'started_at':'$STARTED_AT',
  'finished_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
  'config':'$CONFIG',
  'train_materialization_summary':'$TRAIN_PREFIX_SUMMARY',
  'target_materialization_summary':'$TARGET_PREFIX_SUMMARY',
  'paper_metrics':json.load(open('artifacts/idm/g005_idm_event_state_duration_sequence_prior_prefix320k_paper_metrics.json')),
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
  'claim_boundary':'Sequence-prior event-state-duration prefix diagnostic only; not G005 completion evidence.'
}
pathlib.Path('$RUN_SUMMARY').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
PY
