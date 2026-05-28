#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi

CONFIG="${CONFIG:-configs/model/idm_streaming_d2e_full_frozen_frame_embedding_prefix320k.yaml}"
PAPER_CONFIG="${PAPER_CONFIG:-configs/eval/g005_idm_frozen_frame_embedding_prefix320k_paper_metrics.yaml}"
SOURCE_INPUT_ROOT="${SOURCE_INPUT_ROOT:-outputs/data/d2e_event_state_duration_context_shards_accel64}"
SOURCE_PREFIX_ROOT="${SOURCE_PREFIX_ROOT:-outputs/data/d2e_event_state_duration_hierarchical_prefix320k}"
EMBED_PREFIX_ROOT="${EMBED_PREFIX_ROOT:-outputs/data/d2e_frozen_frame_embedding_prefix320k}"
SOURCE_TRAIN_PREFIX_SUMMARY="${SOURCE_TRAIN_PREFIX_SUMMARY:-artifacts/idm/g005_idm_frozen_frame_embedding_source_train_prefix320k_summary.json}"
SOURCE_TARGET_PREFIX_SUMMARY="${SOURCE_TARGET_PREFIX_SUMMARY:-artifacts/idm/g005_idm_frozen_frame_embedding_source_target_prefix320k_summary.json}"
TRAIN_EMBED_SUMMARY="${TRAIN_EMBED_SUMMARY:-artifacts/idm/g005_idm_frozen_frame_embedding_train_prefix320k_materialization_summary.json}"
TARGET_EMBED_SUMMARY="${TARGET_EMBED_SUMMARY:-artifacts/idm/g005_idm_frozen_frame_embedding_target_prefix320k_materialization_summary.json}"
TRAIN_EMBED_PROGRESS="${TRAIN_EMBED_PROGRESS:-artifacts/idm/g005_idm_frozen_frame_embedding_train_prefix320k_materialization_progress.json}"
TARGET_EMBED_PROGRESS="${TARGET_EMBED_PROGRESS:-artifacts/idm/g005_idm_frozen_frame_embedding_target_prefix320k_materialization_progress.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/idm_streaming_d2e_full_frozen_frame_embedding_prefix320k}"
RUN_SUMMARY="${RUN_SUMMARY:-artifacts/idm/g005_idm_frozen_frame_embedding_prefix320k_run_summary.json}"
GPU_MONITOR_LOG="${GPU_MONITOR_LOG:-artifacts/idm/g005_idm_frozen_frame_embedding_prefix320k_gpu_monitor.csv}"
GPU_MONITOR_PID_FILE="${GPU_MONITOR_PID_FILE:-outputs/cluster/g005_frozen_frame_embedding_prefix_gpu_monitor.pid}"
WANDB_SIDECAR_STATUS="${WANDB_SIDECAR_STATUS:-artifacts/idm/g005_idm_frozen_frame_embedding_prefix320k_wandb_sidecar_status.json}"
WANDB_SIDECAR_LOG="${WANDB_SIDECAR_LOG:-artifacts/idm/g005_idm_frozen_frame_embedding_prefix320k_wandb_sidecar.log}"
WANDB_SIDECAR_PID_FILE="${WANDB_SIDECAR_PID_FILE:-outputs/cluster/g005_frozen_frame_embedding_prefix_wandb_sidecar.pid}"
ENABLE_WANDB_SIDECAR="${ENABLE_WANDB_SIDECAR:-1}"
NPROC="${NPROC:-1}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-320000}"
MAX_TARGET_ROWS="${MAX_TARGET_ROWS:-320000}"
EMBED_BACKEND="${EMBED_BACKEND:-hf-vision}"
EMBED_MODEL_ID="${EMBED_MODEL_ID:-facebook/dinov2-small}"
EMBED_FRAME_OFFSETS="${EMBED_FRAME_OFFSETS:-0,2}"
EMBED_IMAGE_SIZE="${EMBED_IMAGE_SIZE:-224}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-16}"
EMBED_DEVICE="${EMBED_DEVICE:-auto}"
EMBED_POOLING="${EMBED_POOLING:-cls}"
EMBED_PROGRESS_ROWS="${EMBED_PROGRESS_ROWS:-25000}"
MATERIALIZE_ONLY="${MATERIALIZE_ONLY:-0}"

mkdir -p artifacts/idm outputs/cluster "$OUTPUT_DIR" "$SOURCE_PREFIX_ROOT" "$EMBED_PREFIX_ROOT" "$(dirname "$GPU_MONITOR_LOG")"

MONITOR_PID=""
SIDECAR_PID=""
cleanup_background() {
  if [[ -n "${MONITOR_PID:-}" ]]; then
    kill "$MONITOR_PID" >/dev/null 2>&1 || true
    wait "$MONITOR_PID" >/dev/null 2>&1 || true
    MONITOR_PID=""
  fi
  if [[ -n "${SIDECAR_PID:-}" ]]; then
    wait "$SIDECAR_PID" >/dev/null 2>&1 || true
    SIDECAR_PID=""
  fi
}
trap cleanup_background EXIT

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv -l 30 >"$GPU_MONITOR_LOG" 2>&1 &
  MONITOR_PID="$!"
  echo "$MONITOR_PID" >"$GPU_MONITOR_PID_FILE"
fi

STARTED_AT="$(date -Iseconds)"

if [[ ! -s "$SOURCE_PREFIX_ROOT/train_core.jsonl" ]]; then
  uv run python scripts/materialize_chronological_prefix.py \
    --input "$SOURCE_INPUT_ROOT/shard_*/train_core.jsonl" \
    --output "$SOURCE_PREFIX_ROOT/train_core.jsonl" \
    --summary-out "$SOURCE_TRAIN_PREFIX_SUMMARY" \
    --max-rows "$MAX_TRAIN_ROWS" \
    --source-label "g005_frozen_frame_embedding_source_train_prefix320k"
fi
if [[ ! -s "$SOURCE_PREFIX_ROOT/target_all_eval.jsonl" ]]; then
  uv run python scripts/materialize_chronological_prefix.py \
    --input "$SOURCE_INPUT_ROOT/shard_*/target_all_eval.jsonl" \
    --output "$SOURCE_PREFIX_ROOT/target_all_eval.jsonl" \
    --summary-out "$SOURCE_TARGET_PREFIX_SUMMARY" \
    --max-rows "$MAX_TARGET_ROWS" \
    --source-label "g005_frozen_frame_embedding_source_target_prefix320k"
fi

if [[ "$EMBED_BACKEND" == "hf-vision" ]]; then
  EMBED_PY=(uv run --extra train python)
else
  EMBED_PY=(uv run python)
fi

"${EMBED_PY[@]}" scripts/materialize_frame_embedding_features.py \
  --input-path "$SOURCE_PREFIX_ROOT/train_core.jsonl" \
  --output-path "$EMBED_PREFIX_ROOT/train_core.jsonl" \
  --summary-out "$TRAIN_EMBED_SUMMARY" \
  --progress-output "$TRAIN_EMBED_PROGRESS" \
  --backend "$EMBED_BACKEND" \
  --model-id "$EMBED_MODEL_ID" \
  --frame-offsets "$EMBED_FRAME_OFFSETS" \
  --image-size "$EMBED_IMAGE_SIZE" \
  --batch-size "$EMBED_BATCH_SIZE" \
  --device "$EMBED_DEVICE" \
  --embedding-pooling "$EMBED_POOLING" \
  --max-rows "$MAX_TRAIN_ROWS" \
  --progress-rows "$EMBED_PROGRESS_ROWS" \
  --source-label "g005_frozen_frame_embedding_train_prefix320k"

"${EMBED_PY[@]}" scripts/materialize_frame_embedding_features.py \
  --input-path "$SOURCE_PREFIX_ROOT/target_all_eval.jsonl" \
  --output-path "$EMBED_PREFIX_ROOT/target_all_eval.jsonl" \
  --summary-out "$TARGET_EMBED_SUMMARY" \
  --progress-output "$TARGET_EMBED_PROGRESS" \
  --backend "$EMBED_BACKEND" \
  --model-id "$EMBED_MODEL_ID" \
  --frame-offsets "$EMBED_FRAME_OFFSETS" \
  --image-size "$EMBED_IMAGE_SIZE" \
  --batch-size "$EMBED_BATCH_SIZE" \
  --device "$EMBED_DEVICE" \
  --embedding-pooling "$EMBED_POOLING" \
  --max-rows "$MAX_TARGET_ROWS" \
  --progress-rows "$EMBED_PROGRESS_ROWS" \
  --source-label "g005_frozen_frame_embedding_target_prefix320k"

if [[ "$MATERIALIZE_ONLY" == "1" ]]; then
  python3 - <<PY
import json, pathlib, time
summary = {
  "schema": "g005_frozen_frame_embedding_prefix_run_summary.v1",
  "status": "materialized",
  "started_at": "$STARTED_AT",
  "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
  "config": "$CONFIG",
  "paper_metrics": None,
  "train_embedding_summary": "$TRAIN_EMBED_SUMMARY",
  "target_embedding_summary": "$TARGET_EMBED_SUMMARY",
  "gpu_monitor_log": "$GPU_MONITOR_LOG",
  "claim_boundary": "Frozen frame-embedding prefix materialization only; no trained-model metric claim."
}
pathlib.Path("$RUN_SUMMARY").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\\n")
PY
  exit 0
fi

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
    --run-name "g005-frozen-frame-embedding-prefix320k" \
    --group "g005-idm-paper-target" \
    --job-type "train-sidecar" \
    --tags "g005,idm,d2e,frozen-frame-embedding,prefix" \
    --poll-seconds 30 \
    --process-pattern "train_idm_streaming|torchrun|run_g005_idm_frozen_frame_embedding_prefix" \
    --finish-on-run-summary >"$WANDB_SIDECAR_LOG" 2>&1 &
  SIDECAR_PID="$!"
fi

uv run --extra train torchrun --standalone --nproc-per-node="$NPROC" scripts/train_idm_streaming.py --config "$CONFIG" --require-torch
uv run --extra train python scripts/build_g005_idm_paper_metrics.py --config "$PAPER_CONFIG"

python3 - <<PY
import json, pathlib, time
summary = {
  "schema": "g005_frozen_frame_embedding_prefix_run_summary.v1",
  "status": "pass",
  "started_at": "$STARTED_AT",
  "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
  "config": "$CONFIG",
  "train_embedding_summary": "$TRAIN_EMBED_SUMMARY",
  "target_embedding_summary": "$TARGET_EMBED_SUMMARY",
  "paper_metrics": json.load(open("artifacts/idm/g005_idm_frozen_frame_embedding_prefix320k_paper_metrics.json")),
  "gpu_monitor_log": "$GPU_MONITOR_LOG",
  "wandb_sidecar_status": "$WANDB_SIDECAR_STATUS",
  "claim_boundary": "Frozen frame-embedding prefix diagnostic only; not G005 completion evidence unless paper targets pass."
}
pathlib.Path("$RUN_SUMMARY").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\\n")
PY

