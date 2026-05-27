#!/usr/bin/env bash
set -euo pipefail

# Non-terminal diagnostic probe for the raw112 offset-2 video IDM candidate.
# This intentionally supports 1/2 GPU prefix probing while the full G005 gate
# still requires full-corpus 4xH200 evidence before checkpointing G005 complete.

export PATH="$HOME/.local/bin:$PATH"
CONFIG="${CONFIG:-configs/model/idm_video_pair_d2e_full_raw112_offset2_keysoftmax_paper_target.yaml}"
MODEL_SLUG="${MODEL_SLUG:-g005_idm_video_pair_raw112_offset2_keysoftmax}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/idm_video_pair_d2e_full_raw112_offset2_keysoftmax_paper_target}"
PREFIX_ROWS="${PREFIX_ROWS:-320000}"
PREFIX_SHARD="${PREFIX_SHARD:-outputs/data/d2e_full_corpus_shards_accel64/shard_00/target_all_eval.jsonl}"
PREFIX_OUTPUT_DIR="${PREFIX_OUTPUT_DIR:-$OUTPUT_DIR/prefix${PREFIX_ROWS}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
EXPECTED_GPUS="${EXPECTED_GPUS:-$NPROC_PER_NODE}"
EPOCHS_OVERRIDE="${EPOCHS_OVERRIDE:-2}"
CHAIN_SUMMARY="${CHAIN_SUMMARY:-artifacts/idm/${MODEL_SLUG}_prefix${PREFIX_ROWS}_probe_chain_summary.json}"
CHAIN_LOG="${CHAIN_LOG:-artifacts/idm/${MODEL_SLUG}_prefix${PREFIX_ROWS}_probe_chain.log}"
TRAIN_RUN_SUMMARY="${TRAIN_RUN_SUMMARY:-artifacts/idm/${MODEL_SLUG}_${EXPECTED_GPUS}gpu_prefix_probe_run.json}"
GPU_MONITOR="${GPU_MONITOR:-artifacts/idm/${MODEL_SLUG}_${EXPECTED_GPUS}gpu_prefix_probe_gpu_monitor.csv}"
PREFIX_CONFIG="${PREFIX_CONFIG:-outputs/cluster/${MODEL_SLUG}_prefix${PREFIX_ROWS}_config.yaml}"
PREFIX_PAPER_CONFIG="${PREFIX_PAPER_CONFIG:-outputs/cluster/${MODEL_SLUG}_prefix${PREFIX_ROWS}_paper_metrics.yaml}"
PREDICTION_SUMMARY="${PREDICTION_SUMMARY:-artifacts/idm/${MODEL_SLUG}_prefix${PREFIX_ROWS}_prediction_summary.json}"
PAPER_METRICS="${PAPER_METRICS:-artifacts/idm/${MODEL_SLUG}_prefix${PREFIX_ROWS}_paper_metrics.json}"
PAPER_PROGRESS="${PAPER_PROGRESS:-artifacts/idm/${MODEL_SLUG}_prefix${PREFIX_ROWS}_paper_metrics_progress.json}"
CACHE_WANDB_STATUS="${CACHE_WANDB_STATUS:-artifacts/idm/${MODEL_SLUG}_train_cache_wandb_status.json}"
CACHE_WANDB_LOG="${CACHE_WANDB_LOG:-artifacts/idm/${MODEL_SLUG}_train_cache_wandb.log}"
TRAIN_WANDB_STATUS="${TRAIN_WANDB_STATUS:-artifacts/idm/${MODEL_SLUG}_${EXPECTED_GPUS}gpu_train_wandb_status.json}"
TRAIN_WANDB_LOG="${TRAIN_WANDB_LOG:-artifacts/idm/${MODEL_SLUG}_${EXPECTED_GPUS}gpu_train_wandb.log}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-.env}"
ENABLE_WANDB_SIDECAR="${ENABLE_WANDB_SIDECAR:-1}"
MLXP_RESERVATION_ID="${MLXP_RESERVATION_ID:-}"
MLXP_RESERVATION_START_AT="${MLXP_RESERVATION_START_AT:-}"
MLXP_RESERVATION_END_AT="${MLXP_RESERVATION_END_AT:-}"
MLXP_RESERVATION_NODE_ID="${MLXP_RESERVATION_NODE_ID:-}"
MLXP_RESERVATION_GPU_INDICES="${MLXP_RESERVATION_GPU_INDICES:-}"
MLXP_RESERVATION_POD_NAME="${MLXP_RESERVATION_POD_NAME:-}"
MLXP_RESERVATION_CHECKED_AT="${MLXP_RESERVATION_CHECKED_AT:-$(date -Iseconds)}"
QUOTA_REQUEST_ID="${QUOTA_REQUEST_ID:-}"

mkdir -p artifacts/idm artifacts/eval outputs/cluster "$PREFIX_OUTPUT_DIR"
CACHE_SIDECAR_PID=""
TRAIN_SIDECAR_PID=""

cleanup_sidecars() {
  for pid in "$CACHE_SIDECAR_PID" "$TRAIN_SIDECAR_PID"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup_sidecars EXIT

wandb_configured() {
  [[ -n "${WANDB_PROJECT:-}" ]] && return 0
  [[ -f "$WANDB_ENV_FILE" ]] && grep -Eq '^[[:space:]]*WANDB_PROJECT=' "$WANDB_ENV_FILE"
}

write_summary() {
  local status="$1"
  uv run --no-sync python - "$status" <<PY
from __future__ import annotations
import json, os, subprocess, time
from pathlib import Path
status = __import__("sys").argv[1]
def load(path: str):
    p=Path(path)
    if p.exists() and p.is_file():
        try: return json.loads(p.read_text())
        except Exception: return None
    return None
def git(args):
    try: return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return None
paper = load("$PAPER_METRICS") or {}
allm = paper.get("groups", {}).get("all", {}) if isinstance(paper.get("groups"), dict) else {}
payload={
  "schema":"g005_offset2_prefix_probe_chain_summary.v1",
  "status":status,
  "updated_at_unix":time.time(),
  "git_head":git(["rev-parse","HEAD"]),
  "git_status_short":git(["status","--short"]),
  "model_slug":"$MODEL_SLUG",
  "config":"$CONFIG",
  "prefix_rows":int("$PREFIX_ROWS"),
  "prefix_shard":"$PREFIX_SHARD",
  "prefix_config":"$PREFIX_CONFIG",
  "prefix_paper_config":"$PREFIX_PAPER_CONFIG",
  "nproc_per_node":int("$NPROC_PER_NODE"),
  "expected_gpus":int("$EXPECTED_GPUS"),
  "epochs_override":int("$EPOCHS_OVERRIDE"),
  "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
  "mlxp_reservation":{
    "reservation_id":"$MLXP_RESERVATION_ID" or None,
    "start_at":"$MLXP_RESERVATION_START_AT" or None,
    "end_at":"$MLXP_RESERVATION_END_AT" or None,
    "node_id":"$MLXP_RESERVATION_NODE_ID" or None,
    "gpu_indices":"$MLXP_RESERVATION_GPU_INDICES" or None,
    "pod_name":"$MLXP_RESERVATION_POD_NAME" or None,
    "checked_at":"$MLXP_RESERVATION_CHECKED_AT" or None,
  },
  "quota_request_id":"$QUOTA_REQUEST_ID" or None,
  "claim_boundary":"Prefix diagnostic only; not full-corpus G005 completion evidence and not a 4xH200 scaling claim.",
  "train_cache_precompute_run":load("artifacts/idm/${MODEL_SLUG}_train_precompute_run.json"),
  "target_prefix_cache_precompute_run":load("artifacts/idm/${MODEL_SLUG}_target_prefix_precompute_run.json"),
  "train_run":load("$TRAIN_RUN_SUMMARY"),
  "prediction":load("$PREDICTION_SUMMARY"),
  "paper_metrics_payload":paper,
  "paper_all":allm.get("paper_compatible"),
  "strict_local":allm.get("strict_local"),
  "cache_wandb_status":load("$CACHE_WANDB_STATUS"),
  "train_wandb_status":load("$TRAIN_WANDB_STATUS"),
}
Path("$CHAIN_SUMMARY").write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
print(json.dumps({k: payload.get(k) for k in ["status","git_head","prefix_rows","paper_all","strict_local"]}, indent=2, sort_keys=True))
PY
}

start_cache_wandb() {
  [[ "$ENABLE_WANDB_SIDECAR" == "0" ]] && return 0
  wandb_configured || return 0
  nohup env WANDB_RESUME=allow uv run --no-sync --with wandb python scripts/watch_wandb_video_cache.py \
    --env-file "$WANDB_ENV_FILE" \
    --cache-dir "$OUTPUT_DIR/video_cache" \
    --summary "artifacts/idm/${MODEL_SLUG}_cache_precompute_summary.json" \
    --run-summary "artifacts/idm/${MODEL_SLUG}_train_precompute_run.json" \
    --output "$CACHE_WANDB_STATUS" \
    --pid-file "outputs/cluster/${MODEL_SLUG}_train_cache_wandb.pid" \
    --run-name "${MODEL_SLUG}-train-cache" \
    --group g005-idm-paper-target \
    --job-type video-cache \
    --tags g005,idm,d2e,raw112,offset2,cache,prefix-probe \
    --poll-seconds 60 \
    --finish-manifests 64 \
    >"$CACHE_WANDB_LOG" 2>&1 &
  CACHE_SIDECAR_PID="$!"
}

start_train_wandb() {
  [[ "$ENABLE_WANDB_SIDECAR" == "0" ]] && return 0
  wandb_configured || return 0
  nohup env WANDB_RESUME=allow uv run --no-sync --with wandb python scripts/watch_wandb_training.py \
    --env-file "$WANDB_ENV_FILE" \
    --train-history "$OUTPUT_DIR/train_history.json" \
    --rank-progress-dir "$OUTPUT_DIR/rank_progress" \
    --gpu-monitor "$GPU_MONITOR" \
    --run-summary "$TRAIN_RUN_SUMMARY" \
    --checkpoint "$OUTPUT_DIR/checkpoint.pt" \
    --metadata "$OUTPUT_DIR/checkpoint_metadata.json" \
    --output "$TRAIN_WANDB_STATUS" \
    --pid-file "outputs/cluster/${MODEL_SLUG}_${EXPECTED_GPUS}gpu_train_wandb.pid" \
    --run-name "${MODEL_SLUG}-${EXPECTED_GPUS}gpu-train" \
    --group g005-idm-paper-target \
    --job-type train-sidecar \
    --tags g005,idm,d2e,raw112,offset2,prefix-probe \
    --poll-seconds 60 \
    --process-pattern train_idm_video.py \
    --finish-on-run-summary \
    >"$TRAIN_WANDB_LOG" 2>&1 &
  TRAIN_SIDECAR_PID="$!"
}

(
  echo "chain_started_at=$(date -Iseconds)"
  echo "git_head=$(git rev-parse HEAD)"
  echo "config=$CONFIG"
  write_summary running_train_cache
  start_cache_wandb
  CONFIG="$CONFIG" MODEL_SLUG="$MODEL_SLUG" PRECOMPUTE_SPLITS=train \
    LOG_PATH="artifacts/idm/${MODEL_SLUG}_train_precompute.log" \
    RUN_SUMMARY="artifacts/idm/${MODEL_SLUG}_train_precompute_run.json" \
    PID_FILE="outputs/cluster/${MODEL_SLUG}_train_precompute.pid" \
    scripts/run_g005_idm_video_pair_raw112_offset2_precompute.sh

  write_summary running_training
  start_train_wandb
  CONFIG="$CONFIG" MODEL_SLUG="$MODEL_SLUG" \
    NPROC_PER_NODE="$NPROC_PER_NODE" EXPECTED_GPUS="$EXPECTED_GPUS" EPOCHS_OVERRIDE="$EPOCHS_OVERRIDE" \
    SKIP_PREDICTION=1 BUILD_SPLIT_STATS=0 BUILD_PAPER_METRICS=0 VALIDATE_G005=0 \
    LOG_PATH="artifacts/idm/${MODEL_SLUG}_${EXPECTED_GPUS}gpu_prefix_probe.log" \
    RUN_SUMMARY="$TRAIN_RUN_SUMMARY" \
    GPU_MONITOR_LOG="$GPU_MONITOR" \
    PID_FILE="outputs/cluster/${MODEL_SLUG}_${EXPECTED_GPUS}gpu_prefix_probe.pid" \
    GPU_SMOKE_REPORT="outputs/cluster/${MODEL_SLUG}_${EXPECTED_GPUS}gpu_smoke.json" \
    MLXP_RESERVATION_ID="$MLXP_RESERVATION_ID" \
    MLXP_RESERVATION_START_AT="$MLXP_RESERVATION_START_AT" \
    MLXP_RESERVATION_END_AT="$MLXP_RESERVATION_END_AT" \
    MLXP_RESERVATION_NODE_ID="$MLXP_RESERVATION_NODE_ID" \
    MLXP_RESERVATION_GPU_INDICES="$MLXP_RESERVATION_GPU_INDICES" \
    MLXP_RESERVATION_POD_NAME="$MLXP_RESERVATION_POD_NAME" \
    MLXP_RESERVATION_CHECKED_AT="$MLXP_RESERVATION_CHECKED_AT" \
    scripts/run_g005_idm_video_pair_raw112_offset2_4xh200.sh

  write_summary running_target_prefix_cache
  uv run --no-sync python - <<PY
from __future__ import annotations
import json
from pathlib import Path
cfg=json.loads(Path("$CONFIG").read_text())
cfg["target_records"]="$PREFIX_SHARD"
cfg.pop("target_records_glob", None)
cfg["max_target_examples"]=int("$PREFIX_ROWS")
cfg["source_config_path"]="$CONFIG"
cfg["runtime_overrides"]={"target_records":"$PREFIX_SHARD", "max_target_examples":int("$PREFIX_ROWS"), "probe":"prefix"}
Path("$PREFIX_CONFIG").write_text(json.dumps(cfg, indent=2, sort_keys=True)+"\n")
paper=json.loads(Path("configs/eval/g005_idm_video_pair_raw112_offset2_keysoftmax_paper_target.yaml").read_text())
paper["output_path"]="artifacts/idm/${MODEL_SLUG}_prefix${PREFIX_ROWS}_paper_target_audit.json"
paper["paths"]["paper_metrics"]="$PAPER_METRICS"
paper["paths"]["run_summary"]="$TRAIN_RUN_SUMMARY"
paper["paths"]["gpu_monitor"]="$GPU_MONITOR"
paper["paper_metrics"]["output_path"]="$PAPER_METRICS"
paper["paper_metrics"]["progress_output_path"]="$PAPER_PROGRESS"
paper["paper_metrics"]["predictions_path"]="$PREFIX_OUTPUT_DIR/predictions.jsonl"
paper["paper_metrics"]["target_path"]="$PREFIX_SHARD"
paper["paper_metrics"]["max_rows"]=int("$PREFIX_ROWS")
paper["claim_boundary"]="Prefix diagnostic only; not full-corpus G005 completion evidence."
Path("$PREFIX_PAPER_CONFIG").write_text(json.dumps(paper, indent=2, sort_keys=True)+"\n")
PY
  CONFIG="$PREFIX_CONFIG" MODEL_SLUG="${MODEL_SLUG}_target_prefix" PRECOMPUTE_SPLITS=target \
    LOG_PATH="artifacts/idm/${MODEL_SLUG}_target_prefix_precompute.log" \
    RUN_SUMMARY="artifacts/idm/${MODEL_SLUG}_target_prefix_precompute_run.json" \
    PID_FILE="outputs/cluster/${MODEL_SLUG}_target_prefix_precompute.pid" \
    scripts/run_g005_idm_video_pair_raw112_offset2_precompute.sh

  write_summary running_prefix_prediction
  uv run --no-sync python scripts/predict_idm_video.py \
    --config "$PREFIX_CONFIG" \
    --checkpoint-path "$OUTPUT_DIR/checkpoint.pt" \
    --output-dir "$PREFIX_OUTPUT_DIR" \
    --max-target-examples "$PREFIX_ROWS" \
    --prediction-summary-out "$PREDICTION_SUMMARY" \
    --require-torch
  uv run --no-sync python scripts/build_g005_idm_paper_metrics.py --config "$PREFIX_PAPER_CONFIG"
  write_summary pass
  echo "chain_finished_at=$(date -Iseconds)"
) 2>&1 | tee "$CHAIN_LOG"
