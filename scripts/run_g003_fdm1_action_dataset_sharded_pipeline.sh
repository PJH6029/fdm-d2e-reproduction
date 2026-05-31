#!/usr/bin/env bash
set -euo pipefail

# Shard-parallel reset G003 pipeline for full D2E-480p action-token materialization.
# This is CPU/IO-heavy and intended for MLXP/PVC workspaces after the branch is pulled.
# Env knobs:
#   NUM_SHARDS=16
#   MAX_PARALLEL_SHARDS=8
#   CACHE_DIR=/root/work/data/d2e/cache
#   SHARD_ROOT=outputs/data/fdm1_d2e_480p_window_records_shards
#   MERGED_OUTPUT_DIR=outputs/data/fdm1_d2e_480p_window_records
#   EXTRACT_EXTRA_ARGS='--max-bins-per-recording 1000'
#   PREFLIGHT_EXTRA_ARGS='--require-pod --min-free-gb 100'

export PATH="$HOME/.local/bin:$PATH"

NUM_SHARDS="${NUM_SHARDS:-16}"
MAX_PARALLEL_SHARDS="${MAX_PARALLEL_SHARDS:-8}"
SHARD_ROOT="${SHARD_ROOT:-outputs/data/fdm1_d2e_480p_window_records_shards}"
MERGED_OUTPUT_DIR="${MERGED_OUTPUT_DIR:-outputs/data/fdm1_d2e_480p_window_records}"
DECODE_SUMMARY="${DECODE_SUMMARY:-artifacts/sources/fdm1_d2e_480p_window_records_decode_summary.json}"
LOG_DIR="${LOG_DIR:-artifacts/logs/fdm1_g003_shards}"
CACHE_DIR="${CACHE_DIR:-/root/work/data/d2e/cache}"
PIPELINE_SUMMARY="${PIPELINE_SUMMARY:-artifacts/cluster/fdm1_g003_sharded_pipeline_summary.json}"
PID_DIR="${PID_DIR:-outputs/cluster/fdm1_g003_shards}"

mkdir -p "$SHARD_ROOT" "$MERGED_OUTPUT_DIR" "$LOG_DIR" "$PID_DIR" artifacts/cluster artifacts/sources artifacts/reports

uv run python scripts/preflight_g003_fdm1_action_dataset_pod.py \
  ${PREFLIGHT_EXTRA_ARGS:-}

batch_pids=()
failures=0
wait_batch() {
  local item pid shard
  for item in "${batch_pids[@]}"; do
    pid="${item%%:*}"
    shard="${item##*:}"
    if ! wait "$pid"; then
      echo "shard $shard failed; see $LOG_DIR/shard_${shard}.log" >&2
      failures=$((failures + 1))
    fi
  done
  batch_pids=()
}

for shard_index in $(seq 0 $((NUM_SHARDS - 1))); do
  shard_dir="$SHARD_ROOT/shard_${shard_index}"
  mkdir -p "$shard_dir"
  (
    set -euo pipefail
    uv run python scripts/extract_d2e_full_corpus.py \
      --config configs/data/fdm1_d2e_480p_full_corpus_extract.yaml \
      --output-dir "$shard_dir" \
      --summary-out "$shard_dir/decode_summary.json" \
      --cache-dir "$CACHE_DIR" \
      --shard-index "$shard_index" \
      --num-shards "$NUM_SHARDS" \
      ${EXTRACT_EXTRA_ARGS:-}
  ) > "$LOG_DIR/shard_${shard_index}.log" 2>&1 &
  pid=$!
  echo "$pid" > "$PID_DIR/shard_${shard_index}.pid"
  batch_pids+=("$pid:$shard_index")
  if [[ "${#batch_pids[@]}" -ge "$MAX_PARALLEL_SHARDS" ]]; then
    wait_batch
  fi
done

wait_batch

uv run python - <<'PY'
from __future__ import annotations
import json, os, time
from pathlib import Path
num_shards = int(os.environ.get("NUM_SHARDS", "16"))
shard_root = Path(os.environ.get("SHARD_ROOT", "outputs/data/fdm1_d2e_480p_window_records_shards"))
log_dir = Path(os.environ.get("LOG_DIR", "artifacts/logs/fdm1_g003_shards"))
out = Path(os.environ.get("PIPELINE_SUMMARY", "artifacts/cluster/fdm1_g003_sharded_pipeline_summary.json"))
shards = []
for idx in range(num_shards):
    summary_path = shard_root / f"shard_{idx}" / "decode_summary.json"
    log_path = log_dir / f"shard_{idx}.log"
    payload = json.loads(summary_path.read_text()) if summary_path.exists() else None
    shards.append({
        "index": idx,
        "summary_path": str(summary_path),
        "summary_exists": summary_path.exists(),
        "log_path": str(log_path),
        "log_exists": log_path.exists(),
        "selected_recording_variants": payload.get("selected_recording_variants") if payload else None,
        "failures": payload.get("failures") if payload else None,
    })
summary = {
    "schema": "fdm1_g003_sharded_pipeline_summary.v1",
    "status": "pass" if all(item["summary_exists"] and not item.get("failures") for item in shards) else "fail",
    "timestamp_unix": time.time(),
    "num_shards": num_shards,
    "shard_root": str(shard_root),
    "log_dir": str(log_dir),
    "shards": shards,
    "claim_boundary": "Shard extraction summary only; G003 completion still requires merge, finalization, completion audit, evidence bundle, and OMX checkpoint.",
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"sharded_extraction_status": summary["status"], "summary": str(out)}, sort_keys=True))
if summary["status"] != "pass":
    raise SystemExit(2)
PY

uv run python scripts/merge_d2e_full_corpus_shards.py \
  --shard-root "$SHARD_ROOT" \
  --output-dir "$MERGED_OUTPUT_DIR" \
  --summary-out "$DECODE_SUMMARY" \
  --expected-shards "$NUM_SHARDS"

uv run python scripts/finalize_g003_fdm1_action_dataset.py \
  --config configs/data/fdm1_g003_action_dataset_finalization.yaml \
  ${FINALIZE_EXTRA_ARGS:-}

uv run python scripts/build_fdm1_g003_evidence_bundle.py \
  --completion-config configs/eval/fdm1_g003_action_dataset_completion.yaml \
  ${BUNDLE_EXTRA_ARGS:-}

uv run python scripts/monitor_g003_fdm1_action_dataset_pod.py \
  --refresh-audit \
  --build-bundle-if-pass \
  ${MONITOR_EXTRA_ARGS:-}

uv run python scripts/build_fdm1_g003_checkpoint_handoff.py \
  --allow-blocked \
  ${CHECKPOINT_HANDOFF_EXTRA_ARGS:-}

if [[ "$failures" -ne 0 ]]; then
  echo "one or more shard processes failed before summary generation: $failures" >&2
  exit 2
fi
