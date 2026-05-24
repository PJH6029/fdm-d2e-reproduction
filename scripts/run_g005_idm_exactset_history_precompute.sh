#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/model/idm_streaming_d2e_full_luma_pair_exactset_history_paper_target.yaml}"
MODEL_SLUG="${MODEL_SLUG:-g005_idm_exactset_history}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/idm_streaming_d2e_full_luma_pair_exactset_history_paper_target}"
WORKERS="${WORKERS:-32}"
LOG_PATH="${LOG_PATH:-artifacts/idm/g005_idm_exactset_history_precompute.log}"
PID_FILE="${PID_FILE:-outputs/cluster/g005_idm_exactset_history_precompute.pid}"
SUMMARY_PATH="${SUMMARY_PATH:-artifacts/idm/g005_idm_exactset_history_precompute_summary.json}"
PROGRESS_PATH="${PROGRESS_PATH:-artifacts/idm/g005_idm_exactset_history_precompute_progress.json}"
VALIDATION_PATH="${VALIDATION_PATH:-artifacts/idm/g005_idm_exactset_history_precomputed_cache_validation.json}"

mkdir -p "$(dirname "$LOG_PATH")" "$(dirname "$PID_FILE")" "$(dirname "$SUMMARY_PATH")" "$OUTPUT_DIR" outputs/cluster
echo "$$" >"$PID_FILE"

cleanup_pid_file() {
  if [[ -f "$PID_FILE" ]] && [[ "$(cat "$PID_FILE" 2>/dev/null || true)" == "$$" ]]; then
    rm -f "$PID_FILE"
  fi
}
trap cleanup_pid_file EXIT

set +e
(
  set -euo pipefail
  echo "started_at=$(date -Iseconds)"
  echo "git_head=$(git rev-parse HEAD)"
  echo "config=$CONFIG"
  echo "workers=$WORKERS"
  echo "summary=$SUMMARY_PATH"
  echo "progress=$PROGRESS_PATH"
  uv run python scripts/precompute_streaming_idm_stats.py --config "$CONFIG"
  uv run python scripts/precompute_streaming_idm_training_cache.py \
    --config "$CONFIG" \
    --workers "$WORKERS" \
    --stats-path "$OUTPUT_DIR/streaming_stats.json" \
    --output "$SUMMARY_PATH" \
    --progress-output "$PROGRESS_PATH"
  uv run python scripts/precompute_streaming_idm_training_cache.py \
    --config "$CONFIG" \
    --validate-only \
    --stats-path "$OUTPUT_DIR/streaming_stats.json" \
    --output "$VALIDATION_PATH"
  echo "finished_at=$(date -Iseconds)"
) 2>&1 | tee "$LOG_PATH"
STATUS="${PIPESTATUS[0]}"
set -e

uv run python - <<PY
from __future__ import annotations
import json
from pathlib import Path

summary_path = Path("$SUMMARY_PATH")
validation_path = Path("$VALIDATION_PATH")
progress_path = Path("$PROGRESS_PATH")
payload = {
    "schema": "${MODEL_SLUG}_precompute_run.v1",
    "status": "pass" if int("$STATUS") == 0 else "fail",
    "exit_code": int("$STATUS"),
    "config": "$CONFIG",
    "output_dir": "$OUTPUT_DIR",
    "workers": int("$WORKERS"),
    "log_path": "$LOG_PATH",
    "summary_path": str(summary_path),
    "summary_exists": summary_path.exists(),
    "validation_path": str(validation_path),
    "validation_exists": validation_path.exists(),
    "validation_status": (json.loads(validation_path.read_text()).get("status") if validation_path.exists() else None),
    "progress_path": str(progress_path),
    "progress_exists": progress_path.exists(),
    "claim_boundary": "Standalone stats/cache precompute is preprocessing evidence only; it is not IDM quality evidence.",
}
Path("$SUMMARY_PATH.run.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

exit "$STATUS"
