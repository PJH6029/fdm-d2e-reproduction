#!/usr/bin/env bash
set -euo pipefail

SOURCE_CONFIG="${SOURCE_CONFIG:-configs/model/idm_video_pair_d2e_full_raw112_paper_target.yaml}"
TARGET_CONFIG="${TARGET_CONFIG:-configs/model/idm_video_pair_d2e_full_raw112_keysoftmax_paper_target.yaml}"
OUTPUT="${OUTPUT:-artifacts/idm/g005_video_pair_raw112_keysoftmax_cache_migration_summary.json}"
PROGRESS_OUTPUT="${PROGRESS_OUTPUT:-artifacts/idm/g005_video_pair_raw112_keysoftmax_cache_migration_progress.json}"
RUN_SUMMARY="${RUN_SUMMARY:-artifacts/idm/g005_video_pair_raw112_keysoftmax_cache_migration_run.json}"
LOG_PATH="${LOG_PATH:-artifacts/idm/g005_video_pair_raw112_keysoftmax_cache_migration.log}"
GPU_MONITOR_LOG="${GPU_MONITOR_LOG:-artifacts/idm/g005_video_pair_raw112_keysoftmax_cache_migration_gpu_monitor.csv}"
PID_FILE="${PID_FILE:-outputs/cluster/g005_video_pair_raw112_keysoftmax_cache_migration.pid}"
FORCE="${FORCE:-0}"

mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$PROGRESS_OUTPUT")" "$(dirname "$RUN_SUMMARY")" "$(dirname "$LOG_PATH")" "$(dirname "$GPU_MONITOR_LOG")" "$(dirname "$PID_FILE")"
echo "$$" >"$PID_FILE"

START_EPOCH="$(date +%s)"
MONITOR_PID=""
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
    --format=csv \
    -l 30 >"$GPU_MONITOR_LOG" 2>&1 &
  MONITOR_PID="$!"
fi

cleanup() {
  if [[ -n "${MONITOR_PID:-}" ]]; then
    kill "$MONITOR_PID" >/dev/null 2>&1 || true
    wait "$MONITOR_PID" >/dev/null 2>&1 || true
  fi
  if [[ -f "$PID_FILE" ]] && [[ "$(cat "$PID_FILE" 2>/dev/null || true)" == "$$" ]]; then
    rm -f "$PID_FILE"
  fi
}
trap cleanup EXIT

set +e
(
  set -euo pipefail
  echo "started_at=$(date -Iseconds)"
  echo "git_head=$(git rev-parse HEAD)"
  echo "source_config=$SOURCE_CONFIG"
  echo "target_config=$TARGET_CONFIG"
  echo "output=$OUTPUT"
  echo "progress_output=$PROGRESS_OUTPUT"
  args=(
    scripts/migrate_video_idm_cache_keysoftmax.py
    --source-config "$SOURCE_CONFIG"
    --target-config "$TARGET_CONFIG"
    --output "$OUTPUT"
    --progress-output "$PROGRESS_OUTPUT"
  )
  if [[ "$FORCE" != "0" ]]; then
    args+=(--force)
  fi
  uv run python "${args[@]}"
  echo "finished_at=$(date -Iseconds)"
) 2>&1 | tee "$LOG_PATH"
RUN_STATUS="${PIPESTATUS[0]}"
set -e
cleanup
END_EPOCH="$(date +%s)"

uv run python - <<PY
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict | None:
    if not path.exists() or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _git_output(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _gpu_monitor_status(path: Path) -> dict:
    status = {"rows": 0, "unique_gpu_indices": []}
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return status
    index_col = None
    with path.open(newline="", encoding="utf-8") as handle:
        for raw_row in csv.reader(handle):
            row = [cell.strip() for cell in raw_row]
            if not row:
                continue
            lowered = [cell.lower() for cell in row]
            if "index" in lowered:
                index_col = lowered.index("index")
                continue
            if index_col is None:
                index_col = 1 if len(row) > 1 else 0
            if index_col < len(row):
                status["unique_gpu_indices"].append(row[index_col])
            status["rows"] += 1
    status["unique_gpu_indices"] = sorted(set(status["unique_gpu_indices"]))
    return status


output = Path("$OUTPUT")
progress = Path("$PROGRESS_OUTPUT")
gpu_monitor = Path("$GPU_MONITOR_LOG")
summary = _load(output) or {}
payload = {
    "schema": "g005_keysoftmax_cache_migration_run.v1",
    "source_config": "$SOURCE_CONFIG",
    "target_config": "$TARGET_CONFIG",
    "output": str(output),
    "progress_output": str(progress),
    "log_path": "$LOG_PATH",
    "gpu_monitor_log": str(gpu_monitor),
    "pid_file": "$PID_FILE",
    "exit_code": int("$RUN_STATUS"),
    "wall_clock_seconds": int("$END_EPOCH") - int("$START_EPOCH"),
    "git_head": _git_output(["rev-parse", "HEAD"]),
    "git_status_short": _git_output(["status", "--short"]),
    "summary_status": summary.get("status"),
    "train_cache_rows": summary.get("train_cache", {}).get("rows"),
    "target_cache_rows": summary.get("target_cache", {}).get("rows"),
    "train_cache_bytes": summary.get("train_cache", {}).get("bytes"),
    "target_cache_bytes": summary.get("target_cache", {}).get("bytes"),
    "target_keyboard_classes": summary.get("target_keyboard_classes"),
    "target_category_vocab": summary.get("target_category_vocab"),
    "summary_sha256": _sha256(output),
    "progress_sha256": _sha256(progress),
    "gpu_monitor_sha256": _sha256(gpu_monitor),
    "gpu_monitor_status": _gpu_monitor_status(gpu_monitor),
    "claim_boundary": "Cache migration lifecycle evidence only; no training or model-quality claim.",
}
Path("$RUN_SUMMARY").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

exit "$RUN_STATUS"
