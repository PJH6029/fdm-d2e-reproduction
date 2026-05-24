#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/model/idm_video_stack_d2e_full_luma96_offsets012_keysoftmax_paper_target.yaml}"
MODEL_SLUG="${MODEL_SLUG:-g005_idm_video_stack_luma96_offsets012_keysoftmax}"
LOG_PATH="${LOG_PATH:-artifacts/idm/g005_idm_video_stack_luma96_offsets012_keysoftmax_precompute.log}"
PID_FILE="${PID_FILE:-outputs/cluster/g005_idm_video_stack_luma96_offsets012_keysoftmax_precompute.pid}"
RUN_SUMMARY="${RUN_SUMMARY:-artifacts/idm/g005_idm_video_stack_luma96_offsets012_keysoftmax_precompute_run.json}"

mkdir -p "$(dirname "$LOG_PATH")" "$(dirname "$PID_FILE")" "$(dirname "$RUN_SUMMARY")" outputs/cluster
echo "$$" >"$PID_FILE"

cleanup_pid_file() {
  if [[ -f "$PID_FILE" ]] && [[ "$(cat "$PID_FILE" 2>/dev/null || true)" == "$$" ]]; then
    rm -f "$PID_FILE"
  fi
}
trap cleanup_pid_file EXIT

START_EPOCH="$(date +%s)"
set +e
(
  set -euo pipefail
  echo "started_at=$(date -Iseconds)"
  echo "git_head=$(git rev-parse HEAD)"
  echo "config=$CONFIG"
  uv run python scripts/precompute_video_idm_cache.py --config "$CONFIG"
  echo "finished_at=$(date -Iseconds)"
) 2>&1 | tee "$LOG_PATH"
STATUS="${PIPESTATUS[0]}"
set -e
END_EPOCH="$(date +%s)"

uv run python - <<PY
from __future__ import annotations
import hashlib
import json
import subprocess
from pathlib import Path
from fdm_d2e.config import load_config

def _load(path: Path) -> dict | None:
    if path.exists() and path.is_file():
        return json.loads(path.read_text())
    return None

def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()

def _git_output(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None

config = load_config("$CONFIG")
summary_path = Path(config.get("cache_summary_out", "artifacts/idm/g005_video_stack_luma96_offsets012_keysoftmax_cache_precompute_summary.json"))
stats_path = Path(config.get("stats_path", "outputs/idm_video_stack_d2e_full_luma96_offsets012_keysoftmax_paper_target/video_idm_stats.json"))
summary = _load(summary_path)
payload = {
    "schema": "${MODEL_SLUG}_precompute_run.v1",
    "status": "pass" if int("$STATUS") == 0 else "fail",
    "exit_code": int("$STATUS"),
    "config": "$CONFIG",
    "log_path": "$LOG_PATH",
    "wall_clock_seconds": int("$END_EPOCH") - int("$START_EPOCH"),
    "git_head": _git_output(["rev-parse", "HEAD"]),
    "git_status_short": _git_output(["status", "--short"]),
    "cache_summary_path": str(summary_path),
    "cache_summary_exists": summary is not None,
    "cache_summary_sha256": _sha256(summary_path),
    "stats_path": str(stats_path),
    "stats_exists": stats_path.exists(),
    "stats_sha256": _sha256(stats_path),
    "train_cache_rows": (summary or {}).get("train_cache", {}).get("rows"),
    "target_cache_rows": (summary or {}).get("target_cache", {}).get("rows"),
    "train_cache_bytes": (summary or {}).get("train_cache", {}).get("bytes"),
    "target_cache_bytes": (summary or {}).get("target_cache", {}).get("bytes"),
    "claim_boundary": "Video cache precompute is preprocessing evidence only; it is not IDM quality evidence.",
}
Path("$RUN_SUMMARY").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

exit "$STATUS"
