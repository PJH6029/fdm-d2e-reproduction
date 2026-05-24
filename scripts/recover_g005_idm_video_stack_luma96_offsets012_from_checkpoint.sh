#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/model/idm_video_stack_d2e_full_luma96_offsets012_keysoftmax_paper_target.yaml}"
MODEL_SLUG="${MODEL_SLUG:-g005_idm_video_stack_luma96_offsets012_keysoftmax}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/idm_video_stack_d2e_full_luma96_offsets012_keysoftmax_paper_target}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$OUTPUT_DIR/checkpoint.pt}"
PREDICTION_WORKERS="${PREDICTION_WORKERS:-4}"
PREDICTION_CUDA_DEVICES="${PREDICTION_CUDA_DEVICES:-0,1,2,3}"
PREDICTION_PARTS_DIR="${PREDICTION_PARTS_DIR:-$OUTPUT_DIR/prediction_parts}"
PREDICTION_SUMMARY="${PREDICTION_SUMMARY:-artifacts/idm/g005_idm_video_stack_luma96_offsets012_keysoftmax_prediction_summary.json}"
RECOVERY_SUMMARY="${RECOVERY_SUMMARY:-artifacts/idm/g005_idm_video_stack_luma96_offsets012_keysoftmax_checkpoint_recovery_run.json}"
LOG_PATH="${LOG_PATH:-artifacts/idm/g005_idm_video_stack_luma96_offsets012_keysoftmax_checkpoint_recovery.log}"
PID_FILE="${PID_FILE:-outputs/cluster/g005_idm_video_stack_luma96_offsets012_keysoftmax_checkpoint_recovery.pid}"
SPLIT_STATS_CONFIG="${SPLIT_STATS_CONFIG:-configs/eval/g005_idm_video_stack_luma96_offsets012_keysoftmax_split_statistics.yaml}"
PAPER_TARGET_CONFIG="${PAPER_TARGET_CONFIG:-configs/eval/g005_idm_video_stack_luma96_offsets012_keysoftmax_paper_target.yaml}"
VALIDATE_G005="${VALIDATE_G005:-1}"

mkdir -p "$(dirname "$LOG_PATH")" "$(dirname "$PID_FILE")" "$(dirname "$RECOVERY_SUMMARY")" outputs/cluster
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
  echo "checkpoint=$CHECKPOINT_PATH"
  echo "prediction_workers=$PREDICTION_WORKERS"
  uv run python scripts/recover_idm_video_outputs.py \
    --config "$CONFIG" \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --prediction-workers "$PREDICTION_WORKERS" \
    --prediction-cuda-devices "$PREDICTION_CUDA_DEVICES" \
    --prediction-parts-dir "$PREDICTION_PARTS_DIR" \
    --prediction-summary-out "$PREDICTION_SUMMARY" \
    --require-torch
  uv run python scripts/build_split_statistical_comparisons.py --config "$SPLIT_STATS_CONFIG"
  uv run python scripts/build_g005_idm_paper_metrics.py --config "$PAPER_TARGET_CONFIG"
  if [[ "$VALIDATE_G005" != "0" ]]; then
    uv run python scripts/validate_g005_idm_paper_target.py --config "$PAPER_TARGET_CONFIG"
  fi
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

paper_cfg = _load(Path("$PAPER_TARGET_CONFIG")) or {}
paths = paper_cfg.get("paths", {}) if isinstance(paper_cfg.get("paths"), dict) else {}
paper_metrics = _load(Path(paths.get("paper_metrics", "artifacts/idm/g005_idm_video_stack_luma96_offsets012_keysoftmax_paper_metrics.json")))
split_stats = _load(Path(paths.get("split_stats_summary", "artifacts/eval/g005_idm_video_stack_luma96_offsets012_keysoftmax_split_statistical_comparisons_summary.json")))
audit = _load(Path(paper_cfg.get("output_path", "artifacts/idm/g005_idm_video_stack_luma96_offsets012_keysoftmax_paper_target_audit.json")))
prediction_summary = _load(Path("$PREDICTION_SUMMARY"))
metrics = _load(Path("$OUTPUT_DIR") / "metrics.json")
payload = {
    "schema": "${MODEL_SLUG}_checkpoint_recovery_run.v1",
    "status": "pass" if int("$STATUS") == 0 else "fail",
    "exit_code": int("$STATUS"),
    "config": "$CONFIG",
    "checkpoint_path": "$CHECKPOINT_PATH",
    "output_dir": "$OUTPUT_DIR",
    "prediction_workers": int("$PREDICTION_WORKERS"),
    "prediction_cuda_devices": "$PREDICTION_CUDA_DEVICES",
    "prediction_parts_dir": "$PREDICTION_PARTS_DIR",
    "prediction_summary": "$PREDICTION_SUMMARY",
    "prediction_summary_exists": prediction_summary is not None,
    "prediction_rows": (prediction_summary or {}).get("target_records"),
    "paper_target_config": "$PAPER_TARGET_CONFIG",
    "paper_metrics_status": (paper_metrics or {}).get("status"),
    "paper_metrics_rows": (paper_metrics or {}).get("alignment", {}).get("rows_seen") if paper_metrics else None,
    "split_stats_status": (split_stats or {}).get("status"),
    "audit_status": (audit or {}).get("status"),
    "audit_error_count": (audit or {}).get("error_count"),
    "metrics": metrics,
    "log_path": "$LOG_PATH",
    "log_sha256": _sha256(Path("$LOG_PATH")),
    "wall_clock_seconds": int("$END_EPOCH") - int("$START_EPOCH"),
    "git_head": _git_output(["rev-parse", "HEAD"]),
    "git_status_short": _git_output(["status", "--short"]),
    "claim_boundary": "Checkpoint recovery rebuilds full-target video IDM predictions and validation artifacts; it does not retrain the checkpoint.",
}
Path("$RECOVERY_SUMMARY").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

exit "$STATUS"
