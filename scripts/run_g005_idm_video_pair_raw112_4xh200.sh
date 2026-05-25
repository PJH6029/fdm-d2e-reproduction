#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/model/idm_video_pair_d2e_full_raw112_paper_target.yaml}"
MODEL_SLUG="${MODEL_SLUG:-g005_idm_video_pair_raw112}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
EXPECTED_GPUS="${EXPECTED_GPUS:-4}"
LOG_PATH="${LOG_PATH:-artifacts/idm/g005_idm_video_pair_raw112_4xh200.log}"
RUN_SUMMARY="${RUN_SUMMARY:-artifacts/idm/g005_idm_video_pair_raw112_4xh200_run.json}"
GPU_MONITOR_LOG="${GPU_MONITOR_LOG:-artifacts/idm/g005_idm_video_pair_raw112_4xh200_gpu_monitor.csv}"
PID_FILE="${PID_FILE:-outputs/cluster/g005_idm_video_pair_raw112_4xh200.pid}"
GPU_SMOKE_REPORT="${GPU_SMOKE_REPORT:-outputs/cluster/g005_idm_video_pair_raw112_gpu_smoke.json}"
SPLIT_STATS_CONFIG="${SPLIT_STATS_CONFIG:-configs/eval/g005_idm_video_pair_raw112_split_statistics.yaml}"
PAPER_TARGET_CONFIG="${PAPER_TARGET_CONFIG:-configs/eval/g005_idm_video_pair_raw112_paper_target.yaml}"
BUILD_SPLIT_STATS="${BUILD_SPLIT_STATS:-1}"
BUILD_PAPER_METRICS="${BUILD_PAPER_METRICS:-1}"
VALIDATE_G005="${VALIDATE_G005:-1}"
MAX_TARGET_EXAMPLES="${MAX_TARGET_EXAMPLES:-}"
SKIP_PREDICTION="${SKIP_PREDICTION:-0}"
PAPER_MAX_ROWS="${PAPER_MAX_ROWS:-$MAX_TARGET_EXAMPLES}"
EPOCHS_OVERRIDE="${EPOCHS_OVERRIDE:-}"
BATCH_SIZE_OVERRIDE="${BATCH_SIZE_OVERRIDE:-}"
EVAL_BATCH_SIZE_OVERRIDE="${EVAL_BATCH_SIZE_OVERRIDE:-}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-outputs/cluster/${MODEL_SLUG}_runtime_config.yaml}"
RUNTIME_PAPER_TARGET_CONFIG="${RUNTIME_PAPER_TARGET_CONFIG:-outputs/cluster/${MODEL_SLUG}_runtime_paper_target.yaml}"
MLXP_RESERVATION_ID="${MLXP_RESERVATION_ID:-}"
MLXP_RESERVATION_START_AT="${MLXP_RESERVATION_START_AT:-}"
MLXP_RESERVATION_END_AT="${MLXP_RESERVATION_END_AT:-}"
MLXP_RESERVATION_NODE_ID="${MLXP_RESERVATION_NODE_ID:-}"
MLXP_RESERVATION_GPU_INDICES="${MLXP_RESERVATION_GPU_INDICES:-}"
MLXP_RESERVATION_POD_NAME="${MLXP_RESERVATION_POD_NAME:-}"
MLXP_RESERVATION_CHECKED_AT="${MLXP_RESERVATION_CHECKED_AT:-}"

mkdir -p "$(dirname "$LOG_PATH")" "$(dirname "$RUN_SUMMARY")" "$(dirname "$GPU_MONITOR_LOG")" "$(dirname "$PID_FILE")" outputs/cluster
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

cleanup_monitor() {
  if [[ -n "${MONITOR_PID:-}" ]]; then
    kill "$MONITOR_PID" >/dev/null 2>&1 || true
    wait "$MONITOR_PID" >/dev/null 2>&1 || true
    MONITOR_PID=""
  fi
}

cleanup_pid_file() {
  if [[ -f "$PID_FILE" ]] && [[ "$(cat "$PID_FILE" 2>/dev/null || true)" == "$$" ]]; then
    rm -f "$PID_FILE"
  fi
}

cleanup_all() {
  cleanup_monitor
  cleanup_pid_file
}
trap cleanup_all EXIT

set +e
(
  set -euo pipefail
  echo "started_at=$(date -Iseconds)"
  echo "git_head=$(git rev-parse HEAD)"
  echo "config=$CONFIG"
  echo "nproc_per_node=$NPROC_PER_NODE"
  echo "gpu_monitor_log=$GPU_MONITOR_LOG"
  echo "mlxp_reservation_id=${MLXP_RESERVATION_ID:-none}"
  echo "mlxp_reservation_end_at=${MLXP_RESERVATION_END_AT:-none}"
  echo "runtime_overrides=max_target_examples=${MAX_TARGET_EXAMPLES:-none} skip_prediction=$SKIP_PREDICTION epochs=${EPOCHS_OVERRIDE:-none} batch_size=${BATCH_SIZE_OVERRIDE:-none} eval_batch_size=${EVAL_BATCH_SIZE_OVERRIDE:-none}"
  RUN_CONFIG="$CONFIG"
  if [[ -n "$MAX_TARGET_EXAMPLES" || "$SKIP_PREDICTION" != "0" || -n "$EPOCHS_OVERRIDE" || -n "$BATCH_SIZE_OVERRIDE" || -n "$EVAL_BATCH_SIZE_OVERRIDE" ]]; then
    RUN_CONFIG="$RUNTIME_CONFIG"
    uv run python - "$CONFIG" "$RUN_CONFIG" "$MAX_TARGET_EXAMPLES" "$SKIP_PREDICTION" "$EPOCHS_OVERRIDE" "$BATCH_SIZE_OVERRIDE" "$EVAL_BATCH_SIZE_OVERRIDE" <<'PY'
from __future__ import annotations
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
max_target = sys.argv[3]
skip_prediction = sys.argv[4] != "0"
epochs_override = sys.argv[5]
batch_size_override = sys.argv[6]
eval_batch_size_override = sys.argv[7]
config = json.loads(src.read_text(encoding="utf-8"))
config["source_config_path"] = str(src)
config["runtime_overrides"] = {
    "max_target_examples": int(max_target) if max_target else None,
    "skip_prediction": skip_prediction,
    "epochs": int(epochs_override) if epochs_override else None,
    "batch_size": int(batch_size_override) if batch_size_override else None,
    "eval_batch_size": int(eval_batch_size_override) if eval_batch_size_override else None,
}
if max_target:
    config["max_target_examples"] = int(max_target)
if skip_prediction:
    config["skip_prediction"] = True
if epochs_override:
    config["epochs"] = int(epochs_override)
if batch_size_override:
    config["batch_size"] = int(batch_size_override)
if eval_batch_size_override:
    config["eval_batch_size"] = int(eval_batch_size_override)
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  fi
  echo "effective_config=$RUN_CONFIG"
  uv run python scripts/cluster_gpu_smoke.py --expected-gpus "$EXPECTED_GPUS" --report "$GPU_SMOKE_REPORT"
  uv run torchrun --standalone --nproc-per-node="$NPROC_PER_NODE" scripts/train_idm_video.py \
    --config "$RUN_CONFIG" \
    --require-torch
  if [[ "$BUILD_SPLIT_STATS" != "0" ]]; then
    uv run python scripts/build_split_statistical_comparisons.py --config "$SPLIT_STATS_CONFIG"
  fi
  if [[ "$BUILD_PAPER_METRICS" != "0" ]]; then
    RUN_PAPER_CONFIG="$PAPER_TARGET_CONFIG"
    if [[ -n "$PAPER_MAX_ROWS" ]]; then
      RUN_PAPER_CONFIG="$RUNTIME_PAPER_TARGET_CONFIG"
      uv run python - "$PAPER_TARGET_CONFIG" "$RUN_PAPER_CONFIG" "$PAPER_MAX_ROWS" <<'PY'
from __future__ import annotations
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
max_rows = int(sys.argv[3])
config = json.loads(src.read_text(encoding="utf-8"))
config["source_config_path"] = str(src)
metrics = dict(config.get("paper_metrics", {}))
metrics["max_rows"] = max_rows
config["paper_metrics"] = metrics
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
    fi
    uv run python scripts/build_g005_idm_paper_metrics.py --config "$RUN_PAPER_CONFIG"
  fi
  echo "finished_at=$(date -Iseconds)"
) 2>&1 | tee "$LOG_PATH"
RUN_STATUS="${PIPESTATUS[0]}"
set -e
cleanup_monitor
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
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()

def _load(path: Path) -> dict | None:
    if not path.exists() or not path.is_file():
        return None
    return json.loads(path.read_text())

def _git_output(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None

def _git_diff_sha256() -> str | None:
    try:
        diff = subprocess.check_output(["git", "diff", "--binary", "HEAD", "--"], stderr=subprocess.DEVNULL)
    except Exception:
        return None
    if not diff:
        return None
    return hashlib.sha256(diff).hexdigest()

def _gpu_monitor_status(path: Path, expected_gpus: int) -> dict:
    status = {"rows": 0, "unique_gpu_indices": [], "expected_gpus": expected_gpus, "covers_expected_gpus": False}
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
    status["covers_expected_gpus"] = len(status["unique_gpu_indices"]) >= expected_gpus
    return status

config_path = Path("$CONFIG")
model_config = _load(config_path) or {}
output_dir = Path(model_config.get("output_dir", "outputs/idm_video_pair_d2e_full_raw112_paper_target"))
summary_path = Path(model_config.get("summary_out", "artifacts/idm/idm_video_pair_d2e_full_raw112_paper_target_summary.json"))
metadata_path = output_dir / "checkpoint_metadata.json"
metrics_path = output_dir / "metrics.json"
paper_config = _load(Path("$PAPER_TARGET_CONFIG")) or {}
paper_paths = paper_config.get("paths", {}) if isinstance(paper_config.get("paths"), dict) else {}
paper_metrics_cfg = paper_config.get("paper_metrics", {}) if isinstance(paper_config.get("paper_metrics"), dict) else {}
paper_metrics_path = Path(paper_paths.get("paper_metrics") or paper_metrics_cfg.get("output_path") or "artifacts/idm/g005_idm_video_pair_raw112_paper_metrics.json")
split_config = _load(Path("$SPLIT_STATS_CONFIG")) or {}
split_stats_path = Path(split_config.get("summary_out", "artifacts/eval/g005_idm_video_pair_raw112_split_statistical_comparisons_summary.json"))
gpu_monitor_path = Path("$GPU_MONITOR_LOG")
summary = _load(summary_path)
metadata = _load(metadata_path)
split_stats = _load(split_stats_path)
paper_metrics = _load(paper_metrics_path)
payload = {
    "schema": "${MODEL_SLUG}_4xh200_run.v1",
    "config": "$CONFIG",
    "runtime_config": "$RUNTIME_CONFIG",
    "runtime_config_exists": Path("$RUNTIME_CONFIG").exists(),
    "runtime_overrides": {
        "max_target_examples": int("$MAX_TARGET_EXAMPLES") if "$MAX_TARGET_EXAMPLES" else None,
        "skip_prediction": "$SKIP_PREDICTION" != "0",
        "epochs": int("$EPOCHS_OVERRIDE") if "$EPOCHS_OVERRIDE" else None,
        "batch_size": int("$BATCH_SIZE_OVERRIDE") if "$BATCH_SIZE_OVERRIDE" else None,
        "eval_batch_size": int("$EVAL_BATCH_SIZE_OVERRIDE") if "$EVAL_BATCH_SIZE_OVERRIDE" else None,
    },
    "log_path": "$LOG_PATH",
    "gpu_monitor_log": "$GPU_MONITOR_LOG",
    "pid_file": "$PID_FILE",
    "nproc_per_node": int("$NPROC_PER_NODE"),
    "expected_gpus": int("$EXPECTED_GPUS"),
    "mlxp_reservation": {
        "reservation_id": "$MLXP_RESERVATION_ID" or None,
        "start_at": "$MLXP_RESERVATION_START_AT" or None,
        "end_at": "$MLXP_RESERVATION_END_AT" or None,
        "node_id": "$MLXP_RESERVATION_NODE_ID" or None,
        "gpu_indices": "$MLXP_RESERVATION_GPU_INDICES" or None,
        "pod_name": "$MLXP_RESERVATION_POD_NAME" or None,
        "checked_at": "$MLXP_RESERVATION_CHECKED_AT" or None,
    },
    "exit_code": int("$RUN_STATUS"),
    "wall_clock_seconds": int("$END_EPOCH") - int("$START_EPOCH"),
    "git_head": _git_output(["rev-parse", "HEAD"]),
    "git_status_short": _git_output(["status", "--short"]),
    "git_diff_sha256": _git_diff_sha256(),
    "gpu_smoke_report": "$GPU_SMOKE_REPORT",
    "gpu_monitor_sha256": _sha256(gpu_monitor_path),
    "gpu_monitor_status": _gpu_monitor_status(gpu_monitor_path, int("$EXPECTED_GPUS")),
    "summary_path": str(summary_path),
    "summary_exists": summary is not None,
    "metadata_path": str(metadata_path),
    "metadata_exists": metadata is not None,
    "metrics_path": str(metrics_path),
    "metrics_exists": metrics_path.exists(),
    "split_stats_config": "$SPLIT_STATS_CONFIG",
    "split_stats_summary_path": str(split_stats_path),
    "split_stats_status": split_stats.get("status") if split_stats else None,
    "paper_target_config": "$PAPER_TARGET_CONFIG",
    "paper_metrics_path": str(paper_metrics_path),
    "paper_metrics_status": paper_metrics.get("status") if paper_metrics else None,
    "paper_metrics_rows": (paper_metrics or {}).get("alignment", {}).get("rows_seen") if paper_metrics else None,
    "checkpoint_path": (metadata or {}).get("checkpoint_path"),
    "train_records": (metadata or {}).get("train_records"),
    "target_records": (metadata or {}).get("target_records"),
}
Path("$RUN_SUMMARY").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

VALIDATION_STATUS=0
if [[ "$RUN_STATUS" == "0" && "$VALIDATE_G005" != "0" ]]; then
  set +e
  uv run python scripts/validate_g005_idm_paper_target.py --config "$PAPER_TARGET_CONFIG"
  VALIDATION_STATUS="$?"
  set -e
  uv run python - <<PY
from __future__ import annotations
import json
from pathlib import Path
run_path = Path("$RUN_SUMMARY")
payload = json.loads(run_path.read_text())
payload["validation_exit_code"] = int("$VALIDATION_STATUS")
run_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n")
PY
fi

if [[ "$RUN_STATUS" != "0" ]]; then
  exit "$RUN_STATUS"
fi
if [[ "$VALIDATE_G005" != "0" && "$VALIDATION_STATUS" != "0" ]]; then
  exit "$VALIDATION_STATUS"
fi
