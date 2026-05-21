#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/model/fdm_streaming_d2e_full_compact.yaml}"
IDM_PREDICT_CONFIG="${IDM_PREDICT_CONFIG:-configs/model/idm_streaming_d2e_full_compact_predict_fdm_train.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
EXPECTED_GPUS="${EXPECTED_GPUS:-4}"
LOG_PATH="${LOG_PATH:-artifacts/fdm/g004_d2e_full_fdm_4xh200.log}"
RUN_SUMMARY="${RUN_SUMMARY:-artifacts/fdm/g004_d2e_full_fdm_4xh200_run.json}"
GPU_MONITOR_LOG="${GPU_MONITOR_LOG:-artifacts/fdm/g004_d2e_full_fdm_4xh200_gpu_monitor.csv}"
FDM_LABELS="${FDM_LABELS:-outputs/idm_streaming_d2e_full_compact/fdm_train_core_pseudolabels/pseudolabels.jsonl}"
BUILD_SPLIT_STATS="${BUILD_SPLIT_STATS:-1}"
SPLIT_STATS_CONFIG="${SPLIT_STATS_CONFIG:-configs/eval/g004_split_statistics.yaml}"
SPLIT_STATS_SUMMARY="${SPLIT_STATS_SUMMARY:-artifacts/eval/g004_split_statistical_comparisons_summary.json}"
export BUILD_SPLIT_STATS SPLIT_STATS_CONFIG SPLIT_STATS_SUMMARY

mkdir -p "$(dirname "$LOG_PATH")" "$(dirname "$RUN_SUMMARY")" "$(dirname "$GPU_MONITOR_LOG")" outputs/cluster

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
  fi
}
trap cleanup_monitor EXIT

set +e
(
  set -euo pipefail
  echo "started_at=$(date -Iseconds)"
  echo "git_head=$(git rev-parse HEAD)"
  echo "config=$CONFIG"
  echo "idm_predict_config=$IDM_PREDICT_CONFIG"
  echo "nproc_per_node=$NPROC_PER_NODE"
  echo "gpu_monitor_log=$GPU_MONITOR_LOG"
  echo "build_split_stats=$BUILD_SPLIT_STATS"
  echo "split_stats_config=$SPLIT_STATS_CONFIG"
  echo "split_stats_summary=$SPLIT_STATS_SUMMARY"
  uv run python scripts/cluster_gpu_smoke.py --expected-gpus "$EXPECTED_GPUS"
  if [[ ! -s "$FDM_LABELS" ]]; then
    echo "missing FDM train-core pseudo-labels at $FDM_LABELS; generating with trained G003 IDM checkpoint"
    uv run python scripts/predict_idm_streaming.py --config "$IDM_PREDICT_CONFIG"
  fi
  uv run torchrun --standalone --nproc-per-node="$NPROC_PER_NODE" scripts/train_fdm_streaming.py --config "$CONFIG"
  if [[ "$BUILD_SPLIT_STATS" != "0" ]]; then
    uv run python scripts/build_split_statistical_comparisons.py --config "$SPLIT_STATS_CONFIG"
  fi
  echo "finished_at=$(date -Iseconds)"
) 2>&1 | tee "$LOG_PATH"
RUN_STATUS="${PIPESTATUS[0]}"
set -e
cleanup_monitor
trap - EXIT
END_EPOCH="$(date +%s)"

uv run python - <<PY
from __future__ import annotations

import json
import csv
import hashlib
import subprocess
from pathlib import Path

def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def _gpu_monitor_status(path: Path, expected_gpus: int) -> dict:
    status = {
        "rows": 0,
        "unique_gpu_indices": [],
        "expected_gpus": expected_gpus,
        "covers_expected_gpus": False,
    }
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

summary_path = Path("outputs/fdm_streaming_d2e_full_compact/summary.json")
gpu_monitor_path = Path("$GPU_MONITOR_LOG")
split_stats_summary_path = Path("$SPLIT_STATS_SUMMARY")
payload = {
    "schema": "g004_fdm_4xh200_run.v1",
    "config": "$CONFIG",
    "idm_predict_config": "$IDM_PREDICT_CONFIG",
    "log_path": "$LOG_PATH",
    "gpu_monitor_log": "$GPU_MONITOR_LOG",
    "fdm_labels": "$FDM_LABELS",
    "nproc_per_node": int("$NPROC_PER_NODE"),
    "expected_gpus": int("$EXPECTED_GPUS"),
    "exit_code": int("$RUN_STATUS"),
    "wall_clock_seconds": int("$END_EPOCH") - int("$START_EPOCH"),
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "build_split_stats": "$BUILD_SPLIT_STATS" != "0",
    "split_stats_config": "$SPLIT_STATS_CONFIG",
    "split_stats_summary_path": str(split_stats_summary_path),
    "split_stats_summary_exists": split_stats_summary_path.exists(),
    "gpu_monitor_sha256": _sha256(gpu_monitor_path),
    "gpu_monitor_status": _gpu_monitor_status(gpu_monitor_path, int("$EXPECTED_GPUS")),
    "summary_path": str(summary_path),
    "summary_exists": summary_path.exists(),
}
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    checkpoint = summary.get("checkpoint", {})
    payload.update(
        {
            "model": checkpoint.get("model"),
            "num_training_examples": checkpoint.get("num_training_examples"),
            "target_examples": checkpoint.get("target_examples"),
            "predictions_path": checkpoint.get("predictions_path"),
            "metrics_path": checkpoint.get("metrics_path"),
            "statistical_comparison_path": checkpoint.get("statistical_comparison_path"),
            "convergence_report_path": checkpoint.get("convergence_report_path"),
            "convergence_plateau_met": checkpoint.get("convergence_plateau_met"),
        }
    )
if split_stats_summary_path.exists():
    split_stats = json.loads(split_stats_summary_path.read_text())
    payload.update(
        {
            "split_stats_status": split_stats.get("status"),
            "split_stats_outputs": split_stats.get("outputs", []),
        }
    )
Path("$RUN_SUMMARY").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
exit "$RUN_STATUS"
