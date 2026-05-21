#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/model/idm_streaming_d2e_full_compact.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
EXPECTED_GPUS="${EXPECTED_GPUS:-4}"
LOG_PATH="${LOG_PATH:-artifacts/idm/g003_d2e_full_idm_4xh200_train.log}"
RUN_SUMMARY="${RUN_SUMMARY:-artifacts/idm/g003_d2e_full_idm_4xh200_train_run.json}"
GPU_MONITOR_LOG="${GPU_MONITOR_LOG:-artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor.csv}"
TRAIN_RECORDS="${TRAIN_RECORDS:-outputs/data/d2e_full_corpus/train_core.jsonl}"
TARGET_RECORDS="${TARGET_RECORDS:-outputs/data/d2e_full_corpus/target_all_eval.jsonl}"
BUILD_SPLIT_STATS="${BUILD_SPLIT_STATS:-1}"
SPLIT_STATS_CONFIG="${SPLIT_STATS_CONFIG:-configs/eval/g003_split_statistics.yaml}"
SPLIT_STATS_SUMMARY="${SPLIT_STATS_SUMMARY:-artifacts/eval/g003_split_statistical_comparisons_summary.json}"
export BUILD_SPLIT_STATS SPLIT_STATS_CONFIG SPLIT_STATS_SUMMARY

mkdir -p "$(dirname "$LOG_PATH")" "$(dirname "$RUN_SUMMARY")" "$(dirname "$GPU_MONITOR_LOG")" outputs/cluster

if [[ ! -s "$TRAIN_RECORDS" || ! -s "$TARGET_RECORDS" ]]; then
  echo "missing merged full-corpus JSONLs: TRAIN_RECORDS=$TRAIN_RECORDS TARGET_RECORDS=$TARGET_RECORDS" >&2
  exit 3
fi

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
  echo "nproc_per_node=$NPROC_PER_NODE"
  echo "expected_gpus=$EXPECTED_GPUS"
  echo "train_records=$TRAIN_RECORDS"
  echo "target_records=$TARGET_RECORDS"
  echo "gpu_monitor_log=$GPU_MONITOR_LOG"
  echo "build_split_stats=$BUILD_SPLIT_STATS"
  echo "split_stats_config=$SPLIT_STATS_CONFIG"
  echo "split_stats_summary=$SPLIT_STATS_SUMMARY"
  uv run python scripts/cluster_gpu_smoke.py --expected-gpus "$EXPECTED_GPUS"
  uv run torchrun --standalone --nproc-per-node="$NPROC_PER_NODE" scripts/train_idm_streaming.py --config "$CONFIG" --require-torch
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
import subprocess
from pathlib import Path

summary_path = Path("artifacts/idm/idm_streaming_d2e_full_compact_summary.json")
metadata_path = Path("outputs/idm_streaming_d2e_full_compact/checkpoint_metadata.json")
split_stats_summary_path = Path("$SPLIT_STATS_SUMMARY")
payload = {
    "schema": "g003_idm_4xh200_train_run.v1",
    "config": "$CONFIG",
    "log_path": "$LOG_PATH",
    "gpu_monitor_log": "$GPU_MONITOR_LOG",
    "nproc_per_node": int("$NPROC_PER_NODE"),
    "expected_gpus": int("$EXPECTED_GPUS"),
    "exit_code": int("$RUN_STATUS"),
    "wall_clock_seconds": int("$END_EPOCH") - int("$START_EPOCH"),
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "train_records": "$TRAIN_RECORDS",
    "target_records": "$TARGET_RECORDS",
    "build_split_stats": "$BUILD_SPLIT_STATS" != "0",
    "split_stats_config": "$SPLIT_STATS_CONFIG",
    "split_stats_summary_path": str(split_stats_summary_path),
    "split_stats_summary_exists": split_stats_summary_path.exists(),
    "summary_path": str(summary_path),
    "summary_exists": summary_path.exists(),
    "metadata_path": str(metadata_path),
    "metadata_exists": metadata_path.exists(),
}
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    metadata = summary.get("metadata", {})
    payload.update(
        {
            "model": metadata.get("model"),
            "train_records_count": metadata.get("train_records"),
            "target_records_count": metadata.get("target_records"),
            "checkpoint_path": metadata.get("checkpoint_path"),
            "metrics_path": metadata.get("metrics_path"),
            "label_quality_report_path": metadata.get("label_quality_report_path"),
            "statistical_comparison_path": metadata.get("statistical_comparison_path"),
            "convergence_report_path": metadata.get("convergence_report_path"),
            "convergence_plateau_met": metadata.get("convergence_plateau_met"),
            "distributed": metadata.get("distributed"),
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
