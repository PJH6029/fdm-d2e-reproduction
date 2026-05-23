#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/model/idm_streaming_d2e_full_surface_calibrated.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
EXPECTED_GPUS="${EXPECTED_GPUS:-4}"
LOG_PATH="${LOG_PATH:-artifacts/idm/g005_idm_surface_4xh200.log}"
RUN_SUMMARY="${RUN_SUMMARY:-artifacts/idm/g005_idm_surface_4xh200_run.json}"
GPU_MONITOR_LOG="${GPU_MONITOR_LOG:-artifacts/idm/g005_idm_surface_4xh200_gpu_monitor.csv}"
PID_FILE="${PID_FILE:-outputs/cluster/g005_idm_surface_4xh200.pid}"
GPU_SMOKE_REPORT="${GPU_SMOKE_REPORT:-outputs/cluster/g005_idm_surface_gpu_smoke.json}"
RUN_CONFIG_RECORD="${RUN_CONFIG_RECORD:-outputs/cluster/g005_idm_surface_runtime_config_path.txt}"
SPLIT_STATS_CONFIG="${SPLIT_STATS_CONFIG:-configs/eval/g005_idm_surface_split_statistics.yaml}"
PAPER_TARGET_CONFIG="${PAPER_TARGET_CONFIG:-configs/eval/g005_idm_surface_paper_target.yaml}"
STATS_SEED_PATH="${STATS_SEED_PATH:-outputs/idm_streaming_d2e_full_compact_accel64/streaming_stats.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/idm_streaming_d2e_full_surface_calibrated}"
PRESEED_STATS="${PRESEED_STATS:-1}"
ALLOW_CACHE_BUILD="${ALLOW_CACHE_BUILD:-0}"
BUILD_SPLIT_STATS="${BUILD_SPLIT_STATS:-1}"
BUILD_PAPER_METRICS="${BUILD_PAPER_METRICS:-1}"
VALIDATE_G005="${VALIDATE_G005:-1}"

mkdir -p "$(dirname "$LOG_PATH")" "$(dirname "$RUN_SUMMARY")" "$(dirname "$GPU_MONITOR_LOG")" "$(dirname "$PID_FILE")" outputs/cluster "$OUTPUT_DIR"
echo "$$" >"$PID_FILE"
RUN_CONFIG="$CONFIG"
rm -f "$RUN_CONFIG_RECORD"

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
  echo "split_stats_config=$SPLIT_STATS_CONFIG"
  echo "paper_target_config=$PAPER_TARGET_CONFIG"
  uv run python scripts/cluster_gpu_smoke.py --expected-gpus "$EXPECTED_GPUS" --report "$GPU_SMOKE_REPORT"
  if [[ "$PRESEED_STATS" != "0" && ! -s "$OUTPUT_DIR/streaming_stats.json" && -s "$STATS_SEED_PATH" ]]; then
    echo "preseeding streaming stats from $STATS_SEED_PATH"
    cp "$STATS_SEED_PATH" "$OUTPUT_DIR/streaming_stats.json"
  fi
  RUN_CONFIG="$CONFIG"
  if [[ "$ALLOW_CACHE_BUILD" == "0" ]]; then
    CACHE_DIR="$(uv run python - "$CONFIG" <<'PY'
from __future__ import annotations
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle).get("training_cache_dir", ""))
PY
)"
    if [[ -n "$CACHE_DIR" && ! -d "$CACHE_DIR" ]]; then
      RUN_CONFIG="outputs/cluster/g005_idm_surface_runtime_no_cache.yaml"
      echo "training cache $CACHE_DIR is absent; writing no-cache runtime config to $RUN_CONFIG"
      uv run python - "$CONFIG" "$RUN_CONFIG" <<'PY'
from __future__ import annotations
import json, sys
from pathlib import Path
src = Path(sys.argv[1])
dst = Path(sys.argv[2])
config = json.loads(src.read_text(encoding="utf-8"))
config["source_config_path"] = str(src)
config["runtime_cache_policy"] = {
    "training_cache_dir_removed": config.pop("training_cache_dir", None),
    "reason": "cache_missing_and_ALLOW_CACHE_BUILD_0",
}
config.pop("training_cache_num_workers", None)
config.pop("training_cache_shard_assignment", None)
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
      echo "$RUN_CONFIG" > "$RUN_CONFIG_RECORD"
    fi
  fi
  uv run torchrun --standalone --nproc-per-node="$NPROC_PER_NODE" scripts/train_idm_streaming.py \
    --config "$RUN_CONFIG" \
    --require-torch
  if [[ "$BUILD_SPLIT_STATS" != "0" ]]; then
    uv run python scripts/build_split_statistical_comparisons.py --config "$SPLIT_STATS_CONFIG"
  fi
  if [[ "$BUILD_PAPER_METRICS" != "0" ]]; then
    uv run python scripts/build_g005_idm_paper_metrics.py --config "$PAPER_TARGET_CONFIG"
  fi
  echo "finished_at=$(date -Iseconds)"
) 2>&1 | tee "$LOG_PATH"
RUN_STATUS="${PIPESTATUS[0]}"
set -e
cleanup_monitor
END_EPOCH="$(date +%s)"
if [[ -s "$RUN_CONFIG_RECORD" ]]; then
  RUN_CONFIG="$(cat "$RUN_CONFIG_RECORD")"
fi

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

summary_path = Path("artifacts/idm/idm_streaming_d2e_full_surface_calibrated_summary.json")
metadata_path = Path("outputs/idm_streaming_d2e_full_surface_calibrated/checkpoint_metadata.json")
metrics_path = Path("outputs/idm_streaming_d2e_full_surface_calibrated/metrics.json")
paper_metrics_path = Path("artifacts/idm/g005_idm_surface_paper_metrics.json")
split_stats_path = Path("artifacts/eval/g005_idm_surface_split_statistical_comparisons_summary.json")
gpu_monitor_path = Path("$GPU_MONITOR_LOG")
summary = _load(summary_path)
metadata = _load(metadata_path)
split_stats = _load(split_stats_path)
paper_metrics = _load(paper_metrics_path)
payload = {
    "schema": "g005_idm_surface_4xh200_run.v1",
    "config": "$CONFIG",
    "runtime_config": "${RUN_CONFIG:-$CONFIG}",
    "log_path": "$LOG_PATH",
    "gpu_monitor_log": "$GPU_MONITOR_LOG",
    "pid_file": "$PID_FILE",
    "nproc_per_node": int("$NPROC_PER_NODE"),
    "expected_gpus": int("$EXPECTED_GPUS"),
    "exit_code": int("$RUN_STATUS"),
    "wall_clock_seconds": int("$END_EPOCH") - int("$START_EPOCH"),
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
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
    "split_stats_outputs": split_stats.get("outputs", []) if split_stats else [],
    "paper_target_config": "$PAPER_TARGET_CONFIG",
    "paper_metrics_path": str(paper_metrics_path),
    "paper_metrics_status": paper_metrics.get("status") if paper_metrics else None,
    "paper_metrics_rows": (paper_metrics or {}).get("alignment", {}).get("rows_seen") if paper_metrics else None,
    "checkpoint_path": (metadata or {}).get("checkpoint_path"),
    "train_records": (metadata or {}).get("train_records"),
    "target_records": (metadata or {}).get("target_records"),
    "prediction_resume": (summary or {}).get("prediction_resume") if summary else None,
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
audit_path = Path("artifacts/idm/g005_idm_surface_paper_target_audit.json")
audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
payload["validation_exit_code"] = int("$VALIDATION_STATUS")
payload["g005_audit_path"] = str(audit_path)
payload["g005_audit_status"] = audit.get("status") if audit else None
payload["g005_audit_error_count"] = audit.get("error_count") if audit else None
run_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n")
print(json.dumps({"validation_exit_code": int("$VALIDATION_STATUS"), "g005_audit_status": payload["g005_audit_status"]}, sort_keys=True))
PY
fi

if [[ "$RUN_STATUS" != "0" ]]; then
  exit "$RUN_STATUS"
fi
exit "$VALIDATION_STATUS"
