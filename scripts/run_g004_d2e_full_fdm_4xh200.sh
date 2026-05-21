#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/model/fdm_streaming_d2e_full_compact.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
EXPECTED_GPUS="${EXPECTED_GPUS:-4}"
LOG_PATH="${LOG_PATH:-artifacts/fdm/g004_d2e_full_fdm_4xh200.log}"
RUN_SUMMARY="${RUN_SUMMARY:-artifacts/fdm/g004_d2e_full_fdm_4xh200_run.json}"

mkdir -p "$(dirname "$LOG_PATH")" "$(dirname "$RUN_SUMMARY")" outputs/cluster

{
  echo "started_at=$(date -Iseconds)"
  echo "git_head=$(git rev-parse HEAD)"
  echo "config=$CONFIG"
  echo "nproc_per_node=$NPROC_PER_NODE"
  uv run python scripts/cluster_gpu_smoke.py --expected-gpus "$EXPECTED_GPUS"
  uv run torchrun --standalone --nproc-per-node="$NPROC_PER_NODE" scripts/train_fdm_streaming.py --config "$CONFIG"
  echo "finished_at=$(date -Iseconds)"
} 2>&1 | tee "$LOG_PATH"

uv run python - <<PY
from __future__ import annotations

import json
import subprocess
from pathlib import Path

summary_path = Path("outputs/fdm_streaming_d2e_full_compact/summary.json")
payload = {
    "schema": "g004_fdm_4xh200_run.v1",
    "config": "$CONFIG",
    "log_path": "$LOG_PATH",
    "nproc_per_node": int("$NPROC_PER_NODE"),
    "expected_gpus": int("$EXPECTED_GPUS"),
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
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
        }
    )
Path("$RUN_SUMMARY").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
