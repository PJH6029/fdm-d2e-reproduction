#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/model/idm_streaming_d2e_full_luma_pair_exactset_history_paper_target.yaml}"
SPLIT_STATS_CONFIG="${SPLIT_STATS_CONFIG:-configs/eval/g005_idm_exactset_history_split_statistics.yaml}"
PAPER_TARGET_CONFIG="${PAPER_TARGET_CONFIG:-configs/eval/g005_idm_exactset_history_paper_target.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/idm_streaming_d2e_full_luma_pair_exactset_history_paper_target}"
RUN_SUMMARY="${RUN_SUMMARY:-artifacts/idm/g005_idm_exactset_history_4xh200_run.json}"
RECOVERY_LOG="${RECOVERY_LOG:-artifacts/idm/g005_idm_exactset_history_checkpoint_recovery.log}"
RECOVERY_RUN="${RECOVERY_RUN:-artifacts/idm/g005_idm_exactset_history_checkpoint_recovery_run.json}"
GPU_MONITOR_LOG="${GPU_MONITOR_LOG:-artifacts/idm/g005_idm_exactset_history_4xh200_gpu_monitor.csv}"
PREDICTION_WORKERS="${PREDICTION_WORKERS:-64}"
PREDICTION_CUDA_DEVICES="${PREDICTION_CUDA_DEVICES:-0,1,2,3}"
SKIP_PSEUDOLABEL_VALIDATION="${SKIP_PSEUDOLABEL_VALIDATION:-1}"

mkdir -p "$(dirname "$RECOVERY_LOG")" "$(dirname "$RUN_SUMMARY")" "$(dirname "$RECOVERY_RUN")" "$OUTPUT_DIR"

START_EPOCH="$(date +%s)"
STARTED_AT="$(date -Iseconds)"
RECOVERY_STATUS=0
VALIDATION_FLAG=()
if [[ "$SKIP_PSEUDOLABEL_VALIDATION" != "0" ]]; then
  VALIDATION_FLAG=(--no-pseudolabel-validation)
fi

{
  echo "recovery_started_at=$STARTED_AT"
  echo "git_head=$(git rev-parse HEAD)"
  echo "config=$CONFIG"
  ls -lh "$OUTPUT_DIR/checkpoint.pt"
  rm -rf \
    "$OUTPUT_DIR/prediction_recovery_parts" \
    "$OUTPUT_DIR/predictions.jsonl" \
    "$OUTPUT_DIR/pseudolabels.jsonl" \
    "$OUTPUT_DIR/metrics.json" \
    "$OUTPUT_DIR/label_quality_report.json" \
    "$OUTPUT_DIR/statistical_comparison.json" \
    "$OUTPUT_DIR/checkpoint_metadata.json"
  uv run python scripts/recover_idm_streaming_outputs.py \
    --config "$CONFIG" \
    --prediction-workers "$PREDICTION_WORKERS" \
    --prediction-cuda-devices "$PREDICTION_CUDA_DEVICES" \
    "${VALIDATION_FLAG[@]}"
  uv run python scripts/build_split_statistical_comparisons.py --config "$SPLIT_STATS_CONFIG"
  uv run python scripts/build_g005_idm_paper_metrics.py --config "$PAPER_TARGET_CONFIG"
  uv run python scripts/validate_g005_idm_paper_target.py --config "$PAPER_TARGET_CONFIG"
  echo "recovery_finished_at=$(date -Iseconds)"
} >"$RECOVERY_LOG" 2>&1 || RECOVERY_STATUS="$?"

END_EPOCH="$(date +%s)"

uv run python - <<PY
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path


def _load(path: str | Path) -> dict | None:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    return json.loads(p.read_text())


def _sha256(path: str | Path) -> str | None:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _gpu_monitor_status(path: str | Path, expected_gpus: int) -> dict:
    status = {"rows": 0, "unique_gpu_indices": [], "expected_gpus": expected_gpus, "covers_expected_gpus": False}
    p = Path(path)
    if not p.exists() or not p.is_file():
        return status
    index_col = None
    with p.open(newline="", encoding="utf-8") as handle:
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


config = _load("$CONFIG") or {}
summary_path = config.get("summary_out", "artifacts/idm/idm_streaming_d2e_full_luma_pair_exactset_history_paper_target_summary.json")
metadata = _load("$OUTPUT_DIR/checkpoint_metadata.json")
summary = _load(summary_path)
split_stats = _load("artifacts/eval/g005_idm_exactset_history_split_statistical_comparisons_summary.json")
paper_metrics = _load("artifacts/idm/g005_idm_exactset_history_paper_metrics.json")
audit = _load("artifacts/idm/g005_idm_exactset_history_paper_target_audit.json")
precompute = _load("artifacts/idm/g005_idm_exactset_history_precomputed_cache_validation.json")
status = int("$RECOVERY_STATUS")
payload = {
    "schema": "g005_idm_exactset_history_4xh200_run.v1",
    "config": "$CONFIG",
    "runtime_config": "$CONFIG",
    "log_path": "artifacts/idm/g005_idm_exactset_history_4xh200.log",
    "recovery_log_path": "$RECOVERY_LOG",
    "recovery_run_summary_path": "$RECOVERY_RUN",
    "gpu_monitor_log": "$GPU_MONITOR_LOG",
    "nproc_per_node": 4,
    "expected_gpus": 4,
    "exit_code": status,
    "initial_integrated_process_interrupted_after_checkpoint": True,
    "interruption_snapshot": "artifacts/idm/g005_idm_exactset_history_prediction_recovery_snapshot.json",
    "recovery_prediction_workers": int("$PREDICTION_WORKERS"),
    "recovery_prediction_cuda_devices": [item for item in "$PREDICTION_CUDA_DEVICES".split(",") if item],
    "recovery_pseudolabel_validation": "skipped_for_full_corpus_throughput_after_generator_tests"
    if "$SKIP_PSEUDOLABEL_VALIDATION" != "0"
    else "enabled",
    "started_at": "$STARTED_AT",
    "wall_clock_seconds": int("$END_EPOCH") - int("$START_EPOCH"),
    "git_head": _git(["rev-parse", "HEAD"]),
    "git_status_short": _git(["status", "--short"]),
    "gpu_monitor_sha256": _sha256("$GPU_MONITOR_LOG"),
    "gpu_monitor_status": _gpu_monitor_status("$GPU_MONITOR_LOG", 4),
    "summary_path": summary_path,
    "summary_exists": summary is not None,
    "metadata_path": "$OUTPUT_DIR/checkpoint_metadata.json",
    "metadata_exists": metadata is not None,
    "metrics_path": "$OUTPUT_DIR/metrics.json",
    "metrics_exists": Path("$OUTPUT_DIR/metrics.json").exists(),
    "split_stats_config": "$SPLIT_STATS_CONFIG",
    "split_stats_summary_path": "artifacts/eval/g005_idm_exactset_history_split_statistical_comparisons_summary.json",
    "split_stats_status": split_stats.get("status") if split_stats else None,
    "split_stats_outputs": split_stats.get("outputs", []) if split_stats else [],
    "paper_target_config": "$PAPER_TARGET_CONFIG",
    "paper_metrics_path": "artifacts/idm/g005_idm_exactset_history_paper_metrics.json",
    "paper_metrics_status": paper_metrics.get("status") if paper_metrics else None,
    "paper_metrics_rows": (paper_metrics or {}).get("alignment", {}).get("rows_seen") if paper_metrics else None,
    "require_precomputed_cache": True,
    "precompute_cache_validation_path": "artifacts/idm/g005_idm_exactset_history_precomputed_cache_validation.json",
    "precompute_cache_validation_status": precompute.get("status") if precompute else None,
    "precompute_cache_validation_rows": precompute.get("rows") if precompute else None,
    "checkpoint_path": (metadata or {}).get("checkpoint_path") or "$OUTPUT_DIR/checkpoint.pt",
    "checkpoint_sha256": _sha256("$OUTPUT_DIR/checkpoint.pt"),
    "train_records": (metadata or {}).get("train_records"),
    "target_records": (metadata or {}).get("target_records"),
    "prediction_resume": (summary or {}).get("prediction_resume") if summary else None,
    "validation_exit_code": status,
    "g005_audit_path": "artifacts/idm/g005_idm_exactset_history_paper_target_audit.json",
    "g005_audit_status": audit.get("status") if audit else None,
    "g005_audit_error_count": audit.get("error_count") if audit else None,
}
Path("$RUN_SUMMARY").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n")
Path("$RECOVERY_RUN").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n")
print(json.dumps({"exit_code": status, "target_records": payload["target_records"], "paper_metrics_status": payload["paper_metrics_status"], "audit_status": payload["g005_audit_status"]}, sort_keys=True))
PY

exit "$RECOVERY_STATUS"
