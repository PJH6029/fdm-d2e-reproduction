#!/usr/bin/env bash
set -euo pipefail

# Resume the isolated accel64 G003 lane after full shard extraction and merge
# have already succeeded. This intentionally skips extraction and only reruns
# stats precompute + distributed IDM training + split statistics + run evidence.
export OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-accel64}"
export DATA_OUTPUT_DIR="${DATA_OUTPUT_DIR:-outputs/data/d2e_full_corpus_accel64}"
export DECODE_SUMMARY="${DECODE_SUMMARY:-artifacts/sources/d2e_full_corpus_decode_summary_accel64.json}"
export IDM_CONFIG="${IDM_CONFIG:-configs/model/idm_streaming_d2e_full_compact_accel64.yaml}"
export IDM_SUMMARY="${IDM_SUMMARY:-artifacts/idm/idm_streaming_d2e_full_compact_accel64_summary.json}"
export IDM_NPROC_PER_NODE="${IDM_NPROC_PER_NODE:-4}"
export BUILD_SPLIT_STATS="${BUILD_SPLIT_STATS:-1}"
export SPLIT_STATS_CONFIG="${SPLIT_STATS_CONFIG:-configs/eval/g003_split_statistics_accel64.yaml}"
export SPLIT_STATS_SUMMARY="${SPLIT_STATS_SUMMARY:-artifacts/eval/g003_split_statistical_comparisons_accel64_summary.json}"
export PRECOMPUTE_IDM_STATS="${PRECOMPUTE_IDM_STATS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

mkdir -p artifacts/idm artifacts/eval artifacts/sources outputs/cluster

for required in \
  "${DECODE_SUMMARY}" \
  "${DATA_OUTPUT_DIR}/train_core.jsonl" \
  "${DATA_OUTPUT_DIR}/target_all_eval.jsonl"
do
  if [[ ! -s "${required}" ]]; then
    echo "missing required merged G003 accel64 artifact: ${required}" >&2
    exit 2
  fi
done

uv run python scripts/cluster_gpu_smoke.py \
  --expected-gpus "${EXPECTED_GPUS:-4}" \
  --report "outputs/cluster/g003_${OUTPUT_SUFFIX}_resume_gpu_smoke.json"

if [[ "${PRECOMPUTE_IDM_STATS}" != "0" ]]; then
  uv run python scripts/precompute_streaming_idm_stats.py \
    --config "${IDM_CONFIG}"
fi

uv run torchrun --standalone --nproc-per-node="${IDM_NPROC_PER_NODE}" scripts/train_idm_streaming.py \
  --config "${IDM_CONFIG}" \
  --require-torch

if [[ "${BUILD_SPLIT_STATS}" != "0" ]]; then
  uv run python scripts/build_split_statistical_comparisons.py \
    --config "${SPLIT_STATS_CONFIG}"
fi

uv run python - <<'PY'
from __future__ import annotations
import json, os, platform, subprocess
from pathlib import Path

decode_summary = Path(os.environ.get("DECODE_SUMMARY", "artifacts/sources/d2e_full_corpus_decode_summary_accel64.json"))
idm_summary_path = Path(os.environ.get("IDM_SUMMARY", "artifacts/idm/idm_streaming_d2e_full_compact_accel64_summary.json"))
split_stats_summary_path = Path(os.environ.get("SPLIT_STATS_SUMMARY", "artifacts/eval/g003_split_statistical_comparisons_accel64_summary.json"))
suffix = os.environ.get("OUTPUT_SUFFIX", "accel64")
try:
    smi = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip().splitlines()
except Exception as exc:
    smi = [f"nvidia-smi unavailable: {exc!r}"]
evidence = {
    "schema": "g003_d2e_full_idm_run_evidence.v1",
    "suffix": suffix,
    "resume_mode": "training_only_after_successful_accel64_merge",
    "host": platform.node(),
    "gpu": smi,
    "decode_summary": json.loads(decode_summary.read_text()) if decode_summary.exists() else None,
    "idm_summary": json.loads(idm_summary_path.read_text()) if idm_summary_path.exists() else None,
    "split_stats_summary": json.loads(split_stats_summary_path.read_text()) if split_stats_summary_path.exists() else None,
    "artifacts": {
        "gpu_smoke": f"outputs/cluster/g003_{suffix}_resume_gpu_smoke.json",
        "decode_summary": str(decode_summary),
        "idm_summary": str(idm_summary_path),
        "split_stats_summary": str(split_stats_summary_path),
    },
    "idm_nproc_per_node": int(os.environ.get("IDM_NPROC_PER_NODE", "4")),
    "build_split_stats": os.environ.get("BUILD_SPLIT_STATS", "1") != "0",
    "split_stats_config": os.environ.get("SPLIT_STATS_CONFIG", "configs/eval/g003_split_statistics_accel64.yaml"),
    "split_stats_summary_path": str(split_stats_summary_path),
    "split_stats_summary_exists": split_stats_summary_path.exists(),
}
out = Path(f"artifacts/idm/g003_d2e_full_idm_run_{suffix}.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({"wrote": str(out), "suffix": suffix, "resume_mode": evidence["resume_mode"]}, sort_keys=True))
PY
