#!/usr/bin/env bash
set -euo pipefail

OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-full_compact_parallel}"
NUM_SHARDS="${NUM_SHARDS:-16}"
SHARD_ROOT="${SHARD_ROOT:-outputs/data/d2e_full_corpus_shards}"
LOG_DIR="${LOG_DIR:-artifacts/sources}"
DATA_OUTPUT_DIR="${DATA_OUTPUT_DIR:-outputs/data/d2e_full_corpus}"
DECODE_SUMMARY="${DECODE_SUMMARY:-artifacts/sources/d2e_full_corpus_decode_summary.json}"
IDM_CONFIG="${IDM_CONFIG:-configs/model/idm_streaming_d2e_full_compact.yaml}"
CACHE_DIR="${CACHE_DIR:-/root/work/data/d2e/cache}"
IDM_NPROC_PER_NODE="${IDM_NPROC_PER_NODE:-4}"
BUILD_SPLIT_STATS="${BUILD_SPLIT_STATS:-1}"
SPLIT_STATS_CONFIG="${SPLIT_STATS_CONFIG:-configs/eval/g003_split_statistics.yaml}"
SPLIT_STATS_SUMMARY="${SPLIT_STATS_SUMMARY:-artifacts/eval/g003_split_statistical_comparisons_summary.json}"
IDM_SUMMARY="${IDM_SUMMARY:-artifacts/idm/idm_streaming_d2e_full_compact_summary.json}"
export BUILD_SPLIT_STATS SPLIT_STATS_CONFIG SPLIT_STATS_SUMMARY IDM_SUMMARY
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

mkdir -p "${LOG_DIR}" artifacts/sources artifacts/idm artifacts/mlxp outputs/cluster "${SHARD_ROOT}" "${DATA_OUTPUT_DIR}"

uv run python scripts/cluster_gpu_smoke.py \
  --expected-gpus "${EXPECTED_GPUS:-4}" \
  --report "outputs/cluster/g003_${OUTPUT_SUFFIX}_gpu_smoke.json"

echo "Launching ${NUM_SHARDS} full-corpus extraction shards"
pids=()
for shard_index in $(seq 0 $((NUM_SHARDS - 1))); do
  shard_dir="${SHARD_ROOT}/shard_${shard_index}"
  mkdir -p "${shard_dir}" "${LOG_DIR}"
  (
    uv run python scripts/extract_d2e_full_corpus.py \
      --config configs/data/d2e_full_corpus.yaml \
      --data-universe artifacts/sources/d2e_full_data_universe_manifest.json \
      --split-contract artifacts/sources/d2e_full_split_contract.json \
      --output-dir "${shard_dir}" \
      --summary-out "${shard_dir}/decode_summary.json" \
      --cache-dir "${CACHE_DIR}" \
      --shard-index "${shard_index}" \
      --num-shards "${NUM_SHARDS}" \
      --bin-ms "${BIN_MS:-50}" \
      --frame-fps "${FRAME_FPS:-20}" \
      --image-size "${IMAGE_SIZE:-64}" \
      --video-mode "${VIDEO_MODE:-download}"
  ) > "${LOG_DIR}/d2e_full_corpus_shard_${shard_index}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" != "0" ]]; then
  echo "One or more extraction shards failed" >&2
  exit 2
fi

uv run python scripts/merge_d2e_full_corpus_shards.py \
  --shard-root "${SHARD_ROOT}" \
  --output-dir "${DATA_OUTPUT_DIR}" \
  --summary-out "${DECODE_SUMMARY}"

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

decode_summary = Path(os.environ.get("DECODE_SUMMARY", "artifacts/sources/d2e_full_corpus_decode_summary.json"))
idm_summary_path = Path(os.environ.get("IDM_SUMMARY", "artifacts/idm/idm_streaming_d2e_full_compact_summary.json"))
split_stats_summary_path = Path(os.environ.get("SPLIT_STATS_SUMMARY", "artifacts/eval/g003_split_statistical_comparisons_summary.json"))
suffix = os.environ.get("OUTPUT_SUFFIX", "full_compact_parallel")
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
    "host": platform.node(),
    "gpu": smi,
    "decode_summary": json.loads(decode_summary.read_text()) if decode_summary.exists() else None,
    "idm_summary": json.loads(idm_summary_path.read_text()) if idm_summary_path.exists() else None,
    "split_stats_summary": json.loads(split_stats_summary_path.read_text()) if split_stats_summary_path.exists() else None,
    "artifacts": {
        "gpu_smoke": f"outputs/cluster/g003_{suffix}_gpu_smoke.json",
        "decode_summary": str(decode_summary),
        "idm_summary": str(idm_summary_path),
        "split_stats_summary": str(split_stats_summary_path),
    },
    "idm_nproc_per_node": int(os.environ.get("IDM_NPROC_PER_NODE", "4")),
    "build_split_stats": os.environ.get("BUILD_SPLIT_STATS", "1") != "0",
    "split_stats_config": os.environ.get("SPLIT_STATS_CONFIG", "configs/eval/g003_split_statistics.yaml"),
    "split_stats_summary_path": str(split_stats_summary_path),
    "split_stats_summary_exists": split_stats_summary_path.exists(),
}
out = Path(f"artifacts/idm/g003_d2e_full_idm_run_{suffix}.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({"wrote": str(out), "suffix": suffix}, sort_keys=True))
PY
