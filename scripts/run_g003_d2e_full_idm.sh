#!/usr/bin/env bash
set -euo pipefail

OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-full_compact}"
DATA_OUTPUT_DIR="${DATA_OUTPUT_DIR:-outputs/data/d2e_full_corpus}"
DECODE_SUMMARY="${DECODE_SUMMARY:-artifacts/sources/d2e_full_corpus_decode_summary.json}"
IDM_CONFIG="${IDM_CONFIG:-configs/model/idm_streaming_d2e_full_compact.yaml}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
CACHE_DIR="${CACHE_DIR:-/root/work/data/d2e/cache}"

mkdir -p artifacts/sources artifacts/idm artifacts/mlxp outputs/cluster "${DATA_OUTPUT_DIR}"

uv run python scripts/cluster_gpu_smoke.py \
  --expected-gpus "${EXPECTED_GPUS:-1}" \
  --report "outputs/cluster/g003_${OUTPUT_SUFFIX}_gpu_smoke.json"

uv run python scripts/extract_d2e_full_corpus.py \
  --config configs/data/d2e_full_corpus.yaml \
  --data-universe artifacts/sources/d2e_full_data_universe_manifest.json \
  --split-contract artifacts/sources/d2e_full_split_contract.json \
  --output-dir "${DATA_OUTPUT_DIR}" \
  --summary-out "${DECODE_SUMMARY}" \
  --cache-dir "${CACHE_DIR}" \
  --shard-index "${SHARD_INDEX}" \
  --num-shards "${NUM_SHARDS}" \
  --bin-ms "${BIN_MS:-50}" \
  --frame-fps "${FRAME_FPS:-20}" \
  --image-size "${IMAGE_SIZE:-64}" \
  --video-mode "${VIDEO_MODE:-download}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" uv run python scripts/train_idm_streaming.py \
  --config "${IDM_CONFIG}" \
  --require-torch

uv run python - <<'PY'
from __future__ import annotations
import json, os, platform, subprocess
from pathlib import Path

decode_summary = Path(os.environ.get("DECODE_SUMMARY", "artifacts/sources/d2e_full_corpus_decode_summary.json"))
idm_summary_path = Path(os.environ.get("IDM_SUMMARY", "artifacts/idm/idm_streaming_d2e_full_compact_summary.json"))
suffix = os.environ.get("OUTPUT_SUFFIX", "full_compact")
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
    "artifacts": {
        "gpu_smoke": f"outputs/cluster/g003_{suffix}_gpu_smoke.json",
        "decode_summary": str(decode_summary),
        "idm_summary": str(idm_summary_path),
    },
}
out = Path(f"artifacts/idm/g003_d2e_full_idm_run_{suffix}.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({"wrote": str(out), "suffix": suffix}, sort_keys=True))
PY
