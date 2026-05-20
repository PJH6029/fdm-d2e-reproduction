#!/usr/bin/env bash
set -euo pipefail

OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-h200}"
CONFIG_DATA="${CONFIG_DATA:-configs/data/d2e_real_multi_apex8.yaml}"
CONFIG_TORCH="${CONFIG_TORCH:-configs/model/idm_torch_apex8.yaml}"

mkdir -p artifacts/sources artifacts/eval artifacts/idm artifacts/mlxp outputs/cluster

uv run python scripts/cluster_gpu_smoke.py \
  --expected-gpus 1 \
  --report "outputs/cluster/${OUTPUT_SUFFIX}_gpu_smoke_1.json"

uv run python scripts/extract_d2e_real_multi.py \
  --config "${CONFIG_DATA}" \
  --summary-copy "artifacts/sources/d2e_multi_decode_apex8_${OUTPUT_SUFFIX}_summary.json"

uv run python scripts/run_baselines_eval.py \
  --train-records outputs/data/real_multi_apex8/train.jsonl \
  --ground-truth outputs/data/real_multi_apex8/heldout.jsonl \
  --endpoints configs/eval/primary_endpoints.yaml \
  --output-dir "outputs/eval/baselines_multi_apex8_${OUTPUT_SUFFIX}" \
  --summary-out "artifacts/eval/baseline_stat_eval_multi_apex8_${OUTPUT_SUFFIX}.json"

uv run python scripts/train_idm_torch.py \
  --config "${CONFIG_TORCH}" \
  --require-torch

uv run python - <<'PY'
from __future__ import annotations
import json, os, platform, subprocess
from pathlib import Path
suffix=os.environ.get('OUTPUT_SUFFIX','h200')
summary=json.loads(Path('artifacts/idm/idm_torch_apex8_summary.json').read_text())
try:
    smi=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'], text=True).strip().splitlines()
except Exception as exc:
    smi=[f'nvidia-smi unavailable: {exc!r}']
evidence={
  'schema':'g4_h200_idm_run_evidence.v1',
  'suffix': suffix,
  'host': platform.node(),
  'gpu': smi,
  'torch_device': summary.get('device'),
  'torch_metrics': summary.get('metrics'),
  'torch_metadata': summary.get('metadata'),
  'statistical_comparison': summary.get('statistical_comparison'),
  'artifacts': {
    'gpu_smoke': f'outputs/cluster/{suffix}_gpu_smoke_1.json',
    'decode_summary': f'artifacts/sources/d2e_multi_decode_apex8_{suffix}_summary.json',
    'baseline_summary': f'artifacts/eval/baseline_stat_eval_multi_apex8_{suffix}.json',
    'torch_summary': 'artifacts/idm/idm_torch_apex8_summary.json'
  }
}
Path(f'artifacts/idm/g4_h200_idm_run_{suffix}.json').write_text(json.dumps(evidence, indent=2, ensure_ascii=False)+'\n')
print(json.dumps({'wrote': f'artifacts/idm/g4_h200_idm_run_{suffix}.json', 'device': summary.get('device'), 'metrics': summary.get('metrics',{}).get('mouse_move')}, sort_keys=True))
PY
