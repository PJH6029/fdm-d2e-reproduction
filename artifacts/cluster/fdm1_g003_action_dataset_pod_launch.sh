#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
if [[ -z "${KUBERNETES_SERVICE_HOST:-}" ]]; then echo 'refusing: this launch script must run inside the MLXP Kubernetes pod' >&2; exit 2; fi
if [[ ! -d /root/work/code/continuous-gui-poc/fdm-d2e-reproduction ]]; then echo 'refusing: repo dir not found: /root/work/code/continuous-gui-poc/fdm-d2e-reproduction' >&2; exit 2; fi
cd /root/work/code/continuous-gui-poc/fdm-d2e-reproduction
git fetch origin research/fdm1-d2e-ultragoal
git checkout research/fdm1-d2e-ultragoal
git pull --ff-only
uv sync --extra d2e --extra train --extra test
uv run python scripts/preflight_g003_fdm1_action_dataset_pod.py --require-pod --expected-branch research/fdm1-d2e-ultragoal --min-free-gb 100
mkdir -p artifacts/logs outputs/cluster artifacts/cluster artifacts/sources artifacts/reports
git rev-parse HEAD > artifacts/sources/fdm1_g003_pod_launch_commit.txt
if [[ -s outputs/cluster/fdm1_g003_action_dataset_pipeline.pid ]] && kill -0 "$(cat outputs/cluster/fdm1_g003_action_dataset_pipeline.pid)" 2>/dev/null; then echo 'refusing: existing G003 pipeline pid is still active:' $(cat outputs/cluster/fdm1_g003_action_dataset_pipeline.pid) >&2; exit 3; fi
nohup bash scripts/run_g003_fdm1_action_dataset_sharded_pipeline.sh > artifacts/logs/fdm1_g003_action_dataset_pipeline.log 2>&1 & echo $! > outputs/cluster/fdm1_g003_action_dataset_pipeline.pid
echo launched $(cat outputs/cluster/fdm1_g003_action_dataset_pipeline.pid) log=artifacts/logs/fdm1_g003_action_dataset_pipeline.log
uv run python - <<'PY'
import json, os, subprocess, time
path = 'artifacts/cluster/fdm1_g003_action_dataset_pod_launch_context.json'
data = {
  'schema': 'fdm1_g003_pod_launch_context.v1',
  'created_at_unix': time.time(),
  'hostname': os.uname().nodename,
  'branch': 'research/fdm1-d2e-ultragoal',
  'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
  'pid_path': 'outputs/cluster/fdm1_g003_action_dataset_pipeline.pid',
  'log_path': 'artifacts/logs/fdm1_g003_action_dataset_pipeline.log',
  'pid': open('outputs/cluster/fdm1_g003_action_dataset_pipeline.pid').read().strip() if os.path.exists('outputs/cluster/fdm1_g003_action_dataset_pipeline.pid') else None,
  'claim_boundary': 'Launch context only; G003 completion requires completion audit pass and OMX checkpoint.',
}
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, 'w', encoding='utf-8') as f: json.dump(data, f, ensure_ascii=False, indent=2)
print(json.dumps(data, ensure_ascii=False, indent=2))
PY
