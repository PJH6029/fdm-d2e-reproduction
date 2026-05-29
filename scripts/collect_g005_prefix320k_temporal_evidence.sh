#!/usr/bin/env bash
set -euo pipefail

# Collect terminal/snapshot evidence for the G005 actual-luma prefix320k temporal IDM run.
# Does not commit artifacts and does not copy secrets. By default it skips the checkpoint.

REPO_LOCAL="${REPO_LOCAL:-$(pwd)}"
SUMMARY="${SUMMARY:-$REPO_LOCAL/artifacts/cluster/g005_4xh200_auto_launch_20260529.json}"
POD="${POD:-}"
NAMESPACE="${NAMESPACE:-p-production}"
KUBECONFIG_PATH="${KUBECONFIG_PATH:-/home/top321902/.kube/mlxp/jeonghunpark/debug-kubeconfig.yaml}"
REMOTE_BASE="${REMOTE_BASE:-/root/work/code/continuous-gui-poc}"
REMOTE_REPO_PRIMARY="${REMOTE_REPO_PRIMARY:-fdm-d2e-reproduction-g005-lumalong-13023f0}"
REMOTE_REPO_FALLBACK="${REMOTE_REPO_FALLBACK:-fdm-d2e-reproduction}"
MODEL_SLUG="${MODEL_SLUG:-g005_idm_temporal_masked_diffusion_luma2_actual_prefix320k_epoch3}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/idm_temporal_masked_diffusion_d2e_luma2_actual_prefix320k_epoch3}"
COPY_CHECKPOINT="${COPY_CHECKPOINT:-0}"

cd "$REPO_LOCAL"
if [[ -z "$POD" && -s "$SUMMARY" ]]; then
  POD=$(python3 - <<'PY' "$SUMMARY"
import json, sys
try:
    print(json.load(open(sys.argv[1])).get("pod_name") or "")
except Exception:
    print("")
PY
)
fi
if [[ -z "$POD" ]]; then
  echo "pod name is required via POD=... or launch summary $SUMMARY" >&2
  exit 2
fi

REMOTE_REPO="$REMOTE_BASE/$REMOTE_REPO_PRIMARY"
if ! KUBECONFIG="$KUBECONFIG_PATH" kubectl --request-timeout=30s -n "$NAMESPACE" exec -i "$POD" -- test -d "$REMOTE_REPO/.git"; then
  REMOTE_REPO="$REMOTE_BASE/$REMOTE_REPO_FALLBACK"
fi

mkdir -p artifacts/idm "$OUTPUT_DIR/rank_progress"
KUBECONFIG="$KUBECONFIG_PATH" kubectl --request-timeout=180s -n "$NAMESPACE" exec -i "$POD" -- bash -s "$REMOTE_REPO" "$MODEL_SLUG" "$OUTPUT_DIR" "$COPY_CHECKPOINT" <<'REMOTE' | tar -x -C .
set -euo pipefail
repo="$1"
model_slug="$2"
output_dir="$3"
copy_checkpoint="$4"
cd "$repo"
files=(
  "artifacts/idm/${model_slug}_h200_run.json"
  "artifacts/idm/${model_slug}_h200_compact_summary.json"
  "artifacts/idm/${model_slug}_h200_gpu_monitor.csv"
  "artifacts/idm/${model_slug}_h200_wandb_status.json"
  "artifacts/idm/${model_slug}_h200.log"
  "${output_dir}/paper_metrics.json"
  "${output_dir}/summary.json"
  "${output_dir}/train_history.json"
  "${output_dir}/rank_progress/train_rank0.json"
  "${output_dir}/rank_progress/train_rank1.json"
  "${output_dir}/rank_progress/train_rank2.json"
  "${output_dir}/rank_progress/train_rank3.json"
  "${output_dir}/resolved_config.json"
  "${output_dir}/checkpoint_metadata.json"
)
if [[ "$copy_checkpoint" == "1" ]]; then
  files+=("${output_dir}/checkpoint.pt")
fi
existing=()
for f in "${files[@]}"; do
  [[ -f "$f" ]] && existing+=("$f")
done
if [[ ${#existing[@]} -eq 0 ]]; then
  tar -c --files-from /dev/null
else
  tar -c "${existing[@]}"
fi
REMOTE

python3 - <<'PY' "$MODEL_SLUG"
import csv
import json
import pathlib
import sys
model_slug = sys.argv[1]
run_path = pathlib.Path(f"artifacts/idm/{model_slug}_h200_run.json")
compact_path = pathlib.Path(f"artifacts/idm/{model_slug}_h200_compact_summary.json")
gpu_path = pathlib.Path(f"artifacts/idm/{model_slug}_h200_gpu_monitor.csv")
vals = []
indices = set()
rows = 0
if gpu_path.exists():
    with gpu_path.open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) < 4:
                continue
            rows += 1
            try:
                indices.add(int(str(row[1]).strip()))
            except Exception:
                pass
            try:
                vals.append(float(str(row[3]).replace("%", "").strip()))
            except Exception:
                pass
status = {
    "status": "pass" if rows else "missing",
    "rows": rows,
    "unique_gpu_indices": sorted(indices),
    "max_gpu_utilization": max(vals) if vals else None,
    "avg_gpu_utilization": sum(vals) / len(vals) if vals else None,
}
for path in (run_path, compact_path):
    if not path.exists():
        continue
    payload = json.loads(path.read_text())
    payload["gpu_monitor_status"] = status
    payload.setdefault("claim_boundary", "G005 prefix320k temporal IDM evidence snapshot; not completion unless paper-target audit passes.")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps({"copied_run_summary": run_path.exists(), "copied_compact_summary": compact_path.exists(), "gpu_monitor_status": status}, indent=2))
PY

for json_path in \
  "artifacts/idm/${MODEL_SLUG}_h200_run.json" \
  "artifacts/idm/${MODEL_SLUG}_h200_compact_summary.json" \
  "${OUTPUT_DIR}/paper_metrics.json" \
  "${OUTPUT_DIR}/summary.json" \
  "${OUTPUT_DIR}/train_history.json"; do
  [[ -s "$json_path" ]] && python3 -m json.tool "$json_path" >/dev/null
done
