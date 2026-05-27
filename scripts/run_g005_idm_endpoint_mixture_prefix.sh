#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
ENSEMBLE_CONFIG="${ENSEMBLE_CONFIG:-configs/eval/g005_idm_endpoint_mixture_state_luma_gate_context_prefix320k.yaml}"
PAPER_CONFIG="${PAPER_CONFIG:-configs/eval/g005_idm_endpoint_mixture_state_luma_gate_context_prefix320k_paper_metrics.yaml}"
SUMMARY="${SUMMARY:-artifacts/idm/g005_idm_endpoint_mixture_state_luma_gate_context_prefix320k_run_summary.json}"
WANDB_STATUS="${WANDB_STATUS:-artifacts/idm/g005_idm_endpoint_mixture_state_luma_gate_context_prefix320k_wandb_status.json}"
STARTED_AT="$(date -Iseconds)"
uv run python scripts/ensemble_idm_predictions.py --config "$ENSEMBLE_CONFIG"
uv run python scripts/build_g005_idm_paper_metrics.py --config "$PAPER_CONFIG"
uv run --with wandb python scripts/log_wandb_artifacts.py \
  --env-file .env \
  --run-name "g005-endpoint-mixture-state-luma-gate-context-prefix320k" \
  --group "g005-idm-paper-target" \
  --job-type "prefix-diagnostic" \
  --tags "g005,idm,d2e,endpoint-mixture,prefix" \
  --artifact-name "g005-endpoint-mixture-state-luma-gate-context-prefix320k" \
  --output "$WANDB_STATUS" \
  --json "artifacts/idm/g005_idm_endpoint_mixture_state_luma_gate_context_prefix320k_summary.json" \
  --json "artifacts/idm/g005_idm_endpoint_mixture_state_luma_gate_context_prefix320k_paper_metrics.json" || true
python3 - <<PY
import json, pathlib, time
summary={
  'schema':'g005_endpoint_mixture_prefix_run_summary.v1',
  'status':'pass',
  'started_at':'$STARTED_AT',
  'finished_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
  'ensemble_summary':'artifacts/idm/g005_idm_endpoint_mixture_state_luma_gate_context_prefix320k_summary.json',
  'paper_metrics':json.load(open('artifacts/idm/g005_idm_endpoint_mixture_state_luma_gate_context_prefix320k_paper_metrics.json')),
  'wandb_status':'$WANDB_STATUS',
  'claim_boundary':'Prefix post-hoc endpoint mixture diagnostic only; not G005 completion evidence.'
}
pathlib.Path('$SUMMARY').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
PY
