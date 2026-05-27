#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
INPUT_GLOB="${INPUT_GLOB:-outputs/data/d2e_event_state_duration_context_shards_accel64/shard_*/target_all_eval.jsonl}"
CHRONO_OUTPUT="${CHRONO_OUTPUT:-outputs/data/d2e_event_state_duration_context_chrono_prefix320k/target_all_eval.jsonl}"
CHRONO_SUMMARY="${CHRONO_SUMMARY:-artifacts/idm/g005_idm_event_state_duration_context_chrono_prefix320k_materialization_summary.json}"
PREDICT_CONFIG="${PREDICT_CONFIG:-configs/model/idm_streaming_d2e_full_event_state_duration_context_chrono_closed_loop_prefix320k_predict.yaml}"
PAPER_CONFIG="${PAPER_CONFIG:-configs/eval/g005_idm_event_state_duration_context_chrono_closed_loop_prefix320k_paper_metrics.yaml}"
RUN_SUMMARY="${RUN_SUMMARY:-artifacts/idm/g005_idm_event_state_duration_context_chrono_closed_loop_prefix320k_run_summary.json}"
WANDB_STATUS="${WANDB_STATUS:-artifacts/idm/g005_idm_event_state_duration_context_chrono_closed_loop_prefix320k_wandb_status.json}"
MAX_ROWS="${MAX_ROWS:-320000}"
mkdir -p artifacts/idm outputs/cluster "$(dirname "$CHRONO_OUTPUT")"
STARTED_AT="$(date -Iseconds)"
uv run python scripts/materialize_chronological_prefix.py \
  --input "$INPUT_GLOB" \
  --output "$CHRONO_OUTPUT" \
  --summary-out "$CHRONO_SUMMARY" \
  --max-rows "$MAX_ROWS" \
  --source-label "g005_event_state_duration_context_prefix320k"
uv run python scripts/predict_idm_streaming.py --config "$PREDICT_CONFIG" --prediction-workers 1 --prediction-cuda-devices 0
uv run python scripts/build_g005_idm_paper_metrics.py --config "$PAPER_CONFIG"
uv run --with wandb python scripts/log_wandb_artifacts.py \
  --env-file .env \
  --run-name "g005-context-chrono-closed-loop-prefix320k" \
  --group "g005-idm-paper-target" \
  --job-type "prefix-diagnostic" \
  --tags "g005,idm,d2e,chronological,closed-loop,prefix" \
  --artifact-name "g005-context-chrono-closed-loop-prefix320k" \
  --output "$WANDB_STATUS" \
  "$CHRONO_SUMMARY" \
  "artifacts/idm/g005_idm_event_state_duration_context_chrono_closed_loop_prefix320k_prediction_summary.json" \
  "artifacts/idm/g005_idm_event_state_duration_context_chrono_closed_loop_prefix320k_paper_metrics.json" || true
python3 - <<PY
import json, pathlib, time
summary={
  'schema':'g005_chrono_closed_loop_prefix_run_summary.v1',
  'status':'pass',
  'started_at':'$STARTED_AT',
  'finished_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
  'chronological_summary':'$CHRONO_SUMMARY',
  'prediction_summary':'artifacts/idm/g005_idm_event_state_duration_context_chrono_closed_loop_prefix320k_prediction_summary.json',
  'paper_metrics':json.load(open('artifacts/idm/g005_idm_event_state_duration_context_chrono_closed_loop_prefix320k_paper_metrics.json')),
  'wandb_status':'$WANDB_STATUS',
  'claim_boundary':'Prefix diagnostic only; not G005 completion evidence.'
}
pathlib.Path('$RUN_SUMMARY').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
PY
