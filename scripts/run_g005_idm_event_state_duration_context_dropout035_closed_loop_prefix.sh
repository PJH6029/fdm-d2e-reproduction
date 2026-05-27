#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
CONFIG="${CONFIG:-configs/model/idm_streaming_d2e_full_event_state_duration_context_dropout035_closed_loop_prefix320k.yaml}"
DROPOUT_ROOT="${DROPOUT_ROOT:-outputs/data/d2e_event_state_duration_context_dropout035_shards_accel64}"
DROPOUT_SUMMARY="${DROPOUT_SUMMARY:-artifacts/idm/g005_idm_event_state_duration_context_dropout035_materialization_summary.json}"
DROPOUT_PROGRESS="${DROPOUT_PROGRESS:-artifacts/idm/g005_idm_event_state_duration_context_dropout035_materialization_progress.json}"
PAPER_CONFIG="${PAPER_CONFIG:-configs/eval/g005_idm_event_state_duration_context_dropout035_closed_loop_prefix320k_paper_metrics.yaml}"
RUN_SUMMARY="${RUN_SUMMARY:-artifacts/idm/g005_idm_event_state_duration_context_dropout035_closed_loop_prefix320k_run_summary.json}"
MAX_ROWS_PER_FILE="${MAX_ROWS_PER_FILE:-20000}"
NPROC="${NPROC:-2}"

mkdir -p artifacts/idm outputs/cluster
STARTED_AT="$(date -Iseconds)"
uv run python scripts/materialize_state_context_dropout_train.py \
  --input 'outputs/data/d2e_event_state_duration_context_shards_accel64/shard_*/train_core.jsonl' \
  --output-root "$DROPOUT_ROOT" \
  --dropout-rate 0.35 \
  --seed 20260528 \
  --summary-out "$DROPOUT_SUMMARY" \
  --progress-out "$DROPOUT_PROGRESS" \
  --max-rows-per-file "$MAX_ROWS_PER_FILE"
uv run torchrun --standalone --nproc-per-node="$NPROC" scripts/train_idm_streaming.py --config "$CONFIG" --require-torch
uv run python scripts/build_g005_idm_paper_metrics.py --config "$PAPER_CONFIG"
python3 - <<PY
import json, pathlib, time
summary={
  'schema':'g005_dropout_closed_loop_prefix_run_summary.v1',
  'status':'pass',
  'started_at':'$STARTED_AT',
  'finished_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
  'config':'$CONFIG',
  'dropout_summary':'$DROPOUT_SUMMARY',
  'paper_metrics':json.load(open('artifacts/idm/g005_idm_event_state_duration_context_dropout035_closed_loop_prefix320k_paper_metrics.json')),
  'claim_boundary':'Prefix diagnostic only; not G005 completion evidence.'
}
pathlib.Path('$RUN_SUMMARY').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
PY
