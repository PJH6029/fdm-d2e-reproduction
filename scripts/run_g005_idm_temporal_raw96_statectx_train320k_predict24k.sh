#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi

MODEL_SLUG="${MODEL_SLUG:-g005_idm_temporal_masked_diffusion_raw96_patch_axisclass_realvideo_statectx_train320k_stratcal_noadapt_predict24k}"
CHECKPOINT="${CHECKPOINT:-outputs/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_statectx_train320k_target24k/checkpoint.pt}"
CONFIG="${CONFIG:-configs/model/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_statectx_train320k_stratcal_noadapt_predict24k.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${MODEL_SLUG}}"
SUMMARY_OUT="${SUMMARY_OUT:-artifacts/idm/${MODEL_SLUG}_summary.json}"
COMPACT_SUMMARY="${COMPACT_SUMMARY:-artifacts/idm/${MODEL_SLUG}_compact_summary.json}"
WANDB_STATUS="${WANDB_STATUS:-artifacts/idm/${MODEL_SLUG}_wandb_status.json}"
WANDB_TAGS="${WANDB_TAGS:-g005,idm,d2e,fdm1-recipe,real-video,raw96,state-context,stratified-calibration,prediction-only}"

mkdir -p "$OUTPUT_DIR" artifacts/idm outputs/cluster
if [[ ! -s "$CHECKPOINT" ]]; then
  echo "missing checkpoint: $CHECKPOINT" >&2
  exit 2
fi

STARTED_AT="$(date -Iseconds)"
set +e
uv run --extra train python scripts/predict_idm_temporal_masked_diffusion.py \
  --checkpoint "$CHECKPOINT" \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --summary-out "$SUMMARY_OUT"
EXIT_CODE="$?"
set -e
FINISHED_AT="$(date -Iseconds)"

python3 - <<PY
import hashlib, json, pathlib
summary_path=pathlib.Path("$SUMMARY_OUT")
metrics_path=pathlib.Path("$OUTPUT_DIR/paper_metrics.json")
compact_path=pathlib.Path("$COMPACT_SUMMARY")

def load(path):
    return json.loads(path.read_text()) if path.exists() else {}

def sha(path):
    if not path.exists():
        return None
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
summary=load(summary_path)
metrics=load(metrics_path)
all_metrics=(metrics.get('groups') or {}).get('all') or metrics
paper=all_metrics.get('paper_compatible', {})
strict=all_metrics.get('strict_local', {})
keyboard=(paper.get('keyboard') or {}).get('key_accuracy')
button=(paper.get('mouse_button') or {}).get('button_accuracy')
move=paper.get('mouse_move') or {}
strict_button=strict.get('mouse_button') or {}
observed={
  'keyboard_key_accuracy': keyboard,
  'mouse_button_accuracy': button,
  'mouse_button_f1': strict_button.get('f1'),
  'mouse_move_pearson_x': move.get('pearson_x'),
  'mouse_move_pearson_y': move.get('pearson_y'),
  'no_button_false_positive_rate': strict_button.get('no_button_false_positive_rate'),
}
targets={
  'keyboard_key_accuracy': 0.73,
  'mouse_button_accuracy': 0.957,
  'mouse_button_f1_min_nonzero': 1e-12,
  'mouse_move_pearson_x': 0.796,
  'mouse_move_pearson_y': 0.783,
  'no_button_false_positive_rate_max': 0.10,
}
def ge(v,t): return v is not None and float(v) >= float(t)
def le(v,t): return v is not None and float(v) <= float(t)
passes={
  'keyboard_key_accuracy': ge(observed['keyboard_key_accuracy'], targets['keyboard_key_accuracy']),
  'mouse_button_accuracy': ge(observed['mouse_button_accuracy'], targets['mouse_button_accuracy']),
  'mouse_button_f1_nonzero': ge(observed['mouse_button_f1'], targets['mouse_button_f1_min_nonzero']),
  'mouse_move_pearson_x': ge(observed['mouse_move_pearson_x'], targets['mouse_move_pearson_x']),
  'mouse_move_pearson_y': ge(observed['mouse_move_pearson_y'], targets['mouse_move_pearson_y']),
  'no_button_false_positive_rate': le(observed['no_button_false_positive_rate'], targets['no_button_false_positive_rate_max']),
}
compact={
  'schema': 'g005_statectx_train320k_stratified_calibration_prediction_compact.v1',
  'status': 'paper_target_pass' if int('$EXIT_CODE') == 0 and all(passes.values()) else ('nonterminal_negative_probe' if int('$EXIT_CODE') == 0 else 'fail'),
  'started_at': '$STARTED_AT',
  'finished_at': '$FINISHED_AT',
  'exit_code': int('$EXIT_CODE'),
  'model_slug': '$MODEL_SLUG',
  'checkpoint': '$CHECKPOINT',
  'config': '$CONFIG',
  'summary_path': str(summary_path),
  'metrics_path': str(metrics_path),
  'summary_sha256': sha(summary_path),
  'metrics_sha256': sha(metrics_path),
  'target_rows': summary.get('target_rows'),
  'fit_rows': summary.get('fit_rows'),
  'calibration_rows': summary.get('calibration_rows'),
  'calibration_strategy': summary.get('calibration_strategy'),
  'calibration_index_preview': summary.get('calibration_index_preview'),
  'non_noop_budget': summary.get('non_noop_budget'),
  'family_non_noop_budget': summary.get('family_non_noop_budget'),
  'candidate_family_diagnostics': summary.get('candidate_family_diagnostics'),
  'retrieval_action_prior': summary.get('retrieval_action_prior'),
  'candidate_token_prior': summary.get('candidate_token_prior'),
  'paper_target_observed': observed,
  'paper_targets': targets,
  'paper_target_passes': passes,
  'claim_boundary': 'Prediction-only stratified train-heldout calibration sweep from the 320k state-context masked-diffusion checkpoint; no target-label calibration and not G005 completion evidence unless followed by full-corpus promotion/audits.',
}
compact_path.write_text(json.dumps(compact, indent=2, sort_keys=True) + '\n')
print(json.dumps({'status': compact['status'], 'compact_summary': str(compact_path), 'observed': observed}, indent=2, sort_keys=True))
PY

if [[ -n "${WANDB_PROJECT:-}" ]]; then
  set +e
  uv run --extra train --with wandb python scripts/log_wandb_artifacts.py \
    --env-file .env \
    --run-name "$MODEL_SLUG" \
    --group g005-idm-paper-target \
    --job-type prediction-calibration \
    --tags "$WANDB_TAGS" \
    --artifact-name "${MODEL_SLUG}-evidence" \
    --artifact-type evaluation \
    --output "$WANDB_STATUS" \
    --json "$SUMMARY_OUT" \
    --json "$OUTPUT_DIR/paper_metrics.json" \
    --json "$COMPACT_SUMMARY"
  WANDB_EXIT="$?"
  set -e
  if [[ "$WANDB_EXIT" -ne 0 ]]; then
    python3 - <<PY
import json, pathlib, time
path=pathlib.Path("$WANDB_STATUS")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({'schema':'wandb_eval_artifact_logger_status.v1','status':'failed','exit_code':int('$WANDB_EXIT'),'run_name':'$MODEL_SLUG','updated_at_epoch':time.time(),'claim_boundary':'W&B logging failed after local prediction evidence was written; no secrets included.'}, indent=2, sort_keys=True)+'\n')
PY
  fi
else
  python3 - <<PY
import json, pathlib, time
path=pathlib.Path("$WANDB_STATUS")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({'schema':'wandb_eval_artifact_logger_status.v1','status':'skipped','reason':'WANDB_PROJECT not configured','run_name':'$MODEL_SLUG','updated_at_epoch':time.time()}, indent=2, sort_keys=True)+'\n')
PY
fi

exit "$EXIT_CODE"
