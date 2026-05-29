#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi

MODEL_SLUG="${MODEL_SLUG:-g005_idm_temporal_masked_diffusion_raw96_patch_axisclass_realvideo_train320k_offsetensemble_predict5k}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-outputs/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_train320k_target24k}"
CHECKPOINT="${CHECKPOINT:-$BASE_OUTPUT_DIR/checkpoint.pt}"
CONFIG="${CONFIG:-configs/model/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_train320k_offsetensemble_predict5k.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_train320k_candidate_reranker_predict5k}"
SUMMARY_OUT="${SUMMARY_OUT:-artifacts/idm/${MODEL_SLUG}_summary.json}"
COMPACT_SUMMARY="${COMPACT_SUMMARY:-artifacts/idm/${MODEL_SLUG}_compact_summary.json}"
WANDB_STATUS="${WANDB_STATUS:-artifacts/idm/${MODEL_SLUG}_wandb_status.json}"
WANDB_TAGS="${WANDB_TAGS:-g005,idm,d2e,fdm1-recipe,raw96,train320k,offset-ensemble,prediction-only}"

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
import json, pathlib, hashlib
summary_path=pathlib.Path("$SUMMARY_OUT")
metrics_path=pathlib.Path("$OUTPUT_DIR/paper_metrics.json")
compact_path=pathlib.Path("$COMPACT_SUMMARY")
wandb_path=pathlib.Path("$WANDB_STATUS")

def load(path):
    return json.loads(path.read_text()) if path.exists() else None

def sha(path):
    if not path.exists(): return None
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()
summary=load(summary_path) or {}
metrics=load(metrics_path) or {}
all_metrics=(metrics.get('groups') or {}).get('all') or metrics
paper=all_metrics.get('paper_compatible', {})
strict=all_metrics.get('strict_local', {})
keyboard=(paper.get('keyboard') or {}).get('key_accuracy')
button=(paper.get('mouse_button') or {}).get('button_accuracy')
move=paper.get('mouse_move') or {}
strict_button=strict.get('mouse_button') or {}
compact={
  'schema':'g005_temporal_offsetensemble_prediction_compact.v1',
  'status':'pass' if int("$EXIT_CODE") == 0 else 'fail',
  'started_at':'$STARTED_AT',
  'finished_at':'$FINISHED_AT',
  'exit_code':int("$EXIT_CODE"),
  'model_slug':'$MODEL_SLUG',
  'checkpoint':'$CHECKPOINT',
  'config':'$CONFIG',
  'summary_path':str(summary_path),
  'metrics_path':str(metrics_path),
  'summary_sha256':sha(summary_path),
  'metrics_sha256':sha(metrics_path),
  'target_rows':summary.get('target_rows'),
  'candidate_score_reranker':summary.get('candidate_score_reranker'),
  'keyboard_key_accuracy':keyboard,
  'mouse_button_accuracy':button,
  'mouse_button_f1':strict_button.get('f1'),
  'no_button_false_positive_rate':strict_button.get('no_button_false_positive_rate'),
  'mouse_move_pearson_x':move.get('pearson_x'),
  'mouse_move_pearson_y':move.get('pearson_y'),
  'paper_target_status':'not_evaluated' if int("$EXIT_CODE") != 0 else 'nonterminal_probe',
  'claim_boundary':'Prediction-only temporal source-offset ensemble diagnostic from a prior G005 train320k checkpoint; train-heldout calibration only, no target-label calibration, not G005 completion evidence.'
}
compact_path.write_text(json.dumps(compact, indent=2, sort_keys=True)+"\n")
if '$WANDB_PROJECT':
    # Keep a small local status even when explicit W&B artifact logging is not available.
    wandb_path.write_text(json.dumps({'status':'not_logged_by_wrapper','reason':'prediction wrapper records local compact evidence; use log_wandb_artifacts.py for artifact upload if needed','tags':'$WANDB_TAGS'}, indent=2, sort_keys=True)+"\n")
print(json.dumps({'status':compact['status'],'compact_summary':str(compact_path),'keyboard_key_accuracy':keyboard,'mouse_button_accuracy':button,'mouse_move_pearson_x':move.get('pearson_x')}, indent=2, sort_keys=True))
PY
exit "$EXIT_CODE"
