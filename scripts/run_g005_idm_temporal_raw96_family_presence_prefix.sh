#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi

CONFIG="${CONFIG:-configs/model/idm_temporal_masked_diffusion_d2e_raw96_family_presence_prefix80k.yaml}"
MODEL_SLUG="${MODEL_SLUG:-g005_idm_temporal_masked_diffusion_raw96_family_presence_prefix80k}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/idm_temporal_masked_diffusion_d2e_raw96_family_presence_prefix80k}"
SUMMARY_PATH="${SUMMARY_PATH:-artifacts/idm/${MODEL_SLUG}_summary.json}"
RUN_SUMMARY="${RUN_SUMMARY:-artifacts/idm/${MODEL_SLUG}_h200_run.json}"
COMPACT_SUMMARY="${COMPACT_SUMMARY:-artifacts/idm/${MODEL_SLUG}_h200_compact_summary.json}"
GPU_MONITOR_LOG="${GPU_MONITOR_LOG:-artifacts/idm/${MODEL_SLUG}_h200_gpu_monitor.csv}"
GPU_MONITOR_PID_FILE="${GPU_MONITOR_PID_FILE:-outputs/cluster/${MODEL_SLUG}_gpu_monitor.pid}"
WANDB_SIDECAR_STATUS="${WANDB_SIDECAR_STATUS:-artifacts/idm/${MODEL_SLUG}_h200_wandb_status.json}"
WANDB_SIDECAR_LOG="${WANDB_SIDECAR_LOG:-artifacts/idm/${MODEL_SLUG}_h200_wandb.log}"
WANDB_SIDECAR_PID_FILE="${WANDB_SIDECAR_PID_FILE:-outputs/cluster/${MODEL_SLUG}_wandb.pid}"
ENABLE_WANDB_SIDECAR="${ENABLE_WANDB_SIDECAR:-1}"
PROCESS_PATTERN="${PROCESS_PATTERN:-train_idm_temporal_masked_diffusion|run_g005_idm_temporal_raw96_family_presence_prefix}"

mkdir -p artifacts/idm outputs/cluster "$OUTPUT_DIR" "$(dirname "$GPU_MONITOR_LOG")"
MONITOR_PID=""
SIDECAR_PID=""
STARTED_AT="$(date -Iseconds)"
cleanup_background() {
  if [[ -n "${MONITOR_PID:-}" ]]; then
    kill "$MONITOR_PID" >/dev/null 2>&1 || true
    wait "$MONITOR_PID" >/dev/null 2>&1 || true
    MONITOR_PID=""
  fi
  if [[ -n "${SIDECAR_PID:-}" ]]; then
    wait "$SIDECAR_PID" >/dev/null 2>&1 || true
    SIDECAR_PID=""
  fi
}
trap cleanup_background EXIT

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv -l 30 >"$GPU_MONITOR_LOG" 2>&1 &
  MONITOR_PID="$!"
  echo "$MONITOR_PID" >"$GPU_MONITOR_PID_FILE"
fi

if [[ "$ENABLE_WANDB_SIDECAR" != "0" && -n "${WANDB_PROJECT:-}" ]]; then
  uv run --with wandb python scripts/watch_wandb_training.py \
    --env-file .env \
    --train-history "$OUTPUT_DIR/train_history.json" \
    --rank-progress-dir "$OUTPUT_DIR/rank_progress" \
    --gpu-monitor "$GPU_MONITOR_LOG" \
    --run-summary "$RUN_SUMMARY" \
    --checkpoint "$OUTPUT_DIR/checkpoint.pt" \
    --metadata "$OUTPUT_DIR/checkpoint_metadata.json" \
    --output "$WANDB_SIDECAR_STATUS" \
    --pid-file "$WANDB_SIDECAR_PID_FILE" \
    --run-name "$MODEL_SLUG" \
    --group "g005-idm-paper-target" \
    --job-type "train-sidecar" \
    --tags "g005,idm,d2e,fdm1-recipe,raw-video,masked-diffusion,prefix" \
    --poll-seconds 30 \
    --process-pattern "$PROCESS_PATTERN" \
    --finish-on-run-summary >"$WANDB_SIDECAR_LOG" 2>&1 &
  SIDECAR_PID="$!"
fi

set +e
uv run --extra train python scripts/train_idm_temporal_masked_diffusion.py --config "$CONFIG" --require-torch
EXIT_CODE="$?"
set -e
FINISHED_AT="$(date -Iseconds)"

python3 - <<PY
import csv, hashlib, json, pathlib, time
run_summary = pathlib.Path("$RUN_SUMMARY")
compact_summary = pathlib.Path("$COMPACT_SUMMARY")
summary_path = pathlib.Path("$SUMMARY_PATH")
metrics_path = pathlib.Path("$OUTPUT_DIR/paper_metrics.json")
gpu_monitor = pathlib.Path("$GPU_MONITOR_LOG")
wandb_status = pathlib.Path("$WANDB_SIDECAR_STATUS")

def load(path):
    if path.exists():
        return json.loads(path.read_text())
    return None

def sha(path):
    if not path.exists():
        return None
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def gpu_status(path):
    if not path.exists() or path.stat().st_size == 0:
        return {'status':'missing','rows':0,'unique_gpu_indices':[]}
    rows=0; indices=set(); util=[]
    with path.open() as f:
        reader=csv.reader(f)
        header=next(reader, None)
        for row in reader:
            if len(row) < 4:
                continue
            rows += 1
            try: indices.add(int(str(row[1]).strip()))
            except Exception: pass
            try: util.append(float(str(row[3]).strip()))
            except Exception: pass
    return {'status':'pass' if rows else 'empty','rows':rows,'unique_gpu_indices':sorted(indices),'max_gpu_utilization':max(util) if util else None}

payload={
  'schema':'g005_temporal_raw96_family_presence_prefix_h200_run.v1',
  'status':'pass' if int('$EXIT_CODE') == 0 else 'fail',
  'exit_code':int('$EXIT_CODE'),
  'started_at':'$STARTED_AT',
  'finished_at':'$FINISHED_AT',
  'config':'$CONFIG',
  'model_slug':'$MODEL_SLUG',
  'summary_path':str(summary_path),
  'summary':load(summary_path),
  'metrics_path':str(metrics_path),
  'paper_metrics':load(metrics_path),
  'gpu_monitor_log':str(gpu_monitor),
  'gpu_monitor_sha256':sha(gpu_monitor),
  'gpu_monitor_status':gpu_status(gpu_monitor),
  'wandb_sidecar_status_path':str(wandb_status),
  'wandb_sidecar_status':load(wandb_status),
  'claim_boundary':'Bounded 1xH200/prefix raw-video FDM-1-recipe IDM probe; not G005 completion evidence unless paper targets are beaten and audited.'
}
run_summary.parent.mkdir(parents=True, exist_ok=True)
run_summary.write_text(json.dumps(payload, indent=2, sort_keys=True)+'\n')

metrics = payload.get('paper_metrics') or {}
groups = metrics.get('groups') or {}
all_group = groups.get('all') or {}
paper = all_group.get('paper_compatible') or {}
strict = all_group.get('strict_local') or {}
keyboard = paper.get('keyboard') or {}
button = paper.get('mouse_button') or {}
move = paper.get('mouse_move') or {}
strict_button = strict.get('mouse_button') or {}
targets = {
  'keyboard_key_accuracy': 0.73,
  'mouse_button_accuracy': 0.957,
  'mouse_button_f1_min_nonzero': 1e-12,
  'mouse_move_pearson_x': 0.796,
  'mouse_move_pearson_y': 0.783,
  'no_button_false_positive_rate_max': 0.10,
}
observed = {
  'keyboard_key_accuracy': keyboard.get('key_accuracy'),
  'mouse_button_accuracy': button.get('button_accuracy'),
  'mouse_button_f1': strict_button.get('f1'),
  'mouse_move_pearson_x': move.get('pearson_x'),
  'mouse_move_pearson_y': move.get('pearson_y'),
  'no_button_false_positive_rate': strict_button.get('no_button_false_positive_rate'),
}
def ge(value, threshold):
    return value is not None and float(value) >= float(threshold)
def le(value, threshold):
    return value is not None and float(value) <= float(threshold)
gates = {
  'keyboard_key_accuracy': ge(observed['keyboard_key_accuracy'], targets['keyboard_key_accuracy']),
  'mouse_button_accuracy': ge(observed['mouse_button_accuracy'], targets['mouse_button_accuracy']),
  'mouse_button_f1_nonzero': ge(observed['mouse_button_f1'], targets['mouse_button_f1_min_nonzero']),
  'mouse_move_pearson_x': ge(observed['mouse_move_pearson_x'], targets['mouse_move_pearson_x']),
  'mouse_move_pearson_y': ge(observed['mouse_move_pearson_y'], targets['mouse_move_pearson_y']),
  'no_button_false_positive_rate': le(observed['no_button_false_positive_rate'], targets['no_button_false_positive_rate_max']),
}
compact = {
  'schema': 'g005_temporal_raw96_h200_compact_summary.v1',
  'status': 'paper_target_pass' if int('$EXIT_CODE') == 0 and all(gates.values()) else ('fail' if int('$EXIT_CODE') != 0 else 'nonterminal_negative_probe'),
  'exit_code': int('$EXIT_CODE'),
  'model_slug': '$MODEL_SLUG',
  'config': '$CONFIG',
  'run_summary': str(run_summary),
  'summary_path': str(summary_path),
  'metrics_path': str(metrics_path),
  'paper_target_gates': gates,
  'paper_target_observed': observed,
  'paper_targets': targets,
  'train_rows': (payload.get('summary') or {}).get('train_rows'),
  'fit_rows': (payload.get('summary') or {}).get('fit_rows'),
  'target_rows': (payload.get('summary') or {}).get('target_rows'),
  'video_encoder_arch': (payload.get('summary') or {}).get('video_encoder_arch'),
  'video_tokens_per_offset': (payload.get('summary') or {}).get('video_tokens_per_offset'),
  'temporal_window': (payload.get('summary') or {}).get('temporal_window'),
  'loss_weights': (payload.get('summary') or {}).get('loss_weights'),
  'gpu_monitor_status': payload.get('gpu_monitor_status'),
  'wandb_sidecar_status': payload.get('wandb_sidecar_status'),
  'claim_boundary': 'Compact probe evidence only. G005 remains incomplete unless status=paper_target_pass and separate completion/goal checkpoint gates pass.',
}
compact_summary.parent.mkdir(parents=True, exist_ok=True)
compact_summary.write_text(json.dumps(compact, indent=2, sort_keys=True)+'\n')
PY

exit "$EXIT_CODE"
