#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi

MODEL_SLUG="${MODEL_SLUG:-g005_idm_temporal_masked_diffusion_raw96_patch_axisclass_realvideo_prefix32k}"
CONFIG="${CONFIG:-configs/model/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_prefix32k.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_prefix32k}"
SOURCE_INPUT_ROOT="${SOURCE_INPUT_ROOT:-outputs/data/d2e_event_state_duration_context_shards_accel64}"
SOURCE_PREFIX_ROOT="${SOURCE_PREFIX_ROOT:-outputs/data/d2e_event_state_duration_realvideo_balanced_prefix32k}"
SOURCE_TRAIN_PREFIX_SUMMARY="${SOURCE_TRAIN_PREFIX_SUMMARY:-artifacts/idm/${MODEL_SLUG}_source_train_balanced_summary.json}"
SOURCE_TARGET_PREFIX_SUMMARY="${SOURCE_TARGET_PREFIX_SUMMARY:-artifacts/idm/${MODEL_SLUG}_source_target_balanced_summary.json}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-32000}"
MAX_TARGET_ROWS="${MAX_TARGET_ROWS:-24000}"
TRAIN_MAX_PER_RECORDING="${TRAIN_MAX_PER_RECORDING:-2000}"
TARGET_ROWS_PER_SPLIT="${TARGET_ROWS_PER_SPLIT:-8000}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
PROCESS_PATTERN="${PROCESS_PATTERN:-train_idm_temporal_masked_diffusion|run_g005_idm_temporal_raw96_patch_axisclass_realvideo_prefix32k|run_g005_idm_temporal_raw96_family_presence_prefix}"
WANDB_TAGS="${WANDB_TAGS:-g005,idm,d2e,fdm1-recipe,real-video,raw96,axisclass,balanced-prefix}"

mkdir -p "$SOURCE_PREFIX_ROOT" artifacts/idm outputs/cluster "$OUTPUT_DIR"

if [[ ! -s "$SOURCE_PREFIX_ROOT/train_core.jsonl" || ! -s "$SOURCE_TRAIN_PREFIX_SUMMARY" ]]; then
  uv run python scripts/materialize_balanced_prefix.py \
    --input "$SOURCE_INPUT_ROOT/shard_*/train_core.jsonl" \
    --output "$SOURCE_PREFIX_ROOT/train_core.jsonl" \
    --summary-out "$SOURCE_TRAIN_PREFIX_SUMMARY" \
    --balance-key recording_id \
    --max-per-group "$TRAIN_MAX_PER_RECORDING" \
    --max-rows "$MAX_TRAIN_ROWS" \
    --source-label "${MODEL_SLUG}_source_train_balanced"
fi

if [[ ! -s "$SOURCE_PREFIX_ROOT/target_all_eval.jsonl" || ! -s "$SOURCE_TARGET_PREFIX_SUMMARY" ]]; then
  uv run python scripts/materialize_balanced_prefix.py \
    --input "$SOURCE_INPUT_ROOT/shard_*/target_all_eval.jsonl" \
    --output "$SOURCE_PREFIX_ROOT/target_all_eval.jsonl" \
    --summary-out "$SOURCE_TARGET_PREFIX_SUMMARY" \
    --balance-key eval_split_tags \
    --group-value temporal \
    --group-value heldout_recording \
    --group-value heldout_game \
    --per-group-rows "$TARGET_ROWS_PER_SPLIT" \
    --max-rows "$MAX_TARGET_ROWS" \
    --source-label "${MODEL_SLUG}_source_target_balanced"
fi

export MODEL_SLUG CONFIG OUTPUT_DIR NPROC_PER_NODE PROCESS_PATTERN WANDB_TAGS
exec bash scripts/run_g005_idm_temporal_raw96_family_presence_prefix.sh
