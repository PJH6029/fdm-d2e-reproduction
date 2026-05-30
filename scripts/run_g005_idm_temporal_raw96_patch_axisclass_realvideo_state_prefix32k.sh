#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi

export MODEL_SLUG="${MODEL_SLUG:-g005_idm_temporal_masked_diffusion_raw96_patch_axisclass_realvideo_state_prefix32k}"
export CONFIG="${CONFIG:-configs/model/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_state_prefix32k.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_state_prefix32k}"
export SOURCE_PREFIX_ROOT="${SOURCE_PREFIX_ROOT:-outputs/data/d2e_event_state_duration_realvideo_balanced_state_prefix32k}"
export SOURCE_TRAIN_PREFIX_SUMMARY="${SOURCE_TRAIN_PREFIX_SUMMARY:-artifacts/idm/${MODEL_SLUG}_source_train_balanced_summary.json}"
export SOURCE_TARGET_PREFIX_SUMMARY="${SOURCE_TARGET_PREFIX_SUMMARY:-artifacts/idm/${MODEL_SLUG}_source_target_balanced_summary.json}"
export PROCESS_PATTERN="${PROCESS_PATTERN:-train_idm_temporal_masked_diffusion|run_g005_idm_temporal_raw96_patch_axisclass_realvideo_state_prefix32k|run_g005_idm_temporal_raw96_family_presence_prefix}"
export WANDB_TAGS="${WANDB_TAGS:-g005,idm,d2e,fdm1-recipe,real-video,raw96,axisclass,held-state-token,closed-loop-eventify,balanced-prefix}"
export SUMMARY_PATH="${SUMMARY_PATH:-artifacts/idm/${MODEL_SLUG}_summary.json}"

exec bash scripts/run_g005_idm_temporal_raw96_patch_axisclass_realvideo_prefix32k.sh
