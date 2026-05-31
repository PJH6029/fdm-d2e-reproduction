#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi

export MODEL_SLUG="${MODEL_SLUG:-g005_idm_temporal_masked_diffusion_raw96_patch_axisclass_realvideo_statectx_prefix32k}"
export CONFIG="${CONFIG:-configs/model/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_statectx_prefix32k.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_statectx_prefix32k}"
# Reuse the already-balanced event-state-duration prefix rows used by the mouse-aggregate probe.
# They contain prior_action_tokens, hold durations, and previous_event_tokens; the new config only
# changes the masked-diffusion student's conditioning features and output namespace.
export SOURCE_PREFIX_ROOT="${SOURCE_PREFIX_ROOT:-outputs/data/d2e_event_state_duration_realvideo_balanced_mouseagg_prefix32k}"
export SOURCE_TRAIN_PREFIX_SUMMARY="${SOURCE_TRAIN_PREFIX_SUMMARY:-artifacts/idm/g005_idm_temporal_masked_diffusion_raw96_patch_axisclass_realvideo_mouseagg_prefix32k_source_train_balanced_summary.json}"
export SOURCE_TARGET_PREFIX_SUMMARY="${SOURCE_TARGET_PREFIX_SUMMARY:-artifacts/idm/g005_idm_temporal_masked_diffusion_raw96_patch_axisclass_realvideo_mouseagg_prefix32k_source_target_balanced_summary.json}"
export PROCESS_PATTERN="${PROCESS_PATTERN:-train_idm_temporal_masked_diffusion|run_g005_idm_temporal_raw96_patch_axisclass_realvideo_statectx_prefix32k|run_g005_idm_temporal_raw96_family_presence_prefix}"
export WANDB_TAGS="${WANDB_TAGS:-g005,idm,d2e,fdm1-recipe,real-video,raw96,axisclass,mouse-aggregate-decomposed,state-context,balanced-prefix}"
export SUMMARY_PATH="${SUMMARY_PATH:-artifacts/idm/${MODEL_SLUG}_summary.json}"

if [[ ! -s "$SOURCE_PREFIX_ROOT/train_core.jsonl" || ! -s "$SOURCE_PREFIX_ROOT/target_all_eval.jsonl" ]]; then
  exec bash scripts/run_g005_idm_temporal_raw96_patch_axisclass_realvideo_prefix32k.sh
fi

export MODEL_SLUG CONFIG OUTPUT_DIR NPROC_PER_NODE PROCESS_PATTERN WANDB_TAGS SUMMARY_PATH
exec bash scripts/run_g005_idm_temporal_raw96_family_presence_prefix.sh
