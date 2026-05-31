#!/usr/bin/env bash
set -euo pipefail

# Continue the best state-context masked-diffusion IDM checkpoint with a low-weight
# train-only mouse-motion teacher loss. This is still the public FDM-1-shaped
# masked action-token objective; the source checkpoint is an initialization, not
# a prediction shortcut or target-label calibration path.
export MODEL_SLUG="${MODEL_SLUG:-g005_idm_temporal_masked_diffusion_raw96_patch_axisclass_realvideo_statectx_warm_teacher_motiondistill_train320k_target24k}"
export CONFIG="${CONFIG:-configs/model/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_statectx_warm_teacher_motiondistill_train320k_target24k.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_statectx_warm_teacher_motiondistill_train320k_target24k}"
export PROCESS_PATTERN="${PROCESS_PATTERN:-train_idm_temporal_masked_diffusion|run_g005_idm_temporal_raw96_statectx_warm_teacher_motiondistill_train320k|run_g005_idm_temporal_raw96_statectx_teacher_motiondistill_train320k|run_g005_idm_temporal_raw96_family_presence_prefix}"
export WANDB_TAGS="${WANDB_TAGS:-g005,idm,d2e,fdm1-recipe,real-video,raw96,state-context,warm-start,teacher-motion-distill,balanced-prefix,train320k}"

exec bash scripts/run_g005_idm_temporal_raw96_statectx_teacher_motiondistill_train320k.sh
