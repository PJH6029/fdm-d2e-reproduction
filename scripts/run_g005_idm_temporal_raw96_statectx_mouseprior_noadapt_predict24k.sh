#!/usr/bin/env bash
set -euo pipefail

MODEL_SLUG="${MODEL_SLUG:-g005_idm_temporal_masked_diffusion_raw96_patch_axisclass_realvideo_statectx_train320k_mouseprior_noadapt_predict24k}"
CONFIG="${CONFIG:-configs/model/idm_temporal_masked_diffusion_d2e_raw96_patch_axisclass_realvideo_statectx_train320k_mouseprior_noadapt_predict24k.yaml}"
WANDB_TAGS="${WANDB_TAGS:-g005,idm,d2e,fdm1-recipe,real-video,raw96,state-context,prediction-only,train-fit-mouse-prior,no-target-adapt,single-process-cache-write}"
export MODEL_SLUG CONFIG WANDB_TAGS
exec bash scripts/run_g005_idm_temporal_raw96_statectx_train320k_predict24k.sh
