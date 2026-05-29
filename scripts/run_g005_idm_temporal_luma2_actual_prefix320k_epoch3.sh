#!/usr/bin/env bash
set -euo pipefail
MODEL_SLUG="${MODEL_SLUG:-g005_idm_temporal_masked_diffusion_luma2_actual_prefix320k_epoch3}"
CONFIG="${CONFIG:-configs/model/idm_temporal_masked_diffusion_d2e_luma2_actual_prefix320k_epoch3.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/idm_temporal_masked_diffusion_d2e_luma2_actual_prefix320k_epoch3}"
PROCESS_PATTERN="${PROCESS_PATTERN:-train_idm_temporal_masked_diffusion|run_g005_idm_temporal_luma2_actual_prefix320k_epoch3|run_g005_idm_temporal_raw96_family_presence_prefix}"
WANDB_TAGS="${WANDB_TAGS:-g005,idm,d2e,fdm1-recipe,actual-luma,masked-diffusion,prefix320k,ddp-ready}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export MODEL_SLUG CONFIG OUTPUT_DIR PROCESS_PATTERN WANDB_TAGS NPROC_PER_NODE
exec bash scripts/run_g005_idm_temporal_raw96_family_presence_prefix.sh
