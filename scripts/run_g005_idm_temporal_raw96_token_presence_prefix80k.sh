#!/usr/bin/env bash
set -euo pipefail
MODEL_SLUG="${MODEL_SLUG:-g005_idm_temporal_masked_diffusion_raw96_token_presence_prefix80k}"
CONFIG="${CONFIG:-configs/model/idm_temporal_masked_diffusion_d2e_raw96_token_presence_prefix80k.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/idm_temporal_masked_diffusion_d2e_raw96_token_presence_prefix80k}"
export MODEL_SLUG CONFIG OUTPUT_DIR
exec bash scripts/run_g005_idm_temporal_raw96_family_presence_prefix.sh
