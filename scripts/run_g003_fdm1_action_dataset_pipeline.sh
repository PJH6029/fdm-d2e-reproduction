#!/usr/bin/env bash
set -euo pipefail

# CPU/IO-heavy reset G003 pipeline. Run on MLXP/PVC after the repo branch is pulled.
# Optional env knobs:
#   EXTRACT_EXTRA_ARGS='--max-recordings 2 --force'
#   ACTION_EXTRA_ARGS='--max-records 10000'
#   ALIGNMENT_MAX_ROWS=24

export PATH="$HOME/.local/bin:$PATH"

uv run python scripts/extract_d2e_full_corpus.py \
  --config configs/data/fdm1_d2e_480p_full_corpus_extract.yaml \
  ${EXTRACT_EXTRA_ARGS:-}

uv run python scripts/build_fdm1_mouse_bins.py \
  --input-records outputs/data/fdm1_d2e_480p_window_records/all_records.jsonl \
  --base-tokenization-config configs/tokenization/fdm1_action_slots.json \
  --bins-output artifacts/sources/fdm1_g003_fitted_mouse_bins.json \
  --fitted-config-output artifacts/sources/fdm1_action_slots_fitted_config.json

uv run python scripts/materialize_fdm1_action_dataset.py \
  --config configs/data/fdm1_action_dataset.yaml \
  ${ACTION_EXTRA_ARGS:-}

uv run python scripts/build_fdm1_action_alignment_report.py \
  --action-slots outputs/data/fdm1_action_slots/action_slots.jsonl \
  --markdown-out artifacts/reports/fdm1_g003_action_alignment_visual_check.md \
  --audit-out artifacts/sources/fdm1_g003_action_alignment_visual_check.json \
  --max-rows "${ALIGNMENT_MAX_ROWS:-24}"

uv run python scripts/validate_fdm1_g003_action_dataset_completion.py
