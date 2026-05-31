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

uv run python scripts/finalize_g003_fdm1_action_dataset.py \
  --config configs/data/fdm1_g003_action_dataset_finalization.yaml \
  ${FINALIZE_EXTRA_ARGS:-}


uv run python scripts/build_fdm1_g003_evidence_bundle.py   --completion-config configs/eval/fdm1_g003_action_dataset_completion.yaml   ${BUNDLE_EXTRA_ARGS:-}
