# G003 progress — G002 split integration for reset D2E-480p extraction

**Status:** partial progress for `G003-50ms-action-token-dataset-pipeline`; not a G003 completion checkpoint.

## Implemented in this milestone

- Added `src/fdm_d2e/data/fdm1_g003_splits.py` to apply G002 reset split manifests to decoded 50ms window records.
- Extended `scripts/extract_d2e_full_corpus.py` with `--split-mode fdm1-g002`, allowing extraction to use:
  - recording-level train/val/test manifest,
  - held-out-game manifest,
  - pseudo-label simulation manifest,
  - nested data-scale manifest.
- Added `configs/data/fdm1_d2e_480p_full_corpus_extract.yaml` as the reset extraction config for D2E-480p primary rows only.
- Updated `configs/data/fdm1_action_dataset.yaml` so packed action slots consume `outputs/data/fdm1_d2e_480p_window_records/all_records.jsonl` instead of the legacy combined-source output.
- Added `tests/test_fdm1_g003_splits.py` for split-role composition and source namespace de-collision.

## Verification

```text
uv run python -m py_compile scripts/extract_d2e_full_corpus.py src/fdm_d2e/data/fdm1_g003_splits.py
uv run pytest tests/test_fdm1_g003_splits.py tests/test_full_corpus_extraction_contract.py tests/test_fdm1_action_dataset_materializer.py -q
python3 -m json.tool configs/data/fdm1_d2e_480p_full_corpus_extract.yaml
python3 -m json.tool configs/data/fdm1_action_dataset.yaml
git diff --check
```

Result: targeted checks passed (`8 passed`).

## Remaining G003 work

Run the reset extraction/materialization flow on the full D2E-480p corpus, then preserve the resulting decode summary, action-slot dataset summary, overflow summary, alignment summary, fitted mouse-bin evidence, and sampled visual timeline reports before G003 can be checkpointed complete.
