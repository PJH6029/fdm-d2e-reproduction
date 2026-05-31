# G003 progress — fitted mouse bins and completion-gate tooling

**Status:** partial progress for `G003-50ms-action-token-dataset-pipeline`; not a G003 completion checkpoint.

## Implemented in this milestone

- Added train-split global mouse-bin fitting:
  - `src/fdm_d2e/data/fdm1_mouse_bins.py`
  - `scripts/build_fdm1_mouse_bins.py`
  - `artifacts/sources/fdm1_g003_fitted_mouse_bins.json` and `artifacts/sources/fdm1_action_slots_fitted_config.json` are the intended runtime outputs.
- Extended `src/fdm_d2e/tokenization/fdm1_actions.py` with a histogram-based boundary fitter so full-corpus runs do not keep every raw mouse delta in memory.
- Updated `configs/data/fdm1_action_dataset.yaml` to consume the fitted tokenization config rather than the static default config.
- Added reset G003 completion audit:
  - `src/fdm_d2e/reporting/fdm1_g003_completion.py`
  - `scripts/validate_fdm1_g003_action_dataset_completion.py`
  - `configs/eval/fdm1_g003_action_dataset_completion.yaml`
- Added `scripts/run_g003_fdm1_action_dataset_pipeline.sh`, the CPU/IO-heavy full G003 pipeline wrapper:
  1. decode/extract D2E-480p with G002 split manifests,
  2. fit mouse bins from `train_core`,
  3. materialize streaming action-slot packs,
  4. build sampled visual alignment report,
  5. run the G003 action dataset completion audit.

## Verification

```text
uv run python -m py_compile src/fdm_d2e/tokenization/fdm1_actions.py src/fdm_d2e/data/fdm1_mouse_bins.py scripts/build_fdm1_mouse_bins.py
uv run pytest tests/test_fdm1_mouse_bins.py tests/test_fdm1_action_slots.py tests/test_fdm1_action_dataset_materializer.py -q
uv run python -m py_compile src/fdm_d2e/reporting/fdm1_g003_completion.py scripts/validate_fdm1_g003_action_dataset_completion.py
uv run pytest tests/test_fdm1_g003_completion_audit.py tests/test_fdm1_mouse_bins.py tests/test_fdm1_action_dataset_materializer.py -q
python3 -m json.tool configs/eval/fdm1_g003_action_dataset_completion.yaml
bash -n scripts/run_g003_fdm1_action_dataset_pipeline.sh
git diff --check
```

Result: targeted checks passed (`15 passed` for mouse-bin/materializer set; `11 passed` for completion-audit set).

## Remaining G003 work

The tooling is ready, but G003 is still incomplete until the full wrapper runs on D2E-480p storage and produces passing runtime evidence:

- `artifacts/sources/fdm1_d2e_480p_window_records_decode_summary.json`
- `artifacts/sources/fdm1_g003_fitted_mouse_bins.json`
- `artifacts/sources/fdm1_action_slots_fitted_config.json`
- `outputs/data/fdm1_action_slots/action_slots.jsonl`
- `outputs/data/fdm1_action_slots/dataset_summary.json`
- `outputs/data/fdm1_action_slots/overflow_summary.json`
- `outputs/data/fdm1_action_slots/alignment_summary.json`
- `artifacts/reports/fdm1_g003_action_alignment_visual_check.md`
- `artifacts/sources/fdm1_g003_action_alignment_visual_check.json`
- `artifacts/sources/fdm1_g003_action_dataset_completion_audit.json` with `status=pass`.
