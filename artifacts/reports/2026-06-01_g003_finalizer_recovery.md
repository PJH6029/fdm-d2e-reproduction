# G003 progress — resumable action dataset finalizer

**Status:** partial progress for `G003-50ms-action-token-dataset-pipeline`; not a G003 completion checkpoint.

## Implemented in this milestone

- Added `src/fdm_d2e/data/fdm1_g003_finalization.py` to recover/finalize missing G003 action-slot artifacts after decoded D2E-480p window records exist.
- Added `scripts/finalize_g003_fdm1_action_dataset.py` and `configs/data/fdm1_g003_action_dataset_finalization.yaml`.
- Updated `scripts/run_g003_fdm1_action_dataset_pipeline.sh` so the long extraction step is followed by the resumable finalizer rather than several non-resumable ad-hoc commands.
- Added `tests/test_fdm1_g003_finalization.py` covering first-run generation, second-run skip behavior, and CLI execution.

## Verification

```text
uv run python -m py_compile src/fdm_d2e/data/fdm1_g003_finalization.py scripts/finalize_g003_fdm1_action_dataset.py
uv run pytest tests/test_fdm1_g003_finalization.py tests/test_fdm1_g003_completion_audit.py tests/test_fdm1_action_dataset_materializer.py -q
python3 -m json.tool configs/data/fdm1_g003_action_dataset_finalization.yaml
bash -n scripts/run_g003_fdm1_action_dataset_pipeline.sh
git diff --check
```

Result: targeted checks passed (`10 passed`).

## Remaining G003 work

Run `scripts/run_g003_fdm1_action_dataset_pipeline.sh` on the MLXP/PVC D2E workspace. If extraction succeeds but finalization is interrupted, rerun `scripts/finalize_g003_fdm1_action_dataset.py` to regenerate missing fitted-bin/materialization/alignment/audit artifacts.
