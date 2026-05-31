# G003 progress — streaming output hashes for large action-slot packs

**Status:** partial progress for `G003-50ms-action-token-dataset-pipeline`; not a G003 completion checkpoint.

## Implemented in this milestone

- The streaming action-slot materializer now computes SHA-256 digests for every output JSONL while writing, avoiding a second full scan of large D2E-480p files.
- `dataset_summary.json` now records `output_hashes` for `all`, reset split packs, and pseudo-label simulation packs.
- The G003 completion audit now omits direct sha256 rescans for large JSONL artifacts and instead requires their write-time hashes in `dataset_summary.output_hashes`.
- Non-streaming fixture materialization also records `output_hashes` for parity in tests and small debug runs.

## Verification

```text
uv run python -m py_compile src/fdm_d2e/data/fdm1_action_dataset.py src/fdm_d2e/reporting/fdm1_g003_completion.py src/fdm_d2e/data/fdm1_g003_finalization.py
uv run pytest tests/test_fdm1_action_dataset_materializer.py tests/test_fdm1_g003_completion_audit.py tests/test_fdm1_g003_finalization.py -q
python3 -m json.tool configs/eval/fdm1_g003_action_dataset_completion.yaml
git diff --check
```

Result: targeted checks passed (`11 passed`).

## Remaining G003 work

Run the full D2E-480p pipeline on MLXP/PVC and verify `dataset_summary.output_hashes` covers all required output roles before checkpointing G003.
