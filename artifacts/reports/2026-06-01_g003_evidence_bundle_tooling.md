# G003 progress — small evidence bundle tooling

**Status:** partial progress for `G003-50ms-action-token-dataset-pipeline`; not a G003 completion checkpoint.

## Implemented in this milestone

- Added `scripts/build_fdm1_g003_evidence_bundle.py` to stage small post-run evidence files into `artifacts/sources/fdm1_g003_evidence_bundle/`.
- The bundle manifest records large JSONL action-slot packs without copying them; their SHA-256 values come from `dataset_summary.output_hashes` generated during streaming writes.
- Updated `scripts/run_g003_fdm1_action_dataset_pipeline.sh` to build the evidence bundle after the completion audit passes.
- Added `tests/test_fdm1_g003_evidence_bundle.py` for CLI behavior, copied small artifacts, and large-pack hash mapping.

## Verification

```text
uv run python -m py_compile scripts/build_fdm1_g003_evidence_bundle.py
uv run pytest tests/test_fdm1_g003_evidence_bundle.py -q
git diff --check
```

Result: targeted checks passed (`1 passed`).

## Remaining G003 work

Run the full G003 pipeline on MLXP/PVC. After the evidence bundle is generated, copy back and commit the small bundle manifest/artifacts before checkpointing G003.
