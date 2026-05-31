# G003 synthetic pipeline smoke — 2026-06-01 KST

## Target

Verify the post-extraction G003 path composes end-to-end without using D2E data or MLXP resources. This is a local synthetic smoke only: it cannot satisfy the full-corpus G003 completion gate, but it catches integration breakage before the pod run.

## Added artifacts

- `scripts/smoke_g003_fdm1_action_dataset_pipeline.py`
  - Generates a synthetic decoded-window fixture with all required ROADMAP/G002 split roles.
  - Runs `finalize_fdm1_g003_action_dataset` to fit mouse bins, materialize fixed-slot action packs, build visual alignment artifacts, and write the completion audit.
  - Builds the G003 evidence bundle.
  - Collects monitor status.
  - Builds checkpoint handoff in an isolated temp root to prove the post-run handoff can become `ready_to_checkpoint` when pass evidence exists.
  - Writes `artifacts/sources/fdm1_g003_pipeline_smoke_summary.json`.
- `tests/test_smoke_g003_fdm1_action_dataset_pipeline.py`
  - Covers direct smoke execution and CLI summary output.

## Verification

```bash
uv run python -m py_compile scripts/smoke_g003_fdm1_action_dataset_pipeline.py
uv run pytest tests/test_smoke_g003_fdm1_action_dataset_pipeline.py -q
uv run python scripts/smoke_g003_fdm1_action_dataset_pipeline.py --force
uv run python -m json.tool artifacts/sources/fdm1_g003_pipeline_smoke_summary.json
```

Observed evidence:

- Smoke tests: `2 passed`.
- Synthetic smoke summary reports `status=pass`, `finalization_status=pass`, `evidence_bundle_status=pass`, `monitor_status=pass`, and `handoff_status=ready_to_checkpoint`.

## Claim boundary

Synthetic smoke is not D2E full-corpus evidence and must not be used for OMX G003 checkpointing. G003 still requires the real D2E-480p full materialization run, completion audit pass, evidence bundle pass, and fresh Codex `get_goal` snapshot reconciliation.
