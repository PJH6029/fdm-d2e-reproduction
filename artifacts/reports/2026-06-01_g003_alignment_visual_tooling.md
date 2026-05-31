# G003 progress — action-slot alignment visual-check tooling

**Status:** partial progress for `G003-50ms-action-token-dataset-pipeline`; not a G003 completion checkpoint.

## Implemented in this milestone

- Added `src/fdm_d2e/data/fdm1_alignment_report.py` to audit sampled FDM-1 action-slot rows for:
  - 50ms adjacent timestamp spacing,
  - IDM mask/action-token length consistency,
  - movement-token preservation under IDM masking,
  - event-slot/action-token length consistency,
  - sampled frame-path availability warnings.
- Added `scripts/build_fdm1_action_alignment_report.py` to emit a human-inspectable markdown timeline and JSON audit from `action_slots.jsonl`.
- Added `tests/test_fdm1_alignment_report.py` covering timeline rendering, malformed mask failure detection, and CLI smoke execution.

## Verification

```text
uv run python -m py_compile src/fdm_d2e/data/fdm1_alignment_report.py scripts/build_fdm1_action_alignment_report.py src/fdm_d2e/data/fdm1_action_dataset.py
uv run pytest tests/test_fdm1_alignment_report.py tests/test_fdm1_action_dataset_materializer.py tests/test_fdm1_action_slots.py -q
git diff --check
```

Result: targeted checks passed (`14 passed`).

## Remaining G003 work

This tooling must still be run on full D2E-480p action-slot materialization output. G003 completion still requires full-corpus `action_slots.jsonl`/split packs, full overflow/alignment summaries, fitted mouse-bin evidence, and sampled markdown/JSON alignment reports from real D2E recordings.
