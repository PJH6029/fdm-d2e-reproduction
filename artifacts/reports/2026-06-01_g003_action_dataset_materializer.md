# G003 progress — packed action-slot dataset materializer

**Status:** partial progress for `G003-50ms-action-token-dataset-pipeline`; not a G003 completion checkpoint.

## Canonical source

- Roadmap/spec: `ROADMAP.md`
- Active ultragoal story: `.omx/ultragoal/goals.json` → `G003-50ms-action-token-dataset-pipeline`

## Implemented in this milestone

- Added `src/fdm_d2e/data/fdm1_action_dataset.py`, a reusable materializer from decoded D2E 50ms window records to FDM-1-style fixed action-slot rows.
- Added `scripts/materialize_fdm1_action_dataset.py`, a streaming-by-default CLI that reads one or more decoded D2E JSONL record files by consecutive recording group and writes:
  - `action_slots.jsonl`
  - split-specific packed JSONLs under `splits/`
  - `sequence_pack.json`
  - `dataset_summary.json`
  - `alignment_summary.json`
  - `overflow_summary.json`
- Added `configs/data/fdm1_action_dataset.yaml` to pin the intended full-corpus materialization command shape from `outputs/data/d2e_full_corpus/all_records.jsonl` to `outputs/data/fdm1_action_slots`, using streaming mode to avoid loading tens of millions of 50ms rows in memory.
- Extended the tokenizer so `ActionSlotTokenizer` carries the `bin_ms` used by downstream materializers.
- Added tests in `tests/test_fdm1_action_dataset_materializer.py` for packed row semantics, split writers, alignment failure detection, and CLI smoke execution.

## Verification

```text
uv run python -m py_compile src/fdm_d2e/data/fdm1_action_dataset.py scripts/materialize_fdm1_action_dataset.py src/fdm_d2e/tokenization/fdm1_actions.py
uv run pytest tests/test_fdm1_action_slots.py tests/test_fdm1_action_dataset_materializer.py tests/test_d2e_real_contract.py::D2ERealContractTests::test_build_window_records_bins_real_decoded_actions -q
python3 -m json.tool configs/data/fdm1_action_dataset.yaml
git diff --check
```

Result: targeted checks passed (`12 passed`).

## Remaining G003 work

- Integrate the materializer into the full D2E/OWAMcap extraction flow or run it against full decoded `outputs/data/d2e_full_corpus/*.jsonl` artifacts.
- Produce full-corpus action-slot summaries, overflow rates, split counts, token vocabulary statistics, and alignment evidence over all D2E-480p games.
- Add timestamp/alignment visual checks or frame/action overlay evidence before G003 completion.
- Validate that fitted global mouse bins are computed from the training split rather than only using default boundaries.
