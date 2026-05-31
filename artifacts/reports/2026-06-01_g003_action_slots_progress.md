# G003 progress — FDM-1 action-slot tokenization contract

**Status:** partial progress for `G003-50ms-action-token-dataset-pipeline`; this is not a G003 completion checkpoint.

## Canonical source

- Roadmap/spec: `ROADMAP.md`
- Ultragoal story: `.omx/ultragoal/goals.json` → `G003-50ms-action-token-dataset-pipeline`

## Implemented in this milestone

- Added `src/fdm_d2e/tokenization/fdm1_actions.py` for ROADMAP-aligned fixed-slot action serialization.
- Added `configs/tokenization/fdm1_action_slots.json` to pin the first tokenization contract:
  - 50ms bins / 20Hz visual stream assumption.
  - 49 signed mouse bins per axis with compound `MOUSE_MOVE_BIN_<xbin>_<ybin>` default.
  - fixed `K=8` sparse event slots, with `K={4,8,16}` ablation support.
  - keyboard press/release, mouse button transitions, scroll directions, `EVENT_OVERFLOW`, and click-position auxiliary targets.
- Added unit tests in `tests/test_fdm1_action_slots.py` for event-token semantics, 50ms binning, event ordering, overflow accounting, click-position horizon targets, fitted mouse boundaries, and mouse detokenization.

## Verification

```text
uv run python -m py_compile src/fdm_d2e/tokenization/fdm1_actions.py scripts/build_fdm1_g002_data_contract.py
uv run pytest tests/test_fdm1_action_slots.py tests/test_action_tokenization.py -q
python3 -m json.tool configs/tokenization/fdm1_action_slots.json
git diff --check
```

Result: targeted checks passed (`11 passed`).

## Remaining G003 work

G003 still requires D2E/OWAMcap reading integration, 20Hz frame sampling, packed action dataset writing, MCAP/predicted-action or equivalent packed-action outputs, full-dataset overflow/alignment reports, and timestamp/alignment visual checks before it can be checkpointed complete.
