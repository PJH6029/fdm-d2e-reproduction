# G003 tiny real-D2E smoke plan — 2026-06-01 KST

## Target

Plan a tiny real-D2E smoke that can be run before or inside the MLXP pod to verify actual D2E-480p data access and extraction/finalization command shape. This is explicitly not a full-corpus completion artifact.

## Added artifacts

- `configs/data/fdm1_g003_realdata_smoke.yaml`
  - Bounds the smoke to `max_recordings=1`, `max_bins_per_recording=8`, `event_limit=2000`, local `.omx/tmp` cache/output paths, and `video_mode=remote`.
- `scripts/plan_g003_realdata_smoke.py`
  - Selects the first D2E-480p universe row from the pinned G002 data universe.
  - Emits a JSON plan and shell script without running downloads by default.
  - Generates smoke-specific finalization/completion configs under `.omx/tmp` so real-D2E tiny output cannot be confused with canonical G003 full-corpus artifacts.
- `artifacts/cluster/fdm1_g003_realdata_smoke_plan.json`
  - Current selected row: `d2e_480p:Apex_Legends/0805_01`.
- `artifacts/cluster/fdm1_g003_realdata_smoke.sh`
  - Executable smoke script.
- `tests/test_plan_g003_realdata_smoke.py`
  - Verifies row selection, bounded extract args, claim boundary, and CLI generation.

## Intended use

```bash
uv run python scripts/plan_g003_realdata_smoke.py
bash artifacts/cluster/fdm1_g003_realdata_smoke.sh
```

## Verification

```bash
uv run python -m py_compile scripts/plan_g003_realdata_smoke.py
uv run pytest tests/test_plan_g003_realdata_smoke.py -q
uv run python scripts/plan_g003_realdata_smoke.py \
  --output artifacts/cluster/fdm1_g003_realdata_smoke_plan.json \
  --shell-out artifacts/cluster/fdm1_g003_realdata_smoke.sh
uv run python -m json.tool artifacts/cluster/fdm1_g003_realdata_smoke_plan.json
```

Observed evidence:

- Realdata smoke planner tests: `2 passed`.
- Plan generated for one pinned D2E-480p row with 8 bins and 2000-event cap.

## Claim boundary

This is a tiny data-access smoke plan only. It does not replace the required full D2E-480p materialization run, does not satisfy G003 completion, and must not be used for OMX checkpointing.
