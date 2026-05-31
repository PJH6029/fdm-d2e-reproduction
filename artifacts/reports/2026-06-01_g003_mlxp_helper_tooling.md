# G003 progress — MLXP reservation helper tooling

**Status:** helper tooling only; no live production reservation was created.

## Implemented in this milestone

- Added `scripts/mlxp_reservation_helper.py` for safe MLXP production reservation workflows:
  - `validate-payload`
  - `create` with required `--i-confirm-live-production-reservation`
  - `status`
  - `cancel` with required `--i-confirm-cancel-reservation`
- Added `tests/test_mlxp_reservation_helper.py` to verify payload validation and to ensure live create/cancel paths require explicit confirmation flags.
- Validated the G003 draft payload and wrote `artifacts/mlxp/g003_action_dataset_reservation_payload_validation.json`.

## Validation output

The current draft payload validates as `status=pass` and summarizes:

```json
{
  "project_id": "production",
  "node_id": "1",
  "gpu_start": 2,
  "gpu_count": 1,
  "start_at": "2026-06-01T01:00:00+09:00",
  "end_at": "2026-06-01T13:00:00+09:00",
  "managed_image_key": "base",
  "image_path": "",
  "purpose": "Continuous GUI - FDM reproduction: G003 D2E-480p CPU/IO action-slot materialization and audit; reserve 1xH200 only for managed production workspace/PVC access, cancel promptly if GPU remains idle after setup"
}
```

## Verification

```text
uv run python -m py_compile scripts/mlxp_reservation_helper.py
uv run pytest tests/test_mlxp_reservation_helper.py -q
uv run python scripts/mlxp_reservation_helper.py validate-payload --payload artifacts/mlxp/g003_action_dataset_reservation_payload_draft.json --output artifacts/mlxp/g003_action_dataset_reservation_payload_validation.json
python3 -m json.tool artifacts/mlxp/g003_action_dataset_reservation_payload_validation.json
git diff --check
```

Result: targeted checks passed (`5 passed`).

## Claim boundary

This milestone validates the payload and helper safety only. It does not create a reservation, run D2E materialization, or complete G003.
