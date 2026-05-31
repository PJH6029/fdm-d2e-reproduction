# G003 progress — MLXP payload planner

**Status:** planner tooling only; no live production reservation was created.

## Implemented in this milestone

- Added `scripts/plan_mlxp_reservation_payload.py` to generate a fresh MLXP production reservation payload from a board JSON snapshot.
- The planner finds a contiguous same-node free GPU window, uses the board default managed image, validates the payload with `scripts/mlxp_reservation_helper.py` rules, and writes both payload and validation artifacts.
- Added `tests/test_plan_mlxp_reservation_payload.py` for free-window search, busy-cell avoidance, payload construction, and CLI output.

## Latest planned payload

Generated from `.omx/tmp/mlxp_board_latest.json` with board time `2026-06-01T01:28:47+09:00`:

```json
{
  "node_id": "1",
  "gpu_start": 2,
  "gpu_count": 1,
  "gpu_indices": [],
  "start_at": "2026-06-01T01:00:00+09:00",
  "end_at": "2026-06-01T13:00:00+09:00",
  "purpose": "Continuous GUI - FDM reproduction: G003 D2E-480p CPU/IO action-slot materialization and audit; reserve 1xH200 only for managed production workspace/PVC access, cancel promptly if GPU remains idle after setup",
  "managed_image_key": "base",
  "registry_profile_key": "",
  "image_path": "",
  "command": [],
  "args": [],
  "actor_name": "jeonghunpark"
}
```

Validation artifact: `artifacts/mlxp/g003_action_dataset_reservation_payload_validation.json` with `status=pass`.

## Verification

```text
uv run python -m py_compile scripts/plan_mlxp_reservation_payload.py
uv run pytest tests/test_plan_mlxp_reservation_payload.py -q
uv run python scripts/plan_mlxp_reservation_payload.py --board-json .omx/tmp/mlxp_board_latest.json --output artifacts/mlxp/g003_action_dataset_reservation_payload_draft.json --validation-output artifacts/mlxp/g003_action_dataset_reservation_payload_validation.json
python3 -m json.tool artifacts/mlxp/g003_action_dataset_reservation_payload_draft.json
python3 -m json.tool artifacts/mlxp/g003_action_dataset_reservation_payload_validation.json
git diff --check
```

Result: targeted checks passed (`4 passed`).

## Claim boundary

This planner only prepares a payload. It does not create a reservation, run the D2E materialization pipeline, or complete G003.
