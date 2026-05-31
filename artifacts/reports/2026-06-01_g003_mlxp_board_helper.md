# G003 MLXP board helper update — 2026-06-01 KST

## Target

Keep the G003 reservation path reproducible without creating a live production reservation. The previous reservation payload can become stale as board slots pass, so the local helper now supports safe board inspection and summary generation before re-planning the payload.

## Added behavior

- `scripts/mlxp_reservation_helper.py board`
  - Uses the authenticated production board endpoint: `GET /api/projects/production/board?days=<N>`.
  - Writes the full board snapshot when `--output` is supplied.
  - Writes a small summary when `--summary-output` is supplied.
  - Prints the summary by default instead of dumping the full board.
- `summarize_board(...)`
  - Reports node ids, row count, slot size, default managed image key, managed/registry profile keys, and free cell counts by node.

## Safe refresh command

```bash
uv run python scripts/mlxp_reservation_helper.py board \
  --output .omx/tmp/mlxp_board_latest.json \
  --summary-output artifacts/mlxp/g003_action_dataset_board_summary.json

uv run python scripts/plan_mlxp_reservation_payload.py \
  --board-json .omx/tmp/mlxp_board_latest.json \
  --output artifacts/mlxp/g003_action_dataset_reservation_payload_draft.json \
  --validation-output artifacts/mlxp/g003_action_dataset_reservation_payload_validation.json
```

This only inspects and plans. It does not create a reservation.

## Verification

```bash
uv run python -m py_compile scripts/mlxp_reservation_helper.py
uv run pytest tests/test_mlxp_reservation_helper.py -q
```

Observed evidence:

- `tests/test_mlxp_reservation_helper.py`: `7 passed`.
- Live board refresh was skipped in this local shell because no MLXP reservation API token environment variable was present.

## Claim boundary

This helper update does not reserve GPUs and does not complete G003. It only keeps the eventual reservation payload auditable and refreshable before an explicitly confirmed live production create.
