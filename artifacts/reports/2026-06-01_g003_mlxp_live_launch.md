# G003 MLXP live launch progress — 2026-06-01 KST

## Target

Start the ROADMAP-canonical `G003-50ms-action-token-dataset-pipeline` full D2E-480p materialization on MLXP/PVC without consuming 4×H200 for CPU/IO-heavy setup work.

## Reservation

- Reservation: `rsv-jeonghunpark-20260601-5c5df9`
- Project/namespace: `production` / `p-production`
- Pod: `prod-rsv-jeonghunpark-20260601-5c5df9`
- GPU allocation: 1×H200 (`gpu_indices=[2]` at reservation level; visible as GPU 0 in pod)
- Window: `2026-06-01T02:00:00+09:00` → `2026-06-01T14:00:00+09:00`
- Rationale: G003 materialization is CPU/IO-heavy; reserve one GPU only for managed production workspace/PVC access and cancel promptly if no useful pod work remains.
- Evidence:
  - `artifacts/mlxp/g003_action_dataset_board_summary.json`
  - `artifacts/mlxp/g003_action_dataset_reservation_payload_draft.json`
  - `artifacts/mlxp/g003_action_dataset_reservation_payload_validation.json`
  - `artifacts/mlxp/g003_action_dataset_reservation_create_response.safe.json`
  - `artifacts/mlxp/g003_action_dataset_reservation_detail.safe.json` with sensitive runtime token fields redacted.

## Launch attempt and fix

The first in-pod launch pulled branch `research/fdm1-d2e-ultragoal` at `c1d8edb`, installed `uv`, passed the outer launch preflight, and started background PID `290`. The pipeline then stopped before extraction because the internal preflight saw the wrapper-written PID file as an already-active G003 run.

Fix committed locally for relaunch:

- `scripts/run_g003_fdm1_action_dataset_sharded_pipeline.sh` now permits `--allow-active-pid` only when the PID file contains its own `$$`.
- `scripts/mlxp_reservation_helper.py` now redacts sensitive response fields such as `jupyter_token` before writing/printing status/create/cancel responses.

## Verification

```bash
uv run pytest tests/test_mlxp_reservation_helper.py tests/test_run_g003_fdm1_sharded_pipeline_script.py -q
```

Observed evidence: `10 passed` locally. Pod reservation status after redacted status refresh: `running`, pod phase/status `Running`, pod `prod-rsv-jeonghunpark-20260601-5c5df9`.

## Claim boundary

This report records reservation and launch progress only. G003 remains incomplete until the relaunched pod pipeline produces a passing completion audit, evidence bundle, monitor JSON, copyback artifacts, and OMX checkpoint with a fresh active Codex goal snapshot.

## Pod dependency blocker and recovery

The first real extraction shards decoded MCAP successfully but all failed at video frame extraction because the production base image lacked `ffmpeg`:

```text
RuntimeError: ffmpeg is required for real D2E video feature extraction
```

Recovery actions:

- Stopped the failed sharded pipeline (`pid=440`) before merge/finalization.
- Installed `ffmpeg` in the running pod with `apt-get install -y ffmpeg` and verified `ffmpeg version 4.4.2`.
- Hardened `preflight_g003_fdm1_action_dataset_pod.py` to fail before launch when `ffmpeg` is absent from `PATH`.

Verification:

```bash
uv run pytest tests/test_preflight_g003_fdm1_action_dataset_pod.py -q
uv run python -m py_compile scripts/preflight_g003_fdm1_action_dataset_pod.py
```

Observed evidence: `7 passed` locally.
