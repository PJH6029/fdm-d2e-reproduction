# G003 MLXP reservation plan — action-slot materialization

**Status:** draft payload only; no live reservation POST was made in this milestone.

## Why 1 GPU, not 4 GPUs

G003 is CPU/IO-heavy D2E-480p decode/materialization plus action-slot audit work. It does not train IDM/FDM and should not occupy 4 H200s while GPUs are idle. The draft reserves one H200 only because the managed production reservation is the path to the expected `/root/work/code/continuous-gui-poc/fdm-d2e-reproduction` workspace/PVC. Later training/scaling stories must use 4xH200 only when GPU-ready.

## Latest board evidence

- Board inspected: `.omx/tmp/mlxp_board_latest.json`
- Board time: `2026-06-01T01:28:47+09:00`
- Slot size: `60` minutes
- Candidate same-node free window found on node `1`, GPU start `2`.
- Managed/default image: `base` / `ghcr.io/pjh6029/snupi-prod-base:cu124-20260414`

## Exact draft payload

Saved at `artifacts/mlxp/g003_action_dataset_reservation_payload_draft.json`:

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

## Intended pod command after reservation

```bash
cd /root/work/code/continuous-gui-poc/fdm-d2e-reproduction
git fetch origin research/fdm1-d2e-ultragoal
git checkout research/fdm1-d2e-ultragoal
git pull --ff-only
export PATH="$HOME/.local/bin:$PATH"
uv sync --extra d2e --extra train --extra test
bash scripts/run_g003_fdm1_action_dataset_pipeline.sh
```

## Completion gate

G003 should not be checkpointed until `artifacts/sources/fdm1_g003_action_dataset_completion_audit.json` reports `status=pass` and the relevant small evidence artifacts are copied back/committed locally.
