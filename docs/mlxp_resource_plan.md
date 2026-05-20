# MLXP Resource Plan — G0 FDM-D2E Reproduction

- Generated: 2026-05-20
- Skill contract: `mlxp-reservation-api`
- Safe board snapshot: `artifacts/mlxp/production_board_snapshot.safe.json`
- Safe user/quota summary: `artifacts/mlxp/me_summary.safe.json`
- Draft reservation payload: `artifacts/mlxp/production_reservation_payload_draft.json`

## Safety status

No live production reservation was created during G0. The API was used only for authenticated read-only inspection (`/api/auth/me`, `/api/me`, and `/api/projects/production/board`). Per the MLXP skill contract, any future live production reservation POST must show the exact JSON payload and receive explicit user confirmation first.

Token handling: `mlxp.md` was read locally to authenticate API calls, but token values were not copied into docs or safe artifacts.

## API/resource facts from board inspection

- Base URL used successfully: `http://147.46.219.248:8000`
- Project: `production`
- Namespace: `p-production`
- Board slot size: 60 minutes (`1-hour`)
- Production nodes: 6 nodes, each with 8 H200 GPUs (48 GPUs total board capacity)
- Default managed image key: `base`
- Default managed image: `ghcr.io/pjh6029/snupi-prod-base:cu124-20260414`
- Additional managed image: `codex` → `ghcr.io/pjh6029/snupi-personal-codex:20260414`
- Registry profile exposed by board: `snupi-lab-registry` with prefix `snupi-nas2.synology.me:55031/`
- User-facing custom image note from `mlxp.md`: Docker Hub `docker.io`, username `pjh6029`; use only via `image_path` if/when custom image is required.
- Auth/quota summary: token auth works; cluster account active; kubeconfig ready; debug group `b`; effective production quota currently reports 8 GPUs with 400 approved GPU-hours remaining.

## Recommended image strategy

1. **G0/G1/G2 bootstrap:** start with managed production base image (`managed_image_key: "base"`) because it is already board-supported and CUDA 12.4 aligned.
2. **Codex-heavy in-pod editing:** use managed `codex` image if interactive code editing inside the pod becomes materially useful.
3. **Custom training image:** build/push only if dependency installation in production base is too slow or fragile. If using Docker Hub as user requested, payload must use `image_path` and not `managed_image_key`; do not set `registry_profile_key` unless the image path uses a board-provided registry profile.

## Recommended reservation cadence

- Debug/local first: implement G1/G2 contract tests locally and/or on debug resources when enough.
- First production bootstrap: 4x H200, 2-4 hours, validate environment, dataset sample download/decode, tiny train step, and distributed launcher.
- Main IDM/FDM runs: reserve 4x H200 in longer windows after G1-G3 are stable and endpoint config is frozen.
- Checkpoint policy: every production run must write command transcript, git commit, Docker/image identity, dataset fingerprint, GPU count, wall-clock, logs, and metrics.

## Candidate 4-GPU windows

The safe board snapshot includes same-node 4-GPU candidate windows. The current best long window at inspection time was node 5, GPUs 0-3, from `2026-05-21T01:00:00+09:00` to `2026-05-23T17:00:00+09:00` (64 board slots). Availability can change; re-inspect board immediately before any actual reservation.

## Draft payload for future confirmation

The draft below is saved at `artifacts/mlxp/production_reservation_payload_draft.json`. It is **not posted** and must be revalidated against the board immediately before use.

```json
{
  "node_id": "5",
  "gpu_start": 0,
  "gpu_count": 4,
  "gpu_indices": [],
  "start_at": "2026-05-21T01:00:00+09:00",
  "end_at": "2026-05-23T17:00:00+09:00",
  "purpose": "Continuous GUI - FDM reproduction: G0/G1 source validation and real-D2E training stack bootstrap",
  "managed_image_key": "base",
  "registry_profile_key": "",
  "image_path": "",
  "command": [],
  "args": [],
  "actor_name": "jeonghunpark"
}
```

Before posting, show the exact payload again and get explicit confirmation. If using a custom image instead of managed base, replace `managed_image_key` with `image_path` and keep `managed_image_key` empty.

## Cluster workflow reminder

1. Commit/push local repo changes.
2. In reservation pod, work under `/root/work/code/continuous-gui-poc/fdm-d2e-reproduction`.
3. Pull latest branch there.
4. Run setup scripts from G2.
5. Store persistent data/checkpoints/logs under the PVC-backed workspace or approved DDN path; use NVMe only for volatile cache/scratch.

## G0 handoff to G1/G2

- G1 can begin by implementing real D2E source inventory, HF file listing, sample download, MCAP/video decode interfaces, and schema v2 contracts.
- G2 must pin/install the missing Python training/data dependencies and validate the production-base image path.
- No long production training reservation should be made until G1 sample decode and G3 endpoint config are stable.
