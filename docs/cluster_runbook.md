# MLXP / H200 Cluster Runbook

G2 establishes a reproducible path from this local repo to the MLXP PVC checkout:

`/root/work/code/continuous-gui-poc/fdm-d2e-reproduction`

## Image path

Default production base image, from the G0 MLXP board snapshot:

`ghcr.io/pjh6029/snupi-prod-base:cu124-20260414`

Custom-image build command:

```bash
IMAGE_TAG=docker.io/pjh6029/fdm-d2e-reproduction:dev \
BASE_IMAGE=ghcr.io/pjh6029/snupi-prod-base:cu124-20260414 \
bash docker/build_cluster_image.sh
```

Set `PUSH=1` to push to `docker.io/pjh6029` after a successful local build.

## PVC bootstrap in a reserved pod

```bash
cd /root/work/code/continuous-gui-poc/fdm-d2e-reproduction
bash scripts/cluster_bootstrap.sh --self-check
```

The bootstrap installs `ffmpeg` when missing, then does `git fetch`, `git pull --ff-only`, and `uv sync --frozen --extra d2e --extra test --extra train` before running the real-D2E manifest check and the launcher matrix dry run.

## 1/2/4 GPU launcher smoke matrix

Dry-run command contract, safe locally and in the pod:

```bash
uv run python scripts/run_cluster_smoke_matrix.py \
  --gpu-counts 1 2 4 \
  --repo-path /root/work/code/continuous-gui-poc/fdm-d2e-reproduction
```

Actual execution in the MLXP pod:

```bash
uv run python scripts/run_cluster_smoke_matrix.py \
  --execute \
  --gpu-counts 1 2 4 \
  --repo-path /root/work/code/continuous-gui-poc/fdm-d2e-reproduction
```

Single-GPU launches use `uv run python`; 2/4 GPU launches use `uv run torchrun --standalone --nproc-per-node <N>`. Reports are written under `outputs/cluster/` in the PVC checkout.

No live production reservation is created by these scripts; the MLXP reservation exact-payload confirmation gate still applies before POSTing a reservation.

## Current full-corpus unattended chain

For the active full-corpus run, keep the G003 parent alive and let fail-closed
watchers handle post-run handoff:

Watcher scripts self-write their Python PID to their `--watcher-pid-file`.
Do **not** overwrite those files with shell `$!` from `uv run`; `$!` may be a uv
wrapper PID rather than the long-lived watcher process.

```bash
nohup uv run python scripts/watch_g003_then_finalize.py \
  --output artifacts/idm/g003_postrun_watcher_summary.json \
  > artifacts/idm/g003_postrun_watcher.log 2>&1 &

nohup uv run python scripts/watch_g003_then_launch_g004.py \
  --launch \
  --start-g004-watcher \
  --output artifacts/fdm/g003_to_g004_chain_summary.json \
  > artifacts/fdm/g003_to_g004_chain.log 2>&1 &
```

The chain watcher never checkpoints OMX/Codex state. It launches G004 only after
G003 finalization and the G003 audit pass, then starts the G004 post-run watcher.
If G003 finalization/audit fails, it records a blocker instead of launching FDM
training.

After G004 has launched, use the G004→G005 readiness chain to prepare the aux
handoff without starting an unsafe default aux run:

```bash
nohup uv run python scripts/watch_g004_then_plan_g005.py \
  --source-evidence artifacts/aux/<source>_materialization.json \
  --eval-manifest-hashes artifacts/aux/d2e_eval_manifest_hashes.json \
  --require-eval-manifest-hashes \
  --require-namespace-ready \
  --output artifacts/aux/g004_to_g005_readiness_chain_summary.json \
  > artifacts/aux/g004_to_g005_readiness_chain.log 2>&1 &
```

It waits for G004 finalization/audit pass, then records whether G005 is
launch-ready. It does not launch G005 training or checkpoint any story.
