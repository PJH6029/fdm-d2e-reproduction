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
