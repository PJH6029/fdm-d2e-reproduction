#!/usr/bin/env bash
set -euo pipefail

REPO_PATH="${REPO_PATH:-/root/work/code/continuous-gui-poc/fdm-d2e-reproduction}"
REMOTE="${REMOTE:-origin}"
BRANCH="${BRANCH:-main}"
SELF_CHECK=0
for arg in "$@"; do
  case "$arg" in
    --self-check) SELF_CHECK=1 ;;
  esac
done

if [[ ! -d "${REPO_PATH}/.git" ]]; then
  echo "missing git checkout at ${REPO_PATH}; clone/pull must happen on the MLXP PVC before bootstrap" >&2
  exit 2
fi

cd "${REPO_PATH}"

git fetch "${REMOTE}" "${BRANCH}"
git checkout "${BRANCH}"
git pull --ff-only "${REMOTE}" "${BRANCH}"

uv sync --frozen --extra d2e --extra test --extra train

if [[ "${SELF_CHECK}" == "1" ]]; then
  uv run python scripts/prepare_d2e_real.py --config configs/data/d2e_real_sample.yaml
  uv run python scripts/run_cluster_smoke_matrix.py --gpu-counts 1 2 4 --repo-path "${REPO_PATH}"
fi
