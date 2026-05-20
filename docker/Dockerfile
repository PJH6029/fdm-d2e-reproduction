ARG BASE_IMAGE=ghcr.io/pjh6029/snupi-prod-base:cu124-20260414
FROM ${BASE_IMAGE}

ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/fdm-d2e-venv \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1

WORKDIR /root/work/code/continuous-gui-poc/fdm-d2e-reproduction

# The MLXP flow normally git-pulls into the PVC path, but copying the project
# also supports reproducible custom-image builds when production base image deps
# are insufficient.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY schemas ./schemas
COPY docs ./docs

RUN if command -v apt-get >/dev/null 2>&1; then apt-get update -y && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*; fi \
    && python -m pip install --no-cache-dir uv || true \
    && uv sync --frozen --extra d2e --extra test --extra train

CMD ["bash", "scripts/cluster_bootstrap.sh", "--self-check"]
