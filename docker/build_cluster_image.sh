#!/usr/bin/env bash
set -euo pipefail
IMAGE_TAG="${IMAGE_TAG:-docker.io/pjh6029/fdm-d2e-reproduction:dev}"
BASE_IMAGE="${BASE_IMAGE:-ghcr.io/pjh6029/snupi-prod-base:cu124-20260414}"
PUSH="${PUSH:-0}"

docker build \
  -f docker/Dockerfile \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  -t "${IMAGE_TAG}" \
  .

if [[ "${PUSH}" == "1" ]]; then
  docker push "${IMAGE_TAG}"
fi

printf '{"schema":"cluster_image_build.v1","image":"%s","base_image":"%s","pushed":%s}\n' "${IMAGE_TAG}" "${BASE_IMAGE}" "${PUSH}"
