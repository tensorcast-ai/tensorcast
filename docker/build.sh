#!/usr/bin/env bash

set -euo pipefail

# Get the short commit hash (first 5 characters)
COMMIT_HASH=$(git rev-parse --short=5 HEAD)
DATE=$(date +%Y.%m.%d)

# Construct the tag
IMAGE_REPOSITORY="${TENSORCAST_IMAGE_REPOSITORY:-ghcr.io/tensorcast-ai/tensorcast}"
TAG="${IMAGE_REPOSITORY}:${DATE}-${COMMIT_HASH}"

# Prepare proxy arguments
PROXY_ARGS=""
if [ -n "${HTTP_PROXY:-${http_proxy:-}}" ]; then
  PROXY_ARGS="$PROXY_ARGS --build-arg HTTP_PROXY=${HTTP_PROXY:-${http_proxy:-}}"
fi
if [ -n "${HTTPS_PROXY:-${https_proxy:-}}" ]; then
  PROXY_ARGS="$PROXY_ARGS --build-arg HTTPS_PROXY=${HTTPS_PROXY:-${https_proxy:-}}"
fi
if [ -n "${NO_PROXY:-${no_proxy:-}}" ]; then
  PROXY_ARGS="$PROXY_ARGS --build-arg NO_PROXY=${NO_PROXY:-${no_proxy:-}}"
fi

# Build the Docker image
docker build $PROXY_ARGS --build-arg TENSORCAST_BUILD_COMMIT="$COMMIT_HASH" -f ./docker/Dockerfile -t "$TAG" .

docker push "$TAG"
