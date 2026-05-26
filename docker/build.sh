#!/usr/bin/env bash

set -euo pipefail

# Get the short commit hash (first 5 characters)
COMMIT_HASH=$(git rev-parse --short=5 HEAD)
DATE=$(date +%Y.%m.%d)

# Construct the tag
IMAGE_REPOSITORY="${TENSORCAST_IMAGE_REPOSITORY:-ghcr.io/tensorcast-ai/tensorcast}"
TAG="${IMAGE_REPOSITORY}:${DATE}-${COMMIT_HASH}"

# Build the Docker image
docker build --build-arg TENSORCAST_BUILD_COMMIT="$COMMIT_HASH" -f ./docker/Dockerfile -t "$TAG" .

docker push "$TAG"
