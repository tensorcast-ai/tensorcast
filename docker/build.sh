#!/bin/bash

# Get the short commit hash (first 5 characters)
COMMIT_HASH=$(git rev-parse --short=5 HEAD)
DATE=$(date +%Y.%m.%d)

# Construct the tag
TAG="hub.i.basemind.com/tensorcast/scstore:${DATE}-${COMMIT_HASH}"

# Prepare proxy arguments
PROXY_ARGS=""
if [ -n "$http_proxy" ]; then
  PROXY_ARGS="$PROXY_ARGS --build-arg http_proxy=$http_proxy"
fi
if [ -n "$https_proxy" ]; then
  PROXY_ARGS="$PROXY_ARGS --build-arg https_proxy=$https_proxy"
fi
if [ -n "$no_proxy" ]; then
  PROXY_ARGS="$PROXY_ARGS --build-arg no_proxy=$no_proxy"
fi

# Build the Docker image
sudo docker build $PROXY_ARGS -f ./docker/Dockerfile -t $TAG .

docker push $TAG