#!/usr/bin/env bash
# TensorCast Docker Build Wrapper
# --------------------------------
# This script is meant to be called INSIDE the Docker container.
# It temporarily swaps MODULE.bazel and .bazelrc to their Docker-specific
# variants, runs the release build, and restores the originals afterwards.
#
# Usage (inside container):
#   bash docker/build_in_docker.sh [args...]
#
# Typical args are forwarded to tools/release.sh, e.g.:
#   bash docker/build_in_docker.sh --pypi --skip-uv-sync
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ---------------------------------------------------------------------------
# 1. Verify Docker-specific configs exist
# ---------------------------------------------------------------------------
if [[ ! -f "${PROJECT_ROOT}/docker/MODULE.docker.bazel" ]]; then
    echo "error: docker/MODULE.docker.bazel not found. Make sure it exists before running Docker builds." >&2
    exit 1
fi

if [[ ! -f "${PROJECT_ROOT}/docker/.bazelrc.docker" ]]; then
    echo "error: docker/.bazelrc.docker not found. Make sure it exists before running Docker builds." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Backup originals and swap in Docker configs
# ---------------------------------------------------------------------------
echo "==> [docker build] Backing up host configs..."
cp "${PROJECT_ROOT}/MODULE.bazel" "${PROJECT_ROOT}/MODULE.bazel.host-backup"
cp "${PROJECT_ROOT}/.bazelrc" "${PROJECT_ROOT}/.bazelrc.host-backup"

echo "==> [docker build] Applying Docker configs..."
cp "${PROJECT_ROOT}/docker/MODULE.docker.bazel" "${PROJECT_ROOT}/MODULE.bazel"
cp "${PROJECT_ROOT}/docker/.bazelrc.docker" "${PROJECT_ROOT}/.bazelrc"

# ---------------------------------------------------------------------------
# 3. Run the build (via release.sh)
# ---------------------------------------------------------------------------
BUILD_OK=1
echo "==> [docker build] Running: bash tools/release.sh build $@"
if bash "${PROJECT_ROOT}/tools/release.sh" build "$@"; then
    BUILD_OK=0
else
    BUILD_OK=$?
    echo "error: Build failed with exit code ${BUILD_OK}" >&2
fi

# ---------------------------------------------------------------------------
# 4. Restore originals regardless of build outcome
# ---------------------------------------------------------------------------
echo "==> [docker build] Restoring host configs..."
if [[ -f "${PROJECT_ROOT}/MODULE.bazel.host-backup" ]]; then
    mv "${PROJECT_ROOT}/MODULE.bazel.host-backup" "${PROJECT_ROOT}/MODULE.bazel"
fi
if [[ -f "${PROJECT_ROOT}/.bazelrc.host-backup" ]]; then
    mv "${PROJECT_ROOT}/.bazelrc.host-backup" "${PROJECT_ROOT}/.bazelrc"
fi

# ---------------------------------------------------------------------------
# 5. Exit with build status
# ---------------------------------------------------------------------------
if [[ ${BUILD_OK} -eq 0 ]]; then
    echo "==> [docker build] SUCCESS"
    exit 0
else
    echo "==> [docker build] FAILED (exit ${BUILD_OK})" >&2
    exit "${BUILD_OK}"
fi
