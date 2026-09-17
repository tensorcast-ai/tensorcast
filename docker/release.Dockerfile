# TensorCast manylinux wheel builder (Stage B – PyPI uploadable)
#
# Base image   : pytorch/manylinux2_28-builder:cuda13.0
# Purpose      : Produce a manylinux_2_28_x86_64 wheel that is eligible
#                for upload to PyPI.
#
# The pytorch/manylinux2_28-builder image already contains:
#   - glibc 2.28 (manylinux_2_28 compliance)
#   - CUDA 13.0 toolkit (nvcc, headers, libraries)
#   - gcc-toolset / devtoolset toolchain
#   - Python interpreters (cp310-cp310, cp311-cp311, cp312-cp312)
#
# We layer on top:
#   - uv (Python package manager)
#   - bazelisk (Bazel launcher)
#   - git (required by setup.py for version generation)
#   - system libraries for daemon and native extension build
#
# ------------------------------------------------------------------------------
# Build
# ------------------------------------------------------------------------------
#   docker build -f docker/release.Dockerfile -t tensorcast-builder:cu130 .
#
# ------------------------------------------------------------------------------
# Run (non-interactive build)
# ------------------------------------------------------------------------------
#   docker run --rm \
#     -v $(pwd):/io \
#     -w /io \
#     -e IN_DOCKER=1 \
#     tensorcast-builder:cu130 \
#     bash docker/build_in_docker.sh --pypi
#
# ------------------------------------------------------------------------------
# Run (interactive shell for debugging)
# ------------------------------------------------------------------------------
#   docker run -it --rm \
#     -v $(pwd):/io \
#     -w /io \
#     -e IN_DOCKER=1 \
#     tensorcast-builder:cu130 \
#     bash
#
# ------------------------------------------------------------------------------

FROM pytorch/manylinux2_28-builder:cuda13.0

LABEL maintainer="TensorCast Team" \
      description="Manylinux builder for TensorCast PyPI wheel"

# ------------------------------------------------------------------------------
# 1. Base system dependencies
# ------------------------------------------------------------------------------
# manylinux_2_28 is AlmaLinux 8 based; yum is available.
# Install git (setup.py calls `git rev-parse`) and standard build helpers.
RUN yum install -y git curl unzip wget

# ------------------------------------------------------------------------------
# 2. uv – Python package manager
# ------------------------------------------------------------------------------
# Installs to /root/.local/bin/uv by default.
# We add it to PATH for all subsequent layers.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# release.sh invokes Python before the project-level `uv sync`. Keep a small,
# image-local bootstrap environment so those commands can import toml, then let
# `uv sync` populate the same environment with torch 2.13.0 + cu130.
ENV UV_PROJECT_ENVIRONMENT="/opt/tensorcast-release-cu130"
RUN uv venv --python /opt/python/cp310-cp310/bin/python "${UV_PROJECT_ENVIRONMENT}" && \
    uv pip install --python "${UV_PROJECT_ENVIRONMENT}/bin/python" "toml>=0.10.2"
ENV VIRTUAL_ENV="${UV_PROJECT_ENVIRONMENT}"
ENV PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"

# ------------------------------------------------------------------------------
# 3. Bazel (via bazelisk)
# ------------------------------------------------------------------------------
# bazelisk handles version pinning and auto-downloads the correct Bazel.
# We install it as `/usr/local/bin/bazel` so `which bazel` works.
RUN curl -fsSL \
    "https://github.com/bazelbuild/bazelisk/releases/download/v1.28.1/bazelisk-linux-amd64" \
    -o /usr/local/bin/bazel && \
    chmod +x /usr/local/bin/bazel

# ------------------------------------------------------------------------------
# 4. System libraries for daemon and native extension build
# ------------------------------------------------------------------------------
# The daemon and native extensions link against system libraries that are
# NOT bundled into the wheel (they belong to the host / driver / fabric
# install). We install them here so the build can link successfully.
#
# Categories:
#   a) Clang / LLVM toolchain (used instead of hermetic LLVM in Docker)
#   b) RDMA verbs (daemon runtime + build)
#   c) Boost (daemon build)
RUN yum install -y clang clang-devel lld lld-devel llvm llvm-devel

RUN yum install -y rdma-core-devel libibverbs-devel

RUN yum install -y boost-devel

# ------------------------------------------------------------------------------
# 5. Create mount point
# ------------------------------------------------------------------------------
# `/io` is the conventional working directory where the host repo is mounted.
RUN mkdir -p /io

WORKDIR /io

# Default to an interactive bash shell (useful for debugging).
CMD ["/bin/bash"]
