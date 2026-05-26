---
title: Build from Source
description: Build TensorCast from a local checkout and troubleshoot common source-build issues
sidebar_position: 1
---

# Build from Source

Use this path when you need a different torch version, run inside an
unsupported distro, or develop against a checkout.

## Prerequisites

- Linux 5.10+, glibc 2.28+
- `uv` and `pre-commit`
- Bazel, installed via `tools/install-bazel.sh`
- `gcc-13`/`g++-13`, `libstdc++-12-dev`, `libxml2`
- Release toolchain packages (`wheel`, `auditwheel`, `patchelf`, `twine`) from
  the `release` dependency group when building release artifacts

```bash
# uv + pre-commit
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install pre-commit --with pre-commit-uv
pre-commit install

# Bazel
./tools/install-bazel.sh

# Ubuntu toolchain deps
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install -y software-properties-common libxml2 libstdc++-12-dev \
    gcc-13 g++-13 python3
```

## Build

Generate protocol buffer stubs, create the virtualenv, sync dependencies, and
build the C++ core plus Python extension:

```bash
# Generate Python proto stubs and C++ headers
bash tools/build_proto_python.sh

# Set up and activate the virtualenv
uv venv
source .venv/bin/activate
uv sync --all-extras --all-groups --verbose

# Keep MODULE.bazel http_archives aligned with uv.lock
uv run python tools/update_module_http_archives.py --lockfile uv.lock --module MODULE.bazel

# Build C++ core + Python extension (default: torch 2.11.0 + cu128)
BUILD_CORE=1 BUILD_EXTENSION=1 uv run -vvv setup.py build_ext
```

To build against a different torch or CUDA version, use the release driver. It
patches `pyproject.toml` and `MODULE.bazel` in place for the duration of the
build:

```bash
tools/release.sh build --torch-version 2.13.0 --cuda-version cu128
uv pip install dist/tensorcast-*.whl
```

The import-time ABI guard reads the torch version baked into the wheel at build
time, so source edits are not needed when bumping torch versions.

## Run Local Services

After building, run a local Global Store and Store Daemon:

```bash
# Global Store
uv run tensorcast-cli global start --config=examples/config/global_store_config.yaml

# Store Daemon (connect to Global Store)
uv run tensorcast-cli daemon start \
  --config=examples/config/store_daemon_config.yaml \
  --global-store-mode connect \
  --global-store-address 127.0.0.1:50051

# Status / logs / stop
uv run tensorcast-cli daemon status
uv run tensorcast-cli daemon logs -f
uv run tensorcast-cli daemon stop
```

Unified config only: the daemon accepts `--config`, and runtime parameters live
in YAML/JSON. Example files live in `examples/config/`.

## Troubleshoot

If Bazel fails to download LLVM, run:

```bash
bash tools/download_and_set_local_llvm.sh
```

The helper only downloads and verifies the public LLVM release tarball; Bazel
continues to resolve LLVM from the public URLs in `MODULE.bazel`.

If Bazel hits missing header errors like
`fatal error: absl/log/log.h: No such file or directory`, the repo-root
`external/` or `bazel-bin` symlink is likely stale. Fix it with:

```bash
rm -f external && ln -s $(bazel info output_base)/external external
rm -f bazel-bin && ln -s $(bazel info bazel-bin) bazel-bin
```

If importing `tensorcast._C` fails with
`cannot allocate memory in static TLS block`, rebuild on the latest `main`.
TensorCast disables jemalloc initial-exec TLS to make `dlopen()` safe. As a
temporary workaround for older builds, run with:

```bash
GLIBC_TUNABLES=glibc.rtld.optional_static_tls=32768
```

## Tests

For the full test matrix, see the [Testing Guide](testing.md). The common local
entry points are:

```bash
# Python
source .venv/bin/activate
pytest tests/python/

# C++
bazel test //core/...
```

For CPU-only development, set `TENSORCAST_CUDA_BACKEND=fake` in recognized test
environments and pass `--test_env=TENSORCAST_CUDA_BACKEND=fake` to Bazel tests.
