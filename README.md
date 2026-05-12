<div align="center">
<h1>TensorCast</h1>
<p>The shared tensor layer — load once, share everywhere.</p>
</div>

TensorCast is a high-performance distributed artifact storage system. Load
model weights and KV cache once, then share them across processes and
services.

## Install

```bash
pip install tensorcast
```

`pip install tensorcast` pulls in the matching `torch==2.11.0` (CUDA 12.8)
build automatically. If your environment already has a different torch
version, build from source instead — see [Build from source](#build-from-source).

### Compatibility matrix (v0.1.0)

| Axis | Supported |
|---|---|
| Python | 3.10 / 3.11 / 3.12 |
| OS | Linux only, kernel ≥ 5.10 |
| glibc | ≥ 2.28 (RHEL 8, Ubuntu 20.04+, Debian 10+) |
| torch | 2.11.0 (exact pin; ABI-checked at import) |
| CUDA | 12.8 driver + runtime |

The wheel is `manylinux_2_28_x86_64`. The native extension and daemon link
against the cxx11 ABI of PyTorch (`_GLIBCXX_USE_CXX11_ABI=1`), which matches
the official PyTorch wheels on PyPI.

### Python quickstart

```python
import tensorcast as tc
import torch

tc.init(mode="connect")

state_dict = {"layer.weight": torch.randn(8, 8, device="cuda")}
tc.register(state_dict, key="demo:model:001")

handle = tc.artifact(key="demo:model:001")
tensors = handle.tensor_dict(device="cuda:0")
```

For advanced flows (async, views, prefetch, policies), see
[API Architecture](docs/architecture/api/README.md) and
[SDK Startup User Guide](docs/guides/sdk-startup-user-guide.md).

## Docs

- [Developer Guides](docs/README.md) — architecture map and developer docs
- [Architecture Overview](docs/architecture/architecture-overview.md) — system overview
- [API Architecture](docs/architecture/api/README.md) — SDK surface and flows
- [SDK Startup User Guide](docs/guides/sdk-startup-user-guide.md) — `tensorcast.init`, API/SDK startup, TP multi-process usage
- [Store Daemon Deployment](docs/deployment/store-daemon.md) — daemon deployment and config
- [Global Store Deployment](docs/deployment/global-store-deployment.md) — Global Store deployment
- [Testing Guide](docs/development/testing.md) — Python, C++, P2P, and RDMA tests
- [Python Tests](tests/python/README.md) — test layout and commands
- [Release Guide](RELEASE.md) — how releases are cut
- [Repo Automation Rules](AGENTS.md) — automation and sandbox rules

## Build from source

Use this path when you need a different torch version, run inside an
unsupported distro, or want to develop against a checkout.

### Prereqs (Linux)

- Linux 5.10+, glibc 2.28+
- `uv` + `pre-commit`
- Bazel (via `tools/install-bazel.sh`)
- `gcc-13`/`g++-13`, `libstdc++-12-dev`, `libxml2`
- `patchelf` (system package) — required for wheel post-processing

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
    gcc-13 g++-13 python3 patchelf
```

### Build

```bash
# Generate Python proto stubs and C++ headers
bash tools/build_proto_python.sh

# Set up the virtualenv. `uv.lock` is generated on first sync — it is
# .gitignored, so each developer maintains their own.
uv venv
uv sync --all-extras --all-groups --verbose

# Keep MODULE.bazel http_archives aligned with uv.lock
uv run python tools/update_module_http_archives.py --lockfile uv.lock --module MODULE.bazel

# Build C++ core + Python extension (default: torch 2.11.0 + cu128)
BUILD_CORE=1 BUILD_EXTENSION=1 uv run -vvv setup.py build_ext
```

To build against a different torch version, use the release driver — it
patches `pyproject.toml` and `MODULE.bazel` in place for the duration of
the build:

```bash
tools/release.sh build --torch-version 2.13.0 --cuda-version cu128
pip install dist/tensorcast-*.whl
```

The fail-fast ABI guard reads the torch version baked into the wheel at
build time, so source edits are not needed when bumping torch versions.

#### Troubleshoot

- If Bazel fails to download LLVM, run `bash tools/download_and_set_local_llvm.sh`.

- If Bazel hits missing header errors like `fatal error: absl/log/log.h: No such file or directory`, the repo-root `external/` or `bazel-bin` symlink is likely stale. Fix it with:
  ```bash
  rm -f external && ln -s $(bazel info output_base)/external external
  rm -f bazel-bin && ln -s $(bazel info bazel-bin) bazel-bin
  ```

- If importing `tensorcast._C` fails with `cannot allocate memory in static TLS block`, rebuild on the latest `main` (TensorCast disables jemalloc initial-exec TLS to make `dlopen()` safe). As a temporary workaround for older builds, run with `GLIBC_TUNABLES=glibc.rtld.optional_static_tls=32768`.

### Run services (local)

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

Unified config only: the daemon accepts `--config` and all runtime parameters
live in YAML/JSON. Example files live in `examples/config/`.

### Tests

```bash
# Python
uv run pytest tests/python/

# C++
bazel test //core/...
```

For P2P/RDMA and communicator coverage, see [Testing Guide](docs/development/testing.md).

For CPU-only development, set `TENSORCAST_CUDA_BACKEND=fake` (test-only) and
use `--test_env=TENSORCAST_CUDA_BACKEND=fake` for Bazel tests. See [AGENTS.md](AGENTS.md).
