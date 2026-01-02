<div align="center">
<h1>TensorCast</h1>
<p>The shared tensor layer - load once, share everywhere.</p>
</div>

TensorCast is a high-performance distributed artifact storage system. Load model
weights and KV cache once, then share them across processes and services.

## Docs

- [Developer Guides](docs/README.md) - architecture map and developer docs
- [Architecture Overview](docs/architecture/architecture-overview.md) - system overview
- [API Architecture](docs/architecture/api/README.md) - SDK surface and flows
- [Store Daemon Deployment](docs/deployment/store-daemon.md) - daemon deployment and config
- [Global Store Deployment](docs/deployment/global-store-deployment.md) - Global Store deployment
- [Testing Guide](docs/development/testing.md) - Python, C++, P2P, and RDMA tests
- [Python Tests](tests/python/README.md) - test layout and commands
- [Repo Automation Rules](AGENTS.md) - automation and sandbox rules

## Quickstart (dev)

### Prereqs (Linux)

- Linux 5.10+
- uv + pre-commit
- Bazel
- gcc-13/g++-13, libstdc++-12-dev, libxml2

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
sudo apt install -y software-properties-common libxml2 libstdc++-12-dev gcc-13 g++-13
```

If Bazel fails to download LLVM, run `bash tools/download_and_set_local_llvm.sh`.

### Build

```bash
uv venv
uv sync --all-extras --all-groups --verbose

# Build C++ core + Python extension
BUILD_CORE=1 BUILD_EXTENSION=1 uv run -vvv setup.py build_ext
```

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
`docs/architecture/api/README.md` and `tensorcast/api/README.md`.

### Tests

```bash
# Python
uv run pytest tests/python/

# C++
bazel test //core/...
```

For P2P/RDMA and communicator coverage, see `docs/development/testing.md`.

For CPU-only development, set `TENSORCAST_CUDA_BACKEND=fake` (test-only) and
use `--test_env=TENSORCAST_CUDA_BACKEND=fake` for Bazel tests. See `AGENTS.md`.
