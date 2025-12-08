<div align="center">

TensorCast
===========================
<h4> The shared tensor layer — load once, share everywhere. </h4>
</div>


TensorCast provides the shared tensor layer. Load weights and KV cache once, make them available everywhere, and let services attach on demand. Startup becomes fast, scaling becomes smooth, and infrastructure stays efficient.

## Introduction

[AGENTS.md](AGENTS.md)


## Prerequisites

```bash
# Install uv & pre-commit
curl -LsSf https://astral.sh/uv/install.sh | sh

uv tool install pre-commit --with pre-commit-uv

# Install bazel
./tools/install-bazel.sh

# Dependency that need for compile
sudo apt-get install -y libxml2 libstdc++-12-dev

pre-commit install
```

### LLVM download fallback (Bazel init)
If Bazel fails to download LLVM during the initialization stage, you can pre-download it locally and update `MODULE.bazel` automatically:

```bash
bash tools/download_and_set_local_llvm.sh
```

Then re-run your Bazel command.

### Python Environment

```bash

# Create virtual environment
uv venv

uv sync --all-extras --all-groups --verbose

# Generate proto Python code
./tools/build_proto_python.sh

# Install dependencies
#   - BUILD_CORE means cxx files in core/ and daemon/
#   - BUILD_EXTENSION means cxx files in tensorcast/csrc
BUILD_CORE=1 BUILD_EXTENSION=1 uv run -vvv setup.py build_ext

# Run tests
uv run pytest tests/python/**.py

# Run cli target
uv run tensorcast-cli --help

# Run specific file
uv run xxx/xxx.py
```

### StoreDaemon (C++)

The StoreDaemon service is implemented in C++ and launched by the Python CLI.

- Development (from source):
  - Build once with Bazel: `bazel build //daemon:tensorcast_daemon`
  - Start Global Store (single-instance): `uv run -q python -m tensorcast.cli global start --listen-port 0 --wait`
    - Status: `uv run -q python -m tensorcast.cli global status --json`
    - Logs: `uv run -q python -m tensorcast.cli global logs -f`
  - Start daemon (binds to existing GS or auto-starts if needed): `uv run -q python -m tensorcast.cli daemon start --global-store-mode auto`
    - By default non-blocking start waits until the daemon is ready (timeout 30s). Use `--no-wait` to return immediately.
    - Customize readiness timeout: `--timeout 30`
  - Status: `uv run -q python -m tensorcast.cli daemon status --json`
  - Logs (follow): `uv run -q python -m tensorcast.cli daemon logs -f`
  - Stop (owner cascades GS if it was started by daemon start): `uv run -q python -m tensorcast.cli daemon stop`
  - The CLI automatically locates the binary from `bazel-bin/daemon/tensorcast_daemon` or the packaged wheel.

- Packaged (wheel) usage:
  - The wheel packages the daemon at `tensorcast/bin/tensorcast_daemon` and the CLI will use it automatically.

Runtime flags have been removed. The daemon accepts exactly one flag: `--config=/path/to/file.{yaml,json}`. All runtime parameters are configured via this unified config.

See `examples/config/store_daemon_config.yaml` for a complete example, including the `communicator.*` section.
network:
  p2p_port: 9090
communicator:
  enable_rdma: false
  stager:
    stage_cpu_for_rdma: true
    stage_chunk_mb_cpu: 4
    stage_chunk_mb_gpu: 16
    buffers_per_flow: 4
  rdma:
    outstanding_wr: 64
    ack_ttl_ms: 30000
    traffic_class: 186
    qp_timeout: 20
    qp_retry: 7
  pool:
    preregister_mr: true
    pool_size_bytes: 8589934592
    chunk_bytes: 67108864
  transport:
    tcp_conn_count: 8
    connect_timeout_sec: 10
    tcp_tos: 0
```

Pass this file via `--config` when starting the daemon. When `communicator` is present,
the daemon initializes the communication engine using these typed settings (no environment variables).


```bash
# Create a symlink to external packages for header files
# Run this on the root directory of the project
ln -s $(bazel info output_base)/external external
```

## Client Init (Launch/Connect)

Use the unified, Linux-only launch/connect API from Python:

```python
from tensorcast import init

# Connect to current runtime (prefers existing daemon + Global Store discovered from runtime state)
ctx = init()

# Force local launch (private session). If daemon_config_path is omitted,
# init() will try $TENSORCAST_DAEMON_CONFIG, ~/.tensorcast/config/daemon.yaml,
# and finally an embedded loopback config with ephemeral ports and a writable cache
# under ~/.tensorcast (or a temp dir).
# Private launches do not publish ~/.tensorcast/current_session and bind to 127.0.0.1.
ctx = init(
    daemon_config_path="~/.tensorcast/config/daemon.yaml",
    global_store_mode="auto",  # or "connect"/"start"/"none"
    session_id="tc-private-demo",
)

# Connect to a specific daemon
ctx = init(address="127.0.0.1:50052", global_store_mode="connect")

# Context is process-scoped and is automatically closed at process exit (atexit).
# Call tensorcast.shutdown() to close early if needed.
```

Call `tensorcast.init(...)` once per process before interacting with the daemon. The user-facing
API is intentionally lightweight—operate through the top-level helpers and let TensorCast manage
the session state for you:

```python
import tensorcast as tc
import torch

tc.init()

state_dict = {"layer.weight": torch.randn(8, 8, device="cuda")}

registered = tc.register(state_dict, key="demo:model:001")
handle = tc.artifact(key="demo:model:001")
latest = handle.tensor_dict(device="cuda:0")

buffers = {"layer.weight": torch.empty_like(state_dict["layer.weight"])}
handle.tensor_dict_into(buffers, device="cuda:0")

# Stream a single tensor directly into an existing buffer without populating a
# full mapping. TensorCast copies only the requested tensor.
target = torch.empty_like(state_dict["layer.weight"])
handle.tensor_into("layer.weight", target, device="cuda:0")

# Issue background materialization without mutating the original handle.
prefetched, ticket = handle.prefetch(device="cuda:0")
ticket.wait(timeout=5.0)
prefetched.tensor_dict(device="cuda:0")
```

For advanced scenarios (async verbs, fine-grained inspection, or direct access to diagnostics) use
`tc.store()` to obtain the underlying `Store` session object. That object exposes the complete
surface described in [docs/designs/0014-store-session-api-modernization.md](docs/designs/0014-store-session-api-modernization.md),
including async futures, retry telemetry, and session metadata.
For test doubles or advanced embeddings, you can inject custom registration/materialization
handlers via `Store(..., register_fn=..., materialize_fn=...)` or swap them on an existing
session with `set_register_fn`/`set_materialize_fn`—no global monkeypatching required.

Notes on signals and cleanup:
- The SDK does not override your process SIGINT/SIGTERM by default. Child processes are still cleaned up reliably via Linux PDEATHSIG when the parent really exits.
- To opt-in to SDK-installed signal handling (e.g., in standalone scripts), pass `install_signal_handlers=True`:
  - `init(..., install_signal_handlers=True)` installs graceful handlers that stop the owned daemon and then exit.
  - `init(..., install_signal_handlers=True, fate_share_sigterm=True)` installs hard-exit handlers that immediately terminate the process to trigger PDEATHSIG (use sparingly).

CLI-launched daemon sessions are recorded under `~/.tensorcast/sessions/<session_id>` with `session/`, `logs/`, `pids.json`, and `meta.json`. Global Store sessions live under `~/.tensorcast/global_sessions/<gid>`. The authoritative pointer for discovery is `~/.tensorcast/runtime/state.json` (daemon + Global Store + cluster_token); `~/.tensorcast/current_session` and `~/.tensorcast/current_global_session` are convenience pointers used by CLI defaults. When the config specifies `listen.port: 0` (or is omitted), the CLI pre‑assigns a free TCP port and writes an effective config under the session directory; `meta.json` contains the final `address` used by clients. The same applies to `server.p2p_listen.port`.

SDK launches via `init(..., session_id="tc-private-<uuid>")` are private: they do not update `current_session` and bind to loopback only, so other processes and tools cannot auto‑discover them. The returned `Context` manages the lifecycle of that private session.

CLI duplicate protection: `tensorcast daemon start` refuses to start a new local daemon if a current session is already healthy. Use `tensorcast daemon restart` to stop and start, or `tensorcast daemon stop` first. If the daemon start also launched the Global Store, `tensorcast daemon stop` will cascade and stop it.

## Advanced SDK: RegisteredArtifact with Context Manager

For advanced in-memory registration workflows (explicit CPU VA feed via UMA/VS, TTL keepalive,
manual revoke/abort), use the `Store.register_async` path or the low-level SDK handle API with a
Python context manager. When `ttl_ms` is provided, the Store’s lease manager refreshes TTL every
TTL/2 until commit/close.

Note: SDK examples have been aligned to UMA V3 final naming; CPU streaming now uses UMA-owned direct-write grants and ledger updates. Refer to `tensorcast.api` helpers for current usage.

Notes:
- For VRAM lease (FDML), begin with `LeasePlan(kind="lease", ...)` and feed `LeaseSegment` items using IPC handles exported from unique CUDA storage blocks. Each `LeaseSegment` includes `dst_offset` so segment order is irrelevant; the daemon zero-fills PAD and places bytes at the specified destination offsets.
- Coalesced VRAM remains the simplest one-shot path; `Store.put(...)` performs the copy + commit and surfaces the resulting `RegisteredArtifact`.
- Without GPUs, build and run with the Fake CUDA backend (see AGENTS.md). Fake CUDA now simulates cross-process CUDA IPC by backing handles with shared memory so daemon↔client materialization paths (e.g., shared storage round-trips) work in CI without GPUs.

## Run test

### C++ Tests
```bash
# Run all tests except stress, rdma, and multi_gpu tests
# (stress tests are slow, rdma and multi_gpu tests need to be run manually with specific hardware)
bazel test //core/... --test_tag_filters="-stress,-rdma,-multi_gpu"

# Run only stress tests (stress tests are slow)
bazel test //core/... --test_tag_filters="+stress"

# Run only rdma tests
bazel test //core/... --test_tag_filters="+rdma"

# Run only multi_gpu tests
bazel test //core/... --test_tag_filters="+multi_gpu"

# By default, C++ tests run with the Fake CUDA backend so they pass on CPU-only machines.
# You can override explicitly with a single define:
#   - Fake CUDA:  bazel test --define=use_fake_cuda=true  //daemon:grpc_service_impl_registration_test
#   - Real CUDA:  bazel test --define=use_fake_cuda=false //daemon:grpc_service_impl_registration_test
```

### Communicator P2P Test
```bash
# Terminal 1 (Server)
$ ./bazel-bin/tests/cpp/gpu_ce_test -a server -i <SERVER_IP> -p 19099 -c 743838020 -k 4 -g 1 -r 0
# Terminal 2 (Client)
$ ./bazel-bin/tests/cpp/gpu_ce_test -a client -i <SERVER_IP> -p 19099 -c 743838020 -k 4 -g 1 -r 0
```

### Artifact Test (Memory Registration)
```bash
# Artifact Communication Memory Registration Test (Single Process)
# Tests loading from Disk to CPU/GPU and registering for communication.
# Requires CUDA for GPU section and P2P capable environment for full registration success.
$ ./bazel-bin/tests/cpp/replica_p2p_registration_test
```

### Artifact Test (P2P Transfer C/S)
```bash
# Artifact P2P Transfer Test (Server/Client)
# Tests loading a artifact via P2P between two processes (server and client).
# Requires CUDA and P2P capable environment.
#
# Key Flags:
#   --mode: 'server' or 'client'
#   --server_ip: IP address the server listens on (server) or connects to (client)
#   --server_port: Port for communication (default: 50061)
#   --artifact_id: Unique identifier for the artifact (default: p2p_transfer_artifact)
#   --gpu_id: GPU device ID to use (default: 0)
#   --artifact_size_mb: Size of the dummy artifact in MB (default: 16)
#   --register_location: Where the server registers memory ('cpu' or 'gpu', default: 'gpu')
#   --server_register_location: Where the client expects server memory ('cpu' or 'gpu', must match server)
#   --client_target_location: Where the client loads the data ('cpu' or 'gpu')
#   --allocation_mode: Client GPU memory allocation ('pool' or 'borrow', default: 'pool')
#
# These examples assume server and client are on the same machine (use 127.0.0.1 for --server_ip on client).
# Replace 127.0.0.1 with the actual server IP if running distributed.

# Scenario 1: Server registers GPU, Client targets GPU (GPU <-> GPU)
# Terminal 1 (Server)
# Server loads artifact to GPU and registers GPU memory.
$ ./bazel-bin/tests/cpp/replica_p2p_transfer_test --mode=server --register_location=gpu --gpu_id=0
# Terminal 2 (Client)
# Client connects to server, expects GPU memory, loads directly to its GPU using internal pool.
$ ./bazel-bin/tests/cpp/replica_p2p_transfer_test --mode=client --server_ip=127.0.0.1 --server_register_location=gpu --client_target_location=gpu --gpu_id=0

# Scenario 2: Server registers CPU, Client targets CPU (CPU <-> CPU)
# Terminal 1 (Server)
# Server loads artifact to CPU (pinned memory) and registers CPU memory chunks.
$ ./bazel-bin/tests/cpp/replica_p2p_transfer_test --mode=server --register_location=cpu
# Terminal 2 (Client)
# Client connects to server, expects CPU memory, loads directly to its CPU (pinned memory).
$ ./bazel-bin/tests/cpp/replica_p2p_transfer_test --mode=client --server_ip=127.0.0.1 --server_register_location=cpu --client_target_location=cpu

# Unsupported Scenarios (Checked within the test):
# - Server GPU -> Client CPU: The test currently prevents this configuration.
# - Server CPU -> Client GPU: The test currently prevents this configuration.

```

### Python
```bash
## For meta service (global store)
uv run pytest tests/python/test_transport.py
uv run pytest tests/python/test_global_store.py
```

### Unit Tests

Build unit tests:
```bash
bazel build //tests/cpp/unit:all
```

Note that currently, P2P unit tests fail when being called by bazel test, and need to run with shell

```bash
./bazel-bin/tests/cpp/unit/comm_engine_tcp_test

# need a rdma-capable server
./bazel-bin/tests/cpp/unit/comm_engine_rdma_test
```


## Run the Global Store Server
```bash
# Start the Global Store service (single-instance). Default port: 50051, workers: 10.
uv run -q python -m tensorcast.cli global start --config=examples/config/global_store_config.yaml --wait

# Direct module entry remains available for debugging if needed:
# uv run -m tensorcast.global_store --config=examples/config/global_store_config.yaml

# Run the Global Store Server in Docker
sudo docker run -d --name global-store -p 50051:50051 hub.i.basemind.com/tensorcast/global-store:2025.04.27-55f24
```

### Observability (OpenTelemetry)

Enable by configuration file (no environment variables). Set `observability.otel.*` in the Daemon/Global Store config file.

Example (Global Store):

```yaml
observability:
  otel:
    enabled: true
    exporter_protocol: grpc
    exporter_otlp_endpoint: http://127.0.0.1:4317
    service_name: tensorcast-global-store
  logging:
    level: INFO
```

- Enum fields accept friendly values: `exporter_protocol: grpc | http/protobuf`, `logging.level: debug|info|warn|error` (case-insensitive). Loaders normalize these to the canonical protobuf enum names for both C++ and Python.


## RDMA Environment Variables

### `TENSORCAST_IB_HCA`

Specifies the InfiniBand HCA (Host Channel Adapter) device names to use for RDMA communication. Multiple device names can be specified, separated by commas. The "=" character in the value will be automatically removed.

**Usage:**
```bash
# Single device
export TENSORCAST_IB_HCA="mlx5_bond0"

# Multiple devices (comma-separated)
export TENSORCAST_IB_HCA="mlx5_bond0,mlx5_bond1"
```

If not set, TensorCast will automatically discover and use available InfiniBand devices.

#### `TENSORCAST_LLDP_FILE_NAME`

Specifies the path to an LLDP-style configuration file for Rail ID mapping. This file maps network interface names to PCI paths, mlx5 device names, and rail IDs for multi-rail RDMA configurations.

**File Format:**

```
eth1=0000:19:00.0,mlx5_bond100,1
```

**Usage:**
```bash
export TENSORCAST_LLDP_FILE_NAME="/path/to/lldp_config.txt"
```

**Notes:**
- Lines starting with `#` are treated as comments and ignored
- Empty lines are ignored
- Each non-comment line should follow the format: `eth_name=pci_path,mlx5_name,rail_id`
- If the environment variable is not set, Rail ID will be automatically derived from the mlx5 device name (e.g., `mlx5_0` → rail_id `0`, `mlx5_1` → rail_id `1`)
