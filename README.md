# TensorCast

## Introduction

[Developer Guide](web-docs/docs/developer-guides/README.md)

## Prerequisites

```bash
# Install uv & pre-commit
curl -LsSf https://astral.sh/uv/install.sh | sh

uv tool install pre-commit --with pre-commit-uv

# Install bazel
./tools/install-bazel.sh

# Dependency that need for compile
sudo apt-get install -y libxml2

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

# Install dependencies
#   - BUILD_CORE means cxx files in core/
#   - BUILD_EXTENSION means cxx files in tensorcast/csrc
BUILD_CORE=1 BUILD_EXTENSION=1 uv run -vvv setup.py build_ext

# Run tests
uv run pytest tests/python/**.py

# Run cli target
uv run tensorcast-cli --help

# Run specific file
uv run xxx/xxx.py

### StoreDaemon (C++)

The StoreDaemon service is implemented in C++ and launched by the Python CLI.

- Development (from source):
  - Build once with Bazel: `bazel build //daemon:tensorcast_daemon`
  - Start background: `uv run -q python -m tensorcast.cli start --non-blocking --host 127.0.0.1 --port 8073`
  - Stop: `uv run -q python -m tensorcast.cli stop`
  - The CLI automatically locates the binary from `bazel-bin/daemon/tensorcast_daemon`.

- Packaged (wheel) usage:
  - The wheel packages the daemon at `tensorcast/bin/tensorcast_daemon` and the CLI will use it automatically.
  - To override, set `TENSORCAST_DAEMON_BIN` to an absolute path to a `tensorcast_daemon` executable.

Example typed CommunicatorConfig (YAML)

```yaml
server:
  host: 0.0.0.0
  port: 50052
  enable_p2p_engine: true
  enable_rdma: false
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

Pass this to the StoreDaemon via the Python loader (see `tensorcast.daemon_config`). When `communicator` is present,
the daemon initializes the communication engine using these typed settings (no environment variables).


```bash
# Create a symlink to external packages for header files
# Run this on the root directory of the project
ln -s $(bazel info output_base)/external external
```

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
# This script starts the Global Store service, which acts as a central registry
# for discovering and managing distributed artifact replicas.
# It listens for gRPC requests on the specified port.
# Default port: 50051
# Default workers: 10
uv run -m tensorcast.global_store --port 50051 --workers 10

# Run the Global Store Server in Docker
sudo docker run -d --name global-store -p 50051:50051 hub.i.basemind.com/tensorcast/global-store:2025.04.27-55f24
```
