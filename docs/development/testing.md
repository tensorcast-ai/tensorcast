---
title: Testing
description: How to run TensorCast tests (Python, C++, P2P, RDMA)
sidebar_position: 1
---

# Testing

This guide consolidates the common test commands for TensorCast. Python tests
must run with `uv run`; C++ tests use Bazel.

## Python tests

```bash
source .venv/bin/activate
uv run pytest tests/python/
uv run pytest tests/python/test_global_store.py
```

## C++ tests (Bazel)

```bash
# Core tests without stress, rdma, or multi_gpu tags
bazel test //core/... --verbose_failures \
  --test_tag_filters="-stress,-rdma,-multi_gpu" \
  --test_output=errors \
  --test_summary=detailed

# Stress-only
bazel test //core/... --test_tag_filters="+stress"
```

To force the fake CUDA backend in C++ tests, add `--test_env=TENSORCAST_CUDA_BACKEND=fake`.
To use real CUDA, leave the env unset (or set `TENSORCAST_CUDA_BACKEND=real`).

## Communicator tests (TCP/RDMA)

```bash
bazel test //core/communicator:tcp_engine_test
bazel test //core/communicator:tcp_transfer_test
bazel test //core/communicator:rdma_engine_test --test_env=TENSORCAST_CUDA_BACKEND=fake
bazel test //core/communicator:routing_context_test --test_output=errors
```

RDMA device selection and rail mapping are configured via environment variables.
See `docs/deployment/store-daemon.md#rdma-environment-variables`.

## Topology-guided routing validation (GPU-GPU / GPU-CPU / CPU-CPU)

Use `//core/communicator:routing_context_test` as the functional contract test
for topology-guided rail selection. The suite contains explicit coverage for:

- GPU-GPU: cross-node fallback maps 8 GPU pairs to 8 rail-matched NIC paths.
- GPU-CPU: when source preferred rail is unavailable on destination bindings,
  selection falls back to destination CPU-affine NIC.
- CPU-CPU: both source and destination preserve CPU-pool affinity in rail
  selection.

Run:

```bash
bazel test //core/communicator:routing_context_test --test_output=errors
```

### Two-node 8xH800 affinity smoke (brainctl automated)

For a reproducible end-to-end validation (including worker launch, non-root
remote execution, RNIC intersection filtering, and log assertions), run:

```bash
tools/testing/topology_guided_routing_2node_h800_smoke.sh
```

The script enforces the runtime constraints:

- It launches workers with at least 8 GPUs and runs transfer with `--gpu 8 --rdma`.
- Default placement is `POSITIVE_TAGS=H800,ib`; set `EXTRA_LAUNCH_FLAGS` for
  cluster-specific RDMA passthrough requirements.
- It performs strict verbs preflight on both workers:
  `/dev/infiniband` must exist and `ibv_devinfo -l` must report `mlx5_*` HCAs.
- It discovers verbs-visible RNIC sets and uses the 8-device
  intersection as `TENSORCAST_IB_HCA` on both nodes (`HCA_SELECTION_MODE=first|last`
  or `IB_HCA_OVERRIDE=mlx5_x,...`), then runs a short communicator probe and
  keeps only `Dev: mlx5_* added` devices as final candidates.
- It runs transfer verification with bounded retries (`MAX_TRANSFER_ATTEMPTS`,
  default `3`). On handshake timeout (`[rdma_handshake] transport connect failed`
  with `local_dev=mlx5_*`), it excludes failed RNICs from candidate set, rebuilds
  an 8-device selection from remaining common HCAs, and retries automatically.
- It asserts 8 successful with-regmr + 8 successful no-regmr reads, full
  read-path coverage for all 8 tensor keys, and affinity spread/handshake
  evidence with a strict minimum of 7 unique NIC/connect pairs (for 8-GPU run)
  to tolerate one shared rail while still catching collapsed routing.
- It bounds client runtime with `CLIENT_TIMEOUT_SEC` (default 600s) and fails
  fast with server/client log tails on timeout or transfer failure.

By default workers are deleted after validation. Set `KEEP_WORKERS=1` to keep
them for debugging.

Example (inject extra launch flags when your cluster requires explicit RDMA
device/plugin exposure):

```bash
EXTRA_LAUNCH_FLAGS='--custom-resources rdma/mlnx_shared=8 --host-network=true' \
tools/testing/topology_guided_routing_2node_h800_smoke.sh
```

### Two-node 8xH800 affinity smoke (manual, RDMA)

If you prefer running manually, follow the steps below.

1. Build binary:

```bash
bazel build //core/communicator:gpu_ce_test_binary
```

2. On both nodes, restrict to the same 8-HCA subset (example: rails 9-16):

```bash
export TENSORCAST_IB_HCA=mlx5_9,mlx5_10,mlx5_11,mlx5_12,mlx5_13,mlx5_14,mlx5_15,mlx5_16
```

3. Start server (node A):

```bash
./bazel-bin/core/communicator/gpu_ce_test_binary \
  --actor server --port 19099 --gpu 8 --chunk 1 --count 16777216 --rdma
```

4. Start client (node B):

```bash
./bazel-bin/core/communicator/gpu_ce_test_binary \
  --actor client --ip <SERVER_IP> --port 19099 --gpu 8 --chunk 1 --count 16777216 --rdma
```

5. Validate logs (both sides):

- Client has 8 successful with-regmr and 8 successful no-regmr lines:
  - `with regmr result: key=gpu-ce-test-tensor-<i>-0, status=0`
  - `no regmr result: key=gpu-ce-test-tensor-<i>-0, status=0`
- Client read path logs include per-request NIC selection:
  - `read tensor ... key=gpu-ce-test-tensor-<i>-0 ... net_dev=mlx5_*`
- Handshake logs show matched local/peer NICs:
  - `[rdma_handshake] dev=mlx5_* peer=mlx5_*`

If one node has fewer HCAs than the other, keep the test on the intersection
set via `TENSORCAST_IB_HCA`; this avoids asymmetric rail selection and keeps
pairing deterministic.

## Multi-machine communicator tests (manual)

Build the binaries on both hosts:

```bash
bazel build //core/communicator:cpu_ce_test_binary
bazel build //core/communicator:gpu_ce_test_binary
```

CPU transfer test (TCP or RDMA):

```bash
# Host A (server)
./bazel-bin/core/communicator/cpu_ce_test_binary --actor server --ip 0.0.0.0 --port 19099

# Host B (client)
./bazel-bin/core/communicator/cpu_ce_test_binary --actor client --ip <SERVER_IP> --port 19099
```

GPU transfer test (requires CUDA on both hosts):

```bash
# Host A (server)
./bazel-bin/core/communicator/gpu_ce_test_binary --actor server --port 19099 --gpu 1 --chunk 4 --count 67108864

# Host B (client)
./bazel-bin/core/communicator/gpu_ce_test_binary --actor client --ip <SERVER_IP> --port 19099 --gpu 1 --chunk 4 --count 67108864
```

Use `--rdma` on both sides to enable RDMA (requires verbs-capable NICs). Stop
the processes with Ctrl+C when finished.

## Store P2P tests (Replica)

```bash
bazel test //core/store/replica:replica_p2p_registration_test --test_env=TENSORCAST_CUDA_BACKEND=fake
bazel test //core/store/replica:replica_p2p_transfer_test --test_env=TENSORCAST_CUDA_BACKEND=fake
```

These tests are tagged `requires_cuda`. To skip them, add
`--test_tag_filters=-requires_cuda`.

## Store dataplane routing tests

Use this test when validating the new routed-read wrapper interface in
`RemoteKeySource` and its strict fallback to direct `ip:port` reads.

```bash
bazel test //core/store/materialization/dataplane:remote_key_source_routing_fallback_test \
  --test_env=TENSORCAST_CUDA_BACKEND=fake \
  --test_output=errors
```
