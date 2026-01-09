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
```

RDMA device selection and rail mapping are configured via environment variables.
See `docs/deployment/store-daemon.md#rdma-environment-variables`.

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
