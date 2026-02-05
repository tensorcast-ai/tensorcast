---
id: developer-guides/core/communicator/communicator-config-migration
title: CommunicatorConfig Migration (Typed Config)
sidebar_label: CommunicatorConfig Migration
---

This guide explains how to migrate from environment-variable based configuration to the typed `CommunicatorConfig` required by the Communicator engine.

Why: The unified MemoryStager and staged P2P require a consistent, typed configuration model across C++ and Python. Env-based knobs are removed.

## Summary

- New default: `CommunicatorConfig` is required at construction.
- Legacy envs are removed; use typed config fields instead.
- Config sources: explicit injection (CLI/code) > file (YAML/JSON/TOML) > built-in defaults.

## Migration Steps

- Replace legacy constructors with the `CommunicatorConfig` constructor in all engine callsites.
- Replace `register_tensor(...)` with `register_tensor_ex(..., RegisterTensorOptions)` and set options explicitly:
  - `register_mr=true` when RDMA is enabled and you want preregistered MRs.
  - `needs_staging=true` for GPU tensors over TCP (and for any path you want forced staging).
  - `async=true` to avoid blocking on MR registration when applicable.
- Build `CommunicatorConfig` via CLI args or a config file (recommended: YAML).
- Remove direct env reads from services; inject typed config into the daemon and clients.
- Remove any env usage in deployment scripts; switch to config files or explicit injection.

## Example YAML

```yaml
communicator:
  enable_rdma: false
  stager:
    stage_cpu_for_rdma: true
    buffers_per_flow: 4
    expected_gpu_channels: 0
  rdma:
    outstanding_wr: 64
    ack_ttl_ms: 30000
    traffic_class: 186
    qp_timeout: 20
    qp_retry: 7
  simple_numa:
    enable: true
    nodes:
      - id: 0
        nics: ["mlx5_0", "mlx5_1"]
        gpus: [0, 1]
        default: true
      - id: 1
        nics: ["mlx5_2", "mlx5_3"]
        gpus: [2, 3]
  transport:
    tcp_conn_count: 8
    connect_timeout_sec: 10
    tcp_tos: 0
  affinity:
    enable: false
```

Pinned pool sizing and chunking are no longer part of `CommunicatorConfig`. When running inside the Store Daemon, staging pools are injected from `DaemonConfig.pinned_memory` via the `comm_gpu` / `comm_cpu` classes. Standalone `Communicator` constructions (e.g. tests/tools) use internal defaults unless pools are explicitly injected.

## Env Deprecations and Replacements

- `DEFAULT_DEV` → `simple_numa.nodes[n].nics`
- `GPU_TCP_STAGER_CHUNK_SIZE_MB` → `DaemonConfig.pinned_memory.classes[name=comm_gpu].slice_bytes`
- `GPU_TCP_STAGER_NUM_BUFFERS` → `stager.buffers_per_flow`
- `GPU_TCP_RECV_NUM_BUFFERS` → `stager.buffers_per_flow`
- `RDMA_ACK_TTL_MS` → `rdma.ack_ttl_ms`
- `STAGER_NUMA_ENABLE` → `simple_numa.enable`
- `STAGER_NUMA_GPU_MAP` → `simple_numa.nodes[].gpus`
- `STAGER_NUMA_NIC_MAP` → `simple_numa.nodes[].nics`

## Failure Modes (by design)

- Missing typed config: Engine construction fails fast with a clear error. The engine no longer supports any environment-variable based configuration.
- Env-only configuration: Unsupported. Provide a typed config (e.g., via YAML or explicit injection). Missing required fields produce validation errors.
- Invalid `simple_numa` mappings: duplicate node/GPU IDs, empty NIC lists, or empty NIC names are rejected by Simple NUMA topology validation.

## Checklist

- [ ] All Communicator constructions use `CommunicatorConfig`.
- [ ] Services load YAML and inject typed config.
- [ ] No env reads in core codepaths.
