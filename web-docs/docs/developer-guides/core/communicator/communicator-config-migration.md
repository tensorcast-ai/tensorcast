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
- Build `CommunicatorConfig` via CLI args or a config file (recommended: YAML).
- Remove direct env reads from services; inject typed config into the daemon and clients.
- Remove any env usage in deployment scripts; switch to config files or explicit injection.

## Example YAML

```yaml
communicator:
  stager:
    stage_cpu_for_rdma: true
    stage_chunk_mb_cpu: 4
    stage_chunk_mb_gpu: 16
    buffers_per_flow: 4
  rdma:
    outstanding_wr: 64
    ack_ttl_ms: 30000
  pool:
    preregister_mr: true
    pool_size_bytes: 8589934592
    chunk_bytes: 67108864
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
  affinity:
    enable: false
```

## Env Deprecations and Replacements

- `DEFAULT_DEV` → `pool.simple_numa.nodes[n].nics`
- `GPU_TCP_STAGER_CHUNK_SIZE_MB` → `stager.stage_chunk_mb_gpu`
- `GPU_TCP_STAGER_NUM_BUFFERS` → `stager.buffers_per_flow`
- `GPU_TCP_RECV_NUM_BUFFERS` → `stager.buffers_per_flow`
- `RDMA_ACK_TTL_MS` → `rdma.ack_ttl_ms`
- `STAGER_NUMA_ENABLE` → `pool.simple_numa.enable`
- `STAGER_NUMA_GPU_MAP` → `pool.simple_numa.nodes[].gpus`
- `STAGER_NUMA_NIC_MAP` → `pool.simple_numa.nodes[].nics`

## Failure Modes (by design)

- Missing typed config and gate unset: Engine construction fails fast with a clear error.
- Gate set but env missing required fields: Deprecated env loader warns and fills only available fields; validation should fail for missing required fields.

## Checklist

- [ ] All Communicator constructions use `CommunicatorConfig`.
- [ ] Services load YAML and inject typed config.
- [ ] No env reads in core codepaths; deprecated loader only used by legacy shim.
