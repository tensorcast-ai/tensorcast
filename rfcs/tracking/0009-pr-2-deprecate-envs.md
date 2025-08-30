# Tracking Issue — PR-2: Deprecate and gate legacy environment variables

- Owner: Communicator (C++)
- Milestone: P2 (staged RDMA + ACK)
- Status: Planned
- RFC: rfcs/0009-unified-memory-stager-and-staged-p2p.md (Section 13.9)

## Goals
- Gate legacy env-based configuration behind `TENSORCAST_ALLOW_LEGACY_ENV=1` (default off).
- Provide a temporary `CommunicatorConfig::FromEnvDeprecated()` used only by the legacy shim.
- Log WARN once per variable and emit telemetry on use.

## Env mapping
- `DEFAULT_DEV` → `pool.simple_numa.nodes[n].nics`
- `GPU_TCP_STAGER_CHUNK_SIZE_MB` → `stager.stage_chunk_mb_gpu`
- `GPU_TCP_STAGER_NUM_BUFFERS` → `stager.buffers_per_flow`
- `GPU_TCP_RECV_NUM_BUFFERS` → `stager.buffers_per_flow`
- `RDMA_ACK_TTL_MS` → `rdma.ack_ttl_ms`
- `STAGER_NUMA_ENABLE` → `pool.simple_numa.enable`
- `STAGER_NUMA_GPU_MAP` → `pool.simple_numa.nodes[].gpus`
- `STAGER_NUMA_NIC_MAP` → `pool.simple_numa.nodes[].nics`

## Tasks
- [ ] Implement `CommunicatorConfig::FromEnvDeprecated()` builder (guarded by `TENSORCAST_ALLOW_LEGACY_ENV`).
- [ ] Add WARN logs mapping each env var to its typed field.
- [ ] Add counter metric `communicator.deprecated_env_used` with labels per variable.
- [ ] Ensure main codepaths do not read env directly.
- [ ] Update docs to show typed config alternatives and flag semantics.

## Acceptance
- [ ] With gate disabled (default), envs are ignored and engine requires typed config.
- [ ] With gate enabled, envs load with WARN logs and metrics increment.

