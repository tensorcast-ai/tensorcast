---
slug: unified-pinned-memory-authority
title: Unified Pinned Memory Authority (Plan)
areas: ["core", "daemon", "sdk"]
related_code:
  - core/store/runtime/context/runtime_context.cc
  - core/communicator/engine/engine.cc
  - core/checkpoint/checkpoint_streaming.cc
  - proto/tensorcast/config/v1/daemon_config.proto
  - proto/tensorcast/communicator/v1/communicator_config.proto
links:
  design: ../designs/0043-unified-pinned-memory-authority.md
---

# Objective

Implement a unified pinned memory authority in the daemon, simplify runtime configuration, remove redundant pool fields, and route daemon-owned streaming flows through the shared pinned arena.

# Current State & Grounding

- StoreEngine creates a dedicated `PinnedBufferPool` from `engine.mem_pool_size_bytes` and `engine.tx_slice_bytes` in `core/store/runtime/context/runtime_context.cc`.
- Communicator creates its own pinned pool using `communicator.pool.pool_size_bytes` and `communicator.pool.chunk_bytes` in `core/communicator/engine/engine.cc`.
- Checkpoint streaming creates a separate pinned pool in `core/checkpoint/checkpoint_streaming.cc` and is configured via `checkpoint.streaming.*`.
- UMA capacity deductions use the StoreEngine pool size only (`core/store/runtime/context/runtime_context.cc`).
- Configuration fields are split between `proto/tensorcast/config/v1/daemon_config.proto` and `proto/tensorcast/communicator/v1/communicator_config.proto`.
- When `communicator.enable_rdma=true`, the Communicator eagerly registers MRs for every pinned pool slice across every RDMA PD at startup (`core/communicator/engine/engine.cc`). The typed config field `communicator.pool.preregister_mr` exists but is not currently honored by the engine.
- Example configs and CLI defaults still reference legacy fields (`examples/config/store_daemon_config.yaml`, `tensorcast/cli_utils/config.py`, `tensorcast/cli.py`).
- Store metrics expose available pinned bytes via `core/store/components/metrics_collector.{h,cc}` and expect a single pool today.

## Constraints and assumptions

- No backward compatibility: legacy pool fields must be removed and rejected.
- All Python entry points and tests must use `uv run` per repo policy.
- Protobuf regeneration is required after any `.proto` edits (`bash tools/build_proto_python.sh`).

# Phases and Milestones

- [ ] Phase 1: Schema and hard cutover
  - [ ] Milestone: Add `pinned_memory` to `DaemonConfig` and delete legacy pool + staging chunk fields (no compatibility layer).
  - [ ] Milestone: Regenerate protobuf code and update config loaders + example configs to the new schema.
  - [ ] Milestone: Update `examples/config/store_daemon_config.yaml` to the new `pinned_memory` model and validate daemon startup against it.

- [ ] Phase 2: PinnedMemoryAuthority core (ops-safe)
  - [ ] Milestone: Add `core/common/memory/pinned_memory_authority.*` with per-class slab allocation and deadline-based acquisition.
  - [ ] Milestone: Add `StreamingPinnedBuffer` backed by class pools and define its migration path from any legacy streaming buffer type.

- [ ] Phase 3: StoreEngine integration
  - [ ] Milestone: Replace `PinnedBufferPool` creation in `RuntimeContext` with PMA class pools.
  - [ ] Milestone: Replace hard-coded streaming buffer chunk counts with `engine.streaming_buffer_chunks`.

- [ ] Phase 4: Communicator integration
  - [ ] Milestone: Remove `communicator.pool` and staging chunk fields; use PMA `comm_gpu`/`comm_cpu` classes for staging.
  - [ ] Milestone: Update RDMA preregistration to preregister only staged-RDMA pinned classes and register slabs once per NIC/PD (not per slice), gated by `communicator.enable_rdma`.

- [ ] Phase 5: Daemon-owned checkpoint streaming
  - [ ] Milestone: Implement daemon RPC for checkpoint streaming using PMA class pools.
  - [ ] Milestone: Update Python API to call daemon for streaming saves.

- [ ] Phase 6: Documentation and verification
  - [ ] Milestone: Update module docs and deployment guides to the new pinned config model and operational expectations (deadlock avoidance + diagnostics).
  - [ ] Milestone: Add targeted tests/stress coverage for pinned allocation stalls, timeouts, and RDMA preregistration paths.

# Tasks

- Define `PinnedMemory` in `proto/tensorcast/config/v1/daemon_config.proto`.
- Validate `pinned_memory.classes[].slice_bytes % 4096 == 0` and fail startup with a precise error (page alignment; implies 512 B O_DIRECT alignment).
- Add daemon startup validation that required class names exist (`engine`, `comm_gpu`, `comm_cpu`, `checkpoint`) and that StoreEngine/Communicator/checkpoint wiring uses the intended classes.
- Replace Communicator capacity validation formulas (currently derived from `stager.stage_chunk_mb_*`, `buffers_per_flow`, `transport.tcp_conn_count`, and pool sizing) with equivalent validation against PMA class capacities (`comm_gpu`/`comm_cpu`) so startup remains fail-fast.
- Define the new source of truth for direct RDMA window chunking: set `direct_rdma_chunk_bytes = pinned_memory.classes[name=comm_gpu].slice_bytes` and delete `stager.direct_chunk_mb`.
- Remove `communicator.pool`, `communicator.stager.stage_chunk_mb_gpu`, `communicator.stager.stage_chunk_mb_cpu`, and `communicator.stager.direct_chunk_mb` from `proto/tensorcast/communicator/v1/communicator_config.proto`.
- Remove `checkpoint.streaming.pinned_pool_bytes` and `checkpoint.streaming.io_chunk_bytes` from `proto/tensorcast/config/v1/daemon_config.proto` and bind checkpoint streaming IO chunking to the `checkpoint` class `slice_bytes`.
- Update `core/common/config/daemon_config_io.cc` and `tensorcast/daemon_runtime_config.py` for new fields.
- Implement PMA with per-class slab allocation (contiguous slabs per pool) to reduce RDMA MR registrations.
- For classes with `rdma_preregister=true`, enforce `max_bytes == min_bytes` so the preregistered MR set is stable and fully allocated at startup; preregistration is ignored when `communicator.enable_rdma=false`.
- Implement slab-level RDMA preregistration and reuse in Communicator: register each slab once per NIC/PD and plumb the MR metadata so staged RDMA can use `slab_base + offset` without per-slice registrations.
- Remove production-path `LOG(FATAL)` behavior from streaming buffer cleanup (e.g., `StreamingPinnedBuffer::release()` timeout) and replace with diagnosable error returns + metrics/logging.
- Replace `PinnedBufferPool` instantiations in StoreEngine, Communicator, and checkpoint streaming with PMA class pools.
- Update UMA budget calculation and validation to subtract `pinned_memory.total_bytes`.
- Add per-class pinned metrics in `core/store/components/metrics_collector.*`.
- Implement daemon checkpoint streaming RPC and update `tensorcast/api/_io_disk.py` and CLI flows.
- Update docs: `examples/config/store_daemon_config.yaml`, `daemon/README.md`, `core/store/README.md`, `core/communicator/README.md`, `core/checkpoint/README.md`, `docs/deployment/store-daemon.md`.

# Acceptance Checks

- Daemon starts with only `pinned_memory` and required classes; no legacy fields are accepted.
- StoreEngine, Communicator, and checkpoint streaming allocate from PMA class pools.
- `engine.artifact_chunk_bytes % engine.slice_bytes == 0` enforced.
- `comm_gpu` and `comm_cpu` capacity checks fail fast on mis-sizing.
- RDMA preregistration registers slabs once per NIC/PD only when RDMA is enabled.
- Metrics expose per-class pinned usage and global pinned usage; stall logs are emitted on deadline waits.

# Test, Rollout, Backout

- Tests:
  - After proto changes, run `bash tools/build_proto_python.sh` and `bazel test //proto/... --test_output=errors`.
  - Update and run `bazel test //core/store:store_engine_test` and materialization streaming tests.
  - Update and run `bazel test //core/communicator:config_io_test` and transport tests.
  - Update Python config tests in `tests/python/test_config_enum_normalization.py` and daemon lifecycle tests.
- Rollout:
  - Regenerate example configs and validate daemon startup with the new schema.
- Backout:
  - Revert to the previous commit set. No compatibility layer is maintained.

# Risks & Tracking

- Risk: contention or allocation stalls across classes if class sizing is poor.
- Risk: RDMA preregistration cost rises if class min bytes are too large.
- Risk: checkpoint streaming RPC adds new daemon dependency for save flows.

# Progress Log

- (empty)
