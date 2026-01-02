---
slug: unified-pinned-memory-authority
title: Unified Pinned Memory Authority (Plan)
areas: ["core", "daemon", "sdk"]
related_code:
  - core/store/runtime/context/runtime_context.cc
  - core/communicator/engine/engine.cc
  - proto/tensorcast/config/v1/daemon_config.proto
  - proto/tensorcast/communicator/v1/communicator_config.proto
links:
  design: ../designs/0043-unified-pinned-memory-authority.md
---

# Objective

Implement a unified pinned memory authority in the daemon, simplify runtime configuration, remove redundant pool fields, and route daemon-owned streaming flows through the shared pinned arena.

# Current State & Grounding (Updated)

- The daemon requires `pinned_memory` in `DaemonConfig` and validates the required pinned classes exist (`engine`, `comm_gpu`, `comm_cpu`).
- Phase 1 uses fixed-allocation pinned pools: all pinned memory is preallocated at daemon startup; total pinned bytes is derived as `sum(classes[].pool_bytes)`.
- StoreEngine uses the daemon-owned pinned pool for the `engine` class (no `engine.mem_pool_size_bytes` / `engine.tx_slice_bytes` knobs exist).
- Communicator staging pools are injected from pinned classes (`comm_gpu`, `comm_cpu`) and no longer accept `communicator.pool` / staging chunk sizing fields.
- RDMA preregistration is slab-level (one MR per slab per NIC/PD), gated by `communicator.enable_rdma` and per-class `rdma_preregister`.
- StoreEngine streaming buffer depth is configurable via `engine.streaming_buffer_chunks` (used for disk/P2P loads and local CPU→GPU copies; no hard-coded depths in daemon-owned streaming paths).
- MTCP staging does not block inside `MemoryStager::stage()`: it uses non-blocking staging attempts plus bounded retry (deadline from `pinned_memory.allocation_timeout`) to avoid hangs while preserving backpressure.
- Metrics now expose `tc_pinned_*` per-class gauges and global pinned budget/commit gauges.
- SDK local disk save/load utilities are test-only and may allocate pinned memory independently (they are not part of daemon pinned budgeting).

## Constraints and assumptions

- No backward compatibility: legacy pool fields must be removed and rejected.
- All Python entry points and tests must use `uv run` per repo policy.
- Protobuf regeneration is required after any `.proto` edits (`bash tools/build_proto_python.sh`).

# Phases and Milestones

- [x] Phase 1: Schema and hard cutover
  - [x] Milestone: Add `pinned_memory` to `DaemonConfig` and delete legacy pool + staging chunk fields (no compatibility layer).
  - [x] Milestone: Regenerate protobuf code and update config loaders + example configs to the new schema.
  - [x] Milestone: Update `examples/config/store_daemon_config.yaml` to the new `pinned_memory` model and validate daemon startup against it.

- [x] Phase 2: PinnedMemoryAuthority core (ops-safe)
  - [x] Milestone: Add `core/common/memory/pinned_memory_authority.*` with per-class slab allocation and bounded-wait (timeout) acquisition.
  - [x] Milestone: Add `StreamingPinnedBuffer` backed by class pools and define its migration path from any legacy streaming buffer type.

- [x] Phase 3: StoreEngine integration
  - [x] Milestone: Replace `PinnedBufferPool` creation in `RuntimeContext` with PMA class pools.
  - [x] Milestone: Replace hard-coded streaming buffer chunk counts with `engine.streaming_buffer_chunks`.

- [x] Phase 4: Communicator integration
  - [x] Milestone: Remove `communicator.pool` and staging chunk fields; use PMA `comm_gpu`/`comm_cpu` classes for staging.
  - [x] Milestone: Update RDMA preregistration to preregister only staged-RDMA pinned classes and register slabs once per NIC/PD (not per slice), gated by `communicator.enable_rdma`.

- [ ] Phase 5: Documentation and verification
  - [x] Milestone: Update module docs and deployment guides to the new pinned config model and operational expectations (deadlock avoidance + diagnostics).
  - [ ] Milestone: Add targeted tests/stress coverage for pinned allocation stalls, timeouts, and RDMA preregistration paths.

# Tasks

- Define `PinnedMemory` in `proto/tensorcast/config/v1/daemon_config.proto`.
- Validate `pinned_memory.classes[].slice_bytes % 4096 == 0` and fail startup with a precise error (page alignment; implies 512 B O_DIRECT alignment).
- Enforce Phase 1 fixed-allocation invariants: total pinned bytes is derived as `sum(classes[].pool_bytes)` (no runtime growth).
- Apply daemon-side defaults for omitted config: `pinned_memory.allocation_timeout` defaults to 30s; `engine.streaming_buffer_chunks` defaults to 16.
- Add daemon startup validation that required class names exist (`engine`, `comm_gpu`, `comm_cpu`) and that StoreEngine/Communicator wiring uses the intended classes.
- Validate Communicator pinned sizing at daemon startup (derived from `buffers_per_flow`, `transport.tcp_conn_count`, and `expected_gpu_channels`) against PMA class capacities (`comm_gpu`/`comm_cpu`) so startup is fail-fast.
- Define the new source of truth for direct RDMA window chunking: set `direct_rdma_chunk_bytes = pinned_memory.classes[name=comm_gpu].slice_bytes` and delete `stager.direct_chunk_mb`.
- Remove `communicator.pool`, `communicator.stager.stage_chunk_mb_gpu`, `communicator.stager.stage_chunk_mb_cpu`, and `communicator.stager.direct_chunk_mb` from `proto/tensorcast/communicator/v1/communicator_config.proto`.
- Update `core/common/config/daemon_config_io.cc` and `tensorcast/daemon_runtime_config.py` for new fields.
- Implement PMA with per-class slab allocation (contiguous slabs per pool) to reduce RDMA MR registrations.
- For classes with `rdma_preregister=true`, preregister the full `pool_bytes` allocation so the MR set is stable at startup; preregistration is ignored when `communicator.enable_rdma=false`.
- Implement slab-level RDMA preregistration and reuse in Communicator: register each slab once per NIC/PD and plumb the MR metadata so staged RDMA can use `slab_base + offset` without per-slice registrations.
- Remove production-path `LOG(FATAL)` behavior from streaming buffer cleanup (e.g., `StreamingPinnedBuffer::release()` timeout) and replace with diagnosable error returns + metrics/logging.
- Replace `PinnedBufferPool` instantiations in StoreEngine and Communicator with PMA class pools.
- Update UMA budget calculation and validation to subtract total pinned pools (`sum(pinned_memory.classes[].pool_bytes)`).
- Add per-class pinned metrics in `core/store/components/metrics_collector.*`.
- Update docs: `examples/config/store_daemon_config.yaml`, `daemon/README.md`, `core/store/README.md`, `core/communicator/README.md`, `core/checkpoint/README.md`, `docs/deployment/store-daemon.md`.

# Acceptance Checks

- Daemon starts with only `pinned_memory` and required classes; no legacy fields are accepted.
- Phase 1 fixed-allocation is enforced: total pinned bytes is derived as `sum(classes[].pool_bytes)`.
- StoreEngine and Communicator allocate from PMA class pools.
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
- Risk: RDMA preregistration cost rises if class `pool_bytes` is too large.

# Progress Log

- 2026-01-01: Enforced Phase 1 fixed-allocation semantics (derived total pinned bytes = `sum(classes[].pool_bytes)`), removed remaining hard-coded StoreEngine streaming depths, improved pinned allocation diagnostics (named pools + periodic wait logs), updated example/deployment configs, and validated with `bazel test //core/common:streaming_pinned_buffer_test --test_env=TENSORCAST_CUDA_BACKEND=fake`, `bazel test //core/store:store_engine_test --test_env=TENSORCAST_CUDA_BACKEND=fake`, and `USE_BAZEL_VERSION=8.5.0 bazel test //core/communicator:tcp_engine_test --test_env=TENSORCAST_CUDA_BACKEND=fake`.
- 2026-01-02: Switched MTCP staging to non-blocking `StageMode::kTry` and bounded retries by `pinned_memory.allocation_timeout`, removed the last local-copy hard-coded streaming depth cap (now respects `engine.streaming_buffer_chunks`), renamed pinned class sizing from `min_bytes`/`max_bytes` to `pool_bytes`, removed `pinned_memory.total_bytes` from the config schema (derived from `sum(classes[].pool_bytes)`), regenerated protobuf code with `bash tools/build_proto_python.sh`, updated CLI/tests/configs, and validated with `bazel test //core/communicator:staging_flow_controller_test --test_env=TENSORCAST_CUDA_BACKEND=fake`, `bazel test //core/communicator:tcp_error_handling_test --test_env=TENSORCAST_CUDA_BACKEND=fake`, and `bazel test //core/store/runtime/ingestion:materialization_service_test --test_env=TENSORCAST_CUDA_BACKEND=fake`.
