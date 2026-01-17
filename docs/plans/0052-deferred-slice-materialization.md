---
slug: 0052-deferred-slice-materialization
title: Plan - Deferred Slice Materialization for vLLM
links:
  design: ../designs/0052-deferred-slice-materialization.md
---

# Objective

Implement a daemon-owned, deferred slice loading path that returns placeholder CUDA tensors immediately and fills them after an explicit `DeferredLoader.commit()` barrier. Use a session-owned arena so CUDA IPC handle count is bounded (not per-parameter), and optionally publish the resulting slice-artifact as an in-memory (`is_memory_replica`) artifact.

# Current State & Grounding

- Existing async materialization is “alloc then confirm” but SDK forbids returning tensors when `wait_for_completion=False`: `tensorcast/api/_materialize.py`.
- View planning exists and can produce packed outputs for `narrow`: `core/store/materialization/dataplane/view/view_planner.cc`.
- Copy pipeline exists (`SeekableSource → pump_ranges → PositionedSink`) with async H2D support: `core/store/materialization/dataplane/runtime/pump.cc`, `core/common/async_copy_manager.*`.
- Global Store supports memory replica registration (`tensor_index_key`, `remote_memory_keys`): `schema.sql`, `core/store/components/global_store_client.cc`, `proto/tensorcast/common/v1/common.proto`.

# Phases & Milestones

- [ ] Phase 1: Proto + daemon session skeleton
  - [ ] Milestone 1: Add new RPCs/messages to `proto/tensorcast/daemon/v2/store_daemon.proto`
  - [ ] Milestone 2: Implement daemon-side `DeferredSliceLoadSession` registry (PID-bound, TTL cleanup)
  - [ ] Milestone 3: Implement `AllocateDeferredSlice` allocating from a session arena and returning `(storage_id, optional MemCopyHandle, storage_offset_bytes)` + slice metadata

- [ ] Phase 2: Core dataplane (deferred copy execution)
  - [ ] Milestone 1: Implement (or reuse from 0042) a TargetLayout-backed GPU sink supporting ordered concatenation across multiple storage windows
  - [ ] Milestone 2: Implement daemon execution path `CommitDeferredSliceLoad`:
    - build a session SelectionPlan from registered slices (src_offset → dst_offset, with PAD=0 alignment)
    - open source (disk first; P2P/local replica follow-on)
    - run `pump_ranges` to fill the session arena (linear ByteSpace over the arena storages)
  - [ ] Milestone 3: Wire observability (spans/metrics) for copy planning + execution

- [ ] Phase 3: Python SDK API surface
  - [ ] Milestone 1: Add `Artifact.deferred_loader(device=...)` returning `DeferredLoader`
  - [ ] Milestone 2: Add `DeferredLoader.tensor(name, slice=...)` that calls `AllocateDeferredSlice` and rehydrates `torch.Tensor` via `_C.get_cuda_memory_ptr` + `_C.restore_tensors`
  - [ ] Milestone 3: Add `DeferredLoader.commit()` that calls `CommitDeferredSliceLoad` and raises on failure

- [ ] Phase 4: Global Store publication (optional but recommended)
  - [ ] Milestone 1: Define `cgid:` naming scheme for produced slice artifacts (include TTL)
  - [ ] Milestone 2: Register communicator keys for slice buffers and publish via `register_memory_replica`
  - [ ] Milestone 3: (Optional) Upsert `variants` metadata for traceability (not required for routing)

- [ ] Phase 5: Tests + docs
  - [ ] Milestone 1: C++ unit tests for TargetLayout sink mapping + selection-plan assembly (fake CUDA)
  - [ ] Milestone 2: Daemon gRPC tests for allocate/commit lifecycle and cleanup
  - [ ] Milestone 3: Python tests for placeholder semantics and `DeferredLoader.commit()` (use `TENSORCAST_CUDA_BACKEND=fake`)
  - [ ] Milestone 4: Update `docs/internals/model-loading.md` to document the new deferred slice mode

# Rollout / Backout

- Rollout behind an `Artifact` method (`Artifact.deferred_loader`) without changing existing `Artifact.tensor*` semantics.
- Backout by removing the new entrypoint and RPCs; no persistent schema changes are required for Phase 1–3.
