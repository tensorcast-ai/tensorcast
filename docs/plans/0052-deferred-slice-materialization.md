---
slug: 0052-deferred-slice-materialization
title: Plan - Deferred Slice Materialization for vLLM
links:
  design: ../designs/0052-deferred-slice-materialization.md
areas:
  - sdk
  - daemon
  - core
  - global_store
  - proto
related_code:
  - tensorcast/api/store/artifact.py
  - tensorcast/api/store/materialization.py
  - tensorcast/api/store/views.py
  - tensorcast/api/store/view_composer.py
  - tensorcast/api/_region_cache.py
  - tensorcast/api/store/runtime.py
  - tensorcast/api/_materialize.py
  - tensorcast/api/store/types.py
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - daemon/service/controllers/materialization_controller.cc
  - daemon/service/controllers/registration_controller.cc
  - core/store/runtime/ingestion/materialization_facade.cc
  - core/store/materialization/dataplane/sinks/target_layout_gpu_sink.{h,cc}
  - core/store/materialization/dataplane/view/view_planner.{h,cc}
  - core/store/materialization/dataplane/view/view_plan_source.{h,cc}
  - core/store/materialization/dataplane/runtime/pump.{h,cc}
  - docs/internals/model-loading.md
---

# Objective

Enable vLLM's "meta-init + per-param weight_loader" integration by providing a TensorCast `DeferredLoader` that:

- returns placeholder CUDA tensors immediately (views into a client-owned arena),
- fills them with a single daemon data-plane call at `DeferredLoader.commit()` (via `MaterializeIntoTarget`),
- optionally publishes the produced slice-artifact for P2P reuse using existing registration primitives.

# Current State & Grounding

- Region-backed ingestion is implemented in the SDK (`MaterializationPipeline._materialize_into_target`), building a
  `TargetLayout` over client-registered VRAM regions and calling `materialize_into_target_v2`
  (`tensorcast/api/store/materialization.py`).
- View construction and hashing exist (`compute_view_index_bytes`, `compute_view_id`, `compute_view_subset_hash`),
  and the view runtime supports `narrow` selection in core (`core/store/materialization/dataplane/view/*`).
- `TargetLayoutGpuSink` rejects writes that span storage boundaries, so `TargetLayout` offsets must be grouped by storage
  and materialization must respect per-storage range constraints (`core/store/materialization/dataplane/sinks/`).
- SDK region-backed layouts require registered VRAM regions and consult the client cache
  (`tensorcast/api/_region_cache.py`).
- Registration and publish primitives exist (VRAM region registration, begin/feed/commit registered artifacts) via the
  daemon controllers and SDK APIs (`daemon/service/controllers/registration_controller.cc`,
  `tensorcast/api/store/__init__.py`).
- Global Store view-aware routing is still canonical-only; Phase 1 must not depend on view routing for correctness.

# Scope & Guardrails

- No new daemon RPCs in Phase A/B. `MaterializeIntoTarget` is the only data-plane call in `commit()`.
- Do not introduce a new placeholder lease system; placeholders are client-owned CUDA buffers.
- Slices are `narrow` only in Phase A/B. Other view ops stay out of scope.
- Avoid ad-hoc env vars; reuse existing config and runtime discovery paths.

# Phases & Milestones

- [x] Phase A: SDK-first deferred loader (no new daemon RPCs)
  - [x] Milestone A1: Add `Artifact.deferred_loader(device=...)` and a concrete `DeferredLoader` type.
    - Define public surface in `tensorcast/api/store/artifact.py` and export in `tensorcast/api/store/__init__.py`.
    - Implement context manager semantics and lifecycle checks consistent with `Artifact` and `StoreRuntimeContext`.
  - [x] Milestone A2: Client-owned CUDA arena allocation + VRAM region registration.
    - Allocate a single contiguous CUDA buffer for Phase A and map sub-tensors as views.
    - Register the allocation with `RegisterVramRegion` and update the local region cache so
      `_materialize_into_target` can find it.
    - Track TTL and unregister on close/failure paths.
  - [x] Milestone A3: Implement `DeferredLoader.tensor(name, slice=...)`.
    - Parse and validate slice specs against canonical index bytes (dtype, shape, narrow bounds).
    - Record per-tensor view ops and selection list in a loader-local plan structure.
    - Return a CUDA tensor view into the arena; contents undefined until `commit()`.
  - [x] Milestone A4: Implement `DeferredLoader.commit()` using a single `MaterializeIntoTarget`.
    - Reuse `_build_region_backed_layout` logic to build `TargetLayout` and offsets per storage.
    - Compute view index bytes via `compute_view_index_bytes` and pass view metadata consistently
      (`view`, `view_id`, `view_subset_hash`).
    - Enforce per-storage range constraints aligned with `TargetLayoutGpuSink`.
    - Map failures into `ArtifactError` with the same error policy as existing materialization.

- [x] Phase B: Deterministic packing + publish path (still no new RPCs)
  - [x] Milestone B1: Packing modes in the SDK.
    - Append mode (default) for vLLM binding order.
    - Plan-first mode that sorts tensors deterministically (e.g., canonical index order) and pre-computes layout.
  - [x] Milestone B2: Optional publish after commit using existing registration.
    - Register storages as VRAM regions (if not already) and publish a `VRAM_LEASED` (LIP) artifact using
      begin/feed/commit registration flow.
    - Return publish metadata (artifact id, TTL, storage ids, view id/subset hash).
    - Avoid dependence on Global Store view routing; publishing uses canonical identifiers and view metadata only.

- [ ] Phase C: Unified readiness semantics (system-level follow-on, out of Phase A/B scope)
  - [ ] Milestone C1: Define a single wait/cancel/status surface that can back deferred loaders and future async ops.
  - [ ] Milestone C2: Migrate `ConfirmReplica` and new deferred paths to the unified surface.

- [ ] Phase D: First-class view reuse (system-level follow-on)
  - [ ] Milestone D1: Implement Global Store view-aware routing.
  - [ ] Milestone D2: Promote `TargetLayout.TENSOR_TABLE` to a first-class layout mode.

# Tasks (Cross-Cutting)

- [x] SDK wiring: add type hints, error mapping, and structured metrics parity with existing materialize flows.
- [x] Daemon/core review: verify `MaterializeIntoTarget` already accepts view/subset metadata needed by deferred loader.
- [x] Docs: update `docs/internals/model-loading.md` with deferred loader flow and mention target layout constraints.

# Acceptance Checks

- Deferred loader returns CUDA tensors immediately and performs no IO until `commit()`.
- `commit()` executes exactly one daemon data-plane call and fills the arena correctly.
- Slice validation matches canonical index metadata and rejects invalid narrow bounds.
- Optional publish produces a usable artifact id that can be P2P-materialized by another node.

# Test / Rollout / Backout

- Tests:
  - Added C++ `ViewPlanner` subset ordering coverage and daemon validation for ordered full selections.
  - Added Python deferred loader tests (placeholder layout, invalid slices, commit order) gated by CUDA availability.
  - Optional integration test covers publish path and P2P materialization on a second daemon.
- Rollout: add `Artifact.deferred_loader` as an opt-in API; no changes to existing `Artifact.tensor*` semantics.
- Backout: remove the new SDK entry point and leave daemon/core paths untouched.

# Risks & Tracking

- Target layout constraints: multi-storage boundaries must be respected or materialization will fail.
- Memory footprint: arena allocation size needs guardrails and predictable limits in append mode.
- View routing gap: publish path must tolerate canonical-only routing in Global Store.
