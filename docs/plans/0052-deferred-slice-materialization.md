---
slug: 0052-deferred-slice-materialization
title: Plan - Deferred Slice Materialization for vLLM
links:
  design: ../designs/0052-deferred-slice-materialization.md
---

# Objective

Enable vLLM’s “meta-init + per‑param weight_loader” integration by providing a TensorCast `DeferredLoader` that:

- returns placeholder CUDA tensors immediately (views into a client-owned arena),
- fills them with a single daemon data-plane call at `DeferredLoader.commit()` (via `MaterializeIntoTarget`),
- optionally publishes the produced slice-artifact for P2P reuse using existing registration primitives.

# Current State & Grounding

- Region-backed ingestion is implemented (`MaterializeIntoTarget` + `TargetLayout`) and already supports multi-storage ordered concatenation.
- View planning/execution primitives exist (`ViewPlanner`, `ViewPlanSource`, `pump_ranges`).
- Unified lifecycle/leases exist for daemon-exported handles and registration TTLs; avoid inventing new placeholder lease systems.
- Global Store view routing is not implemented (view transport falls back to canonical), so Phase 1 publishing must not depend on view-aware routing.

# Phases & Milestones

## Phase A — SDK-first deferred loader (no new daemon RPCs)

- [ ] Milestone A1: Add `Artifact.deferred_loader(device=...)` API and `DeferredLoader` implementation skeleton.
- [ ] Milestone A2: Implement client-owned arena allocation strategy (single allocation first; multi-storage follow-on).
- [ ] Milestone A3: Implement `DeferredLoader.tensor(name, slice=...)`:
  - validate against canonical index bytes
  - allocate an arena view tensor and record the slice request
- [ ] Milestone A4: Implement `DeferredLoader.commit()` using a single `MaterializeIntoTarget` call:
  - build `TargetLayout` over arena storages
  - encode selection as view/subset (narrow-only; packed + PAD=0)
  - ensure commit respects target storage boundary constraints (per-storage ranges, consistent with existing pipeline)

## Phase B — Determinism and publish-friendly mode

- [ ] Milestone B1: Add explicit packing modes:
  - append mode (default; matches vLLM immediate binding)
  - plan-first mode (deterministic packed layout; required for stable publishing)
- [ ] Milestone B2: Implement optional publication after commit via existing registration:
  - register arena storages as VRAM regions (`RegisterVramRegion`)
  - publish as `VRAM_LEASED` (LIP) with TTL via `Begin/Feed/CommitRegisteredArtifact`
  - return a publish result (`artifact_id`, TTL, and any debug metadata)

## Phase C — Unified readiness semantics (system-level follow-on)

- [ ] Milestone C1: Implement daemon-side `QueryReplicaStatus` + `ReleaseReplica` (or an equivalent unified ticket API) so async readiness is not special-cased via `ConfirmReplica`.
- [ ] Milestone C2: Update SDK so any future async/deferred surfaces share the same wait/cancel/status mechanism.

## Phase D — First-class variant reuse (system-level follow-on)

- [ ] Milestone D1: Implement Global Store view-aware routing (replace canonical fallback for `request_view_transport`).
- [ ] Milestone D2: Promote `TargetLayout.TENSOR_TABLE` to a first-class mode to reduce reliance on packed linear ByteSpaces for future complex layouts.

## Tests + Docs

- [ ] Python tests (fake CUDA): `DeferredLoader` placeholder semantics + commit barrier correctness.
- [ ] (Optional) integration tests: publish via `VRAM_LEASED` and P2P materialize on a second daemon.
- [ ] Update `docs/internals/model-loading.md` to document deferred loader semantics and how it relates to `tensor_dict_into` and publishing.

# Rollout / Backout

- Roll out behind a new `Artifact` method (`Artifact.deferred_loader`) without changing existing `Artifact.tensor*` semantics.
- Back out by removing the SDK entry point and keeping `MaterializeIntoTarget` unchanged; no schema changes required for Phase A/B.
