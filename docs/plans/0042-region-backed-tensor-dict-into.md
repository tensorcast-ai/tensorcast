---
slug: 0042-region-backed-tensor-dict-into
title: Plan - Region-Backed tensor_dict_into
links:
  design: ../designs/0042-region-backed-tensor-dict-into.md
areas: ["core", "daemon", "sdk", "proto"]
related_code:
  - daemon/service/controllers/materialization_controller.cc
  - daemon/state/ipc_region_registry.{h,cc}
  - daemon/service/materialize_into_target_validation_test.cc
  - core/store/store_engine.{h,cc}
  - core/store/runtime/ingestion/materialization_facade.{h,cc}
  - core/store/materialization/dataplane/runtime/pump.{h,cc}
  - core/store/materialization/dataplane/sources/segment_plan_source.{h,cc}
  - core/store/materialization/dataplane/view/view_planner.{h,cc}
  - core/store/materialization/dataplane/view/view_plan_source.{h,cc}
  - core/store/materialization/dataplane/sinks/gpu_memory_sink.{h,cc}
  - proto/tensorcast/daemon/v2/store_daemon.proto
---

# Objective

Implement Phase 2+ support for region-backed `tensor_dict_into` (`MaterializeIntoTarget`): view-indexed layouts
(`INDEX_KIND_VIEW` + `view/view_id`), packed subset selection (`tensor_names` / `view_subset_hash`), multi-storage
COALESCED (ordered concatenation) via a new TargetLayout-backed GPU sink, and (optionally) external-target verification.
This work also requires cross-cutting identity hardening (deterministic `view_id` resolution for non-identity views and
unambiguous subset-hash encoding) so Phase 2+ is safe and becomes a reliable lower-layer for
`docs/designs/0052-deferred-slice-materialization.md`.
These lower-layer primitives are also the dependency for daemon-owned placeholder/deferred fill sessions
(`docs/designs/0052-deferred-slice-materialization.md`).

# Current State & Grounding

- The RPC exists and Phase 1 is wired but intentionally rejects selection and views:
  - controller gating: `daemon/service/controllers/materialization_controller.cc`
  - existing validation coverage: `daemon/service/materialize_into_target_validation_test.cc`
- Core execution entrypoint exists and already streams into a single mapped target:
  - `core/store/store_engine.{h,cc}`
  - `core/store/runtime/ingestion/materialization_facade.{h,cc}`
- View planning / view execution primitives exist and are reused by other materialization flows:
  - `core/store/materialization/dataplane/view/view_planner.{h,cc}`
  - `core/store/materialization/dataplane/view/view_plan_source.{h,cc}`
- COALESCED canonical planning defines PAD=0 semantics (canonical ByteSpace is already “normalized”):
  - `core/store/materialization/dataplane/sources/segment_plan_source.{h,cc}`
- Existing GPU sink is effectively single-storage:
  - `core/store/materialization/dataplane/sinks/gpu_memory_sink.{h,cc}`
- Region mapping + ownership/TTL is tracked by the daemon:
  - `daemon/state/ipc_region_registry.{h,cc}`
- Cross-cutting identity issues must be fixed before enabling view/subset into-target:
  - a non-identity `view` must never execute without a resolved `view_id` (see `docs/designs/0016-artifact-view-v1.md`)
  - `view_subset_hash` / `ViewSubset.subset_hash` must be standardized as raw digest bytes (never UTF-8/hex-string bytes)

## Constraints

- No ad-hoc environment variables; any new toggles must go through the unified runtime config system
  (`docs/designs/0004-unified-runtime-config.md`).
- Avoid blocking shared executors; any bounded waits/IO belong on a blocking executor (root `AGENTS.md`).
- After any `.proto` edits, regenerate code via `bash tools/build_proto_python.sh`.

# Phases & Milestones

- [ ] Phase 0: Identity hardening (prerequisite for Phase 1 and for 0052 reuse)
  - [ ] Milestone 1: Make `view_id` resolution authoritative and non-optional for non-identity views:
    - when a request carries `view` but no `view_id`, compute a deterministic `view_id` (per `docs/designs/0016-artifact-view-v1.md`)
    - propagate the resolved `view_id` into `VariantIdentity.view_id` so core `ReplicaKey` disambiguation and variant verification apply
    - reject non-identity views that still lack a resolved `view_id` (fail fast to avoid canonical/variant collisions)
    - add regression tests that a view-spec request does not reuse/collide with the canonical `ReplicaKey`
  - [ ] Milestone 2: Standardize subset hash encoding end-to-end:
    - define the canonical subset-hash algorithm (sorted+unique `tensor_names` → SHA-256 digest bytes)
    - ensure SDK sends raw digest bytes in `view_subset_hash` and treats response `ViewSubset.subset_hash` as raw digest bytes
    - add round-trip tests that cover request/response stability and caching labels

- [ ] Phase 1: Enable view + subset selection for region-backed targets
  - [ ] Milestone 1: Relax daemon gating and make validation phase-aware:
    - accept `index_kind=VIEW` + `view/view_id` for `MaterializeIntoTarget`
    - require a resolved `view_id` for non-identity `view` specs and validate it matches `target_layout.view_id`
    - accept `tensor_names` / `view_subset_hash` when the request defines a packed subset view ByteSpace
    - validate tensor table entries against the *selected* index bytes (canonical or view)
  - [ ] Milestone 2: Implement selected-index resolution in core `materialize_into_target`:
    - canonical only (Phase 1)
    - view-indexed (Phase 2+)
    - packed subset view index bytes (Phase 2+), including PAD=0 semantics
  - [ ] Milestone 3: Choose a safe packed-subset contract (required before shipping subset-packed into-target):
    - Option A (preferred): add a preflight planning RPC returning `view_index_bytes`, `view_subset`, and `logical_total_size` without writing
    - Option B: centralize packing in one implementation (C++ core + Python binding) and add compatibility tests against the daemon
  - [ ] Milestone 4: Populate response metadata for selection:
    - always return `canonical_index_bytes`
    - when selection is applied, return `view_index_bytes` + `view_subset`
  - [ ] Milestone 5: Extend `//daemon:materialize_into_target_validation_test` with VIEW + subset scenarios (fake CUDA)

- [ ] Phase 2: Multi-storage COALESCED sink (ordered concatenation)
  - [ ] Milestone 1: Implement a TargetLayout-backed GPU sink that supports ordered concatenation across multiple region
    windows (implements `AsyncPositionedSink` so `pump_ranges` overlaps IO + H2D).
  - [ ] Milestone 2: Extend daemon validation to accept multi-storage only when storages satisfy ordered concatenation and
    span `logical_total_size` for the selected ByteSpace.
  - [ ] Milestone 3: Add targeted sink mapping tests (boundary + cross-storage) and an end-to-end daemon test that writes
    into multiple registered regions.

- [ ] Phase 3: Optional external-target verification
  - [ ] Milestone 1: Add a unified-config toggle for external-target verification (off by default) and expose metrics for
    verification enabled/disabled.
  - [ ] Milestone 2: Implement external-target verification using plan-based hashing utilities:
    - canonical: `compute_data_multihash_from_gpu_plan` (`core/store/materialization/dataplane/sources/segment_plan_source.{h,cc}`)
    - view/subset: define and implement the equivalent “selected ByteSpace hash” contract (reusing view/subset identity)
  - [ ] Milestone 3: Add tests for verification failures and ensure failures poison the target region (no silent partial
    success).

- [ ] Phase 4: Hardening + performance (optional follow-ons)
  - [ ] Milestone 1: Cache SegmentPlan/ViewPlan artifacts keyed by `(generation, logical_layout_hash, selection_hash)`
    (exclude physical binding like region ids).
  - [ ] Milestone 2: Add per-request IPC mapping reuse to avoid repeated `cudaIpcOpenMemHandle` for the same region.
  - [ ] Milestone 3: Add a fast path that copies from an existing local replica into the external target once a range-based
    GPU→GPU helper exists (avoid disk/P2P re-stream).

# Acceptance Checks

- Phase 2+ view-indexed and subset-packed requests succeed with no daemon-owned replica allocation and bounded writes into
  the provided regions.
- Non-identity view requests always execute with a resolved `view_id` (no canonical/variant `ReplicaKey` collisions).
- Subset hashes round-trip consistently as raw digest bytes (SDK↔daemon), and are safe to use for cache keys/metrics.
- Multi-storage COALESCED writes are correct at storage boundaries (no off-by-one; no cross-storage writes).
- PAD=0 semantics hold for the selected ByteSpace (either via explicit zero writes or proven pre-zeroed destinations).
- When external-target verification is enabled, corrupt/mismatched writes fail and the region is poisoned; when disabled,
  the explicit “verification skipped” metric increments.

# Test / Rollout / Backout

- Tests (C++):
  - `bazel test //daemon:materialize_into_target_validation_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
  - Add/extend core unit tests for the new multi-storage sink under fake CUDA where possible.
- Rollout:
  - Keep `region_backed_mode=auto` default behavior; Phase 2+ adds capabilities but preserves fallback behavior.
  - Gate external-target verification behind unified config (opt-in).
- Backout:
  - Disable Phase 2+ selection/multi-storage gates in the daemon (returning to Phase 1 constraints) without changing the
    RPC surface.
  - Disable external-target verification via config.
