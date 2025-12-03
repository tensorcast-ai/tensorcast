---
slug: 0036-lazy-artifact-handle
title: Plan – Lazy Artifact Handle Phase 3 (Views, Batching, Prefetch)
links:
  design: ../designs/0036-03-lazy-artifact-handle.md
areas: ["sdk"]
related_code:
  - tensorcast/api/store/artifact.py
  - tensorcast/api/store/views.py
  - tensorcast/api/store/materialization.py
  - tensorcast/api/store/runtime.py
  - tensorcast/api/store/async_ops.py
  - tensorcast/api/store/cache.py
  - tensorcast/api/store/common.py
  - tensorcast/api/store/README.md
  - tensorcast/api/store/view_composer.py
  - tensorcast/api/store/batch_context.py
  - tests/python/api/test_artifact_handle.py
  - tests/python/api/test_artifact_cache.py
  - tests/python/api/test_artifact_tensor_subset.py
  - tests/python/api/test_materialization_pipeline_v2.py
---

# Objective

Deliver the Phase 3 lazy handle capabilities—view composition, batching/async materialization, and prefetch tickets—on top of the Phase 1/2 pipeline + handle foundation while keeping eager `get*` APIs stable and preparing for the Phase 4 disk variant.

# Current State & Grounding
- `tensorcast/api/store/artifact.py` implements lazy handles (metadata, selective tensor fetch, serialization) with double-checked locking; there are no `view/subset/batch/prefetch` surfaces, async tensor APIs, or per-view metadata caches. Disk-backed handles are rejected with `ArtifactError(UNIMPLEMENTED)`, and `FallbackOptions` only exposes `prefer_disk`/`allow_p2p` booleans.
- `tensorcast/api/store/views.py` resolves slices/transpose and view_id for `get_view`/registration using daemon lookups; it re-fetches canonical indices on cache miss and does not compose chained views or cache view-derived metadata per handle.
- `tensorcast/api/store/materialization.py` supports `tensor_names` filtering and streaming v2 payloads; async paths are executor-wrapped sync calls with no batching/coalescing, no replica ticket handling, and no view-spec awareness for handles beyond the resolver inputs.
- `tensorcast/api/store/runtime.py` owns `ArtifactCache` + key cache with TTL/LRU and per-daemon metrics; async infra is a single-thread `ThreadPoolExecutor` via `TrackedExecutor` with no asyncio loop or batcher thread. There is no replica ticket lifecycle, and cache invalidation does not track view metadata.
- `tensorcast/api/store/__init__.py` exports `from_disk()` but the handle path fails for disk identities; there is no `artifact_async` export or async handle factory. `FallbackOptions` lacks the `prefer` enum/`replica_uuid` wiring described in the design.
- Protos: `proto/tensorcast/daemon/v2/store_daemon.proto` includes `ReplicaTicket` fields on materialization responses but has no `QueryReplicaStatus` / `ReleaseReplica` RPCs; daemon client wrappers and Python stubs do not expose ticket polling or release.
- Tests: `tests/python/api/test_artifact_handle.py`, `test_artifact_cache.py`, and `test_artifact_tensor_subset.py` cover metadata, subset materialization, and cache TTL/LRU. There is no coverage for view composition chains, batch contexts, async coroutine APIs, prefetch tickets, or replica-aware fallback policies. `tensorcast/api/store/README.md` documents Phase 2 cache/env vars only.

# Phases & Milestones

- [x] **Phase 1: View composition surfaces**
  - [x] Add `view_composer.py` with `ViewSpecComposer`, `ViewBuilder`, and per-handle `ViewMetadataCache`; reuse `_view_ops` for validation and enforce depth/placement rules from the design.
  - [x] Extend `Artifact` with `.view()/.subset()/.slice()`/`.view_builder()` entry points that clone immutable identity/fallback state, compose view specs, and keep parent/child metadata independence while sharing canonical index bytes when available.
  - [x] Integrate view composition into `MaterializationPipeline` inputs and `Store` exports so `tc.artifact(...).view(...)` flows through the existing view resolver without duplicating daemon index fetches; document behavior deltas in `tensorcast/api/store/README.md`.

- [x] **Phase 2: Batching and async materialization**
  - [x] Implement `batch_context.py` with `BatchContext` (sync context manager) and `MaterializationBatcher` (daemon dispatch thread + coalescing window) matching the design’s threading model; ensure RPC dispatch uses the store event loop and avoids `StoreRuntimeContext._executor` contention.
  - [x] Add handle APIs `tensor_async`, `tensor_dict_async`, `artifact_async` factory, and `Artifact.batch()` that reuse selective fetch + validation, support view specs, and preserve eager `get*` semantics unchanged.
  - [x] Wire batch/async metrics (hits, coalesced count, window latency) via `store_metrics`; add cancellation/shutdown hooks so batcher threads stop on `Store.close()`/fork.

- [ ] **Phase 3: Prefetch tickets and fallback policy expansion**
  - [x] Extend `FallbackOptions` and RPC translation to include `prefer` enum, `disk` preference, checksum toggle, and `replica_uuid` plumbing; keep `prefer_disk` compatibility shim.
  - [x] Surface `Artifact.prefetch()` returning `PrefetchTicket` (with `wait_for_completion=false`) and pass tickets through subsequent materialization requests; add ticket expiry/error handling that invalidates cache + key mappings on `FAILED_PRECONDITION`/`NOT_FOUND`.
  - [x] Introduce daemon client methods for `QueryReplicaStatus`/`ReleaseReplica` (per design) and propagate to `MaterializationPipeline` so async/sync paths can poll or release prefetched replicas; gate disk-path reuse until Phase 4.
  - [x] Add proto/RPC definitions for ticket polling/release in `proto/tensorcast/daemon/v2/store_daemon.proto`, update Bazel + Buf config, and regenerate stubs via `bash tools/build_proto_python.sh`.

- [ ] **Phase 4: Validation, docs, and rollout readiness**
  - [x] Add tests for view chaining and invariants (depth limit, parent/child lifecycle), batch/async coalescing, prefetch ticket reuse/expiry, and replica-aware fallback resolution; extend existing suites instead of adding ad-hoc stubs.
  - [x] Update docs: `tensorcast/api/store/README.md`, `docs/architecture/architecture-overview.md`, and cross-links from `docs/designs/0036-03-lazy-artifact-handle.md` to reflect new APIs, env vars, and metrics; add examples for view chaining + batch usage.
  - [x] Define rollout/backout toggles (env flags or feature knobs) for batcher/prefetch paths and document operational checks (metrics/telemetry) required before enabling by default.

# Tasks
- Ground view composition in existing `ViewOrchestrator` by adding a pure composer that consumes cached canonical indices from `ArtifactCache` when available; avoid duplicate daemon RPCs for repeated view derivations.
- Create per-handle view metadata cache structs and ensure `Artifact.describe()`/`tensor_names` on derived views avoid planner recomputation while respecting `_released`/store-closed checks.
- Build batcher event-loop scaffolding (asyncio loop + dispatch thread) within `StoreRuntimeContext` or adjacent helper, ensuring fork safety and clean shutdown; plumb device/view-spec keys into batch grouping.
- Extend `FallbackOptions` + daemon request builders to support `prefer` enum and `replica_uuid`; add translation layer that preserves legacy kwargs and raises structured `ArtifactError` on invalid combinations.
- Add prefetch lifecycle helpers (issue ticket, poll, reuse, cancel) with metrics + logging, and ensure cache/key invalidation triggers fire on stale tickets or daemon-side failures; add client/daemon RPCs and regenerate Buf/Bazel outputs.
- Expand tests under `tests/python/api/` for view builder, batch/async fetch, and prefetch ticket flows; add fixture helpers for ReplicaTicket-aware stubs to keep suites daemon-independent.
- Refresh SDK docs and examples to highlight lazy handle workflows, batching, and prefetch; note compatibility with existing eager APIs and Phase 4 disk parity dependency.

# Acceptance Checks
- [ ] View composition APIs produce deterministic `ViewSpec` hashes and respect depth/placement invariants; parent/child lifecycle independence tested.
- [ ] Batcher coalesces concurrent async calls and sync context manager emits single RPC per context; shutdown/fork safety verified.
- [ ] Prefetch tickets round-trip through new RPCs and fallback policies; stale/expired tickets trigger invalidation and retry without ticket.
- [ ] Doc sync: `tensorcast/api/store/README.md` and `docs/architecture/architecture-overview.md` reflect new APIs, env/metrics, and feature flags; design cross-links updated.
- [ ] All new/updated tests (`uv run pytest tests/python/api/...`) pass and Buf/Bazel proto generation is refreshed after RPC additions.
# Test / Rollout / Backout
- **Unit/Integration**: `uv run pytest tests/python/api/test_artifact_handle.py`, `uv run pytest tests/python/api/test_artifact_tensor_subset.py`, `uv run pytest tests/python/api/test_materialization_pipeline_v2.py`, plus new suites for view builder/batch/prefetch (e.g., `uv run pytest tests/python/api/test_artifact_views.py`, `test_artifact_batching.py`, `test_artifact_prefetch.py`).
- **Rollout**: Keep eager `get*` as default; guard batcher/prefetch enabling with feature flags/env vars, monitor cache + batcher metrics, and soak in staging before turning on async exports in docs/examples.
- **Backout**: Disable batcher/prefetch flags to fall back to existing executor-based async path and selective fetch; remove new exports from `tensorcast/api/store/__init__.py` if needed while retaining handle metadata APIs. Invalidate caches on rollback to avoid ticket reuse.

# Risks & Tracking
- **Concurrency and shutdown**: Batcher threads/event loops must not race with `Store.close()`/fork; add lifecycle hooks and tests for clean teardown.
- **Cache coherence**: View-derived metadata and replica tickets risk staleness; ensure invalidation hooks fire on registration/deregistration/materialization errors and when tickets expire.
- **API churn/compatibility**: Extending `FallbackOptions` and adding async factories must preserve current eager callers; provide shims and doc updates to avoid breaking downstream code.
- **Telemetry noise**: New metrics for batcher/prefetch could inflate cardinality; cap label sets and sample where needed.
