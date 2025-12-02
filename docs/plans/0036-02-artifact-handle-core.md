---
slug: 0036-artifact-handle-core
title: Plan – Lazy Artifact Handle Phase 2 (Handle + Metadata Cache)
links:
  design: ../designs/0036-02-artifact-handle-core.md
areas: ["sdk"]
related_code:
  - tensorcast/api/store/__init__.py
  - tensorcast/api/store/runtime.py
  - tensorcast/api/store/materialization.py
  - tensorcast/api/store/views.py
  - tensorcast/api/store/common.py
  - tensorcast/api/store/cache.py
  - tensorcast/api/store/artifact.py
  - tests/python/api/test_materialization_pipeline_v2.py
---

# Objective

Ship the Artifact handle and metadata cache defined in the Phase 2 design so SDK users can construct lazy handles (`tc.artifact(...)`) that expose metadata and selective materialization without changing the existing eager `get*` APIs. This plan grounds the work in today’s eager-only SDK, adds cache + lifecycle plumbing that Phase 3 (`0036-03-lazy-artifact-handle.md`) will build on for views/batching/prefetch, and keeps daemon/proto changes out of scope (handled in Phase 1).

# Current State & Grounding
- `tensorcast/api/store/__init__.py` only exposes eager retrieval (`get`, `get_view`, `get_into*`) that always materialize full `dict[str, Tensor]`; there is no `Artifact` handle or factory entry point.
- `StoreRuntimeContext` (`tensorcast/api/store/runtime.py`) maintains a key→artifact_id/disk path cache with `TENSORCAST_STORE_KEY_CACHE_TTL_SECONDS` (30s default) but has no canonical index cache; `_after_fork_child` rebuilds the client/executor only.
- `MaterializationPipeline` (`tensorcast/api/store/materialization.py`) already uses the streaming v2 pipeline and accepts `tensor_names`, but callers always request the full payload; metadata parsing happens per call with no reuse.
- `ViewOrchestrator` (`tensorcast/api/store/views.py`) resolves view inputs by fetching canonical index bytes from the daemon on every request; repeated view calls re-download the same index.
- Tests under `tests/python/api/test_materialization_pipeline_v2.py` cover streaming subset materialization and iterator cleanup; there is no coverage for handle lifecycles, metadata caching, or store shutdown interactions.
- Phase 3 design (`docs/designs/0036-03-lazy-artifact-handle.md`) assumes a stable handle + cache foundation to layer views, batching, and async/prefetch; that foundation does not exist yet.

# Phases & Milestones

- [x] **Phase 1: Runtime cache foundation**
  - [x] Introduce `ArtifactCache` with TTL + LRU limits (`TENSORCAST_STORE_INDEX_CACHE_TTL_SECONDS`, `TENSORCAST_STORE_CACHE_MAX_ENTRIES`), storing `{artifact_id, disk_path, canonical_index_bytes, parsed_index, generation, expires_at}`.
  - [x] Wire cache lifecycle into `StoreRuntimeContext` (construction, `close()`, `_after_fork_child`, and an explicit `invalidate_artifact` helper) and emit hit/miss/eviction metrics via `store_metrics`.
  - [x] Add `get_artifact_index_cached`, `cache_artifact_index`, `invalidate_artifact` APIs (thread-safe), and clear `_key_cache` on invalidations.

- [x] **Phase 2: Artifact handle surfaces + selective materialization**
  - [x] Add `tensorcast/api/store/artifact.py` implementing the Phase 2 state machine (`key=` / `artifact_id=` / `disk_path=` constructors converge to Identified → Indexed), metadata accessors (`tensor_names`, `tensor_meta`, `describe`, `exists`, `is_valid`), `with_fallback`, `release`, and `to_dict`/`from_dict` serialization (with generation and TTL refresh) with double-checked locking.
  - [x] Expose factories `Store.artifact(...)`, `tensorcast.artifact(...)`, and a guarded `tensorcast.from_disk(path)` stub that binds handles to the process Store via `weakref`, preserves eager APIs unchanged, and propagates store-closed errors as `ArtifactError(status_code="FAILED_PRECONDITION")`.
  - [x] Implement handle-driven selective materialization helper (e.g., `_materialize_subset`) that validates names against cached indices and forwards `tensor_names` into `MaterializationPipeline`; open the pipeline helper internally without changing eager `get*` surfaces.
  - [x] Define `TensorMeta` / `ArtifactDescriptor` dataclasses derived from canonical indices so metadata surfaces stay stable for Phase 3 view composition.

- [x] **Phase 3: Cache integration & invalidation hooks**  *(depends on Phase 2 selective fetch foundation)*
  - [x] Populate `ArtifactCache` by reusing `DaemonCtl.get_artifact_index_by_id` (and disk hints from key resolution); share parsed indices with `ViewOrchestrator` to eliminate duplicate daemon fetches and surface cache hit/miss/TTL/eviction metrics.
  - [x] Wire invalidation triggers: successful register/put/register_view overwrite paths, `deregister_artifact`, and `MaterializationPipeline` NOT_FOUND/FAILED_PRECONDITION errors should call `invalidate_artifact` and clear `_key_cache` entries as needed.
  - [x] Integrate `ViewOrchestrator.resolve_view_inputs` with `ArtifactCache` (hit before daemon RPC) and record cache metrics.
  - [x] Ensure observability spans/metrics record cache outcomes and subset materialization selectors while keeping existing `get*` behavior and telemetry unchanged.

- [x] **Phase 4: Validation, docs, and readiness for Phase 3**
  - [x] Add tests: `tests/python/api/test_artifact_cache.py` (TTL expiry, LRU eviction, fork reset, generation carry-through, invalidation hooks), `tests/python/api/test_artifact_handle.py` (state transitions, `release` idempotency, subset fetch, serialization TTL refresh, store-close invalidation), `tests/python/api/test_artifact_exists.py`, `tests/python/api/test_artifact_tensor_subset.py`, and extend `test_materialization_pipeline_v2.py` with a handle-driven subset regression and generation assertions.
  - [x] Update SDK docs (`tensorcast/api/README.md` and a new `tensorcast/api/store/README.md` if needed) plus cross-links in `docs/designs/0036-03-lazy-artifact-handle.md` to describe the handle surface, cache env vars, lifecycle semantics, and selective materialization baseline.
  - [x] Document env vars and metrics in `docs/architecture/architecture-overview.md` and ensure Phase 3 design references Phase 2 selective fetch + cache baseline as implemented.

# Tasks
- Define `tensorcast/api/store/cache.py` with TTL + LRU eviction, hit/miss/eviction counters, and thread-safe access; add an `ArtifactCacheEntry` struct that carries canonical index bytes, parsed index, generation, and disk hint.
- Extend `StoreRuntimeContext` with cache accessors (`get_artifact_index_cached`, `cache_artifact_index`, `invalidate_artifact`) and reuse them inside `ViewOrchestrator.resolve_view_inputs` to stop re-fetching indices per call.
- Implement `Artifact` with a per-handle lock and `weakref` to `Store`; guard metadata fetch via double-checked locking, validate constructor inputs (exactly one of key/artifact_id/disk_path), and surface `exists()` without materializing tensors.
- Add handle-driven selective materialization helper that validates names against cached indices, forwards `tensor_names` into `MaterializationPipeline`, and ensures `MaterializationPayload` carries `generation`.
- Update `tensorcast/api/store/__init__.py` exports and process-store helpers to create/bind handles while keeping eager `get*` behavior untouched; plumb fallback options, disk hints, and generation through factories into the handle.
- Wire invalidation triggers in registration/deregistration/materialization error paths; ensure `_key_cache` entries are cleared on deregister and overwrite.
- Add test fixtures/mocks for daemon index fetches and materialization subsets so cache and handle tests run under `uv run pytest` without daemon dependencies; document new env vars and metrics.

# Test / Rollout / Backout
- **Unit/Integration**: `uv run pytest tests/python/api/test_artifact_cache.py`, `uv run pytest tests/python/api/test_artifact_handle.py`, `uv run pytest tests/python/api/test_artifact_tensor_subset.py`, `uv run pytest tests/python/api/test_artifact_exists.py`, `uv run pytest tests/python/api/test_materialization_pipeline_v2.py`.
- **Rollout**: New APIs are additive; keep eager `get*` paths unchanged while enabling handles in docs/examples. Monitor cache metrics (hits/misses/evictions) and materialization subset telemetry in staging before promoting.
- **Backout**: If issues surface, drop handle exports from `tensorcast/api/store/__init__.py` and disable cache usage in `ViewOrchestrator`/materialization while retaining the existing eager APIs; clear caches on rollback to avoid stale metadata.

# Risks & Tracking
- **Concurrency correctness**: Double-checked locking around metadata and cache updates could deadlock or race; mitigate with targeted unit tests and careful lock scoping in handles + runtime.
- **Cache staleness/memory pressure**: TTL/LRU configuration mistakes could serve stale indices or over-consume memory; mitigate with clear env defaults, metrics, and `invalidate_artifact` hooks when artifacts are deregistered or leases expire.
- **Lifecycle coupling**: Handles rely on process Store weakrefs; misuse after `Store.close()` must raise deterministic errors without leaking executor threads or daemon channels.
- **Disk/key resolution drift**: Disk-path hints and key resolution need to stay consistent between the key cache and the new artifact cache; add coverage for disk-hinted construction even though full disk entry point ships in Phase 4.
- **Selective fetch regression risk**: Opening the `tensor_names` pipeline path for handles must not alter eager `get*` behavior; mitigate with subset-specific tests and metrics assertions.
