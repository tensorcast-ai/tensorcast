# TensorCast Store SDK

The `tensorcast.api.store` package exposes registration, retrieval, and lazy
artifact handle surfaces. This module builds on the process-wide `Store`
singleton (`tensorcast.store()`) so callers can reuse daemon sessions without
managing clients manually.

## Artifact Handles

- `tensorcast.artifact(...)` and `Store.artifact(...)` return a lazy `Artifact`
  bound to the process `Store`. Handles support metadata accessors
  (`tensor_names`, `tensor_meta`, `describe`), existence checks (`exists`), and
  selective materialization via `tensor_dict(names=...)` and `tensor(name, ...)`.
- Handles are tied to the originating `Store` lifecycle. After `Store.close()`
  (or `release()` on the handle), materialization raises
  `ArtifactError(status_code="FAILED_PRECONDITION")` while cached metadata
  remains readable for debugging.
- `with_fallback(...)` clones a handle with different fallback hints; eager
  `get*` APIs remain unchanged.

## Metadata Cache

- The process runtime owns an `ArtifactCache` that stores canonical index bytes,
  parsed indices, and disk hints keyed by `artifact_id`. Cache entries expire by
  TTL and obey an LRU bound.
- Environment defaults:
  - `TENSORCAST_STORE_INDEX_CACHE_TTL_SECONDS=600` (set `<=0` to disable)
  - `TENSORCAST_STORE_CACHE_MAX_ENTRIES=1000` (set `<=0` to disable)
- Cache metrics are emitted as:
  - `tc_store_artifact_cache_hits_total`
  - `tc_store_artifact_cache_misses_total`
  - `tc_store_artifact_cache_evictions_total` (dimensions: `reason=ttl|lru`)
  - `tc_store_artifact_cache_invalidations_total` (dimension: `reason`)
- Invalidation hooks run after registration, deregistration, and materialization
  errors (`NOT_FOUND`/`FAILED_PRECONDITION`) to keep key→artifact mappings and
  cached indices consistent.
