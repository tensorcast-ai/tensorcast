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
- `artifact.tensor_into(name, target_tensor, device=None)` materializes a single
  tensor directly into the provided buffer. Only the requested tensor must be
  present in the target mapping, so multi-tensor artifacts no longer require
  pre-allocating placeholders for every entry when using the helper.
- `tensor_dict_into` / `tensor_into` region-backed paths stream into registered
  CUDA regions via `MaterializeIntoTarget`, supporting view specs, packed
  subset selection (`tensor_names`), and ordered multi-storage layouts. For
  non-identity views the SDK resolves and sends a deterministic `view_id` in
  the `TargetLayout`. `region_backed_mode` (`auto`/`require`/`disable`) controls
  fallback behavior.
- Handles retain whichever identifiers are available (`artifact_id`, `key`,
  `disk_path`). At least one identifier is required when instantiating or
  rehydrating a handle, but resolved handles may keep both `artifact_id` and
  `key` so cloning (`with_fallback`) and serialization (`to_dict`/`from_dict`)
  continue to work.
- `tensorcast.from_disk(path)` / `Store.from_disk(path)` resolve disk-backed
  artifacts via the daemon `ResolveArtifactFromDisk` RPC. The daemon validates
  descriptor multihashes when `verify_checksums=True`, returns canonical
  `canonical_index_bytes` + `generation`, and seeds `ArtifactCache` while
  binding a disk-first `FallbackOptions` so materialization prefers the local
  files without extra resolver RPCs.
  Set `verify_checksums=False` on `FallbackOptions.for_disk(...)` to allow
  descriptor-free local development; checksum validation (and descriptor
  requirements) remain the default in production.
- Handles are tied to the originating `Store` lifecycle. After `Store.close()`
  (or `release()` on the handle), materialization raises
  `ArtifactError(status_code="FAILED_PRECONDITION")` while cached metadata
  remains readable for debugging.
- `with_fallback(...)` clones a handle with different fallback hints; eager
  `get*` APIs remain unchanged.

## View Composition

- `artifact.view(...)`, `.subset(...)`, and `.slice(...)` return derived
  handles with composed view specs. Parent/child handles keep independent locks
  and metadata caches so repeated calls avoid daemon lookups.
- `artifact.view_builder()` exposes a fluent builder for chaining multiple view
  operations before building a child handle.
- Nested slice operations are collapsed into a single narrow so storage offsets
  are computed exactly once in the derived view (no double-application of the
  parent slice).

## Batching, Async, and Prefetch

- Use `with artifact.batch(device="cuda:0")` to coalesce multiple tensor fetches
  into a single RPC while keeping sync semantics.
- Async consumers can `await artifact.tensor_async(...)` or
  `await artifact.tensor_dict_async(...)`; calls are coalesced by the process
  `MaterializationBatcher` (1ms window) on the store event loop.
- `tensor_dict_into_async` / `tensor_into_async` cancellation is best-effort:
  once a region-backed RPC is in-flight, `cancel()` may return `False`; streaming
  materialization remains cancellable before the RPC boundary.
- `artifact.prefetch(device=...)` issues background materialization
  (`wait_for_completion=False`) and returns a tuple of
  `(prefetched_handle, PrefetchTicket)`. The returned handle is a clone whose
  fallback carries the ticket’s `replica_uuid`, allowing callers to opt into the
  hint without mutating the original handle. Use the ticket to `wait()` or
  `cancel()` the staged replica before materializing tensors from the cloned
  handle.

## Fallback Preferences

`FallbackOptions` now supports explicit source preferences:

- `prefer="auto"` (default) — daemon chooses optimal source
- `prefer="local"` — disallow P2P and disk unless an explicit `disk_path` is provided;
  daemon enforces this via `SourcePolicy` gating
- `prefer="p2p"` — allow remote transfer
- `prefer="disk"` — prioritize disk fallback; pass `disk_path` or rely on key→path
  mapping

Compatibility flags `prefer_disk` and `allow_p2p` continue to work; setting
`replica_uuid` hints the daemon to reuse a prefetched replica.

## Feature Toggles

- `TENSORCAST_STORE_ENABLE_BATCHER` (default: enabled) — disable to bypass the
  async batcher and route `tensor_async` through direct fetches.
- `TENSORCAST_STORE_ENABLE_PREFETCH` (default: enabled) — disable to prevent
  `Artifact.prefetch()` from issuing background materialization.

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
- Disk lookups honor cache entries keyed by `disk_path` (not just
  `artifact_id`) to bypass resolver RPCs; mismatched `disk_path` or `generation`
  values trigger cache invalidation and a fresh daemon fetch.
- Disk resolution (`ResolveArtifactFromDisk`) seeds the cache with
  `canonical_index_bytes`, `generation`, and `disk_path` so repeated
  `from_disk` calls avoid extra daemon RPCs and preserve generation metadata.
- Metadata hydration (`_ensure_metadata`) applies `_set_metadata` while holding
  the artifact’s reentrant lock so concurrent callers never observe partially
  populated canonical metadata.
