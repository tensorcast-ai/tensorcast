# TensorCast Python API

The `tensorcast.api` package exposes the high-level registration and loading
helpers that SDK integrations use during artifact lifecycle management.

## Store module layout

Design 0037 refactored `tensorcast.api.store` into a structured subpackage:

- `store/types.py` and `store/handles.py` keep immutable dataclasses and handle wrappers importable from `tensorcast.api.store`.
- `store/runtime.py` owns the process-wide daemon client, session record writes, key/capability caches, and fork-aware executor lifecycle.
- `store/registration.py` and `store/materialization.py` orchestrate register/put/view and artifact materialization flows with shared retry/error mapping.
- `store/views.py` keeps view-spec parsing, placement defaults, and canonical index lookups isolated from the pipelines.
- `store/async_ops.py` centralizes async helpers (`ArtifactFuture`, `TrackedExecutor`) so cancellation/confirm semantics are consistent across verbs.
- `store/__init__.py` is the public façade; it now eagerly wires runtime/registration/materialization without monkeypatch/override hooks or lazy rebuilds.

Module-level helpers (`tensorcast.api.store.register`, `get`, etc.) reuse a process-scoped `Store`. If you close that store (or invoke `shutdown_process_store()`), the next helper invocation transparently reinitializes a fresh instance instead of reusing the closed handle.

## Artifact Handles & Metadata Cache

- `tensorcast.artifact(...)` / `Store.artifact(...)` provide lazy handles that
  expose metadata (`tensor_names`, `tensor_meta`, `describe`) and selective
  materialization (`tensor_dict(names=...)`, `tensor(name, ...)`, `tensor_into(...)`) as the canonical retrieval surface.
- Handles accept whichever identifiers are available (`artifact_id`, key, or
  disk path). At least one identifier is required, but resolved handles keep all
  known hints so `with_fallback(...)` and `to_dict()/from_dict()` remain valid
  even after key resolution.
- Handles are bound to the originating `Store`; materialization after
  `Store.close()` or `Artifact.release()` raises
  `ArtifactError(status_code="FAILED_PRECONDITION")` while cached metadata
  remains readable.
- The runtime now maintains an `ArtifactCache` for canonical indices. Defaults
  can be tuned with `TENSORCAST_STORE_INDEX_CACHE_TTL_SECONDS` (600s) and
  `TENSORCAST_STORE_CACHE_MAX_ENTRIES` (1000). Metrics are emitted for cache
  hits, misses, evictions (`reason=ttl|lru`), and invalidations (`reason=*`).

## Materialization v2 (descriptor streaming)

- `MaterializationPipeline` streams `TensorPayloadDescriptor` + tensor pairs from the daemon v2 surface (`tensorcast.proto.daemon.v2`) by default; the v1 path and `TC_ENABLE_MATERIALIZE_V2` flag have been removed.
- Selective fetch (`tensor_names`) trims descriptors and canonical index bytes; iterator cancellation still routes through `_release_materialized` so CUDA IPC handles are unmapped even on early exit. `tensor_dict_into` / `tensor_into` copies consume descriptors directly without building intermediate dicts.
- Telemetry attaches per-descriptor attributes (`tc.tensor.count`/`tc.tensor.bytes`) and a subset/full selector to the materialization span, and the client metrics surface attaches the same selector to latency/error/retry series.
- Disk fallbacks are forwarded to the daemon through `DiskFallbackHint` + `SourcePreference=PREFER_DISK` (including the `verify_checksums` hint) so all disk reads stay in the daemon data path.

## Device requirements

`Artifact.tensor*`/`tensor_dict*` default to materializing replicas onto CUDA
devices. On hosts without `torch.cuda.is_available()`, callers must either
provide disk fallback options (so bytes can be streamed from disk) or explicitly
select a CPU target alongside those fallback settings. Otherwise the API raises
`ArtifactError("CUDA device required for retrieval")` immediately, keeping
callers from running deeper into retry loops that can never succeed.

## View Retrieval

`Artifact.view(...).tensor*` defaults to executing transforms on the daemon so
transpose views return buffers in the expected orientation. Client-side
execution is intentionally disabled until a local transform engine exists; the
pipeline still accepts `placement="CLIENT"` explicitly for forward
compatibility.

## View Registration

`Store.register_view()` mirrors retrieval semantics so trainers can upload only
the bytes required for a narrow or transpose view while the daemon rebuilds the
canonical artifact. Key behaviours:

- Placement defaults to `SERVER` for pure narrow views and `CLIENT` when any
  transpose is present. Users can override via the `placement` keyword.
- When the daemon lacks GPU support for a server-side transpose it returns a
  `FAILED_PRECONDITION` status. The client surfaces a clear `ArtifactError`
  instructing callers to retry with `placement="CLIENT"`.
- `allow_partial=True` permits uploading subsets of canonical byte-space when
  composing shards; the commit response includes `canonical_ranges` describing
  the covered offsets. These ranges are forwarded to Global Store so metrics can
  expose the remaining backlog for each view.

The API returns a `RegisteredArtifact` whose `registration_result` carries the
view identifier, canonical coverage ranges, and variant hash for downstream
automation.
See the [Variant View Registration Telemetry](../../docs/architecture/p2p-transfer-strategies.md#variant-view-registration-telemetry) guide for the full daemon ↔ Global Store ↔ SDK flow.

## Tensor Storage Graph Helper

`build_tensor_storage_graph()` inspects a `dict[str, torch.Tensor]` and returns
a `TensorStorageGraph` containing:

- **storages** – one `StorageEntry` per unique `torch.Storage` (deduped by
  device id, base pointer, and storage length).
- **aliases** – per logical tensor metadata (shape, stride, dtype, storage
  offset, logical byte length) keyed by tensor name.
- **tensor_meta_index / tensor_source_index** – canonical metadata reused by
  both disk persistence and lease-in-place registration.

Invariants guaranteed by the helper:

- All CUDA tensors in the input must reside on the same device; the helper
  records the device id for each storage and raises on mismatches.
- Storage identifiers are deterministic and stable for the lifetime of the
  process, enabling clients to sort and reference storage groups.
- The aliases map preserves every tensor key from the input dict; consumers
  can rebuild canonical index JSON by combining alias metadata with the device
  offsets supplied by layout planners.

Clients must invoke this helper before feeding lease segments so they can send
the deduplicated storage table alongside per-tensor aliases. The daemon uses
these structures to rebuild canonical indices and avoid repeated CUDA IPC
opens for shared storages.
