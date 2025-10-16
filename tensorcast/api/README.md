# TensorCast Python API

The `tensorcast.api` package exposes the high-level registration and loading
helpers that SDK integrations use during artifact lifecycle management.

## View Retrieval

`Store.get_view()` defaults to executing transforms on the daemon so that
transpose views return buffers in the expected orientation. Client-side
execution is intentionally disabled until a local transform engine exists; the
API still accepts `placement="CLIENT"` explicitly for forward compatibility.

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
