# TensorCast Python API

The `tensorcast.api` package exposes the high-level registration and loading
helpers that SDK integrations use during artifact lifecycle management.

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
