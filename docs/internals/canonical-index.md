---
title: Canonical Index Deep Dive
description: Structural layout identity for TensorCast artifacts across disk, memory, and transport flows
areas: ["core", "daemon", "global_store", "sdk"]
---

# Canonical Index Deep Dive

This note consolidates everything in-tree about the canonical index (Index v2) that TensorCast relies on to make artifact identity, storage deduplication, and transport reproducibility work. It ties together the existing design docs (`docs/designs/0003-unified-memory-registration-avbs-lip.md`, `docs/designs/0007-content-addressed-artifact-id.md`), the API design docs (`docs/architecture/api/api-design.md`), the view semantics in `docs/architecture/artifact-views-and-retrieval.md`, the byte-range executor in `docs/internals/byte-range-mapping-and-execution.md`, the core implementation in `core/store/materialization/dataplane`, and the SDK/daemon call paths that consume it today.

```mermaid
flowchart LR
  A[User tensors<br>(PyTorch storages)] --> B[Index v2 builder<br>(SDK or daemon)]
  B --> C[Canonical index bytes]
  C --> D[Index multihash<br>(structural fingerprint)]
  C --> E[ByteRange map<br>(DATA + PAD)]
  D --> F[mi2 artifact id]
  E --> G[Transport sources<br>(disk, coalesced VRAM, LIP)]
  G --> H[Consumers<br>(daemon, clients, remote nodes)]
```

## Why We Use a Canonical Index

- **Stable structural identity.** The canonical index linearizes tensor layout—offset, size, shape, stride, dtype, storage offset—for every tensor in the artifact. The `index_multihash` over these bytes feeds the mi2 `artifact_id` (`docs/designs/0007-content-addressed-artifact-id.md`), guaranteeing that identical layouts across disk, lease-in-place (LIP), and coalesced VRAM flows collapse onto one identifier. This lets Global Store deduplicate replicas by structure alone before even looking at payload bytes.
- **Re-hydration without ambiguity.** At load time the Store Engine rebuilds canonical `ByteRangeMap`s (`core/store/materialization/dataplane/sources/byte_range_map_builder.cc`) directly from canonical index JSON/CBOR. This allows deterministic reconstruction of the Artifact Virtual Byte Stream (AVBS) and enables PAD-zeroing so that cross-device copies, padding gaps, and staged disk reloads remain consistent.
- **Source-agnostic transport.** Whether bytes originate from disk partitions, daemon-owned coalesced VRAM, or CUDA IPC handles exported by a producer, the canonical index gives every consumer the same byte ranges to hash, copy, or verify. Without it, each transport would have to encode ad-hoc metadata about views, padding, and dtype ordering, undermining the unification promised in `docs/designs/0003-unified-memory-registration-avbs-lip.md`.

## Schema and Invariants

Index v2 encodes each tensor entry as `[offset, size, shape, stride, dtype, storage_offset]`, with the following rules enforced by `core/store/materialization/dataplane/metadata/canonical_index.cc`:

- **Sorted keys.** Tensor names appear in ascending order. Builders sort explicitly (`tensorcast/api/_indices.py:135-150`, `daemon/state/lip_metadata_utils.cc:92`).
- **Alignment.** `offset` values are 8-byte aligned; builders round up coalesced layouts when deduplicating storage ranges (`tensorcast/api/_indices.py:109-132`).
- **Size semantics.** `size` is the total logical bytes occupied by the full storage, not the slice size of an alias. Multiple tensors that share a storage therefore emit identical `offset`/`size` pairs; the differentiator for views is only `storage_offset`.
- **Shape/stride normalization.** Shapes and strides are serialized as unsigned 64-bit lists. Dtypes are lowercased internally to maintain deterministic ordering (`torch_dtype_code` helper).
- **Storage offsets.** `storage_offset` captures each alias's view into the parent storage. When present, it must not push `storage_offset + size` past the storage boundary; both Python (`tensorcast/api/store.py` validation) and daemon (`daemon/state/lip_metadata_utils.cc:67-78`) enforce the bound.

Together these invariants make the canonical index a pure description of *layout* rather than physical allocation. Physical placement is captured elsewhere (byte-range maps and CUDA IPC handles) while the index remains transport-neutral.

## Memory Model Relationships

### PyTorch storages, aliases, and Register metadata

- **Storages.** The SDK gathers unique storages from the state_dict, deduplicating by `data_ptr` per device. During registration each storage becomes a `RegisterStorageMeta` (`daemon/state/types.h:43-58`) containing a storage ID, device, CUDA IPC handle bytes, and the storage-length in bytes.
- **Aliases.** Each tensor alias carries `shape`, `stride`, `dtype`, and `storage_offset` (`RegisterTensorAliasMeta`). The alias refers back to its storage, which guarantees that views into a larger tensor are captured correctly.
- **Lease segments.** When copying or hashing, the daemon works with `LeaseSegMeta` entries describing contiguous CUDA IPC ranges (`daemon/state/types.h:32-41`). These segments are stitched into the AVBS according to canonical index offsets.

### Coalesced VRAM vs. user storage

- **Coalesced plan (`put`).** For coalesced registrations (`tensorcast/api/store.py:1726-1791`), the SDK allocates a daemon-owned contiguous VRAM buffer and computes per-tensor destinations via `calculate_tensor_device_offsets` (`tensorcast/api/_indices.py:19-56`). That yields a new storage layout that still respects canonical index ordering. The canonical index recorded for the artifact uses the coalesced destination offsets while preserving alias metadata, so any later materialization reconstructs exactly what the daemon allocated.
- **Lease-In-Place (LIP).** For LIP flows, the producer's storages remain authoritative. The daemon rebuilds the canonical index from alias+storage metadata (`daemon/state/lip_metadata_utils.cc`) and associates each storage with one destination offset derived from the segments it exports. Consumers later open CUDA IPC handles to map those physical ranges.

## Build Paths

### Disk save (`tensorcast.testing.io_disk.save_dict`, test-only)

`tensorcast/csrc/checkpoint_py.cc` materializes canonical index bytes when persisting to disk:

1. Replays the canonical ordering and writes `tensor_index.(json|cbor)`.
2. Computes `index_multihash` from the canonical index and `data_multihash` from the normalized byte stream (`compute_data_multihash_from_seekable_source`).
3. Emits `artifact_descriptor.json` embedding both hashes and the total size.

Because the canonical index encodes the logical layout, on-disk partitions can be reshuffled or zero-padded without breaking identity; only the canonical offsets matter.

### SDK registration (`put` / `register_artifact`)

During registration (`tensorcast/api/_register.py`):

1. `BuildContext.from_artifact` inspects tensors to produce `tensor_meta_index` and `tensor_source_index`.
2. `calculate_tensor_device_offsets` computes coalesced destination offsets, deduplicating identical `(src_offset, size)` tuples so that shared storages reuse the same destination range.
3. `build_v2_index_bytes` (`tensorcast/api/_indices.py:129-151`) emits canonical index bytes using the destination offsets (coalesced or lease-in-place depending on plan) but retains each alias's `storage_offset`.
4. The SDK ships the canonical index bytes (or at least the multihash) along with storage and alias descriptors to the daemon, ensuring both sides agree on layout before any GPU copy begins.

### Daemon reconstruction (LIP)

When the daemon commits a LIP registration (`daemon/state/lip_manager.cc:490-610`):

1. The commit RPC provides segments (CUDA IPC handles), storages, and aliases.
2. `build_canonical_index_from_metadata` (`daemon/state/lip_metadata_utils.cc`) rebuilds the canonical index so hashing is authoritative server-side.
3. The daemon opens CUDA IPC mappings, compiles the canonical `ByteRangeMap`, and streams a `ByteRangeMappedSource` that zero-fills PAD gaps to compute `data_multihash`.
4. The resulting `index_multihash` determines whether an artifact already exists (`ArtifactDeviceKey` lookup) and whether deduplication or lease extension is possible.

Canonical offsets depend only on storage identity, not per-alias offsets. The daemon mirrors the SDK by emitting storage-level destination offsets and full storage lengths for every alias, so view tensors hash identically across disk, coalesced, and LIP flows.

## Global Store, Artifact Identity, and Deduplication

- **Index caching.** The Global Store persists canonical index blobs keyed by `index_multihash` (`tensorcast/global_store/repositories/artifact_index_repository.py`). Replicas reference the stored key rather than duplicating the bytes.
- **mi2 artifact IDs.** Commit-time multihashes produce the `mi2:` ID returned to clients and used as the primary key for replicas (`docs/designs/0007-content-addressed-artifact-id.md`). As long as canonical index bytes match, artifacts from disk, coalesced VRAM, and LIP share the same ID.
- **Doc sync with Global Store.** When clients call `materialize_replica` or `get`, the Store Engine pulls canonical index bytes via `get_canonical_index_by_id` (`core/store/store_engine.cc:1649-1684`) to reconstruct the expected layout before deciding whether local replicas satisfy the request.

## Materialization, Transport, and CUDA Memory

- **ByteRangeMap + PAD.** From the canonical index JSON, Store Engine builds a canonical `ByteRangeMap` (`core/store/materialization/dataplane/sources/byte_range_map_builder.cc`) and compiles a `ByteRangeProgram`. The map outlines alternating DATA and PAD spans, ensuring consumers zero-fill gaps consistently.
- **Coalesced VRAM consumption.** Consumers opening a coalesced replica map exactly the daemon-allocated buffer using CUDA IPC. Because the canonical index encodes the buffer’s logical layout, tensors restored via `restore_tensors` mirror the original storages regardless of intermediate transfers.
- **LIP consumption.** For lease-in-place replicas, the daemon refuses same-device consumption but can copy over NVLink/PCIe into a coalesced buffer on another device. The canonical index ensures that copy respects view offsets and pad gaps, so the resulting coalesced buffer hashes identically to the producer’s layout.
- **Cross-node P2P.** When Global Store coordinates remote fetches, the sending daemon reads its replica using canonical offsets (coalesced VRAM or disk) and streams DATA segments to the receiver. Because both sides share the canonical index, the receiver knows exactly how to populate its local buffer and can perform verification using the same offsets.

## Cross-Process and Cross-Node Consistency

- **CUDA IPC invariants.** `LeaseSegMeta` and `RegisterStorageMeta` tie canonical offsets to CUDA IPC handles. The daemon verifies that each alias fits within its storage bounds before accepting the registration (`daemon/state/lip_metadata_utils.cc:67-78`), preventing mismatched offsets from propagating.
- **Process isolation.** Consumers never see producer-specific addresses. Canonical offsets are relative to the AVBS, while actual IPC mappings are per-process resources managed inside the daemon (`daemon/state/lip_manager.cc:520-610`).
- **Cross-node transmissions.** The canonical `ByteRangeMap` is agnostic to transport; PAD spans become ranges of zeros, while DATA spans are serialized in the same order independent of source. This uniformity allows hashing and verification to be identical whether bytes came from disk partitions (`tensorcast/csrc/checkpoint_py.cc:316-403`), local GPU memory (`core/store/materialization/dataplane/sources/byte_range_mapped_source.cc`), or remote daemons.

## Current Gaps and Risks

- **Canonical offset drift.** Deduplication relies on every path emitting storage-level destination offsets and storage lengths for each alias. Keep conformance tests between SDK and daemon builders so future edits preserve the shared-storage invariant.
- **Split authority.** Python builds canonical index bytes during registration, while the daemon rebuilds them from metadata. Divergence in either direction (field ordering, offset arithmetic, numeric width) silently corrupts deduplication. Long-term mitigation is to share a single builder implementation (e.g., extend `core/store/materialization/dataplane` with a helper that accepts storage + alias spans) and reuse it in both places.
- **Padding assumptions.** Every consumer assumes PAD ranges are zero. If producers materialize artifacts with uninitialized memory and rely on PAD to be copied through, verification will fail. Documentation and guardrails should make this explicit.
- **Device mismatch.** LIP commits currently rely on the caller to send accurate `device_id` for storages. If a storage is registered on the wrong device, CUDA IPC mapping can succeed but later P2P copies may degrade or fail. Enhancements could include hashing `(device_id, handle_bytes)` to detect device drift early.
- **Global Store consistency.** The Global Store treats `index_multihash` as the canonical key. If we ever emit mismatched canonical index bytes for the same artifact (e.g., due to SDK vs. daemon drift), cached copies become poisoned. Automated validation comparing client-sent index bytes with the daemon-rebuilt version can mitigate this risk; current code only logs a warning (`daemon/state/lip_manager.cc:498-505`).

## Recommended Practices

1. **Guard storage-level offset invariants.** Keep regression tests and CI checks that compare daemon-built indices with SDK output so shared storages continue to yield identical offset/size pairs.
2. **Centralize canonical index construction.** Expose a C++ helper in `core/store/materialization/dataplane` that accepts ordered storages, aliases, and destination offsets; bind it for Python so the SDK and daemon call the same implementation.
3. **Extend test coverage.** Add end-to-end tests covering: shared storage views, zero-sized tensors, multi-device artifacts, and PAD-heavy layouts. Verify that disk save, coalesced registration, and LIP registration yield identical canonical index hashes.
4. **Strengthen observability.** Surface metrics for canonical index cache hits/misses (`tensorcast/global_store/grpc_service.py:297-534`) and mismatches detected during daemon rebuild. Alert when deduplication fails unexpectedly.
5. **Document PAD semantics for users.** Ensure SDK and user guides make it clear that PAD bytes are always treated as zeros and will not survive round trips unless explicitly part of storage data.

## References

- Canonical index builders: `core/store/materialization/dataplane/metadata/canonical_index.{h,cc}`, `tensorcast/api/_indices.py`, `daemon/state/lip_metadata_utils.cc`
- Hashing & byte-range maps: `core/store/materialization/dataplane/sources/byte_range_map_builder.cc`, `core/store/materialization/dataplane/sources/byte_range_mapped_source.cc`, `tensorcast/csrc/checkpoint_py.cc`
- Registration flows: `tensorcast/api/_register.py`, `tensorcast/api/store.py`
- Lease management: `daemon/state/lip_manager.cc`, `daemon/state/types.h`
- Design background: `docs/designs/0003-unified-memory-registration-avbs-lip.md`, `docs/designs/0007-content-addressed-artifact-id.md`, `docs/architecture/api/api-design.md`
