---
slug: 0036-materialization-pipeline-v2
title: Lazy Artifact Handle Phase 1 – Materialization Pipeline v2
areas: ["sdk", "daemon"]
related_code:
  - tensorcast/api/store/materialization.py
  - tensorcast/api/_materialize.py
  - tensorcast/proto/daemon/v2/store_daemon.proto
  - core/store/materialization/**
  - daemon/materialization/**
links:
  successor: ./0036-02-artifact-handle-core.md
  disk_variant: ./0036-04-disk-artifact-variant.md
---

# Summary

Phase 1 extracts the low-level capabilities the Lazy Artifact Handle stack needs from the C++ daemon, protobufs, and the Python `MaterializationPipeline`. Today the pipeline always asks the daemon for a full CUDA IPC buffer and eagerly reconstructs a full `state_dict`, which makes selective tensor fetches, view slicing, and background prefetching impossible.

This design revs the daemon RPCs (`MaterializeReplica*`) to v2, teaches the Store daemon to emit per-tensor payload descriptors, and reshapes the Python pipeline to stream descriptors rather than hydrate Python dictionaries eagerly. Both eager (`store.get`) and future lazy APIs share the same pipeline, so we must land these changes first to keep the end user surface area stable while we ship incremental SDK improvements.

# Goals / Non-Goals

## Goals

1. Ship a v2 daemon surface that accepts `tensor_names`, `view_subset_hash`, and explicit `SourcePreference` hints.
2. Make `MaterializationPipeline` consume iterators of `(TensorPayloadDescriptor, memoryview)` instead of full `state_dict` objects, while keeping the public synchronous and async methods backward compatible.
3. Unify disk and daemon code paths so that slice-aware disk reads share the same descriptor plumbing.
4. Preserve existing tracing, retry, and fallback semantics (`FallbackResolver`) so the change is transparent to callers.

## Non-Goals

- Introducing the `Artifact` handle or any new public SDK entry point (Phase 2).
- Adding async batching, prefetch tickets, or view composition APIs (Phase 3).
- Changing the Global Store metadata model or `tensor_index.json` schema.

# Architecture & Interfaces

## Current pain

`MaterializationPipeline` gathers a complete `state_dict` even when callers only need one tensor, because `_materialize_fn` returns a `MaterializedArtifact` that already contains hydrated PyTorch tensors. The call chain below shows that `get()` returns the dictionary directly, and `get_into()` immediately copies from the temporary dict.

```111:294:tensorcast/api/store/materialization.py
class MaterializationPipeline:
    """Retrieval orchestration for get/get_into/get_view flows."""
    def get(...):
        materialized, _ = self._perform_get_with_retry(...)
        return materialized.state_dict
```

`MaterializedArtifact` itself is tightly coupled to this eager contract: it holds `state_dict` plus `canonical_index_bytes`, which forces every daemon RPC to transfer all tensors before the client can inspect metadata.

```32:189:tensorcast/api/_materialize.py
@dataclass(frozen=True)
class MaterializedArtifact:
    artifact_id: str
    state_dict: dict[str, torch.Tensor]
    canonical_index_bytes: bytes
    replica_uuid: str
    ...
def materialize_artifact(...)-> MaterializedArtifact:
    ...
    state_dict = restore_tensors(...)
```

Under this model, adding selective loading or prefetch tickets would require every caller to fork their own materialization pipeline. Phase 1 fixes the foundation instead.

## RPC surface (daemon + proto)

We introduce the following protobuf changes under a new `tensorcast.proto.daemon.v2` package and bump the daemon handlers:

- `MaterializeReplicaRequest` / `MaterializeByKeyRequest`
  - `repeated string tensor_names`
  - `bytes view_subset_hash`
  - `SourcePreference preference = {AUTO, LOCAL, P2P, DISK}`
  - `DiskFallbackHint disk_fallback = {string disk_path, bool verify_checksums}`
  - `string replica_uuid` (explicit, reused by prefetch later)
- `MaterializeReplicaResponse`
  - `repeated TensorPayloadDescriptor payloads`
  - `bytes canonical_index_bytes`
  - `bytes view_index_bytes`
  - `uint64 generation`
  - `ViewSubset view_subset` (optional, used when the server applies slicing)
  - `ReplicaTicket ticket` (populated when `wait_for_completion=false`)
- `ResolveArtifactFromDisk` (new RPC)
  - `string disk_path`
  - `bool verify_checksums`
  - returns `{artifact_id, canonical_index_bytes, generation}` for cache hydration without materializing

Canonical index fields are standardized as `canonical_index_bytes` / `view_index_bytes` (never `*_json`) and always paired with `generation` so caches and disk resolution stay in sync across SDK layers.

**Hash and identity semantics (required)**
- `view_subset_hash` (request) and `ViewSubset.subset_hash` (response) are opaque **raw digest bytes** identifying the requested selection (e.g., SHA-256 digest of a canonical serialization of sorted, unique `tensor_names`). They MUST NOT contain UTF-8 bytes of a hex string and MUST NOT reuse `view_data_hash`.
- `view_id` identifies a variant ByteSpace (see `docs/designs/0016-artifact-view-v1.md`) and is distinct from subset selection identity.

`TensorPayloadDescriptor` is a thin schema containing `{ string name, string dtype, string device_uuid, uint64 buffer_offset, uint64 byte_length, uint64 storage_offset, repeated int64 shape, repeated int64 stride }`. The daemon uses UMA layout information to only emit descriptors for requested tensors and writes contiguous slices into the exported IPC buffer. Disk loaders use the same message to describe byte ranges in POSIX files.

The daemon and SDK use the v2 RPCs end-to-end; legacy v1 RPCs are no longer part of the supported surface.

## Python pipeline changes

### Materialized payload iterator

We replace `MaterializedArtifact` with a streaming representation:

```python
@dataclass(frozen=True)
class MaterializationPayload:
    artifact_id: str
    canonical_index_bytes: bytes
    descriptors: Sequence[TensorPayloadDescriptor]
    payload_iter: Callable[[], Iterator[tuple[TensorPayloadDescriptor, memoryview]]]
    replica_uuid: str | None
    disk_path: str | None
```

`materialize_artifact_v2()` (new helper in `tensorcast/api/_materialize.py`) opens the CUDA IPC handle once, constructs `memoryview` slices per descriptor, and yields `(descriptor, memoryview)` tuples lazily. Disk paths wrap `mmap` views instead of CUDA pointers but expose the same iterator type.

`MaterializationPipeline` now has two layers:

1. `_perform_get_with_retry_v2()` returns `(MaterializationPayload, device_id)`.
2. Thin adapters convert the iterator into legacy outputs:
   - `get()` drains the iterator and creates `dict[str, torch.Tensor]` exactly as before.
   - `get_into()` copies descriptor-sized chunks directly into the caller’s tensors without building temporaries.
   - `get_view()` reuses the iterator but rehydrates only the requested view subset.

This preserves public compatibility while allowing the forthcoming lazy artifact layer to request descriptors directly.

### Disk parity via daemon

Every disk materialization request now flows through the daemon’s `DiskLoader`, `SelectionPlan`, and `ViewPlanSource`, as mandated in [0038-daemon-only-disk-materialization](./0038-daemon-only-disk-materialization.md). The SDK no longer performs direct disk I/O.

**Design principle**: All disk loading goes through the daemon via gRPC, using the same `MaterializationPipeline` flow as P2P artifacts. The SDK never reads disk directly.

**Flow for disk-backed materialization**:

```
MaterializationPipeline.get(fallback=FallbackOptions.for_disk(path))
        │
        ▼
gRPC: MaterializeReplicaRequest(
    artifact_id=...,
    preference=DISK,
    disk_fallback=DiskFallbackHint(disk_path=path, verify_checksums=true),
    tensor_names=[...],  # optional selective loading
)
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  Daemon (C++ infrastructure - already implemented)            │
│  DiskLoader → FilePartitionSource → SelectionPlan → tensors  │
│  ViewPlanSource applies view-based byte-range selection       │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
MaterializeReplicaResponse(
    payloads=[TensorPayloadDescriptor, ...],
    canonical_index_bytes=...,
    generation=...,
)
        │
        ▼
SDK maps CUDA IPC handles → torch.Tensor dict
```

**Benefits of daemon-mediated disk access**:

1. **Unified flow**: Disk and P2P artifacts use identical `MaterializationPipeline` code paths
2. **Consistent observability**: All operations traced, metered, and logged uniformly
3. **Reuse existing C++ infrastructure**: Daemon's `DiskLoader`, `ViewPlanSource`, `SelectionPlan` are already battle-tested
4. **Simplified SDK**: No direct disk I/O code in Python; smaller attack surface

**Backward compatibility**: All customer-facing flows go through the daemon-backed pipeline (even when preferring disk). For lazy disk artifact construction via `tc.from_disk()`, see Phase 4 (`0036-04-disk-artifact-variant.md`).

**Daemon C++ components used** (no new implementation required):
- `DiskLoader` (`core/store/materialization/dataplane/loaders/disk_loader.h`)
- `FilePartitionSource` (`core/store/materialization/dataplane/sources/file_partition_source.h`)
- `ViewPlanSource` (`core/store/materialization/dataplane/view/view_plan_source.h`)
- `SelectionPlan` (`core/store/materialization/contracts/view/view_plan.h`)

### Telemetry and observability

- Tracing spans (`Client/MaterializeArtifact`) now attach custom attributes per descriptor (`tc.tensor.count`, `tc.tensor.bytes`).
- `store_metrics.materialization_latency_ms` differentiates `all` vs. `subset` pulls.
- Additional counters track cache vs. RPC resolution of canonical indices because the iterator reuses metadata more aggressively.

## Naming compliance

| Symbol | Kind | Compliance |
|--------|------|------------|
| `MaterializationPayload` | Python dataclass | PascalCase |
| `materialize_artifact_v2` | Python function | snake_case |
| `TensorPayloadDescriptor` | Proto message | PascalCase (message) |
| `SourcePreference` | Proto enum | PascalCase |
| `DiskFallbackHint` | Proto message | PascalCase |

All newly introduced Python names follow the SDK naming rules in `AGENTS.md`. Proto enums/messages follow standard PascalCase conventions already used under `tensorcast.proto`.

# Trade-offs & Risks

- **Proto compatibility**: Requires synchronized rollout of daemon binaries and SDK so v2 is available end-to-end (no v1 fallback).
- **Iterator misuse**: A bug in draining the iterator could leak CUDA IPC handles. We add unit tests that intentionally stop consuming early and assert `_release_materialized` cleans up.
- **Daemon dependency for disk loading**: All disk operations now require a running daemon. This is intentional—it ensures unified observability and code paths. For offline scenarios without daemon, the legacy `restore_tensors_from_disk()` C++ API remains available but is not exposed through the new pipeline.
- **Performance regressions**: Descriptor bookkeeping adds per-tensor overhead. We benchmark single tensor fetches vs. baseline to ensure <5% regression when `names=None`, and significantly better results for `names=[one tensor]`.

# Compatibility & Acceptance Criteria

1. **Selective streaming tests**: New integration tests under `tests/python/api/test_materialization_pipeline_v2.py` verify that requesting `tensor_names=["foo"]` only maps that tensor and leaves the rest untouched.
2. **Disk fallback via daemon tests**: `tests/python/api/test_disk_materialization_v2.py` validates:
   - `FallbackOptions.for_disk(path)` with `tensor_names=["a"]` returns only tensor "a"
   - View-based slicing on disk artifacts works identically to P2P artifacts
   - Results match canonical disk artifacts for full loads
3. **Daemon DiskLoader tests**: `bazel test //daemon:disk_loader_materialization_test --test_env=TENSORCAST_CUDA_BACKEND=fake` confirms daemon correctly uses `DiskLoader` + `SelectionPlan` for disk-backed materialization.
4. **Daemon soak**: Run `bazel test //daemon:materialization_v2_test --test_env=TENSORCAST_CUDA_BACKEND=fake` to validate the new RPC struct.
5. **Tracing verification**: Observability smoke test ensures the new span attributes are emitted (checked via OTLP fixture).
6. **Rollout sequencing**: Documented runbook that lands daemon/proto and SDK together so the v2-only surface is consistent.

# References

- `tensorcast/api/store/materialization.py`
- `tensorcast/api/_materialize.py`
- `daemon/materialization/loader.cc`
- `proto/daemon/v2/store_daemon.proto`
- `docs/designs/0036-02-artifact-handle-core.md`

**Daemon C++ components (used for disk loading)**:
- `core/store/materialization/dataplane/loaders/disk_loader.h`
- `core/store/materialization/dataplane/sources/file_partition_source.h`
- `core/store/materialization/dataplane/view/view_plan_source.h`
- `core/store/materialization/contracts/view/view_plan.h` — `SelectionPlan` definition
