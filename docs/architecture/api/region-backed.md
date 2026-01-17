---
title: Region Backed Registration
description: Region registration, LIP reuse, and deregistration
---

# Region Backed Registration

This document explains region-backed registration, lease reuse, and quiesced
cleanup flows.

Related docs:

- Public surface: [API Design](./api-design.md#region-apis)
- LIP registration internals: [Registration Flow](./registration-flow.md#lease-in-place-path)
- Region-backed `get_into` (different mechanism, same motivation): [Materialization Flow](./materialization-flow.md#region-backed-get_into-materializeintotarget-v2)
- Failure semantics: [Error, Retry, Observability](./error-retry-observability.md)

## What is a “VRAM region”?

A VRAM region is a daemon-tracked handle to a long-lived CUDA allocation (via
CUDA IPC handle bytes), scoped to:

- a specific GPU device
- an owner PID (for safety)
- a TTL (to prevent leaks)

Why regions exist:

- LIP registration requires CUDA IPC handles for client-owned VRAM.
- Creating/transporting handles per tensor/storage is expensive and error-prone.
- Many applications already allocate “slabs” and carve them into many tensors;
  region-backed registration turns those into offset-based references.

## Register VRAM Region

The SDK registers reusable CUDA IPC regions using:

- `RegisterVramRegion` RPC
- `Store.register_vram_region(...)` API

A region is scoped to a device and owner PID and is protected by TTL. The daemon
stores region metadata and handle bytes in `IpcRegionRegistry`.

### RegisterVramRegionRequest (field reference)

Proto: [proto/tensorcast/daemon/v2/store_daemon.proto](../../../proto/tensorcast/daemon/v2/store_daemon.proto)

| Field | What it does | Why it exists |
|---|---|---|
| `session_id` | Optional client session tag. | Diagnostics and ownership correlation. |
| `device_id` | GPU ordinal for the region. | Validate/route IPC handle usage. |
| `cuda_ipc_handle` | CUDA IPC handle bytes for the base allocation. | The core “capability” the daemon needs to export/resolve. |
| `size_bytes` | Region size. | Bounds checks for offsets and storages. |
| `ttl_ms` | Region TTL. | Auto-cleanup in crash/leak scenarios. |
| `owner_pid` | Owner process id. | Prevent other processes from hijacking lifecycle. |
| `region_name` | Optional tag. | Operator-friendly debugging (e.g. “model_weights_slab”). |

Example:

```python
import tensorcast

tensorcast.init(mode="connect")
handle = tensorcast.register_vram_region(
    device_id=0,
    base_ptr=int(slab_ptr),
    size_bytes=slab_bytes,
    ttl_ms=60_000,
    name="weights_slab",
)
```

### UnregisterVramRegionRequest (field reference)

Proto: [proto/tensorcast/daemon/v2/store_daemon.proto](../../../proto/tensorcast/daemon/v2/store_daemon.proto)

| Field | What it does | Notes |
|---|---|---|
| `region_id` | Region identifier returned by `RegisterVramRegion`. | Required. |
| `owner_pid` | Owner PID verification. | Safety: mismatches fail. |
| `force` | Best-effort release even if TTL expired. | Useful for cleanup; use with care. |

## Region Referenced LIP Storage

When LIP registration sees a storage fully covered by a registered region, the
SDK emits:

- `StorageEntry.vram_region_id`
- `StorageEntry.mapping_base_offset`

The lease segments then reference the storage id without attaching a per-storage
CUDA IPC handle. The daemon resolves region handles and holds refs for the
lifetime of the export.

Why “fully covered” matters:

- It lets the daemon validate that every byte the artifact needs is within the
  region bounds.
- It avoids a mixed mode where part of a storage would need a separate CUDA IPC
  handle (harder to reason about and easy to get wrong).

## Region-Backed get_into (MaterializeIntoTarget)

Region-backed `get_into` uses the same region registry but a different control
path from LIP registration. The daemon writes directly into an existing CUDA
region when the target layout is fully coalesced and matches the canonical
index. No replica is allocated.

### SDK preconditions

The SDK enforces strict eligibility rules before invoking
`MaterializeIntoTarget`:

- `artifact_id` is required (key-based requests are rejected).
- Full tensor set required (no subset).
- All target tensors must be CUDA, contiguous, and match canonical dtype/shape/stride.
- Canonical layout must be coalesced (segment offsets equal storage offsets).
- The target tensors must map into a single registered region.

These checks are implemented in `tensorcast/api/store/materialization.py` and
`tensorcast/api/_region_cache.py`.

### Daemon validation

The daemon validates the request and the layout strictly:

- `TargetLayout` must be `LAYOUT_KIND_COALESCED_UNSPECIFIED` with
  `INDEX_KIND_CANONICAL_UNSPECIFIED`.
- Exactly one storage entry, using `vram_region_id` and `mapping_base_offset`.
- `tensor_spec_kind` must be offsets or alias format.
- All canonical tensors must be present; offsets/lengths must match canonical
  index entries, and `storage_offset` must equal canonical logical offset.
- `storage_length` must cover the full logical size.
- `device_uuid` and `pid` are required and must match the region device.

### Execution

Once validated, the daemon:

1. Acquires the region from `IpcRegionRegistry` and maps its CUDA IPC handle.
2. Computes the canonical index plan and materializes directly into the region
   via `StoreEngine::materialize_into_target`.
3. Skips verification (`MaterializeHints::Verify::NONE`) by design; metrics
   record that verification was skipped.

On transfer `DataLoss`, the daemon marks the region as poisoned to prevent
reuse. The client then unregisters the region from its cache.

## Deregister Artifact

`DeregisterArtifact` performs a quiesced teardown for LIP replicas:

1. Quiesce new staged exports.
2. Drain active exports if `wait_for_drain=true`.
3. Revoke the commit lease if the owner matches.
4. Best-effort unregister from Global Store.

The SDK exposes this as `Store.deregister_artifact(...)` and returns a
`DeregisterArtifactOutcome` containing drain status and released region ids.

### DeregisterArtifactRequest (field reference)

Proto: [proto/tensorcast/daemon/v2/store_daemon.proto](../../../proto/tensorcast/daemon/v2/store_daemon.proto)

| Field | What it does | Notes |
|---|---|---|
| `artifact_id` | Target content-addressed artifact id. | Required. |
| `wait_for_drain` | Block until active staged exports drain (or timeout). | SDK surface: `wait=`. |
| `drain_timeout_ms` | Optional bounded wait timeout. | `0` uses daemon policy. |
| `extend_ttl_ms` | Optional TTL bump before quiesce. | Useful if lease TTL is close to expiring. |
| `owner_pid` | Optional PID check. | When present, mismatches fail with `PERMISSION_DENIED`. |
| `device_id` | Disambiguate when replicas span devices. | Avoid revoking the wrong resident replica. |
| `release_regions` | Release region references after deregistration. | Prevent ref leaks on long-running daemons. |

## TTL Extension And Transport Hold

- `DeregisterArtifactRequest.extend_ttl_ms` extends TTL before quiesce.
- `GetArtifactOptions.transport_hold_ms` requests a TTL bump during transfers.

## Failure Modes

- Owner mismatch returns `PERMISSION_DENIED`.
- Expired regions return `FAILED_PRECONDITION`.
- Drain timeouts return `DEADLINE_EXCEEDED` and leave the artifact quiesced.

## Code Map

- Region registry: [daemon/state/ipc_region_registry.h](../../../daemon/state/ipc_region_registry.h)
- LIP manager: [daemon/state/lip_manager.cc](../../../daemon/state/lip_manager.cc)
- Daemon RPC wiring: [daemon/service/grpc_service_impl.cc](../../../daemon/service/grpc_service_impl.cc)
- SDK region cache: [tensorcast/api/_region_cache.py](../../../tensorcast/api/_region_cache.py)
- SDK APIs: [tensorcast/api/store/__init__.py](../../../tensorcast/api/store/__init__.py)
