---
title: Region Backed Registration
description: Region registration, local shared-region access, LIP reuse, and deregistration
---

# Region Backed Registration

This document explains region-backed registration, lease reuse, and quiesced
cleanup flows.

Current status:

- `RegisterRegion` is the canonical local region-registration surface and
  supports both `VRAM` and `HOST_SHARED`,
- SDK `register_vram_region(...)` remains a convenience helper, but it lowers
  to `RegisterRegion(memory_kind=VRAM)` rather than a separate daemon RPC,
- byte-artifact `BatchGetIntoRegion` /
  `BatchPutIfAbsentFromRegion` accept region-backed layouts over both `VRAM`
  and `HOST_SHARED`,
- `HOST_SHARED` widens only the local placement layer; it does not change
  artifact identity, routed authority, or inter-daemon transport semantics,
- `MaterializeIntoMappedTarget` remains a separate mapped-target path and
  currently rejects `HOST_SHARED` target layouts.

Related docs:

- Public surface: [API Design](./api-design.md#region-apis)
- LIP registration internals: [Registration Flow](./registration-flow.md#lease-in-place-path)
- Region-backed `get_into` (different mechanism, same motivation): [Materialization Flow](./materialization-flow.md#region-backed-get_into-materializeintotarget-v2)
- View semantics and identity: [Artifact Views and Retrieval](../artifact-views-and-retrieval.md)
- Failure semantics: [Error, Retry, Observability](./error-retry-observability.md)
- Strategy-plane design: [0108 Tensor-Aware Materialization Strategy Plane](../../designs/0108-tensor-aware-materialization-strategy-plane.md)

## What is a region?

A region is a daemon-tracked local shared byte window with:

- one memory kind,
- one owner PID or local lifecycle authority,
- explicit bounds,
- and a TTL or lease-scoped lifetime.

Live region kinds:

- `VRAM`, backed by CUDA IPC handle bytes and tracked in
  `IpcRegionRegistry`,
- `HOST_SHARED`, backed by daemon-managed or otherwise locally shared host
  memory and tracked by the same registry.

Why regions exist:

- region-backed placement lets a caller and the local daemon agree on one
  mutable byte window without making that window part of artifact identity,
- regions amortize registration and safety checks across many item placements,
- many applications naturally manage slabs and offsets rather than one isolated
  allocation per artifact.

## What is a “VRAM region”?

A VRAM region is a daemon-tracked handle to a long-lived CUDA allocation (via
CUDA IPC handle bytes), scoped to:

- a specific GPU device
- an owner PID (for safety)
- a TTL (to prevent leaks)

Why this current live flavor exists:

- LIP registration requires CUDA IPC handles for client-owned VRAM.
- Creating or transporting handles per tensor or storage is expensive and
  error-prone.
- Many applications already allocate “slabs” and carve them into many tensors;
  region-backed registration turns those into offset-based references.

## Register Region

The canonical SDK and RPC surface is:

- `RegisterRegion` RPC
- `Store.register_region(...)` API
- `Store.register_vram_region(...)` API, as a thin SDK helper over
  `RegisterRegion(memory_kind=VRAM)`

A region is scoped to an owner PID, explicit bounds, and TTL. The daemon stores
region metadata in `IpcRegionRegistry` and resolves the concrete attachment
mechanism from `memory_kind`.

### RegisterRegionRequest (field reference)

Proto: [proto/tensorcast/daemon/v2/store_daemon.proto](../../../proto/tensorcast/daemon/v2/store_daemon.proto)

| Field | What it does | Why it exists |
|---|---|---|
| `session_id` | Optional client session tag. | Diagnostics and ownership correlation. |
| `memory_kind` | Selects `VRAM` or `HOST_SHARED`. | One local-only region model across both memory kinds. |
| `device_id` | GPU ordinal for VRAM regions. Ignored for `HOST_SHARED`. | Validate/route CUDA IPC usage. |
| `cuda_ipc_handle` | CUDA IPC handle bytes for a VRAM base allocation. | The attachment the daemon needs for VRAM mapping. |
| `host_shared` | `HOST_SHARED` region spec. | Declares attach metadata, daemon-managed export, and host region class. |
| `size_bytes` | Region size. | Bounds checks for offsets and storages. |
| `ttl_ms` | Region TTL. | Auto-cleanup in crash/leak scenarios. |
| `owner_pid` | Owner process id. | Prevent other processes from hijacking lifecycle. |
| `region_name` | Optional tag. | Operator-friendly debugging (e.g. “model_weights_slab”). |

`HostSharedRegionSpec` carries:

- `attach_token`: opaque local-only attach metadata returned by the daemon or
  supplied by the caller,
- `daemon_managed`: whether the daemon owns the underlying shared host slab,
- `region_class`: `SCRATCH` or `ALLOCATOR`.

Memory-kind rules:

1. `VRAM` requires `device_id >= 0` and a non-empty `cuda_ipc_handle`.
2. `HOST_SHARED` requires `host_shared` and is recorded internally with
   `device_id = -1`.
3. Regions remain local-only mutable placement state; they are not part of
   artifact identity or routed authority truth.

Example (`VRAM`):

```python
import tensorcast

tensorcast.init(mode="connect")
# handle_bytes should come from the local CUDA IPC export helper for slab_ptr.
handle = tensorcast.register_region(
    memory_kind=tensorcast.RegionMemoryKind.VRAM,
    size_bytes=slab_bytes,
    ttl_ms=60_000,
    device_id=0,
    cuda_ipc_handle=handle_bytes,
    name="weights_slab",
)
```

Example (`HOST_SHARED`, daemon-managed):

```python
import mmap
import tensorcast

tensorcast.init(mode="connect")
handle = tensorcast.register_region(
    memory_kind=tensorcast.RegionMemoryKind.HOST_SHARED,
    size_bytes=64 << 20,
    ttl_ms=60_000,
    daemon_managed=True,
    host_shared_region_class=tensorcast.HostSharedRegionClass.SCRATCH,
    name="host_scratch",
)
attachment = tensorcast.attach_host_shared_region(handle)
host_mapping = mmap.mmap(attachment.fd, attachment.size_bytes)
```

## SDK VRAM Region Convenience Helper

The SDK still exposes `Store.register_vram_region(...)` for callers that hold a
base VRAM pointer and want the SDK to mint CUDA IPC handle bytes for them. This
is client-side syntax sugar only: the daemon sees a normal
`RegisterRegionRequest` with `memory_kind = VRAM`, `device_id`, and
`cuda_ipc_handle` populated. There is no separate VRAM-specific daemon RPC or
proto object.

## HOST_SHARED semantics

`HOST_SHARED` keeps the same local-only region model:

1. It is a local mutable placement window, not part of artifact identity or
   routed truth.
2. It carries explicit bounds, owner PID, TTL, and poison state in the same
   registry as `VRAM`.
3. It may be daemon-managed or caller-attached through opaque local-only attach
   metadata.
4. Host pinning is optional performance policy. It is not part of region
   correctness.

Current live `HOST_SHARED` classes:

- `SCRATCH`: a general local shared host window for staging or direct region
  placement,
- `ALLOCATOR`: a caller-managed shared host region whose offsets are interpreted
  together with explicit slot tokens.

For allocator-backed layouts, `slot_index` and `slot_generation` are
caller-supplied lifetime labels:

- TensorCast validates that both are present when required,
- TensorCast echoes them back in per-item outcomes,
- TensorCast does not own slot allocation, slot retirement, or stale-completion
  filtering.

## Region-backed byte-artifact batch IO

`BatchGetIntoRegion` and `BatchPutIfAbsentFromRegion` now accept `TargetLayout`
storages that reference either:

- `StorageEntry.vram_region_id`, or
- `StorageEntry.region_ref` with `memory_kind = VRAM | HOST_SHARED`.

Normative rules:

1. These RPCs remain loopback or UDS-only. Remote or home daemons never write
   directly into caller-visible regions.
2. A single byte-artifact region layout must use one memory kind. Mixed
   `VRAM` and `HOST_SHARED` layouts are rejected.
3. Pure `HOST_SHARED` layouts require `storage.device_id = -1` and empty
   `device_uuid`.
4. `BatchPutIfAbsentFromRegion` reads bytes from the local region mapping
   selected by `TargetLayout`.
5. `BatchGetIntoRegion` writes bytes directly into the local region mapping
   selected by `TargetLayout`.
6. Verification modes, routed authority, `HomeBatch*`, and inter-daemon
   transport semantics are unchanged by the local memory kind.
7. For allocator-backed `HOST_SHARED` byte-artifact layouts, every offset must
   be explicit and must carry `slot_index` plus `slot_generation`.

## HOST_SHARED attach and release

Daemon-managed `HOST_SHARED` regions use the existing local handle plane:

- `RegisterRegion(..., daemon_managed=True, ...)` returns an attach token,
- `attach_host_shared_region(...)` resolves that token to a local memfd,
- `release_host_shared_region(...)` releases the local attachment,
- unregistration or TTL expiry eventually lets the daemon reap the region.

This attach path is local-only. It is not a remote transport mechanism.

### UnregisterRegionRequest (field reference)

Proto: [proto/tensorcast/daemon/v2/store_daemon.proto](../../../proto/tensorcast/daemon/v2/store_daemon.proto)

| Field | What it does | Notes |
|---|---|---|
| `region_id` | Region identifier returned by `RegisterRegion`. | Required. |
| `owner_pid` | Owner PID verification. | Safety: mismatches fail. |
| `force` | Best-effort release even if TTL expired. | Useful for cleanup; use with care. |

`Store.unregister_vram_region(...)` is the matching SDK helper; it sends the
same `UnregisterRegionRequest` as `Store.unregister_region(...)`.

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
path from LIP registration. The daemon writes directly into existing CUDA
regions when the target layout is coalesced and matches the selected
byte-space (canonical or view-indexed). No replica is allocated.

Current limitation:

- `MaterializeIntoTarget` / `MaterializeIntoMappedTarget` still treat this as a
  VRAM-region path.
- `HOST_SHARED` direct-write currently enters through byte-artifact
  `BatchGetIntoRegion`, not the generic mapped-target API.

Boundary note:

- This API is a local external-target front-door only (caller/instance-agent ->
  local daemon).
- Cross-node cache routing must not bypass this boundary: remote/home daemons
  never write directly into caller-visible local regions, whether `VRAM` or
  `HOST_SHARED`.

### SDK preconditions

The SDK enforces strict eligibility rules before invoking
`MaterializeIntoTarget`:

- `artifact_id` is required (key-based requests are rejected).
- Canonical or view-indexed selection supported, including packed subsets
  (`tensor_names`); non-identity views resolve a deterministic `view_id`.
- All target tensors must be CUDA, contiguous, and match the selected index dtype/shape/stride.
- Coalesced layouts may span multiple storages using ordered concatenation.
- Each storage must map into a registered region and cover its logical range.

These checks are implemented in `tensorcast/api/store/materialization.py` and
`tensorcast/api/_region_cache.py`.

### Daemon validation

The daemon validates the request and the layout strictly:

- The RPC is **loopback/UDS only**; non-loopback peers are rejected before any
  write begins.
- `TargetLayout` must be `LAYOUT_KIND_COALESCED_UNSPECIFIED` with
  `INDEX_KIND_CANONICAL_UNSPECIFIED` or `INDEX_KIND_VIEW`.
- One or more storage entries, each using `vram_region_id` and
  `mapping_base_offset`, ordered by concatenation.
- `tensor_spec_kind` must be offsets or alias format.
- Offsets/lengths must match the selected index entries (canonical or view),
  and `storage_offset` must equal logical offset within the concatenated layout.
- `storage_length` must cover the selected logical size.
- `device_uuid` and `pid` are required and must match the region device.
- When `INDEX_KIND_VIEW` is used, the daemon resolves a view plan and validates
  `target_layout.view_id` against the resolved `view_id` (empty for subset-only
  layouts). `view_subset_hash` is treated as raw digest bytes and must match
  the selected `tensor_names` when provided.

### Execution

Once validated, the daemon:

1. Acquires the region from `IpcRegionRegistry` and maps its CUDA IPC handle.
2. Computes the canonical index plan and materializes directly into the region
   via `StoreEngine::materialize_into_target`.
3. Skips external-target verification by default (`engine.enable_external_target_verification=false`);
   when enabled, the daemon hashes the target ByteSpace and compares against the
   expected mi2/view hash, poisoning the region on mismatch and emitting metrics
   for enabled/skipped verification.

For mapped-target region-backed writes, controller validation still owns all
local-only and poison/publication boundaries, but the runtime now lowers the
resolved copy contract through the `0108` strategy plane before falling back to
generic byte-range execution.

## Typed Strategy Config

Region-backed and mapped-target execution now share the same typed strategy
config under `engine.materialization_strategy`. In particular:

- `enable_tensor_aware_mapped_executor` controls whether the mapped strategy
  plane may choose tensor-aware executor ops,
- `enable_owner_file_collective` controls whether owner-file collective
  execution is eligible,
- `allow_mixed_execution` controls whether the runtime may mix specialized ops
  with residual byte-range fallback,
- `diagnostics_verbosity` controls strategy-plane observability in daemon logs.

On transfer `DataLoss`, the daemon marks the region as poisoned to prevent
reuse. The client then unregisters the region from its cache.

## Deregister Artifact

`DeregisterArtifact` performs a quiesced teardown for LIP replicas:

1. Quiesce new staged exports.
2. Drain active exports if `wait_for_drain=true`.
3. Revoke the commit lease if the owner matches.
4. Best-effort unregister from Global Store.
5. By default, also tombstone and delete any managed shared-disk copies for the artifact
   (set `keep_shared_disk_copy=true` to retain shared-disk persistence).

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
| `keep_shared_disk_copy` | Preserve managed shared-disk copies for this artifact. | Default is `false` (purge shared disk on deregister). |

## TTL Extension And Transport Hold

- `DeregisterArtifactRequest.extend_ttl_ms` extends TTL before quiesce.
- `GetArtifactOptions.transport_hold_ms` requests a TTL bump during transfers.

## Failure Modes

- Owner mismatch returns `PERMISSION_DENIED`.
- Expired regions behave as missing and return `NOT_FOUND`.
- Drain timeouts return `DEADLINE_EXCEEDED` and leave the artifact quiesced.
- The drain timeout bounds both Global Store drain waits and local export drain; total wait does not exceed the requested budget.

## Code Map

- Region registry: [daemon/state/ipc_region_registry.h](../../../daemon/state/ipc_region_registry.h)
- LIP manager: [daemon/state/lip_manager.cc](../../../daemon/state/lip_manager.cc)
- Daemon RPC wiring: [daemon/service/grpc_service_impl.cc](../../../daemon/service/grpc_service_impl.cc)
- Byte-artifact region layout validation: [daemon/service/byte_artifact_region_layout.cc](../../../daemon/service/byte_artifact_region_layout.cc)
- Target-storage region acquisition: [daemon/service/controllers/materialization_target_storage_utils.cc](../../../daemon/service/controllers/materialization_target_storage_utils.cc)
- SDK region cache: [tensorcast/api/_region_cache.py](../../../tensorcast/api/_region_cache.py)
- SDK APIs: [tensorcast/api/store/__init__.py](../../../tensorcast/api/store/__init__.py)
- Local handle client helpers: [tensorcast/daemon_ctl.py](../../../tensorcast/daemon_ctl.py)
