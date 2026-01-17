---
title: tensor_dict_into Dataflow
description: Legacy and region-backed tensor_dict_into flows for SDK, daemon, and StoreEngine
areas: ["sdk", "daemon", "core"]
---

# tensor_dict_into Dataflow

This document describes how `tensor_dict_into` / `tensor_into` / `get_into` populate
caller-owned tensors. It covers the legacy daemon-owned replica path and the
region-backed path introduced in Design 0042.

## Legacy path (daemon-owned replica)

The legacy flow materializes a daemon-owned replica, exports a CUDA IPC handle,
and performs client-side copies into the caller’s target tensors.

```mermaid
sequenceDiagram
  participant SDK as SDK (Store)
  participant Daemon as Store Daemon
  participant Engine as StoreEngine
  participant GS as Global Store

  SDK->>Daemon: MaterializeByKey/MaterializeReplica (v2)
  Daemon->>GS: ResolveKeyMapping / RequestReplicaTransport
  Daemon->>Engine: materialize_replica(AUTO/LOAD_ONLY, hints)
  Engine-->>Daemon: ReplicaHandle (CUDA IPC)
  Daemon-->>SDK: mem_handle + descriptor stream
  SDK->>SDK: Validate target layout
  SDK->>SDK: Copy payloads into targets
  SDK->>Daemon: UnloadReplica
```

Key properties:

- Daemon allocates VRAM and owns the replica lifetime.
- SDK validates target tensors against the canonical index before copying.
- Verification applies when requested and uses daemon-owned sources.
- Cancellation is supported because the daemon holds the replica, not the client.

## Region-backed path (external target)

The region-backed flow streams bytes directly into a client-registered CUDA
region. The daemon does not allocate VRAM, and no CUDA IPC handle is returned.

```mermaid
sequenceDiagram
  participant SDK as SDK (Store)
  participant Daemon as Store Daemon
  participant Engine as StoreEngine
  participant GS as Global Store

  SDK->>SDK: Build TargetLayout + TargetTensorOffset
  SDK->>Daemon: MaterializeIntoTarget (artifact_id, layout, device_uuid, pid)
  Daemon->>Daemon: Acquire IpcRegionRegistry ref + map IPC handle
  Daemon->>Engine: materialize_into_target(target_ptr, total_size, index_json)
  Engine->>GS: RequestReplicaTransport (if P2P)
  Engine-->>Daemon: OK (data streamed into target)
  Daemon-->>SDK: success (no mem_handle)
```

Phase 1 constraints:

- `artifact_id` required; key-based resolution is not supported.
- Canonical index only; no view/subset (no `tensor_names` or `view_subset_hash`).
- Single coalesced storage; `storage_length` must match the logical total size.
- `device_uuid` is required and must match `storage.device_id`.
- The RPC is loopback/UDS only; `pid` is required and must match the registered
  region owner PID.
- Verification is skipped for external targets; metrics record the skip.

Failure handling:

- If the data pump starts and fails, the daemon marks the region as poisoned and
  returns `DATA_LOSS`. The SDK evicts the region from its cache so it can be
  re-registered before retrying.
- If validation fails before transfer, the SDK either falls back to the legacy
  path (`region_backed_mode=auto`) or raises `INVALID_ARGUMENT`
  (`region_backed_mode=require`).

## Capability and config gating

The SDK assumes region-backed `MaterializeIntoTarget` support and relies on the
unified runtime config `region_backed_mode` to control fallback behavior for into APIs:

- `auto`: try region-backed first; fall back when validation fails.
- `require`: enforce region-backed and surface errors.

`get` and `get_view` always use the legacy daemon-owned replica path.
