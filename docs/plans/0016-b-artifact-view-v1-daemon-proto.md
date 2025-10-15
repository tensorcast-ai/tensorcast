---
slug: 0016-artifact-view-v1-daemon-proto
title: Plan — Daemon & Proto: ViewSpec, Placement, Controller Wiring (v1)
links:
  design: ../designs/0016-artifact-view-v1.md
areas: ["daemon","proto","core"]
related_code:
  - proto/tensorcast/daemon/v1/store_daemon.proto
  - daemon/**
  - core/store/**
---

# Objective

Extend daemon RPCs and messages to carry `ViewSpec`/`view_id` and `TransformPlacement`, wire controller and engine to compute and execute view plans, and expose `view_index_json` and `view_data_hash` in responses.

# Phases & Milestones

- [x] Phase 1: Proto additions
  - [x] Add `NarrowOp`, `TransposeOp`, `Op`, `TensorViewOps`, `ViewSpec`, `TransformPlacement`
  - [x] Extend `MaterializeReplicaRequest`: `oneof view_identity { ViewSpec view=1001; string view_id=1002; }`, `TransformPlacement placement=1003`
  - [x] Extend `MaterializeReplicaResponse`: `bytes view_index_json=1001; string view_data_hash=1002`
  - [x] Update descriptor/index fields to require canonical index v3; remove enum/field references to v2

- [x] Phase 2: Controller wiring
  - [x] Map request view fields into `MaterializeHints`
  - [x] Decide server/client placement behavior (slice min-bytes; transpose client by default)
  - [x] Construct `VariantIdentity` with resolved canonical id and optional `view_id`
  - [x] Pass `VariantIdentity` through `MaterializeHints` for downstream registry lookup
  - [x] Reject requests whose descriptor/metadata advertises schema versions other than `"v3"` *(GlobalStoreClient::get_artifact_index_by_id now gates on `schema_version=="v3"` and returns `FAILED_PRECONDITION` when absent/mismatched)*

- [x] Phase 3: Engine integration
  - [x] Add optional `ViewSpec`/`view_id` handling in engine materialization
  - [x] Call `ViewPlanner` for slice; attach `view_index_json`, `view_data_hash` to response

- [x] Phase 4: Python stubs & minimal client
  - [x] Regenerate stubs via `tools/build_proto_python.sh`
  - [x] Update `tensorcast/daemon_ctl.py` to pass view fields and read response extras (`return_response=True` exposes metadata; SDK plumbing follows in plan 0016-d)

- [x] Phase 5: Variant-aware hints & orchestrator wiring
  - [x] Populate `VariantIdentity` (canonical id + view data) before invoking the engine
  - [x] Extend AUTO/P2P pathways to forward `view_id` to Global Store transport APIs
  - [x] Add integration tests covering view + canonical coexistence

# Tasks (Detailed TODO)

- [x] Update `proto/tensorcast/daemon/v1/store_daemon.proto`
  - [x] Define view messages and placement enum as per design §3
  - [x] Add fields to `MaterializeReplicaRequest/Response`
  - [x] Keep backward compatibility: when absent, follow canonical path
  - [x] Run: `bash tools/build_proto_python.sh`

- [x] Controller changes (`daemon/service/controllers/materialization_controller.{h,cc}`)
  - [x] Extend `materialize_replica` to populate hints with view fields
  - [x] Respect placement policy: slice SERVER min-bytes; transpose CLIENT by default
  - [x] Construct `VariantIdentity` with resolved canonical id and optional `view_id`
  - [x] Pass `VariantIdentity` through `MaterializeHints` for downstream registry lookup
  - [x] Reject requests whose descriptor/metadata advertises schema versions other than `"v3"` *(covered via GlobalStoreClient schema gate; controller surfaces the failure as gRPC `FAILED_PRECONDITION`)*

- [x] Engine API and call sites
  - [x] Extend `store::loading::MaterializeHints` (core) with optional view spec/id and placement
  - [x] In engine materialization, compute view plan when provided; add to return struct
  - [x] Ensure `ReplicaHandle` returned to controller carries `view_id` info for Confirm/LIP bookkeeping

- [x] AUTO orchestration and GS integration
  - [x] Update `MaterializeOrchestrator` entry to call GS `GetArtifactInfoById(space=view_id)` when variants requested (now wired via `GlobalStoreClient::get_artifact_info_by_id` view branch)
  - [x] Propagate `view_id` to `request_replica_transport` so P2P sources can honour SelectionPlan (transport descriptors carry variant identity and slice ranges)
  - [x] Fallback cleanly to canonical transport when GS lacks variant metadata; surface `PARTIAL_COVERAGE` upstream (orchestrator emits structured detail per design)

- [x] Client plumbing (`tensorcast/daemon_ctl.py`)
  - [x] Add optional arguments to materialize calls to send `view`/`view_id` and `placement`
  - [x] Expose `view_index_json`/`view_data_hash` to SDK *(available through `return_response=True`; high-level helpers will consume in plan 0016-d)*
  - [x] Attach canonical id to outgoing materialize requests to avoid ambiguity when only key is supplied (view flow enforces `artifact_id` and rejects key-only requests)
  - [x] Update client-side validation to assume canonical index v3 when parsing responses (SDK treats non-v3 layouts as errors during index hydration)

# Code Anchors

```99:116:proto/tensorcast/daemon/v1/store_daemon.proto
message MaterializeReplicaRequest {
  optional string artifact_id = 9;
  optional string disk_path = 10;
  string replica_uuid = 2;
  string device_uuid = 3;
  DeviceType target_device_type = 4;
  int32 pinned_allocation_timeout_ms = 5;
  int32 pid = 6;
  int64 size_bytes = 8;
}
```

```33:41:daemon/grpc_service_impl.cc
Status StoreDaemonServiceImpl::MaterializeReplica(
    grpc::ServerContext* ctx,
    const v1::MaterializeReplicaRequest* req,
    v1::MaterializeReplicaResponse* resp) {
  RpcContext rctx{"MaterializeReplica", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->materialize_replica(rctx, *req, *resp);
}
```

```99:116:daemon/service/controllers/materialization_controller.cc
// Engine-backed materialization
store::loading::MaterializeHints hints;
if (req.pinned_allocation_timeout_ms() > 0) {
  hints.pinned_timeout = std::chrono::milliseconds(req.pinned_allocation_timeout_ms());
}
if (has_artifact)
  hints.artifact_id = req.artifact_id();
if (has_disk)
  hints.disk_path = req.disk_path();
const auto mode = has_disk ? store::StoreEngine::MaterializeMode::LOAD_ONLY : store::StoreEngine::MaterializeMode::AUTO;
auto result = d_.engine.materialize_replica(dev, mode, hints);
```

# Commands

```bash
bash tools/build_proto_python.sh
BUILD_CORE=1 BUILD_EXTENSION=1 uv run -vvv setup.py build_ext
```

# Risks & Notes

- Maintain backward compatibility: view fields optional; canonical unchanged.
- Placement defaults to CLIENT for transpose to avoid server GPU compute.
- `view_data_hash` is computed from the loaded ByteSpace for slice requests; fast-path hits may omit it until the replica reaches `LOADED`, so SDK should continue to tolerate empty hashes.

## Status

- **Completed** — Daemon proto, controller, and client plumbing are view-aware; orchestrator + transports honour SelectionPlan with GS-backed fallbacks, and SDK validation is v3-only.
