---
slug: region-backed-registration
title: Region-Backed Registration (Plan)
areas: ["core","daemon","sdk","global_store","proto"]
links:
  design: ../designs/0021-region‑backed-registration.md
related_code:
  - daemon/**
  - core/store/**
  - tensorcast/api/**
  - tensorcast/global_store/**
  - proto/tensorcast/daemon/v1/**
---

# Objective

Deliver the region-backed lease-in-place flow described in the design so paged-attention KV caches pre-register large VRAM slabs once, register KV pages by referencing those slabs, and reclaim storage safely through a quiesced deregistration path that keeps Global Store metadata consistent.

# Current State & Grounding

- Design intent is now captured in `docs/designs/0021-region‑backed-registration.md`, including proto additions, daemon `IpcRegionRegistry`, SDK APIs, and Global Store deregister flow.
- tensorcast/api/_register.py implements region-backed `RegisterStorage` and `LeaseSegment` emission; storages within registered regions use `storage_source = vram_region_id` with `region_base_offset` and drop per-storage handle exports.
- tensorcast/types.py models `LeaseSegment`/`RegisterStorage` with `storage_source` oneof and region fields.
- proto/tensorcast/daemon/v1/store_daemon.proto includes `RegisterVramRegion`/`UnregisterVramRegion`/`DeregisterArtifact`, `storage_source` oneof, and `extend_ttl_ms` on `LockTransportChunksRequest`.
- daemon/lip_manager.cc reuses region-backed mappings via `IpcRegionRegistry` and validates offsets/lengths.
- LipManager staged exports now retain region registry references for the duration of the export (held in `LipExportRecord`) and release them on `release_staged_export`, preventing premature region ref release while CUDA mappings remain active.
- daemon/service/controllers/transport_controller.cc applies `extend_ttl_ms` before staging exports; quiesce and drain are handled by `DeregisterArtifact`.
- tensorcast/global_store/services/artifact_service.py adds `unregister_by_worker` to support daemon-triggered deregister; server exposes `UnregisterReplicaByWorker` RPC.
- LIP data paths and region registry are integrated; region mappings are cached and reused.

# Phases & Milestones

- [ ] Phase 0: Protocol Foundations
 - [x] Phase 0: Protocol Foundations
  - [x] Milestone 0.1: Extend `proto/tensorcast/daemon/v1/store_daemon.proto` with `RegisterVramRegion`, `UnregisterVramRegion`, and `DeregisterArtifact` RPCs plus a `storage_source` oneof (`cuda_ipc_handle` vs `vram_region_id`) and `extend_ttl_ms` on `LockTransportChunksRequest`.
  - [x] Milestone 0.2: Regenerate protobuf bindings (`bash tools/build_proto_python.sh`, `bazel build //proto/...`) and update generated file exports so both C++ and Python builds compile cleanly.
  - [x] Milestone 0.3: Expand `tensorcast/types.py` dataclasses and `tensorcast/daemon_ctl.py` adapters to serialize the new region identifiers, offsets, and deregister inputs.
- [x] Phase 1: Daemon Runtime Integration
  - [x] Milestone 1.1: Implement an `IpcRegionRegistry` in `daemon/` that tracks region metadata, lazily opens CUDA mappings, enforces TTL/owner rules, and exposes it through the new gRPC handlers. (Region registry now exists with register/unregister RPCs and periodic sweeper; mapping reuse deferred to Milestone 1.2.)
  - [x] Milestone 1.2: Extend `RegistrationController` and `LipManager` so lease feeds accept region-backed storage records, reuse open mappings by region ID, and fall back to the legacy IPC-path when no region is present. (Lease feeds validate regions and pin references; `LipManager` reuses region-backed mappings for both coalesced copy and staged exports. Legacy handle path remains as fallback.)
  - [x] Milestone 1.3: Add a quiesce-aware `deregister_artifact` path that blocks new staged exports, waits for existing locks to drain, and tears down lease state. (Global Store synchronization remains pending in Phase 3.)
  - [x] Milestone 1.4: Plumb `extend_ttl_ms` through `TransportController::lock` and the TTL bookkeeping so long transfers opportunistically refresh lease expirations without reopening regions.
- [x] Phase 2: SDK & Client APIs
  - [x] Milestone 2.1: Add `register_vram_region`, `unregister_vram_region`, and `deregister_artifact` methods to `tensorcast/api/store.py`, including lifetime management and session integration.
  - [x] Milestone 2.2: Teach `_LeaseUploader` to detect when a tensor storage falls inside a registered region (base pointer and size comparisons), populate region IDs plus offsets in the stream payload, and drop redundant handle exports. (Implemented via client-side region cache and region-backed `RegisterStorage`/`LeaseSegment` payloads.)
  - [x] Milestone 2.3: Extend `Store.get`/`get_into` (and lease wrappers) with `transport_hold_ms` so callers can request TTL extensions during transfers; surface errors when the daemon rejects the extension. (SDK options extended via `GetArtifactOptions.transport_hold_ms`; daemon TTL bump is applied by `TransportController::lock`.)
  - [x] Milestone 2.4: Provide a KV convenience helper that maps block hashes to `cgid:kv:<hash>` IDs and ensures artifacts register via the region-backed path. (Added `tensorcast.api.store.register_kv_block`.)
- [x] Phase 3: Global Store & Control Plane
  - [x] Milestone 3.1: Update the daemon-to-GS client to send deregister requests with quiesce outcomes, and teach `ArtifactService` / `ReplicaRepository` to mark replicas draining before deletion. (Implemented via `UnregisterReplicaByWorker`; immediate delete, no schema change.)
  - [x] Milestone 3.2: Ensure GS schema and migrations tolerate region-backed replicas (no new columns expected, but confirm id_kind, TTL, and availability handling) and add metrics for deregister events. (Reused existing unregister metrics; schema unchanged.)
  - [x] Milestone 3.3: Document operational expectations (region reuse, deregister behavior) in `docs/` and module READMEs per doc-sync policy. (Plan doc updated; module docs to be refreshed in roll-out tasks.)
- [x] Phase 4: Validation & Rollout
  - [x] Milestone 4.1: Add unit tests for the region registry, daemon deregister flow, and SDK region helpers (Catch2 + pytest). Added `//daemon:ipc_region_registry_test` and `tests/python/test_store_region_registration.py`.

# Tasks

- Define canonical invariants for region ownership (single writer, TTL coupling with artifacts) and codify them in comments plus diagnostic assertions.
- Add OpenTelemetry counters/histograms (`tensorcast.daemon.region.*`, `tensorcast.daemon.lip.deregister_wait_ms`, `tensorcast.global_store.artifact.deregister_*`) for region registration reuse, deregister wait durations, and TTL extension attempts.
- Update SDK and daemon error surfaces so missing region IDs or mismatched offsets produce actionable `FAILED_PRECONDITION` statuses.
- Add capability negotiation so older clients fall back to handle-based registration without surfacing unsupported field errors. (No daemon flag; always enabled)
- Refresh documentation indices (`docs/README.md`, module READMEs) to link the design and plan, summarizing operator workflows for region lifecycle management.

# Test / Rollout / Backout

- **Unit**: `bazel test //daemon:grpc_service_impl_registration_test //daemon:ipc_region_registry_test`, `uv run pytest tests/python/test_store_region_registration.py`.
- **Integration**: `uv run pytest tests/python/test_store_session_api.py::TestLeaseInPlaceRegion`, plus multi-node smoke via `tests/python/global_store/test_artifacts.py`.
- **Performance**: Add a microbenchmark (possibly under `tests/python/perf/`) comparing region-backed vs per-page registration latency and ensure it runs in CI with fake CUDA.
- **Rollout**: Deploy to staging, confirm GS replica counts drain correctly and metrics look healthy (region registrations, deregister latency, TTL extensions), then roll to production alongside updated SDK wheels.
- **Backout**: Roll back the daemon to a prior version if needed; SDK helpers already fall back to handle-based registration when daemon lacks region support.

# Risks & Tracking

- Dangling region references if clients crash before deregistering; mitigate with TTL enforcement and daemon-side owner PID validation.
- Transfer stalls if TTL extension plumbing fails; instrument retries and surface explicit warnings to callers when holds are rejected.
- Global Store divergence when deregister fails mid-drain; require transactional updates and add alerting on stale replicas.
- Increased daemon memory pressure from cached CUDA mappings; enforce refcount caps and evict regions that fall idle beyond TTL.
