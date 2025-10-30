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

- tensorcast/api/_register.py:894 exports a CUDA IPC handle per storage when feeding `LeaseSegment`s, so every KV page registration performs `get_cuda_memory_handle` and uploads handle bytes instead of reusing a pre-registered region.
- tensorcast/types.py:202 models `LeaseSegment` and `RegisterStorage` with `cuda_ipc_handle` fields only; there is no room to identify VRAM regions or relative offsets.
- proto/tensorcast/daemon/v1/store_daemon.proto:399 defines lease registration RPCs without region-aware fields or a deregistration verb; `LockTransportChunksRequest` lacks a TTL extension hint for long transfers.
- daemon/lip_manager.cc:118 opens each lease segment via `CudaIpcMapping::open(seg.handle_bytes, ...)`, reinforcing the per-segment handle contract and offering no integration point for region lookups.
- daemon/service/controllers/transport_controller.cc:19 mints transfer locks without quiesce coordination or TTL extension, so the daemon cannot proactively keep leases alive during long reads.
- tensorcast/global_store/services/artifact_service.py:31 handles replica `register`/`unregister` only; there is no quiesce state, drain tracking, or daemon-triggered deregister flow to keep residency data in sync.

# Phases & Milestones

- [ ] Phase 0: Protocol Foundations
  - [ ] Milestone 0.1: Extend `proto/tensorcast/daemon/v1/store_daemon.proto` with `RegisterVramRegion`, `UnregisterVramRegion`, and `DeregisterArtifact` RPCs plus a `storage_source` oneof (`cuda_ipc_handle` vs `vram_region_id`) and `extend_ttl_ms` on `LockTransportChunksRequest`.
  - [ ] Milestone 0.2: Regenerate protobuf bindings (`bash tools/build_proto_python.sh`, `bazel build //proto/...`) and update generated file exports so both C++ and Python builds compile cleanly.
  - [ ] Milestone 0.3: Expand `tensorcast/types.py` dataclasses and `tensorcast/daemon_ctl.py` adapters to serialize the new region identifiers, offsets, and deregister inputs.
- [ ] Phase 1: Daemon Runtime Integration
  - [ ] Milestone 1.1: Implement an `IpcRegionRegistry` in `daemon/` that tracks region metadata, lazily opens CUDA mappings, enforces TTL/owner rules, and exposes it through the new gRPC handlers.
  - [ ] Milestone 1.2: Extend `RegistrationController` and `LipManager` so lease feeds accept region-backed storage records, reuse open mappings by region ID, and fall back to the legacy IPC-path when no region is present.
  - [ ] Milestone 1.3: Add a quiesce-aware `deregister_artifact` path that blocks new staged exports, waits for existing locks to drain, tears down lease state, and synchronizes the removal with Global Store.
  - [ ] Milestone 1.4: Plumb `extend_ttl_ms` through `TransportController::lock` and the TTL bookkeeping so long transfers opportunistically refresh lease expirations without reopening regions.
- [ ] Phase 2: SDK & Client APIs
  - [ ] Milestone 2.1: Add `register_vram_region`, `unregister_vram_region`, and `deregister_artifact` methods to `tensorcast/api/store.py`, including lifetime management and session integration.
  - [ ] Milestone 2.2: Teach `_LeaseUploader` to detect when a tensor storage falls inside a registered region (base pointer and size comparisons), populate region IDs plus offsets in the stream payload, and drop redundant handle exports.
  - [ ] Milestone 2.3: Extend `Store.get`/`get_into` (and lease wrappers) with `transport_hold_ms` so callers can request TTL extensions during transfers; surface errors when the daemon rejects the extension.
  - [ ] Milestone 2.4: Provide a KV convenience helper that maps block hashes to `cgid:kv:<hash>` IDs and ensures artifacts register via the region-backed path.
- [ ] Phase 3: Global Store & Control Plane
  - [ ] Milestone 3.1: Update the daemon-to-GS client to send deregister requests with quiesce outcomes, and teach `ArtifactService` / `ReplicaRepository` to mark replicas draining before deletion.
  - [ ] Milestone 3.2: Ensure GS schema and migrations tolerate region-backed replicas (no new columns expected, but confirm id_kind, TTL, and availability handling) and add metrics for deregister events.
  - [ ] Milestone 3.3: Document operational expectations (region reuse, deregister behavior) in `docs/` and module READMEs per doc-sync policy.
- [ ] Phase 4: Validation & Rollout
  - [ ] Milestone 4.1: Add unit tests for the region registry, daemon deregister flow, and SDK region helpers (Catch2 + pytest).
  - [ ] Milestone 4.2: Build integration benchmarks that register thousands of KV pages with and without regions to validate latency wins and leak detection.
  - [ ] Milestone 4.3: Stage rollout behind a daemon flag, verify metrics in staging (region registrations, deregister latency, TTL extensions), then default-enable with runbook and backout steps.

# Tasks

- Define canonical invariants for region ownership (single writer, TTL coupling with artifacts) and codify them in comments plus diagnostic assertions.
- Add OpenTelemetry counters for region registration reuse, deregister wait durations, and TTL extension attempts.
- Update SDK and daemon error surfaces so missing region IDs or mismatched offsets produce actionable `FAILED_PRECONDITION` statuses.
- Wire feature toggles or capability negotiation so older clients fall back to handle-based registration without surfacing unsupported field errors.
- Refresh documentation indices (`docs/README.md`, module READMEs) to link the design and plan, summarizing operator workflows for region lifecycle management.

# Test / Rollout / Backout

- **Unit**: `bazel test //daemon:grpc_service_impl_registration_test //daemon:lip_manager_test`, `uv run pytest tests/python/test_store_region_registration.py`.
- **Integration**: `uv run pytest tests/python/test_store_session_api.py::TestLeaseInPlaceRegion`, plus multi-node smoke via `tests/python/global_store/test_artifacts.py`.
- **Performance**: Add a microbenchmark (possibly under `tests/python/perf/`) comparing region-backed vs per-page registration latency and ensure it runs in CI with fake CUDA.
- **Rollout**: Enable region-backed support behind a daemon config flag, deploy to staging, confirm GS replica counts drain correctly, then roll to production alongside updated SDK wheels.
- **Backout**: Disable the daemon flag to reject `RegisterVramRegion` while leaving existing regions untouched, roll back SDK helpers, and remove any staged migrations that depend on region data.

# Risks & Tracking

- Dangling region references if clients crash before deregistering; mitigate with TTL enforcement and daemon-side owner PID validation.
- Transfer stalls if TTL extension plumbing fails; instrument retries and surface explicit warnings to callers when holds are rejected.
- Global Store divergence when deregister fails mid-drain; require transactional updates and add alerting on stale replicas.
- Increased daemon memory pressure from cached CUDA mappings; enforce refcount caps and evict regions that fall idle beyond TTL.
