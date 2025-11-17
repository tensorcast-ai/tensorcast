---
slug: store-engine-materialization-service
title: Store Engine Materialization Service
status: draft
areas: ["core"]
related_code:
  - core/store/store_engine.cc
  - core/store/store_engine.h
  - core/store/loading/**
links:
  plan: ../plans/0025-store-engine-materialization-service.md
---

# Summary

`StoreEngine::materialize_replica` (`core/store/store_engine.cc:1238`) currently spans ~250 lines and mixes hint validation, registry lookups, copy/load orchestration, CUDA IPC export, and telemetry plumbing. This design extracts that logic into a focused `loading::MaterializationService` that owns the COPY_ONLY / LOAD_ONLY / AUTO flows, while a `MaterializationRequest` helper encapsulates all derived context (canonical IDs, device resolution, view metadata). The result is a smaller `StoreEngine`, clearer seams for future modes (RDMA staging, staged P2P), and unit-testable orchestration code that no longer requires spinning up the entire daemon.

# Goals / Non-Goals

## Goals
- Shrink `StoreEngine::materialize_replica` to a thin coordinator that delegates to well-scoped helpers.
- Encapsulate repeated validation (artifact IDs, view IDs, device ordinals) inside a reusable `MaterializationRequest`.
- Provide a `MaterializationService` entry point with explicit dependencies so COPY / LOAD / AUTO behaviors can be unit tested without daemon wiring.
- Preserve existing user-visible behavior (status codes, logs, CUDA IPC handling) and make future materialization modes pluggable.
- Update module docs to describe the new layering to keep the doc-sync rule satisfied.

## Non-Goals
- No protocol or RPC changes for daemon/materialize APIs.
- No change to `MaterializeOrchestrator` semantics beyond calling the new service.
- No redesign of replica lifecycle, UMA chunking, or eviction policies.
- Python client APIs remain untouched.

# Architecture & Interfaces

```mermaid
flowchart LR
  SE[StoreEngine
  (core/store/store_engine.cc)] --> MR[MaterializationRequest
  (loading/materialization/materialization_request.h)]
  MR --> MS[MaterializationService
  (loading/materialization/materialization_service.{h,cc})]
  MS --> REG[ReplicaRegistry]
  MS --> DEV[DeviceManager]
  MS --> MEM[PinnedBufferPool]
  MS --> COMM[CommunicationManager]
  MS --> GS[GlobalStoreClient]
  MS --> ORCH[Materialize Orchestrator]
```

## MaterializationRequest

- **Location**: `core/store/loading/materialization/materialization_request.{h,cc}`.
- **Creation**: `absl::StatusOr<MaterializationRequest> MaterializationRequest::Create(DeviceKey target, MaterializeMode mode, const loading::MaterializeHints& hints, const components::DeviceManager& device_manager)`.
- **Contents**:
  - `ReplicaKey replica_key` with normalized artifact/view IDs (reuses `VariantIdentity` when provided).
  - `DeviceKey target_device` with resolved UUID→ordinal mapping.
  - `common::memory::MemoryLocation target_location`, derived view metadata, resolved disk path (if required), canonical artifact ID, `MaterializeHints`.
  - Validation logic currently at `core/store/store_engine.cc:1244-1302`.
- **Behavior**: ensures target ordinals are valid, enforces artifact_id/disk_path requirements per mode, captures `requested_view_id`, and records whether the request can reuse GPU memory (for COPY_ONLY).

## MaterializationService

- **Location**: `core/store/loading/materialization/materialization_service.{h,cc}` with Bazel target `//core/store/loading:materialization_service`.
- **Dependencies** (injected via `struct MaterializationDeps`):
  - `components::ReplicaRegistry&`
  - `components::DeviceManager&`
  - `components::MetricsCollector&`
  - `std::shared_ptr<components::IGlobalStoreClient>`
  - `std::shared_ptr<components::CommunicationManager>`
  - `std::shared_ptr<common::memory::PinnedBufferPool>`
  - `size_t artifact_chunk_bytes`
  - `std::chrono::milliseconds pinned_memory_timeout`
  - `int num_threads`
- **API**: `absl::StatusOr<loading::ReplicaHandle> Execute(const MaterializationRequest& request);`
- **Internal helpers** (mirroring current code regions):
  - `absl::StatusOr<ReplicaHandle> TryReuseReplica(const MaterializationRequest&)` — lines 1282-1333.
  - `absl::StatusOr<ReplicaHandle> CopyFromPeer(const MaterializationRequest&)` — lines 1362-1469.
  - `absl::StatusOr<ReplicaHandle> LoadFromDisk(const MaterializationRequest&)` — lines 1336-1357, 1473-1480.
  - `absl::StatusOr<ReplicaHandle> RunAuto(const MaterializationRequest&)` — lines 1482-1494.
  - View hashing helper (current `ComputeViewDataHash` call sites at 1321-1330, 1450-1459).
  - Reusable `BuildReplicaHandle(replica::Replica&, const ReplicaKey&, MemoryLocation, std::optional<int>)`.
- **Tracing**: service methods continue to use `SC_TRACE_INIT_GUARD` via contexts passed from `StoreEngine`.

## StoreEngine integration

- `StoreEngine::materialize_replica` reduces to:
  1. `auto request_or = MaterializationRequest::Create(...);`
  2. `loading::MaterializationService svc(MaterializationDeps{...});`
  3. `return svc.Execute(*request_or);`
- `StoreEngine` exposes a private `MaterializationDeps StoreEngine::make_materialization_deps();` to keep wiring localized.
- No other public APIs change; existing `ReplicaHandle` struct stays put.

## Documentation

- `core/store/README.md` gains a "Materialization Service" subsection referencing the new files.
- `docs/architecture/architecture-overview.md` updates the Store Engine box to call out the service boundary and clarify how AUTO routing defers to `MaterializeOrchestrator`.

# Schema Changes (if any)

None. No persistent schema or protobufs are affected.

# Trade-offs & Risks

- **Additional indirection**: Introducing a service layer adds two new files, but it removes inline complexity, improves testability, and unlocks future extensions.
- **Dependency wires**: Injecting registry/device/metrics references risks cyclic includes if not kept in headers-only declarations. Mitigation: keep headers lightweight and forward-declare components.
- **Behavioral regression**: Copying logic into helpers could change subtle timings/statuses. Mitigation: keep code structure identical while moving, and back it with unit tests plus existing daemon suites.
- **Future divergence**: Without ownership discipline, new materialize features could bypass the service. Mitigation: document the boundary and update AGENTS/README to mandate using the service.

# Compatibility & Acceptance Criteria

- Public API (`StoreEngine::materialize_replica`) signature and semantics remain unchanged.
- Existing loading paths continue to honor variant hints, view IDs, CUDA IPC export, and error modes.
- Regression bar:
  - `bazel test //core/store:store_engine_test`
  - `bazel test //daemon:grpc_service_impl_registration_test --define=use_fake_cuda=true`
  - `uv run pytest tests/python/test_store_session_api.py -k materialize`
- Documentation updates merged with the code change (doc-sync rule).

# References

- `core/store/store_engine.cc:1238-1495` (current monolithic logic).
- `core/store/store_engine.h:89-238` (public API exposing materialization helpers).
- `docs/designs/0019-store-engine-modularization.md` (prior modularization effort).
- `docs/architecture/architecture-overview.md` (Store Engine overview to be updated).
