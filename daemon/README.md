---
title: TensorCast Store Daemon (C++)
description: What the daemon is, its boundaries, responsibilities, and interfaces
sidebar_position: 3
---

# Store Daemon

The Store Daemon is the data-plane service process that exposes a stable gRPC API over the C++ StoreEngine. It validates and routes requests, coordinates per-request orchestration, and tracks ephemeral state (sessions, PID refs, locks). It does not own domain invariants implemented in the engine.

- Binary: `//daemon:tensorcast_daemon`
- Public API: `../proto/tensorcast/daemon/v1/store_daemon.proto` (Python import: `tensorcast.proto.daemon.v1`)
- Deployment: see `../docs/deployment/store-daemon.md`

## What It Does

- gRPC surface for artifact loading, lifecycle, key mapping, and status.
- Orchestration via controllers with strong input validation and deadline handling.
- Ephemeral state management with TTL: sessions (`replica_uuid` → key + readiness), PID references, transport locks, verification tracking.
- Event-driven background scheduling with a unified `SessionLifecycleTask` (sessions TTL, PID liveness, registration join TTL), plus Lock TTL and Verification tasks.
- Observability wrappers that attach unified metrics and tracing to each RPC.
- Eviction consults lifecycle counters (use_count, placement_pins) rather than legacy `keep_for_global` flags.
- Maps `MaterializeReplica(keep_for_global=true)` to a TTL prefetch pin by creating a PlacementLease on GPU replicas (default TTL 10 minutes).

## What It Does Not Do

- Reimplement engine invariants: memory lifecycle, DVMP/UMA model, verification semantics, or DVMP chunk-locking rules.
- Own long-lived cache policy beyond explicit hints (e.g., `keep_for_global`). Eviction policies live below or behind explicit feature flags.
- Bypass the StoreEngine for data movement or memory management.
- Break wire compatibility. Protocol changes are additive and guarded.

## Layering and Boundaries

```mermaid
flowchart TB
  SVC[StoreDaemonService (gRPC)] --> CTRLS[Controllers]
  CTRLS --> MGRS[Managers/Registries]
  CTRLS --> OBS[RpcContext (metrics/tracing)]
  MGRS --> RUNTIME[BackgroundScheduler]
  MGRS --> ENGINE[StoreEngine]
  CTRLS --> ENGINE
```

- Service (gRPC): constructs dependencies, gates shutdown, validates inputs, maps `absl::Status` to gRPC.
- Controllers:
  - MaterializationController: `Materialize*`, `Confirm`, `WaitVerification`, `Unload`, `GetArtifactIndexById`.
  - RegistrationController: `Begin`/`Feed`/`KeepAlive`/`Commit`/`Abort`/`Revoke` (unified feed path).
  - TransportController: `LockTransportChunks`/`UnlockTransportChunks`.
  - StatusController: `GetServerConfig`, `GetWorkerStatus`, `GetDetailedStatus`, `GetLoadedReplicasV2`.
- Managers/Registries:
  - RegistrationManager, SessionsService + ReplicaSessionManager, RefTracker, TransportLockManager, VerificationTracker, LipManager/LipBridge.
  - Runtime: BackgroundScheduler runs the unified `SessionLifecycleTask` for sessions/PID/join TTL, plus Lock TTL and Verification tasks, with “sleep until deadline or signal” semantics. PID liveness is event-driven via a `PidMonitor` (pidfd + epoll) with a `/proc` polling fallback when pidfd is unavailable.
- Engine: single source of truth for materialization orchestration, memory lifecycle, DVMP locking semantics, verification futures.

## Interfaces (Public Surface)

- Loading: `MaterializeByKey` (preferred), `MaterializeReplica`, `ConfirmReplica`, `UnloadReplica`, `WaitReplicaVerification`.
- Key mapping: `PublishReplicaKey`, `ResolveKeyMapping`, `GetArtifactIndexById`.
- Status: `GetServerConfig`, `GetWorkerStatus`, `GetDetailedStatus`, `GetLoadedReplicasV2` (paginated).
- Transport: `LockTransportChunks`, `UnlockTransportChunks`.
- In-memory registration: `BeginRegisterArtifact`, `FeedRegisterArtifactStream`, `KeepAliveRegisterArtifact`, `CommitRegisteredArtifact`, `AbortRegisteredArtifact`, `RevokeRegisteredArtifact`.

Contract highlights:
- `Materialize*` returns after allocation with a CUDA IPC handle; clients must `ConfirmReplica` and may `WaitReplicaVerification`.
- `MaterializeByKey` performs key resolution and P2P-first loading with disk fallback inside the daemon; clients do not implement fallback.
- Transport locks infer a unique device when `device_id` is absent; ambiguity returns `INVALID_ARGUMENT`.

## Invariants and Guardrails

- Thin service: only validation, orchestration, status mapping. No duplicated engine logic.
- RAII for external resources (e.g., CUDA IPC) to prevent leaks and double-close.
- TTL for every ephemeral map; all TTL updates and cleanup run under BackgroundScheduler.
- Consistent deadlines: user timeouts are clamped to RPC deadlines.
- Observability is best-effort and never changes control flow; high-cardinality fields are gated.
- Idempotent unload and lock cleanup; expired tokens are unlocked automatically.

## Directory Layout (What lives here)

- `grpc_service_impl.{h,cc}`: thin gRPC service entry points and dependency wiring.
- `service/controllers/*`: controllers for materialization, registration, transport, and status.
- `registration_manager.h`, `replica_session_manager.h`, `sessions_service.h`: unified registration/session lifecycles with TTL.
- `ref_tracker.h`: PID reference tracking; liveness integrated via `SessionLifecycleTask`.
- `transport_lock_manager.h`: tokenized chunk locking with TTL and best-effort unlock.
- `verification_tracker.h`: verification futures, capacity and expiry-based eviction.
- `lip_manager.{h,cc}`, `lip_bridge.{h,cc}`: LIP fast path and cross-device helpers.
- `background_scheduler.h`, `session_lifecycle.h`, `sweep_tasks.h`: event-driven runtime scheduler and lifecycle/task definitions.
- `rpc_context.h`, `grpc_span.h`, `grpc_metrics.h`, `deadline_utils.h`, `device_resolver.h`, `status_utils.h`.
- `worker_lifecycle_manager.{h,cc}`: integration with Global Store (register/heartbeat/reconcile).
- `server_main.cc`: flags/bootstrap and service registration.

## Build, Run, Test

- Build binary: `bazel build //daemon:tensorcast_daemon`
- Launch with unified config: see `../docs/deployment/store-daemon.md`
- C++ tests: `bazel test //daemon:grpc_service_impl_registration_test` (and related `*_test.cc` in this directory)

## Ownership

- Area: `daemon`
- Owners: Store Daemon maintainers (see CODEOWNERS)

## Key Links

- Architecture overview: `../docs/architecture/architecture-overview.md`
- Model loading internals: `../docs/internals/model-loading.md`
- P2P transfer strategies: `../docs/architecture/p2p-transfer-strategies.md`
- Store Engine internals: `../core/store/README.md`
- Global Store internals: `../tensorcast/global_store/README.md`
- Deployment guide: `../docs/deployment/store-daemon.md`

---
This README captures “what and boundaries” of the daemon after the completed architecture and gRPC service refactor. It avoids leaking engine abstractions into the service layer and keeps responsibilities explicit for future evolution.
