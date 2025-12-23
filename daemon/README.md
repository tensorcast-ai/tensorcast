---
title: TensorCast Store Daemon (C++)
description: What the daemon is, its boundaries, responsibilities, and interfaces
sidebar_position: 3
---

# Store Daemon

The Store Daemon is the data-plane service process that exposes a stable gRPC API over the C++ StoreEngine. It validates and routes requests, coordinates per-request orchestration, and tracks ephemeral state (sessions, PID refs, locks). It does not own domain invariants implemented in the engine.

- Binary: `//daemon:tensorcast_daemon`
- Public API: `../proto/tensorcast/daemon/v1/store_daemon.proto` (Python import: `tensorcast.proto.daemon.v1`); streaming descriptor v2 surface lives at `../proto/tensorcast/daemon/v2/store_daemon.proto` (Python import: `tensorcast.proto.daemon.v2`) and is always enabled.
- Deployment: see `../docs/deployment/store-daemon.md`

## What It Does

- gRPC surface for artifact loading, lifecycle, key mapping, and status.
- Orchestration via controllers with strong input validation and deadline handling.
- Ephemeral state management with TTL: sessions (`replica_uuid` → key + readiness), PID references, transport locks, verification tracking.
- Event-driven background scheduling with a unified `SessionLifecycleTask` (sessions TTL, PID liveness, registration join TTL), plus Lock TTL and Verification tasks. The lifecycle manager exposes a schedule hook so the scheduler can be rescheduled immediately when the earliest deadline changes, minimizing expiry drift.
- Observability wrappers that attach unified metrics and tracing to each RPC.
- Handles view registration uploads (slice/transpose) without feature flags: `RegistrationController` streams view chunks to the StoreEngine and publishes variant telemetry through Global Store (see [Variant View Registration Telemetry](../docs/architecture/p2p-transfer-strategies.md#variant-view-registration-telemetry)).
- Enforces non-loopback advertisement when registering with Global Store; startup fails if no routable IP can be determined and `--advertise_host` is unset.
- Immediate reclaim: when the last UseLease retires and no PlacementPins remain for a daemon-owned GPU replica, the lifecycle finalizer unloads the replica immediately (best-effort).
- Eviction consults lifecycle counters (use_count, placement_pins); request-level cache hints removed.
- Join TTL via leases: duplicate coalesced commits (`existed=true`) create a TTL-bound UseLease for the owner PID. On expiry, the lease finalizer drops the lightweight RefTracker ref and may reclaim memory immediately.
- Session keepalive: session principal TTL is tracked by the lifecycle manager via deadline guards instead of a standalone sweep.
- PID unwatch: when the last pid-bound guard retires, the monitor stops watching that PID (pidfd/epoll cleaned). The PID monitor’s polling fallback interval is configurable via service options (`proc_check_interval`).


## What It Does Not Do

- Reimplement engine invariants: memory lifecycle, UMA ledger model, verification semantics. Transfer chunk-locking is removed (handled by UMA plan/commit and VS pin leases).
- Own long-lived cache policy is out of scope for the daemon. Eviction policies live below or behind explicit feature flags.
- Bypass the StoreEngine for data movement or memory management.
- Protocol changes are additive and guarded; no backward‑compat bridging layers are maintained.

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
- Engine: single source of truth for materialization orchestration, memory lifecycle, UMA ledger semantics, verification readiness.

## Interfaces (Public Surface)

- Loading: `MaterializeByKey` (preferred), `MaterializeReplica`, `ConfirmReplica`, `UnloadReplica`, `WaitReplicaVerification`.
- Loading v2 (gated): `MaterializeByKey`/`MaterializeReplica` with descriptor payloads plus `GetMaterializeCapabilities` (SDK probe) under `StoreDaemonServiceV2`. Descriptors are derived from UMA view plans (offset/stride/byte-length) when a view is requested so the exported buffer layout matches the planner, and disk fallbacks stay daemon-owned via `DiskFallbackHint` (respecting `verify_checksums`).
- Loading v2 (region-backed): `MaterializeIntoTarget` streams bytes directly into a client-registered CUDA region when the SDK supplies a full coalesced `TargetLayout` (`layout_kind=LAYOUT_KIND_COALESCED_UNSPECIFIED`, `index_kind=INDEX_KIND_CANONICAL_UNSPECIFIED`, `tensor_spec_kind=TENSOR_SPEC_KIND_OFFSETS`) and `artifact_id`. The daemon validates layout/device constraints, maps the IPC handle, and never allocates a daemon-owned replica.
- Disk fallbacks honor `verify_checksums` on `DiskFallbackHint`/`MaterializeReplicaRequest` and propagate the flag into engine `MaterializeHints` so checksum/descriptor validation is enforced by default but can be disabled for local development.
- Key mapping: `PublishReplicaKey`, `ResolveKeyMapping`, `GetArtifactIndexById`.
- Status: `GetServerConfig`, `GetWorkerStatus`, `GetDetailedStatus`, `GetLoadedReplicasV2` (paginated).
- Transport: `LockTransportChunks`, `UnlockTransportChunks`.
- In-memory registration: `BeginRegisterArtifact`, `FeedRegisterArtifactStream`, `KeepAliveRegisterArtifact`, `CommitRegisteredArtifact`, `AbortRegisteredArtifact`, `RevokeRegisteredArtifact`.
- Persistence (Design 0041): `StartPersistence`/`QueryPersistenceStatus` run through a shard-aware persistence manager that calls Global Store `PlanPlacement`, requests stable leases per shard target, acknowledges lease acquisition, registers remote replicas back to Global Store, and reports status. Shards follow UMA chunking (128MB sharding threshold, 64–256MB shard caps); targets include the local node, any planned remotes, and a shared-disk leg that deduplicates by shard digest (skipped when the digest already exists on the daemon). Background ticks drive bounded lease retries with exponential cooldown, task state, and metrics (`tc_persist_tasks_active`, `tc_persist_errors_total`, `tc_persist_progress_ratio`). An append-only task log at `/tmp/tensorcast_persistence.log` (configurable via `Options.persistence_log_path`) is replayed on startup so in-flight persistence can resume/report after restart. The persistence manager reuses the Global Store client injected by `WorkerLifecycleManager` and is updated with the mutex-guarded `node_id` on registration so lease ACK/reporting carries the registered identity, and status reports are assembled under the persistence mutex then dispatched after releasing it so slow or failed Global Store RPCs cannot stall the scheduler loop. Query responses include shard targets and lease ids (index-aligned; empty lease_id means pending) alongside task-level progress/error/degraded fields; shared-disk failures are treated as fatal even when remotes succeed.
- V2 requests carry `DiskFallbackHint` + `SourcePreference` so the daemon owns all disk reads; the v2 gRPC service is always registered alongside v1 for legacy clients.

Contract highlights:
- `Materialize*` returns after allocation with a CUDA IPC handle; clients must `ConfirmReplica` and may `WaitReplicaVerification`.
- `MaterializeByKey` performs key resolution and P2P-first loading with disk fallback inside the daemon; clients do not implement fallback.
- `MaterializeReplica` shares the same LIP fast-path semantics; same-device denial from LIP is treated as a cache miss and falls back to the engine path rather than surfacing an RPC failure.
- `MaterializeIntoTarget` requires canonical layouts and `artifact_id` in Phase 1, skips verification, and returns `DATA_LOSS` on post-start failures after poisoning the region to prevent reuse.
- Transport locks infer a unique device when `device_id` is absent; ambiguity returns `INVALID_ARGUMENT`.

### Variant Views (v1)

- `MaterializeReplicaRequest` accepts optional `view` (deterministic slice/transpose spec) or `view_id` along with a `placement` hint (`SERVER` for slice/min-byte, `CLIENT` for transpose). Requests using `view` require the canonical `artifact_id` so the daemon can normalise against canonical index v3.
- When a non-identity view is requested, `MaterializeReplicaResponse` populates `view_index_json` (canonicalised layout for the requested ByteSpace) and `view_data_hash` when verification completes. For canonical disk-source loads, `view_index_json` is also filled from the artifact directory so clients can reconstruct tensors without querying Global Store.
- The controller collapses identity views to the canonical path, preserves LIP fast-path semantics, and forwards `VariantIdentity` into the StoreEngine so replica keys and telemetry track `(artifact_id, view_id)` tuples.

## Invariants and Guardrails

- Thin service: only validation, orchestration, status mapping. No duplicated engine logic.
- RAII for external resources (e.g., CUDA IPC) to prevent leaks and double-close.
- Canonical index rebuild mirrors the SDK path: storage-level offsets and lengths are emitted for every alias so tensor views dedupe across disk, coalesced VRAM, and LIP flows.
- TTL for every ephemeral map; all TTL updates and cleanup run under BackgroundScheduler.
- Consistent deadlines: user timeouts are clamped to RPC deadlines.
- Memory tiers: `engine.memory_tiers` drives stable/preemptible behavior; startup fails if configured `stable_bytes` exceed `(cgroup|MemTotal) - mem_pool_size_bytes`, and preemptible markings are skipped entirely when `enable_preemptible` is false. When HA is enabled, the daemon publishes `MemoryTierStatus` heartbeats via `MemoryTierService` using the UMA-backed `MemoryTierBudget` and replays `MemoryTierLease` state on startup/heartbeats (ListOutstandingLeases → UMA stable lease bind/release → ACK carrying artifact_id + chunk_ids + ledger_version for audit).
- Observability is best-effort and never changes control flow; high-cardinality fields are gated. Background HA counters (heartbeat/sync) tolerate missing meters so registration and sync continue even when telemetry providers are unavailable.
- Idempotent unload and lock cleanup; expired tokens are unlocked automatically.
- Lease-in-place commits rebuild the canonical tensor index from the fed `storage_entries` and `tensor_aliases`, emitting `tc_register_storage_count` / `tc_register_tensor_count` metrics so rollouts can confirm dedupe efficacy.

## Directory Layout (What lives here)

- `grpc_service_impl.{h,cc}`: thin gRPC service entry points and dependency wiring.
- `service/controllers/*`: controllers for materialization, registration, transport, and status.
- `registration_manager.h`, `replica_session_manager.h`, `sessions_service.h`: unified registration/session lifecycles with TTL.
- `ref_tracker.h`: PID reference tracking; liveness integrated via `SessionLifecycleTask`.
- `transport_lock_manager.h`: tokenized chunk locking with TTL and best-effort unlock.
- `verification_tracker.h`: completion-driven verification tracking (ReadySignal subscriptions) with capacity/expiry pruning.
- `lip_manager.{h,cc}`, `lip_bridge.{h,cc}`: LIP fast path and cross-device helpers.
- `background_scheduler.h`, `session_lifecycle.h`, `sweep_tasks.h`: event-driven runtime scheduler and lifecycle/task definitions.
- `ipc_region_registry.{h,cc}`: tracks client-registered CUDA IPC regions with TTL, refcounts, and poison state.
- `rpc_context.h`, `grpc_span.h`, `grpc_metrics.h`, `deadline_utils.h`, `device_resolver.h`, `status_utils.h`.
- `worker_lifecycle_manager.{h,cc}`: integration with Global Store (register/heartbeat/reconcile) using a constructor-built `gsl::not_null` `GlobalStoreClient`, fixed node identity captured at construction, and a mutable worker id that is populated during registration; `start()` just performs the handshake and initial registration, and shutdown signals interruptible waits so stop exits promptly.
- `server_main.cc`: flags/bootstrap and service registration.

## Build, Run, Test

- Build binary: `bazel build //daemon:tensorcast_daemon`
- Launch with unified config: see `../docs/deployment/store-daemon.md`
- C++ tests: `bazel test //daemon:grpc_service_impl_registration_test` (and related `*_test.cc` in this directory)

## Error Handling Conventions

- Use wrappers in `daemon/common/safe_sys.h` (e.g., `safe_epoll_add`, `safe_epoll_del`, `safe_eventfd_write/read`) instead of raw syscalls; they map errno to Status and treat expected cases (EEXIST/ENOENT) as OK.

These conventions improve observability while keeping best‑effort paths fast and non‑fatal.

## Key Links

- Architecture overview: `../docs/architecture/architecture-overview.md`
- Model loading internals: `../docs/internals/model-loading.md`
- P2P transfer strategies: `../docs/architecture/p2p-transfer-strategies.md`
- Store Engine internals: `../core/store/README.md`
- Global Store internals: `../tensorcast/global_store/README.md`
- Deployment guide: `../docs/deployment/store-daemon.md`

---
This README captures “what and boundaries” of the daemon after the completed architecture and gRPC service refactor. It avoids leaking engine abstractions into the service layer and keeps responsibilities explicit for future evolution.
