---
slug: 0055-programmable-framework
title: Plan - Programmable API Design (Artifact-First)
links:
  design: ../designs/0055-programmable-framework.md
areas:
  - sdk
  - daemon
  - global_store
  - proto
related_code:
  - tensorcast/api/store/artifact.py
  - tensorcast/api/store/batch_context.py
  - tensorcast/api/_materialize.py
  - daemon/service/controllers/materialization_controller.cc
  - daemon/state/replica_session_manager.h
  - daemon/state/session_lifecycle.h
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - proto/tensorcast/global_store/v1/global_store.proto
  - proto/tensorcast/plan/v1/plan.proto
  - proto/tensorcast/config/v1/node_agent_config.proto
  - schema.sql
---

# Objective

Implement the programmable, artifact-first control-plane primitives defined in `docs/designs/0055-programmable-framework.md`:

- `CallContext`: per-call QoS/deadline/idempotency/tags
- `Operation[T]`: unified sync/blocking status/wait/cancel (Phase-0: no `async def` / `await` SDK surface)
- `pin_device_residency`: process-independent device residency intent (placement pins)
- `Plan`: orchestration IR (worker-first, instance steps via node-local agents)

## API Consistency Rules (required)

- `CallContext` is a pure data container (no `ctx.artifact()` / `ctx.plan()` helpers).
- Handle construction is always context-free: `tensorcast.artifact(...)` / `store.artifact(...)`.
- `ctx` is always passed explicitly as an action parameter: all RPC/scheduling actions accept `*, ctx: CallContext | None = None` (keyword-only).
- `ctx` does not participate in artifact/view/selection identity; only `ctx.idempotency_key` may seed deterministic operation ids (deadline/qos/tags must never affect identity).
- Plan is the only exception: `tensorcast.plan(ctx)` binds `ctx` once; step builder methods do not accept `ctx`.

# Status (2026-01-24)

- Completed Phases 0–3 end-to-end (protos, daemon RPCs/semantics, SDK `CallContext`/`Operation`/`prefetch`/`pin_device_residency`, Global Store `daemon_id` identity).
- Phase 4 is complete: PlanSpec + node agent execution (worker + instance steps) + engine adapter registry are implemented.
- Instance registry and Node Agent runtime config are implemented and wired through the Global Store and unified config system.
- Worker registration invariants are enforced end-to-end (required `daemon_id`, endpoint conflict checks independent of `node_id`,
  `daemon_id` persisted as NOT NULL).

# Prior State & Grounding (before implementation)

- SDK:
  - `Artifact.prefetch(...) -> Operation[PrefetchedReplica]` with operation-scoped wait/cancel/status; no legacy ticket
    wrapper remains.
  - `tensorcast/api/_materialize.py` sends `lease_mode=NO_LEASE` for prefetch to avoid PID-scoped leases.
- Daemon:
  - `proto/tensorcast/daemon/v2/store_daemon.proto` defines `QueryReplicaStatus` and `ReleaseReplica` RPCs, but
    `daemon/service/grpc_service_impl.h` does not override them, so calls are `UNIMPLEMENTED`.
  - `daemon/state/replica_session_manager.h` overwrites `replica_uuid -> ReplicaKey` on `put(...)`; deterministic ids
    are unsafe without PutIfAbsent/JoinIfMatch semantics.
  - `daemon/service/controllers/materialization_controller.cc` is explicit that PID-coupled paths are loopback-only
    (`wait_for_completion` and `MaterializeIntoTarget`).
  - `daemon/state/session_lifecycle.h` already supports placement pinning via
    `SessionLifecycleManager::create_placement_lease(...)`, but it is not exposed via RPC and has no capability token
    model.
- Global Store:
  - `proto/tensorcast/global_store/v1/global_store.proto` has worker identity via `worker_id`, but no stable `daemon_id`.
  - `schema.sql` stores `workers(worker_id, node_id, node_address, grpc_port, ...)` with a uniqueness constraint on
    `(node_address, grpc_port)`, which makes address/port part of identity.
  - Python service code is in `tensorcast/global_store/grpc_service.py` + `tensorcast/global_store/services/worker_service.py`.
  - Daemon HA registration is in `daemon/ha/worker_lifecycle_manager.cc` via `core/store/components/global_store_client.*`.

# Work Breakdown by Area

- **Protos**
  - Daemon: `proto/tensorcast/daemon/v2/store_daemon.proto`
  - Global Store: `proto/tensorcast/global_store/v1/global_store.proto`
  - PlanSpec: `proto/tensorcast/plan/v1/plan.proto`
  - Node Agent: `proto/tensorcast/node_agent/v1/node_agent.proto`
  - Node Agent config: `proto/tensorcast/config/v1/node_agent_config.proto`
  - (Optional but recommended) Daemon config identity: `proto/tensorcast/config/v1/daemon_config.proto`
- **Daemon (C++)**
  - gRPC surface: `daemon/service/grpc_service_impl.{h,cc}`
  - Materialization + lease semantics: `daemon/service/controllers/materialization_controller.cc`
  - Operation/session state: `daemon/state/replica_session_manager.h`, `daemon/state/session_lifecycle.{h,cc}`
  - HA registration: `daemon/ha/worker_lifecycle_manager.{h,cc}`, `core/store/components/global_store_client.{h,cc}`
- **SDK (Python)**
  - Context/operation primitives: `tensorcast/api/context.py`, `tensorcast/api/operation.py`
  - Artifact handle changes: `tensorcast/api/store/artifact.py`, `tensorcast/api/_materialize.py`
  - Daemon RPC client: `tensorcast/daemon_ctl.py`
- **Global Store (Python)**
  - Schema: `schema.sql`
  - Models/repos: `tensorcast/global_store/models/worker.py`, `tensorcast/global_store/repositories/worker_repository.py`
  - Service layer: `tensorcast/global_store/services/worker_service.py`, `tensorcast/global_store/grpc_service.py`
  - Instance registry: `tensorcast/global_store/models/instance.py`,
    `tensorcast/global_store/repositories/instance_repository.py`,
    `tensorcast/global_store/services/instance_service.py`

# Phases & Milestones

- [x] Phase 0: Proto scaffolding + codegen (additive-only)
  - [x] Milestone 0.1: Daemon proto for `NO_LEASE` + operation waits
    - [x] Add `LeaseMode` enum (or equivalent) and `lease_mode` fields to `MaterializeReplicaRequest` and
      `MaterializeByKeyRequest` in `proto/tensorcast/daemon/v2/store_daemon.proto`.
    - [x] Add `WaitReplicaStatus` RPC + request/response messages in `proto/tensorcast/daemon/v2/store_daemon.proto`
      (prefer server-streaming; acceptable alternative is long-poll unary).
    - [x] Add placement lease RPCs + messages in `proto/tensorcast/daemon/v2/store_daemon.proto`:
      `CreatePlacementLease`, `RenewPlacementLease`, `ReleasePlacementLease` returning/accepting `lease_token` (SDK surfaces it as `PlacementPin.capability_token`).
    - [x] Add `WaitPersistenceStatus` RPC in `proto/tensorcast/daemon/v2/store_daemon.proto` (Phase-1; mirrors
      `WaitReplicaStatus` to avoid polling storms).
  - [x] Milestone 0.2: Global Store proto for stable daemon identity
    - [x] Add `daemon_id` to `RegisterWorkerRequest` and `ListActiveWorkersResponse.WorkerInfo` in
      `proto/tensorcast/global_store/v1/global_store.proto`.
    - [x] Decide whether `WorkerHeartbeatRequest` needs `daemon_id` (recommended: keep heartbeat by `worker_id`, but
      allow optional `daemon_id` for sanity-check logging).
  - [x] Milestone 0.3: (Optional) Daemon config identity
    - [x] If `daemon_id` is not derivable safely from existing fields, add an explicit `daemon_id` config field to
      `proto/tensorcast/config/v1/daemon_config.proto` and plumb it through YAML/JSON config parsing (must follow
      `docs/designs/0004-unified-runtime-config.md`).
  - [x] Milestone 0.4: Regenerate + validate protos
    - [x] Run `bash tools/build_proto_python.sh` (required after any `.proto` changes).
    - [x] Run `bazel test //proto/... --test_output=streamed`.

- [x] Phase 1: Daemon operation tracking + `NO_LEASE` (C++)
  - [x] Milestone 1.1: PutIfAbsent/JoinIfMatch for `replica_uuid` operation records
    - [x] Replace overwriting `ReplicaSessionManager::put(...)` with PutIfAbsent/JoinIfMatch semantics in
      `daemon/state/replica_session_manager.h` (reject mismatched reuse with `FAILED_PRECONDITION`).
    - [x] Make session entries carry enough identity for join validation (at minimum `ReplicaKey`; recommended:
      `replica_key_hash` for diagnostics).
    - [x] Update `daemon/state/sessions_service.h` call sites accordingly (avoid silent overwrite).
    - [x] Add a focused test: new `daemon/state/replica_session_manager_join_semantics_test.cc` + target in `daemon/BUILD`.
      - Run: `bazel test //daemon:replica_session_manager_join_semantics_test`.
  - [x] Milestone 1.2: `NO_LEASE` materialization semantics
    - [x] Implement `lease_mode=NO_LEASE` behavior in `daemon/service/controllers/materialization_controller.cc`:
      - never mint IPC handle leases (`MemCopyHandle.lease_token`)
      - never create PID use leases (`SessionLifecycleManager::create_use_lease`)
      - ignore `pid` even for loopback peers (treat as `effective_pid=0`)
      - omit `mem_handle` (or return an explicit empty handle) for `NO_LEASE` requests
    - [x] Add daemon tests for `NO_LEASE` behavior:
      - New: `daemon/service/grpc_service_impl_no_lease_materialize_test.cc` (or equivalent) + target in `daemon/BUILD`.
      - Run: `bazel test //daemon:grpc_service_impl_no_lease_materialize_test --test_env=TENSORCAST_CUDA_BACKEND=fake`.
  - [x] Milestone 1.3: Operation-scoped status/wait/cancel RPCs
    - [x] Implement RPC handlers in `daemon/service/grpc_service_impl.{h,cc}`:
      - `QueryReplicaStatus`
      - `ReleaseReplica` (operation-scoped; must not unload replicas globally)
      - `WaitReplicaStatus`
    - [x] Factor logic into a controller if needed:
      - Option A: extend `daemon/service/controllers/status_controller.{h,cc}`
      - Option B (preferred for cohesion): add `daemon/service/controllers/replica_ticket_controller.{h,cc}`
    - [x] Add tests:
      - New: `daemon/service/grpc_service_impl_replica_status_test.cc` verifying Query/Wait/Release semantics.
      - Run: `bazel test //daemon:grpc_service_impl_replica_status_test --test_env=TENSORCAST_CUDA_BACKEND=fake`.
  - [x] Milestone 1.4: Placement lease RPCs + capability tokens
    - [x] Implement `CreatePlacementLease`/`RenewPlacementLease`/`ReleasePlacementLease` in
      `daemon/service/grpc_service_impl.{h,cc}`.
    - [x] Build a daemon-scoped capability model:
      - Option A: maintain a token→lease-id map in a new module `daemon/state/placement_lease_tokens.{h,cc}` with TTL.
      - Option B: encode a MAC/signed token (requires a daemon secret; store in unified config).
    - [x] Wire to lifecycle: `daemon/state/session_lifecycle.{h,cc}` (`create_placement_lease`, `renew_placement`,
      `release_lease`).
    - [x] Add tests:
      - New: `daemon/service/grpc_service_impl_placement_lease_test.cc` (token required; renew/release fail on invalid).
      - Run: `bazel test //daemon:grpc_service_impl_placement_lease_test --test_env=TENSORCAST_CUDA_BACKEND=fake`.

- [x] Phase 2: SDK surfaces (`CallContext`, `Operation`, `prefetch`, `pin_device_residency`)
  - [x] Milestone 2.1: `CallContext` plumbing
    - [x] Add `tensorcast/api/context.py` (`CallContext`, `tensorcast.context(...)`) and re-export from `tensorcast/__init__.py`.
    - [x] Keep `CallContext` as a pure data container: do not add `CallContext.artifact()` or accept `ctx` in `tensorcast.artifact(...)` / `Store.artifact(...)`.
    - [x] Thread `ctx` through `tensorcast/api/_materialize.py` materialization entry points (deadlines, tags, idempotency).
    - [x] Add tests: `tests/python/api/test_call_context.py`.
      - Run: `uv run pytest tests/python/api/test_call_context.py`.
  - [x] Milestone 2.2: `Operation[T]` primitive (sync/blocking only)
    - [x] Add `tensorcast/api/operation.py` defining `Operation[T]` + status/error models from the design.
    - [x] Implement at least one concrete backend:
      - `DaemonReplicaOperation` (uses `WaitReplicaStatus`, `QueryReplicaStatus`, `ReleaseReplica`)
      - `PollingOperation` for persistence (Phase-0)
    - [x] Add tests: `tests/python/api/test_operation_semantics.py`.
      - Run: `uv run pytest tests/python/api/test_operation_semantics.py`.
  - [x] Milestone 2.3: `Artifact.prefetch(...) -> Operation[PrefetchedReplica]`
    - [x] Update `tensorcast/api/store/artifact.py`:
      - change return type and remove the legacy tuple contract
      - compute deterministic `replica_uuid` when `ctx.idempotency_key` is set
      - default `lease_mode=NO_LEASE`
    - [x] Update `tensorcast/api/_materialize.py` + `tensorcast/daemon_ctl.py` to send/handle `lease_mode` and call
      `WaitReplicaStatus`.
    - [x] Remove `PrefetchTicket` legacy wrapper from `tensorcast/api/store/batch_context.py`
    - [x] Add/adjust tests:
      - Add `tests/python/api/test_prefetch_operation.py`
      - Update `tests/python/api/test_public_surface.py` for new API surface
      - Run: `TENSORCAST_CUDA_BACKEND=fake uv run pytest tests/python/api/test_prefetch_operation.py`.
  - [x] Milestone 2.4: `pin_device_residency` SDK API
    - [x] Implement `Artifact.pin_device_residency(...) -> Operation[PlacementPin]` in `tensorcast/api/store/artifact.py`.
    - [x] Implement `PlacementPin.renew/release` calling daemon placement pin RPCs via `tensorcast/daemon_ctl.py`.
    - [x] Add tests: `tests/python/api/test_placement_pin.py`.
      - Run: `TENSORCAST_CUDA_BACKEND=fake uv run pytest tests/python/api/test_placement_pin.py`.
  - [x] Milestone 2.5: Lint/type checks
    - [x] Run `uv run ruff check .` and `uv run ruff format .`.
    - [x] Run `uv run mypy ./tensorcast`.

- [x] Phase 3: Global Store stable worker identity (`daemon_id`) + schema
  - [x] Milestone 3.1: Schema changes (`schema.sql`)
    - [x] Add `daemon_id` column to `workers` plus a uniqueness constraint/index (DuckDB-compatible, NOT NULL).
    - [x] Update `tests/python/global_store/test_schema_persistence.py` expectations if needed.
  - [x] Milestone 3.2: Global Store service plumbing (Python)
    - [x] Update worker model + repo:
      - `tensorcast/global_store/models/worker.py` (add `daemon_id`)
      - `tensorcast/global_store/repositories/worker_repository.py` (SELECT/INSERT/UPDATE, add `find_by_daemon_id`)
    - [x] Update registration logic to treat `daemon_id` as stable identity:
      - `tensorcast/global_store/services/worker_service.py` (prefer `daemon_id` for upsert; allow address changes)
      - `tensorcast/global_store/grpc_service.py` (plumb proto fields; return `daemon_id` in ListActiveWorkers)
    - [x] Add/adjust tests:
      - `tests/python/global_store/test_grpc_service.py`
      - `tests/python/global_store/test_repositories.py`
      - Run: `uv run pytest tests/python/global_store/...`.
  - [x] Milestone 3.3: Daemon HA registration includes `daemon_id` (C++)
    - [x] Plumb `daemon_id` through daemon options/config:
      - daemon wiring: `daemon/app/*` (composition root)
      - config parsing: `tensorcast/api/_config.py` (if Python-side config helpers involved) and/or daemon C++ config load
    - [x] Include `daemon_id` in Global Store registration:
      - `core/store/components/global_store_client.{h,cc}` (RegisterWorkerRequest)
      - `daemon/ha/worker_lifecycle_manager.{h,cc}`
    - [x] Update HA tests:
      - `daemon/ha/worker_lifecycle_manager_sync_test.cc` assertions to validate `daemon_id` propagation.
      - Run: `bazel test //daemon:worker_lifecycle_manager_sync_test`.

- [x] Phase 4: PlanSpec + node-local agents + engine adapter capabilities
  - [x] Milestone 4.1: PlanSpec (versioned IR)
    - [x] Define `PlanSpec` proto (new): `proto/tensorcast/plan/v1/plan.proto` (or similar) and generate code.
    - [x] Implement `tensorcast/api/plan/plan.py` + `Plan.to_spec()` producing deterministic fingerprints based on
      `(logical_layout_hash, selection_hash)` and stable target identities (`daemon_id`, `instance_id`).
  - [x] Milestone 4.2: Node-local agent service (worker + instance steps)
    - [x] Define agent RPC proto (new): `proto/tensorcast/node_agent/v1/node_agent.proto`.
    - [x] Implement node agent (Python service recommended in v1):
      - `tensorcast/node_agent/server.py` (new)
      - `tensorcast/node_agent/executor.py` (Plan step execution; talks to daemon + engine adapter)
  - [x] Milestone 4.3: Engine adapter capabilities
    - [x] Define `TargetSpec` mint/resolve API data model at the engine boundary and implement capability validation in the SDK layer.
    - [x] Implement `TransformSpec` plugin registry and execution hooks at node-local boundary.
  - [x] Milestone 4.4: Instance registry + Node Agent config
    - [x] Add Global Store instance registry (schema + repo + service + RPCs).
    - [x] Add Node Agent unified config (`NodeAgentConfig`) + entrypoint.
  - [x] Milestone 4.5: Plan execution integration tests
    - [x] Add Node Agent PlanSpec execution tests (success + failure/cancel paths).

# Tests & Acceptance Criteria

- [x] Deterministic operation ids:
  - [x] Re-submitting the same action with the same `CallContext.idempotency_key` returns the same `operation_id`.
  - [x] Reusing an existing `replica_uuid` for a different `ReplicaKey` fails with `FAILED_PRECONDITION` (no overwrite).
- [x] `NO_LEASE` correctness:
  - [x] `NO_LEASE` materialization creates no PID-bound use leases and no handle leases (even for loopback peers).
- [x] Cancellation isolation:
  - [x] Two distinct `replica_uuid`s can map to the same `ReplicaKey`; canceling one does not affect the other.
- [x] Placement leases:
  - [x] `lease_token` is required for renew/release; invalid/expired tokens fail safely.
- [x] PlanSpec determinism:
  - [x] `Plan.to_spec()` is stable for identical inputs (serialization deterministic).
- [x] Node Agent plan execution:
  - [x] Instance step success + failure/cancel coverage (`tests/python/node_agent/test_plan_execution.py`).
- [x] Instance registry:
  - [x] Register/list/unregister instance lifecycle coverage (`tests/python/global_store/test_instances.py`).

# Rollout & Backout

- No legacy compatibility path is required; the SDK ships Operation-only prefetch.
- Backout strategy: revert the 0055 change set as a unit (no dual-path feature gates).
