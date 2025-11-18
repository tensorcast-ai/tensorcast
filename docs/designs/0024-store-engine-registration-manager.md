---
slug: store-engine-registration-manager
title: Store Engine Registration Manager Extraction
areas: ["core","daemon"]
links:
  plan: ../plans/0024-store-engine-registration-manager.md
related_code:
  - core/store/store_engine.*
  - core/store/components/registration/**
  - daemon/**
---

# Summary

`StoreEngine` currently embeds the entire RFC-0006 memory-registration lifecycle (begin → ingest → commit/abort → TTL upkeep). That logic spans ~500 lines scattered across `core/store/store_engine.{h,cc}` and entangles replica management, CUDA IPC export, view ingestion, and Global Store updates in a single class. This design extracts the registration lifecycle into a dedicated `components::ArtifactRegistrationManager` (ARM) so the engine regains focus on orchestration while registration behavior becomes modular, testable, and reusable by other daemons if needed. Because the project has not been released publicly yet, we do not preserve legacy shims or transitional logic—the refactor goes directly to the desired long-term architecture.

Key outcomes:
- Thin `StoreEngine` surface: registration APIs delegate to ARM without managing mutexes or view ingestion directly.
- First-class component: ARM owns pending state, TTL expiry, view executors, hashing, and Global Store publication behind a narrow interface.
- Targeted tests and observability: the new component exposes hooks for unit tests and emits metrics scoped to registration health, reducing regression risk.

# Goals / Non-Goals

Goals
- Encapsulate registration state and synchronization inside `ArtifactRegistrationManager`, shrinking `StoreEngine` and clarifying ownership boundaries.
- Define the simplified architecture without keeping legacy compat layers or transitional flags; code immediately reflects the final desired structure.
- Provide reusable helpers (e.g., `RegistrationResources`, `ReplicaFactory`) so other modules can request registrations without depending on full `StoreEngine`.
- Improve testability by enabling unit-level coverage for TTL expiry, view ingestion, and hashing without bootstrapping the engine.

Non-Goals
- Do not change the public daemon RPC schemas or CLI (no need for compatibility branches, but the observable API surface stays the same).
- Do not alter replica loading/materialization paths (`materialize_replica`, disk/p2p ingest) beyond delegating registration calls.
- Do not introduce new identity kinds or schema updates—ARM simply orchestrates existing flows.

# Architecture & Interfaces

## 1. Component boundaries

```mermaid
flowchart LR
    A[StoreEngine API<br>(Begin/Commit/etc.)] --> B[ArtifactRegistrationManager]
    B --> C[ReplicaRegistry<br>& DeviceManager]
    B --> D[CommunicationManager<br>& PinnedBufferPool]
    B --> E[Global Store Client]
    B --> F[View Planner / Executors]
```

- `StoreEngine` retains lifecycle ownership (construct/destruct, worker identity) but treats registration as a dependency.
- `ArtifactRegistrationManager` lives under `core/store/components/registration/` and is constructed with:
  - `RegistrationResources`: non-owning references to `DeviceManager`, `ReplicaRegistry`, `MetricsCollector`, `PinnedBufferPool`, and optional `CommunicationManager` / `IGlobalStoreClient`.
  - `ReplicaFactory`: callback (e.g., `std::function<std::shared_ptr<replica::Replica>(const replica::ReplicaConfig&)>`) used to instantiate memory-only replicas without exposing engine internals.
  - `WorkerIdentity`: node id/address + ports for Global Store publication; updated via `set_worker_identity`.

## 2. Registration lifecycle

- **Pending state**: ARM introduces `PendingRegistrationContext` (successor to `PendingRegistrationEntry`) that captures allocation metadata, CUDA IPC handles, view options, TTL, and hashing intent. The context lives entirely within ARM’s mutex and is stored in `absl::flat_hash_map<std::string, std::shared_ptr<PendingRegistrationContext>>`.
- **Begin flow**: `Begin(const ArtifactRegistration&) -> RegistrationBeginResult` encapsulates logic formerly in `StoreEngine::begin_register_artifact`, including GPU capacity checks (`components::eviction_service`), Inline buffer replica creation, and view plan computation. ARM is responsible for producing CUDA IPC handle bytes and caching GPU pointers for later ingestion.
- **Ingest/KeepAlive**: ARM exposes `IngestViewChunk`, `GetViewIngestedBytes`, and `KeepAlive` methods with identical semantics, enforcing placement rules (`SERVER` vs `CLIENT`) and TTL extension.
- **Commit/Abort**: `Commit(registration_id)` finalizes view executors, zero-fills uncovered regions, computes canonical hashes via `common::compute_index_multihash` and `loader::verification::compute_data_multihash`, updates Global Store, registers communication exports, and relinquishes pending state. `Abort(registration_id)` cancels and releases memory via the replica factory.
- **TTL enforcement**: ARM owns expiration timestamps and performs cleanup on demand (during Commit/KeepAlive) initially; optional background sweeper can be added later without touching `StoreEngine`.

## 3. Worker identity & Global Store integration

- ARM accepts a `WorkerIdentity` struct mirroring the fields currently stored in `StoreEngine` (`worker_id_`, `node_id_`, `node_address_`, `grpc_port_`, `p2p_port_`). `StoreEngine::set_worker_identity` simply forwards the data to ARM.
- Global Store interaction is mediated entirely through ARM using the shared `IGlobalStoreClient` reference, keeping registration logic close to the data it manages.
- Export registration (`enable_remote_replica_access`) remains in `StoreEngine`; ARM only calls the existing helper through the provided callback when necessary, ensuring no duplication.

## 4. Metrics & instrumentation

- ARM reports registration health to `MetricsCollector` via helper methods that keep gauges/histograms colocated with pending state. The component exposes:
  - `tc_register_pending_gauge` — async gauge updated on every Begin/Commit/Abort/TTL cleanup so dashboards show in‑flight registrations per daemon.
  - `tc_register_commit_seconds{result="ok|aborted|expired"}` — histogram capturing wall time between Begin and outcome, allowing latency SLIs split by result.
- Begin/Commit paths run within `SC_TRACE_INIT_GUARD` scopes so trace timelines now show registration spans without additional logic inside `StoreEngine`.

# Schema Changes (if any)

None. The refactor reuses existing data models and RPC payloads.

# Trade-offs & Risks

- **New dependency layer**: Introducing ARM adds another constructor dependency. Mitigation: keep the interface small (Begin / Commit / Abort / KeepAlive / View APIs) and document required resources.
- **In-flight behavior parity**: Moving logic risks regressions (especially for view ingestion and TTL). Mitigation: migrate code with minimal edits, add dedicated unit tests, and run existing daemon integration tests.
- **Shared ownership updates**: `StoreEngine` setters (e.g., worker identity) now need to forward state to ARM. Mitigation: keep single source of truth in ARM and ensure accessors remain in `StoreEngine` only for compatibility.

# Compatibility & Acceptance Criteria

- No backward-compatibility shims or dual code paths remain; `StoreEngine` delegates entirely to ARM and legacy registration code is removed.
- RPC contracts (`BeginRegisterArtifact`, `CommitRegisteredArtifact`, etc.) are still honored, but implementation details are simplified rather than preserved for compatibility.
- Registration paths continue to support both direct canonical indexes and variant/view registrations, including CUDA IPC handle semantics.
- Perf/latency regressions stay within noise; eviction and memory checks behave the same because logic is migrated byte-for-byte.
- Documentation reflects the new component ownership (core/store/README.md, docs/architecture/architecture-overview.md mention ARM).

# References

- Existing implementation: `core/store/store_engine.cc:1898-2710`, `core/store/store_engine.h:395-454`.
- Eviction helpers: `core/store/components/eviction_service.h`.
- View planning/execution: `core/store/materialization/dataplane/view/view_planner.h`, `core/store/materialization/dataplane/view/view_ingest_executor.h`.
- Doc system spec: `docs/designs/0001-docs-system-design.md`.
