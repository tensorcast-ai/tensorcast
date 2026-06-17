# Global Store — Developer Guide

The Global Store is TensorCast's central control plane for distributed artifact coordination. It provides the essential "where is it?" and "who can serve it?" answers that enable the rest of the system to function.

## Why Does This Exist?

In a distributed artifact storage system, before any data transfer can happen, a fundamental question must be answered: **given an artifact identifier, which nodes hold a replica capable of serving it?**

The Global Store exists because:

1. **Locality matters for performance.** Without centralized tracking, clients would need to broadcast queries across all nodes—an O(N) operation that scales poorly. The Global Store provides O(1) lookups.

2. **Concurrency requires coordination.** Multiple clients requesting the same artifact simultaneously need load balancing to avoid overwhelming a single source. The Global Store maintains real-time request counters and enforces concurrency limits.

3. **Failures require recovery.** When a Store Daemon crashes and restarts, it needs to reconcile its local state with what the cluster believes it holds. The Global Store enables this reconciliation without requiring distributed consensus protocols.

4. **Content-addressing enables deduplication.** Two independent saves of identical tensor data should resolve to the same artifact. The Global Store maintains the canonical mapping from content hashes to artifacts.

## Architecture Philosophy

### Layered Design

The codebase follows a classic three-tier architecture, but with intentional constraints:

```
┌─────────────────────────────────────────────────────────────────┐
│                        gRPC Facade                              │
│  Thin. No business logic. Only:                                 │
│  • Proto ↔ domain object conversion                             │
│  • Input validation (structural, not semantic)                  │
│  • Error mapping to gRPC status codes                           │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│                       Service Layer                             │
│  Owns all business rules:                                       │
│  • Constraint validation (routable addresses, consistent sizes) │
│  • Cross-entity orchestration (register replica → update worker)│
│  • Transaction boundaries and atomic operations                 │
│  • Metrics emission                                             │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│                      Repository Layer                           │
│  Pure data access:                                              │
│  • Thread-local cursor management                               │
│  • SQL query composition                                        │
│  • Result mapping to domain objects                             │
│  • Atomic UPSERT patterns                                       │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                        ┌───────▼───────┐
                        │    DuckDB     │
                        └───────────────┘
```

**Why this separation?**

- The gRPC layer can change (new RPC methods, different wire formats) without touching business logic.
- The service layer is independently testable—unit tests don't need gRPC machinery.
- The repository layer can be swapped (different databases, caching strategies) without rewriting business rules.

### Embedded Database Choice

The Global Store uses DuckDB—an embedded analytical database—rather than a client-server database like PostgreSQL. This choice reflects several design priorities:

1. **Deployment simplicity.** No separate database process to manage, no network configuration, no connection pooling tuning.

2. **Single-node operation is sufficient.** The Global Store is designed for single-node deployment. Cluster coordination happens through the Global Store, not within it.

3. **Analytical query patterns.** Many Global Store queries involve aggregations (count replicas per artifact, find least-loaded source). DuckDB excels at these.

4. **Thread-safety model.** DuckDB handles multi-threading through cursor isolation—each thread gets its own view. This maps cleanly to gRPC's thread-per-request model.

## Core Abstractions

### Identity Model

TensorCast supports two artifact identity schemes:

| Scheme | Format | When to Use |
|--------|--------|-------------|
| **MI2** | `mi2:<index_mh>:<data_mh>` | Content-addressed identity derived from artifact bytes. Enables deduplication. |
| **CGID** | Client-generated string | Pre-computed identifiers for external integration or testing. |

The `artifacts` table stores both schemes with `id_kind` discriminating. MI2 identities are derived from multihashes of the tensor index and data, ensuring that identical saves resolve to identical IDs.

### View and Assembly Awareness

Global Store now treats replicas as ByteSpace-scoped:

- Replicas are keyed by `(artifact_id, view_id)` via `MemoryInfo.byte_space` so
  canonical and view pieces route independently.
- View metadata persists canonical coverage ranges and `view_data_hash` so
  daemons can assemble and seal from pieces (`ListViews` returns ranges).
- `artifact_bindings` records `assembly_id → mi2_id` after sealing so reads can
  resolve to the sealed identity while preserving view routing semantics.
- Assembly CGIDs that provide `tensor_index_data` (or an `ArtifactDescriptor`
  index multihash) persist `index_multihash` so `GetArtifactIndexById` can
  resolve canonical index bytes before sealing.
- Replica domain models expose `ByteSpaceKind` and `ByteSpaceRef` for routing
  and checksum identity across canonical vs view byte spaces.

### Replica Lifecycle

A replica represents a single copy of an artifact on a specific node. Its lifecycle:

```
                    RegisterReplica
                          │
                          ▼
┌─────────────────────────────────────────┐
│              is_available = true        │
│  current_requests ∈ [0, max_concurrency]│
└────────────────────┬────────────────────┘
                     │
       ┌─────────────┴─────────────┐
       │                           │
       ▼                           ▼
  RequestTransport            UnregisterReplica
  (increments counter)              │
       │                            ▼
       │                    [record deleted]
       ▼
  CompleteTransport
  (decrements counter)
```

**Key invariant:** A replica can serve at most `max_concurrency` simultaneous transfers. The Global Store enforces this by atomically incrementing the counter during `RequestTransport` and checking against the limit.

### Worker Lifecycle

Workers (Store Daemons) register with the Global Store and send periodic heartbeats:

```
                    RegisterWorker
                          │
                          ▼
┌─────────────────────────────────────────┐
│           worker_id assigned            │
│    accepting_new_requests = true        │
└────────────────────┬────────────────────┘
                     │
       ┌─────────────┴─────────────┐
       │                           │
       ▼                           ▼
   WorkerHeartbeat           UnregisterWorker
   (resets timeout)                │
       │                           ▼
       │                   [worker marked inactive]
       │                   [replicas marked unavailable]
       │
       ▼
  (heartbeat timeout exceeded)
       │
       ▼
  [cleanup thread marks worker inactive]
  [replicas marked unavailable]
```

Workers must provide a stable `daemon_id` (auto-generated and persisted when omitted, or explicitly configured). Registration treats `daemon_id` as the
primary identity for upserts so a daemon can restart or change advertised address/port without losing its logical
identity. `worker_id` remains an assigned row identifier; `ListActiveWorkers` returns both `worker_id` and `daemon_id`.

When a new daemon registration arrives for an endpoint (`node_address:grpc_port`) that is still occupied by an older row,
Global Store performs endpoint takeover: the previous endpoint owner is reclaimed and the new registration becomes authoritative
without waiting for heartbeat timeout cleanup. This avoids restart races after unclean daemon exits.

Inactive workers remain in the registry with `inactive_at` set so routing can filter them while avoiding delete/update conflicts.

**Design decision: batched heartbeats.** Workers send heartbeats every ~5 seconds. Writing each immediately would create unnecessary database load. Instead, `WorkerService` buffers heartbeats in memory and a background thread flushes batches every ~100ms. This trades sub-second staleness for dramatically reduced write amplification.

### Instance Lifecycle

Engine instances (user processes) register with the Global Store and send periodic heartbeats:

```
                 RegisterInstance
                        │
                        ▼
┌─────────────────────────────────────────┐
│         instance_id assigned            │
└────────────────────┬────────────────────┘
                     │
              InstanceHeartbeat
               (resets timeout)
                     │
                     ▼
             [marked inactive]
```

Instances are keyed by a stable `instance_id` and associated with a `daemon_id` (and
optionally a `worker_id`) to bridge engine processes with the node’s store daemon.
Routable execution hosts publish explicit `execution_endpoint` and
`execution_host_kind` facts; `signals_endpoint` remains observability-only and
is no longer the execution-routing field.

### Capability Directory (Discovery)

The worker/instance registries also act as a low-frequency **capability directory**:

- Each worker/instance persists a bounded `capability_flags` bitset (see `global_store.proto`) advertising control-plane
  capabilities (queue broker, retention handles, capability-token envelope, execution signals, node agent).
- Writes are **update-on-change**: heartbeats update `capability_flags` only when the field is present and the value
  changes. Omitting the field leaves the stored flags untouched; sending an explicit `0` clears capabilities.
- `ListActiveWorkers` / `ListActiveInstances` accept `required_capability_flags` for server-side filtering; responses
  always include the active `capability_flags`.
- `ListActiveInstances` also returns explicit execution routing facts
  (`execution_endpoint`, `execution_host_kind`) so daemon-side directory readers
  can map `instance_id -> execution endpoint` without overloading
  `signals_endpoint`.
- Clients should cache directory results with bounded staleness; this is **advisory discovery**, not a hot path.

**Metrics:** `tc_capability_directory_entries{scope="worker|instance", capability="..."}` tracks active capability
counts by scope and capability.

## Service Responsibilities

### WorkerService

**Purpose:** Manage Store Daemon lifecycle.

**Key behaviors:**
- Generates unique worker IDs incorporating node identity and UUID suffix (avoids collisions across restarts)
- Requires stable `daemon_id` identity (allows address changes without changing `worker_id`)
- Validates address uniqueness—two workers cannot register the same `(address, port)` from different daemon IDs
- Detects stale workers via heartbeat timeout, marks them inactive, and cleans up their replicas

### ArtifactService

**Purpose:** Manage the replica registry.

**Key behaviors:**
- Validates transport metadata consistency (buffer sizes must sum to memory size)
- Uses atomic UPSERT to handle concurrent registrations of the same replica
- Updates Prometheus gauges after registration/unregistration

### TransportService

**Purpose:** Coordinate artifact transfers with load balancing.

**Key behaviors:**
- Enqueues requests into a pending queue and dispatches globally
- Applies group-aware ordering (fairness floor, completion bias, starvation aging)
- Applies source balancing at assignment time (replica load, worker load, assignment recency, diffusion bonus)
- Atomically claims replica capacity and creates transport lease records
- Requires explicit request-level idempotency (`request_id`) and replays duplicate requests safely
- Supports blocking wait with timeout for high-contention scenarios
- Provides swap-safety helpers: `MarkReplicaUnavailable` flips `is_available=false` for a replica, and `WaitReplicaDrain`
  waits until `current_requests==0` without mutating transport state; the drain wait honors the requested timeout and
  RPC deadline by capping sleep intervals to the remaining time budget
- Cleans up stale transports as safety net for crashed clients
- Separates lease closure from requester outcome: completion always releases counters, while group progress uses
  explicit `SUCCESS` outcomes only

**Dispatch strategy (unified path):**
1. Queue and rank pending requests with fairness floor + completion bias + starvation aging.
2. Select replica with source-balance scoring and memory-tier preference (GPU > RAM > DISK).
3. Finalize assignment transactionally with idempotent request replay protection.

### RecoveryService

**Purpose:** Handle high-availability state reconciliation.

**Key behaviors:**
- On Global Store startup, marks all workers/replicas as stale until they re-confirm
- Handles worker re-registration by transferring replicas to new worker ID
- Computes state diffs between worker's local inventory and global state
- Persists `state_version` and `state_checksum` per worker (non-null defaults); heartbeats use cached checksum to avoid full-table scans
  - Tracks per-worker reconcile cursor (`generation`, `request_seq`) and daemon incarnation (`daemon_id`) to reject stale requests and trigger explicit rebase
  - Applies reconcile changes transactionally and only bumps `state_version` + checksum on full success (`NOOP` keeps version, refreshes checksum if needed)

### InstanceService

**Purpose:** Manage engine instance registrations.

**Key behaviors:**
- Validates instance identity (instance_id, daemon_id, engine).
- Updates worker association on heartbeat when daemon_id resolves to a new worker_id.
- Marks inactive instances when heartbeats expire.
- Validates consistency via a stable FNV-1a checksum over sorted replica state (aligned with the daemon)
- Applies endpoint/metadata drift updates (node address/port, memory size, transport keys when reported) during sync
- HA checksum format is `artifact_id:node_id:node_address:node_port:device_id:memory_type:available;` (FNV-1a 64-bit).

**Safe-removal semantics:** When a worker reports an empty inventory, the Global Store does NOT blindly delete all its replicas. An empty list might mean "I haven't enumerated yet" rather than "I have nothing." Removals only apply when the worker explicitly provides a non-empty inventory that excludes replicas the Global Store expects.

### MemoryTierService

**Purpose:** Track UMA memory accounting and lease lifecycle.

**Key behaviors:**
- Persists telemetry snapshots with configurable retention (time-based and row-count-based)
- Manages stable/preemptible memory lease states: `pending → acquired → released` or `pending → revoking → expired`
- Feeds Prometheus gauges for capacity monitoring
- Exposes `ListMemoryTierStatuses` to return the latest per-node telemetry snapshots and `ListOutstandingLeases` for live lease state

### PlacementService

**Purpose:** Plan shard placement and ingest persistence status. See `../../docs/architecture/api/policy-persistence.md`.

**Key behaviors:**
- Generates placement plans that always include the source node and attempt a stable-DRAM remote target when policy is `replicated` or `sharded`; degrades to local-only when no remote capacity is available.
- Persists plan metadata across normalized tables (`artifact_placements`, `artifact_placement_shards`, `artifact_placement_targets`) plus an optional JSON summary for quick fetches.
- Accepts daemon `ReportPersistenceStatus` updates to update per-target state and append task status rows in `artifact_persistence_status`.

### ChunkService

**Purpose:** Support distributed memory pool via chunk directory.

**Key behaviors:**
- Tracks chunk locations across nodes with state (`HOT`, `LOCKED_TX`, `COPIED_GPU`, `COLD`, `EVICTED`)
- Returns chunk locations sorted by node load for intelligent source selection
- Batch updates chunk states from daemon heartbeats

### ViewStateService

**Purpose:** Manage view metadata, TreeHash leaves, and overlap proof digests (v2).

**Key behaviors:**
- Persists view specifications (transformed views of canonical artifacts) and canonical coverage ranges for pieces.
- Exposes `CheckProofCommitmentsMatch` to compare assembly-scoped and MI2-scoped proof commitments for specified
  tensors (used by post-seal view reuse policies).
- Stores Merkle leaf digests keyed by **HashSpaceRef**:
  - canonical hash-space is anchored by `index_multihash`
  - view hash-space is anchored by `view_id`
- Supports partial verification by querying specific leaf indices
  - When requested leaf digests are missing, `GetArtifactInfoById` returns `STATUS_NOT_FOUND` and populates
    `partial_leaf_coverage` (units: leaf indices). `partial_coverage` is reserved for missing *byte* coverage ranges
    (units: bytes) and must not be used for leaf indices.

## Data Model

The canonical schema lives in `/schema.sql`. Key tables:

| Table | Purpose | Hot/Cold |
|-------|---------|----------|
| `workers` | Store Daemon registrations with address uniqueness | Cold |
| `artifacts` | Content-addressed artifact metadata (MI2/CGID) | Cold |
| `artifact_replicas` | Replica descriptors (location, memory tier, availability) | Cold |
| `replica_counters` | Request counters split from replicas | **Hot** |
| `artifact_indices` | Deduplicated tensor index blobs (by SHA-256) | Cold |
| `artifact_transports` | In-flight transport records | Warm |
| `chunk_directory` | VS/UMA chunk-level location tracking | Warm |
| `artifact_placements` | Normalized persistence plan headers | Cold |
| `artifact_placement_shards` | Shard summaries (size, range, digest, chunk_ids) | Cold |
| `artifact_placement_targets` | Shard target nodes and lease IDs | Warm |
| `artifact_placement_summary` | Optional cached plan JSON for fast lookups | Cold |
| `artifact_persistence_status` | Task-level persistence status and progress | Warm |
| `key_mappings` | Human-friendly key → artifact_id lookup | Cold |
| `memory_tier_snapshots` | UMA telemetry time-series | Warm |
| `memory_tier_leases` | Preemptible memory lease lifecycle | Warm |
| `views` | View metadata for transformed artifacts | Cold |
| `view_coverage_ranges` | Canonical coverage ranges for views (pieces) | Cold |
| `leaves` | Merkle leaf digests for integrity verification | Cold |
| `layout_specs` | Immutable, content-addressed layout declarations (v2) | Cold |
| `assembly_layout_bindings` | Versioned `assembly_id → layout_id` pointer (v2) | Warm |
| `assembly_attempts` | Immutable per-attempt truth keyed by `attempt_id` | Warm |
| `assembly_readiness_cuts` | Durable readiness evidence captured before seal | Warm |
| `assembly_slot_occupancies` | Durable `(attempt_id, slot_id)` occupancy state | Warm |
| `artifact_layout_attachments` | Immutable `mi2_id → layout_id` attachments (v2) | Cold |
| `operations` | Unified operation status + coordinator leases (v2) | Warm |
| `assembly_proof_commitments` | Assembly-scoped proof commitments (v2) | Warm |
| `tensor_proof_commitments` | MI2-scoped proof commitments (v2) | Cold |
| `piece_proof_digests` | Per-piece proof digests (v2) | Warm |

**Why separate `replica_counters` from `artifact_replicas`?**

Replica descriptive fields (address, memory size, keys) rarely change. But `current_requests` changes on every transport request/completion—potentially hundreds of times per second across the cluster. Separating them:

1. Reduces write amplification on the main table
2. Enables targeted VACUUM on the hot table
3. Avoids row-level locking contention

## Concurrency Considerations

### Thread-Safe Database Access

Each gRPC request handler runs in its own thread. The Global Store achieves thread safety through cursor isolation:

```python
class BaseRepository:
    def get_cursor(self):
        return self.connection.cursor()  # Thread-local cursor
```

DuckDB cursors are independent views—operations on one cursor don't affect others. Worker-table writes are serialized to avoid delete/update conflicts; most reads remain lock-free.

### Atomic Replica Selection

The transport flow must atomically:
1. Find an available replica
2. Increment its request counter
3. Create a transport record

Without atomicity, concurrent requests could exceed `max_concurrency`. The repository achieves this with a CTE-based UPDATE-RETURNING pattern that combines selection and increment in a single statement.

### Heartbeat Batching

To avoid lock contention on the `workers` table, heartbeats are buffered in memory:

```python
# WorkerService
self._heartbeat_buffer.append((worker_id, mem_avail, accepting))
# Background thread flushes every ~100ms
```

This introduces sub-second staleness but dramatically reduces database pressure under high heartbeat rates.

## Operational Aspects

### Background Threads

The Global Store runs several daemon threads:

1. **Maintenance Thread**: Marks inactive workers, force-completes expired transports, runs VACUUM on hot tables
2. **Heartbeat Batch Thread**: Flushes buffered heartbeats every ~100ms

### Metrics

The service exports Prometheus metrics at `/metrics`:

| Metric | Type | Purpose |
|--------|------|---------|
| `tc_grpc_server_handled_total` | Counter | RPC counts by method and status |
| `tc_grpc_server_handling_seconds` | Histogram | RPC latency distribution |
| `tc_active_workers` | Gauge | Current registered workers |
| `tc_replicas_total` | Gauge | Total replicas across cluster |
| `tc_replicas_per_artifact` | Gauge | Replicas per artifact (cardinality concern) |
| `tc_transport_requests_total` | Counter | Transport outcomes (success/timeout/not_found) |
| `tc_transport_wait_seconds` | Histogram | Wait time for replica availability |
| `tc_active_transports` | Gauge | In-flight transports |
| `tensorcast_memory_tier_*` | Gauge | UMA telemetry by node |

The `PrometheusInterceptor` automatically instruments all unary-unary RPCs.

### Configuration

Configuration uses `tensorcast.config.v1.GlobalStoreConfig` proto, loaded from YAML:

```yaml
database:
  db_file: null  # null for in-memory

server:
  # Bind address (server-side); 0.0.0.0 exposes all interfaces.
  listen:
    host: 0.0.0.0
    port: 50051
  # Advertise address for clients/GetServerInfo; auto-detected if unset.
  advertise:
    host: 10.0.0.5
    port: 50051
  max_workers: 10

worker_policy:
  heartbeat_timeout: "30s"
  cleanup_interval: "60s"
  default_heartbeat_interval: "5s"
  memory_tiers:
    snapshot_retention: "600s"
    snapshot_max_rows: 200
  key_mapping:
    alias_cache_ttl: "1s"
  transport_scheduler:
    mode: TRANSPORT_SCHEDULER_MODE_GROUP_DISPATCH
    source_balance_weights:
      replica_load_weight: 1.0
      worker_load_weight: 1.0
      recent_assignment_penalty_weight: 1.0
      diffusion_bonus_weight: 1.0
    group_dispatch:
      fairness_floor_ratio: 0.25
      completion_bias_weight: 1.0
      starvation_aging_threshold: "5s"
      queue_scan_limit: 128
      dispatch_batch_limit: 16
  progressive_replication:
    enabled: false
    max_outgoing_per_source: 1
    coverage_ttl: "300s"
    assignment_ttl: "30s"
    allow_cross_domain_seed_sources: false
    min_verified_bytes: 16777216
    min_assignment_bytes: 16777216
    max_assignment_bytes: 268435456
    max_assignments_per_materialization: 1024
    assignment_candidate_scan_limit: 64
    max_claim_qps_per_daemon: 100
    cleanup_batch_limit: 256
    source_domain_policy: local_only
    min_report_delta_bytes: 16777216
    min_report_interval: "1s"
```

`server.listen` is the bind address, while `server.advertise` is the routable address returned by GetServerInfo and used for clients when it is routable. If `advertise.host` is set but non-routable, startup fails. If it is unset, the server attempts to auto-detect a suitable IPv4 address and logs the resolved value; clients ignore unspecified advertised hosts (for example, `0.0.0.0`) and fall back to a connectable listen host. When `database.db_file` is set, `~` is expanded and its parent directory is created on startup. When `database.db_file` is null/empty, the CLI leaves it unset and the Global Store uses in-memory DuckDB. When `tensorcast-cli global start` runs without `--config`, it uses `$TENSORCAST_GLOBAL_STORE_CONFIG` when set, otherwise `examples/config/global_store_config.yaml` (repo checkout or packaged wheel); if neither is found, startup fails. The example file defaults to `listen.host: 0.0.0.0` and `db_file: null`.
`worker_policy.key_mapping.alias_cache_ttl` controls the `ResolveKeyMapping` cache TTL returned for alias-style mappings; keep it short to reduce polling pressure.
Transport request handling always uses queue-based group dispatch; `worker_policy.transport_scheduler.mode` is retained only for config compatibility and should be set to `GROUP_DISPATCH`.
`worker_policy.progressive_replication` controls the 0118 progressive dissemination path and is disabled by default. When disabled, coverage reports are rejected and source claims return no eligible source. When enabled, verified byte-prefix coverage is stored separately from ordinary replica availability, source claims are bounded and idempotent, coverage and assignment deadlines are capped by the configured TTLs, per-source fanout caps use `progressive_source_counters`, stale heartbeat/export-generation candidates are filtered, and cross-domain seed sources stay disabled unless `allow_cross_domain_seed_sources=true`.
`UpsertKeyMapping`/`SwapKeyMapping` require descriptor readiness: target `artifact_id` must already resolve to canonical index bytes (`GetArtifactIndexById` path). If artifact row or index bytes are missing, RPC returns `FAILED_PRECONDITION`.

## Extending the Global Store

### Adding a New Entity

1. Add schema to `/schema.sql` with `CREATE TABLE IF NOT EXISTS`
2. Create a repository in `repositories/` extending `BaseRepository`
3. Create a service in `services/` that composes the repository
4. Wire into `GlobalStoreServicer.__init__`
5. Add gRPC methods if externally exposed
6. Update this README

### Adding Business Logic

**Keep gRPC handlers thin.** They should:
- Validate structural input (required fields present, valid UUIDs)
- Call a service method
- Map the result to proto response
- Map exceptions to gRPC status codes

**Business rules belong in services.** Semantic validation (routable addresses, consistent buffer sizes), cross-entity coordination, and metrics emission all happen in the service layer.

### Testing

Use `GlobalStoreServicer.reset_state()` for integration test isolation. It truncates mutable tables and recreates services without restarting the server or background threads.

## Running

```bash
# With explicit config file
uv run -m tensorcast.global_store --config config/global_store.yaml

# Or rely on config discovery:
#   1) $TENSORCAST_GLOBAL_STORE_CONFIG
#   2) examples/config/global_store_config.yaml (repo or packaged wheel)
uv run -m tensorcast.global_store

# Programmatically (for tests)
servicer = GlobalStoreServicer(db_file=None)  # in-memory
server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
register_global_store_servicers(server, servicer)
```
