---
title: High Availability Design
description: High availability design ensuring system resilience and eventual consistency
sidebar_position: 4
---

# High Availability Design

## Overview

This document outlines the high availability design for the distributed artifact storage system, ensuring the system can recover from various failure scenarios while maintaining eventual consistency.

## Goals

1. **Global Store Resilience**: Recover from crashes, restarts, and data corruption
2. **Store Daemon Resilience**: Handle disconnections and maintain service continuity
3. **Eventual Consistency**: Guarantee system convergence to consistent state after failures
4. **Minimal Downtime**: Reduce service interruption during failures
5. **Data Integrity**: Preserve artifact registry data across failures

## Architecture Components

### Current State
- **Global Store**: Centralized coordinator using DuckDB for persistence
- **Store Daemon**: Distributed workers with local artifact storage
- **Communication**: gRPC with heartbeat mechanism
- **Persistence**: DuckDB file-based storage in Global Store

### Enhanced Components for HA

1. **State Synchronization Service** (implemented in `recovery_service.py`)
2. **Connection Manager** (enhanced)
3. **Recovery Coordinator** (implemented within `recovery_service.py`)
4. **Enhanced Heartbeat System** (✔️ completed)

## Failure Scenarios and Solutions

### 1. Global Store Failure Recovery

#### Scenario: Global Store crashes/restarts
**Root Cause**: Hardware failure, software crash, planned restart

**Recovery Strategy**:
1. **Database Recovery**: Load state from persistent DuckDB file
2. **Worker Re-discovery**: Accept worker re-registration
3. **State Synchronization**: Reconcile worker states with database
4. **Artifact Registry Rebuild**: Validate and update artifact replica information

#### Implementation:
```python
class GlobalStoreRecoveryService:
    def recover_from_database(self) -> bool:
        """Recover state from persistent database."""

    def initiate_worker_rediscovery(self) -> None:
        """Trigger worker re-registration process."""

    def reconcile_worker_states(self) -> None:
        """Synchronize worker states with database."""
```

The `RecoveryService` class in `scstore/global_store/services/recovery_service.py` implements these features with additional methods:
- `initiate_recovery()` - Main recovery entry point
- `handle_worker_recovery_registration()` - Handles worker recovery scenarios
- `synchronize_worker_state()` - Performs state synchronization
- `request_full_state_sync()` - Triggers full state synchronization

### 2. Store Daemon Reconnection Strategy

#### Scenario: Global Store becomes unavailable
**Root Cause**: Network partition, Global Store failure

**Strategy**:
1. **Connection Monitoring**: Detect Global Store unavailability
2. **Exponential Backoff Retry**: Implement smart retry mechanism
3. **Local Operation Continuity**: Continue serving cached models
4. **State Queue**: Queue state changes for later synchronization

#### Implementation:
```python
class GlobalStoreConnectionManager:
    def __init__(self):
        self.connection_state = ConnectionState.DISCONNECTED
        self.retry_scheduler = ExponentialBackoffScheduler()
        self.pending_state_changes = StateChangeQueue()

    async def maintain_connection(self) -> None:
        """Continuously monitor and maintain Global Store connection."""

    async def sync_pending_changes(self) -> None:
        """Synchronize queued state changes after reconnection."""
```

### 3. Store Daemon Fatal Failure Recovery

#### Scenario: Store Daemon crashes and restarts
**Root Cause**: Process crash, hardware failure, container restart

**Strategy**:
1. **State Cleanup**: Clear all previous registrations
2. **Fresh Registration**: Re-register as new worker
3. **Artifact Re-discovery**: Scan local storage and re-register models
4. **Gradual Service Restoration**: Progressively restore service capacity

#### Implementation:
```python
class StoreDaemonRecoveryService:
    def cleanup_previous_state(self) -> None:
        """Clean up any previous worker registrations."""

    def rediscover_local_models(self) -> List[ReplicaInfo]:
        """Scan local storage for available models."""

    def gradual_service_restoration(self) -> None:
        """Progressively restore service capacity."""
```

Note: This is currently handled through the Global Store's `RecoveryService.handle_worker_recovery_registration()` method.

## Enhanced Heartbeat and Health Check System

### Unified Health Monitoring

**Objective**: Integrate failure detection, state synchronization, and health monitoring

#### Components:
1. **Enhanced Worker Heartbeat**: Include state checksums and version info
2. **Health Check Aggregation**: Combine application and infrastructure health
3. **State Synchronization Triggers**: Detect inconsistencies during heartbeat

#### Enhanced Heartbeat Protocol:
```protobuf
message EnhancedWorkerHeartbeatRequest {
  string worker_id = 1;
  uint64 mem_pool_available_size = 2;
  bool accepting_new_requests = 3;

  // Enhanced fields for HA
  uint64 state_version = 4;           // Incremental state version
  string state_checksum = 5;          // Checksum of current state
  repeated string registered_models = 6; // List of locally registered models
  int64 last_successful_sync = 7;     // Last successful state sync timestamp
  ConnectionStatus global_store_status = 8; // Connection status from worker perspective
}

message EnhancedWorkerHeartbeatResponse {
  Status status = 1;

  // Enhanced fields for HA
  bool state_sync_required = 2;       // Trigger state synchronization
  uint64 expected_state_version = 3;  // Expected state version
  repeated string obsolete_replicas = 4; // Models that should be unregistered
}

enum ConnectionStatus {
  CONNECTED = 0;
  RECONNECTING = 1;
  DISCONNECTED = 2;
}
```

**Current Implementation Status**: The enhanced heartbeat fields are fully wired end-to-end. The server-side handler (`_handle_enhanced_heartbeat()` in `grpc_service.py`) validates checksums and detects obsolete replicas, while the Store Daemon now performs automatic replica reconciliation, queueing / applying unregistrations, and periodic full state synchronisation via `GlobalStoreConnectionManager`.

### Health Check Integration

**Strategy**: Separate but coordinated health checks and heartbeats

- **HTTP Health Checks**: Infrastructure-level health (process, memory, disk)
- **gRPC Heartbeats**: Application-level state synchronization
- **Combined Status**: Aggregate view for external monitoring

## State Synchronization Protocol

### Consistency Artifact: Eventual Consistency

**Guarantees**:
1. All nodes will eventually converge to the same state
2. Temporary inconsistencies are acceptable during network partitions
3. Last-writer-wins conflict resolution for concurrent updates

### Synchronization Triggers

1. **Worker Re-registration**: Full state sync on connection
2. **Heartbeat Mismatch**: Incremental sync when checksums differ
3. **Periodic Sync**: Regular full synchronization (configurable interval)
4. **Manual Trigger**: Administrative state synchronization

### Replica Authority & Safe-Removal Semantics

When reconciling state **the Store Daemon is always considered the source of truth for the set of replicas physically present on the node**.  The Global Store must therefore treat the daemon-provided `local_replicas` list in every `WorkerLocalState` message as *authoritative*.

Key rules:

1. **Full inventory required** – The daemon **MUST** send the *complete* list of currently resident replicas in `local_replicas` for every state-sync (incremental *or* full).  Relying solely on the `registered_models` string list (checksum optimisation) is insufficient because it does not convey device-level metadata needed to create missing `ReplicaInfo` rows on the Global Store side.
2. **Addition over removal** – When the Global Store receives a state sync it first computes

   ```text
   to_add     = local  \ global
   to_remove  = global \ local
   ```

   However, *removal* **MUST be suppressed** if the candidate replica appears in the daemon-provided inventory **or** if the daemon omitted its inventory entirely (e.g. due to an older client).  In other words: *when in doubt, prefer `ADD_REPLICA` over `REMOVE_REPLICA`.*  A replica is only removed when the daemon has explicitly confirmed its absence.
3. **Safety latch for destructive actions** – Before emitting `REMOVE_REPLICA`, the Global Store must verify that at least two consecutive heartbeat/sync cycles produced the same *missing* result.  This prevents a transient gRPC failure from deleting perfectly healthy replicas.
4. **Worker driven garbage collection** – The Store Daemon may still decide to unload a replica locally (for eviction, OOM, etc.).  In this case it *must* include that absence in its next sync, allowing the Global Store to issue a definitive `REMOVE_REPLICA` back to the worker.  The direction of intent is thus unambiguous.
5. **Checksum remains lightweight** – The MD-5 checksum optimisation stays unchanged: it is still used as a cheap equality guard, but it now guards the *authoritative* replica list coming from the worker.

This change eliminates the class of bugs where a worker’s valid, locally cached replica is erroneously removed after a Global Store restart (because `local_replicas` was empty and the store inferred `REMOVE_REPLICA`).  Implementation notes:

* `GlobalStoreConnectionManager._perform_state_sync()` must populate `WorkerLocalState.local_replicas` with every `ReplicaInfo` it currently serves.
* `RecoveryService._compute_state_changes()` is updated to honour rule 2 (prefer addition over removal when inventory is absent).

> **Operational impact** – Replica counts on the Global Store can briefly oscillate upwards until the first full sync completes, but this is safe because the duplicate entries resolve to the same `replica_id`.

---

## Implementation Strategy

### Phase 1: Enhanced Heartbeat and Connection Management ✅ (Complete)
1. ✅ Extend heartbeat protocol with state information
2. ✅ Implement connection monitoring and retry logic (ConnectionManager in Store Daemon)
3. ✅ Add state version tracking

### Phase 2: Recovery Services ✅ (Complete)
1. ✅ Implement Global Store recovery service
2. ✅ Add Store Daemon recovery mechanisms
3. ✅ Create state synchronization service

### Phase 3: Integration and Testing ✅ (Complete)
1. ✅ Integrate HA components with existing system
2. ✅ Comprehensive failure scenario testing
3. ✅ Performance impact assessment

### Phase 4: Production Hardening & Observability ✅ (Complete)
1. ✅ Extended soak-testing under production-scale load
2. ✅ Prometheus/Grafana dashboards for all new HA metrics
3. ✅ On-call playbooks and alert rules validated in staging

## Configuration

### Global Store Configuration
The Global Store HA configuration is managed through environment variables:

```bash
# Basic settings
GLOBAL_STORE_PORT=50051
GLOBAL_STORE_MAX_WORKERS=10
GLOBAL_STORE_DB_PATH=/path/to/models.db  # For persistence

# HA-related settings
GLOBAL_STORE_HEARTBEAT_TIMEOUT_MS=30000      # Worker timeout (30s)
GLOBAL_STORE_CLEANUP_INTERVAL_MS=60000       # Cleanup interval (1m)
GLOBAL_STORE_HEARTBEAT_INTERVAL_MS=5000      # Heartbeat interval (5s)

# Recovery is automatically enabled on startup
# State sync happens through enhanced heartbeat
```

### Store Daemon Configuration
```yaml
# High-availability features (connection retry, enhanced heartbeat, state sync) are **fully implemented**
high_availability:
  enabled: true   # HA enabled by default
  connection_retry:
    max_attempts: -1                  # Infinite retries
    base_delay_ms: 1000
    max_delay_ms: 60000
    backoff_multiplier: 2.0
  state_sync:
    heartbeat_enhanced: true
    state_checksum_enabled: true
    periodic_sync_interval_ms: 600000 # 10 minutes
```

Note: Full HA support in Store Daemon is **now fully integrated**. Enhanced heartbeat, connection retry, replica reconciliation, and periodic full-sync are all active by default.

## Monitoring and Observability

### Key Metrics
1. **Connection Metrics**:
   - `global_store_connection_status`
   - `connection_retry_attempts_total`
   - `connection_downtime_seconds`

2. **Synchronization Metrics**:
   - `state_sync_operations_total`
   - `state_sync_duration_seconds`
   - `state_inconsistencies_detected_total`

3. **Recovery Metrics**:
   - `recovery_operations_total`
   - `recovery_success_rate`
   - `recovery_duration_seconds`

### Alerting Rules
1. **Critical**: Global Store down for > 5 minutes
2. **Warning**: Store Daemon disconnected for > 2 minutes
3. **Info**: State synchronization took > 30 seconds

## Testing Strategy

### Failure Simulation Tests
1. **Global Store Crash**: Validate recovery from database
2. **Network Partition**: Test Store Daemon retry and reconnection
3. **Store Daemon Crash**: Verify cleanup and re-registration
4. **Split Brain**: Test conflict resolution during recovery

### Load Testing
1. **Recovery under Load**: Test recovery with active artifact requests
2. **Concurrent Failures**: Multiple Store Daemon failures
3. **State Synchronization Performance**: Large-scale state sync operations

## Current Status and TODOs

### Implemented ✅
- Global Store recovery service (`recovery_service.py`)
- Enhanced heartbeat protocol definition
- State version tracking
- Worker recovery registration
- State synchronization framework
- Startup recovery process
- Enhanced heartbeat state checksum validation
- **Obsolete artifact detection (Global Store)**
- **Store Daemon Connection Manager with retry back-off**
- **Enhanced heartbeat client in Store Daemon**
- **Production hardening & operational dashboards**

### In Progress 🚧
*None – all planned HA features have shipped to production.*

### TODO ⚠️
*None*

## Recommended Next Steps

*All previously planned steps have been completed. Any further enhancements will be tracked in the project issue tracker.*

## Future Enhancements

1. **Multi-Master Global Store**: Distributed Global Store with consensus
2. **Conflict-free Replicated Data Types (CRDTs)**: Advanced consistency models
3. **Cross-Region Replication**: Geographic distribution support
4. **Automated Failover**: Automatic promotion of standby Global Store instances

## Conclusion

This high availability design provides robust failure recovery mechanisms while maintaining the simplicity and performance of the current architecture. The enhanced heartbeat system serves as the foundation for both health monitoring and state synchronization, ensuring eventual consistency across the distributed system.

All components are now fully implemented and enabled by default. The system has passed comprehensive failure-scenario and performance tests and is considered production-ready.