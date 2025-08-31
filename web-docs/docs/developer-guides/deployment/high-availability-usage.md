---
title: High Availability
description: Configuration and usage of high availability features for system resilience
sidebar_position: 3
---

# High Availability

## Overview

The artifact storage system now includes comprehensive high availability features that ensure system resilience and eventual consistency in the face of failures. This guide explains how to configure and use these features.

## Features

### Global Store High Availability
- **Database Recovery**: Automatic recovery from persistent DuckDB storage
- **Worker Rediscovery**: Automatic rediscovery of workers after restart
- **State Synchronization**: Reconciles worker states with global state
- **Enhanced Heartbeat**: State version tracking and consistency checking

### Store Daemon High Availability
- **Connection Manager**: Intelligent reconnection with exponential backoff
- **State Queueing**: Queues operations during disconnection
- **Automatic Re-registration**: Re-registers with Global Store after failures
- **State Synchronization**: Syncs local state with global state

## Configuration

### Global Store Configuration

```yaml
# global_store_config.yaml
high_availability:
  enabled: true
  state_sync_interval_ms: 300000      # 5 minutes
  worker_timeout_ms: 30000            # 30 seconds
  recovery:
    auto_recovery_enabled: true
    max_recovery_attempts: 3
    recovery_timeout_ms: 120000       # 2 minutes
```

### Store Daemon Configuration

```yaml
# store_daemon_config.yaml
high_availability:
  enabled: true
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

## Usage Examples

### Starting Global Store with HA

```python
from tensorcast.global_store.grpc_service import GlobalStoreServicer

# Start with persistent database for recovery
global_store = GlobalStoreServicer(db_file="/path/to/persistent.db")

# Recovery is automatically initiated on startup
```

### Starting Store Daemon with HA

```python
# C++ daemon is now the implementation; use Python config models to launch it
from tensorcast.daemon_config import StoreDaemonConfig

# Configure with HA enabled
config = StoreDaemonConfig(
    storage_path="/path/to/storage",
    global_store_address="global-store:50051",
    enable_p2p_engine=True,
    # HA configuration will be read from config file
)

servicer = StoreDaemonServicer(config=config)

# HA connection manager is started automatically during initialization
# when global_store_address is provided in the config
```

### Worker Recovery Registration

When a Store Daemon restarts, it can recover its previous state:

```python
# The connection manager automatically handles recovery registration
# with the previous worker ID if available

# Manual recovery registration
request = global_store_pb2.RegisterWorkerRequest(
    node_id="worker-node-1",
    node_address="10.0.0.100",
    grpc_port=50052,
    p2p_port=9090,
    mem_pool_total_size=10 * 1024**3,  # 10GB
    mem_pool_available_size=8 * 1024**3,  # 8GB
    is_recovery_registration=True,
    previous_worker_id="worker_node-1_1640995200"
)

response = global_store_stub.RegisterWorker(request)
if response.state_sync_required:
    # Perform state synchronization
    pass
```

### Enhanced Heartbeat

Store Daemons send enhanced heartbeats with state information:

```python
request = global_store_pb2.WorkerHeartbeatRequest(
    worker_id="worker_node-1_1640995200",
    mem_pool_available_size=7 * 1024**3,
    accepting_new_requests=True,
    # Enhanced HA fields
    state_version=15,
    state_checksum="md5_hash_of_local_state",
    registered_models=["model1", "model2", "model3"],
    last_successful_sync=1640995200,
    global_store_status=global_store_pb2.CONNECTED
)

response = global_store_stub.WorkerHeartbeat(request)
if response.state_sync_required:
    # Global Store detected inconsistency, perform sync
    perform_state_sync(response.expected_state_version)
```

### State Synchronization

Manual state synchronization when needed:

```python
# Prepare local state
local_state = global_store_pb2.WorkerLocalState(
    worker_id="worker_node-1_1640995200",
    state_version=15,
    state_checksum="local_state_checksum",
    local_replicas=[...],  # List of local artifact replicas
    last_update_timestamp=int(time.time())
)

# Request synchronization
sync_request = global_store_pb2.SynchronizeWorkerStateRequest(
    worker_id="worker_node-1_1640995200",
    local_state=local_state,
    force_full_sync=False
)

response = global_store_stub.SynchronizeWorkerState(sync_request)
# Apply state changes
for change in response.state_changes:
    if change.type == global_store_pb2.StateChange.ADD_REPLICA:
        # Add replica locally
        pass
    elif change.type == global_store_pb2.StateChange.REMOVE_REPLICA:
        # Remove replica locally
        pass
```

## Failure Scenarios

### Global Store Crash/Restart

1. **Automatic Recovery**: Global Store automatically recovers from persistent database
2. **Worker Rediscovery**: Workers are marked as stale until they reconnect
3. **State Reconciliation**: When workers reconnect, states are synchronized

### Store Daemon Crash/Restart

1. **Clean Registration**: Store Daemon registers as new worker (clears previous state)
2. **Artifact Rediscovery**: Local storage is scanned for available artifacts
3. **State Rebuild**: Artifact replicas are re-registered with Global Store

### Network Partition

1. **Connection Monitoring**: Store Daemon detects Global Store unavailability
2. **Local Operation**: Continues serving cached artifacts locally
3. **State Queueing**: Artifact registrations are queued for later sync
4. **Automatic Reconnection**: Exponential backoff retry until reconnection
5. **State Synchronization**: Queued changes are synced after reconnection

## Monitoring

### Key Metrics

Monitor these Prometheus metrics for HA health:

```
# Connection status
global_store_connection_status{worker_id="..."}
connection_retry_attempts_total{worker_id="..."}
connection_downtime_seconds{worker_id="..."}

# State synchronization
state_sync_operations_total{worker_id="...", result="..."}
state_sync_duration_seconds{worker_id="..."}
state_inconsistencies_detected_total{worker_id="..."}

# Recovery operations
recovery_operations_total{type="...", result="..."}
recovery_success_rate{type="..."}
recovery_duration_seconds{type="..."}
```

### Health Checks

```bash
# Check Global Store via gRPC
grpcurl -plaintext global-store:50051 \
  global_store.GlobalStore/ListActiveWorkers
```

## Best Practices

### Deployment

1. **Persistent Storage**: Always use persistent database files for Global Store
2. **Health Monitoring**: Monitor connection and sync metrics
3. **Gradual Rollout**: Update Global Store first, then Store Daemons
4. **Testing**: Test failure scenarios in staging environment

### Configuration

1. **Timeout Values**: Adjust timeouts based on network latency
2. **Retry Limits**: Use infinite retries for mission-critical deployments
3. **Sync Intervals**: Balance consistency vs performance needs
4. **Resource Allocation**: Ensure adequate memory for state queuing

### Troubleshooting

1. **Check Logs**: Both components log HA operations extensively
2. **Verify Configuration**: Ensure HA is enabled in both components
3. **Network Connectivity**: Verify gRPC connectivity between components
4. **Database Access**: Ensure Global Store can read/write database file
5. **Resource Limits**: Check memory and disk space for state operations

## Migration from Legacy Setup

### Phase 1: Enable HA in Global Store
```bash
# Update Global Store with persistent database
global-store --db-file=/data/global-store.db
```

### Phase 2: Update Store Daemons
```bash
# Enable HA in Store Daemon configuration
store-daemon --config=/etc/store-daemon-ha.yaml
```

### Phase 3: Verify Operation
```bash
# Check all workers are using enhanced heartbeat
# Monitor connection status and state sync metrics
# Test failure scenarios
```

## Limitations

1. **Single Global Store**: Current implementation supports single Global Store instance
2. **Eventual Consistency**: System guarantees eventual, not immediate consistency
3. **Memory Overhead**: State tracking and queueing requires additional memory
4. **Network Dependency**: HA features require reliable network connectivity

## Future Enhancements

- Multi-master Global Store with consensus
- Cross-region replication
- Automated failover with standby instances
- Advanced conflict resolution for concurrent updates
