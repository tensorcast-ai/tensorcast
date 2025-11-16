---
title: Global Store
description: Guide for deploying Global Store service
sidebar_position: 2
---

# Global Store Deployment Best Practices

The Global Store is a centralized artifact registry service that manages artifact metadata, worker registration, and state synchronization across the distributed artifact serving infrastructure.

## Architecture Overview

The Global Store provides:
- **Centralized Artifact Registry**: Tracks all artifacts and their replicas across the cluster
- **Worker Management**: Handles worker registration, heartbeats, and health monitoring
- **State Synchronization**: Ensures consistency across distributed workers
- **High Availability**: Supports persistent storage and recovery mechanisms
- **Metrics**: Prometheus-compatible metrics endpoint

## Configuration

Use a unified file-based configuration (YAML/JSON → Proto with strict validation); CLI flags and environment variables are not supported. See `examples/config/global_store_config.yaml` and pass it via `--config`.

## Deployment Methods

### 1. Direct Python Module Execution (Unified Config)

#### Development/Testing
```bash
# Use the configuration file (strictly validated)
uv run -m tensorcast.global_store --config=examples/config/global_store_config.yaml
```

#### Production with Persistence
Set the persistent database path, thread counts, and other parameters in the configuration file, and start with `--config`.

### 2. Environment Variable Configuration (Deprecated)
Environment variables are no longer supported. Please use the unified configuration file.

### 3. Docker Deployment

```bash
# Build the image
./docker/build.sh

# Run with persistent storage (mount config)
docker run -d \
  --name global-store \
  -p 50051:50051 \
  -p 8001:8001 \
  -v /var/lib/global_store:/data \
  -v $(pwd)/examples/config/global_store_config.yaml:/etc/tensorcast/global_store.yaml:ro \
  hub.i.basemind.com/tensorcast/global-store:latest \
  uv run -m tensorcast.global_store --config=/etc/tensorcast/global_store.yaml
```

### 4. Kubernetes Deployment (High Availability)

See the [Kubernetes High Availability](#kubernetes-high-availability) section below for detailed StatefulSet configuration.

## High Availability Configuration

### 1. Persistent Storage

For production deployments, **always use persistent storage**:

```bash
# Ensure data directory exists with proper permissions
mkdir -p /var/lib/global_store
chown -R app:app /var/lib/global_store

# Start with persistent database via unified config
uv run -m tensorcast.global_store --config=/etc/tensorcast/global_store.yaml
```

Benefits:
- **Automatic Recovery**: Global Store recovers state from database on restart
- **Worker State Preservation**: Maintains worker registrations and artifact assignments
- **Audit Trail**: Preserves historical data for debugging

### 2. Recovery Mechanisms

The Global Store implements several recovery features:

#### Database Recovery
- Automatically initiated on startup when using persistent storage
- Restores worker registrations, artifact metadata, and replica assignments
- Validates data integrity during recovery

#### Worker Re-registration
- Workers can perform recovery registration after Global Store restart
- Preserves worker identity and artifact assignments when possible
- Supports state synchronization after recovery

#### State Synchronization
- Enhanced heartbeat protocol includes state versioning
- Automatic detection of state divergence
- Full state sync available for major discrepancies

### 3. Monitoring and Health Checks

#### Prometheus Metrics
Exposed via the unified metrics system:
- Worker registration/deregistration counts
- Active worker count
- Artifact registration metrics
  - tc_register_begin_coalesced_total
  - tc_register_begin_cpu_total
  - tc_register_begin_lease_total
  - tc_register_commit_coalesced_total
  - tc_register_commit_cpu_total
  - tc_register_commit_lease_total
  - tc_register_abort_total
  - tc_register_keepalive_total
  - tc_register_revoke_total
  - tc_register_feed_cpu_bytes_total
  - tc_register_feed_lease_segments_total
  - tc_register_feed_lease_bytes_total
  - tc_register_pending_gauge (current number of in-flight registrations per daemon)
  - tc_register_commit_seconds (latency histogram with `result` label: ok/aborted/expired)
- Request latencies
- Error rates

#### Health Check Endpoints
```bash
# gRPC health check
grpc_health_probe -addr=localhost:50051

# HTTP health via metrics endpoint
curl http://localhost:8001/health
```

## Kubernetes High Availability

### StatefulSet Configuration

For production Kubernetes deployments, use a StatefulSet with persistent volumes:

```yaml
# See docker/k8s/global_store.yaml for complete configuration
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: global-store
spec:
  replicas: 1  # Single instance with persistent storage
  serviceName: global-store-headless
  template:
    spec:
      containers:
      - name: global-store
        image: hub.i.basemind.com/tensorcast/global-store:latest
        env:
        - name: GLOBAL_STORE_DB_PATH
          value: /data/models.db
        volumeMounts:
        - name: data
          mountPath: /data
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 20Gi
```

### Service Configuration

```yaml
# Headless service for StatefulSet
apiVersion: v1
kind: Service
metadata:
  name: global-store-headless
spec:
  clusterIP: None
  ports:
  - name: grpc
    port: 50051
  selector:
    app: global-store

---
# Load-balanced service for client access
apiVersion: v1
kind: Service
metadata:
  name: global-store
spec:
  type: ClusterIP
  ports:
  - name: grpc
    port: 50051
  - name: metrics
    port: 8001
  selector:
    app: global-store
```

## Best Practices

### 1. Resource Allocation

- **CPU**: 2-4 cores for moderate load, 4-8 cores for high load
- **Memory**: 4-8 GB minimum, 16 GB recommended for large deployments
- **Storage**: 20-50 GB SSD for database, depending on artifact count

### 2. Database Maintenance

```bash
# Periodic backup (while service is running)
cp /var/lib/global_store/models.db /backup/models.db.$(date +%Y%m%d)

# Database optimization (automatic, but can be tuned)
export GLOBAL_STORE_OPTIMIZE_INTERVAL_MS=1800000  # 30 minutes
```

### 3. Security Considerations

- Run as non-root user
- Use TLS for gRPC connections in production
- Restrict network access to trusted sources

### 4. Scaling Strategy

While Global Store is designed as a single instance with persistent storage:

1. **Vertical Scaling**: Increase CPU/memory for the single instance
2. **Read Replicas**: Future versions may support read-only replicas
3. **Backup Instance**: Maintain a standby instance with replicated database

### 5. Integration with Store Daemons

Ensure Store Daemons are configured to:
```bash
# Point to Global Store service
export GLOBAL_STORE_ADDRESS=global-store.namespace.svc.cluster.local:50051

# Enable reconnection with exponential backoff
export ENABLE_GLOBAL_STORE_RECONNECT=true
export RECONNECT_MAX_RETRIES=10
```

## Troubleshooting

### Common Issues

1. **Worker Registration Failures**
   - Check network connectivity
   - Verify Global Store address in worker configuration
   - Check for port conflicts

2. **Database Corruption**
   - Stop service immediately
   - Restore from backup
   - Run with fresh database if needed

3. **High Memory Usage**
   - Check worker count and cleanup settings
   - Increase cleanup frequency
   - Monitor for memory leaks

### Debug Logging

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python -m tensorcast.global_store
```

### Recovery Procedures

1. **Complete System Recovery**
   ```bash
   # Stop all services
   systemctl stop global-store

   # Restore database from backup
   cp /backup/models.db.latest /var/lib/global_store/models.db

   # Start Global Store
   systemctl start global-store

   # Workers will auto-reconnect and re-register
   ```

2. **Worker State Resync**
   ```bash
   # Trigger full state sync for specific worker
   grpcurl -d '{"worker_id": "worker-123"}' \
     localhost:50051 \
     global_store.GlobalStoreService/RequestFullStateSync
   ```

## Conclusion

The Global Store is designed for high availability through:
- Persistent storage with automatic recovery
- Robust worker management with heartbeat monitoring
- State synchronization protocols
- Comprehensive monitoring and metrics

Follow these best practices to ensure reliable operation of your artifact serving infrastructure.
