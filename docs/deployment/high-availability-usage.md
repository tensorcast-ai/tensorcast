---
title: High Availability
description: Configuration and usage of high availability features for system resilience
sidebar_position: 3
---

# High Availability

## Overview

TensorCast HA keeps the single Global Store resilient and consistent with Store Daemon state by combining startup recovery, enhanced heartbeats, incremental/full state sync, and drift pruning. Configuration is file-based (proto-backed) and orchestrated via the `tensorcast` CLI.

## Features

### Global Store
- **Startup recovery**: marks workers and replicas stale, cleans orphaned replicas, and resets per-worker state versions.
- **State sync pipeline**: enhanced heartbeats advertise version + checksum + registered artifacts; incremental sync applies additions/removals with “addition over removal” semantics; full-state sync returns the expected set without bumping versions.
- **Identity guardrails**: rejects loopback/unspecified registration addresses; `HealthCheck` surfaces `cluster_token`, listen endpoints, metrics port, and version.

### Store Daemon
- **Routable registration**: requires a non-loopback advertise host and non-zero `server.p2p_listen.port` before enabling HA.
- **Initial drift pruning**: on startup (and recovery re-registration) runs `RequestFullStateSync` then unloads local replicas not expected by the Global Store.
- **Enhanced heartbeat loop**: sends version/checksum/inventory; if the server returns `NOT_FOUND` but the channel is healthy, the daemon re-registers with the previous worker id and resyncs.
- **Incremental sync handling**: applies obsolete removals immediately and prefetches server-requested `ADD_REPLICA` updates; toggles remote access for `UPDATE_REPLICA`.
- **Bounded retries**: all RPCs use jittered exponential backoff (`max_retries=3`, base 100ms).

## Configuration

### Global Store (proto: `tensorcast.config.v1.GlobalStoreConfig`)

```yaml
database:
  db_file: /var/lib/tensorcast/global_store.duckdb   # persistent for HA
server:
  listen:
    host: 0.0.0.0
    port: 50051
  max_workers: 20
worker_policy:
  heartbeat_timeout: 30s
  cleanup_interval: 60s
  default_heartbeat_interval: 5s
  memory_tiers:
    snapshot_retention: 10m
    snapshot_max_rows: 200
    publish_interval: 5s
meta:
  cluster_token: <optional-cluster-id>               # echoed by HealthCheck
```

### Store Daemon (proto: `tensorcast.config.v1.DaemonConfig`)

```yaml
server:
  listen:
    host: 0.0.0.0
    port: 50052
  advertise:
    host: 10.0.0.20           # must be routable (non-loopback/unspecified)
  p2p_listen:
    host: 0.0.0.0
    port: 65090               # required for HA registration
high_availability:
  enabled: true
  global_store_endpoints:
    - host: 10.0.0.5          # first entry is used today
      port: 50051
  heartbeat_interval: 5s      # default when omitted
  periodic_sync_interval: 10s # chunk sync loop; 0 disables
```

> The CLI (`tensorcast daemon start`) will inject `high_availability.global_store_endpoints` when you pass `--global-store-address` or `--global-store-endpoints`, and will auto-fill ports when set to 0. Keep `server.advertise.host` routable to avoid registration failures.
>
> Note: `listen.host: 0.0.0.0` is a bind-all address (server-side). Clients should connect using a routable IP/DNS name (e.g. `10.0.0.5:50051`). TensorCast will not treat `0.0.0.0` as a dial target even if the Global Store reports it via health metadata.

## Usage

### Start Global Store (single instance)

```bash
uv run tensorcast global start --config=/etc/tensorcast/global_store.yaml
```

- Use a persistent DuckDB path (`database.db_file`) for recovery.
- The CLI persists `cluster_token` under `~/.tensorcast/runtime/cluster_token` to detect split-brain and returns the metrics port in the startup log.
- Add `--blocking` to keep the Global Store attached to the CLI and stop it when the CLI exits.

### Start Store Daemon with HA

```bash
uv run tensorcast daemon start \
  --config=/etc/tensorcast/daemon.yaml \
  --global-store-mode connect \
  --global-store-address 10.0.0.5:50051
```

- The orchestrator writes an effective config that enables HA, injects the Global Store endpoint, and fills missing listen/p2p ports.
- Startup will fail fast if `server.p2p_listen.port` is zero or if `advertise.host` is loopback/unspecified.
- Add `--blocking` to keep the daemon attached to the CLI and stop it when the CLI exits (SIGTERM with a ~35s grace before SIGKILL).

### Manual RPC examples (generated stubs)

Enhanced heartbeat:

```python
import time
from tensorcast.proto.global_store.v1 import global_store_pb2

req = global_store_pb2.WorkerHeartbeatRequest(
    worker_id="worker-node-1",
    mem_pool_available_size=7 * 1024**3,
    accepting_new_requests=True,
    state_version=15,
    state_checksum="local_state_md5",
    registered_artifact_ids=["artifact1", "artifact2"],
    last_successful_sync=int(time.time()),
    global_store_status=global_store_pb2.CONNECTION_STATUS_CONNECTED,
)
resp = stub.WorkerHeartbeat(req)
if resp.state_sync_required:
    # initiate sync
    ...
```

State sync:

```python
import time
from google.protobuf import timestamp_pb2
from tensorcast.proto.global_store.v1 import global_store_pb2
from tensorcast.proto.common.v1 import common_pb2

ts = timestamp_pb2.Timestamp()
ts.FromSeconds(int(time.time()))
local_state = global_store_pb2.WorkerLocalState(
    worker_id="worker-node-1",
    state_version=15,
    state_checksum="local_state_checksum",
    last_update_ts=ts,
)
replica = local_state.local_replicas.add()
replica.ref.artifact_id = "artifact1"
replica.memory_info.memory_type = common_pb2.MEMORY_TYPE_GPU
replica.memory_info.device_id = 0
replica.memory_info.memory_size = 1 * 1024**3
resp = stub.SynchronizeWorkerState(
    global_store_pb2.SynchronizeWorkerStateRequest(
        worker_id="worker-node-1",
        local_state=local_state,
        force_full_sync=False,
    )
)
```

## Failure Scenarios

- **Global Store crash/restart**: recovery marks workers/replicas stale; daemons continue heartbeating, re-register on `NOT_FOUND`, run full-state sync, and prune drift. Persistent `db_file` keeps registry state.
- **Daemon crash/restart**: a fresh process registers a new worker id, runs full-state sync, and unloads replicas not expected by the Global Store.
- **Network partition**: RPC retries are bounded; once connectivity returns the next heartbeat triggers sync. No offline queue is kept—state is rebuilt from the current engine snapshot.
- **Registration rejected**: ensure `advertise.host` is routable and `p2p_listen.port` is set.

## Monitoring

- **Metrics**: Global Store exports Prometheus metrics (e.g., `tc_state_sync_total`, `tc_state_sync_seconds`, `tc_active_workers`, `tc_replicas_total`, `tc_grpc_server_handled_total`). Scrape `http://<host>:<metrics_port>/metrics`.
- **Health**:
  - gRPC: `grpcurl -plaintext <host>:50051 tensorcast.global_store.v1.GlobalStoreService/HealthCheck`
  - Inventory: `grpcurl -plaintext <host>:50051 tensorcast.global_store.v1.GlobalStoreService/ListActiveWorkers`
- **Daemon visibility**: OTEL spans wrap all Global Store RPCs; counters (hb_success/failure, sync_success/failure) are exported via the configured OTEL sink when enabled.

## Best Practices

- Use persistent storage for the Global Store database; back up `cluster_token` with it.
- Set a routable `server.advertise.host` and non-zero `server.p2p_listen.port` before enabling HA.
- Only enable `high_availability.force_full_sync_on_empty_inventory` when deliberately draining/retiring a node; otherwise keep the default conservative behavior.
- Start always waits for readiness and returns only when services are healthy (or on error) before clients connect; `--blocking` keeps the process attached after readiness.
- Keep configs proto-valid (strict parsing) and avoid relying on multiple endpoints—the first `global_store_endpoints` entry is used today.

## Migration from Legacy Setup

1. Add a persistent `database.db_file` and optional `meta.cluster_token` to the Global Store config, then restart with `uv run tensorcast global start --config=...`.
2. Update daemon config to include `high_availability` and a routable `server.advertise.host`/`p2p_listen.port`, then restart with `uv run tensorcast daemon start --global-store-address <addr>`.
3. Verify via `HealthCheck`, metrics scrape, and a forced `RequestFullStateSync` (daemon restart) before rolling out broadly.

## Limitations

- Single Global Store instance; no automatic failover or multi-endpoint rotation (first endpoint only).
- RPC retries are bounded; there is no durable queue of state changes while disconnected.
- Eventual consistency: reconciles on heartbeat/sync, not instantly.

## Future Enhancements

- Endpoint rotation/failover across multiple Global Store addresses.
- Multi-master Global Store with consensus and cross-region replication.
- Automated standby promotion and richer conflict resolution for concurrent updates.
