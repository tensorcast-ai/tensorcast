---
title: High Availability — Developer Guide
description: Implementation-aligned HA behavior in Global Store and Store Daemon
sidebar_position: 4
---

# High Availability — Developer Guide

## Overview

This document explains High Availability (HA) as implemented in code: startup recovery, enhanced heartbeat, state synchronization, and daemon lifecycle behavior. It links directly to source files and proto definitions used in production.

## Key Components (sources)

- Global Store
  - Recovery service: `tensorcast/global_store/services/recovery_service.py`
  - gRPC service (registration, heartbeat, sync): `tensorcast/global_store/grpc_service.py`
- Store Daemon (C++)
  - Lifecycle + heartbeat + sync: `daemon/worker_lifecycle_manager.cc`
  - Global Store client + retries/backoff: `core/store/components/global_store_client.{h,cc}`
- Protocol buffers
  - Enhanced heartbeat and sync: `proto/tensorcast/global_store/v1/global_store.proto`

## Startup Recovery (Global Store)

Global Store runs a recovery pass on startup; workers/replicas are marked stale until workers re-confirm via registration/heartbeat.

```42:71:tensorcast/global_store/services/recovery_service.py
def initiate_recovery(self) -> bool:
    if self.recovery_in_progress:
        return False
    self.recovery_in_progress = True
    try:
        self._validate_database_state()
        self._mark_workers_as_stale()
        self._mark_replicas_as_stale()
        self.worker_state_versions.clear()
        self.last_recovery_time = int(time.time())
        return True
    except Exception:
        return False
    finally:
        self.recovery_in_progress = False
```

## Enhanced Heartbeat

The daemon sends an enhanced heartbeat including state metadata; the server validates and signals whether synchronization is required.

```93:115:proto/tensorcast/global_store/v1/global_store.proto
message WorkerHeartbeatRequest { string worker_id = 1; uint64 mem_pool_available_size = 2; bool accepting_new_requests = 3; uint64 state_version = 4; string state_checksum = 5; repeated string registered_artifact_ids = 6; int64 last_successful_sync = 7; ConnectionStatus global_store_status = 8; }
message WorkerHeartbeatResponse { Status status = 1; bool state_sync_required = 2; uint64 expected_state_version = 3; repeated string obsolete_replicas = 4; google.protobuf.Timestamp server_timestamp_ts = 5; }
```

```721:780:tensorcast/global_store/grpc_service.py
def _handle_enhanced_heartbeat(...):
    success = self.worker_service.heartbeat(...)
    current_version = self.recovery_service.get_worker_state_version(request.worker_id)
    state_sync_required = request.state_version < current_version
    if request.state_checksum:
        server_checksum = self.recovery_service.get_worker_state_checksum(request.worker_id)
        if request.state_checksum != server_checksum:
            state_sync_required = True
    obsolete_replicas = self.recovery_service.get_obsolete_artifacts(...)
    if obsolete_replicas:
        state_sync_required = True
    return WorkerHeartbeatResponse(status=STATUS_OK, state_sync_required=state_sync_required, expected_state_version=current_version, obsolete_replicas=obsolete_replicas, server_timestamp_ts=ts)
```

```104:128:daemon/worker_lifecycle_manager.cc
const auto infos = engine_->get_all_replicas_info();
std::vector<std::string> registered_ids; for (const auto& i : infos) registered_ids.push_back(i.artifact_id);
state_checksum_ = compute_state_checksum(infos);
auto hb_or = global_store_->send_heartbeat_enhanced(*worker_id_, engine_->get_available_memory(), accepting, state_version_, state_checksum_, registered_ids, last_sync_success_ts_, global_store::CONNECTION_STATUS_CONNECTED);
```

The lifecycle manager now constructs its `GlobalStoreClient` (channel + stub) during instantiation; the handle is `gsl::not_null`, node identity is captured once at construction, and the worker id is populated during registration so `start()` only performs the health-check handshake and worker registration.

## State Synchronization

When `state_sync_required` is true or version/checksum diverge, the daemon submits the authoritative inventory via `SynchronizeWorkerState`.

Global Store computes changes with “addition over removal” semantics and applies them; removals are suppressed when the daemon does not supply inventory.

```227:304:tensorcast/global_store/services/recovery_service.py
# to_add = local \ global; to_remove = (global \ local) only if local inventory is non-empty
```

```699:727:core/store/components/global_store_client.cc
absl::StatusOr<std::pair<uint64_t, std::string>> GlobalStoreClient::synchronize_worker_state(...)
```

## Full-State Sync

The daemon can request expected replicas for a full reconciliation after cold start or drift.

```729:756:core/store/components/global_store_client.cc
absl::StatusOr<std::pair<uint64_t, std::string>> GlobalStoreClient::request_full_state_sync(...)
```

## Connection Management & Retries (Daemon)

All RPCs are wrapped with bounded retries, exponential backoff and deadlines.

```579:626:core/store/components/global_store_client.cc
template <typename Request, typename Response, typename RpcMethod>
absl::Status GlobalStoreClient::execute_rpc_with_retry(...) { /* retry with jitter */ }
```

## Responsibilities

- Global Store: persist registry; accept heartbeats and synchronize state; compute and apply changes; serve expected state for full sync.
- Store Daemon: maintain authoritative local inventory; send enhanced heartbeats; perform incremental/full syncs; apply obsolete removals.

## Protocols (authoritative)

```151:177:proto/tensorcast/global_store/v1/global_store.proto
message WorkerLocalState { string worker_id = 1; uint64 state_version = 2; string state_checksum = 3; repeated tensorcast.common.v1.ReplicaInfo local_replicas = 4; google.protobuf.Timestamp last_update_ts = 5; }
message SynchronizeWorkerStateRequest { string worker_id = 1; WorkerLocalState local_state = 2; bool force_full_sync = 3; }
message SynchronizeWorkerStateResponse { Status status = 1; uint64 new_state_version = 2; repeated StateChange state_changes = 3; string new_state_checksum = 4; }
message StateChange { enum ChangeType { CHANGE_TYPE_UNSPECIFIED = 0; CHANGE_TYPE_REMOVE_REPLICA = 1; CHANGE_TYPE_UPDATE_REPLICA = 2; CHANGE_TYPE_ADD_REPLICA = 3; } ChangeType type = 1; tensorcast.common.v1.ReplicaInfo replica_info = 2; string reason = 3; }
```

## End-to-End Flows

- Startup recovery: validate DB, mark stale, reset versions → workers re-register/heartbeat → server may request sync.
- Heartbeat loop: daemon sends enhanced heartbeat → server compares version/checksum and obsolete set → may request sync.
- Incremental sync: daemon sends `WorkerLocalState.local_replicas` snapshot → server computes StateChange list and applies → returns new version/checksum.
- Full-state sync: daemon requests expected replicas → applies locally, resumes incremental syncs.

## Reconciliation Invariants

- Authoritative inventory: daemon’s `local_replicas` is the source of truth for local presence.
- Addition over removal: removals are suppressed if the daemon provides no inventory; only add/remove based on explicit differences.
- Idempotency: repeated adds resolve to the same replica logically.

## Configuration

### Global Store (Unified Config)

请在 Global Store 配置文件中设置 `server.listen.port`、`server.max_workers`、`database.db_file`、`worker_policy.{heartbeat_timeout,cleanup_interval,default_heartbeat_interval}` 等字段。

### Store Daemon

```yaml
high_availability:
  enabled: true
  connection_retry:
    max_attempts: -1
    base_delay_ms: 1000
    max_delay_ms: 60000
    backoff_multiplier: 2.0
  state_sync:
    heartbeat_enhanced: true
    state_checksum_enabled: true
    periodic_sync_interval_ms: 600000
```

## Observability

- Connection: `global_store_connection_status`, `connection_retry_attempts_total`, `connection_downtime_seconds`
- Sync: `state_sync_operations_total`, `state_sync_duration_seconds`, `state_inconsistencies_detected_total`
- Recovery: `recovery_operations_total`, `recovery_success_rate`, `recovery_duration_seconds`

## Troubleshooting

- Heartbeat OK but frequent sync requests: verify checksum computation on daemon; check for rapidly changing inventories.
- Unexpected removals: ensure daemon includes full `local_replicas`; server suppresses removals if inventory is empty.
- RPC failures: tune `GlobalStoreClientConfig` timeouts/backoff; check network and server logs.

## References

- `docs/architecture/architecture-overview.md`
- `docs/architecture/p2p-transfer-strategies.md`
- `docs/deployment/global-store-deployment.md`
