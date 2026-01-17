---
slug: ha-heartbeat-sync-decoupling
title: HA Heartbeat and State Sync Decoupling (Plan)
areas: ["daemon", "core", "global_store"]
related_code:
  - daemon/ha/worker_lifecycle_manager.{h,cc}
  - daemon/app/server_main.cc
  - core/store/components/global_store_client.{h,cc}
  - proto/tensorcast/config/v1/daemon_config.proto
  - daemon/ha/worker_lifecycle_manager_sync_test.cc
links:
  design: ../designs/0046-ha-heartbeat-sync-decoupling.md
---

# Objective

Split heartbeat and state sync responsibilities so heartbeats stay responsive, sync runs in the background, and monitor restarts do not block on stalled threads. Add per-RPC timeout policies so heartbeat calls remain short and sync calls remain bounded.

# Current State & Grounding

- Heartbeat loop handles heartbeat, state sync, memory tier publish, and lease reconcile in one thread.
  - `daemon/ha/worker_lifecycle_manager.cc`
- Monitor loop restarts by `join`ing the stalled thread, which can block indefinitely if RPCs hang.
  - `daemon/ha/worker_lifecycle_manager.cc`
- Global Store client uses a single default `rpc_timeout` and `max_retries` across all methods.
  - `core/store/components/global_store_client.{h,cc}`
- HA config already exists but lacks per-RPC timeout settings.
  - `proto/tensorcast/config/v1/daemon_config.proto`

# Phases & Milestones

- [ ] Phase 1: RPC policy plumbing
  - [ ] Milestone: add per-RPC timeout and retry fields to `HighAvailability` config.
  - [ ] Milestone: add `RpcOptions` overrides in `GlobalStoreClient` and use them from the daemon.

- [ ] Phase 2: Loop separation and cancellation
  - [ ] Milestone: split state sync into a background worker (queue + coalesce).
  - [ ] Milestone: add cancellation or generation controls for heartbeat and sync restarts.
  - [ ] Milestone: move memory tier publish/lease reconcile out of heartbeat loop or into sync worker.

- [ ] Phase 3: Observability and tests
  - [ ] Milestone: add metrics for sync queue depth, in-flight sync, and restart causes.
  - [ ] Milestone: extend lifecycle manager tests for stall detection without monitor deadlock.

# Tasks

- Extend `tensorcast.config.v1.HighAvailability` with per-RPC timeout and retry fields.
- Regenerate protos after proto changes (`bash tools/build_proto_python.sh`).
- Add `RpcOptions` struct and per-call overrides in `GlobalStoreClient`.
- Introduce a sync queue or semaphore and coalescing logic in `WorkerLifecycleManager`.
- Ensure monitor restarts do not block on `join` (use cancellation plus a reaper or bounded wait model).
- Update metrics in `WorkerLifecycleManager` and any dashboards that consume them.
- Update daemon docs (`daemon/README.md`) to describe the new HA loop split and configuration knobs.

# Test / Rollout / Backout

- Tests (C++):
  - `bazel test //daemon:worker_lifecycle_manager_sync_test`
  - `bazel test //proto/... --test_output=errors` (only if proto changed)
- Rollout:
  - Keep default timeouts aligned with current behavior; only enable shorter heartbeat deadlines when configured.
- Backout:
  - Revert to previous configuration defaults and remove loop split if needed; protocol remains unchanged.

# Risks & Tracking

- More concurrency paths can introduce ordering bugs; mitigate with single in-flight sync and clear metrics.
- Aggressive heartbeat timeouts can increase false state_sync_required; mitigate with conservative defaults and retry limits.
