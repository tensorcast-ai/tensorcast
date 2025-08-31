---
title: Store Daemon Deployment
description: Running the C++ StoreDaemon and configuring compatibility flags
sidebar_position: 4
---

# Store Daemon Deployment

This page describes how to run the C++ StoreDaemon (`daemon/tensorcast_daemon`) in development and production, and how to configure compatibility flags for parity with the legacy Python daemon.

## Binary

- Build from source (development):

```
bazel build //daemon:tensorcast_daemon
```

- Packaged wheel includes the daemon at `tensorcast/bin/tensorcast_daemon` and the Python CLI will use it automatically.

## Launch via Python CLI

Use the typed YAML config (`tensorcast.store_daemon.config`) and the CLI translates it into C++ flags.

```
uv run -q python -m tensorcast.cli start --non-blocking --host 127.0.0.1 --port 8073
```

The CLI locates the binary in this order:

- `TENSORCAST_DAEMON_BIN` env var
- Installed package `tensorcast/bin/tensorcast_daemon`
- Development path `bazel-bin/daemon/tensorcast_daemon`

## Observability

Metrics are exposed via the unified system; the daemon no longer provides an HTTP metrics endpoint.

## Global Store Mode

When `--global_store_addr` is set, the daemon registers as a worker and sends periodic heartbeats. It can also optionally synchronize DVMP chunk states.

Recommended flags:

- `--global_store_addr=HOST:PORT`
- `--heartbeat_interval_ms=5000`
- `--chunk_sync_interval_ms=10000` (set `0` to disable)

## Compatibility Flags

These flags align the C++ daemon’s external behavior to the Python daemon where needed:

- `--enable_p2p_access` (default `true`): Global toggle for remote memory registration
- `--confirm_requires_disk_path` (binary default `false`; Python launcher defaults to `true`): Strict ConfirmReplica; reject empty `disk_path` and non‑GPU targets
- `--verification_timeout_status=ok|deadline` (default `ok`): Map verification timeouts
- `--auto_register_disk_loads` (default `false`): Auto register local disk loads after ready
- `--force_full_digest_on_load` (default `false`): Strong verification via content addressing
- `--evict_on_dead_pid` (default `false`): Unload replicas when all PID refs are gone

## Lifecycle & Eviction Flags

Daemon-level periodic eviction mirrors the Python `LifecycleWorker` policy and is disabled by default.

- `--enable_periodic_eviction` (default `false`): Enable periodic eviction loop
- `--eviction_check_interval_ms` (default `30000`): Eviction evaluation interval
- `--gpu_memory_limit_fraction` (default `0.75`): Per‑GPU usage threshold to trigger LRU eviction

When enabled, the daemon will evict least‑recently‑used GPU replicas on devices whose used/total exceeds the threshold, skipping replicas that have active PID refs or are marked keep_for_global.

All flags are available via the Python config:

```yaml
server:
  enable_p2p_engine: true
  enable_rdma: false
  enable_p2p_access: true
  confirm_requires_disk_path: false
  verification_timeout_status: ok   # or: deadline
  auto_register_disk_loads: false
  force_full_digest_on_load: false
network:
high_availability:
  heartbeat_interval_ms: 5000
  chunk_sync_interval_ms: 10000
lifecycle:
  evict_on_dead_pid: false
  enable_periodic_eviction: false
  eviction_check_interval_s: 30.0
  gpu_memory_limit_fraction: 0.75
global_store_address: 127.0.0.1:6000
```

## Example Direct Launch

```
bazel-bin/daemon/tensorcast_daemon \
  --listen_addr=0.0.0.0:50051 \
  --metrics_port=9095 \
  --global_store_addr=127.0.0.1:6000 \
  --heartbeat_interval_ms=5000 \
  --chunk_sync_interval_ms=10000 \
  --enable_p2p_engine=false \
  --enable_p2p_access=true

## Prometheus Metrics (Daemon)

The C++ daemon exposes a minimal Prometheus endpoint on `metrics_port`:

- Memory pool gauges:
  - `store_daemon_memory_pool_total_bytes`
  - `store_daemon_memory_pool_available_bytes`
- HA lifecycle counters/gauges:
  - `store_daemon_hb_success`, `store_daemon_hb_failure`
  - `store_daemon_sync_success`, `store_daemon_sync_failure`
  - `store_daemon_last_hb_ts_s`, `store_daemon_last_sync_ts_s`
  - `store_daemon_hb_alive`, `store_daemon_sync_alive`
  - `store_daemon_hb_restarts`, `store_daemon_sync_restarts`

These reflect the Global Store heartbeat/sync loops and their liveness/restarts. Use them for alerting and dashboards.
```
