---
title: Store Daemon Deployment
description: Running the C++ StoreDaemon with unified runtime config
sidebar_position: 4
---

# Store Daemon Deployment

This page describes how to run the C++ StoreDaemon (`daemon/tensorcast_daemon`) in development and production using the unified runtime configuration.

## Binary

- Build from source (development):

```
bazel build //daemon:tensorcast_daemon
```

- Packaged wheel includes the daemon at `tensorcast/bin/tensorcast_daemon` and the Python CLI will use it automatically.

## Launch via Python CLI

Use the unified YAML config and start via CLI:

```
uv run -q python -m tensorcast.cli start --non-blocking --config=examples/config/store_daemon_config.yaml
```

The CLI locates the binary from the wheel or development path automatically.

## Observability

Metrics are exposed via the unified system; the daemon no longer provides an HTTP metrics endpoint.

## Configuration

All runtime parameters are configured via the unified config. See `examples/config/store_daemon_config.yaml`.
```yaml
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

## Communicator Configuration (YAML)

P2P/RDMA communicator is configured via a single YAML/JSON file (no per‑field flags). Example:

```yaml
enable_rdma: false
stager:
  stage_cpu_for_rdma: true
  stage_chunk_mb_cpu: 4
  stage_chunk_mb_gpu: 16
  buffers_per_flow: 4
rdma:
  outstanding_wr: 64
  ack_ttl_ms: 30000
  traffic_class: 186
  qp_timeout: 20
  qp_retry: 7
pool:
  preregister_mr: true
  pool_size_bytes: 8589934592
  chunk_bytes: 67108864
transport:
  tcp_conn_count: 8
  connect_timeout_sec: 10
  tcp_tos: 0
```

Merge the communicator configuration into the daemon's unified config under `communicator.*`, and launch with `--config`.

## Launch Example (Unified Config)

```
bazel-bin/daemon/tensorcast_daemon --config=examples/config/store_daemon_config.yaml
```

### Metrics Exposure (Unified)

- The daemon no longer serves metrics directly. Use the central observability pipeline for `tc_*` metrics:

```
# Metrics HTTP sidecar has been removed. Use central observability pipeline.
```

Metrics include, e.g.:
- `tc_memory_pool_bytes{location=cpu|gpu,device_id?,memory_type=total|free}`
- `tc_p2p_bytes_total`

Note: The HTTP metrics sidecar is no longer spawned by the CLI.
