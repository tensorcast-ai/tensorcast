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

If you omit --config, the CLI tries $TENSORCAST_DAEMON_CONFIG, ~/.tensorcast/store_daemon_config.yaml, or examples/config/store_daemon_config.yaml.
Use `--host` / `--port` to override the gRPC listen address for CI or sandboxed runs.
```

The CLI locates the binary from the wheel or development path automatically
and extends ``LD_LIBRARY_PATH`` with the TensorCast shared library bundle as
well as the PyTorch, NVIDIA, and auxiliary CUDA runtime directories (including
packages such as ``cusparselt`` that are installed outside the ``nvidia``
namespace) that live inside the active Python environment. This allows the
daemon to resolve ``libstore_engine``, ``libtorch`` and CUDA components even
when only the binary is present on disk.

When launching in non-blocking mode (`tensorcast start --non-blocking`), the CLI now
mirrors daemon stdout and stderr into the invoking terminal in addition to
persisting them under `~/.tensorcast/sessions/<id>/logs`. Library callers can
disable console mirroring by invoking `start_service(..., to_console=False)` if
needed. The CLI also validates that the daemon process stays alive before
waiting for the gRPC ready probe, failing fast with the daemon's exit code when
configuration issues prevent startup. Failed launches leave the session log
directory intact so the captured `daemon.out` / `daemon.err` streams remain
available for troubleshooting.

## Manage Daemon Sessions

Daemon sessions are tracked under `~/.tensorcast/sessions/<session_id>` and the
current session id is stored in `~/.tensorcast/current_session`.

Common commands:

```
# Status (connects to daemon gRPC if available, otherwise shows process info)
uv run -q python -m tensorcast.cli status

# Logs (stdout by default, --stderr for stderr; add -f to follow)
uv run -q python -m tensorcast.cli logs -f

# Stop current session (SIGTERM with SIGKILL fallback)
uv run -q python -m tensorcast.cli stop
```

## Observability

Metrics are exposed via the unified system; the daemon no longer provides an HTTP metrics endpoint.

### Logging

- `observability.logging.level` drives the daemon's stderr threshold and minimum log level (DEBUG is routed through VLOG).
- `observability.logging.vlog_level` sets the global `VLOG` verbosity; values <= 0 disable verbose logging.
- `observability.logging.file` writes plain-text logs to disk in addition to stderr; the sink is hot-swappable at runtime via config reloads.
- When `observability.logging.otel_context_enabled` and `observability.logging.sink_file` are set, the daemon writes a second log file enriched with OpenTelemetry `trace_id`/`span_id` for correlation.

## Configuration

All runtime parameters are configured via the unified config. The daemon only
accepts `--config=/path/to/file`. See `examples/config/store_daemon_config.yaml`.
Enum fields accept friendly values and are normalized (case-insensitive): `observability.otel.exporter_protocol: grpc | http/protobuf`, `observability.logging.level: debug|info|warn|error`.
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
