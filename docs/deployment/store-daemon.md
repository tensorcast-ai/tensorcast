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

Use the unified YAML config and start via CLI (default `--global-store-mode` is `none`):

```
uv run tensorcast-cli daemon start --global-store-mode connect --global-store-address 127.0.0.1:50051
```

If you omit `--config`, the CLI tries `$TENSORCAST_DAEMON_CONFIG` or
`~/.tensorcast/config/daemon.yaml` and otherwise materializes an embedded
loopback config (port=0, storage under `~/.tensorcast`). Set listen/advertise
addresses through the config file instead of CLI flags.
You can also override config values inline with `--set KEY=VALUE` (repeatable).
Example: `--set engine.memory_tiers.stable_bytes=4GB`.
Common shortcuts: `--stable-bytes`, `--mem-pool-size-bytes`, `--pinned-pool-bytes`,
`--enable-rdma`, `--log-level`.

The CLI locates the binary from the wheel or development path automatically
and extends ``LD_LIBRARY_PATH`` with the TensorCast shared library bundle as
well as the PyTorch, NVIDIA, and auxiliary CUDA runtime directories (including
packages such as ``cusparselt`` that are installed outside the ``nvidia``
namespace) that live inside the active Python environment. This allows the
daemon to resolve ``libstore_engine``, ``libtorch`` and CUDA components even
when only the binary is present on disk.

During startup, the CLI can mirror daemon stdout and stderr into the invoking
terminal in addition to persisting them under `~/.tensorcast/sessions/<id>/logs`.

## Manage Daemon Sessions

Daemon sessions are tracked under `~/.tensorcast/sessions/<session_id>` and the
current session id is stored in `~/.tensorcast/current_session`.

Only one Store Daemon instance is allowed per `$TENSORCAST_HOME`. If you run
`tensorcast-cli daemon start` while another daemon is already running, the CLI
returns an error and prints the existing daemon session details; reuse that
instance or stop it before starting a new one.

When `tensorcast-cli daemon stop` is invoked without a session id, the CLI
resolves the active daemon from `~/.tensorcast/runtime/state.json` first, then
falls back to `~/.tensorcast/current_session`.

Common commands:

```
# Status (connects to daemon gRPC if available, otherwise shows process info)
uv run tensorcast-cli daemon status

# Logs (stdout by default, --stderr for stderr; add -f to follow)
uv run tensorcast-cli daemon logs -f

# Stop current session (SIGTERM with SIGKILL fallback)
uv run tensorcast-cli daemon stop
```

## Observability

Metrics are exposed via the unified system; the daemon no longer provides an HTTP metrics endpoint.

### Store Client Sessions

- `uv run tensorcast daemon status` now prints a *Store Sessions* section after the daemon health report. Data is sourced from `~/.tensorcast/store_sessions/<session_id>.json`, which the Python SDK refreshes whenever a Store verb completes. Use this view to spot clients that still hold leases or in-flight futures before forcing revocation.
- Each session entry includes daemon endpoint, client PID, timestamps, active lease count, pending futures, and any capabilities reported by `Store.__init__` (pool size, transfer slice, lease support).

### Store Client Metrics (Grafana Example)

```json
{
  "title": "Store Operation Latency",
  "type": "timeseries",
  "fieldConfig": {
    "defaults": {
      "unit": "ms",
      "transformations": []
    },
    "overrides": []
  },
  "targets": [
    {
      "expr": "histogram_quantile(0.95, sum by (le, verb) (rate(tc_store_operation_latency_seconds_bucket{daemon="$daemon"}[5m])))",
      "legendFormat": "{{verb}} p95"
    },
    {
      "expr": "sum by (verb) (rate(tc_store_operation_errors_total{daemon="$daemon"}[5m]))",
      "legendFormat": "{{verb}} errors/s",
      "yaxis": 2
    }
  ],
  "options": {
    "tooltip": {
      "mode": "single"
    }
  }
}
```

Pair this panel with a counter visualization for `tc_store_operation_retries_total` to highlight retry-heavy verbs. Filter on the `daemon` label to compare multiple Store sessions in the same dashboard.

## Store Session API Rollout & Backout

### Rollout checklist

1. **Version alignment**: Ensure the staged Global Store schema, Store Daemon binary, and Python SDK wheel come from the same release. Run `uv run tensorcast --version` and `uv run tensorcast daemon status` to confirm the daemon reports the expected build metadata.
2. **Pre-traffic validation**: Against staging, execute `uv run pytest tests/python/test_register_lease_in_place_helper.py`, `uv run pytest tests/python/test_register_vram_leased_and_dvmp_stream.py`, and `bazel test //daemon:session_lifecycle_test --define=use_fake_cuda=true`. These suites cover lease renewal, VRAM leased-in-place flows, and daemon session lifecycle.
3. **Metrics watch**: Monitor the OpenTelemetry metrics defined in [Design 0010](../designs/0010-opentelemetry-unified-observability-design.md)—`tc_store_operation_latency_seconds`, `tc_store_operation_errors_total`, and `tc_store_operation_retries_total`—while introducing production traffic. Alert thresholds should track the historical p95 latency and error envelopes before legacy helpers are disabled.
4. **Session audit**: Use `uv run tensorcast daemon status` to inspect the *Store Sessions* section and verify the session registry under `~/.tensorcast/store_sessions` reflects active clients with the expected lease/future counts.
5. **Release checklist**: Cross-check the deployment steps against the [Store Session Release Checklist](./store-session-release-checklist.md) before announcing completion.

### Backout checklist

- **Binary rollback**: Redeploy the previous Store Daemon binary and Python SDK wheel (pre-Store-session release). Older clients ignore the `.tensorcast/store_sessions` manifests, so no cleanup is required beyond optional file pruning.
- **Verification**: Re-run the validation suites above and confirm observability indicators (`tc_store_operation_errors_total`, `tc_store_operation_retries_total`) return to baseline values.
- **Communication**: Notify on-call and consumer teams when rollback occurs, document the failure mode, and schedule a postmortem before attempting another rollout.

### Logging

- `observability.logging.level` drives the daemon's stderr threshold and minimum log level (DEBUG is routed through VLOG).
- `observability.logging.vlog_level` sets the global `VLOG` verbosity; values <= 0 disable verbose logging.
- `observability.logging.file` writes plain-text logs to disk in addition to stderr; the sink is hot-swappable at runtime via config reloads.
- When `observability.logging.otel_context_enabled` and `observability.logging.sink_file` are set, the daemon writes a second log file enriched with OpenTelemetry `trace_id`/`span_id` for correlation.

## Configuration

All runtime parameters are configured via the unified config. The daemon only
accepts `--config=/path/to/file`. See `examples/config/store_daemon_config.yaml`.
Enum fields accept friendly values and are normalized (case-insensitive): `observability.otel.exporter_protocol: grpc | http/protobuf`, `observability.logging.level: debug|info|warn|error`.

For long-lived clients, prefer a non-zero `server.grpc.max_connection_idle` and
explicit keepalive settings (for example `keepalive_time: 30s`,
`keepalive_timeout: 10s`) to avoid idle GOAWAY churn. The Python SDK reuses a
single gRPC channel/stub per `DaemonCtl` instance, so keep the instance around
instead of recreating it for each RPC.
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
  pool_size_bytes: 8GB
  chunk_bytes: 64MB
transport:
  tcp_conn_count: 8
  connect_timeout_sec: 10
  tcp_tos: 0
```

Merge the communicator configuration into the daemon's unified config under `communicator.*`, and launch with `--config`. `pool_size_bytes` and `chunk_bytes` accept humanized sizes like `8GB`/`64MB` when embedded in the daemon config.

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
