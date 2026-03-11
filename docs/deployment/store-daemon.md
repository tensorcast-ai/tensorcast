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

If you omit `--config`, the CLI tries `$TENSORCAST_DAEMON_CONFIG`, then
`examples/config/store_daemon_config.yaml` (repo checkout or packaged wheel),
and errors if no config is found. Set listen/advertise addresses through the
config file instead of CLI flags.
You can also override config values inline with `--set KEY=VALUE` (repeatable).
Example: `--set engine.memory_tiers.stable_bytes=4GB`.
Common shortcuts: `--stable-bytes`, `--mem-pool-size-bytes`, `--enable-rdma`, `--log-level`.

The CLI locates the binary from the wheel or development path automatically
and extends ``LD_LIBRARY_PATH`` with the TensorCast shared library bundle as
well as the PyTorch, NVIDIA, and auxiliary CUDA runtime directories (including
packages such as ``cusparselt`` that are installed outside the ``nvidia``
namespace) that live inside the active Python environment. This allows the
daemon to resolve ``libstore_engine``, ``libtorch`` and CUDA components even
when only the binary is present on disk.
If you need extra launcher environment for the daemon binary, set `envs` in the
daemon config. `envs.LD_LIBRARY_PATH` is merged in this order: inherited shell
entries, then configured entries, then the auto-discovered TensorCast/PyTorch/CUDA
directories. Other `envs.*` keys are passed through directly to the daemon
process.

By default the daemon runs in the background after `tensorcast-cli daemon start`
returns, and logs are persisted under `~/.tensorcast/hosts/<host_id>/sessions/<id>/logs`
(view them with `uv run tensorcast-cli daemon logs`). Add `--blocking` to keep the daemon
attached to the CLI, stream logs directly to the terminal (stdio inherited to
avoid buffered crash output), and stop it when the CLI exits (SIGTERM with a
~35s grace before SIGKILL). In blocking mode, logs are not persisted to the
session log files.

## Manage Daemon Sessions

Daemon sessions are tracked under `~/.tensorcast/hosts/<host_id>/sessions/<session_id>` and the
current session id is stored in `~/.tensorcast/hosts/<host_id>/current_session`
(where `<host_id>` is derived from the hostname and machine-id).
Session metadata is written as soon as the daemon process starts (before
readiness), so SDK `tc.init(mode="connect")` can discover the session while
startup continues; retry if the daemon is still initializing. The CLI emits
periodic readiness status messages if startup takes longer than a few seconds
and probes both the configured listen host and loopback to avoid hanging on a
non-local listen address. Transient readiness probe errors are suppressed during
startup and only surfaced if the daemon fails to become ready.

Only one Store Daemon instance is allowed per host-scoped runtime root under
`$TENSORCAST_HOME`. If you run `tensorcast-cli daemon start` while another daemon
is already running, the CLI
returns an error and prints the existing daemon session details; reuse that
instance or stop it before starting a new one.

When `tensorcast-cli daemon stop` is invoked without a session id, the CLI
resolves the active daemon from `~/.tensorcast/hosts/<host_id>/runtime/state.json`
first, then falls back to `~/.tensorcast/hosts/<host_id>/current_session`.

Common commands:

```
# Status (connects to daemon gRPC if available, otherwise shows process info)
uv run tensorcast-cli daemon status

# Logs (stdout by default, --stderr for stderr; add -f to follow)
uv run tensorcast-cli daemon logs -f

# Stop current session (SIGTERM with a ~35s grace before SIGKILL)
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
2. **Pre-traffic validation**: Against staging, execute `uv run pytest tests/python/test_register_lease_in_place_helper.py`, `uv run pytest tests/python/test_register_vram_leased_and_dvmp_stream.py`, and `bazel test //daemon:session_lifecycle_test --test_env=TENSORCAST_CUDA_BACKEND=fake`. These suites cover lease renewal, VRAM leased-in-place flows, and daemon session lifecycle.
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
Launcher-only daemon environment can also be declared in config via `envs`, for
example:

```yaml
envs:
  LD_LIBRARY_PATH: /data/cuda/compat
  NCCL_DEBUG: INFO
```

When HA is enabled, the daemon advertises a routable address to the Global Store. If `server.advertise.host` is set but non-routable, startup fails; if it is unset, the daemon resolves it using a routable `server.listen.host`, the outbound route IP to the Global Store endpoint, and finally the default interface IP. The resolved advertise address is logged at startup.

For long-lived clients, prefer a non-zero `server.grpc.max_connection_idle` and
explicit keepalive settings (for example `keepalive_time: 30s`,
`keepalive_timeout: 10s`) to avoid idle GOAWAY churn. The Python SDK reuses a
single gRPC channel/stub per `DaemonCtl` instance, so keep the instance around
instead of recreating it for each RPC.

Replica promotion/export for P2P routing is controlled via the `promotion` block
(`policy`, `require_verified`, `demotion_drain_timeout`). The default
`promotion.policy` is `never`, which means replicas remain presence-only unless
promotion is explicitly enabled and requested by the SDK.
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
communicator:
  enable_rdma: false
  stager:
    stage_cpu_for_rdma: true
    buffers_per_flow: 16
    expected_gpu_channels: 0
  rdma:
    outstanding_wr: 64
    ack_ttl_ms: 30000
    traffic_class: 186
    qp_timeout: 20
    qp_retry: 7
  transport:
    tcp_conn_count: 20
    connect_timeout_sec: 10
    tcp_tos: 0
```

Pinned staging pool sizing and chunking come from the daemon-wide pinned memory configuration (`DaemonConfig.pinned_memory`) via the `comm_gpu` / `comm_cpu` classes (`slice_bytes` + `pool_bytes`), not from `CommunicatorConfig`.

## RDMA Environment Variables

These are optional and only affect RDMA device selection. Runtime parameters
still live in the unified config file.

### TENSORCAST_IB_HCA

Specifies InfiniBand HCA device names to use for RDMA. Multiple values are
comma-separated. Any `=` characters in the value are stripped.

```bash
export TENSORCAST_IB_HCA="mlx5_bond0"
export TENSORCAST_IB_HCA="mlx5_bond0,mlx5_bond1"
```

If unset, TensorCast auto-discovers available devices.

### TENSORCAST_LLDP_FILE_NAME

Path to an LLDP-style mapping file for rail ID selection in multi-rail
configurations. Each non-comment line maps a network interface to PCI path,
mlx5 device name, and rail ID.

```
eth1=0000:19:00.0,mlx5_bond100,1
```

```bash
export TENSORCAST_LLDP_FILE_NAME="/path/to/lldp_config.txt"
```

Lines starting with `#` and blank lines are ignored. If unset, rail IDs are
derived from the mlx5 device name (for example `mlx5_0` -> rail 0).

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
