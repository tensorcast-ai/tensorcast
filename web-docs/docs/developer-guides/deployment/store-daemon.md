---
title: Store Daemon Deployment
description: A guide for deploying and managing the StoreDaemon in production.
sidebar_position: 2
---

# Store Daemon Deployment Guide

This guide provides practical instructions for deploying, configuring, and managing the `StoreDaemon` service in a production environment.

## Prerequisites

- A server with a supported Linux distribution.
- Python 3.10+ installed.
- Access to the StepCast Store artifacts (Python wheel, dependencies).
- (Optional but Recommended) NVIDIA drivers, CUDA Toolkit, and `nvcc` installed for GPU support.
- (Optional) A running [Global Store](./global-store.md) instance if using worker management features.

## Installation

1.  **Create a dedicated user** for running the service to enhance security:

    ```bash
    sudo useradd --system --no-create-home --shell /bin/false store-daemon
    ```

2.  **Set up the environment**:
    - Create an installation directory (e.g., `/opt/stepcast-store`).
    - Create a Python virtual environment.
    - Install the `stepcast-store` wheel and its dependencies using `uv` or `pip`.

    ```bash
    sudo mkdir -p /opt/stepcast-store
    sudo chown -R $(whoami):$(whoami) /opt/stepcast-store
    cd /opt/stepcast-store

    python3 -m venv .venv
    source .venv/bin/activate
    pip install /path/to/stepcast_model_store-*.whl
    ```

3.  **Create necessary directories** for storage, logs, and PID files:

    ```bash
    sudo mkdir -p /data/models /var/log/store-daemon /var/run/store-daemon
    sudo chown -R store-daemon:store-daemon /data/models /var/log/store-daemon /var/run/store-daemon
    ```

## Configuration

The `StoreDaemon` is configured via a YAML file. It is recommended to place this file in a standard location like `/etc/stepcast/store-daemon.yaml`.

### Sample Configuration (`store-daemon.yaml`)

```yaml
# /etc/stepcast/store-daemon.yaml

server:
  host: "0.0.0.0"
  port: 50052
  storage_path: "/data/models"
  num_threads: 16
  chunk_size: 256MB        # Human-readable units now supported
  mem_pool_size: 8GB       # Ditto
  enable_p2p_engine: true
  enable_p2p_access: true
  enable_rdma: false
  pinned_memory_timeout_ms: 30000

# Register with a Global Store (optional)
global_store_address: "global-store.service.consul:50051"

high_availability:
  enabled: true
  heartbeat_interval_ms: 5000
  registration_retry_delay_ms: 1000
  max_retries: 10

network:
  p2p_port: 9090     # RDMA/TCP data plane
  metrics_port: 9091 # Prometheus scrape
  health_check_port: 8080

shutdown:
  grace_period_ms: 30000  # 30 seconds

lifecycle:
  gpu_memory_limit_fraction: 0.75 # Start evicting when usage > 75 %
  global_cache_fraction: 0.20     # Reserve 20 % for globally cached artifacts
  proc_check_interval_s: 5.0
  eviction_check_interval_s: 30.0
```

### Key Parameters

- `server.storage_path`: The **most critical** setting. This is the root directory where artifacts are stored on the shared filesystem (e.g., JuiceFS, NFS).
- `server.mem_pool_size`: The size of the pre-allocated pinned memory pool for staging data. A larger pool can improve performance for concurrent loads but consumes more system RAM.
- `global_store_address`: Set this to register with the Global Store. If omitted, the daemon runs in standalone mode.
- `high_availability.heartbeat_interval_ms`: How often the daemon sends heartbeat messages when registered.
- `high_availability.registration_retry_delay_ms`: Delay between retries when registration with the Global Store fails.
- `lifecycle.gpu_memory_limit_fraction`: GPU usage threshold that triggers automatic eviction of unused artifacts. Default is now **0.75**.

## Running the Service

While you can run the daemon directly, using a service manager like `systemd` is highly recommended for production environments to handle logging, automatic restarts, and resource management.

### Using the CLI (for testing)

The `scstore` command-line tool can be used to start, stop, and check the status of the daemon.

```bash
# Start the daemon in the background using a config file
scstore start --config /etc/stepcast/store-daemon.yaml

# Check the status
scstore status

# Stop the daemon gracefully
scstore stop
```

### 🏃‍♂️ Local Quick-start (No Global Store)

If you simply want to explore the **StoreDaemon** on a single machine without a running Global Store, the following one-liner is enough:

```bash
scstore start \
  --storage-path "$(pwd)/models" \
  --mem-pool-size 4GB \
  --enable-p2p-engine \
  --enable-p2p-access \
  --blocking
```

Key points:

* The daemon will listen on `0.0.0.0:8073` (default).
* Artifacts will be stored under `./models` in the current directory.
* `--blocking` keeps the process in the foreground so logs are printed to your terminal. Omit it to run in the background.
* No Global Store registration occurs because we did **not** pass `--global-store-address`.

---

### 🔧 Command-line Flags Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--config, -c` | *(none)* | Path to YAML configuration file. CLI flags always override values read from the file. |
| `--host` | `0.0.0.0` | Bind address for the gRPC server. |
| `--port` | `8073` | gRPC port for client connections. |
| `--storage-path` | `""` | Directory for artifact files (required for local mode). |
| `--mem-pool-size` | `8GB` | Size of the pinned memory staging pool. Accepts human-readable units (e.g. `64MB`, `1GB`). |
| `--num-threads` | `10` | Size of the internal worker thread pool. |
| `--chunk-size` | `32MB` | Maximum chunk size sent over the wire. |
| `--enable-p2p-engine` | `False` | Enable the high-performance communication engine. |
| `--enable-p2p-access` | `False` | Require explicit artifact registration before load. Useful for fine-grained access control. |
| `--enable-rdma` | `False` | Enable RDMA transport inside the communication engine. Requires compatible NIC & kernel modules. |
| `--global-store-address` | *(none)* | Address of the Global Store in the form `host:port`. If omitted, the daemon runs in standalone mode. |
| `--p2p-port` | `9090` | Port used by the P2P data plane (TCP or RDMA). |
| `--metrics-port` | `9091` | HTTP port that exposes Prometheus metrics. |
| `--health-check-port` | `8080` | HTTP port for `/health`, `/ready`, and `/status` endpoints. |
| `--pinned-memory-timeout-ms` | `30000` | Timeout (ms) when allocating from the pinned memory pool. |
| `--blocking / --non-blocking` | `--non-blocking` | Run in the foreground or as a daemon. |
| `--pid-file` | `/var/run/store-daemon/store-daemon.pid` | PID file for daemon mode. |
| `--log-file` | `/var/log/store-daemon/store-daemon.log` | Log file for daemon mode. |
| `--verbose, -v` | *(flag)* | Enable verbose logging. |

> 💡 **Tip**
> Any parameter not available as a CLI flag can be set in the YAML file (see below). Command-line flags always take precedence over YAML.

---

### 📝 Configuration File Schema (YAML)

Below is a flattened list of the most important keys recognised by `StoreDaemon`.  They correspond 1-to-1 with the [`StoreDaemonConfig`](https://github.com/stepai/stepcast-store/blob/main/scstore/store_daemon/config.py) Pydantic artifact.

| Key | Default | Description |
|-----|---------|-------------|
| `server.host` | `0.0.0.0` | Bind address for the gRPC server. |
| `server.port` | `50052` | gRPC port. |
| `server.storage_path` | `/tmp/models` | Local directory for artifact storage. **Must exist**. |
| `server.mem_pool_size` | `8GB` | Pinned memory pool size (human-readable units supported). |
| `server.enable_p2p_engine` | `true` | Enable the communication engine. |
| `global_store_address` | *(none)* | Global Store address (`host:port`). |
| `high_availability.heartbeat_interval_ms` | `5000` | Heartbeat interval when connected to Global Store. |
| `high_availability.registration_retry_delay_ms` | `1000` | Retry delay when registration fails. |
| `network.p2p_port` | `9090` | P2P data plane port. |
| `network.metrics_port` | `9091` | Prometheus scrape port. |
| `network.health_check_port` | `8080` | Health check & readiness probe port. |
| `lifecycle.gpu_memory_limit_fraction` | `0.75` | GPU memory utilisation threshold before eviction kicks in. |
| `lifecycle.global_cache_fraction` | `0.20` | GPU memory reserved for global cache. |
| `shutdown.grace_period_ms` | `30000` | Graceful shutdown window after receiving SIGTERM. |

You can start the daemon with **only** the YAML file:

```bash
scstore start --config /etc/stepcast/store-daemon.yaml
```

and optionally override individual keys via CLI flags.

## Monitoring

Effective monitoring is crucial for a healthy `StoreDaemon` deployment.

### Health Checks
Use the HTTP health endpoints with a load balancer or monitoring system (like Consul, Kubernetes, or Nagios).

- **Liveness Probe**: `curl http://<daemon-host>:8080/health`
  - Use this to determine if the process is running. If it fails, the process should be restarted.
- **Readiness Probe**: `curl http://<daemon-host>:8080/ready`
  - Use this to determine if the daemon should receive traffic. It checks if the daemon is initialized and registered with the Global Store.
- **Detailed Status**: `curl http://<daemon-host>:8080/status`
  - Provides a detailed JSON object for debugging, showing worker ID, memory usage, active operations, and more.

### Prometheus Metrics
Scrape the `/metrics` endpoint (default port `9091`) with a Prometheus server.

**Key Metrics to Alert On**:
```text
- `up{job="store-daemon"} == 0`: The daemon is unreachable.
- `store_daemon_models_load_failures_total`: A high or increasing rate of load failures indicates a problem with storage access or configuration.
- `store_daemon_memory_pool_available_bytes`: If available memory is consistently low, consider increasing `mem_pool_size` or adjusting eviction policies.
- `store_daemon_worker_registered == 0`: The worker is not registered with the Global Store, indicating a connectivity or configuration issue.
- `store_daemon_p2p_transfer_errors_total`: An increasing count indicates problems with the P2P communication layer.
```

