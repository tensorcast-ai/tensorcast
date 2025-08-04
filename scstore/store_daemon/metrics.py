#  Copyright (c) 2025, StepCast Team.

"""Prometheus metrics definitions for StoreDaemon."""

from prometheus_client import Counter, Gauge, Histogram, Info

# Model operation metrics
MODELS_LOADED_TOTAL = Counter(
    "store_daemon_models_loaded_total",
    "Total number of models loaded",
    ["device_type", "source_type"],  # source_type: remote, disk
)

MODELS_LOAD_FAILURES_TOTAL = Counter(
    "store_daemon_models_load_failures_total",
    "Total number of model load failures",
    ["device_type", "error_type"],
)

MODEL_LOAD_DURATION = Histogram(
    "store_daemon_model_load_duration_seconds",
    "Time spent loading models",
    ["device_type", "source_type"],
)

MODELS_UNLOADED_TOTAL = Counter(
    "store_daemon_models_unloaded_total",
    "Total number of models unloaded",
    ["device_type"],
)

# Async loading metrics
MODELS_ALLOCATED_TOTAL = Counter(
    "store_daemon_models_allocated_total",
    "Total number of models with memory allocated (async)",
    ["device_type"],
)

PENDING_LOADS = Gauge(
    "store_daemon_pending_loads",
    "Number of models currently being loaded asynchronously",
)

ASYNC_LOAD_WAIT_DURATION = Histogram(
    "store_daemon_async_load_wait_duration_seconds",
    "Time spent waiting for async model loading in ConfirmModel",
    ["device_type"],
)

# System metrics
ACTIVE_OPERATIONS = Gauge(
    "store_daemon_active_operations", "Number of currently active operations"
)

MEMORY_POOL_TOTAL = Gauge(
    "store_daemon_memory_pool_total_bytes", "Total memory pool size in bytes"
)

MEMORY_POOL_AVAILABLE = Gauge(
    "store_daemon_memory_pool_available_bytes", "Available memory pool size in bytes"
)

# Worker management metrics
WORKER_REGISTERED = Gauge(
    "store_daemon_worker_registered",
    "Whether worker is registered (1=registered, 0=not registered)",
)

WORKER_HEALTHY = Gauge(
    "store_daemon_worker_healthy", "Whether worker is healthy (1=healthy, 0=unhealthy)"
)

HEARTBEATS_SENT_TOTAL = Counter(
    "store_daemon_heartbeats_sent_total", "Total number of heartbeats sent"
)

HEARTBEAT_FAILURES_TOTAL = Counter(
    "store_daemon_heartbeat_failures_total", "Total number of heartbeat failures"
)

WORKER_UPTIME_SECONDS = Gauge(
    "store_daemon_worker_uptime_seconds", "Worker uptime in seconds"
)

WORKER_INFO = Info("store_daemon_worker_info", "Worker information")

# Verification metrics
MODEL_VERIFICATION_TOTAL = Counter(
    "store_daemon_model_verification_total",
    "Total number of model integrity verification results",
    ["status"],  # passed, failed
)

MODEL_VERIFICATION_LATENCY = Histogram(
    "store_daemon_model_verification_latency_seconds",
    "Latency of model integrity verification",
)

# Lifecycle management metrics
MODEL_REF_COUNT = Gauge(
    "store_daemon_model_ref_count",
    "Current reference count for loaded models",
    ["model", "device_id"],
)

GPU_CACHE_BYTES = Gauge(
    "store_daemon_gpu_cache_bytes",
    "GPU memory used for model caching",
    ["type"],  # local, global
)

EVICTIONS_TOTAL = Counter(
    "store_daemon_evictions_total",
    "Total number of model evictions",
    ["reason"],  # memory, shutdown
)

PROC_LEAK_DETECTED_TOTAL = Counter(
    "store_daemon_proc_leak_detected_total",
    "Total number of process reference leaks detected",
)

# High Availability metrics
HA_CONNECTION_STATE = Gauge(
    "store_daemon_ha_connection_state",
    "Current HA connection state (0=disconnected, 1=connecting, 2=connected, 3=reconnecting, 4=failed)",
)

HA_STATE_VERSION = Gauge(
    "store_daemon_ha_state_version",
    "Current local state version for HA synchronization",
)

HA_REGISTERED_MODELS = Gauge(
    "store_daemon_ha_registered_models",
    "Number of models registered with Global Store",
)

HA_HEARTBEAT_TOTAL = Counter(
    "store_daemon_ha_heartbeat_total",
    "Total number of heartbeats sent to Global Store",
    ["status"],  # success, failure
)

HA_STATE_SYNC_TOTAL = Counter(
    "store_daemon_ha_state_sync_total",
    "Total number of state synchronizations performed",
    ["type", "status"],  # _type: full, incremental; status: success, failure
)

HA_STATE_CHANGES_TOTAL = Counter(
    "store_daemon_ha_state_changes_total",
    "Total number of state changes applied from Global Store",
    ["change_type"],  # ADD_REPLICA, UPDATE_REPLICA, REMOVE_REPLICA
)

HA_CONNECTION_RETRIES_TOTAL = Counter(
    "store_daemon_ha_connection_retries_total",
    "Total number of connection retry attempts to Global Store",
)

HA_THREAD_RESTARTS_TOTAL = Counter(
    "store_daemon_ha_thread_restarts_total",
    "Total number of HA thread restarts due to failures",
    ["thread_name"],  # connection, heartbeat, sync
)

HA_PENDING_CHANGES = Gauge(
    "store_daemon_ha_pending_changes",
    "Number of pending state changes in queue",
    ["type"],  # registrations, unregistrations
)


def get_device_type_label(device_type) -> str:
    """Get device type label for metrics.

    Args:
        device_type: Device type from proto definition

    Returns:
        String label for the device type
    """
    # Import here to avoid circular imports
    from scstore.proto import store_daemon_pb2

    if device_type == store_daemon_pb2.DEVICE_TYPE_CPU:
        return "cpu"
    elif device_type == store_daemon_pb2.DEVICE_TYPE_GPU:
        return "gpu"
    else:
        return "unknown"
