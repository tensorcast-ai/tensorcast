#  Copyright (c) 2025, StepCast Team.

"""Prometheus metrics definitions for StoreDaemon."""

from prometheus_client import Counter, Gauge, Histogram, Info

# Artifact operation metrics
ARTIFACTS_MATERIALIZED_TOTAL = Counter(
    "store_daemon_artifacts_materialized_total",
    "Total number of artifacts materialized",
    ["device_type", "source_type"],  # source_type: remote, disk
)

ARTIFACTS_MATERIALIZE_FAILURES_TOTAL = Counter(
    "store_daemon_artifacts_materialize_failures_total",
    "Total number of artifact materialization failures",
    ["device_type", "error_type"],
)

ARTIFACT_MATERIALIZE_DURATION = Histogram(
    "store_daemon_artifact_materialize_duration_seconds",
    "Time spent materializing artifacts",
    ["device_type", "source_type"],
)

ARTIFACTS_UNLOADED_TOTAL = Counter(
    "store_daemon_artifacts_unloaded_total",
    "Total number of artifacts unloaded",
    ["device_type"],
)

# Async materialization metrics
ARTIFACTS_ALLOCATED_TOTAL = Counter(
    "store_daemon_artifacts_allocated_total",
    "Total number of artifacts with memory allocated (async)",
    ["device_type"],
)

PENDING_MATERIALIZATIONS = Gauge(
    "store_daemon_pending_materializations",
    "Number of artifacts currently being materialized asynchronously",
)

MATERIALIZE_WAIT_DURATION = Histogram(
    "store_daemon_materialize_wait_duration_seconds",
    "Time spent waiting for async artifact materialization in ConfirmReplica",
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
ARTIFACT_VERIFICATION_TOTAL = Counter(
    "store_daemon_artifact_verification_total",
    "Total number of artifact integrity verification results",
    ["status"],  # passed, failed
)

ARTIFACT_VERIFICATION_LATENCY = Histogram(
    "store_daemon_artifact_verification_latency_seconds",
    "Latency of artifact integrity verification",
)

# Lifecycle management metrics
ARTIFACT_REF_COUNT = Gauge(
    "store_daemon_artifact_ref_count",
    "Current reference count for loaded artifacts",
    ["artifact", "device_id"],
)

GPU_CACHE_BYTES = Gauge(
    "store_daemon_gpu_cache_bytes",
    "GPU memory used for replica caching",
    ["type"],  # local, global
)

EVICTIONS_TOTAL = Counter(
    "store_daemon_evictions_total",
    "Total number of replica evictions",
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

HA_REGISTERED_ARTIFACTS = Gauge(
    "store_daemon_ha_registered_artifacts",
    "Number of artifacts registered with Global Store",
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
