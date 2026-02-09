#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import logging
from collections.abc import Sequence
from contextlib import contextmanager
from time import monotonic
from typing import Callable, Iterator

import grpc
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from tensorcast.proto.global_store.v1 import global_store_pb2

"""Prometheus metrics for the Global Store service.

This module centralises the Prometheus metric definitions and exposes a
`PrometheusInterceptor` that can be attached to the gRPC server so that
all RPC calls are automatically tracked without having to instrument each
handler manually.

The interceptor records two key metrics per-RPC:
1. `grpc_server_handled_total` – total number of requests grouped by method
   name and final status code.
2. `grpc_server_handling_seconds` – request latency histogram grouped by
   method name.

It also exposes Gauges for high-level cluster state that are updated via
convenience helpers which can be called from anywhere in the code-base.
"""

# ---------------------------------------------------------------------------
# Metric definitions – keep the cardinality low and bounded!
# ---------------------------------------------------------------------------

GRPC_SERVER_HANDLED_COUNTER = Counter(
    "tc_grpc_server_handled_total",
    "Total number of gRPC requests processed on the server side.",
    labelnames=("method", "code"),
)

GRPC_SERVER_HANDLING_LATENCY_SECONDS = Histogram(
    "tc_grpc_server_handling_seconds",
    "Latency in seconds per gRPC method handled by the server.",
    labelnames=("method",),
    buckets=(
        0.0005,
        0.001,
        0.0025,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        float("inf"),
    ),
)

ACTIVE_WORKERS_GAUGE = Gauge(
    "tc_active_workers",
    "Current number of active workers as perceived by the Global Store.",
)

ARTIFACT_REPLICAS_GAUGE = Gauge(
    "tc_replicas_total",
    "Total number of registered artifact replicas across all workers.",
)

# ---------------------------------------------------------------------------
# Extended, fine-grained metrics (per-artifact, per-replica, transport, recovery)
# ---------------------------------------------------------------------------

# Replica lifecycle ---------------------------------------------------------

REPLICA_REGISTER_COUNTER = Counter(
    "tc_replica_register_total",
    "Total number of artifact replicas registered (create or update).",
    labelnames=("artifact_id", "memory_type"),
)

REPLICA_UNREGISTER_COUNTER = Counter(
    "tc_replica_unregister_total",
    "Total number of artifact replicas unregistered/deleted.",
    labelnames=("artifact_id", "memory_type"),
)

REPLICA_PER_ARTIFACT_GAUGE = Gauge(
    "tc_replicas_per_artifact",
    "Current number of replicas per artifact (content-addressed artifact_id).",
    labelnames=("artifact_id",),
)

# Per-memory-type gauge – GPU/RAM/DISK spread.
REPLICA_PER_MEMTYPE_GAUGE = Gauge(
    "tc_replicas_per_memtype",
    "Current number of replicas per memory type.",
    labelnames=("memory_type",),
)

# Transport lifecycle -------------------------------------------------------

TRANSPORT_REQUEST_COUNTER = Counter(
    "tc_transport_requests_total",
    "Total number of transport requests processed by the Global Store.",
    labelnames=("artifact_id", "status"),  # status=success|timeout|error
)

TRANSPORT_WAIT_SECONDS = Histogram(
    "tc_transport_wait_seconds",
    "Time spent waiting for an available replica during transport request.",
    labelnames=("artifact_id",),
    buckets=(
        0.0,
        0.05,
        0.1,
        0.25,
        0.5,
        1,
        2,
        5,
        10,
        float("inf"),
    ),
)

ACTIVE_TRANSPORTS_GAUGE = Gauge(
    "tc_active_transports",
    "Current number of in-flight (not yet completed) artifact transports.",
)

TRANSPORT_FILTER_COUNTER = Counter(
    "tc_transport_filter_total",
    "Total number of replicas filtered during transport selection.",
    labelnames=("artifact_id", "reason"),
)

TRANSPORT_NO_EXPORTABLE_COUNTER = Counter(
    "tc_transport_no_exportable_total",
    "Total number of transport requests rejected due to no exportable sources.",
    labelnames=("artifact_id",),
)

# Capability directory -------------------------------------------------------

CAPABILITY_DIRECTORY_GAUGE = Gauge(
    "tc_capability_directory_entries",
    "Active capability directory entries by scope and capability.",
    labelnames=("scope", "capability"),
)


def _normalize_capability_name(raw: str) -> str:
    for prefix in (
        "WORKER_CAPABILITY_FLAG_",
        "INSTANCE_CAPABILITY_FLAG_",
    ):
        if raw.startswith(prefix):
            return raw[len(prefix) :].lower()
    return raw.lower()


_WORKER_CAPABILITIES: list[tuple[int, str]] = [
    (value, _normalize_capability_name(name))
    for name, value in global_store_pb2.WorkerCapabilityFlag.items()
    if value != 0
]

_INSTANCE_CAPABILITIES: list[tuple[int, str]] = [
    (value, _normalize_capability_name(name))
    for name, value in global_store_pb2.InstanceCapabilityFlag.items()
    if value != 0
]

# View registration ----------------------------------------------------------

VIEW_REGISTRATION_COUNTER = Counter(
    "tc_view_registration_total",
    "Total number of view registrations recorded by the Global Store.",
    labelnames=("result",),  # result=complete|partial
)

VIEW_PARTIAL_BACKLOG_GAUGE = Gauge(
    "tc_view_partial_backlog_bytes",
    "Canonical byte backlog for partial view registrations.",
    labelnames=("artifact_id", "view_id"),
)

# Digest grids (leaves + overlap proofs) -------------------------------------

DIGEST_ENTRIES_WRITTEN_COUNTER = Counter(
    "tc_digest_entries_written_total",
    "Total number of digest entries newly written (inserted) by the Global Store.",
    labelnames=(
        "grid",
    ),  # grid=leaves|assembly_proof_commitments|piece_proof_digests|tensor_proof_commitments
)

DIGEST_CONFLICT_COUNTER = Counter(
    "tc_digest_conflicts_total",
    "Total number of digest conflicts (mismatched writes) rejected by the Global Store.",
    labelnames=("grid",),
)

DIGEST_REQUEST_REJECTED_COUNTER = Counter(
    "tc_digest_requests_rejected_total",
    "Total number of digest write requests rejected by the Global Store.",
    labelnames=("reason",),  # reason=too_large|conflict|invalid
)

# Retention / GC -------------------------------------------------------------

GC_ROWS_DELETED_COUNTER = Counter(
    "tc_gc_rows_deleted_total",
    "Total number of rows deleted by Global Store retention policies.",
    labelnames=(
        "table",
    ),  # table=operations|assembly_proof_commitments|piece_proof_digests
)

# Recovery / state-sync ------------------------------------------------------

STATE_SYNC_COUNTER = Counter(
    "tc_state_sync_total",
    "Total number of worker state-synchronisation operations.",
    labelnames=("result",),  # result=success|error
)

STATE_SYNC_DURATION_SECONDS = Histogram(
    "tc_state_sync_seconds",
    "Duration of worker state-synchronisation operations in seconds.",
    buckets=(
        0.01,
        0.05,
        0.1,
        0.25,
        0.5,
        1,
        2,
        5,
        10,
        float("inf"),
    ),
)

# Memory tier telemetry ------------------------------------------------------

MEMORY_TIER_STABLE_BYTES = Gauge(
    "tensorcast_memory_tier_stable_bytes",
    "Stable memory bytes per node (state=total|used).",
    labelnames=("node", "state"),
)

MEMORY_TIER_PREEMPTIBLE_BYTES = Gauge(
    "tensorcast_memory_tier_preemptible_bytes",
    "Preemptible memory bytes per node (state=total|marked).",
    labelnames=("node", "state"),
)

MEMORY_TIER_FAULTS_PER_SEC = Gauge(
    "tensorcast_memory_tier_faults_per_sec",
    "Faults per second observed for UMA preemptible memory.",
    labelnames=("node",),
)

MEMORY_TIER_REHYDRATE_P99_NS = Gauge(
    "tensorcast_memory_tier_rehydrate_p99_ns",
    "p99 rehydration latency (ns) reported by UMA for preemptible chunks.",
    labelnames=("node",),
)

MEMORY_TIER_ENABLE_PREEMPTIBLE = Gauge(
    "tensorcast_memory_tier_enable_preemptible",
    "Info gauge indicating whether preemptible memory is enabled on the node.",
    labelnames=("node", "enable_preemptible"),
)

# ---------------------------------------------------------------------------
# Helper functions to update high-level gauges from the business logic layer.
# ---------------------------------------------------------------------------


def set_active_workers(count: int) -> None:
    """Set the ACTIVE_WORKERS_GAUGE to *count*."""

    ACTIVE_WORKERS_GAUGE.set(count)


def set_total_replicas(count: int) -> None:
    """Set the total replicas gauge to *count*."""

    ARTIFACT_REPLICAS_GAUGE.set(count)


def set_capability_counts(*, scope: str, entries: Sequence[object]) -> None:
    """Set capability directory gauges for the given entries."""

    if scope == "worker":
        capabilities = _WORKER_CAPABILITIES
    elif scope == "instance":
        capabilities = _INSTANCE_CAPABILITIES
    else:
        return

    counts = {name: 0 for _, name in capabilities}
    for entry in entries:
        flags = int(getattr(entry, "capability_flags", 0))
        for bit, name in capabilities:
            if flags & (1 << int(bit)):
                counts[name] += 1

    for name, count in counts.items():
        CAPABILITY_DIRECTORY_GAUGE.labels(scope=scope, capability=name).set(count)


def observe_memory_tier_snapshot(
    node_id: str,
    stable_total_bytes: int,
    stable_used_bytes: int,
    preemptible_total_bytes: int,
    preemptible_marked_bytes: int,
    faults_per_sec: float,
    rehydrate_p99_ns: int,
    enable_preemptible: bool,
) -> None:
    """Record the latest memory tier telemetry for a node."""

    MEMORY_TIER_STABLE_BYTES.labels(node=node_id, state="total").set(stable_total_bytes)
    MEMORY_TIER_STABLE_BYTES.labels(node=node_id, state="used").set(stable_used_bytes)
    MEMORY_TIER_PREEMPTIBLE_BYTES.labels(node=node_id, state="total").set(
        preemptible_total_bytes
    )
    MEMORY_TIER_PREEMPTIBLE_BYTES.labels(node=node_id, state="marked").set(
        preemptible_marked_bytes
    )
    MEMORY_TIER_FAULTS_PER_SEC.labels(node=node_id).set(faults_per_sec)
    MEMORY_TIER_REHYDRATE_P99_NS.labels(node=node_id).set(rehydrate_p99_ns)
    MEMORY_TIER_ENABLE_PREEMPTIBLE.labels(node=node_id, enable_preemptible="true").set(
        1.0 if enable_preemptible else 0.0
    )
    MEMORY_TIER_ENABLE_PREEMPTIBLE.labels(node=node_id, enable_preemptible="false").set(
        0.0 if enable_preemptible else 1.0
    )


# ---------------------------------------------------------------------------
# Replica helpers
# ---------------------------------------------------------------------------


def inc_replica_register(artifact_id: str, memory_type: str) -> None:
    """Increment the replica register counter and update per-artifact gauges."""

    REPLICA_REGISTER_COUNTER.labels(
        artifact_id=artifact_id, memory_type=memory_type
    ).inc()


def inc_replica_unregister(artifact_id: str, memory_type: str) -> None:
    """Increment the replica unregister counter."""

    REPLICA_UNREGISTER_COUNTER.labels(
        artifact_id=artifact_id, memory_type=memory_type
    ).inc()


def set_replicas_per_artifact(artifact_id: str, count: int) -> None:
    """Set per-artifact replica gauge."""

    REPLICA_PER_ARTIFACT_GAUGE.labels(artifact_id=artifact_id).set(count)


def set_replicas_per_memtype(memory_type: str, count: int) -> None:
    """Set per-memory-type replica gauge."""

    REPLICA_PER_MEMTYPE_GAUGE.labels(memory_type=memory_type).set(count)


def inc_view_registration(result: str) -> None:
    """Increment the view registration counter."""

    VIEW_REGISTRATION_COUNTER.labels(result=result).inc()


def set_view_partial_backlog(
    artifact_id: str, view_id: str, backlog_bytes: int
) -> None:
    """Track outstanding canonical bytes for a view registration."""

    VIEW_PARTIAL_BACKLOG_GAUGE.labels(artifact_id=artifact_id, view_id=view_id).set(
        backlog_bytes
    )


def inc_digest_entries_written(*, grid: str, count: int) -> None:
    if count <= 0:
        return
    DIGEST_ENTRIES_WRITTEN_COUNTER.labels(grid=grid).inc(count)


def inc_digest_conflict(*, grid: str, count: int = 1) -> None:
    if count <= 0:
        return
    DIGEST_CONFLICT_COUNTER.labels(grid=grid).inc(count)


def inc_digest_request_rejected(*, reason: str, count: int = 1) -> None:
    if count <= 0:
        return
    DIGEST_REQUEST_REJECTED_COUNTER.labels(reason=reason).inc(count)


def inc_gc_rows_deleted(*, table: str, count: int) -> None:
    if count <= 0:
        return
    GC_ROWS_DELETED_COUNTER.labels(table=table).inc(count)


# ---------------------------------------------------------------------------
# Transport helpers
# ---------------------------------------------------------------------------


def inc_transport_request(artifact_id: str, status: str) -> None:
    """Increment transport request counter with *status*."""

    TRANSPORT_REQUEST_COUNTER.labels(artifact_id=artifact_id, status=status).inc()


def observe_transport_wait(artifact_id: str, wait_seconds: float) -> None:
    """Record *wait_seconds* into histogram."""

    TRANSPORT_WAIT_SECONDS.labels(artifact_id=artifact_id).observe(wait_seconds)


def inc_active_transports() -> None:
    ACTIVE_TRANSPORTS_GAUGE.inc()


def dec_active_transports() -> None:
    # Ensure we do not go below zero.
    if ACTIVE_TRANSPORTS_GAUGE._value.get() > 0:
        ACTIVE_TRANSPORTS_GAUGE.dec()


def inc_transport_filter(artifact_id: str, reason: str) -> None:
    """Increment transport filter counter with *reason*."""

    TRANSPORT_FILTER_COUNTER.labels(artifact_id=artifact_id, reason=reason).inc()


def inc_transport_no_exportable(artifact_id: str) -> None:
    """Increment no-exportable transport counter."""

    TRANSPORT_NO_EXPORTABLE_COUNTER.labels(artifact_id=artifact_id).inc()


# ---------------------------------------------------------------------------
# Recovery helpers
# ---------------------------------------------------------------------------


def observe_state_sync(duration_seconds: float, success: bool) -> None:
    """Record a state-sync operation duration and result."""

    result_label = "success" if success else "error"
    STATE_SYNC_COUNTER.labels(result=result_label).inc()
    STATE_SYNC_DURATION_SECONDS.observe(duration_seconds)


# ---------------------------------------------------------------------------
# gRPC Server-side interceptor for automatic RPC instrumentation.
# ---------------------------------------------------------------------------


class PrometheusInterceptor(grpc.ServerInterceptor):
    """A minimalist gRPC interceptor that records per-RPC metrics."""

    def intercept_service(
        self,
        continuation: Callable[[grpc.HandlerCallDetails], grpc.RpcMethodHandler | None],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler | None:
        method = handler_call_details.method
        handler = continuation(handler_call_details)

        if handler is None:
            return handler  # pragma: no cover – pass through to default behaviour

        # We need to wrap the handler depending on its RPC type.
        if handler.unary_unary:
            # Use the helper to create a compliant unary-unary handler instead of
            # instantiating RpcMethodHandler directly (namedtuple construction
            # with keyword arguments is not supported and triggers a run-time
            # TypeError inside the gRPC server, which then surfaces as
            # "Error in service handler!" on the client side).
            return grpc.unary_unary_rpc_method_handler(
                self._wrap_unary_unary(method, handler.unary_unary),
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        # For brevity we instrument only unary-unary RPCs for now.
        # The Global Store currently exposes only unary-unary methods.
        return handler

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _wrap_unary_unary(
        self,
        method: str,
        inner: Callable[[object, grpc.ServicerContext], object],
    ) -> Callable[[object, grpc.ServicerContext], object]:
        """Wrap a unary-unary RPC handler to collect metrics."""

        def wrapper(request, context):
            with _record_rpc_metrics(method):
                response = inner(request, context)
                code = context.code() or grpc.StatusCode.OK
                GRPC_SERVER_HANDLED_COUNTER.labels(method=method, code=code.name).inc()
                return response

        return wrapper


@contextmanager
def _record_rpc_metrics(method: str) -> Iterator[None]:
    """Context manager that observes latency histogram for *method*."""

    start_time = monotonic()
    try:
        yield
    finally:
        duration = monotonic() - start_time
        GRPC_SERVER_HANDLING_LATENCY_SECONDS.labels(method=method).observe(duration)


# ---------------------------------------------------------------------------
# HTTP server bootstrap helper
# ---------------------------------------------------------------------------


def start_metrics_http_server(port: int, addr: str = "") -> int:
    """Idempotently start the Prometheus HTTP server on *port*.

    Calling this multiple times is safe; `start_http_server` is a no-op after
    the first successful invocation in the current process.
    """

    try:
        server = start_http_server(port, addr=addr)
        actual_port = getattr(server, "server_port", None) or port
        return int(actual_port)
    except OSError as exc:
        logging.getLogger(__name__).warning(
            "Could not start Prometheus metrics HTTP server on port %s: %s",
            port,
            exc,
        )
        return port
