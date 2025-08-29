#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

from contextlib import contextmanager
from time import monotonic
from typing import Callable, Iterator

import grpc
from prometheus_client import Counter, Gauge, Histogram, start_http_server

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
    "grpc_server_handled_total",
    "Total number of gRPC requests processed on the server side.",
    labelnames=("method", "code"),
)

GRPC_SERVER_HANDLING_LATENCY_SECONDS = Histogram(
    "grpc_server_handling_seconds",
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
    "global_store_active_workers",
    "Current number of active workers as perceived by the Global Store.",
)

ARTIFACT_REPLICAS_GAUGE = Gauge(
    "global_store_replicas_total",
    "Total number of registered artifact replicas across all workers.",
)

# ---------------------------------------------------------------------------
# Extended, fine-grained metrics (per-artifact, per-replica, transport, recovery)
# ---------------------------------------------------------------------------

# Replica lifecycle ---------------------------------------------------------

REPLICA_REGISTER_COUNTER = Counter(
    "global_store_replica_register_total",
    "Total number of artifact replicas registered (create or update).",
    labelnames=("artifact_id", "memory_type"),
)

REPLICA_UNREGISTER_COUNTER = Counter(
    "global_store_replica_unregister_total",
    "Total number of artifact replicas unregistered/deleted.",
    labelnames=("artifact_id", "memory_type"),
)

# Per-artifact replica gauge – allows tracking of hot artifacts.
REPLICA_PER_ARTIFACT_GAUGE = Gauge(
    "global_store_replicas_per_artifact",
    "Current number of replicas per artifact (content-addressed artifact_id).",
    labelnames=("artifact_id",),
)

# Per-memory-type gauge – GPU/RAM/DISK spread.
REPLICA_PER_MEMTYPE_GAUGE = Gauge(
    "global_store_replicas_per_memtype",
    "Current number of replicas per memory type.",
    labelnames=("memory_type",),
)

# Transport lifecycle -------------------------------------------------------

TRANSPORT_REQUEST_COUNTER = Counter(
    "global_store_transport_requests_total",
    "Total number of transport requests processed by the Global Store.",
    labelnames=("artifact_id", "status"),  # status=success|timeout|error
)

TRANSPORT_WAIT_SECONDS = Histogram(
    "global_store_transport_wait_seconds",
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
    "global_store_active_transports",
    "Current number of in-flight (not yet completed) artifact transports.",
)

# Recovery / state-sync ------------------------------------------------------

STATE_SYNC_COUNTER = Counter(
    "global_store_state_sync_operations_total",
    "Total number of worker state-synchronisation operations.",
    labelnames=("result",),  # result=success|error
)

STATE_SYNC_DURATION_SECONDS = Histogram(
    "global_store_state_sync_duration_seconds",
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

# ---------------------------------------------------------------------------
# Helper functions to update high-level gauges from the business logic layer.
# ---------------------------------------------------------------------------


def set_active_workers(count: int) -> None:
    """Set the ACTIVE_WORKERS_GAUGE to *count*."""

    ACTIVE_WORKERS_GAUGE.set(count)


def set_total_replicas(count: int) -> None:
    """Set the total replicas gauge to *count*."""

    ARTIFACT_REPLICAS_GAUGE.set(count)


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


def start_metrics_http_server(port: int) -> None:
    """Idempotently start the Prometheus HTTP server on *port*.

    Calling this multiple times is safe; `start_http_server` is a no-op after
    the first successful invocation in the current process.
    """

    try:
        start_http_server(port)
    except OSError as exc:
        # The most common failure is the port already in use when the server is
        # restarted quickly.  We log and continue because metrics are optional.
        import logging

        logging.getLogger(__name__).warning(
            "Could not start Prometheus metrics HTTP server on port %s: %s",
            port,
            exc,
        )
