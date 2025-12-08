#  Copyright (c) 2025, TensorCast Team.

"""Prometheus metrics for the dashboard backend."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

REQUEST_COUNTER = Counter(
    "tensorcast_dashboard_http_requests_total",
    "Total HTTP requests handled by the dashboard backend.",
    labelnames=("endpoint", "method", "status"),
)

REQUEST_LATENCY = Histogram(
    "tensorcast_dashboard_http_request_duration_seconds",
    "HTTP request latency for the dashboard backend.",
    labelnames=("endpoint", "method"),
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
)

GRPC_COUNTER = Counter(
    "tensorcast_dashboard_grpc_calls_total",
    "Total Global Store gRPC calls issued by the dashboard backend.",
    labelnames=("rpc", "status"),
)

GRPC_LATENCY = Histogram(
    "tensorcast_dashboard_grpc_call_duration_seconds",
    "Latency for Global Store gRPC calls made by the dashboard backend.",
    labelnames=("rpc",),
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
)
