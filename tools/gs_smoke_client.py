#  Copyright (c) 2025-2026, TensorCast Team.

"""Global Store gRPC one-shot client with OpenTelemetry.

Sends a HealthCheck (and optionally ListActiveWorkers) request to a running
Global Store server to generate real RPC spans via gRPC instrumentation.

Usage:
  # Console exporter
  TC_OTEL_CONSOLE_EXPORTER=1 \
  OTEL_SERVICE_NAME=tensorcast-gs-smoke \
  uv run tools/gs_smoke_client.py --host 127.0.0.1 --port 50051 --list-workers

  # With OTLP collector
  OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 \
  OTEL_SERVICE_NAME=tensorcast-gs-smoke \
  uv run tools/gs_smoke_client.py --host 127.0.0.1 --port 50051
"""

from __future__ import annotations

import argparse
import contextlib

import grpc
from opentelemetry import trace

from tensorcast.global_store.composite_stub import GlobalStoreCompositeStub
from tensorcast.observability.otel import setup_otel
from tensorcast.proto.global_store.v1 import (
    global_store_pb2,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Global Store OTel smoke client")
    parser.add_argument("--host", default="127.0.0.1", help="Global Store host")
    parser.add_argument("--port", type=int, default=50051, help="Global Store port")
    parser.add_argument(
        "--list-workers",
        action="store_true",
        help="Also call ListActiveWorkers to exercise another RPC",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Per-RPC timeout seconds (default: 10.0)",
    )
    parser.add_argument(
        "--service-name",
        default="tensorcast-gs-smoke",
        help="OTEL service.name if env var not set",
    )
    args = parser.parse_args()

    setup_otel(args.service_name, role="smoke-client")

    address = f"{args.host}:{args.port}"
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("Smoke/GSClient"):
        channel = grpc.insecure_channel(address)
        try:
            stub = GlobalStoreCompositeStub(channel)
            # HealthCheck
            try:
                resp = stub.HealthCheck(
                    global_store_pb2.HealthCheckRequest(),
                    timeout=min(2.0, args.timeout),
                )
                print("HealthCheck status:", resp.status)
            except grpc.RpcError as e:  # noqa: BLE001
                print("HealthCheck RPC error:", e)

            if args.list_workers:
                try:
                    lr = stub.ListActiveWorkers(
                        global_store_pb2.ListActiveWorkersRequest(
                            include_unavailable=True
                        ),
                        timeout=args.timeout,
                    )
                    print("ListActiveWorkers count:", len(lr.workers))
                except grpc.RpcError as e:  # noqa: BLE001
                    print("ListActiveWorkers RPC error:", e)
        finally:
            channel.close()

    # Ensure spans are exported before process exit
    provider = trace.get_tracer_provider()
    for meth in ("force_flush", "shutdown"):
        fn = getattr(provider, meth, None)
        if callable(fn):
            with contextlib.suppress(Exception):
                fn()


if __name__ == "__main__":
    main()
