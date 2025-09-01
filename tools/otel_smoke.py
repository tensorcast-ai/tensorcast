#  Copyright (c) 2025, TensorCast Team.

"""OTel smoke test for TensorCast Python.

Emits a manual span and performs a dummy gRPC client call to exercise
instrumentation and the exporter pipeline. Uses standard OTel env vars.

Examples:
  TC_OTEL_CONSOLE_EXPORTER=1 \
  OTEL_SERVICE_NAME=tensorcast-smoke \
  uv run tools/otel_smoke.py --grpc-target 127.0.0.1:65534

  # With OTLP collector
  OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 \
  OTEL_SERVICE_NAME=tensorcast-smoke \
  uv run tools/otel_smoke.py
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time

import grpc
from opentelemetry import trace

from tensorcast.observability.otel import setup_otel


def _manual_span() -> None:
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("Smoke/ManualSpan") as span:
        span.set_attribute("tc.smoke.attr", "manual")
        span.set_attribute("tc.smoke.pid", os.getpid())
        time.sleep(0.05)


def _grpc_sync_call(target: str) -> None:
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("Smoke/GrpcSyncCall"):
        # Dynamic unary call – will fail with UNAVAILABLE if no server, but still traced
        channel = grpc.insecure_channel(target)
        try:
            stub = channel.unary_unary("/smoke.Test/Unary")
            try:
                _ = stub(b"payload", timeout=0.5)
            except grpc.RpcError:
                pass
        finally:
            channel.close()


async def _grpc_aio_call(target: str) -> None:
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("Smoke/GrpcAioCall"):
        channel = grpc.aio.insecure_channel(target)
        try:
            stub = channel.unary_unary("/smoke.Test/Unary")
            try:
                _ = await stub(b"payload", timeout=0.5)
            except grpc.RpcError:
                pass
        finally:
            await channel.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="TensorCast OpenTelemetry smoke test")
    parser.add_argument(
        "--grpc-target",
        default="127.0.0.1:65534",
        help="gRPC address to attempt a client call against (default: unused port)",
    )
    parser.add_argument(
        "--service-name",
        default="tensorcast-smoke",
        help="OTEL service.name when OTEL_SERVICE_NAME is not set",
    )
    parser.add_argument(
        "--role",
        default="dev",
        help="Resource attribute tc.node.role (default: dev)",
    )
    parser.add_argument(
        "--use-aio",
        action="store_true",
        help="Exercise asyncio gRPC client path as well",
    )
    args = parser.parse_args()

    setup_otel(args.service_name, role=args.role)

    _manual_span()
    _grpc_sync_call(args.grpc_target)
    if args.use_aio:
        asyncio.run(_grpc_aio_call(args.grpc_target))

    # Flush and shutdown provider to ensure export
    provider = trace.get_tracer_provider()
    try:
        provider.force_flush()
    except Exception:
        pass
    try:
        provider.shutdown()
    except Exception:
        pass

    print("OTel smoke test completed")


if __name__ == "__main__":
    main()

