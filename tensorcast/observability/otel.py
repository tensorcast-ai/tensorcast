#  Copyright (c) 2025, TensorCast Team.

"""OpenTelemetry initialization for TensorCast (Python Global Store and clients).

Strict mode: OpenTelemetry is required. We configure a proper SDK provider,
OTLP gRPC exporter, and gRPC instrumentation. Fail fast on errors to avoid
silent loss of observability.

Environment knobs (standard OTel):
- `OTEL_SERVICE_NAME`: logical service name; default provided by caller
- `OTEL_EXPORTER_OTLP_ENDPOINT` / `*_TRACES_ENDPOINT` / `*_PROTOCOL`
- `OTEL_TRACES_SAMPLER` / `OTEL_TRACES_SAMPLER_ARG`
"""

from __future__ import annotations

import os
import threading
from importlib import metadata as importlib_metadata
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as GrpcExporter,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HttpExporter,
)
from opentelemetry.instrumentation.grpc import (
    GrpcInstrumentorClient,
    GrpcInstrumentorServer,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.sampling import (
    ALWAYS_OFF,
    ALWAYS_ON,
    ParentBased,
    ParentBasedTraceIdRatio,
    TraceIdRatioBased,
)

from tensorcast.client_runtime import get_client_config
from tensorcast.proto.config.v1 import common_pb2 as commonpb

_INIT_LOCK = threading.Lock()
_OTEL_INITIALIZED = False
_GRPC_INSTRUMENTED = False
_GRPC_AIO_INSTRUMENTED = False


def _instrument_grpc() -> None:
    """Instrument both sync and asyncio gRPC (idempotent).

    Assumes instrumentation packages are present (we depend on them).
    Raises RuntimeError on failure.
    """
    global _GRPC_INSTRUMENTED, _GRPC_AIO_INSTRUMENTED
    # Sync gRPC
    if not _GRPC_INSTRUMENTED:
        try:
            GrpcInstrumentorServer().instrument()
            GrpcInstrumentorClient().instrument()
            _GRPC_INSTRUMENTED = True
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to instrument gRPC: {exc}") from exc

    # AsyncIO gRPC – available in instrumentation >=0.57b0
    if not _GRPC_AIO_INSTRUMENTED:
        try:
            from opentelemetry.instrumentation.grpc import (
                GrpcAioInstrumentorClient,
                GrpcAioInstrumentorServer,
            )

            GrpcAioInstrumentorServer().instrument()
            GrpcAioInstrumentorClient().instrument()
            _GRPC_AIO_INSTRUMENTED = True
        except Exception as exc:  # noqa: BLE001
            # Treat missing aio or instrument failure as an error to surface
            # misconfiguration early in development.
            raise RuntimeError(f"Failed to instrument gRPC aio: {exc}") from exc


def _sampler_from_env():
    """Build a sampler based on OTEL_TRACES_SAMPLER and _ARG.

    Defaults to ParentBased(ALWAYS_ON) per spec when unset or invalid.
    """
    sampler_name = (
        os.getenv("OTEL_TRACES_SAMPLER", "parentbased_always_on").strip().lower()
    )
    arg = os.getenv("OTEL_TRACES_SAMPLER_ARG", "1.0")
    try:
        rate = float(arg)
    except Exception:  # noqa: BLE001
        rate = 1.0

    if sampler_name == "always_on":
        return ALWAYS_ON
    if sampler_name == "always_off":
        return ALWAYS_OFF
    if sampler_name == "traceidratio":
        return TraceIdRatioBased(rate)
    if sampler_name == "parentbased_always_on":
        return ParentBased(ALWAYS_ON)
    if sampler_name == "parentbased_always_off":
        return ParentBased(ALWAYS_OFF)
    if sampler_name == "parentbased_traceidratio":
        return ParentBasedTraceIdRatio(rate)

    # Fallback
    return ParentBased(ALWAYS_ON)


def setup_otel(service_default: str, role: str) -> bool:
    """Initialize OpenTelemetry tracing and gRPC instrumentation (required).

    This function is idempotent. It requires OTel packages to be installed and
    an OTLP exporter to be constructible. Any failure raises an exception to
    prevent silent loss of observability.

    Args:
        service_default: Default value for `service.name` when OTEL_SERVICE_NAME
            is not set.
        role: Value for resource attribute `tc.node.role` (e.g., "global-store").

    Returns:
        True if OpenTelemetry tracing was successfully initialized.

    Raises:
        ImportError: If required OTel packages are not installed.
        RuntimeError: If exporter or instrumentation cannot be initialized.
    """
    # Sampler built from env via helper (Python SDK doesn't expose a helper)

    # Idempotent installation and instrumentation
    global _OTEL_INITIALIZED, _GRPC_INSTRUMENTED, _GRPC_AIO_INSTRUMENTED
    with _INIT_LOCK:
        if not _OTEL_INITIALIZED:
            # Build resource with service name and custom attributes.
            service_name = os.getenv("OTEL_SERVICE_NAME", service_default)
            # Best-effort package version lookup.
            try:
                service_version = importlib_metadata.version("tensorcast")
            except Exception:
                service_version = "unknown"

            resource = Resource.create(
                {
                    "service.name": service_name,
                    "service.namespace": "tensorcast",
                    "service.version": service_version,
                    "tc.node.role": role,
                }
            )

            # Respect env-driven sampler
            sampler = _sampler_from_env()
            provider = TracerProvider(resource=resource, sampler=sampler)
            # Exporter configuration primarily driven by environment variables.
            # Creation must succeed; otherwise we fail fast.
            try:
                exporter = OTLPSpanExporter()
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"Failed to create OTLPSpanExporter: {exc}") from exc
            provider.add_span_processor(BatchSpanProcessor(exporter))
            # Optional console exporter for local debugging (prints spans)
            if os.getenv("TC_OTEL_CONSOLE_EXPORTER", "0").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }:
                from opentelemetry.sdk.trace.export import ConsoleSpanExporter

                provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

            trace.set_tracer_provider(provider)
            _OTEL_INITIALIZED = True

        # Instrument both sync and asyncio gRPC
        _instrument_grpc()

    return _OTEL_INITIALIZED


# New: setup OTel directly from Observability proto (no environment required)
def setup_otel_from_observability(obs: Any, role: str) -> bool:
    # Idempotent
    global _OTEL_INITIALIZED
    with _INIT_LOCK:
        if _OTEL_INITIALIZED:
            return True

        # Respect explicit disable: do nothing when not enabled
        if not getattr(obs.otel, "enabled", False):
            return False

        # Resource
        service_name = obs.otel.service_name or "tensorcast"
        try:
            service_version = importlib_metadata.version("tensorcast")
        except Exception:
            service_version = "unknown"
        resource = Resource.create(
            {
                "service.name": service_name,
                "service.namespace": "tensorcast",
                "service.version": service_version,
                "tc.node.role": role,
            }
        )

        # Sampler
        from opentelemetry.sdk.trace.sampling import (
            ALWAYS_ON,
            ParentBased,
            ParentBasedTraceIdRatio,
            TraceIdRatioBased,
        )

        sampler_name = (obs.otel.sampler or "parentbased_traceidratio").strip().lower()
        try:
            rate = float(obs.otel.sampler_arg or "1.0")
        except Exception:
            rate = 1.0
        if sampler_name == "always_on":
            sampler = ALWAYS_ON
        elif sampler_name == "traceidratio":
            sampler = TraceIdRatioBased(rate)
        elif sampler_name == "parentbased_always_on":
            sampler = ParentBased(ALWAYS_ON)
        else:
            sampler = ParentBasedTraceIdRatio(rate)

        provider = TracerProvider(resource=resource, sampler=sampler)

        # Exporter
        proto = obs.otel.exporter_protocol
        endpoint = obs.otel.exporter_otlp_endpoint or "http://127.0.0.1:4317"
        if (
            proto == obs.OTelProtocol.O_TEL_PROTOCOL_HTTP_PROTOBUF
            and HttpExporter is not None
        ):
            exporter = HttpExporter(endpoint=endpoint)
        else:
            exporter = GrpcExporter(endpoint=endpoint)

        provider.add_span_processor(BatchSpanProcessor(exporter))

        trace.set_tracer_provider(provider)
        _instrument_grpc()
        _OTEL_INITIALIZED = True
        return True


def _has_active_sdk_provider() -> bool:
    """Return True if a real SDK TracerProvider is already installed.

    This helps SDK-library contexts avoid reconfiguring the application's
    tracing pipeline. We check for the SDK type rather than relying on
    implementation details of the default proxy provider.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider as _SdkTracerProvider

        provider = trace.get_tracer_provider()
        return isinstance(provider, _SdkTracerProvider)
    except Exception:
        return False


def ensure_client_otel(
    service_default: str = "tensorcast-client", role: str = "client"
) -> None:
    """Ensure OTel is configured for client SDK when explicitly enabled.

    Default behavior: do not auto-initialize OpenTelemetry for the client SDK.
    - If an SDK TracerProvider is already installed by the application, ensure
      gRPC instrumentation is active (idempotent) and return.
    - Otherwise, initialize only when client config provides
      `observability.otel.enabled: true`.
    """
    if _has_active_sdk_provider():
        # Respect the application's provider; just ensure instrumentation.
        with _INIT_LOCK:
            from contextlib import suppress

            with suppress(Exception):
                _instrument_grpc()
        return

    cfg = get_client_config()
    if cfg is not None and cfg.HasField("observability"):
        obs: commonpb.Observability = cfg.observability
        # Initialize only when explicitly enabled
        if getattr(obs.otel, "enabled", False):
            setup_otel_from_observability(obs, role)
        return
    # No client config or observability not provided → default off (no-op)


def set_span_attributes(attrs: dict[str, Any]) -> None:
    """Attach attributes to the current active span, if any.

    Best-practice filtering: by default, high-cardinality attributes are
    dropped to avoid exploding attribute cardinality in backends. Set
    `TC_OTEL_ALLOW_HIGH_CARDINALITY_ATTRS=1` to allow them for debugging.

    Considered high-cardinality by default (filtered):
    - tc.artifact.id, tc.replica.id, tc.transport.id, tc.request.id
    - tc.registration.id, tc.worker.id, tc.device.uuid
    - tc.disk.path, tc.daemon.address, *.uuid, *.address

    Explicitly allowed (low-cardinality) examples: tc.device.id, tc.source,
    tc.memory.type, tc.location, server.address, server.port.
    """
    try:
        from opentelemetry import trace

        def _truthy(s: str | None) -> bool:
            return (s or "").strip().lower() in {"1", "true", "yes", "on"}

        allow_hc = _truthy(os.getenv("TC_OTEL_ALLOW_HIGH_CARDINALITY_ATTRS"))

        # Exceptions that we always allow even if they match *.id heuristics
        allowlist = {
            # Business identifiers explicitly allowed per requirement
            "tc.artifact.id",
            "tc.device.id",
            "tc.memory.type",
            "tc.memory.size",
            "tc.source",
            "tc.location",
            "server.address",
            "server.port",
        }
        # Exact keys considered high-card by default
        highcard_exact = {
            "tc.replica.id",
            "tc.transport.id",
            "tc.request.id",
            "tc.registration.id",
            "tc.worker.id",
            "tc.device.uuid",
            "tc.disk.path",
            "tc.daemon.address",
        }

        def _is_high_card(k: str) -> bool:
            if k in allowlist:
                return False
            if k in highcard_exact:
                return True
            if k.endswith(".uuid") or k.endswith(".address"):
                return True
            # Heuristic: any other *.id not in allowlist
            return bool(k.endswith(".id") and k not in allowlist)

        span = trace.get_current_span()
        if span is None:
            return
        if hasattr(span, "is_recording") and not span.is_recording():
            return
        for k, v in attrs.items():
            if not allow_hc and _is_high_card(str(k)):
                continue
            try:
                span.set_attribute(k, v)
            except Exception:  # noqa: BLE001
                # Ignore per-attribute failures to avoid affecting RPC handling
                continue
    except Exception:  # noqa: BLE001
        # OpenTelemetry not installed or not initialized — ignore
        return
