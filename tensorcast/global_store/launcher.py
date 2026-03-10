#  Copyright (c) 2025-2026, TensorCast Team.

"""Shared Global Store server bootstrap/runtime helpers."""

from __future__ import annotations

import socket
from concurrent import futures
from dataclasses import dataclass
from pathlib import Path

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from tensorcast import __version__ as _tc_version
from tensorcast.global_store.config import GlobalStoreConfig
from tensorcast.global_store.config.settings import set_config
from tensorcast.global_store.grpc_service import (
    GlobalStoreServicer,
    register_global_store_servicers,
)
from tensorcast.global_store.launch_config import (
    load_global_store_config_with_overrides,
)
from tensorcast.global_store.memory_tier_grpc_service import MemoryTierGrpcServicer
from tensorcast.global_store.metrics import (
    PrometheusInterceptor,
    ThreadPoolTelemetryReporter,
    start_metrics_http_server,
    start_thread_pool_telemetry,
)
from tensorcast.global_store.repositories.memory_tier_lease_repository import (
    MemoryTierLeaseRepository,
)
from tensorcast.global_store.repositories.memory_tier_snapshot_repository import (
    MemoryTierSnapshotRepository,
)
from tensorcast.global_store.services.address_validation import (
    is_loopback_address,
    is_unspecified_address,
)
from tensorcast.global_store.services.memory_tier_service import MemoryTierService
from tensorcast.logger import init_logger
from tensorcast.observability.otel import (
    setup_otel_from_observability as _setup_otel_from_observability,
)
from tensorcast.proto.config.v1 import global_store_config_pb2 as gsc_pb
from tensorcast.proto.memory_tier.v1 import memory_tier_pb2_grpc

logger = init_logger(__name__)


@dataclass(frozen=True)
class StartedGlobalStore:
    server: grpc.Server
    servicer: GlobalStoreServicer
    listen_host: str
    listen_port: int
    advertise_host: str
    advertise_port: int
    advertise_source: str
    metrics_port: int
    control_plane_reporter: ThreadPoolTelemetryReporter | None = None


def _resolve_route_ip(target_host: str, target_port: int) -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect((target_host, target_port))
            return sock.getsockname()[0]
    except OSError:
        return None


def _resolve_default_advertise_host() -> str | None:
    candidate = _resolve_route_ip("8.8.8.8", 80)
    if (
        candidate
        and not is_loopback_address(candidate)
        and not is_unspecified_address(candidate)
    ):
        return candidate
    try:
        hostname_ip = socket.gethostbyname(socket.gethostname())
    except OSError:
        hostname_ip = None
    if (
        hostname_ip
        and not is_loopback_address(hostname_ip)
        and not is_unspecified_address(hostname_ip)
    ):
        return hostname_ip
    return None


def _resolve_advertise_address(
    *,
    listen_host: str,
    bound_port: int,
    explicit_host: str | None,
    explicit_port: int | None,
) -> tuple[str, int, str]:
    if explicit_host and not (
        is_loopback_address(explicit_host) or is_unspecified_address(explicit_host)
    ):
        port = explicit_port if explicit_port else bound_port
        return explicit_host, port, "explicit"

    if listen_host and not (
        is_loopback_address(listen_host) or is_unspecified_address(listen_host)
    ):
        port = explicit_port if explicit_port else bound_port
        return listen_host, port, "listen"

    detected = _resolve_default_advertise_host()
    if detected:
        port = explicit_port if explicit_port else bound_port
        return detected, port, "default"

    port = explicit_port if explicit_port else bound_port
    return listen_host or "0.0.0.0", port, "listen"


def _validate_advertise_host(config: GlobalStoreConfig) -> None:
    if config.advertise_host and (
        is_loopback_address(config.advertise_host)
        or is_unspecified_address(config.advertise_host)
    ):
        raise RuntimeError(
            "Global Store advertise.host must be routable when configured "
            f"(got '{config.advertise_host}')."
        )


def start_global_store_server(
    config: GlobalStoreConfig,
    pb_cfg: gsc_pb.GlobalStoreConfig,
) -> StartedGlobalStore:
    """Start the Global Store gRPC server from resolved config values."""

    _validate_advertise_host(config)
    set_config(config)

    try:
        _setup_otel_from_observability(pb_cfg.observability, role="global-store")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to initialize OpenTelemetry from config: %s", exc)
    logger.info("OpenTelemetry tracing enabled for Global Store")

    servicer = GlobalStoreServicer(
        db_file=str(config.db_file) if config.db_file else None
    )
    snapshot_repo = MemoryTierSnapshotRepository(servicer.connection)
    lease_repo = MemoryTierLeaseRepository(servicer.connection)
    memory_tier_service = MemoryTierService(
        snapshot_repository=snapshot_repo,
        lease_repository=lease_repo,
        snapshot_retention_ns=config.memory_tier_snapshot_retention_ms * 1_000_000,
        snapshot_max_rows=config.memory_tier_snapshot_max_rows,
    )
    memory_tier_servicer = MemoryTierGrpcServicer(memory_tier_service)

    metrics_port = start_metrics_http_server(
        config.metrics_port, addr=config.listen_host or ""
    )

    executor = futures.ThreadPoolExecutor(
        max_workers=config.max_workers,
        thread_name_prefix="gs-control-plane",
    )
    control_plane_reporter = start_thread_pool_telemetry(
        executor=executor,
        poll_interval_s=1.0,
        logger=logger,
    )
    server = grpc.server(
        executor,
        interceptors=[PrometheusInterceptor()],
    )
    register_global_store_servicers(server, servicer)
    memory_tier_pb2_grpc.add_MemoryTierServiceServicer_to_server(
        memory_tier_servicer, server
    )

    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    for service_name in (
        "tensorcast.global_store.v1.ClusterRuntimeService",
        "tensorcast.global_store.v1.ArtifactCatalogService",
        "tensorcast.global_store.v1.AssemblyViewService",
        "tensorcast.global_store.v1.WorkflowOrchestrationService",
        "tensorcast.global_store.v1.ClusterAdminService",
    ):
        health_servicer.set(service_name, health_pb2.HealthCheckResponse.SERVING)

    bind_addr = f"{config.listen_host}:{config.listen_port}"
    bound_port = server.add_insecure_port(bind_addr)
    if bound_port == 0:
        raise RuntimeError(f"Failed to bind Global Store on {bind_addr}")
    listen_host = config.listen_host or "0.0.0.0"
    server.start()

    advertise_host, advertise_port, advertise_source = _resolve_advertise_address(
        listen_host=listen_host,
        bound_port=bound_port,
        explicit_host=config.advertise_host,
        explicit_port=config.advertise_port,
    )

    logger.info(
        "Global Store server started on %s:%s with %s workers",
        listen_host,
        bound_port,
        config.max_workers,
    )
    logger.info(
        "Global Store advertise address resolved to %s:%s (source=%s)",
        advertise_host,
        advertise_port,
        advertise_source,
    )
    if is_loopback_address(advertise_host) or is_unspecified_address(advertise_host):
        logger.warning(
            "Global Store advertise address is not routable (%s); set "
            "server.advertise.host for a stable dial address.",
            advertise_host,
        )
    if config.db_file:
        logger.info("Using persistent database: %s", config.db_file)
    else:
        logger.info("Using in-memory database")
    logger.info("Prometheus metrics exposed on port %s", metrics_port)

    servicer.set_runtime_info(
        listen_host=listen_host,
        listen_port=bound_port,
        advertise_host=advertise_host,
        advertise_port=advertise_port,
        metrics_port=metrics_port,
        cluster_token=config.cluster_token,
        db_file=str(config.db_file) if config.db_file else "",
        version=_tc_version,
    )
    return StartedGlobalStore(
        server=server,
        servicer=servicer,
        listen_host=listen_host,
        listen_port=bound_port,
        advertise_host=advertise_host,
        advertise_port=advertise_port,
        advertise_source=advertise_source,
        metrics_port=metrics_port,
        control_plane_reporter=control_plane_reporter,
    )


def run_global_store(
    *,
    config_path: str | Path | None = None,
    listen_host: str | None = None,
    listen_port: int | None = None,
    metrics_port: int | None = None,
) -> None:
    """Load config, start server, and block until termination."""

    config, pb_cfg = load_global_store_config_with_overrides(
        config_path,
        listen_host=listen_host,
        listen_port=listen_port,
        metrics_port=metrics_port,
    )
    started = start_global_store_server(config, pb_cfg)
    try:
        started.server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down Global Store server...")
        started.server.stop(grace=5)
    finally:
        if started.control_plane_reporter is not None:
            started.control_plane_reporter.stop()


__all__ = [
    "StartedGlobalStore",
    "run_global_store",
    "start_global_store_server",
]
