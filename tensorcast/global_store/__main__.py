#  Copyright (c) 2025-2026, TensorCast Team.

"""Main entry point for Global Store server."""

import argparse
import socket
from concurrent import futures
from typing import cast

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from tensorcast import __version__ as _tc_version
from tensorcast.global_store.config import GlobalStoreConfig
from tensorcast.global_store.config.settings import set_config
from tensorcast.global_store.grpc_service import (
    GlobalStoreServicer,
    register_global_store_servicers,
)
from tensorcast.global_store.memory_tier_grpc_service import MemoryTierGrpcServicer

# Prometheus metrics
from tensorcast.global_store.metrics import (
    PrometheusInterceptor,
    start_metrics_http_server,
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
from tensorcast.proto.memory_tier.v1 import memory_tier_pb2_grpc

logger = init_logger(__name__)


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


def main():
    """Start the Global Store server."""
    parser = argparse.ArgumentParser(
        description="Global Store Server - Centralized artifact registry"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to Global Store config (YAML/JSON)",
    )
    parser.add_argument(
        "--listen-host",
        type=str,
        default=None,
        help="Override listen host from config",
    )
    parser.add_argument(
        "--listen-port",
        type=int,
        default=None,
        help="Override listen port from config (0 = auto)",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=None,
        help="Override metrics port from config (0 = auto)",
    )
    args = parser.parse_args()

    # Load configuration (strict via proto)
    config: GlobalStoreConfig = GlobalStoreConfig.from_file(args.config)
    pb_cfg = GlobalStoreConfig.load_proto_from_file(args.config)
    cfg_updates: dict[str, object] = {}
    if args.listen_host is not None:
        cfg_updates["listen_host"] = args.listen_host
        pb_cfg.server.listen.host = args.listen_host
    if args.listen_port is not None:
        cfg_updates["listen_port"] = args.listen_port
        pb_cfg.server.listen.port = args.listen_port
    if args.metrics_port is not None:
        cfg_updates["metrics_port"] = args.metrics_port
        pb_cfg.server.metrics_port = args.metrics_port
    if cfg_updates:
        copy_fn = getattr(config, "model_copy", None)
        if callable(copy_fn):
            config = cast(GlobalStoreConfig, copy_fn(update=cfg_updates))
        else:
            config = cast(GlobalStoreConfig, config.copy(update=cfg_updates))
    if config.advertise_host and (
        is_loopback_address(config.advertise_host)
        or is_unspecified_address(config.advertise_host)
    ):
        raise RuntimeError(
            f"Global Store advertise.host must be routable when configured (got '{config.advertise_host}')."
        )
    set_config(config)

    # Initialize OpenTelemetry from config (no env bridging)
    try:
        _setup_otel_from_observability(pb_cfg.observability, role="global-store")
    except Exception as _exc:  # noqa: BLE001
        logger.exception("Failed to initialize OpenTelemetry from config: %s", _exc)
    logger.info("OpenTelemetry tracing enabled for Global Store")

    # Initialize the service
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

    # Start Prometheus metrics HTTP server (idempotent)
    metrics_port = start_metrics_http_server(
        config.metrics_port, addr=config.listen_host or ""
    )

    # Create gRPC server
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=config.max_workers),
        interceptors=[PrometheusInterceptor()],
    )
    register_global_store_servicers(server, servicer)
    memory_tier_pb2_grpc.add_MemoryTierServiceServicer_to_server(
        memory_tier_servicer, server
    )

    # Bind to port (capture actual port if 0 was requested)
    bind_addr = f"{config.listen_host}:{config.listen_port}"
    bound_port = server.add_insecure_port(bind_addr)
    if bound_port == 0:
        raise RuntimeError(f"Failed to bind Global Store on {bind_addr}")
    listen_host = config.listen_host or "0.0.0.0"
    server.start()

    # Register gRPC health
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

    advertise_host, advertise_port, advertise_source = _resolve_advertise_address(
        listen_host=listen_host,
        bound_port=bound_port,
        explicit_host=config.advertise_host,
        explicit_port=config.advertise_port,
    )

    logger.info(
        f"Global Store server started on {listen_host}:{bound_port} "
        f"with {config.max_workers} workers"
    )
    logger.info(
        "Global Store advertise address resolved to %s:%s (source=%s)",
        advertise_host,
        advertise_port,
        advertise_source,
    )
    if is_loopback_address(advertise_host) or is_unspecified_address(advertise_host):
        logger.warning(
            "Global Store advertise address is not routable (%s); set server.advertise.host for a stable dial address.",
            advertise_host,
        )
    if config.db_file:
        logger.info(f"Using persistent database: {config.db_file}")
    else:
        logger.info("Using in-memory database")
    logger.info(f"Prometheus metrics exposed on port {metrics_port}")

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

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down Global Store server...")
        server.stop(grace=5)  # 5 second grace period


if __name__ == "__main__":
    main()
