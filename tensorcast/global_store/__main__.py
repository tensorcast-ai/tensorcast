#  Copyright (c) 2025, TensorCast Team.

"""Main entry point for Global Store server."""

import argparse
from concurrent import futures

import grpc

from tensorcast.global_store.config import GlobalStoreConfig
from tensorcast.global_store.config.settings import set_config
from tensorcast.global_store.grpc_service import GlobalStoreServicer

# Prometheus metrics
from tensorcast.global_store.metrics import (
    PrometheusInterceptor,
    start_metrics_http_server,
)
from tensorcast.logger import init_logger
from tensorcast.observability.otel import (
    setup_otel_from_observability as _setup_otel_from_observability,
)
from tensorcast.proto.global_store.v1 import global_store_pb2_grpc

logger = init_logger(__name__)


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
    args = parser.parse_args()

    # Load configuration (strict via proto)
    config = GlobalStoreConfig.from_file(args.config)
    pb_cfg = GlobalStoreConfig.load_proto_from_file(args.config)
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

    # Start Prometheus metrics HTTP server (idempotent)
    start_metrics_http_server(config.metrics_port)

    # Create gRPC server
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=config.max_workers),
        interceptors=[PrometheusInterceptor()],
    )
    global_store_pb2_grpc.add_GlobalStoreServiceServicer_to_server(servicer, server)

    # Bind to port
    server.add_insecure_port(f"[::]:{config.port}")
    server.start()

    logger.info(
        f"Global Store server started on port {config.port} "
        f"with {config.max_workers} workers"
    )
    if config.db_file:
        logger.info(f"Using persistent database: {config.db_file}")
    else:
        logger.info("Using in-memory database")
    logger.info(f"Prometheus metrics exposed on port {config.metrics_port}")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down Global Store server...")
        server.stop(grace=5)  # 5 second grace period


if __name__ == "__main__":
    main()
