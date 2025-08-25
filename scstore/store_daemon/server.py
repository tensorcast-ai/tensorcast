#  Copyright (c) 2025, StepCast Team.

"""Server startup and management for StoreDaemon."""

from concurrent import futures
from typing import Callable

import grpc
from prometheus_client import start_http_server
from py_grpc_prometheus.prometheus_server_interceptor import PromServerInterceptor

from scstore.logger import init_logger
from scstore.proto import store_daemon_pb2_grpc

from .config import StoreDaemonConfig
from .servicer import StoreDaemonServicer

logger = init_logger(__name__)


def serve(
    *,
    config: StoreDaemonConfig,
    on_started: Callable | None = None,
) -> None:
    """
    Start the gRPC server for replica storage.

    Args:
        config: StoreDaemonConfig instance containing all server parameters.
        on_started: Optional callback invoked once the gRPC server has started.
    """

    # Extract commonly used fields for readability
    host = config.server.host
    port = config.server.port
    num_thread = config.server.num_threads
    metrics_port = config.network.metrics_port

    # Start Prometheus metrics server
    try:
        start_http_server(metrics_port)
        logger.info(f"Prometheus metrics server started on port {metrics_port}")
    except Exception as e:
        logger.error(f"Failed to start metrics server on port {metrics_port}: {e}")

    # Create gRPC server
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=num_thread),
        interceptors=[PromServerInterceptor(enable_handling_time_histogram=True)],
    )

    # Create and add servicer
    servicer = StoreDaemonServicer(config)

    store_daemon_pb2_grpc.add_StoreDaemonServicer_to_server(servicer, server)

    # Start server
    listen_addr = f"{host}:{port}"
    server.add_insecure_port(listen_addr)
    server.start()
    logger.info(f"Started gRPC server on {listen_addr}")

    if on_started:
        on_started()

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down gRPC server")
        servicer.graceful_shutdown()
