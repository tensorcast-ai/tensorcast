#  Copyright (c) 2025, TensorCast Team.

"""Standalone entry point for the Global Store Web UI.

This module allows running the Web UI as a **separate process** from the
main Global Store gRPC server. The Web UI communicates with the Global Store
exclusively via gRPC, eliminating direct database access and preventing
any potential lock conflicts.

Example
-------
    python -m tensorcast.global_store.webui_backend.main \
        --grpc-host 127.0.0.1 --grpc-port 50051 \
        --ui-host 0.0.0.0 --ui-port 9000

This entry no longer reads environment variables; pass necessary options
via CLI arguments or the Global Store process configuration.
"""

from __future__ import annotations

import argparse
import sys

import uvicorn

from tensorcast.global_store.config import GlobalStoreConfig
from tensorcast.global_store.webui_backend.app import WebUIApp
from tensorcast.global_store.webui_backend.grpc_client import GlobalStoreClientConfig
from tensorcast.logger import init_logger

logger = init_logger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:  # noqa: D401,E501
    parser = argparse.ArgumentParser(description="Global Store Web UI (stand-alone)")

    parser.add_argument(
        "--grpc-host",
        type=str,
        default="127.0.0.1",
        help="Hostname of the Global Store gRPC server",
    )
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=50051,
        help="Port of the Global Store gRPC server",
    )
    parser.add_argument(
        "--ui-host",
        type=str,
        default="0.0.0.0",
        help="Host interface to bind the Web UI HTTP server to",
    )
    parser.add_argument(
        "--ui-port",
        type=int,
        default=9000,
        help="Port to expose the Web UI HTTP server",
    )

    return parser.parse_args(argv)


def cli(argv: list[str] | None = None) -> None:  # noqa: D401
    """CLI entry point used by ``python -m``."""

    args = _parse_args(argv)

    # ------------------------------------------------------------------
    # Create gRPC client configuration
    # ------------------------------------------------------------------
    grpc_config = GlobalStoreClientConfig(host=args.grpc_host, port=args.grpc_port)

    # ------------------------------------------------------------------
    # Compose a minimal config object for the Web UI.  We do *not* rely on the
    # complete GlobalStoreConfig because only a small subset of fields is
    # required by the UI backend (mainly CORS and static directory settings).
    # If the environment defines compatible variables (e.g. TENSORCAST_UI_*), the
    # usual config loader still handles them.
    # ------------------------------------------------------------------

    base_cfg = GlobalStoreConfig()
    ui_config = base_cfg.model_copy(
        update={
            "ui_host": args.ui_host,
            "ui_port": args.ui_port,
        }
    )

    # ------------------------------------------------------------------
    # Construct the Web UI FastAPI application.
    # ------------------------------------------------------------------
    app_wrapper = WebUIApp(grpc_config, ui_config)

    logger.info(
        "Starting Web UI at http://%s:%d (gRPC backend: %s:%d)",
        args.ui_host,
        args.ui_port,
        args.grpc_host,
        args.grpc_port,
    )

    # ------------------------------------------------------------------
    # Launch Uvicorn – the call blocks until the process receives a shutdown
    # signal (SIGTERM / KeyboardInterrupt).
    # ------------------------------------------------------------------
    uvicorn.run(app_wrapper.app, host=args.ui_host, port=args.ui_port, log_level="info")


if __name__ == "__main__":  # pragma: no cover – executed only when run directly
    try:
        cli()
    except KeyboardInterrupt:
        # Allow clean exit when terminated by parent process
        logger.info("Web UI process interrupted – shutting down.")
        sys.exit(0)
