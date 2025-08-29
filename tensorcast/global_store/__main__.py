#  Copyright (c) 2025, TensorCast Team.

"""Main entry point for Global Store server."""

import argparse
import logging
import subprocess
import sys
import threading
import time
from concurrent import futures
from pathlib import Path
from typing import IO

import grpc

from tensorcast.global_store.config import GlobalStoreConfig
from tensorcast.global_store.grpc_service import GlobalStoreServicer

# Prometheus metrics
from tensorcast.global_store.metrics import (
    PrometheusInterceptor,
    start_metrics_http_server,
)
from tensorcast.logger import init_logger
from tensorcast.proto import global_store_pb2_grpc

logger = init_logger(__name__)


def main():
    """Start the Global Store server."""
    parser = argparse.ArgumentParser(
        description="Global Store Server - Centralized artifact registry"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to listen on (default: from config or 50051)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker threads (default: from config or 10)",
    )
    parser.add_argument(
        "--db-file",
        type=str,
        default=None,
        help="Path to DuckDB file for persistent storage (default: in-memory)",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=8001,
        help="Port to expose Prometheus metrics (default: 8001)",
    )
    parser.add_argument(
        "--webui-log-file",
        type=str,
        default=None,
        help="Path to log file for Web UI process (default: /tmp/global-store-webui.log)",
    )
    args = parser.parse_args()

    # Load configuration
    config = GlobalStoreConfig.from_env()

    # Override with command line arguments by creating a new config instance
    if any(
        arg is not None
        for arg in [
            args.port,
            args.workers,
            args.db_file,
            args.metrics_port,
            args.webui_log_file,
        ]
    ):
        # Build update dictionary with only non-None values
        updates = {}
        if args.port is not None:
            updates["port"] = args.port
        if args.workers is not None:
            updates["max_workers"] = args.workers
        if args.db_file is not None:
            updates["db_file"] = args.db_file
        if args.metrics_port is not None:
            updates["metrics_port"] = args.metrics_port
        if args.webui_log_file is not None:
            updates["ui_log_file"] = Path(args.webui_log_file).expanduser()

        # Create new config instance with updates using Pydantic's model_copy
        config = config.model_copy(update=updates)

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
    global_store_pb2_grpc.add_GlobalStoreServicer_to_server(servicer, server)

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

    # ------------------------------------------------------------------
    # Wait until the gRPC server is actually ready to accept RPCs before
    # spawning the (optional) Web-UI process.  Although `server.start()`
    # returns immediately, there can be a short window where the server
    # socket is listening yet the service handlers are not fully ready.
    # This helper performs an active readiness probe by attempting to
    # establish a gRPC channel and waits (with a timeout) until it is
    # reported ready.  This prevents an early UI startup from failing the
    # very first RPC (e.g. ListActiveWorkers) with a generic
    # "Error in service handler!".
    # ------------------------------------------------------------------

    def _wait_for_grpc_ready(host: str, port: int, timeout_sec: float = 10.0) -> bool:
        """Block until the gRPC server at *host:port* becomes ready.

        Args:
            host: Hostname or IP address to connect to.
            port: TCP port number.
            timeout_sec: Maximum time to wait before giving up.

        Returns:
            True if the server reported readiness within *timeout_sec*,
            otherwise False.
        """

        deadline = time.time() + timeout_sec
        address = f"{host}:{port}"

        while time.time() < deadline:
            channel = grpc.insecure_channel(address)
            try:
                grpc.channel_ready_future(channel).result(timeout=0.5)
                logger.info("gRPC server is ready")
                return True  # ready!
            except grpc.FutureTimeoutError:  # noqa: BLE001  (keep retrying)
                time.sleep(0.25)
            finally:
                channel.close()
        return False  # timed-out

    ui_proc = None
    ui_log_handle: IO[str] | None = None

    # Start Web-UI only after the gRPC server passes readiness probe.
    if (
        config.ui_enabled
        and config.ui_port > 0
        and _wait_for_grpc_ready("127.0.0.1", config.port)
    ):

        def _spawn_webui(cfg: GlobalStoreConfig):
            """Launch the stand-alone Web UI process and return the Popen handle."""

            # Build command – keep minimal required arguments to decouple from
            # future CLI changes.
            cmd = [
                sys.executable,
                "-m",
                "tensorcast.global_store.webui_backend.main",
                "--grpc-host",
                "127.0.0.1",
                "--grpc-port",
                str(cfg.port),
                "--ui-host",
                cfg.ui_host,
                "--ui-port",
                str(cfg.ui_port),
            ]

            logger.info("Spawning Web UI process: %s", " ".join(cmd))
            # Capture stdout/stderr so we can forward logs to the parent logger
            return subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,  # decode bytes automatically
                bufsize=1,  # line-buffered for timely log forwarding
            )

        from pathlib import Path as _Path

        def _forward_process_logs(
            proc: subprocess.Popen[str],
            log_file_path: _Path,
        ) -> IO[str]:
            """Forward subprocess output to logger *and* a file in background threads.

            Returns:
                The opened log file handle that must be closed when done.
            """

            # Ensure parent directories exist
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            # Use open with manual close - file needs to stay open for background threads
            log_handle = open(log_file_path, "a", buffering=1, encoding="utf-8")  # noqa: SIM115

            def _pipe(stream, level: int):
                # Iteratively read lines; exit when stream closes.
                for line in stream:
                    # Write to log file and parent logger
                    log_handle.write(line)
                    log_handle.flush()

            # Start background threads for stdout/stderr
            if proc.stdout is not None:
                threading.Thread(
                    target=_pipe, args=(proc.stdout, logging.INFO), daemon=True
                ).start()
            if proc.stderr is not None:
                threading.Thread(
                    target=_pipe, args=(proc.stderr, logging.ERROR), daemon=True
                ).start()

            # Return the handle for later cleanup
            return log_handle

        try:
            ui_proc = _spawn_webui(config)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to spawn Web UI process: %s", exc)
        else:
            if ui_proc.poll() is None:
                # Stream child-process logs to parent logger.
                ui_log_handle = _forward_process_logs(ui_proc, config.ui_log_file)

                logger.info(
                    "Web UI server spawned on http://%s:%d (PID %s)",
                    config.ui_host,
                    config.ui_port,
                    ui_proc.pid,
                )
                logger.info("Web UI logs are being written to %s", config.ui_log_file)
            else:
                # Process exited immediately – surface its output.
                stdout, stderr = ui_proc.communicate()
                logger.error(
                    "Web UI process exited prematurely with code %s", ui_proc.returncode
                )
                if stdout:
                    logger.error("[WebUI stdout]\n%s", stdout)
                if stderr:
                    logger.error("[WebUI stderr]\n%s", stderr)

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down Global Store server...")
        server.stop(grace=5)  # 5 second grace period
    finally:
        if ui_proc and ui_proc.poll() is None:
            logger.info("Terminating Web UI process (PID %s)", ui_proc.pid)
            try:
                ui_proc.terminate()
                ui_proc.wait(timeout=10)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to terminate Web UI process: %s", exc)
            else:
                logger.info("Web UI process terminated.")
                # Close log file handle if present
                if ui_log_handle is not None:
                    ui_log_handle.close()


if __name__ == "__main__":
    main()
