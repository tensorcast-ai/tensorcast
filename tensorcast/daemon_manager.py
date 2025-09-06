#  Copyright (c) 2025, TensorCast Team.

import atexit
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager, suppress
from typing import Iterator, List, Optional, Tuple

import grpc
from opentelemetry import trace
from opentelemetry.trace import SpanKind

from tensorcast.daemon_config import StoreDaemonConfig
from tensorcast.logger import init_logger
from tensorcast.observability.otel import ensure_client_otel, set_span_attributes
from tensorcast.proto.daemon.v1 import (
    store_daemon_pb2 as store_daemon_pb2,
)
from tensorcast.proto.daemon.v1 import (
    store_daemon_pb2_grpc as store_daemon_pb2_grpc,
)

logger = init_logger(__name__)


class DaemonManager:
    """
    Manages the lifecycle of Store Daemon automatically.

    This class can detect if a daemon is already running and reuse it,
    or start a new daemon if none exists. The daemon's lifecycle is tied
    to the current process.
    """

    def __init__(
        self,
        config: StoreDaemonConfig,
        auto_start: bool = True,
    ):
        # Initialize OTel in a library-friendly manner (no downgrade).
        ensure_client_otel("tensorcast-client", role="client")

        self.config = config
        self.auto_start = auto_start

        self.server_address = f"{config.server.host}:{config.server.port}"
        self.daemon_process: Optional[subprocess.Popen] = None
        self._daemon_started_by_us = False

        # Register cleanup function
        atexit.register(self.cleanup)

    def close(self) -> None:
        """Close resources held by the manager (terminate daemon we started)."""
        self.cleanup()

    def __enter__(self) -> "DaemonManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _host_port(self) -> Tuple[str, int]:
        try:
            host, port_s = self.server_address.split(":", 1)
            return host, int(port_s)
        except Exception:
            return self.server_address, 0

    @contextmanager
    def _span(self, name: str, kind: SpanKind) -> Iterator[trace.Span]:
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(name, kind=kind) as span:
            host, port = self._host_port()
            set_span_attributes({"server.address": host, "server.port": port})
            yield span

    def _build_start_command(self) -> List[str]:
        python = os.environ.get("TENSORCAST_PYTHON", sys.executable)
        cmd: List[str] = [
            python,
            "-m",
            "tensorcast.cli",
            "start",
            "--host",
            self.config.server.host,
            "--port",
            str(self.config.server.port),
            "--storage-path",
            str(self.config.server.storage_path),
            "--num-thread",
            str(self.config.server.num_threads),
            "--chunk-size",
            f"{self.config.server.chunk_size}B",
            "--mem-pool-size",
            f"{self.config.server.mem_pool_size}B",
            "--non-blocking",
        ]
        if self.config.server.enable_p2p_access:
            cmd.extend(["--enable-p2p-access", "True"])
        if self.config.server.enable_p2p_engine:
            cmd.extend(["--enable-p2p-engine", "True"])
        return cmd

    def _wait_until_ready(
        self, max_wait_time: int = 30, wait_interval: float = 1.0
    ) -> bool:
        elapsed = 0.0
        while elapsed < max_wait_time:
            if self.is_daemon_running():
                return True
            if self.daemon_process and self.daemon_process.poll() is not None:
                # Process died
                return False
            time.sleep(wait_interval)
            elapsed += wait_interval
        return False

    def is_daemon_running(self) -> bool:
        """Check if a daemon is already running at the specified address."""
        with self._span("Client/CheckDaemon", SpanKind.CLIENT) as span:
            channel = grpc.insecure_channel(self.server_address)
            stub = store_daemon_pb2_grpc.StoreDaemonServiceStub(channel)
            try:
                request = store_daemon_pb2.GetServerConfigRequest()
                _ = stub.GetServerConfig(request, timeout=2.0)
                logger.info(f"Found existing daemon at {self.server_address}")
                return True
            except grpc.RpcError as e:
                span.record_exception(e)
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    logger.debug(f"No daemon found at {self.server_address}")
                    return False
                logger.warning(f"Error checking daemon status: {e}")
                return False
            except Exception as e:
                span.record_exception(e)
                logger.warning(f"Unexpected error checking daemon: {e}")
                return False
            finally:
                with suppress(Exception):
                    channel.close()

    def start_daemon(self) -> bool:
        """Start a new daemon process."""
        if not self.auto_start:
            logger.error("Auto-start is disabled, cannot start daemon")
            return False

        logger.info(f"Starting new daemon at {self.server_address}")
        cmd = self._build_start_command()

        with self._span("Client/StartDaemon", SpanKind.INTERNAL) as span:
            try:
                # Start daemon as subprocess
                self.daemon_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid,  # Create new process group
                )

                self._daemon_started_by_us = True

                if self._wait_until_ready(max_wait_time=30, wait_interval=1):
                    logger.info(f"Daemon started successfully at {self.server_address}")
                    return True

                # If not ready within time, check if it died and capture output
                if self.daemon_process and self.daemon_process.poll() is not None:
                    stdout, stderr = self.daemon_process.communicate()
                    logger.error("Daemon process exited unexpectedly")
                    with suppress(Exception):
                        logger.error(f"stdout: {stdout.decode()}")
                    with suppress(Exception):
                        logger.error(f"stderr: {stderr.decode()}")
                else:
                    logger.error("Daemon failed to start within 30 seconds")
                self.cleanup()
                return False

            except Exception as e:
                span.record_exception(e)
                logger.error(f"Failed to start daemon: {e}")
                return False

    def stop_daemon(self) -> bool:
        """Stop the daemon process if it was started by this manager."""
        if not self.daemon_process or not self._daemon_started_by_us:
            logger.debug("stop_daemon called, but no managed daemon to stop")
            return False
        self.cleanup()
        return True

    def ensure_daemon_running(self) -> bool:
        """Ensure a daemon is running, starting one if necessary."""
        if self.is_daemon_running():
            return True

        if self.auto_start:
            return self.start_daemon()
        else:
            logger.error(
                f"No daemon found at {self.server_address} and auto-start is disabled. "
                f"Please start the daemon manually with: "
                f"python -m tensorcast.cli start --storage-path {self.config.server.storage_path} "
                f"--mem-pool-size {self.config.server.mem_pool_size}"
            )
            return False

    def cleanup(self):
        """Clean up daemon process if we started it."""
        if self.daemon_process and self._daemon_started_by_us:
            logger.info("Cleaning up daemon process")
            try:
                # Send SIGTERM to the process group
                os.killpg(os.getpgid(self.daemon_process.pid), signal.SIGTERM)

                # Wait for graceful shutdown
                try:
                    self.daemon_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't shut down gracefully
                    logger.warning("Daemon didn't shut down gracefully, force killing")
                    os.killpg(os.getpgid(self.daemon_process.pid), signal.SIGKILL)
                    self.daemon_process.wait()

                logger.info("Daemon process cleaned up")

            except Exception as e:
                logger.error(f"Error cleaning up daemon process: {e}")
            finally:
                self.daemon_process = None
                self._daemon_started_by_us = False


# Global daemon manager instance
_global_daemon_manager: Optional[DaemonManager] = None


def get_daemon_manager(
    config: StoreDaemonConfig, auto_start: bool = True
) -> DaemonManager:
    """Get or create the global daemon manager instance."""
    global _global_daemon_manager

    if _global_daemon_manager is None:
        _global_daemon_manager = DaemonManager(config, auto_start)

    return _global_daemon_manager


def ensure_daemon_running(config: StoreDaemonConfig, auto_start: bool = True) -> bool:
    """Convenience function to ensure daemon is running."""
    manager = get_daemon_manager(config, auto_start)
    return manager.ensure_daemon_running()
