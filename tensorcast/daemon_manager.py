#  Copyright (c) 2025, TensorCast Team.

import atexit
import os
import signal
import subprocess
import sys
import time
from typing import Optional

import grpc

import tensorcast.proto.store_daemon_pb2 as store_daemon_pb2
import tensorcast.proto.store_daemon_pb2_grpc as store_daemon_pb2_grpc
from tensorcast.logger import init_logger
from tensorcast.store_daemon.config import StoreDaemonConfig

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
        self.config = config
        self.auto_start = auto_start

        self.server_address = f"{config.server.host}:{config.server.port}"
        self.daemon_process: Optional[subprocess.Popen] = None
        self._daemon_started_by_us = False

        # Register cleanup function
        atexit.register(self.cleanup)

    def is_daemon_running(self) -> bool:
        """Check if a daemon is already running at the specified address."""
        try:
            channel = grpc.insecure_channel(self.server_address)
            stub = store_daemon_pb2_grpc.StoreDaemonStub(channel)

            # Try to get server config to verify the daemon is responsive
            request = store_daemon_pb2.GetServerConfigRequest()
            _ = stub.GetServerConfig(request, timeout=2.0)

            channel.close()
            logger.info(f"Found existing daemon at {self.server_address}")
            return True

        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                logger.debug(f"No daemon found at {self.server_address}")
                return False
            else:
                logger.warning(f"Error checking daemon status: {e}")
                return False
        except Exception as e:
            logger.warning(f"Unexpected error checking daemon: {e}")
            return False

    def start_daemon(self) -> bool:
        """Start a new daemon process."""
        if not self.auto_start:
            logger.error("Auto-start is disabled, cannot start daemon")
            return False

        logger.info(f"Starting new daemon at {self.server_address}")

        python = os.environ.get("TENSORCAST_PYTHON", sys.executable)

        # Build command to start daemon
        cmd = [
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

        try:
            # Start daemon as subprocess
            self.daemon_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,  # Create new process group
            )

            self._daemon_started_by_us = True

            # Wait for daemon to be ready
            max_wait_time = 30  # seconds
            wait_interval = 1  # seconds
            elapsed = 0

            while elapsed < max_wait_time:
                if self.is_daemon_running():
                    logger.info(f"Daemon started successfully at {self.server_address}")
                    return True

                # Check if process is still alive
                if self.daemon_process.poll() is not None:
                    stdout, stderr = self.daemon_process.communicate()
                    logger.error("Daemon process exited unexpectedly")
                    logger.error(f"stdout: {stdout.decode()}")
                    logger.error(f"stderr: {stderr.decode()}")
                    return False

                time.sleep(wait_interval)
                elapsed += wait_interval

            logger.error(f"Daemon failed to start within {max_wait_time} seconds")
            self.cleanup()
            return False

        except Exception as e:
            logger.error(f"Failed to start daemon: {e}")
            return False

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
