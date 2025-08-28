#  Copyright (c) 2025, StepCast Team.

"""Service management utilities for StoreDaemon."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import click
import grpc

from scstore.cli_utils.config_loader import (
    ConfigError,
    load_config,
    print_config_summary,
    validate_config,
)
from scstore.cli_utils.pid_manager import (
    cleanup_pid_file,
    get_process_info,
    is_process_running,
    read_pid_file,
    stop_process,
)
from scstore.logger import init_logger, setup_logging
from scstore.proto import store_daemon_pb2, store_daemon_pb2_grpc
from scstore.store_daemon.config import StoreDaemonConfig

logger = init_logger(__name__)


class ServiceError(Exception):
    """Base exception for service operations."""

    pass


def start_service(
    config_file: Path | None,
    cli_args: dict[str, Any],
    pid_file: Path,
    log_file: Path,
    blocking: bool,
    verbose: bool,
) -> None:
    """
    Start the StoreDaemon service.

    Args:
        config_file: Optional path to YAML configuration file
        cli_args: Dictionary of CLI arguments
        pid_file: Path to PID file
        log_file: Path to log file for daemon mode
        blocking: Run in blocking (foreground) mode
        verbose: Enable verbose logging

    Raises:
        ServiceError: If service cannot be started
    """
    # Expand user symbols (e.g., ~) and convert to absolute paths early so
    # that subsequent operations (particularly daemonisation which changes
    # the working directory to '/') continue to refer to the intended files
    # regardless of cwd changes.
    pid_file = pid_file.expanduser().resolve()
    log_file = log_file.expanduser().resolve()

    # ------------------------------------------------------------------
    # Configure stdio behaviour based on execution mode.
    # ------------------------------------------------------------------
    setup_logging(
        level="DEBUG" if verbose else "INFO",
        log_file=log_file,
    )

    # Load and validate configuration
    try:
        if config_file:
            click.echo(f"Loading configuration from {config_file}")
        config = load_config(config_file, cli_args)
        validate_config(config)
    except ConfigError as e:
        raise ServiceError(f"Configuration error: {e}") from e

    # Check for existing daemon
    _check_existing_daemon(pid_file)

    # Print configuration summary
    _print_startup_info(config, blocking, pid_file, log_file)
    print_config_summary(config)

    # Setup signal handlers
    _setup_signal_handlers(pid_file)

    # Start the C++ daemon service
    _start_cpp_daemon_service(
        config=config,
        pid_file=pid_file,
        log_file=log_file,
        blocking=blocking,
        verbose=verbose,
    )


def stop_service(pid_file: Path, force: bool) -> None:
    """
    Stop the StoreDaemon service.

    Args:
        pid_file: Path to PID file
        force: Force kill the daemon

    Raises:
        ServiceError: If service cannot be stopped
    """
    pid = read_pid_file(pid_file)

    if not pid:
        click.echo(f"No PID file found at {pid_file}")
        click.echo("StoreDaemon may not be running or is using a different PID file")
        raise ServiceError("Service not found")

    if not is_process_running(pid):
        click.echo(f"Process with PID {pid} is not running")
        cleanup_pid_file(pid_file)
        return

    # Stop the process
    sig_name = "SIGKILL" if force else "SIGTERM"
    click.echo(f"Sending {sig_name} to PID {pid}...")

    if stop_process(pid, force=force):
        click.echo(f"StoreDaemon (PID {pid}) stopped successfully")
        cleanup_pid_file(pid_file)
    else:
        raise ServiceError(f"Failed to stop process {pid}")


def _format_bytes(bytes_value: int) -> str:
    """Format bytes into human-readable format."""
    if bytes_value < 0:
        return "N/A"

    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    value = float(bytes_value)

    while value >= 1024 and unit_index < len(units) - 1:
        value /= 1024
        unit_index += 1

    if unit_index == 0:
        return f"{int(value)} {units[unit_index]}"
    return f"{value:.1f} {units[unit_index]}"


def _format_duration(seconds: int) -> str:
    """Format duration in seconds to human-readable format."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes}m {secs}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs}s"


def _display_detailed_status(
    pid: int, response: "store_daemon_pb2.GetDetailedStatusResponse"
) -> None:
    """Display detailed status information from GetDetailedStatus response."""
    click.echo("\n" + "=" * 60)
    click.echo("StoreDaemon Status")
    click.echo("=" * 60)

    # Basic info
    click.echo(f"PID: {pid}")
    click.echo(f"Uptime: {_format_duration(response.uptime_seconds)}")
    click.echo(f"Health: {'Healthy' if response.is_healthy else 'Unhealthy'}")
    if response.is_shutting_down:
        click.echo("Status: SHUTTING DOWN")
    click.echo(f"Registered: {'Yes' if response.is_registered else 'No'}")
    if response.worker_id:
        click.echo(f"Worker ID: {response.worker_id}")

    # Memory pool status
    mp = response.memory_pool_info
    click.echo("\nMemory Pool Status")
    click.echo("=" * 60)
    click.echo(f"Total Size: {_format_bytes(mp.total_size_bytes)}")
    if mp.total_size_bytes > 0:
        percent_avail = (mp.available_bytes / mp.total_size_bytes) * 100
        click.echo(
            f"Available: {_format_bytes(mp.available_bytes)} ({percent_avail:.1f}%)"
        )
        click.echo(
            f"Allocated: {_format_bytes(mp.allocated_bytes)} ({100 - percent_avail:.1f}%)"
        )
    else:
        click.echo(f"Available: {_format_bytes(mp.available_bytes)}")
        click.echo(f"Allocated: {_format_bytes(mp.allocated_bytes)}")
    if mp.allocated_chunks_count > 0:
        click.echo(f"Chunks: {mp.allocated_chunks_count} allocated")
    if mp.chunk_size_bytes > 0:
        click.echo(f"Chunk Size: {_format_bytes(mp.chunk_size_bytes)}")

    # GPU devices
    if response.gpu_devices:
        click.echo(f"\nGPU Devices ({len(response.gpu_devices)} total)")
        click.echo("=" * 60)
        for gpu in response.gpu_devices:
            click.echo(f"\nGPU {gpu.device_id}")
            if gpu.device_uuid:
                click.echo(f"  UUID: {gpu.device_uuid}")
            click.echo(f"  Total Memory: {_format_bytes(gpu.total_memory_bytes)}")
            if gpu.total_memory_bytes > 0:
                used_percent = (gpu.used_memory_bytes / gpu.total_memory_bytes) * 100
                click.echo(
                    f"  Used: {_format_bytes(gpu.used_memory_bytes)} ({used_percent:.1f}%)"
                )
                click.echo(
                    f"  Free: {_format_bytes(gpu.free_memory_bytes)} ({100 - used_percent:.1f}%)"
                )
            else:
                click.echo(f"  Used: {_format_bytes(gpu.used_memory_bytes)}")
                click.echo(f"  Free: {_format_bytes(gpu.free_memory_bytes)}")

            if gpu.loaded_replicas:
                click.echo(f"  Replicas ({len(gpu.loaded_replicas)} loaded):")
                for artifact in gpu.loaded_replicas:
                    click.echo(
                        f"    - {artifact.artifact_id} ({_format_bytes(artifact.artifact_size_bytes)})"
                    )

    # CPU replicas
    if response.cpu_replicas:
        click.echo(f"\nCPU Memory Replicas ({len(response.cpu_replicas)} total)")
        click.echo("=" * 60)
        for artifact in response.cpu_replicas:
            click.echo(
                f"- {artifact.artifact_id} ({_format_bytes(artifact.artifact_size_bytes)})"
            )
            if artifact.is_registered_for_comm:
                click.echo("  [Registered for RDMA]")

    # Communication status
    comm = response.communication_info
    if comm.enabled:
        click.echo("\nCommunication Status")
        click.echo("=" * 60)
        click.echo("RDMA: Enabled")
        if comm.total_transfers > 0:
            click.echo(f"Total Transfers: {comm.total_transfers}")
            click.echo(
                f"Bytes Transferred: {_format_bytes(comm.total_bytes_transferred)}"
            )
            if comm.total_transfer_errors > 0:
                click.echo(f"Transfer Errors: {comm.total_transfer_errors}")

    # Summary
    click.echo("\nSummary")
    click.echo("=" * 60)
    click.echo(f"Total Replicas Loaded: {response.total_replicas_loaded}")
    click.echo(
        f"Total Artifact Size: {_format_bytes(response.total_artifact_size_bytes)}"
    )
    click.echo(f"Storage Path: {response.storage_path}")
    if response.num_worker_threads > 0:
        click.echo(f"Worker Threads: {response.num_worker_threads}")

    click.echo("")  # Empty line at the end


def check_service_status(
    pid_file: Path, *, host: str | None = None, port: int | None = None
) -> None:
    """
    Check the status of StoreDaemon service.

    Args:
        pid_file: Path to PID file
    """
    pid = read_pid_file(pid_file)

    if not pid:
        click.echo("StoreDaemon is not running (no PID file found)")
        sys.exit(1)

    if not is_process_running(pid):
        click.echo(f"StoreDaemon is not running (stale PID {pid} in {pid_file})")
        cleanup_pid_file(pid_file)
        sys.exit(1)

    # Try to connect to the StoreDaemon and get detailed status
    try:
        import grpc

        from scstore.cli_utils.config_loader import load_config
        from scstore.proto import store_daemon_pb2, store_daemon_pb2_grpc

        # Resolve address preference: explicit host/port -> config default
        if host is None or port is None:
            config = load_config(None, {})  # default config
            host = host or "127.0.0.1"
            port = port or config.server.port

        # Connect to the daemon
        channel = grpc.insecure_channel(f"{host}:{port}")
        stub = store_daemon_pb2_grpc.StoreDaemonStub(channel)

        # Get detailed status
        request = store_daemon_pb2.GetDetailedStatusRequest()
        response = stub.GetDetailedStatus(request, timeout=5.0)

        # Display the detailed status
        _display_detailed_status(pid, response)

    except Exception as e:
        # Fall back to simple process info if RPC fails
        click.echo(f"StoreDaemon is running with PID {pid}")
        click.echo(f"  (Unable to connect to gRPC service: {e})")

        # Get process info if available
        info = get_process_info(pid)
        if info and len(info) > 1:  # More than just PID
            if "started" in info:
                click.echo(f"  Started: {info['started']}")
            if "cpu_percent" in info:
                click.echo(f"  CPU: {info['cpu_percent']}%")
            if "memory_mb" in info:
                click.echo(f"  Memory: {info['memory_mb']:.1f} MB")

    sys.exit(0)


def _check_existing_daemon(pid_file: Path) -> None:
    """Check if daemon is already running."""
    existing_pid = read_pid_file(pid_file)
    if existing_pid and is_process_running(existing_pid):
        click.echo(
            f"StoreDaemon is already running with PID {existing_pid}",
            err=True,
        )
        click.echo(f"Use 'scstore-cli stop' to stop it or remove {pid_file}")
        raise ServiceError("Service already running")
    elif existing_pid:
        # Clean up stale PID file
        cleanup_pid_file(pid_file)


def _print_startup_info(
    config: StoreDaemonConfig,
    blocking: bool,
    pid_file: Path,
    log_file: Path,
) -> None:
    """Print startup information."""
    click.echo("=" * 60)
    click.echo("StoreDaemon Starting")
    click.echo("=" * 60)
    click.echo(
        f"Mode: {'Blocking (Foreground)' if blocking else 'Non-blocking (Background)'}"
    )
    if not blocking:
        click.echo(f"PID File: {pid_file}")
        click.echo(f"Log File: {log_file}")


def _setup_signal_handlers(pid_file: Path) -> None:
    """Setup signal handlers for graceful shutdown."""

    def signal_handler(signum: int, frame: Any) -> None:
        logger.info(f"Received signal {signum}, shutting down...")
        cleanup_pid_file(pid_file)
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


def _ensure_cpp_daemon_binary() -> Path:
    """Locate the C++ daemon binary without performing any build actions.

    Resolution order:
    - SCSTORE_DAEMON_BIN env var (absolute path)
    - Installed package resource: scstore/bin/scstore_daemon
    - Development workspace: <repo-root>/bazel-bin/daemon/scstore_daemon
    """
    # 1) Explicit override via env
    env_path = os.environ.get("SCSTORE_DAEMON_BIN")
    if env_path:
        p = Path(env_path)
        if p.exists() and os.access(p, os.X_OK):
            return p
        raise ServiceError(f"SCSTORE_DAEMON_BIN set but not executable: {p}")

    # 2) Packaged binary under scstore/bin/
    try:
        import importlib.resources as ir  # py3.9+

        pkg = ir.files("scstore").joinpath("bin", "scstore_daemon")
        p = Path(str(pkg))
        if p.exists() and os.access(p, os.X_OK):
            return p
    except Exception:
        pass

    # 3) Development workspace bazel-bin
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / "bazel-bin" / "daemon" / "scstore_daemon"
    if candidate.exists() and os.access(candidate, os.X_OK):
        return candidate

    raise ServiceError(
        "scstore_daemon binary not found. Set SCSTORE_DAEMON_BIN, install a package containing scstore/bin/scstore_daemon, or build it via Bazel in development (bazel build //daemon:scstore_daemon)."
    )


def _cpp_daemon_args(config: StoreDaemonConfig) -> list[str]:
    """Translate StoreDaemonConfig to C++ daemon flags."""
    listen = f"{config.server.host}:{config.server.port}"
    mem_pool = int(config.server.mem_pool_size)
    chunk = int(config.server.chunk_size)
    return [
        f"--listen_addr={listen}",
        f"--storage_path={str(config.server.storage_path)}",
        f"--p2p_port={config.network.p2p_port}",
        f"--mem_pool_size={mem_pool}",
        f"--chunk_size={chunk}",
        f"--io_threads={config.server.num_threads}",
        f"--metrics_port={config.network.metrics_port}",
    ]


def _wait_grpc_ready(host: str, port: int, timeout_s: float = 20.0) -> bool:
    """Poll GetServerConfig until the daemon responds or times out."""
    deadline = time.time() + timeout_s
    addr = f"{host}:{port}"
    while time.time() < deadline:
        try:
            channel = grpc.insecure_channel(addr)
            stub = store_daemon_pb2_grpc.StoreDaemonStub(channel)
            stub.GetServerConfig(store_daemon_pb2.GetServerConfigRequest(), timeout=1.0)
            channel.close()
            return True
        except Exception:
            time.sleep(0.2)
    return False


def _start_cpp_daemon_service(
    *,
    config: StoreDaemonConfig,
    pid_file: Path,
    log_file: Path,
    blocking: bool,
    verbose: bool,
) -> None:
    """Start the C++ daemon binary in foreground or background without waiting for readiness."""
    bin_path = _ensure_cpp_daemon_binary()
    args = [str(bin_path), *_cpp_daemon_args(config)]

    if blocking:
        # Foreground: run binary attached to this terminal, write child PID, wait, cleanup.
        try:
            proc = subprocess.Popen(args)
        except Exception as e:
            raise ServiceError(f"Failed to start daemon: {e}") from e
        try:
            # Write child PID for discoverability
            try:
                pid_file.parent.mkdir(parents=True, exist_ok=True)
                pid_file.write_text(str(proc.pid))
                logger.info("PID %d written to %s", proc.pid, pid_file)
            except Exception as e:
                proc.terminate()
                raise ServiceError(f"Could not write PID file: {e}") from e
            proc.wait()
        finally:
            cleanup_pid_file(pid_file)
        return

    # Background: spawn and detach IO to log file and return immediately
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", buffering=1) as log_fd:
            proc = subprocess.Popen(
                args, stdout=log_fd, stderr=log_fd, stdin=subprocess.DEVNULL
            )
    except Exception as e:
        raise ServiceError(f"Failed to start daemon (background): {e}") from e

    # Record PID and exit without readiness wait to avoid blocking callers.
    try:
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(str(proc.pid))
        logger.info("(bg) PID %d written to %s", proc.pid, pid_file)
    except Exception as e:
        click.echo(f"Warning: failed to write PID file: {e}", err=True)

    click.echo(
        "StoreDaemon launched in background. Use 'scstore status' to check readiness, 'scstore stop' to stop it."
    )
