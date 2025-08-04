#  Copyright (c) 2025, StepCast Team.

"""Service management utilities for StoreDaemon."""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path
from typing import Any

import click

from scstore.cli_utils.config_loader import (
    ConfigError,
    load_config,
    print_config_summary,
    validate_config,
)
from scstore.cli_utils.daemon import daemonize
from scstore.cli_utils.pid_manager import (
    PidManagerError,
    cleanup_pid_file,
    get_process_info,
    is_process_running,
    read_pid_file,
    stop_process,
    write_pid_file,
)
from scstore.logger import configure_logging, init_logger
from scstore.proto import store_daemon_pb2
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
    configure_logging(
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

    # Start the service
    _start_daemon_service(
        config=config,
        config_file=config_file,
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

            if gpu.loaded_models:
                click.echo(f"  Models ({len(gpu.loaded_models)} loaded):")
                for model in gpu.loaded_models:
                    click.echo(
                        f"    - {model.model_identifier} ({_format_bytes(model.model_size_bytes)})"
                    )

    # CPU models
    if response.cpu_models:
        click.echo(f"\nCPU Memory Models ({len(response.cpu_models)} total)")
        click.echo("=" * 60)
        for model in response.cpu_models:
            click.echo(
                f"- {model.model_identifier} ({_format_bytes(model.model_size_bytes)})"
            )
            if model.is_registered_for_comm:
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
    click.echo(f"Total Models Loaded: {response.total_models_loaded}")
    click.echo(f"Total Model Size: {_format_bytes(response.total_model_size_bytes)}")
    click.echo(f"Storage Path: {response.storage_path}")
    if response.num_worker_threads > 0:
        click.echo(f"Worker Threads: {response.num_worker_threads}")

    click.echo("")  # Empty line at the end


def check_service_status(pid_file: Path) -> None:
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

        config = load_config(None, {})  # Load default config

        # Connect to the daemon
        channel = grpc.insecure_channel(f"127.0.0.1:{config.server.port}")
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


def _start_daemon_service(
    config: StoreDaemonConfig,
    config_file: Path | None,
    pid_file: Path,
    log_file: Path,
    blocking: bool,
    verbose: bool,
) -> None:
    """Start the actual daemon service.

    For non-blocking mode we now perform *all* daemonisation **before** the
    gRPC server starts.  We use a pipe to let the daemon process inform the
    original CLI process when the server is ready so that the latter only
    exits after a successful start-up – mimicking the previous UX while
    avoiding the undefined behaviour of forking after the server has already
    created threads.
    """

    # ------------------------------------------------------------------
    # Foreground mode – write PID file so that helper commands (e.g.,
    # `scstore-cli status`) can locate the running process just like in
    # background mode.
    # ------------------------------------------------------------------

    if blocking:
        try:
            write_pid_file(pid_file)
        except PidManagerError as e:
            logger.error("Failed to write PID file: %s", e)
            raise ServiceError("Could not write PID file") from e

        from scstore.store_daemon import serve

        try:
            serve(
                config=config,
            )
        finally:
            # Ensure the PID file is removed when the process exits for any
            # reason to prevent stale files from lingering.
            cleanup_pid_file(pid_file)

        return

    # ------------------------------------------------------------------
    # Non-blocking (background) mode – new implementation.
    # ------------------------------------------------------------------
    # Create a pipe so the daemon can notify us when the server is ready
    read_fd, write_fd = os.pipe()

    pid = os.fork()
    if pid > 0:
        # ------------------------------
        # Parent – wait for notification
        # ------------------------------
        os.close(write_fd)
        try:
            # Wait (blocking) for exactly one byte from the daemon indicating
            # successful start-up.  EOF or any other value is treated as a
            # failure.
            started_flag = os.read(read_fd, 1)
        finally:
            os.close(read_fd)

        if started_flag == b"1":
            click.echo(
                "StoreDaemon started successfully. Use 'scstore-cli stop' to stop the daemon"
            )
            sys.exit(0)

        click.echo("Failed to start StoreDaemon. Check logs for details.", err=True)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Child – will become the daemon.
    # ------------------------------------------------------------------
    os.close(read_fd)
    try:
        # Perform the classic double-fork.
        daemonize(log_file)

        # Write the PID file *after* daemonisation so it contains the daemon PID
        try:
            write_pid_file(pid_file)
        except PidManagerError as e:
            logger.error(f"Failed to write PID file: {e}")
            os.write(write_fd, b"0")
            os.close(write_fd)
            sys.exit(1)

        # Import here to avoid unnecessarily importing heavy modules in the
        # original CLI process.
        from scstore.store_daemon import serve

        def on_server_started() -> None:
            """Notify the original process that the server is ready."""
            try:
                os.write(write_fd, b"1")
            finally:
                # Close our copy of the write-end; the original process will
                # get EOF once we are done writing.
                os.close(write_fd)

        # Start the server (this call blocks until termination).
        serve(
            config=config,
            on_started=on_server_started,
        )
    except KeyboardInterrupt:
        click.echo("\nReceived interrupt signal, shutting down...")
        cleanup_pid_file(pid_file)
        sys.exit(0)
    except Exception as e:
        logger.exception("Error starting StoreDaemon")
        # Notify parent about the failure if possible
        try:
            os.write(write_fd, b"0")
            os.close(write_fd)
        except Exception:  # pragma: no cover – best-effort cleanup
            pass
        cleanup_pid_file(pid_file)
        raise ServiceError(f"Failed to start service: {e}") from e
