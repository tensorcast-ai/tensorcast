#  Copyright (c) 2025, StepCast Team.

"""scstore command-line interface."""

import sys
from pathlib import Path

import click

from scstore.cli_utils import (
    DEFAULT_LOG_FILE,
    DEFAULT_PID_FILE,
    ServiceError,
    check_service_status,
    start_service,
    stop_service,
)
from scstore.logger import init_logger

logger = init_logger(__name__)


@click.group()
def cli():
    """scstore CLI - Manage the StoreDaemon service."""
    pass


@cli.command(name="start")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to YAML configuration file",
)
@click.option(
    "--host",
    default="0.0.0.0",
    help="Host address to bind (default: 0.0.0.0)",
)
@click.option(
    "--port",
    default=8073,
    type=int,
    help="gRPC port to bind (default: 8073)",
)
@click.option(
    "--storage-path",
    default="",
    help="Path to artifact storage directory (default: '')",
)
@click.option(
    "--mem-pool-size",
    default="8GB",
    help="Memory pool size, e.g., 8GB",
)
@click.option(
    "--num-thread",
    default=10,
    type=int,
    help="Number of worker threads (default: 10)",
)
@click.option(
    "--chunk-size",
    default="32MB",
    help="Chunk size, e.g., 32MB",
)
@click.option(
    "--enable-p2p-engine",
    is_flag=True,
    help="Whether communication engine is enabled (default: True)",
)
@click.option(
    "--enable-p2p-access",
    is_flag=True,
    help="Whether artifact registration is required (default: True)",
)
@click.option(
    "--enable-rdma",
    is_flag=True,
    help="Enable RDMA layer within communication engine (default: False)",
)
@click.option(
    "--global-store-address",
    help="Global store address (e.g., localhost:50051)",
)
@click.option(
    "--p2p-port",
    default=9090,
    type=int,
    help="p2p data transfer port (default: 9090)",
)
@click.option(
    "--metrics-port",
    default=9091,
    type=int,
    help="Prometheus metrics port (default: 9091)",
)
@click.option(
    "--health-check-port",
    default=8080,
    type=int,
    help="Health check HTTP port (default: 8080)",
)
@click.option(
    "--pinned-memory-timeout-ms",
    default=30000,
    type=int,
    help="Timeout for pinned memory allocation in milliseconds (default: 30000)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
@click.option(
    "--blocking/--non-blocking",
    is_flag=True,
    default=True,
    help="Run in blocking mode (foreground). Default is non-blocking (background)",
)
@click.option(
    "--pid-file",
    type=click.Path(path_type=Path),
    default=DEFAULT_PID_FILE,
    help=f"PID file location (default: {DEFAULT_PID_FILE})",
)
@click.option(
    "--log-file",
    type=click.Path(path_type=Path),
    default=DEFAULT_LOG_FILE,
    help=f"Log file location for daemon mode (default: {DEFAULT_LOG_FILE})",
)
def start(**kwargs):
    """Start the StoreDaemon service."""
    try:
        # Extract config file if provided
        config_file = kwargs.pop("config", None)

        # Extract service management options
        pid_file = kwargs.pop("pid_file")
        log_file = kwargs.pop("log_file")
        blocking = kwargs.pop("blocking")
        verbose = kwargs.pop("verbose")

        # Start the service
        start_service(
            config_file=config_file,
            cli_args=kwargs,
            pid_file=pid_file,
            log_file=log_file,
            blocking=blocking,
            verbose=verbose,
        )
    except ServiceError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    # except Exception as e:
    #     click.echo(f"Unexpected error: {e}", err=True)
    #     logger.exception("Unexpected error during service start")
    #     sys.exit(1)


@cli.command(name="stop")
@click.option(
    "--pid-file",
    type=click.Path(path_type=Path),
    default=DEFAULT_PID_FILE,
    help=f"PID file location (default: {DEFAULT_PID_FILE})",
)
@click.option(
    "--force",
    is_flag=True,
    help="Force kill the daemon (SIGKILL instead of SIGTERM)",
)
def stop(pid_file: Path, force: bool):
    """Stop the StoreDaemon service."""
    try:
        stop_service(pid_file, force)
    except ServiceError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        logger.exception("Unexpected error during service stop")
        sys.exit(1)


@cli.command(name="status")
@click.option(
    "--pid-file",
    type=click.Path(path_type=Path),
    default=DEFAULT_PID_FILE,
    help=f"PID file location (default: {DEFAULT_PID_FILE})",
)
@click.option(
    "--host",
    default=None,
    help="Optional host override for status RPC (default: config/default)",
)
@click.option(
    "--port",
    default=None,
    type=int,
    help="Optional port override for status RPC (default: config/default)",
)
def status(pid_file: Path, host: str | None, port: int | None):
    """Check the status of StoreDaemon service."""
    try:
        check_service_status(pid_file, host=host, port=port)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command(name="restart")
@click.option(
    "--pid-file",
    type=click.Path(path_type=Path),
    default=DEFAULT_PID_FILE,
    help=f"PID file location (default: {DEFAULT_PID_FILE})",
)
@click.option(
    "--force",
    is_flag=True,
    help="Force kill the daemon during stop",
)
@click.pass_context
def restart(ctx: click.Context, pid_file: Path, force: bool):
    """Restart the StoreDaemon service."""
    # First stop the service
    ctx.invoke(stop, pid_file=pid_file, force=force)

    # Then start it again with the same configuration
    # Note: This assumes the service will use the same configuration
    # as before. For full restart with new options, use stop then start.
    click.echo("Starting service...")
    ctx.invoke(start, pid_file=pid_file)


def main():
    """Entry point for the scstore command."""
    cli()


if __name__ == "__main__":
    main()
