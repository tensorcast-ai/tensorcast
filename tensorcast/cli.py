#  Copyright (c) 2025, TensorCast Team.

"""tensorcast command-line interface."""

import sys
from pathlib import Path

import click

from tensorcast.cli_utils import (
    DEFAULT_LOG_FILE,
    DEFAULT_PID_FILE,
    ServiceError,
    check_service_status,
    start_service,
    stop_service,
)
from tensorcast.logger import init_logger

logger = init_logger(__name__)


@click.group()
def cli():
    """tensorcast CLI - Manage the StoreDaemon service."""
    pass


@cli.command(name="start")
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to unified daemon config (YAML/JSON)",
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
        config_file = kwargs.pop("config")

        # Extract service management options
        pid_file = kwargs.pop("pid_file")
        log_file = kwargs.pop("log_file")
        blocking = kwargs.pop("blocking")
        verbose = kwargs.pop("verbose")

        # Start the service
        start_service(
            config_file=config_file,
            cli_args={},
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
    """Entry point for the tensorcast command."""
    cli()


if __name__ == "__main__":
    main()
