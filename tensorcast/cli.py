#  Copyright (c) 2025, TensorCast Team.

"""tensorcast command-line interface."""

import sys
from pathlib import Path

import click

from tensorcast.cli_utils import (
    ServiceError,
    check_service_status,
    logs_tail,
    start_service,
    stop_service,
)
from tensorcast.cli_utils.resolve import ping_daemon
from tensorcast.cli_utils.service_manager import (
    discover_default_config_path,
    get_current_session_id,
    get_session_address,
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
    required=False,
    type=click.Path(exists=True, path_type=Path),
    help=(
        "Path to unified daemon config (YAML/JSON). "
        "If omitted, tries $TENSORCAST_DAEMON_CONFIG, ~/.tensorcast/store_daemon_config.yaml, "
        "or examples/config/store_daemon_config.yaml"
    ),
)
@click.option(
    "--block/--no-block",
    is_flag=True,
    default=False,
    help="Run in blocking mode (foreground). Default is non-blocking",
)
@click.option(
    "--wait/--no-wait",
    default=True,
    show_default=True,
    help="Wait for daemon readiness before returning (non-blocking mode)",
)
@click.option(
    "--timeout",
    type=float,
    default=20.0,
    show_default=True,
    help="Readiness wait timeout in seconds (with --wait)",
)
def start(config: Path | None, block: bool, wait: bool, timeout: float):
    """Start the StoreDaemon service.

    Refuses to start if a current daemon session is already healthy. Use
    'tensorcast restart' or 'tensorcast stop' first.
    """
    try:
        # Prevent duplicate starts: if a current session is healthy, refuse to start another
        sid = get_current_session_id()
        if sid:
            addr = get_session_address(sid)
            if addr and ping_daemon(addr):
                raise ServiceError(
                    f"A StoreDaemon is already running for session {sid} at {addr}. "
                    "Stop it with 'tensorcast stop' or use 'tensorcast restart'."
                )

        cfg = config or discover_default_config_path()
        if not cfg:
            raise ServiceError(
                "No config provided and no default config found. "
                "Provide --config or set $TENSORCAST_DAEMON_CONFIG."
            )
        start_service(
            config_path=cfg,
            blocking=block,
            to_console=True,
            wait=wait,
            timeout=timeout,
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
    "--force",
    is_flag=True,
    help="Force kill the daemon (SIGKILL instead of SIGTERM)",
)
def stop(force: bool):
    """Stop the StoreDaemon service."""
    try:
        stop_service(session_id=None, force=force)
    except ServiceError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        logger.exception("Unexpected error during service stop")
        sys.exit(1)


@cli.command(name="status")
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
def status(host: str | None, port: int | None):
    """Check the status of StoreDaemon service."""
    try:
        check_service_status(session_id=None, host=host, port=port)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command(name="restart")
@click.option(
    "--force",
    is_flag=True,
    help="Force kill the daemon during stop",
)
@click.option(
    "--config",
    "-c",
    required=False,
    type=click.Path(exists=True, path_type=Path),
    help=(
        "Path to unified daemon config (YAML/JSON). If omitted, uses same discovery as 'start'"
    ),
)
@click.option(
    "--block/--no-block",
    is_flag=True,
    default=False,
    help="Run in blocking mode (foreground). Default is non-blocking",
)
@click.option(
    "--wait/--no-wait",
    default=True,
    show_default=True,
    help="Wait for daemon readiness before returning (non-blocking mode)",
)
@click.option(
    "--timeout",
    type=float,
    default=20.0,
    show_default=True,
    help="Readiness wait timeout in seconds (with --wait)",
)
@click.pass_context
def restart(
    ctx: click.Context,
    force: bool,
    config: Path | None,
    block: bool,
    wait: bool,
    timeout: float,
):
    """Restart the StoreDaemon service (UX aligned with start)."""
    ctx.invoke(stop, force=force)
    click.echo("Starting service...")
    ctx.invoke(start, config=config, block=block, wait=wait, timeout=timeout)


@cli.command(name="logs")
@click.option("--stderr", is_flag=True, help="Show stderr instead of stdout")
@click.option("-f", "--follow", is_flag=True, help="Follow log output (tail -f)")
def logs(stderr: bool, follow: bool):
    """Show or tail logs for the local daemon session."""
    try:
        logs_tail(session_id=None, stderr=stderr, follow=follow)
    except ServiceError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def main():
    """Entry point for the tensorcast command."""
    cli()


if __name__ == "__main__":
    main()
