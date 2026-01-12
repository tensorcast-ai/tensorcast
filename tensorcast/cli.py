#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

import click

from tensorcast import runtime
from tensorcast.cli_utils import ServiceError, global_store_manager, service_manager
from tensorcast.cli_utils.health import ping_global_store
from tensorcast.cli_utils.paths import (
    current_global_session_path,
    global_session_paths,
    runtime_state_path,
)
from tensorcast.cli_utils.process import read_json_default, read_runtime_state
from tensorcast.logger import init_logger

logger = init_logger(__name__)

GlobalStoreMode = Literal["connect", "start", "none"]

_LOG_LEVELS = ("debug", "info", "warn", "warning", "error")


def _extract_override_key(raw: str) -> str:
    if "=" not in raw:
        raise ServiceError(f"Invalid --set override '{raw}'; expected key=value")
    key = raw.split("=", 1)[0].strip()
    if not key:
        raise ServiceError(f"Invalid --set override '{raw}'; empty key")
    return key


def _paths_conflict(a: str, b: str) -> bool:
    if a == b:
        return True
    return a.startswith(f"{b}.") or b.startswith(f"{a}.")


def _parse_endpoint(value: str) -> str:
    raw = value.strip()
    if not raw:
        raise ServiceError("Global Store endpoint cannot be empty")
    try:
        host, port_s = raw.rsplit(":", 1)
        if not host:
            raise ValueError("missing host")
        port = int(port_s)
    except Exception as exc:  # noqa: BLE001
        raise ServiceError(f"Invalid Global Store endpoint '{value}'") from exc
    if port < 0 or port > 65535:
        raise ServiceError(
            f"Invalid Global Store endpoint '{value}'; port out of range"
        )
    return f"{host}:{port}"


def _runtime_session_to_dict(session: runtime.RuntimeSession) -> dict:
    return {
        "session_id": session.session_id,
        "daemon": {
            "pid": session.daemon_pid,
            "address": session.daemon_address,
            "p2p_address": session.daemon_p2p_address,
            "logs_dir": str(session.logs_dir) if session.logs_dir else None,
            "owner": session.owner,
        },
        "global_store": {
            "mode": session.global_store_mode,
            "address": session.global_store_address,
            "session": session.global_store_session,
            "owner": session.global_store_owner,
            "cluster_token": session.cluster_token,
        },
        "started_at": session.started_at,
    }


def _echo_daemon_status(session: runtime.RuntimeSession) -> None:
    click.echo(f"Daemon session: {session.session_id}")
    if session.daemon_pid:
        click.echo(f"  status      : running (pid={session.daemon_pid})")
    else:
        click.echo("  status      : unknown (pid not recorded)")
    if session.daemon_address:
        click.echo(f"  address     : {session.daemon_address}")
    if session.daemon_p2p_address:
        click.echo(f"  p2p address : {session.daemon_p2p_address}")
    if session.logs_dir:
        click.echo(f"  logs        : {session.logs_dir}")
    gs_mode = session.global_store_mode or "none"
    gs_owner = "owner" if session.global_store_owner else "borrowed"
    if session.global_store_address:
        click.echo(
            f"  global store: mode={gs_mode} addr={session.global_store_address} ({gs_owner})"
        )
    else:
        click.echo(f"  global store: mode={gs_mode} addr=none")


def _read_current_global_session() -> str | None:
    path = current_global_session_path()
    if not path.exists():
        return None
    try:
        data = path.read_text(encoding="utf-8").strip()
        return data or None
    except Exception:
        return None


def _global_status_payload(session_id: str | None) -> tuple[str | None, dict]:
    state = read_runtime_state(runtime_state_path())
    gs_state = state.get("global_store") if isinstance(state, dict) else None
    sid = (
        session_id
        or (gs_state.get("session_id") if isinstance(gs_state, dict) else None)
        or _read_current_global_session()
    )
    payload: dict = {}
    inst = global_session_paths(sid) if sid else None
    if inst and inst.state_json.exists():
        payload = read_json_default(inst.state_json, {})
    elif isinstance(gs_state, dict):
        payload = dict(gs_state)
    address = None
    if isinstance(payload, dict):
        gs = payload.get("global_store", payload)
        if isinstance(gs, dict):
            address = (
                gs.get("address") or gs_state.get("address")
                if isinstance(gs_state, dict)
                else None
            )
    if address is None and isinstance(gs_state, dict):
        address = gs_state.get("address")
    return sid, {
        "address": address,
        "state": payload,
    }


@click.group()
def cli():
    """TensorCast CLI - manage the Store Daemon and Global Store."""


@cli.group()
def daemon():
    """Manage the local Store Daemon."""


@daemon.command("start")
@click.option(
    "--config",
    "-c",
    required=False,
    type=click.Path(exists=True, path_type=Path),
    help=(
        "Path to unified daemon config (YAML/JSON). "
        "If omitted, tries $TENSORCAST_DAEMON_CONFIG or ~/.tensorcast/config/daemon.yaml, "
        "falling back to an embedded default."
    ),
)
@click.option(
    "--global-store-mode",
    type=click.Choice(["connect", "start", "none"]),
    default="none",
    show_default=True,
    help="Global Store orchestration mode.",
)
@click.option(
    "--global-store-address",
    default=None,
    help="Explicit Global Store host:port (connect-only when provided).",
)
@click.option(
    "--global-store-endpoints",
    "global_store_endpoints",
    multiple=True,
    help="Repeatable Global Store host:port list for HA injection and connect mode.",
)
@click.option(
    "--stable-bytes",
    default=None,
    help="Override engine.memory_tiers.stable_bytes (supports KB/MB/GB).",
)
@click.option(
    "--enable-rdma",
    is_flag=True,
    help="Enable RDMA (communicator.enable_rdma=true).",
)
@click.option(
    "--log-level",
    type=click.Choice(_LOG_LEVELS, case_sensitive=False),
    default=None,
    help="Override observability.logging.level.",
)
@click.option(
    "--session",
    default=None,
    help="Override session id to use for this daemon",
)
@click.option(
    "--set",
    "config_overrides",
    multiple=True,
    help=(
        "Override daemon config values (key=value). Repeatable; values are YAML-parsed. "
        "Quote strings containing ':' or spaces."
    ),
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit daemon status as JSON after startup",
)
@click.option(
    "--blocking",
    is_flag=True,
    help="Run in foreground; stream logs to console and stop on exit.",
)
def daemon_start(
    config: Path | None,
    global_store_mode: GlobalStoreMode,
    global_store_address: str | None,
    global_store_endpoints: tuple[str, ...],
    stable_bytes: str | None,
    enable_rdma: bool,
    log_level: str | None,
    session: str | None,
    config_overrides: tuple[str, ...],
    as_json: bool,
    blocking: bool,
):
    """Start the Store Daemon via the unified runtime orchestrator."""
    try:
        overrides = list(config_overrides or ())
        override_paths = {_extract_override_key(entry) for entry in overrides}

        def _check_conflict(path: str) -> None:
            for existing in override_paths:
                if _paths_conflict(existing, path):
                    raise ServiceError(
                        f"Config override '{path}' conflicts with existing override '{existing}'"
                    )

        if stable_bytes:
            _check_conflict("engine.memory_tiers.stable_bytes")
            overrides.append(f"engine.memory_tiers.stable_bytes={stable_bytes}")
            override_paths.add("engine.memory_tiers.stable_bytes")
        if enable_rdma:
            _check_conflict("communicator.enable_rdma")
            overrides.append("communicator.enable_rdma=true")
            override_paths.add("communicator.enable_rdma")
        if log_level:
            _check_conflict("observability.logging.level")
            overrides.append(f"observability.logging.level={log_level}")
            override_paths.add("observability.logging.level")

        endpoints = [_parse_endpoint(ep) for ep in global_store_endpoints if ep]
        if endpoints:
            _check_conflict("high_availability.global_store_endpoints")
            if global_store_mode == "start":
                raise ServiceError(
                    "--global-store-endpoints cannot be used with --global-store-mode start"
                )
            if global_store_address:
                normalized_addr = _parse_endpoint(global_store_address)
                global_store_address = normalized_addr
                if normalized_addr not in endpoints:
                    raise ServiceError(
                        "--global-store-address must match one of --global-store-endpoints"
                    )
            else:
                global_store_address = endpoints[0]
            global_store_mode = "connect"
        if global_store_address:
            if global_store_mode == "start":
                raise ServiceError(
                    "--global-store-address cannot be used with --global-store-mode start"
                )
            if global_store_mode != "connect":
                global_store_mode = "connect"
            global_store_address = _parse_endpoint(global_store_address)

        existing = runtime.status()
        if existing is not None:
            click.echo(
                "Error: A StoreDaemon is already running. Use the existing daemon "
                "or stop it before starting another.",
                err=True,
            )
            if as_json:
                click.echo(json.dumps(_runtime_session_to_dict(existing), indent=2))
                sys.exit(1)
            _echo_daemon_status(existing)
            sys.exit(1)

        session_obj = runtime.start(
            daemon_config=config,
            session_id=session,
            global_store_mode=global_store_mode,
            global_store_address=global_store_address,
            config_overrides=overrides or None,
            ha_endpoints=endpoints or None,
            blocking=blocking,
            to_console=True,
            reuse_existing=False,
            fate_share=blocking,
        )
        if blocking:
            return
    except ServiceError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if as_json:
        click.echo(json.dumps(_runtime_session_to_dict(session_obj), indent=2))
        return
    _echo_daemon_status(session_obj)


@daemon.command("stop")
@click.option(
    "--force",
    is_flag=True,
    help="Force kill the daemon (SIGKILL instead of SIGTERM)",
)
@click.option(
    "--session",
    default=None,
    help="Override session id to stop",
)
def daemon_stop(force: bool, session: str | None):
    """Stop the Store Daemon."""
    try:
        runtime.stop(session_id=session, force=force)
    except ServiceError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        click.echo(f"Unexpected error: {e}", err=True)
        logger.exception("Unexpected error during daemon stop")
        sys.exit(1)


@daemon.command("status")
@click.option(
    "--session",
    default=None,
    help="Override session id to query",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit status as JSON",
)
def daemon_status(session: str | None, as_json: bool):
    """Check daemon status using runtime state + health checks."""
    try:
        session_obj = runtime.status(session)
        if session_obj is None:
            click.echo(
                "No local daemon session found. Start one with 'tensorcast daemon start'."
            )
            sys.exit(1)
        if as_json:
            click.echo(json.dumps(_runtime_session_to_dict(session_obj), indent=2))
            return
        _echo_daemon_status(session_obj)
    except Exception as e:  # noqa: BLE001
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@daemon.command("restart")
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
    help="Path to unified daemon config (YAML/JSON).",
)
@click.option(
    "--global-store-mode",
    type=click.Choice(["connect", "start", "none"]),
    default="none",
    show_default=True,
    help="Global Store orchestration mode.",
)
@click.option(
    "--global-store-address",
    default=None,
    help="Explicit Global Store host:port (connect-only when provided).",
)
@click.option(
    "--global-store-endpoints",
    "global_store_endpoints",
    multiple=True,
    help="Repeatable Global Store host:port list for HA injection and connect mode.",
)
@click.option(
    "--stable-bytes",
    default=None,
    help="Override engine.memory_tiers.stable_bytes (supports KB/MB/GB).",
)
@click.option(
    "--enable-rdma",
    is_flag=True,
    help="Enable RDMA (communicator.enable_rdma=true).",
)
@click.option(
    "--log-level",
    type=click.Choice(_LOG_LEVELS, case_sensitive=False),
    default=None,
    help="Override observability.logging.level.",
)
@click.option(
    "--session",
    default=None,
    help="Override session id to restart",
)
@click.option(
    "--set",
    "config_overrides",
    multiple=True,
    help=(
        "Override daemon config values (key=value). Repeatable; values are YAML-parsed. "
        "Quote strings containing ':' or spaces."
    ),
)
def daemon_restart(
    force: bool,
    config: Path | None,
    global_store_mode: GlobalStoreMode,
    global_store_address: str | None,
    global_store_endpoints: tuple[str, ...],
    stable_bytes: str | None,
    enable_rdma: bool,
    log_level: str | None,
    session: str | None,
    config_overrides: tuple[str, ...],
):
    """Restart the Store Daemon (stop then start)."""
    daemon_stop(force=force, session=session)
    daemon_start(
        config=config,
        global_store_mode=global_store_mode,
        global_store_address=global_store_address,
        global_store_endpoints=global_store_endpoints,
        stable_bytes=stable_bytes,
        enable_rdma=enable_rdma,
        log_level=log_level,
        session=session,
        config_overrides=config_overrides,
        as_json=False,
    )


@daemon.command("logs")
@click.option("--stderr", is_flag=True, help="Show stderr instead of stdout")
@click.option("-f", "--follow", is_flag=True, help="Follow log output (tail -f)")
@click.option(
    "--session",
    default=None,
    help="Override session id to tail",
)
def daemon_logs(stderr: bool, follow: bool, session: str | None):
    """Show or tail logs for the local daemon session."""
    try:
        service_manager.logs_tail(session_id=session, stderr=stderr, follow=follow)
    except ServiceError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.group(name="global")
def global_group():
    """Manage the embedded Global Store."""


@global_group.command("start")
@click.option(
    "--config", type=click.Path(path_type=Path), help="Global Store config path"
)
@click.option("--listen-host", default=None, help="Listen host override")
@click.option(
    "--listen-port", type=int, default=None, help="Listen port override (0 allowed)"
)
@click.option("--gs-session", default=None, help="Override global session id")
@click.option("--json", "as_json", is_flag=True, help="Emit status as JSON after start")
@click.option(
    "--blocking",
    is_flag=True,
    help="Run in foreground; stream logs to console and stop on exit.",
)
def global_start(
    config: Path | None,
    listen_host: str | None,
    listen_port: int | None,
    gs_session: str | None,
    as_json: bool,
    blocking: bool,
):
    """Start the Global Store (single-instance)."""
    try:
        inst = global_store_manager.start_global_store(
            config_path=config,
            session_id=gs_session,
            listen_host=listen_host,
            listen_port=listen_port,
            to_console=True,
            fate_share=blocking,
            blocking=blocking,
            announce=False,
        )
        if blocking:
            return
    except ServiceError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if as_json:
        click.echo(
            json.dumps(
                {
                    "session_id": inst.id,
                    "pid": inst.pid,
                    "address": inst.address,
                    "metrics_port": inst.metrics_port,
                },
                indent=2,
            )
        )
    else:
        click.echo(
            f"Global Store started (session={inst.id}) at {inst.address}. Logs: {inst.logs_dir}"
        )


@global_group.command("stop")
@click.option("--force", is_flag=True, help="Force kill the Global Store")
@click.option("--gs-session", default=None, help="Override global session id")
def global_stop(force: bool, gs_session: str | None):
    """Stop the Global Store instance."""
    try:
        global_store_manager.stop_global_store(session_id=gs_session, force=force)
    except ServiceError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@global_group.command("status")
@click.option("--gs-session", default=None, help="Override global session id")
@click.option("--json", "as_json", is_flag=True, help="Emit status as JSON")
def global_status(gs_session: str | None, as_json: bool):
    """Report Global Store status."""
    sid, payload = _global_status_payload(gs_session)
    address = payload.get("address")
    health = ping_global_store(address, timeout=1.0) if address else None
    output = {
        "session_id": sid,
        "address": address,
        "state": payload.get("state", {}),
        "health": (
            {
                "address": health.address,
                "listen_host": health.listen_host,
                "listen_port": health.listen_port,
                "advertise_host": health.advertise_host,
                "advertise_port": health.advertise_port,
                "advertise_address": (
                    f"{health.advertise_host}:{health.advertise_port}"
                    if health.advertise_host and health.advertise_port
                    else None
                ),
                "metrics_port": health.metrics_port,
                "cluster_token": health.cluster_token,
                "version": health.version,
                "db_file": health.db_file,
            }
            if health
            else None
        ),
    }
    if as_json:
        click.echo(json.dumps(output, indent=2))
        return
    click.echo(f"Global Store session: {sid or 'unknown'}")
    if address:
        click.echo(f"  address: {address}")
    if payload.get("state"):
        click.echo("  state  : recorded")
    if health:
        click.echo("  health : SERVING")
        if health.advertise_host and health.advertise_port:
            click.echo(f"  advertise: {health.advertise_host}:{health.advertise_port}")
        if health.metrics_port:
            click.echo(f"  metrics: {health.metrics_port}")
    else:
        click.echo("  health : unavailable")


@global_group.command("logs")
@click.option("--stderr", is_flag=True, help="Show stderr instead of stdout")
@click.option("-f", "--follow", is_flag=True, help="Follow log output (tail -f)")
@click.option("--gs-session", default=None, help="Override global session id")
def global_logs(stderr: bool, follow: bool, gs_session: str | None):
    """Tail Global Store logs."""
    try:
        global_store_manager.global_store_logs(
            session_id=gs_session, stderr=stderr, follow=follow
        )
    except ServiceError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def main():
    """Entry point for the tensorcast command."""
    cli()


if __name__ == "__main__":
    main()
