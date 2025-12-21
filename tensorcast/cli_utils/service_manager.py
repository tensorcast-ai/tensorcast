#  Copyright (c) 2025, TensorCast Team.

"""Slim service management façade for StoreDaemon (Linux-only).

This module composes focused helpers from sibling modules to provide a clean
API used by the CLI and SDK: discover_default_config_path, start_service,
stop_service, check_service_status, logs_tail, and session helpers.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import click

from tensorcast.cli_utils.config import discover_daemon_config
from tensorcast.cli_utils.errors import ServiceError
from tensorcast.cli_utils.filesys import open_log_binary
from tensorcast.cli_utils.health import get_daemon_config
from tensorcast.cli_utils.logs import logs_tail as _logs_tail
from tensorcast.cli_utils.network import (
    pick_free_tcp_port,
    resolve_connect_host,
    wait_daemon_ready,
)
from tensorcast.cli_utils.paths import (
    DaemonSession,
    clear_current_session_if_matches,
    get_current_session_id,
    get_session_address,
    list_sessions,
    runtime_lock_path,
    runtime_state_path,
    session_paths,
    set_current_session_id,
)
from tensorcast.cli_utils.proc import (
    build_daemon_process_env,
    ensure_cpp_daemon_binary,
    is_matching_daemon_process,
    kill_force,
    kill_gracefully,
    preexec_detached,
    preexec_fate_sharing,
)
from tensorcast.cli_utils.process import (
    atomic_write_json,
    clear_runtime_daemon,
    ensure_process_started,
    file_lock,
    instance_fingerprint,
    join_threads,
    prune_process_records,
    read_json_default,
    read_runtime_state,
    register_process,
    start_log_threads,
    update_runtime_daemon,
    write_session_state,
)
from tensorcast.cli_utils.status import check_service_status as _check_service_status
from tensorcast.daemon_runtime_config import (
    apply_daemon_config_overrides,
    dump_daemon_config,
    load_daemon_config,
)


@contextlib.contextmanager
def _runtime_and_session_lock(inst: DaemonSession) -> Any:
    """Enforce runtime -> session lock order for state writes."""

    with file_lock(runtime_lock_path()), file_lock(inst.pids_lock):
        yield


def discover_default_config_path() -> Path | None:
    """Discover a default StoreDaemon config path for convenience starts.

    Order:
    1) $TENSORCAST_DAEMON_CONFIG if set and exists
    2) ~/.tensorcast/config/daemon.yaml or ~/.tensorcast/config/daemon.yml
    """
    return discover_daemon_config()


def start_service(
    *,
    config_path: Path,
    session_id: str | None = None,
    blocking: bool = False,
    to_console: bool = True,
    register_current: bool = True,
    publish_meta: bool = True,
    ephemeral: bool = False,
    persist_runtime_state: bool = True,
    owner: bool = True,
    restrict_to_localhost: bool = False,
    global_store: dict[str, Any] | None = None,
    listen_host: str | None = None,
    listen_port: int | None = None,
    config_overrides: tuple[str, ...] | list[str] | None = None,
    ha_endpoints: list[str] | None = None,
    ha_enabled: bool | None = None,
    fate_share: bool = True,
) -> DaemonSession:
    """Start the C++ StoreDaemon.

    - If config listen.port is 0 or missing, pick a free TCP port and pass derived config to daemon.
    - Startup is always blocking: this call returns only once the daemon is ready (or the process exits).
    """
    cfg = load_daemon_config(config_path)
    cfg_modified = False
    if config_overrides:
        try:
            apply_daemon_config_overrides(cfg, config_overrides)
        except ValueError as exc:
            raise ServiceError(f"Invalid config override: {exc}") from exc
        cfg_modified = True

    if listen_host is not None:
        cfg.server.listen.host = listen_host
        cfg_modified = True
    if listen_port is not None:
        if listen_port < 0 or listen_port > 65535:
            raise ServiceError("listen_port must be between 0 and 65535")
        cfg.server.listen.port = listen_port
        cfg_modified = True

    # Restrict to loopback if requested (SDK private launch)
    if restrict_to_localhost:
        with contextlib.suppress(Exception):
            if cfg.server.listen.host != "127.0.0.1":
                cfg.server.listen.host = "127.0.0.1"
                cfg_modified = True
            if cfg.server.p2p_listen.host and cfg.server.p2p_listen.host != "127.0.0.1":
                cfg.server.p2p_listen.host = "127.0.0.1"
                cfg_modified = True

    host = cfg.server.listen.host or "127.0.0.1"
    port = int(cfg.server.listen.port or 0)

    if port <= 0:
        port = pick_free_tcp_port()
        cfg.server.listen.port = port
        cfg_modified = True
    if not cfg.server.listen.host:
        cfg.server.listen.host = host
        cfg_modified = True

    p2p_host = cfg.server.p2p_listen.host or host
    p2p_port = int(cfg.server.p2p_listen.port or 0)
    if p2p_port <= 0:
        p2p_port = pick_free_tcp_port()
        cfg.server.p2p_listen.port = p2p_port
        cfg_modified = True
    if not cfg.server.p2p_listen.host:
        cfg.server.p2p_listen.host = p2p_host
        cfg_modified = True

    if ha_enabled is not None and bool(ha_enabled) != bool(
        cfg.high_availability.enabled
    ):
        cfg.high_availability.enabled = bool(ha_enabled)
        cfg_modified = True
    if ha_endpoints is not None:
        endpoints = list(ha_endpoints)
        if ha_enabled is None and endpoints:
            cfg.high_availability.enabled = True
            cfg_modified = True
        if not endpoints and cfg.high_availability.global_store_endpoints:
            cfg.high_availability.ClearField("global_store_endpoints")
            cfg_modified = True
        if endpoints:
            cfg.high_availability.ClearField("global_store_endpoints")
            for endpoint in endpoints:
                try:
                    host, port_str = endpoint.rsplit(":", 1)
                    port_val = int(port_str)
                except Exception as exc:  # noqa: BLE001
                    raise ServiceError(f"Invalid HA endpoint '{endpoint}'") from exc
                ep = cfg.high_availability.global_store_endpoints.add()
                ep.host = host
                ep.port = port_val
            cfg_modified = True
    if ha_enabled is False and cfg.high_availability.global_store_endpoints:
        cfg.high_availability.ClearField("global_store_endpoints")
        cfg_modified = True

    inst = session_paths(session_id)
    started_at = time.time()
    runtime_state_file = runtime_state_path()
    effective_cfg_path = inst.effective_config_path
    try:
        if cfg_modified:
            dump_daemon_config(cfg, effective_cfg_path)
            daemon_cfg_arg = f"--config={str(effective_cfg_path)}"
            meta_cfg_path = str(effective_cfg_path)
        else:
            daemon_cfg_arg = f"--config={str(Path(config_path).expanduser().resolve())}"
            meta_cfg_path = str(Path(config_path).expanduser().resolve())
    except Exception as e:  # noqa: BLE001
        raise ServiceError(f"Failed to write effective config: {e}") from e

    bin_path = ensure_cpp_daemon_binary()
    args = [str(bin_path), daemon_cfg_arg]

    # Persist logs only when running non-blocking. In blocking mode, mirror to console only.
    persist_logs = not blocking
    so_path = inst.logs / "daemon.out"
    se_path = inst.logs / "daemon.err"
    so = open_log_binary(so_path) if persist_logs else None
    se = open_log_binary(se_path) if persist_logs else None

    env = build_daemon_process_env({**os.environ, "TENSORCAST_INSTANCE": inst.id})

    try:
        preexec_fn = preexec_fate_sharing if fate_share else preexec_detached
        use_pipes = blocking or fate_share
        stdout_target = (
            subprocess.PIPE
            if use_pipes
            else (so if so is not None else subprocess.DEVNULL)
        )
        stderr_target = (
            subprocess.PIPE
            if use_pipes
            else (se if se is not None else subprocess.DEVNULL)
        )
        proc = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=stdout_target,
            stderr=stderr_target,
            cwd=inst.session,
            env=env,
            preexec_fn=preexec_fn,
            close_fds=True,
        )
    except Exception:
        with contextlib.suppress(Exception):
            if so is not None:
                so.close()
        with contextlib.suppress(Exception):
            if se is not None:
                se.close()
        raise

    connect_host = resolve_connect_host(host)
    p2p_connect_host = resolve_connect_host(p2p_host)
    object.__setattr__(inst, "address", f"{connect_host}:{port}")
    object.__setattr__(inst, "p2p_address", f"{p2p_connect_host}:{p2p_port}")
    fingerprint = instance_fingerprint(proc.pid)

    with _runtime_and_session_lock(inst):
        register_process(
            inst_id=inst.id,
            pids_path=inst.pids_json,
            proc=proc,
            role="daemon",
            stdout_path=so_path,
            stderr_path=se_path,
            lock_path=None,
        )

    if not use_pipes:
        with contextlib.suppress(Exception):
            if so is not None:
                so.close()
        with contextlib.suppress(Exception):
            if se is not None:
                se.close()
        log_threads: list[Any] = []
    else:
        sinks_out: list[Any] = []
        sinks_err: list[Any] = []
        if persist_logs and so is not None:
            sinks_out.append(so)
        if persist_logs and se is not None:
            sinks_err.append(se)
        if to_console:
            sinks_out.append(sys.stdout)
            sinks_err.append(sys.stderr)
        log_threads = start_log_threads(proc, sinks_out, sinks_err)

    def _persist_state() -> None:
        session_state = {
            "session_id": inst.id,
            "ephemeral": ephemeral,
            "started_at": started_at,
            "daemon": {
                "pid": int(proc.pid),
                "address": inst.address,
                "p2p_address": inst.p2p_address,
                "config_path": meta_cfg_path,
                "binary": str(bin_path),
            },
            "global_store": global_store or {},
            "logs_dir": str(inst.logs),
        }
        with contextlib.suppress(Exception):
            from tensorcast import __version__ as _tc_version

            session_state["version"] = _tc_version
        with _runtime_and_session_lock(inst):
            if publish_meta:
                _write_meta(
                    inst,
                    address=f"{connect_host}:{port}",
                    config_path=meta_cfg_path,
                    p2p_address=inst.p2p_address,
                    daemon_bin=str(bin_path),
                )
            write_session_state(inst.session_state_json, session_state)
            if persist_runtime_state:
                update_runtime_daemon(
                    path=runtime_state_file,
                    session_id=inst.id,
                    pid=int(proc.pid),
                    address=inst.address,
                    p2p_address=inst.p2p_address,
                    owner=owner,
                    fingerprint=fingerprint,
                )
            if register_current:
                set_current_session_id(inst.id)

    # In blocking mode, ensure graceful shutdown:
    # - Register atexit to stop the daemon session we started
    # - Install SIGINT/SIGTERM handlers to call stop_service before exiting
    daemon_cfg = None
    try:
        ensure_process_started(
            proc,
            inst.logs if persist_logs else None,
            startup_grace=1.0,
        )
    except ServiceError:
        _cleanup_failed_start(proc, log_threads, inst, so, se)
        raise

    ready = wait_daemon_ready(connect_host, port, timeout=None, proc=proc)
    if not ready:
        retcode = proc.poll()
        _cleanup_failed_start(proc, log_threads, inst, so, se)
        log_hint = f" See logs under {inst.logs}" if persist_logs else ""
        if retcode is not None:
            raise ServiceError(
                f"Daemon exited with code {retcode} during startup (address={connect_host}:{port})."
                + log_hint
            )
        raise ServiceError(
            f"Daemon exited during startup (address={connect_host}:{port})." + log_hint
        )

    # Best-effort backfill of effective listen/p2p ports from daemon once it is serving.
    with contextlib.suppress(Exception):
        daemon_cfg = get_daemon_config(f"{connect_host}:{port}", timeout=1.0)
    if daemon_cfg is not None:
        listen = None
        p2p_listen = None
        with contextlib.suppress(Exception):
            listen = daemon_cfg.server.listen
            p2p_listen = daemon_cfg.server.p2p_listen
        if listen is None:
            listen = getattr(daemon_cfg, "listen", None)
        if p2p_listen is None:
            p2p_listen = getattr(daemon_cfg, "p2p_listen", None)

        eff_host = getattr(listen, "host", None) or host
        eff_port = int(getattr(listen, "port", 0) or port)
        eff_p2p_host = getattr(p2p_listen, "host", None) or p2p_host or eff_host
        eff_p2p_port = int(getattr(p2p_listen, "port", 0) or p2p_port)

        connect_host = resolve_connect_host(eff_host)
        p2p_connect_host = resolve_connect_host(eff_p2p_host or eff_host)
        port = eff_port
        p2p_port = eff_p2p_port
        object.__setattr__(inst, "address", f"{connect_host}:{port}")
        object.__setattr__(inst, "p2p_address", f"{p2p_connect_host}:{p2p_port}")

    # Avoid race where daemon_cfg is None but daemon is already serving; fall back to existing values.
    if inst.address is None:
        object.__setattr__(inst, "address", f"{connect_host}:{port}")
    if inst.p2p_address is None:
        object.__setattr__(inst, "p2p_address", f"{p2p_connect_host}:{p2p_port}")

    _persist_state()

    if blocking:

        def _cleanup() -> None:
            with contextlib.suppress(Exception):
                stop_service(session_id=inst.id)

        atexit.register(_cleanup)

        def _handle_signal(signum, _frame) -> None:  # noqa: ANN001
            _cleanup()
            try:
                sys.exit(128 + int(signum))
            except SystemExit:
                raise

        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)

        ret = proc.wait()
        join_threads(log_threads)
        click.echo(f"daemon exited with code {ret}")
        return inst

    click.echo(
        f"StoreDaemon started (daemon session={inst.id}) at {connect_host}:{port}. Logs: {inst.logs}"
    )
    return inst


def stop_service(
    *, session_id: str | None = None, grace: float = 10.0, force: bool = False
) -> None:
    sid = session_id or get_current_session_id()
    if not sid:
        raise ServiceError("No session id provided and no current session found")
    inst = session_paths(sid)
    runtime_state_file = runtime_state_path()
    if not inst.pids_json.exists():
        click.echo(f"No pids.json found for daemon session {sid}")
        with contextlib.suppress(Exception), file_lock(runtime_lock_path()):
            state = read_runtime_state(runtime_state_file)
            daemon_state = state.get("daemon")
            if isinstance(daemon_state, dict) and daemon_state.get("session_id") == sid:
                clear_runtime_daemon(runtime_state_file)
        return
    with _runtime_and_session_lock(inst):
        data = read_json_default(inst.pids_json, {"processes": []})
    procs = data.get("processes", [])
    targets: list[int] = []
    for proc in reversed(procs):
        pid = int(proc.get("pid", 0))
        if pid <= 0:
            continue
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            continue
        first_cmd = None
        try:
            cmd_list = proc.get("cmd", [])
            first_cmd = cmd_list[0] if cmd_list else None
        except Exception:
            first_cmd = None
        if not is_matching_daemon_process(pid, first_cmd):
            continue
        targets.append(pid)
        if force:
            kill_force(pgid)
        else:
            if not kill_gracefully(pgid, grace=grace):
                kill_force(pgid)
    with _runtime_and_session_lock(inst):
        if targets:
            prune_process_records(
                pids_path=inst.pids_json,
                predicate=lambda entry: int(entry.get("pid", 0)) in targets,
                lock_path=None,
            )
        with contextlib.suppress(Exception):
            state = read_runtime_state(runtime_state_file)
            daemon_state = state.get("daemon")
            if isinstance(daemon_state, dict) and daemon_state.get("session_id") == sid:
                clear_runtime_daemon(runtime_state_file)
        clear_current_session_if_matches(sid)
    click.echo(f"Stopped daemon session {sid}")


def check_service_status(
    *, session_id: str | None = None, host: str | None = None, port: int | None = None
) -> None:
    sid = session_id or get_current_session_id()
    if not sid:
        click.echo(
            "No local daemon session found. Start one with 'tensorcast daemon start'."
        )
        sys.exit(1)
    _check_service_status(session_id=sid, host=host, port=port)


def logs_tail(
    *, session_id: str | None = None, stderr: bool = False, follow: bool = False
) -> None:
    _logs_tail(session_id=session_id, stderr=stderr, follow=follow)


def _cleanup_failed_start(
    proc: subprocess.Popen[Any],
    log_threads: list[Any],
    inst: DaemonSession,
    so,
    se,
) -> None:
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        pgid = None
    if (
        pgid is not None
        and proc.poll() is None
        and not kill_gracefully(pgid, grace=2.0)
    ):
        kill_force(pgid)
    with contextlib.suppress(Exception):
        proc.wait(timeout=2.0)
    join_threads(log_threads, timeout=1.0)
    for stream in (so, se):
        if stream is None:
            continue
        with contextlib.suppress(Exception):
            stream.flush()
            stream.close()
    with _runtime_and_session_lock(inst):
        prune_process_records(
            pids_path=inst.pids_json,
            predicate=lambda entry: int(entry.get("pid", 0)) == int(proc.pid),
            lock_path=None,
        )
        with contextlib.suppress(FileNotFoundError):
            inst.session_state_json.unlink()
        with contextlib.suppress(FileNotFoundError):
            inst.meta_json.unlink()
        with contextlib.suppress(Exception):
            state = read_runtime_state(runtime_state_path())
            daemon_state = state.get("daemon")
            if (
                isinstance(daemon_state, dict)
                and daemon_state.get("session_id") == inst.id
                and int(daemon_state.get("pid", 0)) == int(proc.pid)
            ):
                clear_runtime_daemon(runtime_state_path())
    with contextlib.suppress(Exception):
        clear_current_session_if_matches(inst.id)


def _write_meta(
    inst: DaemonSession,
    *,
    address: str,
    config_path: str,
    p2p_address: str | None = None,
    daemon_bin: str | None = None,
) -> None:
    from tensorcast import __version__ as _tc_version

    meta = {
        "schema_version": 1,
        "session_id": inst.id,
        "address": address,
        "config_path": config_path,
        "created_at": __import__("time").time(),
        "user": os.environ.get("USER") or os.environ.get("USERNAME", ""),
        "version": _tc_version,
    }
    if p2p_address:
        meta["p2p_address"] = p2p_address
    if daemon_bin:
        meta["daemon_bin"] = daemon_bin
    atomic_write_json(inst.meta_json, meta)


__all__ = [
    "ServiceError",
    "discover_default_config_path",
    "start_service",
    "stop_service",
    "check_service_status",
    "logs_tail",
    "get_current_session_id",
    "get_session_address",
    "list_sessions",
]
