#  Copyright (c) 2025, TensorCast Team.

"""Slim service management façade for StoreDaemon (Linux-only).

This module composes focused helpers from sibling modules to provide a clean
API used by the CLI and SDK: discover_default_config_path, start_service,
stop_service, check_service_status, logs_tail, and session helpers.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import click

from tensorcast.daemon_runtime_config import dump_daemon_config, load_daemon_config

from .errors import ServiceError
from .filesys import open_log_binary, read_json_locked, write_json_locked
from .logs import logs_tail as _logs_tail
from .network import pick_free_tcp_port, resolve_connect_host, wait_daemon_ready
from .paths import (
    DaemonSession,
    clear_current_session_if_matches,
    get_current_session_id,
    get_session_address,
    list_sessions,
    session_paths,
    set_current_session_id,
)
from .proc import (
    build_daemon_process_env,
    ensure_cpp_daemon_binary,
    is_matching_daemon_process,
    kill_force,
    kill_gracefully,
    preexec_fate_sharing,
    pump,
)
from .status import check_service_status as _check_service_status


def discover_default_config_path() -> Path | None:
    """Discover a default StoreDaemon config path for convenience starts.

    Order:
    1) $TENSORCAST_DAEMON_CONFIG if set and exists
    2) ~/.tensorcast/store_daemon_config.yaml or ~/.tensorcast/store_daemon.yaml
    3) examples/config/store_daemon_config.yaml (in repo checkout)
    """
    env = os.environ.get("TENSORCAST_DAEMON_CONFIG")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p

    home = Path.home() / ".tensorcast"
    for name in ("store_daemon_config.yaml", "store_daemon.yaml"):
        candidate = home / name
        if candidate.exists():
            return candidate

    repo_root = Path(__file__).resolve().parents[2]
    example = repo_root / "examples" / "config" / "store_daemon_config.yaml"
    if example.exists():
        return example
    return None


def start_service(
    *,
    config_path: Path,
    session_id: str | None = None,
    blocking: bool = False,
    to_console: bool = True,
    wait: bool = False,
    timeout: float = 15.0,
    register_current: bool = True,
    publish_meta: bool = True,
    restrict_to_localhost: bool = False,
) -> DaemonSession:
    """Start the C++ StoreDaemon.

    - If config listen.port is 0 or missing, pick a free TCP port and pass derived config to daemon.
    - In non-blocking mode, when wait=True, wait up to `timeout` seconds for readiness before returning.
    """
    cfg = load_daemon_config(config_path)

    # Restrict to loopback if requested (SDK private launch)
    if restrict_to_localhost:
        with contextlib.suppress(Exception):
            if cfg.server.listen.host != "127.0.0.1":
                cfg.server.listen.host = "127.0.0.1"
            if cfg.server.p2p_listen.host and cfg.server.p2p_listen.host != "127.0.0.1":
                cfg.server.p2p_listen.host = "127.0.0.1"

    host = cfg.server.listen.host or "127.0.0.1"
    port = int(cfg.server.listen.port or 0)

    cfg_modified = False
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

    inst = session_paths(session_id)
    effective_cfg_path = inst.session / "effective_daemon_config.yaml"
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

    so_path = inst.logs / "daemon.out"
    se_path = inst.logs / "daemon.err"
    so = open_log_binary(so_path)
    se = open_log_binary(se_path)

    env = build_daemon_process_env({**os.environ, "TENSORCAST_INSTANCE": inst.id})

    try:
        proc = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=inst.session,
            env=env,
            preexec_fn=lambda: preexec_fate_sharing(),
            close_fds=True,
        )
    except Exception:
        with contextlib.suppress(Exception):
            so.close()
        with contextlib.suppress(Exception):
            se.close()
        raise

    _register_process(inst, role="daemon", proc=proc, so=so_path, se=se_path)
    connect_host = resolve_connect_host(host)
    if publish_meta:
        _write_meta(
            inst,
            address=f"{connect_host}:{port}",
            config_path=meta_cfg_path,
            p2p_address=f"{resolve_connect_host(p2p_host)}:{p2p_port}",
            daemon_bin=str(bin_path),
        )
    if register_current:
        set_current_session_id(inst.id)
    object.__setattr__(inst, "address", f"{connect_host}:{port}")
    object.__setattr__(
        inst, "p2p_address", f"{resolve_connect_host(p2p_host)}:{p2p_port}"
    )

    sinks_out: list[Any] = [so]
    sinks_err: list[Any] = [se]
    if to_console:
        sinks_out.append(sys.stdout)
        sinks_err.append(sys.stderr)
    log_threads = _start_daemon_log_threads(proc, sinks_out, sinks_err)

    if blocking:
        ret = proc.wait()
        _join_threads(log_threads)
        click.echo(f"daemon exited with code {ret}")
        return inst

    if wait:
        try:
            _ensure_process_started(proc, so_path, startup_grace=min(timeout, 1.0))
        except ServiceError:
            _cleanup_failed_start(proc, log_threads, inst, so, se)
            raise

        ready = wait_daemon_ready(connect_host, port, timeout=timeout, proc=proc)
        if not ready:
            retcode = proc.poll()
            _cleanup_failed_start(proc, log_threads, inst, so, se)
            if retcode is not None:
                raise ServiceError(
                    f"Daemon exited with code {retcode} during startup (address={connect_host}:{port}). "
                    f"Check logs at {so_path}"
                )
            raise ServiceError(
                f"Daemon failed to become ready within {timeout:.0f}s (address={connect_host}:{port}). "
                f"Check logs at {so_path}"
            )

    click.echo(
        f"StoreDaemon started (daemon session={inst.id}) at {connect_host}:{port}. Logs: {so_path}"
    )
    return inst


def stop_service(
    *, session_id: str | None = None, grace: float = 10.0, force: bool = False
) -> None:
    sid = session_id or get_current_session_id()
    if not sid:
        raise ServiceError("No session id provided and no current session found")
    inst = session_paths(sid)
    if not inst.pids_json.exists():
        click.echo(f"No pids.json found for daemon session {sid}")
        return
    data = read_json_locked(inst.pids_json)
    procs = data.get("processes", [])
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
        if force:
            kill_force(pgid)
        else:
            if not kill_gracefully(pgid, grace=grace):
                kill_force(pgid)
    click.echo(f"Stopped daemon session {sid}")
    clear_current_session_if_matches(sid)


def check_service_status(
    *, session_id: str | None = None, host: str | None = None, port: int | None = None
) -> None:
    sid = session_id or get_current_session_id()
    if not sid:
        click.echo("No local daemon session found. Start one with 'tensorcast start'.")
        sys.exit(1)
    _check_service_status(session_id=sid, host=host, port=port)


def logs_tail(
    *, session_id: str | None = None, stderr: bool = False, follow: bool = False
) -> None:
    _logs_tail(session_id=session_id, stderr=stderr, follow=follow)


def _start_daemon_log_threads(
    proc: subprocess.Popen[Any], stdout_sinks: list[Any], stderr_sinks: list[Any]
) -> list[threading.Thread]:
    threads: list[threading.Thread] = []
    if proc.stdout is not None:
        threads.append(_spawn_log_thread(proc.stdout, stdout_sinks))
    if proc.stderr is not None:
        threads.append(_spawn_log_thread(proc.stderr, stderr_sinks))
    return threads


def _spawn_log_thread(src, sinks: list[Any]) -> threading.Thread:
    def _worker() -> None:
        try:
            pump(src, sinks)
        finally:
            with contextlib.suppress(Exception):
                src.close()
            for sink in sinks:
                if sink in (sys.stdout, sys.stderr):
                    continue
                with contextlib.suppress(Exception):
                    sink.flush()
                    sink.close()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread


def _join_threads(threads: list[threading.Thread], timeout: float = 1.0) -> None:
    if not threads:
        return
    deadline = time.monotonic() + max(timeout, 0.0)
    for thread in threads:
        remaining = max(0.0, deadline - time.monotonic())
        if remaining <= 0.0:
            break
        thread.join(remaining)


def _ensure_process_started(
    proc: subprocess.Popen[Any], log_path: Path, startup_grace: float
) -> None:
    grace = max(startup_grace, 0.1)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        retcode = proc.poll()
        if retcode is not None:
            raise ServiceError(
                f"Daemon exited with code {retcode} during startup. Check logs at {log_path}"
            )
        time.sleep(0.05)
    retcode = proc.poll()
    if retcode is not None:
        raise ServiceError(
            f"Daemon exited with code {retcode} during startup. Check logs at {log_path}"
        )


def _cleanup_failed_start(
    proc: subprocess.Popen[Any],
    log_threads: list[threading.Thread],
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
    _join_threads(log_threads, timeout=1.0)
    for stream in (so, se):
        with contextlib.suppress(Exception):
            stream.flush()
            stream.close()
    with contextlib.suppress(Exception):
        clear_current_session_if_matches(inst.id)


def _register_process(
    inst: DaemonSession, *, role: str, proc: subprocess.Popen, so: Path, se: Path
) -> None:
    entry = {
        "role": role,
        "pid": int(proc.pid),
        "cmd": [str(x) for x in proc.args]
        if isinstance(proc.args, (list, tuple))
        else [str(proc.args)],
        "stdout": str(so),
        "stderr": str(se),
        "start_time": __import__("time").time(),
    }
    data = {"session_id": inst.id, "processes": [entry]}
    write_json_locked(inst.pids_json, data)


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
    write_json_locked(inst.meta_json, meta)


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
