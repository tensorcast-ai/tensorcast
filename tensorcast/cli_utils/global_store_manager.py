#  Copyright (c) 2025-2026, TensorCast Team.

"""Lifecycle management for the Global Store service (single-instance)."""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import psutil

from tensorcast.cli_utils.config import (
    discover_global_store_config,
    dump_global_store_config,
    load_or_create_cluster_token,
    select_free_port,
)
from tensorcast.cli_utils.errors import ServiceError
from tensorcast.cli_utils.filesys import open_log_binary
from tensorcast.cli_utils.health import (
    GlobalStoreHealth,
    ping_global_store,
    wait_for_global_store,
)
from tensorcast.cli_utils.network import resolve_connect_host
from tensorcast.cli_utils.paths import (
    GlobalSession,
    clear_current_global_session_if_matches,
    current_global_session_path,
    global_session_paths,
    global_store_lock_path,
    runtime_lock_path,
    runtime_state_path,
    set_current_global_session_id,
)
from tensorcast.cli_utils.proc import (
    kill_force,
    kill_gracefully,
    preexec_detached,
    preexec_fate_sharing,
)
from tensorcast.cli_utils.process import (
    atomic_write_json,
    clear_runtime_global_store,
    ensure_process_started,
    file_lock,
    instance_fingerprint,
    join_threads,
    prune_process_records,
    read_json_default,
    read_runtime_state,
    register_blocking_cleanup,
    register_process,
    start_log_threads,
    update_runtime_global_store,
)
from tensorcast.global_store.config.settings import GlobalStoreConfig
from tensorcast.proto.config.v1 import global_store_config_pb2 as gsc_pb


@dataclass
class GlobalStoreInstance:
    id: str
    pid: int
    address: str
    listen_host: str | None
    listen_port: int | None
    metrics_port: int | None
    config_path: Path
    logs_dir: Path
    cluster_token: str | None
    db_file: str | None
    owner: bool


def _normalize_exit_code(return_code: int) -> int:
    if return_code < 0:
        return 128 + (-return_code)
    return return_code


def _extract_config_path(cmd: Any) -> str | None:
    if not isinstance(cmd, list):
        return None
    for idx, item in enumerate(cmd):
        if not isinstance(item, str):
            continue
        if item == "--config" and idx + 1 < len(cmd):
            value = cmd[idx + 1]
            return value if isinstance(value, str) and value else None
        if item.startswith("--config="):
            value = item.split("=", 1)[1].strip()
            return value or None
    return None


def _process_group_contains_global_store(pgid: int, *, config_path: str | None) -> bool:
    if pgid <= 0:
        return False
    wanted_cfg = None
    if config_path:
        with contextlib.suppress(Exception):
            wanted_cfg = str(Path(config_path).expanduser().resolve())
        wanted_cfg = wanted_cfg or config_path
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            pid = int(proc.info.get("pid") or 0)
        except Exception:
            continue
        if pid <= 0:
            continue
        try:
            if os.getpgid(pid) != pgid:
                continue
        except ProcessLookupError:
            continue
        cmdline = proc.info.get("cmdline") or []
        if not isinstance(cmdline, list) or not cmdline:
            continue
        cmd_str = " ".join(str(part) for part in cmdline)
        if "tensorcast.global_store" not in cmd_str:
            continue
        if wanted_cfg and wanted_cfg not in cmd_str:
            continue
        return True
    return False


def _current_global_session_id() -> str | None:
    path = current_global_session_path()
    if not path.exists():
        return None
    try:
        data = path.read_text(encoding="utf-8").strip()
        return data or None
    except Exception:
        return None


@contextlib.contextmanager
def _locked(inst: GlobalSession):
    with (
        file_lock(runtime_lock_path()),
        file_lock(global_store_lock_path()),
        file_lock(inst.pids_lock),
    ):
        yield


def _load_runtime_global_state() -> dict[str, Any] | None:
    try:
        state = read_runtime_state(runtime_state_path())
    except Exception:
        return None
    gs_state = state.get("global_store") if isinstance(state, dict) else None
    return gs_state if isinstance(gs_state, dict) else None


def _resolve_config(
    *,
    inst: GlobalSession,
    config_path: Path | None,
    cluster_token: str,
    listen_host: str | None,
    listen_port: int | None,
    metrics_port: int | None,
) -> tuple[gsc_pb.GlobalStoreConfig, Path]:
    """Load or build a GlobalStoreConfig and persist it under the session."""

    pb: gsc_pb.GlobalStoreConfig | None = None
    cfg_source: Path | None = None
    if config_path is not None:
        cfg_source = Path(config_path)
    else:
        discovered = discover_global_store_config()
        if discovered:
            cfg_source = Path(discovered)

    if cfg_source is None:
        raise ServiceError(
            "No Global Store config found. Provide --config or set "
            "TENSORCAST_GLOBAL_STORE_CONFIG. Expected examples/config/global_store_config.yaml "
            "to be available in the repo or packaged wheel."
        )

    try:
        pb = GlobalStoreConfig.load_proto_from_file(str(cfg_source))
    except Exception as exc:  # noqa: BLE001
        raise ServiceError(
            f"Failed to load Global Store config from {cfg_source}: {exc}"
        ) from exc

    assert pb is not None

    if listen_host is not None:
        pb.server.listen.host = listen_host
    if listen_port is not None:
        pb.server.listen.port = max(0, int(listen_port))
    if not pb.server.listen.host:
        pb.server.listen.host = "0.0.0.0"
    if metrics_port is not None:
        pb.server.metrics_port = max(0, int(metrics_port))
    if not pb.database.db_file:
        pb.database.db_file = str(inst.root / "global_store.duckdb")
    if not pb.observability.logging.file:
        pb.observability.logging.file = str(inst.logs / "global_store.out")
    if not pb.meta.schema_version:
        pb.meta.schema_version = "v1"
    if not pb.meta.description:
        pb.meta.description = "generated-by-cli"
    pb.meta.cluster_token = cluster_token

    listen_host_eff = pb.server.listen.host or "0.0.0.0"
    if pb.server.listen.port <= 0:
        pb.server.listen.port = select_free_port(
            preferred=50051, host=listen_host_eff, probe_span=32
        )
    if pb.server.metrics_port <= 0:
        pb.server.metrics_port = select_free_port(
            preferred=8000, host=listen_host_eff, probe_span=32
        )

    effective_cfg_path = inst.session / "effective_global_store_config.yaml"
    effective_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    dump_global_store_config(pb, effective_cfg_path)
    return pb, effective_cfg_path


def _build_instance_from_health(
    *,
    inst: GlobalSession,
    gs_state: dict[str, Any] | None,
    health: GlobalStoreHealth,
    owner: bool,
) -> GlobalStoreInstance:
    pid = int(gs_state.get("pid", 0)) if isinstance(gs_state, dict) else None
    db_file = None
    if isinstance(gs_state, dict):
        db_file = gs_state.get("db_file") or None
    db_file = db_file or health.db_file
    if health.advertise_host and health.advertise_port:
        address = f"{health.advertise_host}:{health.advertise_port}"
    elif health.listen_host and health.listen_port:
        address = f"{health.listen_host}:{health.listen_port}"
    else:
        address = health.address
    return GlobalStoreInstance(
        id=inst.id,
        pid=int(pid or 0),
        address=address,
        listen_host=health.listen_host,
        listen_port=health.listen_port,
        metrics_port=health.metrics_port,
        config_path=inst.session / "effective_global_store_config.yaml",
        logs_dir=inst.logs,
        cluster_token=health.cluster_token
        or (gs_state.get("cluster_token") if isinstance(gs_state, dict) else None),
        db_file=db_file,
        owner=owner,
    )


def start_global_store(
    *,
    config_path: Path | None = None,
    cluster_token: str | None = None,
    session_id: str | None = None,
    listen_host: str | None = None,
    listen_port: int | None = None,
    metrics_port: int | None = None,
    to_console: bool = False,
    fate_share: bool = True,
    blocking: bool = False,
    announce: bool = True,
) -> GlobalStoreInstance:
    """Start the Global Store (single-instance guarded).

    This entry point always waits for readiness; with blocking=True it remains
    attached until the process exits.
    """

    provided_cluster_token = cluster_token
    runtime_state_file = runtime_state_path()
    gs_state_hint = _load_runtime_global_state()
    hinted_token = cluster_token or (
        gs_state_hint.get("cluster_token") if isinstance(gs_state_hint, dict) else None
    )
    cluster_token = load_or_create_cluster_token(hinted_token)
    existing_gid: str | None = None
    existing_addr: str | None = None
    state_token: str | None = None
    inst: GlobalSession | None = None
    gs_state: dict[str, Any] | None = None

    with file_lock(runtime_lock_path()), file_lock(global_store_lock_path()):
        runtime_state = read_runtime_state(runtime_state_file)
        gs_state = (
            runtime_state.get("global_store")
            if isinstance(runtime_state, dict)
            else None
        )
        existing_gid = (
            gs_state.get("session_id") if isinstance(gs_state, dict) else None
        )
        existing_addr = gs_state.get("address") if isinstance(gs_state, dict) else None
        state_token = (
            gs_state.get("cluster_token") if isinstance(gs_state, dict) else None
        )
        if state_token:
            if provided_cluster_token and state_token != provided_cluster_token:
                raise ServiceError(
                    "Global Store cluster token mismatch; clear ~/.tensorcast/runtime/cluster_token "
                    "if you intend to replace the existing cluster."
                )
            cluster_token = state_token
        if session_id and existing_gid and existing_addr and session_id != existing_gid:
            raise ServiceError(
                f"Global Store already running under session {existing_gid}"
            )

        inst = global_session_paths(session_id or existing_gid)
        with file_lock(inst.pids_lock):
            health = (
                ping_global_store(existing_addr, timeout=1.0) if existing_addr else None
            )
            if health:
                if (
                    cluster_token
                    and health.cluster_token
                    and health.cluster_token != cluster_token
                ):
                    raise ServiceError(
                        "Global Store is running with a different cluster token; "
                        "clear ~/.tensorcast/runtime/cluster_token to reset."
                    )
                set_current_global_session_id(inst.id)
                return _build_instance_from_health(
                    inst=inst,
                    gs_state=gs_state,
                    health=health,
                    owner=bool(gs_state.get("owner"))
                    if isinstance(gs_state, dict)
                    else False,
                )

            prune_process_records(
                pids_path=inst.pids_json,
                predicate=(lambda entry: True),
                lock_path=None,
            )
            with contextlib.suppress(FileNotFoundError):
                inst.state_json.unlink()
            clear_runtime_global_store(runtime_state_file, preserve_cluster_token=True)

    # Prepare effective config and launch (outside locks)
    assert inst is not None
    started_at = time.time()
    pb_cfg, effective_cfg_path = _resolve_config(
        inst=inst,
        config_path=config_path,
        cluster_token=cluster_token,
        listen_host=listen_host,
        listen_port=listen_port,
        metrics_port=metrics_port,
    )

    so_path = inst.logs / "global_store.out"
    se_path = inst.logs / "global_store.err"
    so = open_log_binary(so_path)
    se = open_log_binary(se_path)
    args = [
        "uv",
        "run",
        "-m",
        "tensorcast.global_store",
        "--config",
        str(effective_cfg_path),
    ]
    try:
        preexec_fn = preexec_fate_sharing if fate_share else preexec_detached
        use_pipes = blocking or fate_share
        stdout_target = subprocess.PIPE if use_pipes else so
        stderr_target = subprocess.PIPE if use_pipes else se
        proc = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=stdout_target,
            stderr=stderr_target,
            cwd=inst.session,
            preexec_fn=preexec_fn,
            close_fds=True,
        )
    except Exception:
        with contextlib.suppress(Exception):
            so.close()
        with contextlib.suppress(Exception):
            se.close()
        raise

    fingerprint = instance_fingerprint(proc.pid)
    with _locked(inst):
        register_process(
            inst_id=inst.id,
            pids_path=inst.pids_json,
            proc=proc,
            role="global_store",
            stdout_path=so_path,
            stderr_path=se_path,
            lock_path=None,
        )

    if not use_pipes:
        with contextlib.suppress(Exception):
            so.close()
        with contextlib.suppress(Exception):
            se.close()
        log_threads: list[Any] = []
    else:
        sinks_out: list[Any] = []
        sinks_err: list[Any] = []
        if so is not None:
            sinks_out.append(so)
        if se is not None:
            sinks_err.append(se)
        if to_console:
            sinks_out.append(sys.stdout)
            sinks_err.append(sys.stderr)
        log_threads = start_log_threads(proc, sinks_out, sinks_err)

    try:
        ensure_process_started(proc, inst.logs, startup_grace=1.0)
    except ServiceError:
        _cleanup_failed_start(proc, log_threads, inst, so, se)
        raise

    connect_host = resolve_connect_host(pb_cfg.server.listen.host)
    connect_port = int(pb_cfg.server.listen.port or 0)
    address = f"{connect_host}:{connect_port}"
    health = wait_for_global_store(address, timeout=None, proc=proc)
    if health is None:
        retcode = proc.poll()
        _cleanup_failed_start(proc, log_threads, inst, so, se)
        hint = f" See logs under {inst.logs}"
        if retcode is not None:
            raise ServiceError(
                f"Global Store exited with code {retcode} during startup.{hint}"
            )
        raise ServiceError("Global Store exited during startup." + hint)

    if health and health.cluster_token:
        cluster_token = health.cluster_token

    listen_host_eff = (
        health.listen_host if health else pb_cfg.server.listen.host or "0.0.0.0"
    )
    listen_port_eff = (
        health.listen_port if health else int(pb_cfg.server.listen.port or 0) or None
    )
    advertise_host_eff = (
        health.advertise_host if health and health.advertise_host else None
    )
    advertise_port_eff = (
        health.advertise_port if health and health.advertise_port else None
    )
    metrics_port_eff = (
        health.metrics_port if health else int(pb_cfg.server.metrics_port or 0) or None
    )
    db_file = (
        (health.db_file if health else None)
        or pb_cfg.database.db_file
        or str(inst.root / "global_store.duckdb")
    )
    if listen_host_eff is None:
        raise ServiceError("Global Store listen host unavailable after configuration")
    advertise_host_eff = advertise_host_eff or listen_host_eff
    advertise_port_eff = advertise_port_eff or listen_port_eff
    address_eff = (
        f"{advertise_host_eff}:{advertise_port_eff}"
        if advertise_host_eff and advertise_port_eff is not None
        else listen_host_eff
    )
    session_state = {
        "schema_version": 1,
        "session_id": inst.id,
        "started_at": started_at,
        "global_store": {
            "pid": int(proc.pid),
            "address": address_eff,
            "listen_host": listen_host_eff,
            "listen_port": listen_port_eff,
            "advertise_host": advertise_host_eff,
            "advertise_port": advertise_port_eff,
            "metrics_port": metrics_port_eff,
            "config_path": str(effective_cfg_path),
            "db_file": db_file,
            "cluster_token": cluster_token,
        },
        "logs_dir": str(inst.logs),
    }
    with contextlib.suppress(Exception):
        from tensorcast import __version__ as _tc_version

        session_state["version"] = _tc_version

    with _locked(inst):
        atomic_write_json(inst.state_json, session_state)
        update_runtime_global_store(
            path=runtime_state_file,
            session_id=inst.id,
            pid=int(proc.pid),
            address=address_eff,
            listen_host=listen_host_eff,
            listen_port=listen_port_eff,
            metrics_port=metrics_port_eff,
            db_file=db_file,
            cluster_token=cluster_token,
            owner=True,
            fingerprint=fingerprint,
        )
        set_current_global_session_id(inst.id)

    if cluster_token:
        with contextlib.suppress(Exception):
            load_or_create_cluster_token(cluster_token)

    instance = GlobalStoreInstance(
        id=inst.id,
        pid=int(proc.pid),
        address=address_eff,
        listen_host=listen_host_eff,
        listen_port=listen_port_eff,
        metrics_port=metrics_port_eff,
        config_path=effective_cfg_path,
        logs_dir=inst.logs,
        cluster_token=cluster_token,
        db_file=db_file,
        owner=True,
    )
    if blocking:

        def _cleanup() -> None:
            with contextlib.suppress(Exception):
                stop_global_store(session_id=inst.id, quiet=True)

        register_blocking_cleanup(_cleanup)

        ret = proc.wait()
        join_threads(log_threads)
        click.echo(f"global store exited with code {ret}")
        if ret != 0:
            raise click.exceptions.Exit(code=_normalize_exit_code(ret))
        return instance

    if announce:
        click.echo(
            f"Global Store started (session={inst.id}) at {address_eff}. Logs: {inst.logs}"
        )
    return instance


def stop_global_store(
    session_id: str | None = None, *, force: bool = False, quiet: bool = False
) -> None:
    gid = session_id or _current_global_session_id()
    if not gid:
        raise ServiceError(
            "No global store session id provided and none set as current"
        )
    inst = global_session_paths(gid)
    runtime_state_file = runtime_state_path()
    if not inst.pids_json.exists():
        click.echo(f"No pids.json found for global store session {gid}")
        with contextlib.suppress(Exception), file_lock(runtime_lock_path()):
            state = read_runtime_state(runtime_state_file)
            gs_state = state.get("global_store")
            if isinstance(gs_state, dict) and gs_state.get("session_id") == gid:
                clear_runtime_global_store(runtime_state_file)
        return

    with _locked(inst):
        data = read_json_default(inst.pids_json, {"processes": []})
        procs = data.get("processes", [])
        targets: list[int] = []
        for proc in reversed(procs):
            pid = int(proc.get("pid", 0))
            if pid <= 0:
                continue
            config_path = _extract_config_path(proc.get("cmd"))
            pgid = None
            try:
                pgid = os.getpgid(pid)
            except ProcessLookupError:
                pgid = pid
                try:
                    os.killpg(pgid, 0)
                except ProcessLookupError:
                    continue
                if not _process_group_contains_global_store(
                    pgid, config_path=config_path
                ):
                    continue
            targets.append(pid)
            if force:
                kill_force(int(pgid))
            else:
                if not kill_gracefully(int(pgid), grace=10.0):
                    kill_force(int(pgid))
        if targets:
            prune_process_records(
                pids_path=inst.pids_json,
                predicate=lambda entry: int(entry.get("pid", 0)) in targets,
                lock_path=None,
            )
        with contextlib.suppress(Exception):
            state = read_runtime_state(runtime_state_file)
            gs_state = state.get("global_store")
            if isinstance(gs_state, dict) and gs_state.get("session_id") == gid:
                clear_runtime_global_store(runtime_state_file)
        clear_current_global_session_if_matches(gid)
    if not quiet:
        click.echo(f"Stopped global store session {gid}")


def global_store_logs(
    *, session_id: str | None = None, stderr: bool = False, follow: bool = False
) -> None:
    gid = session_id or _current_global_session_id()
    if not gid:
        raise ServiceError(
            "No global store session id provided and none set as current"
        )
    inst = global_session_paths(gid)
    log_file = inst.logs / ("global_store.err" if stderr else "global_store.out")
    if not log_file.exists():
        click.echo(f"No log file found for global store session: {log_file}")
        return
    click.echo(f"==> {log_file} <==")
    if not follow:
        try:
            with open(log_file, "rb") as f:
                data = f.read()
                lines = data.splitlines()[-200:]
                for ln in lines:
                    click.echo(ln.decode("utf-8", errors="replace"))
        except Exception as e:  # noqa: BLE001
            raise ServiceError(f"Failed to read logs: {e}") from e
        return
    try:
        with open(log_file, "rb") as f:
            f.seek(0, os.SEEK_END)
            while True:
                line = f.readline()
                if line:
                    click.echo(line.decode("utf-8", errors="replace"), nl=False)
                else:
                    time.sleep(0.2)
    except KeyboardInterrupt:
        return
    except Exception as e:  # noqa: BLE001
        raise ServiceError(f"Failed to tail logs: {e}") from e


def _cleanup_failed_start(
    proc: subprocess.Popen[Any],
    log_threads: list[Any],
    inst: GlobalSession,
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
    with _locked(inst):
        prune_process_records(
            pids_path=inst.pids_json,
            predicate=lambda entry: int(entry.get("pid", 0)) == int(proc.pid),
            lock_path=None,
        )
        with contextlib.suppress(FileNotFoundError):
            inst.state_json.unlink()
        with contextlib.suppress(Exception):
            clear_runtime_global_store(runtime_state_path())
    with contextlib.suppress(Exception):
        clear_current_global_session_if_matches(inst.id)


__all__ = [
    "GlobalStoreInstance",
    "global_store_logs",
    "start_global_store",
    "stop_global_store",
]
