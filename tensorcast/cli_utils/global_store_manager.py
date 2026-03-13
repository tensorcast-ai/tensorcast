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
    dump_global_store_config,
    select_free_port,
    write_cluster_token,
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
from tensorcast.global_store.launch_config import (
    apply_global_store_proto_defaults,
    load_global_store_config_with_overrides,
    resolve_global_store_config_path,
)
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


def _global_store_launch_env() -> dict[str, str]:
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / "tensorcast" / "__init__.py").exists():
        existing = env.get("PYTHONPATH", "")
        parts = [str(repo_root)]
        if existing:
            parts.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


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


def _normalize_config_paths(config_paths: set[str]) -> set[str]:
    normalized: set[str] = set()
    for cfg in config_paths:
        if not cfg:
            continue
        resolved = None
        with contextlib.suppress(Exception):
            resolved = str(Path(cfg).expanduser().resolve())
        normalized.add(resolved or cfg)
    return normalized


def _find_global_store_pgids(config_paths: set[str]) -> set[int]:
    normalized = _normalize_config_paths(config_paths)
    if not normalized:
        return set()
    pgids: set[int] = set()
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            pid = int(proc.info.get("pid") or 0)
        except Exception:
            continue
        if pid <= 0:
            continue
        cmdline = proc.info.get("cmdline") or []
        if not isinstance(cmdline, list) or not cmdline:
            continue
        cmd_str = " ".join(str(part) for part in cmdline)
        if "tensorcast.global_store" not in cmd_str:
            continue
        if not any(cfg in cmd_str for cfg in normalized):
            continue
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            continue
        if pgid > 0:
            pgids.add(pgid)
    return pgids


def _current_global_session_id() -> str | None:
    path = current_global_session_path()
    if not path.exists():
        return None
    try:
        data = path.read_text(encoding="utf-8").strip()
        return data or None
    except Exception:
        return None


def _has_live_recorded_global_store(inst: GlobalSession) -> bool:
    records = read_json_default(inst.pids_json, {"processes": []})
    processes = records.get("processes", [])
    if not isinstance(processes, list):
        return False
    for entry in processes:
        if not isinstance(entry, dict):
            continue
        pid = int(entry.get("pid", 0))
        if pid <= 0 or not psutil.pid_exists(pid):
            continue
        with contextlib.suppress(Exception):
            cmdline = psutil.Process(pid).cmdline()
            cmd = " ".join(str(part) for part in cmdline)
            if "tensorcast.global_store" in cmd:
                return True
    return False


@contextlib.contextmanager
def _locked(inst: GlobalSession):
    with (
        file_lock(runtime_lock_path()),
        file_lock(global_store_lock_path()),
        file_lock(inst.pids_lock),
    ):
        yield


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

    try:
        cfg_source = resolve_global_store_config_path(config_path)
    except FileNotFoundError as exc:
        raise ServiceError(str(exc)) from exc

    try:
        _, pb = load_global_store_config_with_overrides(
            cfg_source,
            listen_host=listen_host,
            listen_port=listen_port,
            metrics_port=metrics_port,
        )
    except Exception as exc:  # noqa: BLE001
        raise ServiceError(
            f"Failed to load Global Store config from {cfg_source}: {exc}"
        ) from exc

    apply_global_store_proto_defaults(
        pb,
        default_listen_host="0.0.0.0",
        default_log_file=str(inst.logs / "global_store.out"),
        default_schema_version="v1",
        default_description="generated-by-cli",
        cluster_token=cluster_token,
    )

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
    existing_gid: str | None = None
    existing_addr: str | None = None
    inst: GlobalSession | None = None
    gs_state: dict[str, Any] | None = None
    reused_instance: GlobalStoreInstance | None = None
    reused_cluster_token: str | None = None

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
        if session_id and existing_gid and existing_addr and session_id != existing_gid:
            raise ServiceError(
                f"Global Store already running under session {existing_gid}"
            )

        inst = global_session_paths(session_id or existing_gid)
        with file_lock(inst.pids_lock):
            health = (
                ping_global_store(existing_addr, timeout=1.0) if existing_addr else None
            )
            if not health and isinstance(gs_state, dict):
                listen_host_state = gs_state.get("listen_host")
                listen_port_state = gs_state.get("listen_port")
                if listen_host_state and listen_port_state:
                    fallback_addr = (
                        f"{resolve_connect_host(str(listen_host_state))}:"
                        f"{int(listen_port_state)}"
                    )
                    if fallback_addr != existing_addr:
                        health = ping_global_store(fallback_addr, timeout=1.0)
            if health:
                if (
                    provided_cluster_token
                    and health.cluster_token
                    and health.cluster_token != provided_cluster_token
                ):
                    raise ServiceError(
                        "Global Store is running with a different cluster token; "
                        "expected the explicitly requested token."
                    )
                set_current_global_session_id(inst.id)
                reused_instance = _build_instance_from_health(
                    inst=inst,
                    gs_state=gs_state,
                    health=health,
                    owner=bool(gs_state.get("owner"))
                    if isinstance(gs_state, dict)
                    else False,
                )
                reused_cluster_token = reused_instance.cluster_token
            else:
                if _has_live_recorded_global_store(inst):
                    raise ServiceError(
                        "Global Store process appears to be running but is not healthy via "
                        "recorded endpoints. Refusing to start a second instance on the "
                        "same session. Run 'tensorcast-cli global stop --force' and retry."
                    )

                prune_process_records(
                    pids_path=inst.pids_json,
                    predicate=(lambda entry: True),
                    lock_path=None,
                )
                with contextlib.suppress(FileNotFoundError):
                    inst.state_json.unlink()
                clear_runtime_global_store(
                    runtime_state_file, preserve_cluster_token=False
                )

    # Prepare effective config and launch (outside locks)
    assert inst is not None
    if reused_instance is not None:
        if reused_cluster_token:
            with contextlib.suppress(Exception):
                write_cluster_token(reused_cluster_token)
        return reused_instance

    cluster_token = write_cluster_token(provided_cluster_token)
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
    # Launch Global Store with the current interpreter so the serving process is
    # the direct child protected by PR_SET_PDEATHSIG. Wrapping with `uv run`
    # leaves an extra process layer that can orphan the actual server when the
    # parent dies abruptly (for example via SIGKILL).
    args = [
        sys.executable,
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
        launch_env = _global_store_launch_env()
        proc = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=stdout_target,
            stderr=stderr_target,
            cwd=inst.session,
            env=launch_env,
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
        cluster_token = write_cluster_token(health.cluster_token)

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
    db_file = (health.db_file if health and health.db_file else None) or (
        pb_cfg.database.db_file or None
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
        target_pgids: set[int] = set()
        config_paths: set[str] = set()
        for proc in reversed(procs):
            pid = int(proc.get("pid", 0))
            if pid <= 0:
                continue
            config_path = _extract_config_path(proc.get("cmd"))
            if config_path:
                config_paths.add(config_path)
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
            if pgid is not None and int(pgid) > 0:
                target_pgids.add(int(pgid))

        extra_pgids = _find_global_store_pgids(config_paths) - target_pgids
        for pgid in sorted(extra_pgids):
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
