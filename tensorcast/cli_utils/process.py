#  Copyright (c) 2025, TensorCast Team.

"""Process helpers and persistent runtime state contracts.

Schema (authoritative, versioned):

Runtime state (schema_version=1)
```json
{
  "schema_version": 1,
  "daemon": {
    "session_id": "20240604T120000Z",
    "pid": 12346,
    "address": "127.0.0.1:50052",
    "p2p_address": "127.0.0.1:50053",
    "owner": true,
    "updated_at": 1717500001.0,
    "instance_fingerprint": {"host_id": "...", "boot_id": "...", "pid": 12346}
  },
  "global_store": {
    "session_id": "gs-20240604T1159Z",
    "pid": 22345,
    "address": "127.0.0.1:50051",
    "listen_host": "127.0.0.1",
    "listen_port": 50051,
    "metrics_port": 8000,
    "db_file": "/home/user/.tensorcast/global_sessions/.../global_store.duckdb",
    "cluster_token": "d6c3...e7",
    "owner": true,
    "updated_at": 1717500002.0,
    "instance_fingerprint": {"host_id": "...", "boot_id": "...", "pid": 22345}
  }
}
```

Session state (schema_version=1)
```json
{
  "schema_version": 1,
  "session_id": "20240604T120000Z",
  "ephemeral": false,
  "started_at": 1717500000.0,
  "daemon": {
    "pid": 12346,
    "address": "127.0.0.1:50052",
    "p2p_address": "127.0.0.1:50053",
    "config_path": "/path/to/effective_daemon_config.yaml",
    "binary": "/path/to/tensorcast_daemon"
  },
  "global_store": {
    "mode": "connect",
    "required": true,
    "address": "127.0.0.1:50051",
    "session": "gs-20240604T1159Z"
  },
  "logs_dir": "/home/user/.tensorcast/sessions/20240604/logs"
}
```
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import psutil

from tensorcast.cli_utils.errors import ServiceError
from tensorcast.cli_utils.filesys import write_text_atomic
from tensorcast.cli_utils.proc import pump

RUNTIME_SCHEMA_VERSION = 1
SESSION_STATE_SCHEMA_VERSION = 1
PIDS_SCHEMA_VERSION = 1


@contextlib.contextmanager
def file_lock(path: Path) -> Iterator[Path]:
    """Acquire an exclusive flock on the given path, creating it if missing."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        with contextlib.suppress(OSError):
            os.chmod(path, 0o600)
        yield path
    finally:
        with contextlib.suppress(Exception):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def atomic_write_json(path: Path, data: dict[str, Any], mode: int = 0o600) -> None:
    serialized = json.dumps(data, indent=2, sort_keys=True)
    write_text_atomic(path, serialized, mode=mode)


def read_runtime_state(path: Path) -> dict[str, Any]:
    data = read_json_default(path, {"schema_version": RUNTIME_SCHEMA_VERSION})
    _ensure_schema_version(data, RUNTIME_SCHEMA_VERSION)
    return data


def write_runtime_state(path: Path, state: dict[str, Any]) -> dict[str, Any]:
    payload = dict(state)
    payload["schema_version"] = RUNTIME_SCHEMA_VERSION
    atomic_write_json(path, payload)
    return payload


def update_runtime_daemon(
    *,
    path: Path,
    session_id: str,
    pid: int,
    address: str | None,
    p2p_address: str | None,
    owner: bool,
    fingerprint: dict[str, Any],
    cluster_token: str | None = None,
) -> dict[str, Any]:
    state = read_runtime_state(path)
    daemon_state = {
        "session_id": session_id,
        "pid": pid,
        "address": address,
        "p2p_address": p2p_address,
        "owner": owner,
        "updated_at": time.time(),
        "instance_fingerprint": fingerprint,
    }
    if cluster_token:
        daemon_state["cluster_token"] = cluster_token
    state["daemon"] = daemon_state
    return write_runtime_state(path, state)


def clear_runtime_daemon(path: Path) -> dict[str, Any]:
    state = read_runtime_state(path)
    state.pop("daemon", None)
    return write_runtime_state(path, state)


def update_runtime_global_store(
    *,
    path: Path,
    session_id: str,
    pid: int | None,
    address: str | None,
    listen_host: str | None,
    listen_port: int | None,
    metrics_port: int | None,
    db_file: str | None,
    cluster_token: str | None,
    owner: bool,
    fingerprint: dict[str, Any] | None,
) -> dict[str, Any]:
    state = read_runtime_state(path)
    gs_state = {
        "session_id": session_id,
        "pid": pid,
        "address": address,
        "listen_host": listen_host,
        "listen_port": listen_port,
        "metrics_port": metrics_port,
        "db_file": db_file,
        "cluster_token": cluster_token,
        "owner": owner,
        "updated_at": time.time(),
        "instance_fingerprint": fingerprint,
    }
    # Drop None values to keep payload compact while retaining cluster_token if set.
    state["global_store"] = {k: v for k, v in gs_state.items() if v is not None}
    return write_runtime_state(path, state)


def clear_runtime_global_store(
    path: Path, *, preserve_cluster_token: bool = True
) -> dict[str, Any]:
    state = read_runtime_state(path)
    existing = state.get("global_store") if isinstance(state, dict) else None
    token = None
    if preserve_cluster_token and isinstance(existing, dict):
        token = existing.get("cluster_token")
    if token:
        state["global_store"] = {"cluster_token": token, "updated_at": time.time()}
    else:
        state.pop("global_store", None)
    return write_runtime_state(path, state)


def write_session_state(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    updated = dict(payload)
    updated["schema_version"] = SESSION_STATE_SCHEMA_VERSION
    atomic_write_json(path, updated)
    return updated


def read_session_state(path: Path) -> dict[str, Any]:
    data = read_json_default(path, {"schema_version": SESSION_STATE_SCHEMA_VERSION})
    _ensure_schema_version(data, SESSION_STATE_SCHEMA_VERSION)
    return data


def read_json_default(
    path: Path, default: dict[str, Any] | None = None
) -> dict[str, Any]:
    if not path.exists():
        return dict(default or {})
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:  # noqa: BLE001
        raise ServiceError(f"Invalid JSON in {path}: {exc}") from exc
    except Exception:
        return dict(default or {})


def _ensure_schema_version(
    payload: dict[str, Any], expected: int, *, allow_missing: bool = True
) -> None:
    schema_version = payload.get("schema_version")
    if schema_version is None and allow_missing:
        payload["schema_version"] = expected
        return
    if schema_version != expected:
        raise ServiceError(
            f"Expected schema_version={expected} but found {schema_version} in {payload}"
        )


def append_process_record(
    *,
    pids_path: Path,
    entry: dict[str, Any],
    session_id: str,
    lock_path: Path | None = None,
    schema_version: int = PIDS_SCHEMA_VERSION,
    dedupe: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Append or merge a process entry, keeping schema_version stable."""

    lock_ctx = (
        file_lock(lock_path) if lock_path is not None else contextlib.nullcontext()
    )
    with lock_ctx:
        data = read_json_default(
            pids_path, {"schema_version": schema_version, "session_id": session_id}
        )
        _ensure_schema_version(data, schema_version)
        processes = data.get("processes", [])
        if not isinstance(processes, list):
            processes = []
        key_fn = dedupe or (lambda proc: proc.get("pid"))
        existing: dict[Any, dict[str, Any]] = {}
        for proc in processes:
            try:
                existing[key_fn(proc)] = proc
            except Exception:
                continue
        try:
            existing[key_fn(entry)] = entry
        except Exception:
            existing[entry.get("pid")] = entry
        merged = [proc for proc in existing.values() if proc is not None]
        data["processes"] = merged
        data["session_id"] = session_id or data.get("session_id") or session_id
        atomic_write_json(pids_path, data)
        return data


def prune_process_records(
    *,
    pids_path: Path,
    predicate: Callable[[dict[str, Any]], bool],
    lock_path: Path | None = None,
    schema_version: int = PIDS_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Remove process entries matching predicate; no-op if file missing."""

    lock_ctx = (
        file_lock(lock_path) if lock_path is not None else contextlib.nullcontext()
    )
    with lock_ctx:
        if not pids_path.exists():
            return {"schema_version": schema_version, "processes": []}
        data = read_json_default(
            pids_path, {"schema_version": schema_version, "processes": []}
        )
        _ensure_schema_version(data, schema_version)
        processes = data.get("processes", [])
        if not isinstance(processes, list):
            processes = []
        retained = [proc for proc in processes if not predicate(proc)]
        data["processes"] = retained
        atomic_write_json(pids_path, data)
        return data


def is_process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        return psutil.pid_exists(pid)
    except Exception:
        return False


def read_boot_id() -> str | None:
    try:
        path = Path("/proc/sys/kernel/random/boot_id")
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    return None


def instance_fingerprint(pid: int | None = None) -> dict[str, Any]:
    """Return a fingerprint for the current instance (host + boot + pid)."""

    hostname = socket.gethostname()
    machine_id = None
    with contextlib.suppress(Exception):
        machine_id_path = Path("/etc/machine-id")
        if machine_id_path.exists():
            machine_id = machine_id_path.read_text(encoding="utf-8").strip()
    host_id = f"{hostname}|{machine_id}" if machine_id else hostname
    boot_id = read_boot_id() or "unknown"
    return {
        "host_id": host_id,
        "boot_id": boot_id,
        "pid": int(pid or os.getpid()),
    }


def register_process(
    *,
    inst_id: str,
    pids_path: Path,
    proc: subprocess.Popen[Any],
    role: str,
    stdout_path: Path,
    stderr_path: Path,
    lock_path: Path | None = None,
) -> dict[str, Any]:
    entry = {
        "role": role,
        "pid": int(proc.pid),
        "cmd": [str(x) for x in proc.args]
        if isinstance(proc.args, (list, tuple))
        else [str(proc.args)],
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "start_time": time.time(),
    }
    append_process_record(
        pids_path=pids_path,
        entry=entry,
        session_id=inst_id,
        lock_path=lock_path,
    )
    return entry


def spawn_log_thread(src, sinks: Iterable[Any]) -> threading.Thread:
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


def start_log_threads(
    proc: subprocess.Popen[Any], stdout_sinks: list[Any], stderr_sinks: list[Any]
) -> list[threading.Thread]:
    threads: list[threading.Thread] = []
    if proc.stdout is not None:
        threads.append(spawn_log_thread(proc.stdout, stdout_sinks))
    if proc.stderr is not None:
        threads.append(spawn_log_thread(proc.stderr, stderr_sinks))
    return threads


def join_threads(threads: list[threading.Thread], timeout: float = 1.0) -> None:
    if not threads:
        return
    deadline = time.monotonic() + max(timeout, 0.0)
    for thread in threads:
        remaining = max(0.0, deadline - time.monotonic())
        if remaining <= 0.0:
            break
        thread.join(remaining)


def ensure_process_started(
    proc: subprocess.Popen[Any], log_path: Path | None, startup_grace: float
) -> None:
    grace = max(startup_grace, 0.1)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        retcode = proc.poll()
        if retcode is not None:
            hint = f" Check logs at {log_path}" if log_path is not None else ""
            raise ServiceError(
                f"Process exited with code {retcode} during startup." + hint
            )
        time.sleep(0.05)
    retcode = proc.poll()
    if retcode is not None:
        hint = f" Check logs at {log_path}" if log_path is not None else ""
        raise ServiceError(f"Process exited with code {retcode} during startup." + hint)


__all__ = [
    "RUNTIME_SCHEMA_VERSION",
    "SESSION_STATE_SCHEMA_VERSION",
    "PIDS_SCHEMA_VERSION",
    "append_process_record",
    "atomic_write_json",
    "clear_runtime_daemon",
    "clear_runtime_global_store",
    "ensure_process_started",
    "file_lock",
    "instance_fingerprint",
    "is_process_alive",
    "join_threads",
    "prune_process_records",
    "register_process",
    "read_json_default",
    "read_runtime_state",
    "read_session_state",
    "spawn_log_thread",
    "start_log_threads",
    "update_runtime_daemon",
    "update_runtime_global_store",
    "write_runtime_state",
    "write_session_state",
]
