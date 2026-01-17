#  Copyright (c) 2025-2026, TensorCast Team.

"""Session and path management utilities for TensorCast CLI.

This module centralizes filesystem layout, current-session bookkeeping, and
helpers to construct per-session directories. Runtime state is host-scoped to
avoid collisions when multiple machines share the same filesystem.
"""

from __future__ import annotations

import contextlib
import os
import socket
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from tensorcast.cli_utils.filesys import read_json_locked, write_text_atomic

HOME_DIRNAME = ".tensorcast"
HOSTS_DIRNAME = "hosts"
ENV_HOME_VAR = "TENSORCAST_HOME"


def _home_base() -> Path:
    override = os.environ.get(ENV_HOME_VAR)
    if override:
        return Path(override).expanduser()
    return Path.home() / HOME_DIRNAME


def home_dir() -> Path:
    p = _home_base()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sanitize_component(value: str) -> str:
    safe = value.replace(os.sep, "_")
    if os.altsep:
        safe = safe.replace(os.altsep, "_")
    return safe


def _read_machine_id() -> str | None:
    try:
        path = Path("/etc/machine-id")
        if path.exists():
            data = path.read_text(encoding="utf-8").strip()
            return data or None
    except Exception:
        return None
    return None


@lru_cache(maxsize=1)
def _host_id() -> str:
    hostname = socket.gethostname().strip() or "unknown"
    machine_id = _read_machine_id()
    raw = f"{hostname}-{machine_id}" if machine_id else hostname
    return _sanitize_component(raw)


def host_root() -> Path:
    root = home_dir() / HOSTS_DIRNAME / _host_id()
    root.mkdir(parents=True, exist_ok=True)
    return root


def runtime_root() -> Path:
    root = host_root() / "runtime"
    root.mkdir(parents=True, exist_ok=True)
    return root


def locks_dir() -> Path:
    root = host_root() / "locks"
    root.mkdir(parents=True, exist_ok=True)
    return root


def runtime_state_path() -> Path:
    return runtime_root() / "state.json"


def runtime_lock_path() -> Path:
    path = locks_dir() / "runtime.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)
    return path


def global_store_lock_path() -> Path:
    path = locks_dir() / "global_store.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)
    return path


def sessions_root() -> Path:
    p = host_root() / "sessions"
    p.mkdir(parents=True, exist_ok=True)
    return p


def global_sessions_root() -> Path:
    p = host_root() / "global_sessions"
    p.mkdir(parents=True, exist_ok=True)
    return p


def current_session_path() -> Path:
    return host_root() / "current_session"


def current_global_session_path() -> Path:
    return host_root() / "current_global_session"


def _read_text_optional(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        data = path.read_text(encoding="utf-8").strip()
        return data or None
    except Exception:
        return None


def get_current_session_id() -> str | None:
    return _read_text_optional(current_session_path())


def set_current_session_id(session_id: str) -> None:
    p = current_session_path()
    write_text_atomic(p, session_id, mode=0o600)


def set_current_global_session_id(session_id: str) -> None:
    p = current_global_session_path()
    write_text_atomic(p, session_id, mode=0o600)


def clear_current_session_if_matches(sid: str) -> None:
    p = current_session_path()
    if not p.exists():
        return
    try:
        current = p.read_text(encoding="utf-8").strip()
    except Exception:
        return
    if current == sid:
        with contextlib.suppress(Exception):
            p.unlink()


def clear_current_global_session_if_matches(session_id: str) -> None:
    p = current_global_session_path()
    if not p.exists():
        return
    try:
        current = p.read_text(encoding="utf-8").strip()
    except Exception:
        return
    if current == session_id:
        with contextlib.suppress(Exception):
            p.unlink()


def list_sessions() -> list[str]:
    d = sessions_root()
    return sorted([ch.name for ch in d.iterdir() if ch.is_dir()])


def _now_ts() -> float:
    import time

    return time.time()


def _now_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _rand4() -> str:
    import random

    return f"{random.randrange(0, 65536):04x}"


def gen_session_id() -> str:
    return f"{_now_id()}-{_rand4()}"


def gen_global_session_id() -> str:
    return f"gs-{_now_id()}-{_rand4()}"


@dataclass(frozen=True)
class DaemonSession:
    id: str
    root: Path
    session: Path
    logs: Path
    pids_json: Path
    meta_json: Path
    session_state_json: Path
    effective_config_path: Path
    pids_lock: Path
    address: str | None = None
    p2p_address: str | None = None


@dataclass(frozen=True)
class GlobalSession:
    id: str
    root: Path
    session: Path
    logs: Path
    pids_json: Path
    state_json: Path
    pids_lock: Path
    # Layout (schema_version=1):
    # - session/state.json: Global Store session metadata (pid, address, ports, cluster_token)
    # - logs/global_store.out|err: stdout/stderr streams
    # - pids.json: append-only process records with role="global_store"


def session_paths(session_id: str | None = None) -> DaemonSession:
    sid = session_id or gen_session_id()
    root = sessions_root() / sid
    session = root / "session"
    logs = root / "logs"
    session.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    return DaemonSession(
        id=sid,
        root=root,
        session=session,
        logs=logs,
        pids_json=root / "pids.json",
        meta_json=root / "meta.json",
        session_state_json=session / "session_state.json",
        effective_config_path=session / "effective_daemon_config.yaml",
        pids_lock=root / "pids.lock",
        address=None,
        p2p_address=None,
    )


def global_session_paths(session_id: str | None = None) -> GlobalSession:
    gid = session_id or gen_global_session_id()
    root = global_sessions_root() / gid
    session = root / "session"
    logs = root / "logs"
    session.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    return GlobalSession(
        id=gid,
        root=root,
        session=session,
        logs=logs,
        pids_json=root / "pids.json",
        state_json=session / "state.json",
        pids_lock=root / "pids.lock",
    )


def get_session_address(session_id: str | None = None) -> str | None:
    """Return host:port of a session from meta.json (default: current)."""
    sid = session_id or get_current_session_id()
    if not sid:
        return None
    try:
        meta = read_json_locked(session_paths(sid).meta_json)
        addr = meta.get("address")
        return addr if isinstance(addr, str) else None
    except Exception:
        return None


def daemon_state_dir(session_id: str) -> Path:
    """Return the daemon session state directory without creating the session dir."""
    return sessions_root() / session_id / "session"


def discover_local_handle_socket_path(session_id: str | None = None) -> Path | None:
    """Discover the default local handle socket path for a daemon session."""
    sid = session_id or get_current_session_id()
    if not sid:
        return None
    state_dir = daemon_state_dir(sid)
    if not state_dir.exists():
        return None
    return state_dir / "local_handle.sock"
