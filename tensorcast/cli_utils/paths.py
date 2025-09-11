#  Copyright (c) 2025, TensorCast Team.

"""Session and path management utilities for TensorCast CLI.

This module centralizes filesystem layout, current-session bookkeeping, and
helpers to construct per-session directories.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .filesys import read_json_locked, write_text_atomic

HOME_DIRNAME = ".tensorcast"


def home_dir() -> Path:
    return Path.home() / HOME_DIRNAME


def sessions_root() -> Path:
    p = home_dir() / "sessions"
    p.mkdir(parents=True, exist_ok=True)
    return p


def current_session_path() -> Path:
    return home_dir() / "current_session"


def get_current_session_id() -> str | None:
    p = current_session_path()
    if not p.exists():
        return None
    try:
        data = p.read_text(encoding="utf-8").strip()
        return data or None
    except Exception:
        return None


def set_current_session_id(session_id: str) -> None:
    p = current_session_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(p, session_id)
    with contextlib.suppress(Exception):
        os.chmod(p, 0o600)


def clear_current_session_if_matches(sid: str) -> None:
    p = current_session_path()
    if not p.exists():
        return
    try:
        current = p.read_text(encoding="utf-8").strip()
        if current == sid:
            with contextlib.suppress(Exception):
                p.unlink()
    except Exception:
        pass


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


@dataclass(frozen=True)
class DaemonSession:
    id: str
    root: Path
    session: Path
    logs: Path
    pids_json: Path
    meta_json: Path
    address: str | None = None
    p2p_address: str | None = None


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
        address=None,
        p2p_address=None,
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
