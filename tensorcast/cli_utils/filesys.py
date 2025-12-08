#  Copyright (c) 2025, TensorCast Team.

"""Filesystem helpers with locking and safe writes."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import tempfile
from pathlib import Path


def write_json_locked(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        json.dump(data, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def read_json_locked(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        data = json.load(f)
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return data


def open_log_binary(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, "ab", buffering=0)


def write_text_atomic(path: Path, text: str, mode: int = 0o600) -> None:
    """Atomically write text to a path with restrictive permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name, suffix=".tmp"
    )
    tmp = Path(tmp_path)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    os.replace(tmp, path)
    with contextlib.suppress(OSError):
        os.chmod(path, mode)
    with contextlib.suppress(FileNotFoundError):
        os.remove(tmp_path)
