#  Copyright (c) 2025-2026, TensorCast Team.

"""Network helpers: port selection, listen probes, and address normalization."""

from __future__ import annotations

import socket
import subprocess
import time
from typing import Any, Callable, Iterable


def resolve_connect_host(listen_host: str | None) -> str:
    if not listen_host:
        return "127.0.0.1"
    s = str(listen_host).strip().lower()
    if s in {"0.0.0.0", "::", "[::]", "*"}:
        return "127.0.0.1"
    return listen_host


def is_unspecified_host(host: str | None) -> bool:
    if not host:
        return True
    s = str(host).strip().lower()
    return s in {"0.0.0.0", "::", "[::]", "*"}


def pick_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(s.getsockname()[1])


def wait_daemon_listening(
    host: str,
    port: int,
    timeout: float | None = 20.0,
    *,
    proc: subprocess.Popen[Any] | None = None,
    progress: Callable[[float, Exception | None], None] | None = None,
    progress_interval: float = 5.0,
    extra_hosts: Iterable[str] | None = None,
) -> str | None:
    deadline = None if timeout is None else time.time() + timeout
    start_ts = time.time()
    last_report = start_ts
    last_err: Exception | None = None
    candidates: list[str] = []
    seen: set[str] = set()
    for candidate in (host, *(extra_hosts or [])):
        if not candidate:
            continue
        if candidate in seen:
            continue
        candidates.append(candidate)
        seen.add(candidate)
    if proc is not None and proc.poll() is not None:
        return None
    while deadline is None or time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return None
        for candidate in candidates:
            try:
                with socket.create_connection((candidate, port), timeout=0.2):
                    pass
                return candidate
            except Exception as e:  # noqa: BLE001
                last_err = e
                if proc is not None and proc.poll() is not None:
                    return None
        now = time.time()
        if progress is not None and now - last_report >= max(progress_interval, 0.5):
            progress(now - start_ts, last_err)
            last_report = now
        time.sleep(0.2)
    # For debugging purposes; caller can log if desired
    _ = last_err
    return None
