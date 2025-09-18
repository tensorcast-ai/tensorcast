#  Copyright (c) 2025, TensorCast Team.

"""Process management helpers: fate sharing, spawn, kill, and binary discovery.

Linux-only features are guarded to avoid crashing on import in other OSes.
"""

from __future__ import annotations

import contextlib
import importlib.resources as ir
import os
import platform
import signal
from pathlib import Path
from typing import Any, Iterable, Optional

import psutil

from .errors import ServiceError


def _require_linux() -> None:
    if platform.system() != "Linux":
        raise ServiceError("This operation requires Linux (PR_SET_PDEATHSIG)")


def set_pdeathsig(sig: int = signal.SIGKILL) -> None:
    """Install PR_SET_PDEATHSIG for the current (child) process. Linux-only."""
    _require_linux()
    import ctypes  # defer to avoid import-time failures on non-Linux

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    PR_SET_PDEATHSIG = 1
    res = libc.prctl(PR_SET_PDEATHSIG, sig, 0, 0, 0)
    if res != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"prctl(PR_SET_PDEATHSIG) failed: errno={err}")


def preexec_fate_sharing(
    ignore_sigint: bool = True, pdeathsig: int = signal.SIGKILL
) -> None:
    os.setsid()
    set_pdeathsig(pdeathsig)
    if ignore_sigint:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    with contextlib.suppress(AttributeError, OSError):
        signal.pthread_sigmask(
            signal.SIG_BLOCK, [signal.SIGTTOU, signal.SIGTTIN, signal.SIGTSTP]
        )


def pump(src, sinks: Iterable[Any]) -> None:
    import sys as _sys

    for line in iter(src.readline, b""):
        for s in sinks:
            try:
                if s in (_sys.stdout, _sys.stderr):
                    s.write(line.decode("utf-8", errors="replace"))
                    s.flush()
                else:
                    s.write(line)
                    s.flush()
            except Exception:
                pass


def kill_gracefully(pgid: int, grace: float = 10.0) -> bool:
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    import time

    deadline = time.time() + grace
    while time.time() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.2)
    return False


def kill_force(pgid: int) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGKILL)


def is_matching_daemon_process(pid: int, expected_cmd0: Optional[str]) -> bool:
    """Lightweight validation to avoid killing an unrelated PID due to reuse."""
    try:
        p = psutil.Process(pid)
        exe = p.exe() if p else ""
        name = p.name() if p else ""
        cmdline = p.cmdline() if p else []
        if expected_cmd0:
            try:
                if os.path.samefile(exe, expected_cmd0):
                    return True
            except Exception:
                pass
            if cmdline and os.path.basename(cmdline[0]) == os.path.basename(
                expected_cmd0
            ):
                return True
        if "tensorcast_daemon" in (name or ""):
            return True
        if cmdline and any(
            "tensorcast_daemon" in os.path.basename(x) for x in cmdline[:1]
        ):
            return True
    except Exception:
        pass
    return False


def ensure_cpp_daemon_binary() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / "bazel-bin" / "daemon" / "tensorcast_daemon"
    if candidate.exists() and os.access(candidate, os.X_OK):
        return candidate

    pkg = ir.files("tensorcast").joinpath("bin").joinpath("tensorcast_daemon")
    p = Path(str(pkg))
    if p.exists() and os.access(p, os.X_OK):
        return p

    raise ServiceError("tensorcast_daemon binary not found.")
