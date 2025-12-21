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
from typing import Any, Iterable, Mapping, Optional

import psutil

from .errors import ServiceError


def _discover_daemon_library_paths() -> list[Path]:
    """Return library search paths required by the daemon runtime.

    The packaged daemon depends on TensorCast shared objects as well as the
    libtorch and NVIDIA CUDA runtime distributed with the Python wheels. Bazel
    does not set an rpath for these, so we explicitly surface the relevant
    directories for ``LD_LIBRARY_PATH`` augmentation.
    """

    paths: list[Path] = []

    package_root = Path(__file__).resolve().parents[1]
    package_lib = package_root / "lib"
    if package_lib.is_dir():
        paths.append(package_lib)

    site_packages: Path | None = None

    try:
        import torch
    except Exception:
        torch = None

    if torch is not None:
        try:
            torch_root = Path(torch.__file__).resolve().parent
        except Exception:
            torch_root = None
        if torch_root is not None:
            torch_lib = torch_root / "lib"
            if torch_lib.is_dir():
                paths.append(torch_lib)
            site_packages = torch_root.parent

    if site_packages is None:
        import site as _site

        for candidate in map(Path, _site.getsitepackages()):
            if candidate.exists():
                site_packages = candidate
                break

    if site_packages is not None:
        nvidia_root = site_packages / "nvidia"
        if nvidia_root.is_dir():
            paths.extend(
                candidate
                for candidate in sorted(nvidia_root.rglob("lib"))
                if candidate.is_dir()
            )
            paths.extend(
                candidate
                for candidate in sorted(nvidia_root.rglob("lib64"))
                if candidate.is_dir()
            )

        cuda_components = (
            "cublas",
            "cublaslt",
            "cudnn",
            "cufft",
            "curand",
            "cusolver",
            "cusparse",
            "cusparselt",
            "cuda_cupti",
            "cuda_nvrtc",
            "cuda_runtime",
            "nccl",
            "nvjitlink",
            "nvrtc",
            "nvtx",
        )

        for component in cuda_components:
            lowered = component.lower()
            for subdir_name in ("lib", "lib64"):
                direct = site_packages / component / subdir_name
                if direct.is_dir():
                    paths.append(direct)

                direct_lower = site_packages / lowered / subdir_name
                if direct_lower.is_dir():
                    paths.append(direct_lower)

                nested = site_packages / "nvidia" / component / subdir_name
                if nested.is_dir():
                    paths.append(nested)

                nested_lower = site_packages / "nvidia" / lowered / subdir_name
                if nested_lower.is_dir():
                    paths.append(nested_lower)

    return _dedupe_library_paths(paths)


def _dedupe_library_paths(candidates: Iterable[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[Path] = set()
    for p in candidates:
        try:
            resolved = p.resolve()
        except Exception:
            resolved = p
        if not resolved.exists():
            continue
        if resolved in seen:
            continue
        deduped.append(resolved)
        seen.add(resolved)
    return deduped


def build_daemon_process_env(
    base_env: Mapping[str, str] | None = None,
    extra_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Prepare an environment mapping for launching the C++ daemon."""

    env = dict(base_env or os.environ)
    if extra_env:
        env.update(extra_env)

    ld_paths = _discover_daemon_library_paths()
    if ld_paths:
        existing = env.get("LD_LIBRARY_PATH", "")
        ld_entries = [str(p) for p in ld_paths if str(p)]
        if existing:
            ld_entries.append(existing)
        env["LD_LIBRARY_PATH"] = ":".join(ld_entries)

    return env


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


def preexec_detached(ignore_sigint: bool = True) -> None:
    os.setsid()
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

    raise ServiceError(
        "tensorcast_daemon binary not found. Build it with bazel build //daemon:tensorcast_daemon."
    )
