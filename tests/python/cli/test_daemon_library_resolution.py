#  Copyright (c) 2025-2026, TensorCast Team.

"""Validate daemon library environment discovery resolves required shared libs."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Iterable

import pytest

from tensorcast.cli_utils.proc import build_daemon_process_env

_REQUIRED_LIBRARIES: tuple[str, ...] = (
    "libnvrtc.so.12",
    "libnvrtc-builtins.so.12.4",
    "libcublas.so.12",
    "libcublasLt.so.12",
    "libcudnn.so.9",
    "libcufft.so.11",
    "libcupti.so.12",
    "libcurand.so.10",
    "libcusparse.so.12",
    "libcusparseLt.so.0",
    "libnccl.so.2",
    "libnvJitLink.so.12",
    "libcudart.so.12",
    "libgomp.so.1",
    "libtorch.so",
    "libtorch_cpu.so",
    "libtorch_cuda.so",
    "libtorch_global_deps.so",
    "libc10_cuda.so",
    "libc10.so",
)


_ENV_MANAGED_LIBRARIES: tuple[str, ...] = tuple(
    lib for lib in _REQUIRED_LIBRARIES if lib != "libgomp.so.1"
)


def _collect_library_dirs(ld_library_path: str) -> tuple[Path, ...]:
    dirs: list[Path] = []
    for raw_entry in ld_library_path.split(":"):
        entry = raw_entry.strip()
        if not entry:
            continue
        candidate = Path(entry)
        if candidate.is_dir():
            dirs.append(candidate)
    return tuple(dirs)


def _resolve_library_entries(
    library_dirs: tuple[Path, ...], libs: tuple[str, ...]
) -> tuple[list[str], list[str]]:
    resolved: list[str] = []
    missing: list[str] = []
    for lib_name in libs:
        if any((directory / lib_name).exists() for directory in library_dirs):
            resolved.append(lib_name)
            continue
        if lib_name.startswith("libnvrtc-builtins.so."):
            major_prefix = lib_name.rsplit(".", 1)[0]
            candidates: list[Path] = []
            for directory in library_dirs:
                candidates.extend(directory.glob(f"{major_prefix}*"))
            candidates = [candidate for candidate in candidates if candidate.is_file()]
            if candidates:
                candidates.sort()
                resolved.append(str(candidates[-1]))
                continue
        missing.append(lib_name)
    return resolved, missing


def _load_shared_objects(libraries: Iterable[str]) -> list[str]:
    missing: list[str] = []
    for lib_name in libraries:
        try:
            ctypes.CDLL(lib_name)  # noqa: F401 - keep handle alive for validation
        except OSError as exc:  # pragma: no cover - exercised during failures
            missing.append(f"{lib_name}: {exc}")
    return missing


@pytest.mark.integration
def test_daemon_library_environment_loads_required_shared_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    env = build_daemon_process_env(os.environ)
    ld_library_path = env.get("LD_LIBRARY_PATH")
    assert ld_library_path, "daemon environment must provide LD_LIBRARY_PATH"

    monkeypatch.setenv("LD_LIBRARY_PATH", ld_library_path)

    library_dirs = _collect_library_dirs(ld_library_path)
    assert library_dirs, "expected at least one library directory from LD_LIBRARY_PATH"

    resolved_env_libs, env_missing = _resolve_library_entries(
        library_dirs, _ENV_MANAGED_LIBRARIES
    )
    assert not env_missing, (
        "expected CUDA/Torch libraries discoverable via LD_LIBRARY_PATH, missing: "
        + ", ".join(env_missing)
    )

    resolved_load_libs, load_missing = _resolve_library_entries(
        library_dirs, _REQUIRED_LIBRARIES
    )
    load_missing.extend(_load_shared_objects(resolved_load_libs))
    assert not load_missing, (
        "ldd-listed libraries failed to load by exact name, missing: "
        + ", ".join(load_missing)
    )
