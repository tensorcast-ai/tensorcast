#  Copyright (c) 2025, TensorCast Team.

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


def _find_missing_libraries(library_dirs: tuple[Path, ...], libs: tuple[str, ...]) -> list[str]:
    return [
        lib_name
        for lib_name in libs
        if not any((directory / lib_name).exists() for directory in library_dirs)
    ]


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

    env_missing = _find_missing_libraries(library_dirs, _ENV_MANAGED_LIBRARIES)
    assert not env_missing, (
        "expected CUDA/Torch libraries discoverable via LD_LIBRARY_PATH, missing: "
        + ", ".join(env_missing)
    )

    load_missing = _load_shared_objects(_REQUIRED_LIBRARIES)
    assert not load_missing, (
        "ldd-listed libraries failed to load by exact name, missing: "
        + ", ".join(load_missing)
    )
