#  Copyright (c) 2025-2026, TensorCast Team.

"""Validate daemon library environment discovery resolves required shared libs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Iterable, Mapping

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


def _load_shared_objects_in_subprocess(
    env: Mapping[str, str], libraries: Iterable[str]
) -> list[str]:
    """Load shared objects in a fresh process.

    Linux dynamic loaders typically parse ``LD_LIBRARY_PATH`` at process start;
    mutating it inside the current pytest process does not reliably affect
    subsequent ``dlopen`` calls. We therefore validate loadability in a child
    process with the daemon environment applied from the start.
    """

    payload = json.dumps(list(libraries))
    script = textwrap.dedent(
        """\
        from __future__ import annotations

        import ctypes
        import json
        import sys

        libs = json.loads(sys.argv[1])
        missing: list[str] = []
        for lib in libs:
            try:
                ctypes.CDLL(lib)  # noqa: F401 - validate loadability
            except OSError as exc:
                missing.append(f"{lib}: {exc}")
        print(json.dumps(missing))
        """
    )

    proc = subprocess.run(
        [sys.executable, "-c", script, payload],
        env=dict(env),
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        details = (proc.stderr or proc.stdout).strip()
        return [f"subprocess failed (rc={proc.returncode}): {details}"]

    try:
        decoded = json.loads(proc.stdout.strip() or "[]")
    except json.JSONDecodeError as exc:
        combined = (proc.stdout + "\n" + proc.stderr).strip()
        return [f"subprocess output not JSON ({exc}): {combined}"]

    if not isinstance(decoded, list):
        return [f"subprocess output not list: {decoded!r}"]

    return [str(item) for item in decoded]


@pytest.mark.integration
def test_daemon_library_environment_loads_required_shared_objects() -> None:
    env = build_daemon_process_env(os.environ)
    ld_library_path = env.get("LD_LIBRARY_PATH")
    assert ld_library_path, "daemon environment must provide LD_LIBRARY_PATH"

    library_dirs = _collect_library_dirs(ld_library_path)
    assert library_dirs, "expected at least one library directory from LD_LIBRARY_PATH"

    resolved_env_libs, env_missing = _resolve_library_entries(
        library_dirs, _ENV_MANAGED_LIBRARIES
    )
    assert not env_missing, (
        "expected CUDA/Torch libraries discoverable via LD_LIBRARY_PATH, missing: "
        + ", ".join(env_missing)
    )

    libraries_to_load: list[str] = list(resolved_env_libs)
    libraries_to_load.append("libgomp.so.1")
    load_missing = _load_shared_objects_in_subprocess(env, libraries_to_load)
    assert not load_missing, (
        "required shared objects failed to load with daemon environment: "
        + ", ".join(load_missing)
    )
