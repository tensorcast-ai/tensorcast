#  Copyright (c) 2025-2026, TensorCast Team.

"""Helpers for preloading CUDA NVRTC runtime libraries."""

from __future__ import annotations

import ctypes
import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Iterable

_logger = logging.getLogger(__name__)

# Track whether we have already succeeded in loading the NVRTC builtins library.
_preload_attempted = False
_preload_succeeded = False


def ensure_nvrtc_builtins_preloaded() -> None:
    """Ensure the NVRTC builtins shared library is loaded into the process.

    PyTorch extensions depending on NVRTC symbols may fail to import when the
    dynamic loader cannot locate ``libnvrtc-builtins``.  The NVIDIA ``cuda_nvrtc``
    wheels ship the necessary shared objects alongside the Python package, but
    they are not added to ``LD_LIBRARY_PATH`` automatically.  We defensively load
    the library with ``RTLD_GLOBAL`` so that subsequent imports of ``tensorcast._C``
    see the symbols regardless of the user's environment configuration.

    The function is idempotent and will only attempt to load once per process.
    """

    global _preload_attempted, _preload_succeeded

    if _preload_attempted:
        return

    _preload_attempted = True

    candidates = (
        "libnvrtc-builtins.so.13",
        "libnvrtc-builtins.so.13.0",
        "libnvrtc-builtins.so.12.4",
        "libnvrtc-builtins.so.12",
        "libnvrtc-builtins.so",
    )

    if _try_load_shared_objects(candidates):
        _preload_succeeded = True
        return

    lib_dir = _find_cuda_nvrtc_lib_dir()
    if lib_dir is None:
        return

    so_candidates = sorted(lib_dir.glob("libnvrtc-builtins.so*"), reverse=True)
    if not so_candidates:
        _logger.debug("No libnvrtc-builtins candidates found under %s", lib_dir)
        return

    if _try_load_shared_objects(str(path) for path in so_candidates):
        _preload_succeeded = True
        return

    _logger.debug("Failed to preload libnvrtc-builtins from %s", lib_dir)


def _find_cuda_nvrtc_lib_dir() -> Path | None:
    """Locate the directory holding libnvrtc-builtins for both cu12 and cu13 wheel layouts.

    cu12 wheels ship ``nvidia/cuda_nvrtc/lib/...``; cu13 wheels ship
    ``nvidia/cu13/lib/...`` (umbrella shared across cu13 libs). Walk the
    ``nvidia`` namespace-package roots and return the first ``lib/`` we find.
    """

    try:
        spec = importlib.util.find_spec("nvidia")
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        _logger.debug(
            "Neither nvidia-cuda-nvrtc-cu12 nor nvidia-cuda-nvrtc (cu13) package "
            "installed; skipping NVRTC preload"
        )
        return None

    roots = list(getattr(spec, "submodule_search_locations", None) or [])
    if not roots and spec.origin:
        roots = [str(Path(spec.origin).parent)]

    for root in roots:
        for subdir in ("cuda_nvrtc", "cu13"):
            lib_dir = Path(root) / subdir / "lib"
            if lib_dir.is_dir():
                return lib_dir

    _logger.debug("No nvrtc lib dir found under any of %r", roots)
    return None


def _try_load_shared_objects(paths: Iterable[str]) -> bool:
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    for path in paths:
        try:
            ctypes.CDLL(path, mode=mode)
        except OSError:
            continue
        else:
            _logger.debug("Preloaded NVRTC builtins library from %s", path)
            return True
    return False


__all__ = ["ensure_nvrtc_builtins_preloaded"]
