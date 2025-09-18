#  Copyright (c) 2025, TensorCast Team.

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
        "libnvrtc-builtins.so.12.4",
        "libnvrtc-builtins.so.12",
        "libnvrtc-builtins.so",
    )

    if _try_load_shared_objects(candidates):
        _preload_succeeded = True
        return

    module = _import_cuda_nvrtc()
    if not module:
        return

    lib_dir = Path(module.__file__).resolve().parent / "lib"
    if not lib_dir.is_dir():
        _logger.debug(
            "cuda_nvrtc package present but lib/ directory missing at %s", lib_dir
        )
        return

    so_candidates = sorted(lib_dir.glob("libnvrtc-builtins.so*"), reverse=True)
    if not so_candidates:
        _logger.debug("No libnvrtc-builtins candidates found under %s", lib_dir)
        return

    if _try_load_shared_objects(str(path) for path in so_candidates):
        _preload_succeeded = True
        return

    _logger.debug("Failed to preload libnvrtc-builtins from %s", lib_dir)


def _import_cuda_nvrtc():
    try:
        return importlib.import_module("nvidia.cuda_nvrtc")
    except ModuleNotFoundError:
        _logger.debug(
            "nvidia-cuda-nvrtc-cu12 package not installed; skipping NVRTC preload"
        )
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
