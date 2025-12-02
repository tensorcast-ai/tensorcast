#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import contextlib

# -----------------------------------------------------------------------------
# Early patch for a PyTorch bug that can raise the following exception when
# importing `torch.overrides` multiple times within the same interpreter:
#
#   RuntimeError: function '_has_torch_function' already has a docstring
#
# The root cause is a duplicate call to `torch._C._add_docstr` which (by design)
# throws if a docstring already exists.  We defensively wrap the original
# implementation so that it silently ignores this specific error.  To avoid
# eagerly importing PyTorch (and its dependency chain) we install the patch via
# a meta path hook that runs immediately before ``tensorcast._C`` is imported.
# -----------------------------------------------------------------------------
import importlib.abc
import sys
import threading
from types import ModuleType
from typing import Any, Callable


class _TensorCastCExtensionBootstrap(importlib.abc.MetaPathFinder):
    """Installs prerequisites immediately before ``tensorcast._C`` loads."""

    _lock = threading.Lock()
    _prepared = False

    @classmethod
    def _ensure_prepared(cls) -> None:
        import torch  # pylint: disable=import-outside-toplevel

        from tensorcast._cuda_loader import ensure_nvrtc_builtins_preloaded

        ensure_nvrtc_builtins_preloaded()

        # Patch torch._C._add_docstr to tolerate duplicate docstrings (PyTorch bug).
        _c_mod: ModuleType = torch._C
        _orig_add_docstr: Callable[[Any, str], Any] = _c_mod._add_docstr

        try:
            already_patched = _orig_add_docstr._tensorcast_patched  # pyright: ignore[reportFunctionMemberAccess]
        except AttributeError:
            already_patched = False

        if already_patched:
            return

        def _safe_add_docstr(obj: Any, doc: str) -> Any:
            try:
                return _orig_add_docstr(obj, doc)
            except RuntimeError as exc:  # noqa: BLE001
                if "already has a docstring" in str(exc):
                    return obj
                raise

        _safe_add_docstr._tensorcast_patched = True  # pyright: ignore[reportFunctionMemberAccess]
        _c_mod._add_docstr = _safe_add_docstr

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> Any:  # noqa: D401, ANN401
        if fullname != "tensorcast._C":
            return None

        finder_cls = type(self)

        if finder_cls._prepared:
            return None

        with finder_cls._lock:
            if not finder_cls._prepared:
                finder_cls._ensure_prepared()
                finder_cls._prepared = True
                with contextlib.suppress(ValueError):
                    sys.meta_path.remove(self)

        return None


def _install_c_extension_bootstrap() -> None:
    for finder in sys.meta_path:
        if isinstance(finder, _TensorCastCExtensionBootstrap):
            return

    sys.meta_path.insert(0, _TensorCastCExtensionBootstrap())


_install_c_extension_bootstrap()

# -----------------------------------------------------------------------------
# Build configuration validation
# -----------------------------------------------------------------------------
# Validate that the C extension's CUDA backend matches the runtime environment.
# This catches a common misconfiguration where the extension is built with
# USE_FAKE_CUDA=1 but PyTorch has real CUDA available.
from tensorcast._build_config import (  # noqa: E402
    BuildConfigMismatchError,
    validate_cuda_backend_consistency,
)

validate_cuda_backend_consistency()

# -----------------------------------------------------------------------------
# Public package interface remains unchanged below.
# -----------------------------------------------------------------------------

from tensorcast._version import __version__  # noqa: E402
from tensorcast.api import (  # noqa: E402
    ArtifactDescriptor,
    ArtifactError,
    ArtifactFuture,
    FallbackOptions,
    GetArtifactOptions,
    PlanType,
    RegisterArtifactOptions,
    RegisteredArtifact,
    RegisteredLease,
    RegistrationResult,
    Store,
    StoreOptions,
    build_indices_from_safetensors,
    calculate_tensor_device_offsets,
    save_dict,
)
from tensorcast.api.store import (  # noqa: E402
    BatchContext,
    PrefetchTicket,
    artifact,
    artifact_async,
    from_disk,
    get,
    get_async,
    get_into,
    get_into_async,
    put,
    put_async,
    register,
    register_async,
    register_view,
    store,
)
from tensorcast.startup import init, is_initialized, shutdown  # noqa: E402

__all__ = [
    "__version__",
    "init",
    "is_initialized",
    "shutdown",
    "Store",
    "StoreOptions",
    "RegisteredArtifact",
    "ArtifactError",
    "ArtifactFuture",
    "FallbackOptions",
    "save_dict",
    "RegisteredLease",
    "RegistrationResult",
    "PlanType",
    "RegisterArtifactOptions",
    "GetArtifactOptions",
    "calculate_tensor_device_offsets",
    "build_indices_from_safetensors",
    "from_disk",
    "ArtifactDescriptor",
    "artifact",
    "artifact_async",
    "BatchContext",
    "store",
    "register",
    "register_async",
    "register_view",
    "put",
    "put_async",
    "get",
    "get_async",
    "get_into",
    "get_into_async",
    "PrefetchTicket",
    "BuildConfigMismatchError",
]
