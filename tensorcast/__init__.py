#  Copyright (c) 2025-2026, TensorCast Team.

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
import importlib
import importlib.abc
import sys
import threading
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable


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
# Build configuration helpers
# -----------------------------------------------------------------------------
# Validation runs on first C-extension load via tensorcast._c_ext to avoid
# importing the heavy extension at package import time.
from tensorcast._build_config import (  # noqa: E402, F401
    BuildConfigMismatchError,
    validate_cuda_backend_consistency,
)
from tensorcast._version import __version__  # noqa: E402, F401

# -----------------------------------------------------------------------------
# Lazy public API re-exports
# -----------------------------------------------------------------------------
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "Artifact": ("tensorcast.api", "Artifact"),
    "ArtifactDescriptor": ("tensorcast.api", "ArtifactDescriptor"),
    "ArtifactError": ("tensorcast.api", "ArtifactError"),
    "ArtifactFuture": ("tensorcast.api", "ArtifactFuture"),
    "FallbackOptions": ("tensorcast.api", "FallbackOptions"),
    "GetArtifactOptions": ("tensorcast.api", "GetArtifactOptions"),
    "PlanType": ("tensorcast.api", "PlanType"),
    "RegisterArtifactOptions": ("tensorcast.api", "RegisterArtifactOptions"),
    "RegisteredArtifact": ("tensorcast.api", "RegisteredArtifact"),
    "RegisteredLease": ("tensorcast.api", "RegisteredLease"),
    "RegistrationResult": ("tensorcast.api", "RegistrationResult"),
    "SealAssemblyResult": ("tensorcast.api", "SealAssemblyResult"),
    "ViewRegistrationKind": ("tensorcast.api", "ViewRegistrationKind"),
    "Store": ("tensorcast.api", "Store"),
    "StoreOptions": ("tensorcast.api", "StoreOptions"),
    "build_indices_from_safetensors": (
        "tensorcast.api",
        "build_indices_from_safetensors",
    ),
    "calculate_tensor_device_offsets": (
        "tensorcast.api",
        "calculate_tensor_device_offsets",
    ),
    "CallContext": ("tensorcast.api", "CallContext"),
    "CollectiveLoadGroup": ("tensorcast.api", "CollectiveLoadGroup"),
    "CapabilityDirectoryClient": (
        "tensorcast.capability_directory",
        "CapabilityDirectoryClient",
    ),
    "CapabilityDirectoryOptions": (
        "tensorcast.capability_directory",
        "CapabilityDirectoryOptions",
    ),
    "Operation": ("tensorcast.api", "Operation"),
    "OperationError": ("tensorcast.api", "OperationError"),
    "OperationStatus": ("tensorcast.api", "OperationStatus"),
    "OperationTimeoutError": ("tensorcast.api", "OperationTimeoutError"),
    "Plan": ("tensorcast.api", "Plan"),
    "PlanFailedError": ("tensorcast.api", "PlanFailedError"),
    "PlanResult": ("tensorcast.api", "PlanResult"),
    "PlanStepRef": ("tensorcast.api", "PlanStepRef"),
    "PlanStepResult": ("tensorcast.api", "PlanStepResult"),
    "Instance": ("tensorcast.api", "Instance"),
    "Worker": ("tensorcast.api", "Worker"),
    "TargetSpec": ("tensorcast.api", "TargetSpec"),
    "TransformSpec": ("tensorcast.api", "TransformSpec"),
    "context": ("tensorcast.api", "context"),
    "plan": ("tensorcast.api", "plan"),
    "RetentionHandle": ("tensorcast.retention", "RetentionHandle"),
    "acquire_retention_handle": ("tensorcast.retention", "acquire_retention_handle"),
    "renew_retention_handle": ("tensorcast.retention", "renew_retention_handle"),
    "release_retention_handle": ("tensorcast.retention", "release_retention_handle"),
    "artifact": ("tensorcast.api.store", "artifact"),
    "artifact_async": ("tensorcast.api.store", "artifact_async"),
    "deregister_artifact": ("tensorcast.api.store", "deregister_artifact"),
    "from_disk": ("tensorcast.api.store", "from_disk"),
    "put": ("tensorcast.api.store", "put"),
    "put_async": ("tensorcast.api.store", "put_async"),
    "persistence_operation": ("tensorcast.api.store", "persistence_operation"),
    "query_persistence_status": ("tensorcast.api.store", "query_persistence_status"),
    "register": ("tensorcast.api.store", "register"),
    "register_async": ("tensorcast.api.store", "register_async"),
    "register_view": ("tensorcast.api.store", "register_view"),
    "register_piece": ("tensorcast.api.store", "register_piece"),
    "register_vram_region": ("tensorcast.api.store", "register_vram_region"),
    "seal_assembly": ("tensorcast.api.store", "seal_assembly"),
    "store": ("tensorcast.api.store", "store"),
    "unregister_vram_region": ("tensorcast.api.store", "unregister_vram_region"),
    "init": ("tensorcast.startup", "init"),
    "PortConfig": ("tensorcast.startup", "PortConfig"),
    "is_initialized": ("tensorcast.startup", "is_initialized"),
    "shutdown": ("tensorcast.startup", "shutdown"),
}

context: Any
plan: Any


def __getattr__(name: str) -> Any:
    if name not in _LAZY_ATTRS:
        raise AttributeError(name)
    module_name, attr_name = _LAZY_ATTRS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(_LAZY_ATTRS))


if TYPE_CHECKING:
    from tensorcast.api import (  # noqa: F401
        Artifact,
        ArtifactDescriptor,
        ArtifactError,
        ArtifactFuture,
        CallContext,
        CapabilityDirectoryClient,
        CapabilityDirectoryOptions,
        CollectiveLoadGroup,
        FallbackOptions,
        GetArtifactOptions,
        Instance,
        Operation,
        OperationError,
        OperationStatus,
        OperationTimeoutError,
        Plan,
        PlanFailedError,
        PlanResult,
        PlanStepRef,
        PlanStepResult,
        PlanType,
        RegisterArtifactOptions,
        RegisteredArtifact,
        RegisteredLease,
        RegistrationResult,
        RetentionHandle,
        Store,
        StoreOptions,
        TargetSpec,
        TransformSpec,
        Worker,
        acquire_retention_handle,
        build_indices_from_safetensors,
        calculate_tensor_device_offsets,
        release_retention_handle,
        renew_retention_handle,
    )
    from tensorcast.api.store import (  # type: ignore[no-redef]  # noqa: F401
        artifact,
        artifact_async,
        deregister_artifact,
        from_disk,
        persistence_operation,
        put,
        put_async,
        query_persistence_status,
        register,
        register_async,
        register_view,
        register_vram_region,
        store,
        unregister_vram_region,
    )
    from tensorcast.startup import (  # noqa: F401
        PortConfig,
        init,
        is_initialized,
        shutdown,
    )


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
    "RegisteredLease",
    "RegistrationResult",
    "PlanType",
    "RegisterArtifactOptions",
    "GetArtifactOptions",
    "calculate_tensor_device_offsets",
    "build_indices_from_safetensors",
    "CallContext",
    "CollectiveLoadGroup",
    "Operation",
    "OperationError",
    "OperationStatus",
    "OperationTimeoutError",
    "Plan",
    "PlanFailedError",
    "PlanResult",
    "PlanStepRef",
    "PlanStepResult",
    "Worker",
    "Instance",
    "TargetSpec",
    "TransformSpec",
    "context",
    "plan",
    "from_disk",
    "Artifact",
    "ArtifactDescriptor",
    "store",
    "register",
    "register_async",
    "register_view",
    "persistence_operation",
    "put",
    "put_async",
    "query_persistence_status",
    "artifact",
    "artifact_async",
    "register_vram_region",
    "unregister_vram_region",
    "deregister_artifact",
    "BuildConfigMismatchError",
]
