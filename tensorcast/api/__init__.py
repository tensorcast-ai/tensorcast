#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import atexit
import logging
import os
import threading
import warnings
from os import PathLike

import torch

from tensorcast import startup
from tensorcast.api._config import (
    GetArtifactOptions,
    PlanType,
    RegisterArtifactOptions,
    get_daemon_address,
    has_daemon_address,
)
from tensorcast.api._device import resolve_device
from tensorcast.api._indices import (
    build_indices_from_safetensors,
    calculate_tensor_device_offsets,
)
from tensorcast.api._io_disk import load_dict_from_disk, save_dict
from tensorcast.api._loader import (
    LoadHandle,
)
from tensorcast.api._loader import (
    get_artifact_async as _legacy_get_artifact_async,
)
from tensorcast.api._loader import (
    get_artifact_sync as _legacy_get_artifact_sync,
)
from tensorcast.api._loader import (
    load_dict_async as _legacy_load_dict_async,
)
from tensorcast.api._loader import (
    load_dict_sync as _legacy_load_dict_sync,
)
from tensorcast.api._register import (
    RegisteredArtifact as _LegacyRegisteredArtifact,
)
from tensorcast.api._register import (
    RegisteredLease,
    RegistrationResult,
    begin_register_artifact_sdk,
)
from tensorcast.api._register import (
    register_artifact as _legacy_register_artifact,
)
from tensorcast.api.store import (
    ArtifactError,
    ArtifactFuture,
    FallbackOptions,
    Store,
    StoreOptions,
)
from tensorcast.client_runtime import daemon_target_default
from tensorcast.types import ArtifactDescriptor, CommitResult

logger = logging.getLogger(__name__)

_legacy_store: Store | None = None
_legacy_store_lock = threading.RLock()
_legacy_store_atexit = False
_legacy_warnings: set[str] = set()


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    normalized = value.strip().lower()
    return normalized in {"1", "true", "yes", "on"}


_STORE_SESSION_REQUIRED = _truthy(os.getenv("TENSORCAST_STORE_SESSION_REQUIRED"))

RegisteredArtifact = _LegacyRegisteredArtifact


def _close_legacy_store() -> None:
    global _legacy_store
    store = _legacy_store
    if store is None:
        return
    try:
        store.close()
    finally:
        _legacy_store = None


def _emit_deprecation(symbol: str, guidance: str) -> None:
    if symbol in _legacy_warnings:
        return
    _legacy_warnings.add(symbol)
    warnings.warn(
        f"tensorcast.api.{symbol} is deprecated; {guidance}",
        DeprecationWarning,
        stacklevel=2,
    )


def _require_store_session(symbol: str) -> None:
    if not _STORE_SESSION_REQUIRED:
        return
    raise RuntimeError(
        "Legacy helper usage is disabled by TENSORCAST_STORE_SESSION_REQUIRED. "
        f"Instantiate tensorcast.api.Store and call {symbol} instead."
    )


def _resolve_daemon_endpoint() -> str:
    if has_daemon_address():
        return get_daemon_address()
    default_target = daemon_target_default()
    if default_target:
        return default_target
    try:
        ctx = startup.require_initialized()
    except RuntimeError as exc:  # noqa: BLE001
        raise RuntimeError(
            "TensorCast daemon target is not configured. Call tensorcast.startup.init() "
            "or set a daemon target via client config before using legacy helpers."
        ) from exc
    return ctx.address


def _get_or_create_store() -> Store:
    global _legacy_store, _legacy_store_atexit
    with _legacy_store_lock:
        if _legacy_store is not None and not getattr(_legacy_store, "_closed", False):
            return _legacy_store
        endpoint = _resolve_daemon_endpoint()
        store = Store(endpoint, opts=StoreOptions())
        _legacy_store = store
        if not _legacy_store_atexit:
            atexit.register(_close_legacy_store)
            _legacy_store_atexit = True
        return store


def _fallback_from_options(
    options: GetArtifactOptions | None,
) -> FallbackOptions | None:
    if options is None:
        return None
    prefer = options.prefer.strip().lower()
    if prefer == "disk":
        return FallbackOptions(
            prefer_disk=True,
            allow_p2p=False,
            verify_checksums=options.enable_verification,
        )
    return None


def _artifact_future_to_load_handle(
    future: ArtifactFuture[dict[str, torch.Tensor]],
) -> LoadHandle:
    state_holder: dict[str, torch.Tensor] = {}
    confirmed = {"value": False}

    def _confirm() -> bool:
        if confirmed["value"]:
            return True
        try:
            result = future.result()
        except ArtifactError:
            return False
        except Exception:  # noqa: BLE001
            return False
        state_holder.clear()
        state_holder.update(result)
        confirmed["value"] = True
        return True

    return LoadHandle(state_holder, _confirm)


def _device_selector_from_legacy(device: int | torch.device) -> torch.device:
    resolved = resolve_device(device)
    return torch.device("cuda", resolved)


def register_artifact(
    artifact: dict[str, torch.Tensor],
    *,
    options: RegisterArtifactOptions,
    device_id: int | torch.device | None = None,
    ttl_ms: int | None = None,
) -> RegistrationResult:
    _require_store_session("Store.register/Store.put")
    opts = options
    _emit_deprecation(
        "register_artifact", "instantiate tensorcast.api.Store and call register/put"
    )
    try:
        store = _get_or_create_store()
    except Exception:  # noqa: BLE001
        logger.debug("Falling back to legacy register_artifact path", exc_info=True)
        return _legacy_register_artifact(
            artifact,
            options=opts,
            device_id=device_id,
            ttl_ms=ttl_ms,
        )

    effective_plan = opts.plan
    if opts.lease_in_place:
        effective_plan = PlanType.VRAM_LEASED

    try:
        if effective_plan is PlanType.VRAM_LEASED:
            result = store.register(
                artifact,
                key=opts.key,
                options=opts,
                ttl_ms=ttl_ms,
            )
        else:
            result = store.put(
                artifact,
                key=opts.key,
                options=opts,
                device=_device_selector_from_legacy(device_id)
                if device_id is not None
                else None,
            )
    except ArtifactError:
        logger.debug("Store register failed; using legacy helper", exc_info=True)
        return _legacy_register_artifact(
            artifact,
            options=opts,
            device_id=device_id,
            ttl_ms=ttl_ms,
        )

    if result.registration_result is None:
        return _legacy_register_artifact(
            artifact,
            options=opts,
            device_id=device_id,
            ttl_ms=ttl_ms,
        )
    return result.registration_result


def get_artifact_sync(
    *,
    key: str,
    device_id: int | torch.device = 0,
    options: GetArtifactOptions | None = None,
) -> dict[str, torch.Tensor]:
    opts = options or GetArtifactOptions()
    _require_store_session("Store.get")
    _emit_deprecation("get_artifact_sync", "use Store.get instead")
    fallback = _fallback_from_options(opts)
    try:
        store = _get_or_create_store()
        return store.get(
            key=key,
            device=_device_selector_from_legacy(device_id),
            fallback=fallback,
            options=opts,
        )
    except ArtifactError:
        logger.debug("Store get failed; using legacy helper", exc_info=True)
        return _legacy_get_artifact_sync(
            key=key,
            device_id=device_id,
            options=opts,
        )


def get_artifact_async(
    *,
    key: str,
    device_id: int | torch.device = 0,
    options: GetArtifactOptions | None = None,
) -> LoadHandle:
    opts = options or GetArtifactOptions()
    _require_store_session("Store.get_async")
    _emit_deprecation("get_artifact_async", "use Store.get_async instead")
    fallback = _fallback_from_options(opts)
    try:
        store = _get_or_create_store()
        future = store.get_async(
            key=key,
            device=_device_selector_from_legacy(device_id),
            fallback=fallback,
            options=opts,
        )
    except ArtifactError:
        logger.debug("Store get_async failed; using legacy helper", exc_info=True)
        return _legacy_get_artifact_async(
            key=key,
            device_id=device_id,
            options=opts,
        )
    return _artifact_future_to_load_handle(future)


def load_dict_sync(
    disk_path: str | PathLike[str] | None = None,
    device_id: int | torch.device = 0,
    storage_path: str | PathLike[str] | None = None,
    enable_verification: bool = True,
    pinned_allocation_timeout_ms: int = 30_000,
) -> dict[str, torch.Tensor]:
    _require_store_session("Store.get with disk fallback")
    _emit_deprecation("load_dict_sync", "use Store.get with disk fallback")
    return _legacy_load_dict_sync(
        disk_path=disk_path,
        device_id=device_id,
        storage_path=storage_path,
        enable_verification=enable_verification,
        pinned_allocation_timeout_ms=pinned_allocation_timeout_ms,
    )


def load_dict_async(
    disk_path: str | PathLike[str] | None = None,
    device_id: int | torch.device = 0,
    storage_path: str | PathLike[str] | None = None,
    enable_verification: bool = True,
    pinned_allocation_timeout_ms: int = 30_000,
) -> LoadHandle:
    _require_store_session("Store.get_async with disk fallback")
    _emit_deprecation("load_dict_async", "use Store.get_async with disk fallback")
    return _legacy_load_dict_async(
        disk_path=disk_path,
        device_id=device_id,
        storage_path=storage_path,
        enable_verification=enable_verification,
        pinned_allocation_timeout_ms=pinned_allocation_timeout_ms,
    )


__all__ = [
    # Store session API
    "Store",
    "StoreOptions",
    "ArtifactError",
    "ArtifactFuture",
    "FallbackOptions",
    # Legacy helpers (deprecated)
    "load_dict_sync",
    "load_dict_async",
    "get_artifact_sync",
    "get_artifact_async",
    "save_dict",
    "load_dict_from_disk",
    "register_artifact",
    "begin_register_artifact_sdk",
    "RegisteredArtifact",
    "RegisteredLease",
    # Config/Options
    "PlanType",
    "RegisterArtifactOptions",
    "GetArtifactOptions",
    # Low-level helpers (used by tests/examples)
    "calculate_tensor_device_offsets",
    "build_indices_from_safetensors",
    # Types
    "CommitResult",
    "ArtifactDescriptor",
]
