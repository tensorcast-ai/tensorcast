#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

from typing import Tuple

from tensorcast import startup
from tensorcast.api._runtime_handle import RuntimeHandle
from tensorcast.daemon_ctl import DaemonCtl


def is_initialized() -> bool:
    return startup.is_initialized()


def require_runtime() -> RuntimeHandle:
    ctx = startup.require_initialized()
    return RuntimeHandle(address=ctx.address, client=ctx.client)


def require_client() -> DaemonCtl:
    return require_runtime().client


def apply_client_load_defaults_if_present(
    pinned_allocation_timeout_ms: int,
    enable_verification: bool,
    wait_for_completion: bool,
    *,
    runtime_address: str,
) -> Tuple[int, bool, bool]:
    from tensorcast.api._config import DEFAULT_PINNED_TIMEOUT_MS
    from tensorcast.client_runtime import client_defaults, daemon_target_default

    cfg_timeout_ms, cfg_enable_ver, cfg_wait = client_defaults()
    cfg_target = daemon_target_default()
    if cfg_target and cfg_target != runtime_address:
        raise RuntimeError(
            "ClientConfig daemon target does not match initialized daemon address. "
            "Call tensorcast.startup.init(mode='connect'|'create') with the desired daemon first."
        )

    if (
        pinned_allocation_timeout_ms == DEFAULT_PINNED_TIMEOUT_MS
        and cfg_timeout_ms is not None
    ):
        pinned_allocation_timeout_ms = int(cfg_timeout_ms)
    if enable_verification is True and cfg_enable_ver is not None:
        enable_verification = bool(cfg_enable_ver)
    if wait_for_completion is True and cfg_wait is not None:
        wait_for_completion = bool(cfg_wait)

    return pinned_allocation_timeout_ms, enable_verification, wait_for_completion


__all__ = [
    "RuntimeHandle",
    "apply_client_load_defaults_if_present",
    "is_initialized",
    "require_client",
    "require_runtime",
]
