#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

from typing import Optional

from tensorcast.proto.config.v1 import client_config_pb2 as cc_pb2

_CLIENT_CFG: Optional[cc_pb2.ClientConfig] = None


def set_client_config(cfg: cc_pb2.ClientConfig) -> None:
    global _CLIENT_CFG
    _CLIENT_CFG = cfg


def get_client_config() -> Optional[cc_pb2.ClientConfig]:
    return _CLIENT_CFG


def storage_root_default() -> Optional[str]:
    if _CLIENT_CFG and _CLIENT_CFG.HasField("storage"):
        return _CLIENT_CFG.storage.default_root or None
    return None


def daemon_target_default() -> Optional[str]:
    if (
        _CLIENT_CFG
        and _CLIENT_CFG.HasField("daemon")
        and _CLIENT_CFG.daemon.HasField("target")
    ):
        host = _CLIENT_CFG.daemon.target.host
        port = _CLIENT_CFG.daemon.target.port
        if host and port:
            return f"{host}:{port}"
    return None


def client_defaults() -> tuple[Optional[int], Optional[bool], Optional[bool]]:
    """Return (pinned_allocation_timeout_ms, enable_verification, wait_for_completion)."""
    if _CLIENT_CFG and _CLIENT_CFG.HasField("defaults"):
        d = _CLIENT_CFG.defaults
        timeout_ms = (
            d.pinned_allocation_timeout.seconds * 1000
            + d.pinned_allocation_timeout.nanos // 1_000_000
        )
        return (
            timeout_ms if timeout_ms > 0 else None,
            d.enable_verification,
            d.wait_for_completion,
        )
    return (None, None, None)
