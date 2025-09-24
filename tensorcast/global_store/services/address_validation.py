#  Copyright (c) 2025, TensorCast Team.

"""Utilities for validating advertised node addresses."""

from __future__ import annotations

import ipaddress

from tensorcast.global_store.exceptions import ValidationError

_LOOPBACK_HOSTNAMES = {"localhost", "ip6-localhost"}
_UNSPECIFIED_SENTINELS = {"0.0.0.0", "::", "[::]", "*"}


def _normalize(addr: str) -> str:
    return str(addr).strip().lower()


def is_loopback_address(addr: str) -> bool:
    """Return True when the address refers to loopback."""

    normalized = _normalize(addr)
    if normalized in _LOOPBACK_HOSTNAMES:
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def is_unspecified_address(addr: str) -> bool:
    """Return True when the address is unspecified (e.g., 0.0.0.0, ::)."""

    normalized = _normalize(addr)
    if normalized in _UNSPECIFIED_SENTINELS:
        return True
    try:
        return ipaddress.ip_address(normalized).is_unspecified
    except ValueError:
        return False


def ensure_routable_address(addr: str, field_name: str = "node_address") -> None:
    """Raise when the supplied address is loopback or unspecified."""

    if not addr:
        raise ValidationError(f"{field_name} is required")
    if is_loopback_address(addr) or is_unspecified_address(addr):
        raise ValidationError(
            f"Invalid {field_name} '{addr}'. Use a routable (non-loopback, non-unspecified) IP of the external interface; "
            "127.0.0.1 and 0.0.0.0 are not allowed."
        )
