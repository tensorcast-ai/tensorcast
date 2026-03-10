#  Copyright (c) 2026, TensorCast Team.

# Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import os

_DEBUG_ERRORS_ENV = "TENSORCAST_DEBUG_ERRORS"
_TRUTHY = {"1", "true", "yes", "on"}


def debug_errors_enabled() -> bool:
    value = os.environ.get(_DEBUG_ERRORS_ENV)
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY


def debug_errors_hint() -> str:
    return f"Set {_DEBUG_ERRORS_ENV}=1 to show full RPC error details."


__all__ = [
    "debug_errors_enabled",
    "debug_errors_hint",
]
