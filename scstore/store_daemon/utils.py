#  Copyright (c) 2025, TensorCast Team.

"""Utility classes and functions for StoreDaemon."""

from __future__ import annotations

import json
from threading import Lock
from typing import Any


class AtomicCounter:
    """Thread-safe counter."""

    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = Lock()

    def increment(self) -> int:
        """Increment the counter and return the new value."""
        with self._lock:
            self._value += 1
            return self._value

    def decrement(self) -> int:
        """Decrement the counter and return the new value."""
        with self._lock:
            self._value -= 1
            return self._value

    def get(self) -> int:
        """Get the current value of the counter."""
        with self._lock:
            return self._value

    def set(self, value: int) -> None:
        """Set the counter to a specific value."""
        with self._lock:
            self._value = value


def resolve_device_id(device_uuid: str, default: int = 0) -> int:
    """Resolve a CUDA device ordinal from its UUID.

    Falls back to *default* if the mapping cannot be resolved – this keeps the
    daemon functional even when the C++ extension is unavailable (for example
    in CPU-only CI environments or unit tests).
    """
    try:
        from scstore import _C as _ckpt_helpers  # pylint: disable=import-error

        dev_uuid_map = _ckpt_helpers.get_device_uuid_map()
        for d_id, d_uuid in dev_uuid_map.items():
            if d_uuid == device_uuid:
                return d_id  # pyright: ignore[reportReturnType]
    except Exception:  # pragma: no cover – best-effort helper
        pass

    return default


def read_verification_json(path: str) -> dict[str, Any]:
    """Read *verification.json* and return the parsed dict.

    Returns an empty dict on any error (file missing, JSON decode error, etc.)
    so callers don't need to guard for exceptions.
    """
    try:
        with open(path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:  # pragma: no cover – best-effort helper
        return {}
