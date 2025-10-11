#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

from dataclasses import dataclass

from tensorcast.daemon_ctl import DaemonCtl


@dataclass(frozen=True, slots=True)
class RuntimeHandle:
    """Lightweight view over the active TensorCast runtime context."""

    address: str
    client: DaemonCtl


__all__ = ["RuntimeHandle"]
