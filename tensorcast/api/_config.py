#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from enum import Enum

from ._errors import InvalidPlan

# Module-level constants
DEFAULT_ALIGN: int = 8
DEFAULT_PINNED_TIMEOUT_MS: int = 30000
# Final model: no default CPU stream chunking heuristic


# Global daemon address configuration
_daemon_address_lock = threading.RLock()
_global_daemon_address: str | None = None
_global_store_address = os.environ.get("TENSORCAST_GLOBAL_STORE", "127.0.0.1:8085")


def set_daemon_address(address: str) -> None:
    with _daemon_address_lock:
        global _global_daemon_address
        _global_daemon_address = address


def get_daemon_address() -> str:
    with _daemon_address_lock:
        if _global_daemon_address is None:
            raise RuntimeError(
                "TensorCast runtime is not initialized. Call tensorcast.startup.init() first."
            )
        return _global_daemon_address


def clear_daemon_address() -> None:
    with _daemon_address_lock:
        global _global_daemon_address
        _global_daemon_address = None


def has_daemon_address() -> bool:
    with _daemon_address_lock:
        return _global_daemon_address is not None


def set_global_store_address(address: str) -> None:
    global _global_store_address
    _global_store_address = address


def get_global_store_address() -> str:
    return _global_store_address


class PlanType(Enum):
    VRAM_COALESCED = "vram_coalesced"
    VRAM_LEASED = "vram_leased"

    @staticmethod
    def parse(value: "PlanType | str") -> "PlanType":
        if isinstance(value, PlanType):
            return value
        s = str(value).strip().lower()
        if s in ("vram_coalesced", "coalesced"):
            return PlanType.VRAM_COALESCED
        if s in ("vram_leased", "lease"):
            return PlanType.VRAM_LEASED
        raise InvalidPlan(f"Unknown plan: {value}")


@dataclass(slots=True, frozen=True)
class RegisterArtifactOptions:
    plan: PlanType | str = PlanType.VRAM_COALESCED
    p2p_prefer: str = "vram"
    max_inflight_bytes: int = 512 * 1024 * 1024
    release_on_tensor_commit: bool = True
    min_tensor_bytes: int = 64 * 1024
    max_tensor_count: int = 8192
    lease_bytes_limit: int = 0
    # Lease/LIP specific: opt-in in-place mode per RFC-0014
    lease_in_place: bool = False
    # CPU path removed
    key: str | None = None
    disk_path: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan", PlanType.parse(self.plan))


@dataclass(slots=True, frozen=True)
class GetArtifactOptions:
    prefer: str = "p2p"  # "p2p" or "disk"
    pinned_allocation_timeout_ms: int = DEFAULT_PINNED_TIMEOUT_MS
    wait_for_completion: bool = True
    enable_verification: bool = True
