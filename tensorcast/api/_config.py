#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

from ._errors import InvalidPlan

# Module-level constants
DEFAULT_ALIGN: int = 8
DEFAULT_PINNED_TIMEOUT_MS: int = 30000
DEFAULT_CPU_CHUNK_SIZE: int = 4 * 1024 * 1024


# Global daemon address configuration
_global_daemon_address = "127.0.0.1:8073"
_global_store_address = os.environ.get("TENSORCAST_GLOBAL_STORE", "127.0.0.1:8085")


def set_daemon_address(address: str) -> None:
    global _global_daemon_address
    _global_daemon_address = address


def get_daemon_address() -> str:
    return _global_daemon_address


def set_global_store_address(address: str) -> None:
    global _global_store_address
    _global_store_address = address


def get_global_store_address() -> str:
    return _global_store_address


class PlanType(Enum):
    VRAM_COALESCED = "vram_coalesced"
    CPU = "cpu"
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
        if s in ("cpu", "uma"):
            return PlanType.CPU
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
    cpu_preferred_channel: int = 2
    cpu_ring_bytes: int = 0
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
