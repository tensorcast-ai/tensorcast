#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import os
import threading
from enum import Enum

from pydantic import BaseModel, ConfigDict, field_validator

from tensorcast.api._errors import InvalidPlan

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
    def parse(value: object) -> "PlanType":
        if isinstance(value, PlanType):
            return value
        s = str(value).strip().lower()
        if s in ("vram_coalesced", "coalesced", "copy"):
            return PlanType.VRAM_COALESCED
        if s in ("vram_leased", "lease"):
            return PlanType.VRAM_LEASED
        raise InvalidPlan(
            f"Unknown plan '{value}'; expected 'lease' or 'copy' (or PlanType enum)."
        )


class PlacementPolicy(Enum):
    LOCAL_ONLY = "local_only"
    REPLICATED = "replicated"
    SHARDED = "sharded"

    @staticmethod
    def parse(value: object) -> "PlacementPolicy":
        if isinstance(value, PlacementPolicy):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"local_only", "local"}:
            return PlacementPolicy.LOCAL_ONLY
        if normalized in {"replicated", "replica"}:
            return PlacementPolicy.REPLICATED
        if normalized in {"sharded", "shard"}:
            return PlacementPolicy.SHARDED
        raise ValueError(
            f"Unknown placement_policy '{value}'; expected local_only, replicated, or sharded."
        )


class RegisterArtifactOptions(BaseModel):
    model_config = ConfigDict(frozen=True)

    plan: PlanType = PlanType.VRAM_LEASED
    placement_policy: PlacementPolicy = PlacementPolicy.LOCAL_ONLY
    persist: bool = False
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

    @field_validator("plan", mode="before")
    @classmethod
    def _normalize_plan(cls, value: object) -> PlanType:
        try:
            return PlanType.parse(value)
        except InvalidPlan:
            raise
        except Exception as exc:  # noqa: BLE001
            raise InvalidPlan(str(exc)) from exc

    @field_validator("placement_policy", mode="before")
    @classmethod
    def _normalize_placement_policy(cls, value: object) -> PlacementPolicy:
        try:
            return PlacementPolicy.parse(value)
        except Exception as exc:  # noqa: BLE001
            raise InvalidPlan(str(exc)) from exc


class GetArtifactOptions(BaseModel):
    model_config = ConfigDict(frozen=True)

    prefer: str = "auto"  # "auto" | "local" | "p2p" | "disk"
    pinned_allocation_timeout_ms: int = DEFAULT_PINNED_TIMEOUT_MS
    wait_for_completion: bool = True
    enable_verification: bool = True
    # Hint for P2P transport lock TTL extension; forwarded by daemons
    transport_hold_ms: int | None = None

    @field_validator("prefer", mode="before")
    @classmethod
    def _normalize_prefer(cls, value: object) -> str:
        normalized = "auto" if value is None else str(value).strip().lower()
        if normalized not in {"auto", "local", "p2p", "disk"}:
            raise ValueError(
                "GetArtifactOptions.prefer must be one of: auto, local, p2p, disk"
            )
        return normalized


__all__ = [
    "DEFAULT_ALIGN",
    "DEFAULT_PINNED_TIMEOUT_MS",
    "GetArtifactOptions",
    "PlanType",
    "PlacementPolicy",
    "RegisterArtifactOptions",
    "clear_daemon_address",
    "get_daemon_address",
    "get_global_store_address",
    "has_daemon_address",
    "set_daemon_address",
    "set_global_store_address",
]
