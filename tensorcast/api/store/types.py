#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import torch
from pydantic import BaseModel, ConfigDict, field_validator

from tensorcast.api._config import PlanType
from tensorcast.api.errors import ArtifactError, ArtifactStatusCode
from tensorcast.types import ServerConfig

TensorDict = Mapping[str, torch.Tensor]

SpanAttributeValue = bool | int | float | str


FallbackPreference = Literal["auto", "local", "p2p", "disk"]


@dataclass(frozen=True)
class RetryPolicy:
    deadline_seconds: float
    max_attempts: int
    base_backoff_seconds: float
    backoff_multiplier: float
    jitter: float


class FallbackOptions(BaseModel):
    """Source selection and replica hints for materialization."""

    model_config = ConfigDict(frozen=True)

    prefer: FallbackPreference = "auto"
    allow_p2p: bool = True
    allow_disk: bool = True
    verify_checksums: bool = True
    prefer_disk: bool | None = None  # Deprecated compatibility flag
    replica_uuid: str | None = None

    @staticmethod
    def _to_prefer_literal(value: str) -> FallbackPreference:
        normalized = value.strip().lower()
        if normalized == "auto":
            return "auto"
        if normalized == "local":
            return "local"
        if normalized == "p2p":
            return "p2p"
        if normalized == "disk":
            return "disk"
        raise ArtifactError(
            f"Unknown fallback preference '{value}' (expected auto, local, p2p, or disk)",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )

    @field_validator("prefer", mode="before")
    @classmethod
    def _normalize_prefer(cls, value: object) -> FallbackPreference:
        normalized = "auto" if value is None else str(value).strip().lower()
        return cls._to_prefer_literal(normalized)

    @classmethod
    def local_only(cls) -> "FallbackOptions":
        return cls(
            prefer="local",
            allow_p2p=False,
            allow_disk=False,
            verify_checksums=True,
            prefer_disk=False,
        )

    @classmethod
    def parse(cls, value: object) -> "FallbackOptions | None":
        """Accept either a FallbackOptions instance or a string shortcut."""
        if value is None:
            return None
        if isinstance(value, FallbackOptions):
            return value
        if isinstance(value, str):
            raw = value.strip()
            if raw.lower().startswith("disk:"):
                raise ArtifactError(
                    "Fallback string 'disk:' is no longer supported; use Store.from_disk(...) to import "
                    "and then materialize by artifact_id or key.",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            normalized = raw.lower()
            if normalized in {"auto", "local", "p2p", "disk"}:
                if normalized == "local":
                    return cls.local_only()
                if normalized == "disk":
                    return cls(
                        prefer="disk",
                        prefer_disk=True,
                    )
                prefer_literal = cls._to_prefer_literal(normalized)
                return cls(prefer=prefer_literal)
        raise ArtifactError(
            "Fallback must be a FallbackOptions instance or string "
            "('auto', 'local', 'p2p', 'disk')",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )


class StoreOptions(BaseModel):
    model_config = ConfigDict(frozen=True)

    fallback: FallbackOptions | None = None
    retry_overrides: Mapping[str, RetryPolicy] | None = None

    @field_validator("fallback", mode="before")
    @classmethod
    def _coerce_fallback(cls, value: object) -> FallbackOptions | None:
        return FallbackOptions.parse(value)


@dataclass(frozen=True)
class CanonicalIndexEntry:
    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    segment_offset: int
    size_bytes: int


@dataclass(frozen=True)
class CanonicalIndex:
    entries: tuple[CanonicalIndexEntry, ...]
    total_size_bytes: int
    avbs_hash: str


ReplicaType = Literal[
    "COALESCED_VRAM",
    "DRAM_STABLE",
    "VRAM_LEASE_IN_PLACE",
    "VRAM_LEASED",
]


@dataclass(frozen=True)
class ReplicaInfo:
    replica_id: str
    replica_type: ReplicaType
    device: torch.device
    plan: PlanType
    size_bytes: int


@dataclass(frozen=True)
class LeaseHandle:
    lease_id: str
    ttl_ms: int
    expires_at_monotonic: float
    owner_pid: int


@dataclass(frozen=True)
class StoreCapabilities:
    mem_pool_bytes: int
    tx_slice_bytes: int
    artifact_chunk_bytes: int
    server_config: ServerConfig | None = None


@dataclass(frozen=True)
class PersistenceShardStatus:
    shard_id: str
    shard_idx: int
    state: str
    progress: float
    degraded_reason: str | None = None
    last_error: str | None = None
    target_nodes: tuple[str, ...] = ()
    lease_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class PersistenceStatusResult:
    task_id: str
    artifact_id: str
    plan_id: str
    state: str
    progress: float
    degraded_reason: str | None = None
    last_error: str | None = None
    shards: tuple[PersistenceShardStatus, ...] = ()


__all__ = [
    "ArtifactError",
    "ArtifactStatusCode",
    "FallbackPreference",
    "CanonicalIndex",
    "CanonicalIndexEntry",
    "FallbackOptions",
    "LeaseHandle",
    "ReplicaInfo",
    "ReplicaType",
    "RetryPolicy",
    "SpanAttributeValue",
    "StoreCapabilities",
    "StoreOptions",
    "TensorDict",
    "PersistenceStatusResult",
    "PersistenceShardStatus",
]
