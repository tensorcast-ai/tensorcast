#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import torch

from tensorcast.api._config import PlanType
from tensorcast.types import ServerConfig

TensorDict = Mapping[str, torch.Tensor]

SpanAttributeValue = bool | int | float | str

ArtifactStatusCode = Literal[
    "OK",
    "CANCELLED",
    "UNKNOWN",
    "INVALID_ARGUMENT",
    "DEADLINE_EXCEEDED",
    "NOT_FOUND",
    "ALREADY_EXISTS",
    "PERMISSION_DENIED",
    "RESOURCE_EXHAUSTED",
    "FAILED_PRECONDITION",
    "ABORTED",
    "OUT_OF_RANGE",
    "UNIMPLEMENTED",
    "INTERNAL",
    "UNAVAILABLE",
    "DATA_LOSS",
    "UNAUTHENTICATED",
]


class ArtifactError(RuntimeError):
    """Structured exception surfaced by Store verbs."""

    def __init__(
        self,
        message: str,
        *,
        status_code: ArtifactStatusCode,
        retryable: bool,
    ) -> None:
        super().__init__(message)
        self.status_code: ArtifactStatusCode = status_code
        self.retryable = bool(retryable)


@dataclass(frozen=True)
class RetryPolicy:
    deadline_seconds: float
    max_attempts: int
    base_backoff_seconds: float
    backoff_multiplier: float
    jitter: float


@dataclass(frozen=True)
class FallbackOptions:
    """Source selection and replica hints for materialization."""

    prefer: Literal["auto", "local", "p2p", "disk"] = "auto"
    disk_path: str | None = None
    allow_p2p: bool = True
    verify_checksums: bool = True
    prefer_disk: bool | None = None  # Deprecated compatibility flag
    replica_uuid: str | None = None

    @classmethod
    def for_disk(cls, path: str, *, verify: bool = True) -> "FallbackOptions":
        return cls(
            prefer="disk",
            disk_path=path,
            allow_p2p=False,
            verify_checksums=verify,
            prefer_disk=True,
        )

    @classmethod
    def local_only(cls) -> "FallbackOptions":
        return cls(
            prefer="local",
            allow_p2p=False,
            verify_checksums=True,
            prefer_disk=False,
        )


@dataclass(frozen=True)
class StoreOptions:
    fallback: FallbackOptions | None = None
    retry_overrides: Mapping[str, RetryPolicy] | None = None


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


ReplicaType = Literal["COALESCED_VRAM", "VRAM_LEASE_IN_PLACE", "VRAM_LEASED"]


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
    supports_coalesced: bool
    supports_lease: bool
    server_config: ServerConfig | None = None


__all__ = [
    "ArtifactError",
    "ArtifactStatusCode",
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
]
