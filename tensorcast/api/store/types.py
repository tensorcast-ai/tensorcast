#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import torch
from pydantic import BaseModel, ConfigDict

from tensorcast.api._config import GetArtifactOptions, PlanType
from tensorcast.api.errors import ArtifactError, ArtifactStatusCode
from tensorcast.types import ServerConfig

TensorDict = Mapping[str, torch.Tensor]

SpanAttributeValue = bool | int | float | str


@dataclass(frozen=True)
class RetryPolicy:
    deadline_seconds: float
    max_attempts: int
    base_backoff_seconds: float
    backoff_multiplier: float
    jitter: float


class StoreOptions(BaseModel):
    model_config = ConfigDict(frozen=True)

    get: GetArtifactOptions | None = None
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
    "CanonicalIndex",
    "CanonicalIndexEntry",
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
