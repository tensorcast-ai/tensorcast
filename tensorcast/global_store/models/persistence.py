#  Copyright (c) 2025, TensorCast Team.

"""Placement and persistence domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class PlacementShard:
    """Shard summary for a placement plan."""

    plan_id: str
    shard_idx: int
    shard_id: str
    size_bytes: int
    content_digest: str
    byte_range_start: int
    byte_range_length: int
    chunk_ids: Sequence[int]
    degraded_reason: str | None = None


@dataclass
class PlacementTarget:
    """Target node for a shard."""

    plan_id: str
    shard_idx: int
    node_id: str
    lease_id: str | None
    target_state: str
    degraded_reason: str | None = None


@dataclass
class PlacementPlan:
    """Placement plan persisted in Global Store."""

    plan_id: str
    artifact_id: str
    policy: str
    shard_count: int
    shards: list[PlacementShard] = field(default_factory=list)
    targets: list[PlacementTarget] = field(default_factory=list)
    degraded_reason: str | None = None


@dataclass
class PersistenceStatus:
    """Aggregated persistence task status."""

    task_id: str
    plan_id: str
    artifact_id: str
    state: str
    progress: float
    last_error: str | None = None
    degraded_reason: str | None = None


@dataclass
class PersistenceShardStatus:
    """Shard-level persistence status."""

    shard_id: str
    shard_idx: int
    state: str
    progress: float
    degraded_reason: str | None = None
    last_error: str | None = None
    targets: list[PlacementTarget] = field(default_factory=list)
