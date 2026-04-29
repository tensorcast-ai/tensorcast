#  Copyright (c) 2026, TensorCast Team.

"""Broadcast domain models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from uuid import UUID


class BroadcastSessionState(str, Enum):
    """Lifecycle states for a broadcast session."""

    PLANNING = "planning"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BroadcastTargetState(str, Enum):
    """Lifecycle states for one broadcast target worker."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    MATERIALIZING = "materializing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BroadcastEdgeState(str, Enum):
    """Lifecycle states for one parent-child broadcast attempt."""

    PLANNED = "planned"
    ASSIGNED = "assigned"
    MATERIALIZING = "materializing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BroadcastSession:
    """Persistent state for a tree broadcast attempt."""

    session_id: str
    artifact_id: str
    requested_view_id: str | None
    epoch: int
    fanout: int
    max_attempts: int
    strict_parent: bool
    state: BroadcastSessionState = BroadcastSessionState.PLANNING
    root_replica_id: UUID | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass
class BroadcastTarget:
    """Persistent state for one target in a broadcast session."""

    session_id: str
    target_worker_id: str
    target_daemon_id: str | None
    state: BroadcastTargetState = BroadcastTargetState.PENDING
    level: int | None = None
    attempt: int = 0
    assigned_edge_id: str | None = None
    completed_replica_id: UUID | None = None
    failure_reason: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass
class BroadcastEdge:
    """Persistent state for one parent-child broadcast edge attempt."""

    edge_id: str
    session_id: str
    parent_worker_id: str
    parent_replica_id: UUID
    child_worker_id: str
    level: int
    attempt: int = 1
    state: BroadcastEdgeState = BroadcastEdgeState.PLANNED
    transport_request_id: str | None = None
    failure_reason: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None
