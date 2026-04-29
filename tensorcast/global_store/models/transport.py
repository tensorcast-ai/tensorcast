#  Copyright (c) 2025-2026, TensorCast Team.

"""Transport domain artifact."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4


class TransportCompletionOutcome(str, Enum):
    """Completion outcome reported by requester-side transport lifecycle."""

    UNSPECIFIED = "unspecified"
    SUCCESS = "success"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class BroadcastTransportHint:
    """Optional broadcast-tree transport routing hint."""

    session_id: str
    strict_parent: bool = True


@dataclass(frozen=True)
class TransportSchedulingGroup:
    """Optional scheduling-group metadata attached to transport requests."""

    group_id: str
    group_kind: str
    total_parts: int
    part_id: str
    priority: int = 0
    epoch: int = 0


@dataclass
class Transport:
    """Represents an active artifact transport between nodes."""

    # Identity
    transport_id: UUID = field(default_factory=uuid4)
    replica_id: UUID = field(default_factory=uuid4)
    artifact_id: str = ""  # content-addressed ID
    requested_view_id: str | None = None
    disk_path: str | None = None

    # Source information
    source_node_id: str = ""
    source_address: str = ""
    source_port: int = 0
    # Immutable bytes snapshot captured at dispatch time for transport analytics.
    replica_memory_size_bytes: int | None = None
    requester_worker_id: str | None = None
    request_id: str | None = None
    request_fingerprint: str | None = None
    broadcast_session_id: str | None = None
    broadcast_edge_id: str | None = None

    # Optional scheduling-group metadata
    group_id: str | None = None
    group_kind: str | None = None
    group_total_parts: int | None = None
    group_part_id: str | None = None
    group_priority: int | None = None
    group_epoch: int | None = None

    # Timestamps
    created_at: datetime | None = None
    completed_at: datetime | None = None

    # Status
    status: str = "in_progress"  # in_progress, completed
    completion_outcome: TransportCompletionOutcome = (
        TransportCompletionOutcome.UNSPECIFIED
    )
    completion_detail: str | None = None

    @property
    def age_seconds(self) -> float:
        """Get the age of this transport in seconds."""
        if not self.created_at:
            return 0.0

        # Ensure we compare *aware* vs *aware* or *naive* vs *naive* datetimes
        now = (
            datetime.now(tz=self.created_at.tzinfo)
            if self.created_at.tzinfo is not None
            else datetime.now()
        )
        return (now - self.created_at).total_seconds()

    @property
    def duration_seconds(self) -> float | None:
        """Get the duration of this transport in seconds if completed."""
        if not self.created_at or not self.completed_at:
            return None
        return (self.completed_at - self.created_at).total_seconds()

    @property
    def scheduling_group(self) -> TransportSchedulingGroup | None:
        """Return scheduling group metadata when all required fields are present."""
        if (
            not self.group_id
            or not self.group_kind
            or self.group_total_parts is None
            or not self.group_part_id
        ):
            return None
        return TransportSchedulingGroup(
            group_id=self.group_id,
            group_kind=self.group_kind,
            total_parts=int(self.group_total_parts),
            part_id=self.group_part_id,
            priority=int(self.group_priority or 0),
            epoch=int(self.group_epoch or 0),
        )

    def set_scheduling_group(self, group: TransportSchedulingGroup | None) -> None:
        """Attach or clear scheduling-group metadata."""
        if group is None:
            self.group_id = None
            self.group_kind = None
            self.group_total_parts = None
            self.group_part_id = None
            self.group_priority = None
            self.group_epoch = None
            return
        self.group_id = group.group_id
        self.group_kind = group.group_kind
        self.group_total_parts = int(group.total_parts)
        self.group_part_id = group.part_id
        self.group_priority = int(group.priority)
        self.group_epoch = int(group.epoch)
