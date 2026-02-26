#  Copyright (c) 2025-2026, TensorCast Team.

"""Pending transport request queue model."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from tensorcast.global_store.models.transport import TransportSchedulingGroup


class PendingTransportState(str, Enum):
    """Lifecycle state for queued transport requests."""

    ENQUEUED = "enqueued"
    DISPATCHED = "dispatched"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass
class PendingTransportRequest:
    """Queue record used by group-dispatch scheduler mode."""

    request_id: str
    request_fingerprint: str
    artifact_id: str
    requested_view_id: str | None
    source_node_id: str
    source_address: str
    source_port: int
    requester_worker_id: str | None = None
    state: PendingTransportState = PendingTransportState.ENQUEUED
    deadline_at: datetime | None = None
    created_at: datetime | None = None
    dispatched_at: datetime | None = None
    updated_at: datetime | None = None

    # Scheduling-group fields
    group_id: str | None = None
    group_kind: str | None = None
    group_total_parts: int | None = None
    group_part_id: str | None = None
    group_priority: int | None = None
    group_epoch: int | None = None

    @property
    def scheduling_group(self) -> TransportSchedulingGroup | None:
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
