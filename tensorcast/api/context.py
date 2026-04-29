#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

SpanAttributeValue = bool | int | float | str
QosClass = Literal["realtime", "interactive", "background"]


@dataclass(frozen=True, slots=True)
class CollectiveLoadGroup:
    """Explicit collective load contract for a single materialization call."""

    group_id: str
    world_size: int
    rank: int


@dataclass(frozen=True, slots=True)
class TransportSchedulingGroup:
    """Control-plane transport scheduling group for coordinated P2P source selection."""

    group_id: str
    group_kind: str
    total_parts: int
    part_id: str
    priority: int = 0
    epoch: int = 0
    request_id: str | None = None

    def __post_init__(self) -> None:
        group_kind = str(self.group_kind).strip()
        group_id = str(self.group_id).strip()
        part_id = str(self.part_id).strip()
        total_parts = int(self.total_parts)
        priority = int(self.priority)
        epoch = int(self.epoch)
        request_id = None if self.request_id is None else str(self.request_id).strip()
        if not group_kind:
            raise ValueError("TransportSchedulingGroup.group_kind must be non-empty")
        if not group_id:
            raise ValueError("TransportSchedulingGroup.group_id must be non-empty")
        if total_parts <= 0:
            raise ValueError("TransportSchedulingGroup.total_parts must be positive")
        if not part_id:
            raise ValueError("TransportSchedulingGroup.part_id must be non-empty")
        if priority < 0:
            raise ValueError("TransportSchedulingGroup.priority must be non-negative")
        if epoch < 0:
            raise ValueError("TransportSchedulingGroup.epoch must be non-negative")
        object.__setattr__(self, "group_kind", group_kind)
        object.__setattr__(self, "group_id", group_id)
        object.__setattr__(self, "total_parts", total_parts)
        object.__setattr__(self, "part_id", part_id)
        object.__setattr__(self, "priority", priority)
        object.__setattr__(self, "epoch", epoch)
        object.__setattr__(self, "request_id", request_id or None)


@dataclass(frozen=True, slots=True)
class GovernanceContext:
    """Typed low-cardinality governance hints propagated with a plan."""

    lane: str | None = None
    policy_version: int | None = None
    staleness_budget_ms: int | None = None


@dataclass(frozen=True, slots=True)
class CallContext:
    """Pure per-call container for deadlines, idempotency, and execution hints."""

    request_id: str | None = None
    qos: QosClass = "interactive"
    deadline_ms: int | None = None
    idempotency_key: str | None = None
    tags: Mapping[str, SpanAttributeValue] | None = None
    collective: CollectiveLoadGroup | None = None
    transport_group: TransportSchedulingGroup | None = None
    governance: GovernanceContext | None = None


def context(
    *,
    request_id: str | None = None,
    qos: QosClass = "interactive",
    deadline_ms: int | None = None,
    idempotency_key: str | None = None,
    tags: Mapping[str, SpanAttributeValue] | None = None,
    collective: CollectiveLoadGroup | None = None,
    transport_group: TransportSchedulingGroup | None = None,
    governance: GovernanceContext | None = None,
) -> CallContext:
    return CallContext(
        request_id=request_id,
        qos=qos,
        deadline_ms=deadline_ms,
        idempotency_key=idempotency_key,
        tags=tags,
        collective=collective,
        transport_group=transport_group,
        governance=governance,
    )


__all__ = [
    "CallContext",
    "CollectiveLoadGroup",
    "GovernanceContext",
    "QosClass",
    "SpanAttributeValue",
    "TransportSchedulingGroup",
    "context",
]
