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
class GroupVersionSetRef:
    """Daemon-returned immutable group version-set reference."""

    version_set_id: str
    manifest_hash: bytes = b""
    manifest_generation: int = 0


@dataclass(frozen=True, slots=True)
class GroupRealization:
    """Daemon-mediated semantic group realization for one materialization call."""

    group_kind: str
    group_id: str
    epoch: int
    total_parts: int
    part_id: str
    required_part_ids: tuple[str, ...]
    require_staged_publish: bool = False
    deadline_unix_nanos: int = 0
    version_set: GroupVersionSetRef | None = None


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
    group_realization: GroupRealization | None = None
    governance: GovernanceContext | None = None


def context(
    *,
    request_id: str | None = None,
    qos: QosClass = "interactive",
    deadline_ms: int | None = None,
    idempotency_key: str | None = None,
    tags: Mapping[str, SpanAttributeValue] | None = None,
    collective: CollectiveLoadGroup | None = None,
    group_realization: GroupRealization | None = None,
    governance: GovernanceContext | None = None,
) -> CallContext:
    return CallContext(
        request_id=request_id,
        qos=qos,
        deadline_ms=deadline_ms,
        idempotency_key=idempotency_key,
        tags=tags,
        collective=collective,
        group_realization=group_realization,
        governance=governance,
    )


__all__ = [
    "CallContext",
    "CollectiveLoadGroup",
    "GroupRealization",
    "GroupVersionSetRef",
    "GovernanceContext",
    "QosClass",
    "SpanAttributeValue",
    "context",
]
