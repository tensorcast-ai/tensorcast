#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

SpanAttributeValue = bool | int | float | str
QosClass = Literal["realtime", "interactive", "background"]


@dataclass(frozen=True, slots=True)
class CallContext:
    """Pure per-call container for deadlines, idempotency, and trace tags."""

    request_id: str | None = None
    qos: QosClass = "interactive"
    deadline_ms: int | None = None
    idempotency_key: str | None = None
    tags: Mapping[str, SpanAttributeValue] | None = None


def context(
    *,
    request_id: str | None = None,
    qos: QosClass = "interactive",
    deadline_ms: int | None = None,
    idempotency_key: str | None = None,
    tags: Mapping[str, SpanAttributeValue] | None = None,
) -> CallContext:
    return CallContext(
        request_id=request_id,
        qos=qos,
        deadline_ms=deadline_ms,
        idempotency_key=idempotency_key,
        tags=tags,
    )


__all__ = [
    "CallContext",
    "QosClass",
    "SpanAttributeValue",
    "context",
]
