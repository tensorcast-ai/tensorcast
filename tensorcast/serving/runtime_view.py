#  Copyright (c) 2026, TensorCast Team.
"""Framework-neutral runtime view aggregation helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

_PENDING_PUBLICATION_STATES = {"publishing", "retiring"}


def _worker_publication_state(payload: Mapping[str, Any]) -> str:
    projection = payload.get("published_replica")
    if not isinstance(projection, Mapping):
        return "unpublished"
    state = str(projection.get("state") or "").strip().lower()
    return state or "unpublished"


def publication_aggregate(
    worker_payloads: Iterable[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Aggregate per-worker published replica projections.

    The aggregate is intentionally conservative: a single published worker is
    ``partial`` until every required worker reports ``published``.
    """

    payloads = [dict(payload) for payload in worker_payloads]
    if not any("published_replica" in payload for payload in payloads):
        return None

    states = [_worker_publication_state(payload) for payload in payloads]
    required_workers = len(states)
    published_workers = sum(state == "published" for state in states)
    failed_workers = sum(state == "failed" for state in states)
    stale_workers = sum(state == "stale" for state in states)
    pending_workers = sum(state in _PENDING_PUBLICATION_STATES for state in states)
    disabled_workers = sum(state == "disabled" for state in states)

    if failed_workers:
        aggregate_state = "failed"
    elif stale_workers:
        aggregate_state = "stale"
    elif published_workers == required_workers:
        aggregate_state = "published"
    elif published_workers:
        aggregate_state = "partial"
    elif pending_workers:
        aggregate_state = "publishing"
    elif disabled_workers == required_workers:
        aggregate_state = "disabled"
    else:
        aggregate_state = "unpublished"

    return {
        "schema_version": 1,
        "state": aggregate_state,
        "mode": "runtime_view",
        "published_workers": published_workers,
        "required_workers": required_workers,
        "failed_workers": failed_workers,
        "pending_workers": pending_workers,
        "stale_workers": stale_workers,
    }


def aggregate_runtime_view_outputs(
    outputs: Iterable[Any],
    *,
    response_name: str,
) -> dict[str, Any] | None:
    """Aggregate runtime endpoint payloads returned by multiple workers."""

    worker_payloads: list[dict[str, Any]] = []
    for payload in outputs:
        if payload is None:
            continue
        if not isinstance(payload, dict):
            raise RuntimeError(f"{response_name} worker response must be a dict")
        worker_payloads.append(dict(payload))
    if not worker_payloads:
        return None

    result = dict(worker_payloads[0])
    aggregate = publication_aggregate(worker_payloads)
    if aggregate is not None:
        result["publication_aggregate"] = aggregate
    return result


__all__ = [
    "aggregate_runtime_view_outputs",
    "publication_aggregate",
]
