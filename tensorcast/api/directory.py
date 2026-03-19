#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Generic, TypeVar

from tensorcast.daemon_ctl import DaemonCtl

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class DirectorySnapshot(Generic[T]):
    value: T
    as_of_ms: int
    staleness_ms: int
    cache_epoch: int | None = None
    freshness_state: str = "unknown"
    authority_mode: str = "unknown"


@dataclass(frozen=True, slots=True)
class WorkerRoute:
    daemon_id: str
    worker_id: str | None = None
    daemon_address: str | None = None
    capability_flags: int = 0


@dataclass(frozen=True, slots=True)
class InstanceExecutionRoute:
    instance_id: str
    daemon_id: str
    execution_host_kind: str
    execution_endpoint: str
    engine: str | None = None
    capability_flags: int = 0


def _snapshot_fields(message: object) -> tuple[int, int, int | None, str, str]:
    now_ms = int(time.time() * 1000)
    as_of_ms = int(getattr(message, "as_of_ms", 0) or 0)
    staleness_ms = int(getattr(message, "staleness_ms", 0) or 0)
    cache_epoch_raw = int(getattr(message, "cache_epoch", 0) or 0)
    freshness_state = str(getattr(message, "freshness_state", "") or "")
    authority_mode = str(getattr(message, "authority_mode", "") or "")
    if as_of_ms <= 0:
        as_of_ms = now_ms
        staleness_ms = 0
        freshness_state = freshness_state or "current"
    return (
        as_of_ms,
        staleness_ms,
        cache_epoch_raw if cache_epoch_raw > 0 else None,
        freshness_state or "current",
        authority_mode or "unknown",
    )


class TensorCastDirectory:
    def __init__(self, client: DaemonCtl) -> None:
        self._client = client

    def list_workers(
        self,
        *,
        include_unavailable: bool = False,
        required_capability_flags: int = 0,
        max_staleness_ms: int | None = None,
    ) -> DirectorySnapshot[list[WorkerRoute]]:
        response = self._client.list_directory_workers(
            include_unavailable=include_unavailable,
            required_capability_flags=required_capability_flags,
            max_staleness_ms=max_staleness_ms,
        )
        as_of_ms, staleness_ms, cache_epoch, freshness_state, authority_mode = (
            _snapshot_fields(response)
        )
        return DirectorySnapshot(
            value=[
                WorkerRoute(
                    daemon_id=str(worker.daemon_id),
                    worker_id=str(worker.worker_id) or None,
                    daemon_address=str(worker.daemon_address) or None,
                    capability_flags=int(worker.capability_flags),
                )
                for worker in response.workers
            ],
            as_of_ms=as_of_ms,
            staleness_ms=staleness_ms,
            cache_epoch=cache_epoch,
            freshness_state=freshness_state,
            authority_mode=authority_mode,
        )

    def list_instances(
        self,
        *,
        include_unavailable: bool = False,
        required_capability_flags: int = 0,
        max_staleness_ms: int | None = None,
    ) -> DirectorySnapshot[list[InstanceExecutionRoute]]:
        response = self._client.list_directory_instances(
            include_unavailable=include_unavailable,
            required_capability_flags=required_capability_flags,
            max_staleness_ms=max_staleness_ms,
        )
        as_of_ms, staleness_ms, cache_epoch, freshness_state, authority_mode = (
            _snapshot_fields(response)
        )
        return DirectorySnapshot(
            value=[
                InstanceExecutionRoute(
                    instance_id=str(instance.instance_id),
                    daemon_id=str(instance.daemon_id),
                    execution_host_kind=str(instance.execution_host_kind),
                    execution_endpoint=str(instance.execution_endpoint),
                    engine=str(instance.engine) or None,
                    capability_flags=int(instance.capability_flags),
                )
                for instance in response.instances
            ],
            as_of_ms=as_of_ms,
            staleness_ms=staleness_ms,
            cache_epoch=cache_epoch,
            freshness_state=freshness_state,
            authority_mode=authority_mode,
        )

    def resolve_instance_execution(
        self,
        instance_id: str,
        *,
        max_staleness_ms: int | None = None,
    ) -> DirectorySnapshot[InstanceExecutionRoute]:
        response = self._client.resolve_instance_execution(
            instance_id=instance_id,
            max_staleness_ms=max_staleness_ms,
        )
        as_of_ms, staleness_ms, cache_epoch, freshness_state, authority_mode = (
            _snapshot_fields(response)
        )
        route = response.route
        return DirectorySnapshot(
            value=InstanceExecutionRoute(
                instance_id=str(route.instance_id),
                daemon_id=str(route.daemon_id),
                execution_host_kind=str(route.execution_host_kind),
                execution_endpoint=str(route.execution_endpoint),
                engine=str(route.engine) or None,
                capability_flags=int(route.capability_flags),
            ),
            as_of_ms=as_of_ms,
            staleness_ms=staleness_ms,
            cache_epoch=cache_epoch,
            freshness_state=freshness_state,
            authority_mode=authority_mode,
        )


__all__ = [
    "DirectorySnapshot",
    "InstanceExecutionRoute",
    "TensorCastDirectory",
    "WorkerRoute",
]
