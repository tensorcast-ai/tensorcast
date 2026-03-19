#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Generic, TypeVar

from tensorcast.api.directory import (
    DirectorySnapshot,
    InstanceExecutionRoute,
    TensorCastDirectory,
    WorkerRoute,
)
from tensorcast.daemon_ctl import DaemonCtl

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class SignalSnapshot(Generic[T]):
    value: T
    as_of_ms: int
    staleness_ms: int
    cache_epoch: int | None = None
    freshness_state: str = "current"


@dataclass(frozen=True, slots=True)
class WorkerStatus:
    worker_id: str | None
    daemon_id: str | None
    is_registered: bool
    is_healthy: bool
    is_shutting_down: bool
    mem_pool_total_size: int
    mem_pool_available_size: int
    uptime_seconds: int


class TensorCastSignals:
    def __init__(self, client: DaemonCtl) -> None:
        self._client = client

    def get_worker_status(self) -> SignalSnapshot[WorkerStatus]:
        response = self._client.get_worker_status()
        now_ms = int(time.time() * 1000)
        as_of_ms = int(getattr(response, "as_of_ms", 0) or 0)
        staleness_ms = int(getattr(response, "staleness_ms", 0) or 0)
        cache_epoch_raw = int(getattr(response, "cache_epoch", 0) or 0)
        freshness_state = str(getattr(response, "freshness_state", "") or "")
        if as_of_ms <= 0:
            as_of_ms = now_ms
            staleness_ms = 0
            freshness_state = freshness_state or "current"
        return SignalSnapshot(
            value=WorkerStatus(
                worker_id=str(response.worker_id) or None,
                daemon_id=str(response.daemon_id) or None,
                is_registered=bool(response.is_registered),
                is_healthy=bool(response.is_healthy),
                is_shutting_down=bool(response.is_shutting_down),
                mem_pool_total_size=int(response.mem_pool_total_size),
                mem_pool_available_size=int(response.mem_pool_available_size),
                uptime_seconds=int(response.uptime_seconds),
            ),
            as_of_ms=as_of_ms,
            staleness_ms=staleness_ms,
            cache_epoch=cache_epoch_raw if cache_epoch_raw > 0 else None,
            freshness_state=freshness_state or "current",
        )

    def list_workers(
        self,
        *,
        include_unavailable: bool = False,
        required_capability_flags: int = 0,
        max_staleness_ms: int | None = None,
    ) -> DirectorySnapshot[list[WorkerRoute]]:
        return TensorCastDirectory(self._client).list_workers(
            include_unavailable=include_unavailable,
            required_capability_flags=required_capability_flags,
            max_staleness_ms=max_staleness_ms,
        )

    def list_instances(
        self,
        *,
        include_unavailable: bool = False,
        required_capability_flags: int = 0,
        max_staleness_ms: int | None = None,
    ) -> DirectorySnapshot[list[InstanceExecutionRoute]]:
        return TensorCastDirectory(self._client).list_instances(
            include_unavailable=include_unavailable,
            required_capability_flags=required_capability_flags,
            max_staleness_ms=max_staleness_ms,
        )


__all__ = [
    "SignalSnapshot",
    "TensorCastSignals",
    "WorkerStatus",
]
