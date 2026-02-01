#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import grpc

from tensorcast.proto.global_store.v1 import global_store_pb2, global_store_pb2_grpc

_T = TypeVar("_T")


@dataclass(slots=True)
class _CacheState(Generic[_T]):
    entries: list[_T] = field(default_factory=list)
    last_refresh: float | None = None
    next_refresh_at: float = 0.0
    backoff_s: float = 0.0


@dataclass(frozen=True, slots=True)
class CapabilityDirectoryOptions:
    target: str
    rpc_timeout_s: float = 3.0
    max_staleness_s: float = 10.0
    min_refresh_interval_s: float = 1.0
    max_backoff_s: float = 30.0


class CapabilityDirectoryClient:
    """Cached Global Store capability directory client with bounded staleness."""

    def __init__(self, options: CapabilityDirectoryOptions) -> None:
        self._options = options
        self._channel = grpc.insecure_channel(options.target)
        self._stub = global_store_pb2_grpc.GlobalStoreServiceStub(self._channel)
        self._lock = threading.RLock()
        self._worker_cache: dict[
            bool, _CacheState[global_store_pb2.ListActiveWorkersResponse.WorkerInfo]
        ] = {}
        self._instance_cache: dict[
            bool, _CacheState[global_store_pb2.ListActiveInstancesResponse.InstanceInfo]
        ] = {}

    def close(self) -> None:
        with self._lock:
            self._channel.close()

    def list_workers(
        self,
        *,
        required_capability_flags: int = 0,
        include_unavailable: bool = False,
    ) -> list[global_store_pb2.ListActiveWorkersResponse.WorkerInfo]:
        entries = self._get_workers(include_unavailable=include_unavailable)
        if required_capability_flags:
            required = int(required_capability_flags)
            entries = [
                worker
                for worker in entries
                if (int(worker.capability_flags) & required) == required
            ]
        return list(entries)

    def list_instances(
        self,
        *,
        required_capability_flags: int = 0,
        include_unavailable: bool = False,
    ) -> list[global_store_pb2.ListActiveInstancesResponse.InstanceInfo]:
        entries = self._get_instances(include_unavailable=include_unavailable)
        if required_capability_flags:
            required = int(required_capability_flags)
            entries = [
                inst
                for inst in entries
                if (int(inst.capability_flags) & required) == required
            ]
        return list(entries)

    # ------------------------------------------------------------------
    # Internal refresh helpers
    # ------------------------------------------------------------------
    def _get_workers(
        self, *, include_unavailable: bool
    ) -> list[global_store_pb2.ListActiveWorkersResponse.WorkerInfo]:
        state = self._worker_cache.setdefault(include_unavailable, _CacheState())
        self._maybe_refresh_workers(state, include_unavailable)
        return state.entries

    def _get_instances(
        self, *, include_unavailable: bool
    ) -> list[global_store_pb2.ListActiveInstancesResponse.InstanceInfo]:
        state = self._instance_cache.setdefault(include_unavailable, _CacheState())
        self._maybe_refresh_instances(state, include_unavailable)
        return state.entries

    def _maybe_refresh_workers(
        self,
        state: _CacheState[global_store_pb2.ListActiveWorkersResponse.WorkerInfo],
        include_unavailable: bool,
    ) -> None:
        now = time.monotonic()
        stale = state.last_refresh is None or (
            now - state.last_refresh > self._options.max_staleness_s
        )
        if not stale and now < state.next_refresh_at:
            return
        if now < state.next_refresh_at and state.last_refresh is not None:
            return
        try:
            response = self._stub.ListActiveWorkers(
                global_store_pb2.ListActiveWorkersRequest(
                    include_unavailable=include_unavailable
                ),
                timeout=self._options.rpc_timeout_s,
            )
            state.entries = list(response.workers)
            state.last_refresh = now
            state.backoff_s = self._options.min_refresh_interval_s
            state.next_refresh_at = now + self._options.min_refresh_interval_s
            return
        except grpc.RpcError as e:
            state.backoff_s = (
                self._options.min_refresh_interval_s
                if state.backoff_s <= 0
                else min(state.backoff_s * 2.0, self._options.max_backoff_s)
            )
            state.next_refresh_at = now + state.backoff_s
            if stale:
                raise RuntimeError("Capability directory worker cache is stale") from e

    def _maybe_refresh_instances(
        self,
        state: _CacheState[global_store_pb2.ListActiveInstancesResponse.InstanceInfo],
        include_unavailable: bool,
    ) -> None:
        now = time.monotonic()
        stale = state.last_refresh is None or (
            now - state.last_refresh > self._options.max_staleness_s
        )
        if not stale and now < state.next_refresh_at:
            return
        if now < state.next_refresh_at and state.last_refresh is not None:
            return
        try:
            response = self._stub.ListActiveInstances(
                global_store_pb2.ListActiveInstancesRequest(
                    include_unavailable=include_unavailable
                ),
                timeout=self._options.rpc_timeout_s,
            )
            state.entries = list(response.instances)
            state.last_refresh = now
            state.backoff_s = self._options.min_refresh_interval_s
            state.next_refresh_at = now + self._options.min_refresh_interval_s
            return
        except grpc.RpcError as e:
            state.backoff_s = (
                self._options.min_refresh_interval_s
                if state.backoff_s <= 0
                else min(state.backoff_s * 2.0, self._options.max_backoff_s)
            )
            state.next_refresh_at = now + state.backoff_s
            if stale:
                raise RuntimeError(
                    "Capability directory instance cache is stale"
                ) from e


__all__ = [
    "CapabilityDirectoryClient",
    "CapabilityDirectoryOptions",
]
