#  Copyright (c) 2025, TensorCast Team.

# Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import asyncio
import contextlib
import queue
import threading
import time
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import torch

from tensorcast.api import _metrics as store_metrics
from tensorcast.api.store.types import ArtifactError
from tensorcast.proto.daemon.v1 import store_daemon_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2 as store_daemon_v2_pb2

if TYPE_CHECKING:
    from tensorcast.api.store.artifact import Artifact
    from tensorcast.api.store.materialization import MaterializationPipeline
    from tensorcast.api.store.runtime import StoreRuntimeContext


@dataclass(slots=True)
class PendingFetch:
    artifact_ref: weakref.ReferenceType["Artifact"]
    tensor_name: str
    device: torch.device | str
    target_loop: asyncio.AbstractEventLoop
    future: asyncio.Future[torch.Tensor]
    batch_key: tuple[str, str, str]
    submitted_at: float


@dataclass(slots=True)
class _BatchGroup:
    key: tuple[str, str, str]
    fetches: list[PendingFetch]
    deadline: float


class BatchContext:
    """Synchronous batching context manager for grouped tensor fetches."""

    def __init__(self, artifact: "Artifact", *, device: torch.device | str) -> None:
        self._artifact_ref = weakref.ref(artifact)
        self._device = device
        self._names: list[str] = []
        self._materialized: dict[str, torch.Tensor] | None = None
        self._lock = threading.Lock()
        self._closed = False

    def __enter__(self) -> "BatchContext":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._closed = True
        if exc_type is None:
            self._ensure_materialized()

    def add(self, name: str) -> None:
        if self._closed:
            raise ArtifactError(
                "BatchContext is closed",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        self._names.append(name)

    def _ensure_materialized(self) -> None:
        with self._lock:
            if self._materialized is not None:
                return
        artifact = self._artifact_ref()
        if artifact is None:
            raise ArtifactError(
                "Artifact no longer available",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        tensors = artifact.tensor_dict(device=self._device, names=self._names or None)
        with self._lock:
            self._materialized = tensors

    def get(self, name: str) -> torch.Tensor:
        self._ensure_materialized()
        assert self._materialized is not None
        if name not in self._materialized:
            raise ArtifactError(
                f"Tensor '{name}' not found in batch result",
                status_code="NOT_FOUND",
                retryable=False,
            )
        return self._materialized[name]

    @property
    def tensors(self) -> Mapping[str, torch.Tensor]:
        self._ensure_materialized()
        assert self._materialized is not None
        return dict(self._materialized)


class MaterializationBatcher:
    """Coalesces async tensor fetches within a short dispatch window."""

    def __init__(
        self,
        runtime: "StoreRuntimeContext",
        pipeline: "MaterializationPipeline",
        *,
        window_ms: float = 1.0,
    ) -> None:
        self._runtime = runtime
        self._pipeline_ref = weakref.ref(pipeline)
        self._window_ms = max(0.1, float(window_ms))
        self._pending_queue: queue.Queue[PendingFetch] = queue.Queue()
        self._groups: dict[tuple[str, str, str], _BatchGroup] = {}
        self._group_lock = threading.Lock()
        self._shutdown = threading.Event()
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop,
            daemon=True,
            name="MaterializationBatcher-Dispatch",
        )
        self._dispatch_thread.start()

    def close(self) -> None:
        if self._shutdown.is_set():
            return
        self._shutdown.set()
        self._dispatch_thread.join(timeout=1.0)

    @staticmethod
    def _device_key(device: torch.device | str) -> str:
        dev = torch.device(device)
        if dev.type == "cuda":
            return f"cuda:{dev.index or 0}"
        return str(dev)

    def submit(
        self,
        artifact: "Artifact",
        tensor_name: str,
        *,
        device: torch.device | str,
        view_hash: str | None,
        target_loop: asyncio.AbstractEventLoop,
    ) -> asyncio.Future[torch.Tensor]:
        artifact_id = artifact.artifact_id
        future: asyncio.Future[torch.Tensor] = target_loop.create_future()
        batch_key = (artifact_id, view_hash or "", self._device_key(device))
        fetch = PendingFetch(
            artifact_ref=weakref.ref(artifact),
            tensor_name=tensor_name,
            device=device,
            target_loop=target_loop,
            future=future,
            batch_key=batch_key,
            submitted_at=time.monotonic(),
        )
        self._pending_queue.put_nowait(fetch)
        return future

    def _dispatch_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                fetch = self._pending_queue.get(timeout=self._window_ms / 1000.0)
                self._assign(fetch)
            except queue.Empty:
                pass
            self._flush_ready()

    def _assign(self, fetch: PendingFetch) -> None:
        with self._group_lock:
            group = self._groups.get(fetch.batch_key)
            if group is None:
                deadline = time.monotonic() + self._window_ms / 1000.0
                group = _BatchGroup(
                    key=fetch.batch_key,
                    fetches=[],
                    deadline=deadline,
                )
                self._groups[fetch.batch_key] = group
            group.fetches.append(fetch)

    def _flush_ready(self) -> None:
        now = time.monotonic()
        ready: list[_BatchGroup] = []
        with self._group_lock:
            expired_keys = [
                key for key, grp in self._groups.items() if grp.deadline <= now
            ]
            ready.extend(self._groups.pop(key) for key in expired_keys)
        for group in ready:
            self._dispatch_group(group)

    def _dispatch_group(self, group: _BatchGroup) -> None:
        loop = self._runtime.event_loop
        start_time = min(fetch.submitted_at for fetch in group.fetches)
        window = max(0.0, time.monotonic() - start_time)
        store_metrics.record_batch_latency(self._runtime.daemon_endpoint, window)
        if len(group.fetches) > 1:
            store_metrics.increment_batch_coalesced(self._runtime.daemon_endpoint)
        coro = self._materialize_group(group)
        asyncio.run_coroutine_threadsafe(coro, loop)

    def _set_result(
        self, fetch: PendingFetch, tensor: torch.Tensor | None, error: Exception | None
    ) -> None:
        if fetch.future.cancelled() or fetch.future.done():
            return
        if error is not None:
            fetch.target_loop.call_soon_threadsafe(fetch.future.set_exception, error)
            return
        if tensor is None:
            err = ArtifactError(
                f"Tensor '{fetch.tensor_name}' missing from batch result",
                status_code="NOT_FOUND",
                retryable=False,
            )
            fetch.target_loop.call_soon_threadsafe(fetch.future.set_exception, err)
            return
        fetch.target_loop.call_soon_threadsafe(fetch.future.set_result, tensor)

    async def _materialize_group(self, group: _BatchGroup) -> None:
        fetches = group.fetches
        if not fetches:
            return
        artifact = fetches[0].artifact_ref()
        if artifact is None:
            for fetch in fetches:
                self._set_result(
                    fetch,
                    None,
                    ArtifactError(
                        "Artifact no longer available",
                        status_code="FAILED_PRECONDITION",
                        retryable=False,
                    ),
                )
            return
        names = [fetch.tensor_name for fetch in fetches]
        device = fetches[0].device
        try:
            tensors = await asyncio.to_thread(
                artifact.tensor_dict,
                device=device,
                names=names,
            )
        except Exception as exc:  # noqa: BLE001
            for fetch in fetches:
                self._set_result(fetch, None, exc)
            return

        for fetch in fetches:
            self._set_result(fetch, tensors.get(fetch.tensor_name), None)
            store_metrics.increment_batch_hit(
                self._runtime.daemon_endpoint, coalesced=len(fetches) > 1
            )


@dataclass(slots=True)
class PrefetchTicket:
    replica_uuid: str
    artifact_id: str
    device: torch.device
    expires_at: float | None
    started_at: float
    view_hash: str | None
    runtime_ref: weakref.ReferenceType["StoreRuntimeContext"]

    @property
    def is_expired(self) -> bool:
        return bool(self.expires_at is not None and time.monotonic() > self.expires_at)

    def _runtime(self) -> "StoreRuntimeContext":
        runtime = self.runtime_ref()
        if runtime is None or runtime.closed:
            raise ArtifactError(
                "Store runtime is closed",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        return runtime

    def wait(self, timeout: float | None = None) -> bool:
        if self.is_expired:
            store_metrics.record_prefetch_event(
                self._runtime().daemon_endpoint, status="expired"
            )
            return False
        runtime = self._runtime()
        client = runtime.ensure_client()
        ticket = store_daemon_v2_pb2.ReplicaTicket(replica_uuid=self.replica_uuid)
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if deadline is not None and time.monotonic() > deadline:
                store_metrics.record_prefetch_event(
                    runtime.daemon_endpoint, status="timeout"
                )
                return False
            if self.is_expired:
                store_metrics.record_prefetch_event(
                    runtime.daemon_endpoint, status="expired"
                )
                return False
            try:
                resp = client.query_replica_status(ticket)
                status = (
                    resp.ticket.status if resp and resp.HasField("ticket") else None
                )
                if status not in (
                    store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_ALLOCATED,
                    store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_UNSPECIFIED,
                ):
                    store_metrics.record_prefetch_event(
                        runtime.daemon_endpoint, status="ready"
                    )
                    return True
            except Exception:
                return False
            time.sleep(0.05)

    def cancel(self) -> None:
        runtime = self.runtime_ref()
        if runtime is None or runtime.closed:
            return
        client = runtime.ensure_client()
        ticket = store_daemon_v2_pb2.ReplicaTicket(replica_uuid=self.replica_uuid)
        with contextlib.suppress(Exception):
            client.release_replica(ticket)
            store_metrics.record_prefetch_event(
                runtime.daemon_endpoint, status="released"
            )


__all__ = ["BatchContext", "MaterializationBatcher", "PrefetchTicket"]
