#  Copyright (c) 2025-2026, TensorCast Team.

"""Single-writer reducer for worker control-plane mutations."""

from __future__ import annotations

import queue
import threading
import time
import zlib
from concurrent.futures import Future
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast

from tensorcast.global_store.metrics import (
    inc_worker_control_reducer_intent,
    observe_worker_control_reducer_queue_latency,
    set_worker_control_reducer_queue_depth,
)
from tensorcast.global_store.repositories.base import bind_tx_context
from tensorcast.global_store.services.worker_control_intents import WorkerControlIntent

T = TypeVar("T")
_STOP_SENTINEL = object()


class ReducerOverloadedError(RuntimeError):
    """Raised when reducer queues are full and cannot accept new intents."""


@dataclass
class _ReducerShard:
    shard_id: int
    queue: queue.Queue[object]
    thread: threading.Thread | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    pending_heartbeats: dict[str, WorkerControlIntent[object]] = field(
        default_factory=dict
    )


class WorkerControlReducer:
    """Sharded reducer that serializes worker control-plane writes."""

    def __init__(
        self,
        *,
        shard_count: int,
        queue_capacity: int,
        coalesce_window_ms: int,
        logger,
    ) -> None:
        self._logger = logger
        self._shard_count = max(1, int(shard_count))
        self._queue_capacity = max(1, int(queue_capacity))
        self._coalesce_window_s = max(0, int(coalesce_window_ms)) / 1000.0
        self._stop_event = threading.Event()
        self._started = False
        self._shards = [
            _ReducerShard(
                shard_id=i,
                queue=queue.Queue(maxsize=self._queue_capacity),
            )
            for i in range(self._shard_count)
        ]

    def start(self) -> None:
        if self._started:
            return
        self._stop_event.clear()
        for shard in self._shards:
            thread = threading.Thread(
                target=self._shard_loop,
                args=(shard,),
                daemon=True,
                name=f"worker-control-reducer-{shard.shard_id}",
            )
            thread.start()
            shard.thread = thread
            set_worker_control_reducer_queue_depth(shard=shard.shard_id, depth=0)
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._stop_event.set()
        for shard in self._shards:
            with shard.lock:
                shard.pending_heartbeats.clear()
            with suppress(queue.Full):
                shard.queue.put_nowait(_STOP_SENTINEL)
        for shard in self._shards:
            if shard.thread is not None:
                shard.thread.join(timeout=1.0)
            shard.thread = None
            set_worker_control_reducer_queue_depth(shard=shard.shard_id, depth=0)
        self._started = False

    def submit(
        self,
        *,
        worker_key: str,
        kind: str,
        operation,
        timeout_s: float | None = None,
    ):
        if not self._started:
            raise RuntimeError("WorkerControlReducer must be started before submit()")
        shard = self._select_shard(worker_key)
        future: Future = Future()
        now_s = time.monotonic()
        intent = WorkerControlIntent(
            worker_key=worker_key,
            kind=kind,
            operation=operation,
            enqueued_at_s=now_s,
            futures=[future],
        )
        if kind == "heartbeat" and self._coalesce_window_s > 0:
            with shard.lock:
                pending = shard.pending_heartbeats.get(worker_key)
                if pending is not None and (
                    now_s - pending.enqueued_at_s <= self._coalesce_window_s
                ):
                    pending.operation = operation
                    pending.futures.append(future)
                    inc_worker_control_reducer_intent(
                        shard=shard.shard_id,
                        kind=kind,
                        result="coalesced",
                    )
                    self._logger.debug(
                        "Worker control reducer heartbeat coalesced worker_key=%s shard=%s queue_depth=%s",
                        worker_key,
                        shard.shard_id,
                        shard.queue.qsize(),
                    )
                    return future.result(timeout=timeout_s)
                shard.pending_heartbeats[worker_key] = intent
        try:
            shard.queue.put_nowait(intent)
            queue_depth = shard.queue.qsize()
            set_worker_control_reducer_queue_depth(
                shard=shard.shard_id, depth=queue_depth
            )
            inc_worker_control_reducer_intent(
                shard=shard.shard_id,
                kind=kind,
                result="submitted",
            )
        except queue.Full as exc:
            if kind == "heartbeat":
                with shard.lock:
                    shard.pending_heartbeats.pop(worker_key, None)
            inc_worker_control_reducer_intent(
                shard=shard.shard_id,
                kind=kind,
                result="overloaded",
            )
            self._logger.warning(
                "Worker control reducer queue full worker_key=%s kind=%s shard=%s queue_depth=%s",
                worker_key,
                kind,
                shard.shard_id,
                shard.queue.qsize(),
            )
            raise ReducerOverloadedError(
                f"worker control reducer shard={shard.shard_id} queue is full"
            ) from exc
        return future.result(timeout=timeout_s)

    def _select_shard(self, worker_key: str) -> _ReducerShard:
        shard_id = zlib.crc32(worker_key.encode("utf-8")) % self._shard_count
        return self._shards[shard_id]

    def _shard_loop(self, shard: _ReducerShard) -> None:
        while not self._stop_event.is_set():
            try:
                item = shard.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is _STOP_SENTINEL:
                break
            intent: WorkerControlIntent[Any] = cast(WorkerControlIntent[Any], item)
            assert intent is not None
            wait_s = max(0.0, time.monotonic() - intent.enqueued_at_s)
            observe_worker_control_reducer_queue_latency(wait_seconds=wait_s)
            try:
                with bind_tx_context(
                    source="worker_control_reducer",
                    intent_kind=intent.kind,
                    worker_key=intent.worker_key,
                    reducer_shard=shard.shard_id,
                ):
                    result = intent.operation()
                inc_worker_control_reducer_intent(
                    shard=shard.shard_id,
                    kind=intent.kind,
                    result="succeeded",
                )
                for future in intent.futures:
                    if not future.done():
                        future.set_result(result)
            except Exception as exc:  # noqa: BLE001
                inc_worker_control_reducer_intent(
                    shard=shard.shard_id,
                    kind=intent.kind,
                    result="failed",
                )
                for future in intent.futures:
                    if not future.done():
                        future.set_exception(exc)
                self._logger.exception(
                    "Worker control reducer intent failed kind=%s worker_key=%s shard=%s queue_depth=%s wait_seconds=%.6f",
                    intent.kind,
                    intent.worker_key,
                    shard.shard_id,
                    shard.queue.qsize(),
                    wait_s,
                )
            finally:
                if intent is not None and intent.kind == "heartbeat":
                    with shard.lock:
                        pending = shard.pending_heartbeats.get(intent.worker_key)
                        if pending is intent:
                            shard.pending_heartbeats.pop(intent.worker_key, None)
                shard.queue.task_done()
                set_worker_control_reducer_queue_depth(
                    shard=shard.shard_id, depth=shard.queue.qsize()
                )
