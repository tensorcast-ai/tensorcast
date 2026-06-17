#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import concurrent.futures
import logging
from collections.abc import Callable
from typing import Generic, TypeVar, cast

from tensorcast.api.store.runtime import StoreRuntimeContext

T = TypeVar("T")
logger = logging.getLogger(__name__)


class ArtifactFuture(Generic[T]):
    """Wrapper around concurrent futures with confirm/cancel hooks."""

    def __init__(
        self,
        future: concurrent.futures.Future[T],
        *,
        confirm: Callable[[], None] | None = None,
        cancel_callback: Callable[[], bool] | None = None,
    ) -> None:
        self._future = future
        self._confirm = confirm
        self._cancel = cancel_callback

    def done(self) -> bool:
        return self._future.done()

    def result(self, timeout: float | None = None) -> T:
        if not self._future.done() and self._confirm is not None:
            self._confirm()
        return self._future.result(timeout=timeout)

    def confirm(self, timeout: float | None = None) -> T:
        return self.result(timeout=timeout)

    def exception(self, timeout: float | None = None) -> BaseException | None:
        if not self._future.done() and self._confirm is not None:
            self._confirm()
        return self._future.exception(timeout=timeout)

    def add_done_callback(
        self, callback: Callable[["ArtifactFuture[T]"], None]
    ) -> None:
        def _wrap(_: concurrent.futures.Future[T]) -> None:
            callback(self)

        self._future.add_done_callback(_wrap)

    def cancel(self) -> bool:
        if self._future.done():
            return False
        cancelled = False
        if self._cancel is not None:
            try:
                cancelled = bool(self._cancel())
            except Exception:  # noqa: BLE001
                logger.exception("artifact future cancel callback failed")
        if not cancelled:
            cancelled = self._future.cancel()
        return cancelled


class TrackedExecutor:
    """Executor wrapper that syncs lifecycle with the runtime context."""

    def __init__(self, runtime: StoreRuntimeContext) -> None:
        self._runtime = runtime

    def submit(
        self,
        func: Callable[[], T],
        *,
        confirm: Callable[[], None] | None = None,
        cancel_callback: Callable[[], bool] | None = None,
    ) -> ArtifactFuture[T]:
        future = cast(concurrent.futures.Future[T], self._runtime.executor.submit(func))
        tracked_future = cast(concurrent.futures.Future[object], future)
        self._runtime.track_future(tracked_future)
        return ArtifactFuture(future, confirm=confirm, cancel_callback=cancel_callback)


__all__ = ["ArtifactFuture", "TrackedExecutor"]
