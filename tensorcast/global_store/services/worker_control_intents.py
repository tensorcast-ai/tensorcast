#  Copyright (c) 2025-2026, TensorCast Team.

"""Intent payloads for worker control-plane reducer execution."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class WorkerControlIntent(Generic[T]):
    """A reducer intent bound to one worker key and operation kind."""

    worker_key: str
    kind: str
    operation: Callable[[], T]
    enqueued_at_s: float
    futures: list[Future[T]] = field(default_factory=list)
