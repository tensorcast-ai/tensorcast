#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Mapping

from tensorcast.api import _metrics as store_metrics
from tensorcast.api.store.types import CanonicalIndex

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ArtifactCacheEntry:
    artifact_id: str
    canonical_index_bytes: bytes
    parsed_index: CanonicalIndex
    generation: int | None
    expires_at: float
    hit_count: int = 0


class ArtifactCache:
    """Process-wide canonical index cache with TTL and LRU eviction."""

    def __init__(
        self,
        *,
        daemon_endpoint: str,
        ttl_seconds: float,
        max_entries: int,
    ) -> None:
        self._daemon_endpoint = daemon_endpoint
        self._ttl_seconds = max(0.0, float(ttl_seconds))
        self._max_entries = int(max_entries)
        self._lock = threading.RLock()
        self._entries: OrderedDict[str, ArtifactCacheEntry] = OrderedDict()

    @property
    def ttl_seconds(self) -> float:
        return self._ttl_seconds

    @property
    def max_entries(self) -> int:
        return self._max_entries

    @property
    def enabled(self) -> bool:
        return self._ttl_seconds > 0 and self._max_entries > 0

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    def get_artifact_index_cached(self, artifact_id: str) -> ArtifactCacheEntry | None:
        if not self.enabled:
            return None
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(artifact_id)
            if entry is None:
                store_metrics.increment_artifact_cache_miss(self._daemon_endpoint)
                return None
            if entry.expires_at <= now:
                del self._entries[artifact_id]
                store_metrics.increment_artifact_cache_eviction(
                    self._daemon_endpoint, reason="ttl"
                )
                store_metrics.increment_artifact_cache_miss(self._daemon_endpoint)
                return None
            entry.hit_count += 1
            self._entries.move_to_end(artifact_id)
            store_metrics.increment_artifact_cache_hit(self._daemon_endpoint)
            return replace(entry)

    def cache_artifact_index(self, entry: ArtifactCacheEntry) -> None:
        if not self.enabled:
            return
        if not entry.artifact_id:
            return
        expires_at = time.monotonic() + self._ttl_seconds
        cached = replace(entry, expires_at=expires_at, hit_count=0)
        with self._lock:
            if entry.artifact_id in self._entries:
                del self._entries[entry.artifact_id]
            self._entries[entry.artifact_id] = cached
            self._entries.move_to_end(entry.artifact_id)
            self._evict_lru_locked()

    def invalidate_artifact(
        self, artifact_id: str, *, reason: str | None = None
    ) -> None:
        if not artifact_id:
            return
        with self._lock:
            removed = self._entries.pop(artifact_id, None)
        if removed is not None:
            store_metrics.increment_artifact_cache_invalidation(
                self._daemon_endpoint, reason=reason or "explicit"
            )

    def _evict_lru_locked(self) -> None:
        if self._max_entries <= 0:
            return
        while len(self._entries) > self._max_entries:
            _, removed = self._entries.popitem(last=False)
            store_metrics.increment_artifact_cache_eviction(
                self._daemon_endpoint, reason="lru"
            )
            logger.debug(
                "store.artifact_cache.evict",
                extra={
                    "tc.store.daemon": self._daemon_endpoint,
                    "tc.artifact.id": removed.artifact_id,
                    "tc.cache.reason": "lru",
                },
            )

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def to_debug_dict(self) -> Mapping[str, float]:
        with self._lock:
            return {key: entry.expires_at for key, entry in self._entries.items()}
