#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import time

import torch

from tensorcast.api.store.cache import ArtifactCache, ArtifactCacheEntry
from tensorcast.api.store.types import CanonicalIndex, CanonicalIndexEntry


def _make_entry(artifact_id: str, generation: int | None = None) -> ArtifactCacheEntry:
    entry = CanonicalIndexEntry(
        name="x",
        dtype=torch.float32,
        shape=(1,),
        stride=(1,),
        storage_offset=0,
        segment_offset=0,
        size_bytes=4,
    )
    index = CanonicalIndex(entries=(entry,), total_size_bytes=4, avbs_hash="")
    return ArtifactCacheEntry(
        artifact_id=artifact_id,
        canonical_index_bytes=b'{"x":[0,4,[1],[1],"float32",0]}',
        parsed_index=index,
        generation=generation,
        disk_path=None,
        expires_at=time.monotonic(),
    )


def test_cache_hit_and_ttl_expiry():
    cache = ArtifactCache(daemon_endpoint="daemon", ttl_seconds=0.01, max_entries=4)
    cache.cache_artifact_index(_make_entry("a1"))
    cached = cache.get_artifact_index_cached("a1")
    assert cached is not None
    assert cached.artifact_id == "a1"
    assert cached.hit_count == 1
    time.sleep(0.02)
    assert cache.get_artifact_index_cached("a1") is None


def test_cache_round_trip_preserves_generation():
    cache = ArtifactCache(daemon_endpoint="daemon", ttl_seconds=10, max_entries=2)
    cache.cache_artifact_index(_make_entry("g1", generation=9))
    cached = cache.get_artifact_index_cached("g1")
    assert cached is not None
    assert cached.generation == 9


def test_cache_lru_eviction():
    cache = ArtifactCache(daemon_endpoint="daemon", ttl_seconds=10, max_entries=2)
    cache.cache_artifact_index(_make_entry("first"))
    cache.cache_artifact_index(_make_entry("second"))
    cache.cache_artifact_index(_make_entry("third"))

    assert cache.get_artifact_index_cached("first") is None
    assert cache.get_artifact_index_cached("second") is not None
    assert cache.get_artifact_index_cached("third") is not None


def test_cache_invalidate_removes_entry():
    cache = ArtifactCache(daemon_endpoint="daemon", ttl_seconds=10, max_entries=2)
    cache.cache_artifact_index(_make_entry("dead"))
    assert cache.get_artifact_index_cached("dead") is not None
    cache.invalidate_artifact("dead", reason="explicit")
    assert cache.get_artifact_index_cached("dead") is None
