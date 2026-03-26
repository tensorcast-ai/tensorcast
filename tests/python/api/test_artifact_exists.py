#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import json
import weakref

import torch

from tensorcast.api.store.artifact import Artifact
from tensorcast.api.store.cache import ArtifactCache
from tensorcast.api.store.retry import build_retry_policies
from tensorcast.api.store.types import StoreOptions


def _canonical_index_bytes() -> bytes:
    tensor = torch.ones(1)
    meta = [0, int(tensor.element_size() * tensor.numel()), [1], [1], str(tensor.dtype), 0]
    return json.dumps({"foo": meta}, separators=(",", ":")).encode("utf-8")


class _ClientStub:
    def __init__(self, canonical_index_bytes: bytes) -> None:
        self.canonical_index_bytes = canonical_index_bytes
        self.calls = 0
        self.requested_artifact_ids: list[str] = []

    def get_artifact_index_by_id(self, artifact_id: str) -> bytes:
        self.calls += 1
        self.requested_artifact_ids.append(artifact_id)
        return self.canonical_index_bytes


class _RuntimeStub:
    def __init__(self, client: _ClientStub) -> None:
        self.daemon_endpoint = "daemon"
        self.session_id = "sess"
        self.opts = StoreOptions()
        self.retry_policies = build_retry_policies()
        self._artifact_cache = ArtifactCache(
            daemon_endpoint="daemon", ttl_seconds=5, max_entries=4
        )
        self._client = client
        self._key_cache: dict[str, str | None] = {}

    def ensure_client(self) -> _ClientStub:
        return self._client

    def cache_artifact_index(self, entry) -> None:
        self._artifact_cache.cache_artifact_index(entry)

    def get_artifact_index_cached(self, artifact_id: str):
        return self._artifact_cache.get_artifact_index_cached(artifact_id)

    def invalidate_artifact(self, artifact_id: str | None, *, key=None, reason=None) -> None:
        self._artifact_cache.invalidate_artifact(artifact_id or "", reason=reason)
        if key and key in self._key_cache:
            del self._key_cache[key]

    def resolve_key_mapping_cached(self, *, key: str) -> str | None:
        return self._key_cache.get(key)

    def cache_key_mapping(
        self, key: str, *, artifact_id: str | None, ttl_override=None
    ) -> None:
        self._key_cache[key] = artifact_id


class _StoreStub:
    def __init__(self, runtime: _RuntimeStub) -> None:
        self._runtime = runtime
        self._materialization = None
        self.closed = False


def test_exists_false_when_key_unmapped():
    runtime = _RuntimeStub(_ClientStub(_canonical_index_bytes()))
    store = _StoreStub(runtime)
    artifact = Artifact(store_ref=weakref.ref(store), key="missing")

    assert artifact.exists() is False


def test_exists_populates_cache_and_id():
    canonical_bytes = _canonical_index_bytes()
    runtime = _RuntimeStub(_ClientStub(canonical_bytes))
    store = _StoreStub(runtime)
    runtime.cache_key_mapping("mapped", artifact_id="aid")
    artifact = Artifact(store_ref=weakref.ref(store), key="mapped")

    assert artifact.exists() is True
    assert artifact.artifact_id == "aid"
    cached = runtime.get_artifact_index_cached("aid")
    assert cached is not None
    assert cached.canonical_index_bytes == canonical_bytes


def test_exists_handles_tuple_key_mapping_cache_result():
    canonical_bytes = _canonical_index_bytes()
    client = _ClientStub(canonical_bytes)
    runtime = _RuntimeStub(client)
    store = _StoreStub(runtime)
    runtime._key_cache["mapped"] = ("aid", "/tmp/cached.index")
    artifact = Artifact(store_ref=weakref.ref(store), key="mapped")

    assert artifact.exists() is True
    assert artifact.artifact_id == "aid"
    assert client.requested_artifact_ids == ["aid"]
