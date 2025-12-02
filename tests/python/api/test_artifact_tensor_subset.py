#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import json
import weakref

import pytest
import torch

from tensorcast.api.store.artifact import Artifact
from tensorcast.api.store.cache import ArtifactCache
from tensorcast.api.store.common import canonical_index_from_bytes
from tensorcast.api.store.types import ArtifactError, StoreOptions
from tensorcast.api.store.retry import build_retry_policies


def _canonical_index_bytes() -> bytes:
    tensor = torch.ones(1)
    meta = [0, int(tensor.element_size() * tensor.numel()), [1], [1], str(tensor.dtype), 0]
    return json.dumps({"foo": meta}, separators=(",", ":")).encode("utf-8")


class _RuntimeStub:
    def __init__(self) -> None:
        self.daemon_endpoint = "daemon"
        self.session_id = "sess"
        self.opts = StoreOptions()
        self.retry_policies = build_retry_policies()
        self._artifact_cache = ArtifactCache(
            daemon_endpoint="daemon", ttl_seconds=5, max_entries=4
        )
        self._client = None

    def ensure_client(self):
        raise RuntimeError("should not be called")

    def cache_artifact_index(self, entry) -> None:
        self._artifact_cache.cache_artifact_index(entry)

    def get_artifact_index_cached(self, artifact_id: str):
        return self._artifact_cache.get_artifact_index_cached(artifact_id)

    def invalidate_artifact(self, artifact_id: str | None, *, key=None, reason=None) -> None:
        self._artifact_cache.invalidate_artifact(artifact_id or "", reason=reason)

    def resolve_key_mapping_cached(
        self, *, key: str
    ) -> tuple[str | None, str | None]:
        return None, None


class _StoreStub:
    def __init__(self, runtime: _RuntimeStub) -> None:
        self._runtime = runtime
        self._materialization = object()
        self.closed = False


def test_invalid_tensor_name_raises_before_materialization():
    canonical_bytes = _canonical_index_bytes()
    runtime = _RuntimeStub()
    store = _StoreStub(runtime)
    artifact = Artifact(
        store_ref=weakref.ref(store),
        artifact_id="aid",
        canonical_index_bytes=canonical_bytes,
        canonical_index=canonical_index_from_bytes(canonical_bytes),
    )

    with pytest.raises(ArtifactError) as excinfo:
        artifact.tensor_dict(device=0, names=["missing"])

    assert excinfo.value.status_code == "INVALID_ARGUMENT"
