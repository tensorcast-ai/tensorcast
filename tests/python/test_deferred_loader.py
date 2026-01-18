#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import concurrent.futures
import json
import types
from typing import Any, Iterator

import pytest
import torch

import tensorcast.api.store as store_mod
from tensorcast.api import _region_cache as region_cache
from tensorcast.api.store import ArtifactError, Store
from tensorcast.api.store.cache import ArtifactCacheEntry
from tensorcast.api.store import deferred_loader as deferred_loader_mod
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.types import VramRegionHandle


def _skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available - deferred loader requires CUDA tensors")


def _make_index_bytes() -> bytes:
    index = {
        "alpha": [0, 16, [4], [1], "torch.float32", 0],
        "beta": [16, 16, [4], [1], "torch.float32", 0],
    }
    return json.dumps(index, separators=(",", ":"), sort_keys=True).encode("utf-8")


class FakeDeferredClient:
    def __init__(self, index_bytes: bytes) -> None:
        self._index_bytes = index_bytes
        self.materialize_calls: list[dict[str, Any]] = []
        self.register_calls: list[dict[str, Any]] = []
        self.unregister_calls: list[str] = []

    def get_artifact_index_by_id(self, artifact_id: str) -> bytes:
        return self._index_bytes

    def register_vram_region(
        self,
        *,
        device_id: int,
        size_bytes: int,
        ttl_ms: int,
        cuda_ipc_handle: bytes,
        region_name: str | None = None,
    ) -> VramRegionHandle:
        self.register_calls.append(
            {
                "device_id": device_id,
                "size_bytes": size_bytes,
                "ttl_ms": ttl_ms,
                "handle": cuda_ipc_handle,
                "region_name": region_name,
            }
        )
        return VramRegionHandle(region_id="region:deferred", ttl_ms=ttl_ms)

    def unregister_vram_region(self, region_id: str, *, force: bool | None = None) -> bool:
        self.unregister_calls.append(region_id)
        return True

    def materialize_into_target_v2(self, **kwargs: Any) -> Any:
        self.materialize_calls.append(kwargs)
        return types.SimpleNamespace(
            status=store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_ALLOCATED
        )


class FakeRuntime:
    _DEFAULT_LEASE_TTL_MS = 600_000

    def __init__(self, client: FakeDeferredClient) -> None:
        self._client = client
        self.daemon_endpoint = "fake://daemon"
        self.closed = False
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._cache: dict[str, ArtifactCacheEntry] = {}

    def ensure_client(self) -> FakeDeferredClient:
        return self._client

    def track_future(self, future: concurrent.futures.Future[object]) -> None:
        return None

    def get_artifact_index_cached(self, artifact_id: str) -> ArtifactCacheEntry | None:
        return self._cache.get(artifact_id)

    def cache_artifact_index(self, entry: ArtifactCacheEntry) -> None:
        self._cache[entry.artifact_id] = entry

    def invalidate_artifact(self, *args: object, **kwargs: object) -> None:
        return None


@pytest.fixture(autouse=True)
def _clear_region_cache() -> Iterator[None]:
    yield
    for device_id in list(region_cache._REGIONS_BY_DEVICE.keys()):
        for rec in list(region_cache._REGIONS_BY_DEVICE[device_id]):
            region_cache.unregister_region(rec.region_id)


@pytest.fixture
def store_and_client(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[Store, FakeDeferredClient]]:
    _skip_if_no_cuda()
    client = FakeDeferredClient(_make_index_bytes())
    runtime = FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    monkeypatch.setattr(store_mod, "get_cuda_memory_handle", lambda *args, **kwargs: b"fake-handle")
    monkeypatch.setattr(deferred_loader_mod, "device_uuid_for", lambda device_id: "gpu-0")
    try:
        yield store, client
    finally:
        runtime.executor.shutdown(wait=False)


def test_deferred_loader_invalid_slice_raises(
    store_and_client: tuple[Store, FakeDeferredClient],
) -> None:
    store, _ = store_and_client
    artifact = store.artifact(artifact_id="artifact-1")
    with artifact.deferred_loader(device="cuda:0") as loader:
        with pytest.raises(ArtifactError):
            loader.tensor("alpha", slice=slice(5, 6))


def test_deferred_loader_commit_preserves_order(
    store_and_client: tuple[Store, FakeDeferredClient],
) -> None:
    store, client = store_and_client
    artifact = store.artifact(artifact_id="artifact-1")

    with artifact.deferred_loader(device="cuda:0") as loader:
        tensor_beta = loader.tensor("beta")
        tensor_alpha = loader.tensor("alpha")
        arena = loader._arena
        assert arena is not None
        assert tensor_beta.data_ptr() == arena.data_ptr() + loader._offsets["beta"]
        assert tensor_alpha.data_ptr() == arena.data_ptr() + loader._offsets["alpha"]
        assert tensor_beta.is_cuda
        assert tensor_alpha.is_cuda
        assert tuple(tensor_beta.shape) == (4,)
        assert tuple(tensor_alpha.shape) == (4,)
        result = loader.commit()

    assert len(client.materialize_calls) == 1
    call = client.materialize_calls[0]
    assert tuple(call["tensor_names"]) == ("beta", "alpha")
    layout = call["target_layout"]
    assert layout.index_kind == store_daemon_pb2.TargetLayout.INDEX_KIND_VIEW
    assert result.tensor_names == ("beta", "alpha")
    assert result.view_id is None
    assert result.view_subset_hash is not None
