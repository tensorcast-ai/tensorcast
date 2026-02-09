#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import concurrent.futures
import json
import types
from pathlib import Path
from typing import Any, Iterator

import pytest
import torch

import tensorcast.api.store as store_mod
from tensorcast._c_ext import build_canonical_index_from_safetensors
from tensorcast.api import _region_cache as region_cache
from tensorcast.api.store import ArtifactError, Store
from tensorcast.api.store.cache import ArtifactCacheEntry
from tensorcast.api.store import deferred_loader as deferred_loader_mod
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.types import VramRegionHandle
from tests.python.utils.artifact_utils import create_dummy_safetensors


def _skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available - deferred loader requires CUDA tensors")


def _make_index_bytes(
    *, shape: list[int] | None = None, stride: list[int] | None = None
) -> bytes:
    shape_value = shape if shape is not None else [4]
    stride_value = stride if stride is not None else [1]
    numel = 1
    for dim in shape_value:
        numel *= int(dim)
    size_bytes = int(numel) * 4
    elem_bytes = 4
    beta_storage_offset = size_bytes // elem_bytes
    index = {
        "alpha": [0, size_bytes, shape_value, stride_value, "torch.float32", 0],
        "beta": [
            size_bytes,
            size_bytes,
            shape_value,
            stride_value,
            "torch.float32",
            beta_storage_offset,
        ],
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
    monkeypatch.setattr(
        store_mod,
        "get_cuda_memory_handle_with_offset",
        lambda *args, **kwargs: (b"fake-handle", 0),
    )
    monkeypatch.setattr(deferred_loader_mod, "device_uuid_for", lambda device_id: "gpu-0")
    try:
        yield store, client
    finally:
        runtime.executor.shutdown(wait=False)


@pytest.fixture
def store_and_client_empty_stride(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[Store, FakeDeferredClient]]:
    _skip_if_no_cuda()
    client = FakeDeferredClient(_make_index_bytes(stride=[]))
    runtime = FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    monkeypatch.setattr(
        store_mod, "get_cuda_memory_handle", lambda *args, **kwargs: b"fake-handle"
    )
    monkeypatch.setattr(
        store_mod,
        "get_cuda_memory_handle_with_offset",
        lambda *args, **kwargs: (b"fake-handle", 0),
    )
    monkeypatch.setattr(deferred_loader_mod, "device_uuid_for", lambda device_id: "gpu-0")
    try:
        yield store, client
    finally:
        runtime.executor.shutdown(wait=False)


@pytest.fixture
def store_and_client_safetensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[Store, FakeDeferredClient]]:
    _skip_if_no_cuda()
    storage_root = tmp_path / "models"
    storage_root.mkdir(parents=True, exist_ok=True)
    artifact_id = "st_simple"
    create_dummy_safetensors(storage_root, artifact_id)
    index_path = storage_root / artifact_id / "tensor_index.json"
    if index_path.exists():
        index_path.unlink()
    index_bytes = build_canonical_index_from_safetensors(str(storage_root / artifact_id))
    client = FakeDeferredClient(index_bytes)
    runtime = FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    monkeypatch.setattr(store_mod, "get_cuda_memory_handle", lambda *args, **kwargs: b"fake-handle")
    monkeypatch.setattr(
        store_mod,
        "get_cuda_memory_handle_with_offset",
        lambda *args, **kwargs: (b"fake-handle", 0),
    )
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
        slot = loader.commit()

    assert len(client.materialize_calls) == 1
    call = client.materialize_calls[0]
    assert tuple(call["tensor_names"]) == ("beta", "alpha")
    layout = call["target_layout"]
    assert layout.index_kind == store_daemon_pb2.TargetLayout.INDEX_KIND_VIEW
    commit_result = slot.commit_result
    assert commit_result.tensor_names == ("beta", "alpha")
    assert commit_result.view_id is None
    assert commit_result.view_subset_hash == b""


def test_deferred_loader_byte_space_full_selection_has_empty_selection(
    store_and_client: tuple[Store, FakeDeferredClient],
) -> None:
    store, client = store_and_client
    artifact = store.artifact(artifact_id="artifact-1")

    with artifact.deferred_loader(device="cuda:0", packing="byte_space") as loader:
        tensor_beta = loader.tensor("beta")
        tensor_alpha = loader.tensor("alpha")
        arena = loader._arena
        assert arena is not None
        assert tensor_alpha.data_ptr() == arena.data_ptr() + loader._offsets["alpha"]
        assert tensor_beta.data_ptr() == arena.data_ptr() + loader._offsets["beta"]
        slot = loader.commit()

    assert len(client.materialize_calls) == 1
    call = client.materialize_calls[0]
    assert tuple(call["tensor_names"]) == ()
    assert call["view_subset_hash"] == b""
    layout = call["target_layout"]
    assert (
        layout.index_kind
        == store_daemon_pb2.TargetLayout.INDEX_KIND_CANONICAL_UNSPECIFIED
    )
    commit_result = slot.commit_result
    assert commit_result.tensor_names == ()
    assert commit_result.view_subset_hash == b""


def test_deferred_loader_byte_space_safetensors_index(
    store_and_client_safetensors: tuple[Store, FakeDeferredClient],
) -> None:
    store, client = store_and_client_safetensors
    artifact = store.artifact(artifact_id="artifact-st")

    with artifact.deferred_loader(device="cuda:0", packing="byte_space") as loader:
        tensor = loader.tensor("t")
        assert tensor.is_cuda
        slot = loader.commit()

    assert len(client.materialize_calls) == 1
    call = client.materialize_calls[0]
    assert tuple(call["tensor_names"]) == ()
    assert call["view_subset_hash"] == b""
    layout = call["target_layout"]
    assert (
        layout.index_kind
        == store_daemon_pb2.TargetLayout.INDEX_KIND_CANONICAL_UNSPECIFIED
    )
    commit_result = slot.commit_result
    assert commit_result.tensor_names == ()
    assert commit_result.view_subset_hash == b""


def test_deferred_loader_empty_stride_is_contiguous(
    store_and_client_empty_stride: tuple[Store, FakeDeferredClient],
) -> None:
    store, _ = store_and_client_empty_stride
    artifact = store.artifact(artifact_id="artifact-1")
    with artifact.deferred_loader(device="cuda:0") as loader:
        tensor = loader.tensor("alpha")
        assert tensor.is_contiguous()
        assert tuple(tensor.stride()) == (1,)


def test_deferred_loader_append_retry_does_not_duplicate_order(
    store_and_client: tuple[Store, FakeDeferredClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, client = store_and_client
    artifact = store.artifact(artifact_id="artifact-1")

    with artifact.deferred_loader(device="cuda:0") as loader:
        original_ensure = loader._ensure_arena

        def _boom() -> None:
            raise ArtifactError(
                "arena unavailable",
                status_code="RESOURCE_EXHAUSTED",
                retryable=True,
            )

        monkeypatch.setattr(loader, "_ensure_arena", _boom)
        with pytest.raises(ArtifactError):
            loader.tensor("alpha")
        assert loader._order == []
        assert loader._cursor_bytes == 0
        assert "alpha" not in loader._offsets

        monkeypatch.setattr(loader, "_ensure_arena", original_ensure)
        loader.tensor("alpha")
        slot = loader.commit()

    assert slot.commit_result.tensor_names == ("alpha",)
    assert len(client.materialize_calls) == 1
