#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import concurrent.futures
import json
import time
import types
from typing import Any, Iterator

import pytest
import torch

import tensorcast.api.store as store_mod
from tensorcast.api import _region_cache as region_cache
from tensorcast.api.store import ArtifactError, Store
from tensorcast.api.store.cache import ArtifactCacheEntry
from tensorcast.api.store.common import canonical_index_from_bytes
from tensorcast.api.store import deferred_loader as deferred_loader_mod
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.types import VramRegionHandle


def _skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available - inplace slot tests require CUDA tensors")


def _make_index_bytes() -> bytes:
    size_bytes = 16
    elem_bytes = 4
    beta_storage_offset = size_bytes // elem_bytes
    index = {
        "alpha": [0, size_bytes, [4], [1], "torch.float32", 0],
        "beta": [
            size_bytes,
            size_bytes,
            [4],
            [1],
            "torch.float32",
            beta_storage_offset,
        ],
    }
    return json.dumps(index, separators=(",", ":"), sort_keys=True).encode("utf-8")


class FakeSlotClient:
    def __init__(self, index_bytes: bytes) -> None:
        self._index_bytes = index_bytes
        self.materialize_calls: list[dict[str, Any]] = []
        self.publish_calls: list[dict[str, Any]] = []
        self.retire_calls: list[dict[str, Any]] = []
        self.register_calls: list[dict[str, Any]] = []
        self.unregister_calls: list[str] = []
        self.publish_failures = 0
        self._token_counter = 0

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
        return VramRegionHandle(region_id="region:slot", ttl_ms=ttl_ms)

    def unregister_vram_region(self, region_id: str, *, force: bool | None = None) -> bool:
        self.unregister_calls.append(region_id)
        return True

    def materialize_into_target_v2(self, **kwargs: Any) -> Any:
        self.materialize_calls.append(kwargs)
        self._token_counter += 1
        token = f"token-{self._token_counter}".encode("utf-8")
        return types.SimpleNamespace(
            status=store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_ALLOCATED,
            target_write_token=token,
        )

    def publish_target_replica(self, **kwargs: Any) -> Any:
        self.publish_calls.append(kwargs)
        if self.publish_failures > 0:
            self.publish_failures -= 1
            raise ArtifactError(
                "publish failed",
                status_code="UNAVAILABLE",
                retryable=True,
            )
        return types.SimpleNamespace(lease_id="lease-1", replica_id="replica-1")

    def retire_published_replica(self, **kwargs: Any) -> Any:
        self.retire_calls.append(kwargs)
        return types.SimpleNamespace(drained=True, removed=True)


class FakeRuntime:
    _DEFAULT_LEASE_TTL_MS = 600_000

    def __init__(self, client: FakeSlotClient) -> None:
        self._client = client
        self.daemon_endpoint = "fake://daemon"
        self.closed = False
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._cache: dict[str, ArtifactCacheEntry] = {}

    def ensure_client(self) -> FakeSlotClient:
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


def _cache_index(runtime: FakeRuntime, artifact_id: str, index_bytes: bytes) -> None:
    parsed = canonical_index_from_bytes(index_bytes)
    entry = ArtifactCacheEntry(
        artifact_id=artifact_id,
        canonical_index_bytes=index_bytes,
        parsed_index=parsed,
        generation=1,
        disk_path=None,
        expires_at=time.monotonic(),
    )
    runtime.cache_artifact_index(entry)


def test_inplace_slot_swap_preserves_data_ptr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _skip_if_no_cuda()
    index_bytes = _make_index_bytes()
    client = FakeSlotClient(index_bytes)
    runtime = FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    _cache_index(runtime, "artifact-1", index_bytes)
    _cache_index(runtime, "artifact-2", index_bytes)
    monkeypatch.setattr(store_mod, "get_cuda_memory_handle", lambda *args, **kwargs: b"fake-handle")
    monkeypatch.setattr(deferred_loader_mod, "device_uuid_for", lambda device_id: "gpu-0")

    artifact1 = store.artifact(artifact_id="artifact-1")
    artifact2 = store.artifact(artifact_id="artifact-2")

    with artifact1.deferred_loader(device="cuda:0", packing="byte_space") as loader:
        tensor_alpha = loader.tensor("alpha")
        tensor_beta = loader.tensor("beta")
        ptrs = {"alpha": tensor_alpha.data_ptr(), "beta": tensor_beta.data_ptr()}
        slot = loader.commit()

    slot.swap(artifact2, publish=False)

    assert slot.tensors["alpha"].data_ptr() == ptrs["alpha"]
    assert slot.tensors["beta"].data_ptr() == ptrs["beta"]


def test_publish_failure_keeps_slot_clean_and_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _skip_if_no_cuda()
    index_bytes = _make_index_bytes()
    client = FakeSlotClient(index_bytes)
    client.publish_failures = 1
    runtime = FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    _cache_index(runtime, "artifact-1", index_bytes)
    _cache_index(runtime, "artifact-2", index_bytes)
    monkeypatch.setattr(store_mod, "get_cuda_memory_handle", lambda *args, **kwargs: b"fake-handle")
    monkeypatch.setattr(deferred_loader_mod, "device_uuid_for", lambda device_id: "gpu-0")

    artifact1 = store.artifact(artifact_id="artifact-1")
    artifact2 = store.artifact(artifact_id="artifact-2")

    with artifact1.deferred_loader(device="cuda:0", packing="byte_space") as loader:
        _ = loader.tensor("alpha")
        _ = loader.tensor("beta")
        slot = loader.commit()

    with pytest.raises(ArtifactError):
        slot.swap(artifact2, publish=True)

    assert slot.dirty is False
    assert slot.published_lease_id is None

    slot.publish_replica()
    assert slot.published_lease_id == "lease-1"
    assert len(client.publish_calls) == 2


def test_artifact_ref_parsing() -> None:
    index_bytes = _make_index_bytes()
    client = FakeSlotClient(index_bytes)
    runtime = FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)

    artifact = store.artifact(ref="llama")
    assert artifact.key == "llama"

    artifact_id = store.artifact(ref="mi2:abc123").artifact_id
    assert artifact_id == "mi2:abc123"

    with pytest.raises(ValueError):
        store.artifact(ref="disk:")
    with pytest.raises(ValueError):
        store.artifact(ref="llama", key="other")
    with pytest.raises(ValueError):
        store.artifact(ref="")
