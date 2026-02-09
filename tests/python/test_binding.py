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
from tensorcast.api.store import deferred_loader as deferred_loader_mod
from tensorcast.api.store.cache import ArtifactCacheEntry
from tensorcast.api.store.common import canonical_index_from_bytes
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.types import VramRegionHandle


def _skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available - binding tests require CUDA tensors")


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


class FakeBindingClient:
    def __init__(self, index_bytes: bytes) -> None:
        self._index_bytes = index_bytes
        self.materialize_calls: list[dict[str, Any]] = []
        self.publish_calls: list[dict[str, Any]] = []
        self.retire_calls: list[dict[str, Any]] = []
        self.register_calls: list[dict[str, Any]] = []
        self.unregister_calls: list[str] = []
        self.swap_key_calls: list[dict[str, Any]] = []
        self.keepalive_calls: list[tuple[str, int, int]] = []
        self._token_counter = 0
        self._key_state: dict[str, tuple[str, int]] = {}

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
        return VramRegionHandle(region_id="region:binding", ttl_ms=ttl_ms)

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
        return types.SimpleNamespace(lease_id="lease-1", replica_id="replica-1")

    def retire_published_replica(self, **kwargs: Any) -> Any:
        self.retire_calls.append(kwargs)
        return types.SimpleNamespace(drained=True, removed=True)

    def keep_alive_registered_artifact(self, registration_id: str, ttl_ms: int, epoch: int) -> bool:
        self.keepalive_calls.append((registration_id, ttl_ms, epoch))
        return True

    def swap_key_mapping(
        self,
        *,
        key: str,
        new_artifact_id: str,
        expected_artifact_id: str | None = None,
        expected_generation: int | None = None,
        operation_id: str | None = None,
        timeout_s: float = 10.0,
    ) -> Any:
        self.swap_key_calls.append(
            {
                "key": key,
                "new_artifact_id": new_artifact_id,
                "expected_artifact_id": expected_artifact_id,
                "expected_generation": expected_generation,
                "operation_id": operation_id,
                "timeout_s": timeout_s,
            }
        )
        current_id, generation = self._key_state.get(key, ("", 0))
        if expected_artifact_id and expected_artifact_id != current_id:
            return types.SimpleNamespace(ok=False, artifact_id=current_id, generation=generation)
        if expected_generation is not None and expected_generation != generation:
            return types.SimpleNamespace(ok=False, artifact_id=current_id, generation=generation)
        if current_id == new_artifact_id:
            return types.SimpleNamespace(ok=True, artifact_id=current_id, generation=generation)
        generation += 1
        self._key_state[key] = (new_artifact_id, generation)
        return types.SimpleNamespace(ok=True, artifact_id=new_artifact_id, generation=generation)


class FakeRuntime:
    _DEFAULT_LEASE_TTL_MS = 600_000

    def __init__(self, client: FakeBindingClient) -> None:
        self._client = client
        self.daemon_endpoint = "fake://daemon"
        self.closed = False
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._cache: dict[str, ArtifactCacheEntry] = {}

    def ensure_client(self) -> FakeBindingClient:
        return self._client

    def track_future(self, future: concurrent.futures.Future[object]) -> None:
        return None

    def get_artifact_index_cached(self, artifact_id: str) -> ArtifactCacheEntry | None:
        return self._cache.get(artifact_id)

    def cache_artifact_index(self, entry: ArtifactCacheEntry) -> None:
        self._cache[entry.artifact_id] = entry

    def invalidate_artifact(self, *_args: object, **_kwargs: object) -> None:
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
        expires_at=time.monotonic(),
    )
    runtime.cache_artifact_index(entry)


def _setup_store(monkeypatch: pytest.MonkeyPatch) -> tuple[Store, FakeRuntime, FakeBindingClient]:
    _skip_if_no_cuda()
    index_bytes = _make_index_bytes()
    client = FakeBindingClient(index_bytes)
    runtime = FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    _cache_index(runtime, "artifact-1", index_bytes)
    _cache_index(runtime, "artifact-2", index_bytes)
    monkeypatch.setattr(store_mod, "get_cuda_memory_handle", lambda *args, **kwargs: b"fake-handle")
    monkeypatch.setattr(
        store_mod,
        "get_cuda_memory_handle_with_offset",
        lambda *args, **kwargs: (b"fake-handle", 0),
    )
    monkeypatch.setattr(deferred_loader_mod, "device_uuid_for", lambda device_id: "gpu-0")
    return store, runtime, client


def test_binding_swap_preserves_data_ptr(monkeypatch: pytest.MonkeyPatch) -> None:
    store, _runtime, _client = _setup_store(monkeypatch)
    artifact1 = store.artifact(artifact_id="artifact-1")
    artifact2 = store.artifact(artifact_id="artifact-2")

    binding = artifact1.bind(device="cuda:0", packing="byte_space")
    ptrs = {name: tensor.data_ptr() for name, tensor in binding.tensors.items()}

    binding.swap(artifact2)

    assert binding.tensors["alpha"].data_ptr() == ptrs["alpha"]
    assert binding.tensors["beta"].data_ptr() == ptrs["beta"]


def test_binding_view_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
    store, _runtime, _client = _setup_store(monkeypatch)
    artifact1 = store.artifact(artifact_id="artifact-1")
    artifact2 = store.artifact(artifact_id="artifact-2")

    slices = {"alpha": (slice(0, 2),)}
    view = artifact1.view(slices=slices)
    binding = view.bind(device="cuda:0", packing="byte_space")
    selection_before = binding.selection

    binding.swap(artifact2)

    selection_after = binding.selection
    assert selection_before.view_id == selection_after.view_id
    assert selection_before.selection_hash == selection_after.selection_hash


def test_binding_publishability_gating(monkeypatch: pytest.MonkeyPatch) -> None:
    store, _runtime, _client = _setup_store(monkeypatch)
    artifact = store.artifact(artifact_id="artifact-1")
    with pytest.raises(ArtifactError) as excinfo:
        _ = artifact.bind(device="cuda:0", packing="append", publish=True)
    assert excinfo.value.status_code == "FAILED_PRECONDITION"


def test_binding_activation_cas(monkeypatch: pytest.MonkeyPatch) -> None:
    store, _runtime, client = _setup_store(monkeypatch)
    artifact1 = store.artifact(artifact_id="artifact-1")
    artifact2 = store.artifact(artifact_id="artifact-2")

    binding = artifact1.bind(device="cuda:0", packing="byte_space")
    binding.swap(
        artifact2,
        activate_key="model:latest",
        expected_active_artifact_id="",
    )
    assert client.swap_key_calls

    with pytest.raises(ArtifactError) as excinfo:
        binding.swap(
            artifact1,
            activate_key="model:latest",
            expected_active_artifact_id="artifact-1",
        )
    assert excinfo.value.status_code == "FAILED_PRECONDITION"
