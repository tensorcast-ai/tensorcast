#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import concurrent.futures
import json
import time
import types
from collections.abc import Iterator
from typing import Any

import pytest
import torch

from tensorcast.api import _region_cache as region_cache
from tensorcast.api.store import ArtifactError, Store
from tensorcast.api.store.cache import ArtifactCacheEntry
from tensorcast.api.store.common import canonical_index_from_bytes
from tensorcast.api.store.mapped_binding import (
    CopyPlanEntry,
    Range,
    copy_plan_from_json,
    copy_plan_to_json,
    normalize_copy_plan,
    validate_copy_plan,
)
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2


def _make_index_bytes() -> bytes:
    index = {
        "src": [0, 8, [8], [1], "torch.uint8", 0],
    }
    return json.dumps(index, separators=(",", ":"), sort_keys=True).encode("utf-8")


@pytest.fixture(autouse=True)
def _clear_region_cache() -> Iterator[None]:
    yield
    for device_id in list(region_cache._REGIONS_BY_DEVICE.keys()):
        for rec in list(region_cache._REGIONS_BY_DEVICE[device_id]):
            region_cache.unregister_region(rec.region_id)


def test_copy_plan_json_roundtrip() -> None:
    plan = [
        CopyPlanEntry(
            ckpt_name="src",
            ckpt_range=Range(dim=0, start=0, end=4),
            dst_name="a",
            dst_range=Range(dim=0, start=0, end=4),
        ),
        CopyPlanEntry(
            ckpt_name="src",
            ckpt_range=Range(dim=0, start=4, end=8),
            dst_name="b",
            dst_range=Range(dim=0, start=0, end=4),
        ),
    ]
    encoded = copy_plan_to_json(plan)
    assert (
        encoded
        == '{"entries":[{"ckpt_name":"src","ckpt_range":{"dim":0,"end":4,"start":0},"dst_name":"a","dst_range":{"dim":0,"end":4,"start":0}},{"ckpt_name":"src","ckpt_range":{"dim":0,"end":8,"start":4},"dst_name":"b","dst_range":{"dim":0,"end":4,"start":0}}],"version":1}'
    )
    decoded = copy_plan_from_json(encoded)
    assert decoded == tuple(plan)


def test_normalize_copy_plan_accepts_dicts_and_tuples() -> None:
    plan = normalize_copy_plan(
        [
            {
                "ckpt_name": "src",
                "ckpt_range": {"dim": 0, "start": 0, "end": 4},
                "dst_name": "a",
                "dst_range": {"dim": 0, "start": 0, "end": 4},
            },
            (
                "src",
                {"dim": 0, "start": 4, "end": 8},
                "b",
                {"dim": 0, "start": 0, "end": 4},
            ),
        ]
    )
    assert plan == (
        CopyPlanEntry(
            ckpt_name="src",
            ckpt_range=Range(dim=0, start=0, end=4),
            dst_name="a",
            dst_range=Range(dim=0, start=0, end=4),
        ),
        CopyPlanEntry(
            ckpt_name="src",
            ckpt_range=Range(dim=0, start=4, end=8),
            dst_name="b",
            dst_range=Range(dim=0, start=0, end=4),
        ),
    )


def test_validate_copy_plan_requires_dst_names_match_targets() -> None:
    canonical_index = canonical_index_from_bytes(_make_index_bytes())
    target_tensors = {"a": torch.empty((4,), dtype=torch.uint8)}
    plan = [
        CopyPlanEntry(
            ckpt_name="src",
            ckpt_range=Range(dim=0, start=0, end=4),
            dst_name="b",
            dst_range=Range(dim=0, start=0, end=4),
        )
    ]
    with pytest.raises(ArtifactError) as excinfo:
        validate_copy_plan(
            plan=plan,
            canonical_index=canonical_index,
            target_tensors=target_tensors,
            view_narrows=None,
            require_full_coverage=True,
        )
    assert excinfo.value.status_code == "FAILED_PRECONDITION"


def test_validate_copy_plan_detects_dst_gaps() -> None:
    canonical_index = canonical_index_from_bytes(_make_index_bytes())
    target_tensors = {
        "a": torch.empty((4,), dtype=torch.uint8),
        "b": torch.empty((4,), dtype=torch.uint8),
    }
    plan = [
        CopyPlanEntry(
            ckpt_name="src",
            ckpt_range=Range(dim=0, start=0, end=3),
            dst_name="a",
            dst_range=Range(dim=0, start=0, end=3),
        ),
        CopyPlanEntry(
            ckpt_name="src",
            ckpt_range=Range(dim=0, start=4, end=8),
            dst_name="b",
            dst_range=Range(dim=0, start=0, end=4),
        ),
    ]
    with pytest.raises(ArtifactError) as excinfo:
        validate_copy_plan(
            plan=plan,
            canonical_index=canonical_index,
            target_tensors=target_tensors,
            view_narrows=None,
            require_full_coverage=True,
        )
    assert excinfo.value.status_code == "INVALID_ARGUMENT"


class _FakeMappedClient:
    def __init__(self, index_bytes: bytes) -> None:
        self._index_bytes = index_bytes
        self.register_calls: list[dict[str, Any]] = []
        self.unregister_calls: list[str] = []
        self.into_target_calls: list[dict[str, Any]] = []
        self.into_mapped_calls: list[dict[str, Any]] = []
        self.publish_calls: list[dict[str, Any]] = []
        self._token_counter = 0
        self._region_counter = 0
        self._lease_counter = 0

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
    ) -> Any:
        self._region_counter += 1
        self.register_calls.append(
            {
                "device_id": device_id,
                "size_bytes": size_bytes,
                "ttl_ms": ttl_ms,
                "cuda_ipc_handle": cuda_ipc_handle,
                "region_name": region_name,
            }
        )
        return types.SimpleNamespace(
            region_id=f"region:mapped:{self._region_counter}", ttl_ms=ttl_ms
        )

    def unregister_vram_region(
        self, region_id: str, *, force: bool | None = None
    ) -> bool:
        self.unregister_calls.append(region_id)
        return True

    def materialize_into_target_v2(self, **kwargs: Any) -> Any:
        self.into_target_calls.append(kwargs)
        self._token_counter += 1
        return types.SimpleNamespace(
            status=1,  # MATERIALIZE_REPLICA_STATUS_ALLOCATED
            target_write_token=f"token-{self._token_counter}".encode("utf-8"),
        )

    def materialize_into_mapped_target(self, **kwargs: Any) -> Any:
        self.into_mapped_calls.append(kwargs)
        self._token_counter += 1
        return types.SimpleNamespace(
            status=1,  # MATERIALIZE_REPLICA_STATUS_ALLOCATED
            target_write_token=f"token-{self._token_counter}".encode("utf-8"),
        )

    def publish_target_replica(self, **kwargs: Any) -> Any:
        self.publish_calls.append(kwargs)
        self._lease_counter += 1
        return types.SimpleNamespace(
            lease_id=f"lease-{self._lease_counter}",
            replica_id=f"replica-{self._lease_counter}",
        )


class _FakeRuntime:
    def __init__(self, client: _FakeMappedClient) -> None:
        self._client = client
        self.daemon_endpoint = "fake://daemon"
        self.closed = False
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._cache: dict[str, ArtifactCacheEntry] = {}

    def ensure_client(self) -> _FakeMappedClient:
        return self._client

    def track_future(self, future: concurrent.futures.Future[object]) -> None:
        return None

    def get_artifact_index_cached(self, artifact_id: str) -> ArtifactCacheEntry | None:
        return self._cache.get(artifact_id)

    def cache_artifact_index(self, entry: ArtifactCacheEntry) -> None:
        self._cache[entry.artifact_id] = entry

    def invalidate_artifact(self, *_args: object, **_kwargs: object) -> None:
        return None


def _cache_index(runtime: _FakeRuntime, artifact_id: str, index_bytes: bytes) -> None:
    parsed = canonical_index_from_bytes(index_bytes)
    entry = ArtifactCacheEntry(
        artifact_id=artifact_id,
        canonical_index_bytes=index_bytes,
        parsed_index=parsed,
        generation=1,
        expires_at=time.monotonic(),
    )
    runtime.cache_artifact_index(entry)


@pytest.mark.requires_cuda_or_fake
def test_mapped_binding_uses_materialize_into_mapped_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA tensors unavailable; mapped binding requires torch CUDA")

    index_bytes = _make_index_bytes()
    client = _FakeMappedClient(index_bytes)
    runtime = _FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    _cache_index(runtime, "artifact-1", index_bytes)
    _cache_index(runtime, "artifact-2", index_bytes)

    import tensorcast.api._device as device_mod
    import tensorcast.api.store as store_mod

    monkeypatch.setattr(
        store_mod, "get_cuda_memory_handle", lambda *args, **kwargs: b"fake-handle"
    )
    monkeypatch.setattr(
        store_mod,
        "get_cuda_memory_handle_with_offset",
        lambda *args, **kwargs: (b"fake-handle", 0),
    )
    monkeypatch.setattr(device_mod, "device_uuid_for", lambda device_id: "gpu-0")

    dst_tensors = {
        "a": torch.empty((4,), dtype=torch.uint8, device="cuda:0"),
        "b": torch.empty((4,), dtype=torch.uint8, device="cuda:0"),
    }
    plan = [
        CopyPlanEntry(
            ckpt_name="src",
            ckpt_range=Range(dim=0, start=0, end=4),
            dst_name="a",
            dst_range=Range(dim=0, start=0, end=4),
        ),
        CopyPlanEntry(
            ckpt_name="src",
            ckpt_range=Range(dim=0, start=4, end=8),
            dst_name="b",
            dst_range=Range(dim=0, start=0, end=4),
        ),
    ]

    artifact1 = store.artifact(artifact_id="artifact-1")
    artifact2 = store.artifact(artifact_id="artifact-2")

    binding = artifact1.bind_into(dst_tensors, mapping=plan)
    assert len(client.into_mapped_calls) == 1
    assert not client.into_target_calls
    first_selection = client.into_mapped_calls[0]["selection"]
    first_layout = client.into_mapped_calls[0]["target_layout"]
    assert first_selection.view_id
    assert first_layout.index_kind == store_daemon_pb2.TargetLayout.INDEX_KIND_VIEW

    pointers = {name: tensor.data_ptr() for name, tensor in binding.tensors.items()}
    binding.swap(artifact2)

    assert len(client.into_mapped_calls) == 2
    assert not client.into_target_calls
    second_selection = client.into_mapped_calls[1]["selection"]
    assert second_selection.view_id == first_selection.view_id
    assert binding.tensors["a"].data_ptr() == pointers["a"]
    assert binding.tensors["b"].data_ptr() == pointers["b"]


@pytest.mark.requires_cuda_or_fake
def test_mapped_binding_swap_publish_calls_publish_target_replica(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA tensors unavailable; mapped binding requires torch CUDA")

    index_bytes = _make_index_bytes()
    client = _FakeMappedClient(index_bytes)
    runtime = _FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    _cache_index(runtime, "artifact-1", index_bytes)
    _cache_index(runtime, "artifact-2", index_bytes)

    import tensorcast.api._device as device_mod
    import tensorcast.api.store as store_mod

    monkeypatch.setattr(
        store_mod, "get_cuda_memory_handle", lambda *args, **kwargs: b"fake-handle"
    )
    monkeypatch.setattr(
        store_mod,
        "get_cuda_memory_handle_with_offset",
        lambda *args, **kwargs: (b"fake-handle", 0),
    )
    monkeypatch.setattr(device_mod, "device_uuid_for", lambda device_id: "gpu-0")

    dst_tensors = {
        "a": torch.empty((4,), dtype=torch.uint8, device="cuda:0"),
        "b": torch.empty((4,), dtype=torch.uint8, device="cuda:0"),
    }
    plan = [
        CopyPlanEntry(
            ckpt_name="src",
            ckpt_range=Range(dim=0, start=0, end=4),
            dst_name="a",
            dst_range=Range(dim=0, start=0, end=4),
        ),
        CopyPlanEntry(
            ckpt_name="src",
            ckpt_range=Range(dim=0, start=4, end=8),
            dst_name="b",
            dst_range=Range(dim=0, start=0, end=4),
        ),
    ]

    artifact1 = store.artifact(artifact_id="artifact-1")
    artifact2 = store.artifact(artifact_id="artifact-2")

    binding = artifact1.bind_into(dst_tensors, mapping=plan)
    assert not client.publish_calls
    binding.swap(artifact2, publish=True)
    assert len(client.publish_calls) == 1
    publish_call = client.publish_calls[0]
    byte_space = publish_call["byte_space"]
    assert byte_space.kind == common_pb2.BYTE_SPACE_KIND_VIEW
    assert byte_space.id


@pytest.mark.requires_cuda_or_fake
def test_mapped_binding_bind_publish_calls_publish_target_replica(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA tensors unavailable; mapped binding requires torch CUDA")

    index_bytes = _make_index_bytes()
    client = _FakeMappedClient(index_bytes)
    runtime = _FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    _cache_index(runtime, "artifact-1", index_bytes)

    import tensorcast.api._device as device_mod
    import tensorcast.api.store as store_mod

    monkeypatch.setattr(
        store_mod, "get_cuda_memory_handle", lambda *args, **kwargs: b"fake-handle"
    )
    monkeypatch.setattr(
        store_mod,
        "get_cuda_memory_handle_with_offset",
        lambda *args, **kwargs: (b"fake-handle", 0),
    )
    monkeypatch.setattr(device_mod, "device_uuid_for", lambda device_id: "gpu-0")

    dst_tensors = {
        "a": torch.empty((4,), dtype=torch.uint8, device="cuda:0"),
        "b": torch.empty((4,), dtype=torch.uint8, device="cuda:0"),
    }
    plan = [
        CopyPlanEntry(
            ckpt_name="src",
            ckpt_range=Range(dim=0, start=0, end=4),
            dst_name="a",
            dst_range=Range(dim=0, start=0, end=4),
        ),
        CopyPlanEntry(
            ckpt_name="src",
            ckpt_range=Range(dim=0, start=4, end=8),
            dst_name="b",
            dst_range=Range(dim=0, start=0, end=4),
        ),
    ]

    artifact1 = store.artifact(artifact_id="artifact-1")
    binding = artifact1.bind_into(dst_tensors, mapping=plan)
    assert not client.publish_calls

    binding.publish_replica()
    assert len(client.publish_calls) == 1
    publish_call = client.publish_calls[0]
    byte_space = publish_call["byte_space"]
    assert byte_space.kind == common_pb2.BYTE_SPACE_KIND_VIEW
    assert byte_space.id
