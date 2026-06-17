#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import concurrent.futures
import importlib
import json
import time
import types
from collections.abc import Iterator
from typing import Any

import pytest
import torch

from tensorcast.api import _region_cache as region_cache
from tensorcast.api.context import CallContext, CollectiveLoadGroup
from tensorcast.api.store import ArtifactError, Store
from tensorcast.api.store.cache import ArtifactCacheEntry
from tensorcast.api.store.common import canonical_index_from_bytes
from tensorcast.api.store.mapped_binding import (
    CopyPlanEntry,
    Range,
    copy_plan_from_json,
    copy_plan_to_json,
    infer_mapped_target_entries,
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
        self.create_binding_calls: list[dict[str, Any]] = []
        self.create_calls: list[dict[str, Any]] = []
        self.commit_calls: list[dict[str, Any]] = []
        self.begin_update_calls: list[dict[str, Any]] = []
        self.seal_calls: list[dict[str, Any]] = []
        self.refill_calls: list[dict[str, Any]] = []
        self.close_calls: list[str] = []
        self.publish_calls: list[dict[str, Any]] = []
        self.omit_create_binding_current_value = False
        self._token_counter = 0
        self._region_counter = 0
        self._lease_counter = 0
        self._binding_counter = 0
        self._binding_selections: dict[str, common_pb2.ArtifactSelection] = {}
        self._binding_layout_ids: dict[str, str] = {}
        self._binding_generations: dict[str, int] = {}
        self._binding_value_counter = 0

    def _make_binding_value(
        self,
        *,
        binding_id: str,
        selection: common_pb2.ArtifactSelection | None,
    ) -> store_daemon_pb2.BindingValue:
        self._binding_value_counter += 1
        generation = self._binding_generations.get(binding_id, 0) + 1
        self._binding_generations[binding_id] = generation
        value = store_daemon_pb2.BindingValue(
            binding_id=binding_id,
            binding_layout_id=self._binding_layout_ids.get(binding_id, ""),
            binding_value_id=f"value-{self._binding_value_counter}",
            seal_generation=generation,
            is_artifact_backed=selection is not None,
        )
        if selection is not None:
            value.source_artifact_id = str(selection.artifact_id)
            value.selection.CopyFrom(selection)
        return value

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

    def materialize_into_target(self, **kwargs: Any) -> Any:
        self.into_target_calls.append(kwargs)
        selection = common_pb2.ArtifactSelection()
        selection.CopyFrom(kwargs["selection"])
        return types.SimpleNamespace(
            status=1,  # MATERIALIZE_REPLICA_STATUS_ALLOCATED
            resolved_selection=selection,
        )

    def materialize_into_mapped_target(self, **kwargs: Any) -> Any:
        self.into_mapped_calls.append(kwargs)
        selection = common_pb2.ArtifactSelection()
        selection.CopyFrom(kwargs["selection"])
        return types.SimpleNamespace(
            status=1,  # MATERIALIZE_REPLICA_STATUS_ALLOCATED
            resolved_selection=selection,
        )

    def publish_target_replica(self, **kwargs: Any) -> Any:
        self.publish_calls.append(kwargs)
        self._lease_counter += 1
        return types.SimpleNamespace(
            lease_id=f"lease-{self._lease_counter}",
            replica_id=f"replica-{self._lease_counter}",
        )

    def create_binding(self, **kwargs: Any) -> Any:
        self.create_binding_calls.append(kwargs)
        self._binding_counter += 1
        self._token_counter += 1
        binding_id = f"mapped-client-binding-{self._binding_counter}"
        self._binding_layout_ids[binding_id] = str(kwargs["binding_layout_id"])
        current_value = None
        if (
            kwargs.get("initial_selection") is not None
            and not self.omit_create_binding_current_value
        ):
            selection = common_pb2.ArtifactSelection()
            selection.CopyFrom(kwargs["initial_selection"])
            self._binding_selections[binding_id] = selection
            current_value = self._make_binding_value(
                binding_id=binding_id,
                selection=selection,
            )
        return types.SimpleNamespace(
            binding_id=binding_id,
            target_index_bytes=bytes(kwargs["target_index_bytes"]),
            current_value=current_value,
            binding_current_value_publication_token=f"token-{self._token_counter}".encode(
                "utf-8"
            ),
        )

    def create_owned_binding(self, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        self._binding_counter += 1
        self._token_counter += 1
        binding_id = f"mapped-binding-{self._binding_counter}"
        self._binding_layout_ids[binding_id] = str(kwargs["binding_layout_id"])
        selection = common_pb2.ArtifactSelection(
            artifact_id=str(kwargs["source_selection"].artifact_id),
            view_id=str(kwargs["target_layout"].view_id),
            logical_layout_hash=bytes(kwargs["target_layout"].logical_layout_hash),
            selection_hash=b"mapped-selection",
        )
        self._binding_selections[binding_id] = selection
        return types.SimpleNamespace(
            binding_id=binding_id,
            artifact_id=str(selection.artifact_id),
            target_index_bytes=bytes(kwargs["target_index_bytes"]),
            resolved_selection=selection,
            binding_current_value_publication_token=f"token-{self._token_counter}".encode(
                "utf-8"
            ),
            created_staged_value=False,
            current_value=self._make_binding_value(
                binding_id=binding_id,
                selection=selection,
            ),
        )

    def commit_binding_artifact(self, **kwargs: Any) -> Any:
        self.commit_calls.append(kwargs)
        binding_id = str(kwargs["binding_id"])
        self._token_counter += 1
        selection = common_pb2.ArtifactSelection()
        selection.CopyFrom(kwargs["selection"])
        self._binding_selections[binding_id] = selection
        return types.SimpleNamespace(
            binding_current_value_publication_token=f"token-{self._token_counter}".encode(
                "utf-8"
            ),
            current_value=self._make_binding_value(
                binding_id=binding_id,
                selection=selection,
            )
        )

    def begin_binding_update(self, **kwargs: Any) -> Any:
        self.begin_update_calls.append(kwargs)
        return types.SimpleNamespace(update_epoch=f"bue:{kwargs['binding_id']}:1")

    def seal_binding(self, **kwargs: Any) -> Any:
        self.seal_calls.append(kwargs)
        return types.SimpleNamespace(
            current_value=self._make_binding_value(
                binding_id=str(kwargs["binding_id"]),
                selection=None,
            )
        )

    def refill_owned_binding(self, **kwargs: Any) -> Any:
        self.refill_calls.append(kwargs)
        self._token_counter += 1
        binding_id = str(kwargs["binding_id"])
        selection = common_pb2.ArtifactSelection()
        selection.CopyFrom(self._binding_selections[binding_id])
        selection.artifact_id = str(kwargs["artifact_id"])
        self._binding_selections[binding_id] = selection
        return types.SimpleNamespace(
            artifact_id=str(selection.artifact_id),
            resolved_selection=selection,
            binding_current_value_publication_token=f"token-{self._token_counter}".encode(
                "utf-8"
            ),
            current_value=self._make_binding_value(
                binding_id=binding_id,
                selection=selection,
            ),
        )

    def close_owned_binding(self, *, binding_id: str, **_kwargs: Any) -> Any:
        self.close_calls.append(str(binding_id))
        return types.SimpleNamespace(closed=True)


class _FailingRegionClient(_FakeMappedClient):
    def __init__(self, index_bytes: bytes, *, fail_after: int) -> None:
        super().__init__(index_bytes)
        self._fail_after = int(fail_after)

    def register_vram_region(
        self,
        *,
        device_id: int,
        size_bytes: int,
        ttl_ms: int,
        cuda_ipc_handle: bytes,
        region_name: str | None = None,
    ) -> Any:
        if len(self.register_calls) >= self._fail_after:
            raise MemoryError("region registry capacity reached")
        return super().register_vram_region(
            device_id=device_id,
            size_bytes=size_bytes,
            ttl_ms=ttl_ms,
            cuda_ipc_handle=cuda_ipc_handle,
            region_name=region_name,
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


def _patch_owner_restore(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    artifact_mod = importlib.import_module("tensorcast.api.store.artifact")
    store_mod = importlib.import_module("tensorcast.api.store")

    restore = lambda *, response, runtime, device_id: {  # noqa: E731
        entry.name: torch.empty_strided(
            size=tuple(int(v) for v in entry.shape),
            stride=tuple(int(v) for v in entry.stride),
            dtype=entry.dtype,
            device=torch.device("cuda", int(device_id)),
        )
        for entry in canonical_index_from_bytes(
            bytes(response.target_index_bytes)
        ).entries
    }
    monkeypatch.setattr(artifact_mod, "restore_owned_binding_tensors", restore)
    monkeypatch.setattr(store_mod, "restore_owned_binding_tensors", restore)


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
def test_mapped_bind_into_closes_binding_and_releases_regions_for_malformed_create_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA tensors unavailable; mapped binding requires torch CUDA")

    index_bytes = _make_index_bytes()
    client = _FakeMappedClient(index_bytes)
    client.omit_create_binding_current_value = True
    runtime = _FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    _cache_index(runtime, "artifact-1", index_bytes)

    import tensorcast.api._device as device_mod
    import tensorcast.api.store as store_mod

    artifact_mod = importlib.import_module("tensorcast.api.store.artifact")
    monkeypatch.setattr(
        store_mod,
        "get_cuda_memory_handle_with_offset",
        lambda *args, **kwargs: (b"fake-handle", 0),
    )
    monkeypatch.setattr(device_mod, "device_uuid_for", lambda device_id: "gpu-0")
    monkeypatch.setattr(artifact_mod, "device_uuid_for", lambda device_id: "gpu-0")

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

    with pytest.raises(ArtifactError) as excinfo:
        store.artifact(artifact_id="artifact-1").bind_into(dst_tensors, mapping=plan)

    assert excinfo.value.status_code == "DATA_LOSS"
    assert len(client.into_mapped_calls) == 1
    assert client.close_calls == ["mapped-client-binding-1"]
    assert client.unregister_calls == ["region:mapped:1", "region:mapped:2"]


@pytest.mark.requires_cuda_or_fake
def test_bind_into_surfaces_region_registry_capacity_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA tensors unavailable; mapped binding requires torch CUDA")

    index_bytes = _make_index_bytes()
    client = _FailingRegionClient(index_bytes, fail_after=1)
    runtime = _FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    _cache_index(runtime, "artifact-1", index_bytes)

    import tensorcast.api._device as device_mod
    import tensorcast.api.store as store_mod

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

    artifact = store.artifact(artifact_id="artifact-1")

    with pytest.raises(ArtifactError) as excinfo:
        artifact.bind_into(dst_tensors, mapping=plan)

    assert excinfo.value.status_code == "RESOURCE_EXHAUSTED"
    assert "region registry capacity" in str(excinfo.value)
    assert "requested_regions=2" in str(excinfo.value)
    assert "registered_before_failure=1" in str(excinfo.value)


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


@pytest.mark.requires_cuda_or_fake
def test_bind_mapping_uses_owner_path(
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

    import importlib

    import tensorcast.api._device as device_mod

    artifact_mod = importlib.import_module("tensorcast.api.store.artifact")

    monkeypatch.setattr(device_mod, "device_uuid_for", lambda device_id: "gpu-0")
    monkeypatch.setattr(artifact_mod, "device_uuid_for", lambda device_id: "gpu-0")
    _patch_owner_restore(monkeypatch)

    artifact1 = store.artifact(artifact_id="artifact-1")
    artifact2 = store.artifact(artifact_id="artifact-2")
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

    binding = artifact1.bind(device="cuda:0", mapping=plan)
    assert len(client.create_calls) == 1
    assert not client.into_target_calls
    assert not client.into_mapped_calls
    assert binding.selection.view_id
    assert {spec.name: spec.storage_offset for spec in binding.layout.dst_specs} == {
        "a": 0,
        "b": 0,
    }
    assert len(binding.layout.target_layout.storages) == 2
    assert {
        offset.name: offset.storage_offset
        for offset in binding.layout.target_layout.offsets
    } == {
        "a": 0,
        "b": 0,
    }
    ptrs = {name: tensor.data_ptr() for name, tensor in binding.tensors.items()}

    binding.swap(artifact2)

    assert len(client.refill_calls) == 1
    assert binding.tensors["a"].data_ptr() == ptrs["a"]
    assert binding.tensors["b"].data_ptr() == ptrs["b"]


@pytest.mark.requires_cuda_or_fake
def test_bind_into_mapping_propagates_collective_hint_in_operation_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA tensors unavailable; mapped binding requires torch CUDA")

    index_bytes = _make_index_bytes()
    client = _FakeMappedClient(index_bytes)
    runtime = _FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    _cache_index(runtime, "artifact-1", index_bytes)

    import importlib

    import tensorcast.api._device as device_mod

    artifact_mod = importlib.import_module("tensorcast.api.store.artifact")
    store_mod = importlib.import_module("tensorcast.api.store")

    monkeypatch.setattr(device_mod, "device_uuid_for", lambda device_id: "gpu-0")
    monkeypatch.setattr(artifact_mod, "device_uuid_for", lambda device_id: "gpu-0")
    monkeypatch.setattr(
        store_mod,
        "get_cuda_memory_handle_with_offset",
        lambda *args, **kwargs: (b"fake-handle", 0),
    )

    artifact = store.artifact(artifact_id="artifact-1")
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
    ctx = CallContext(
        collective=CollectiveLoadGroup(
            group_id="same-host-tp-load",
            world_size=8,
            rank=3,
        )
    )

    artifact.bind_into(dst_tensors, mapping=plan, ctx=ctx)

    assert len(client.into_mapped_calls) == 1
    operation_id = client.into_mapped_calls[0]["operation_id"]
    assert "clid=same-host-tp-load" not in operation_id
    assert "clws=8" not in operation_id
    assert "clrk=3" not in operation_id


@pytest.mark.requires_cuda_or_fake
def test_layout_seeded_create_binding_mapping_forwards_copy_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA tensors unavailable; mapped binding requires torch CUDA")

    index_bytes = _make_index_bytes()
    client = _FakeMappedClient(index_bytes)
    runtime = _FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    _cache_index(runtime, "artifact-1", index_bytes)

    import importlib

    import tensorcast.api._device as device_mod

    artifact_mod = importlib.import_module("tensorcast.api.store.artifact")

    monkeypatch.setattr(device_mod, "device_uuid_for", lambda device_id: "gpu-0")
    monkeypatch.setattr(artifact_mod, "device_uuid_for", lambda device_id: "gpu-0")
    _patch_owner_restore(monkeypatch)

    artifact = store.artifact(artifact_id="artifact-1")
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

    layout = artifact.bind(device="cuda:0", mapping=plan).layout

    created = store.create_binding(
        layout,
        ownership="daemon",
        device="cuda:0",
        mapping=plan,
    )

    assert created.current_value is None
    assert len(client.create_binding_calls) == 1
    create_call = client.create_binding_calls[0]
    assert create_call["copy_plan"] is not None
    assert len(create_call["copy_plan"].entries) == 2
    assert tuple(create_call["dst_specs"]) == layout.dst_specs


def test_bind_mapping_fails_on_ambiguous_inference() -> None:
    index = canonical_index_from_bytes(
        json.dumps(
            {
                "src_a": [0, 8, [8], [1], "torch.uint8", 0],
                "src_b": [8, 12, [3, 4], [4, 1], "torch.uint8", 0],
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    plan = [
        CopyPlanEntry(
            ckpt_name="src_a",
            ckpt_range=Range(dim=0, start=0, end=4),
            dst_name="mixed",
            dst_range=Range(dim=0, start=0, end=4),
        ),
        CopyPlanEntry(
            ckpt_name="src_b",
            ckpt_range=Range(dim=0, start=0, end=3),
            dst_name="mixed",
            dst_range=Range(dim=0, start=4, end=7),
        ),
    ]
    with pytest.raises(ArtifactError) as excinfo:
        _ = infer_mapped_target_entries(
            plan=plan,
            canonical_index=index,
            view_narrows={},
        )
    assert excinfo.value.status_code == "INVALID_ARGUMENT"


def test_infer_mapped_target_entries_allows_packed_qkv_style_merge() -> None:
    index = canonical_index_from_bytes(
        json.dumps(
            {
                "k_proj": [0, 16, [2, 4], [4, 1], "torch.uint16", 0],
                "q_proj": [16, 64, [8, 4], [4, 1], "torch.uint16", 0],
                "v_proj": [80, 16, [2, 4], [4, 1], "torch.uint16", 0],
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    plan = [
        CopyPlanEntry(
            ckpt_name="q_proj",
            ckpt_range=Range(dim=0, start=0, end=8),
            dst_name="qkv_proj",
            dst_range=Range(dim=0, start=0, end=8),
        ),
        CopyPlanEntry(
            ckpt_name="k_proj",
            ckpt_range=Range(dim=0, start=0, end=2),
            dst_name="qkv_proj",
            dst_range=Range(dim=0, start=8, end=10),
        ),
        CopyPlanEntry(
            ckpt_name="v_proj",
            ckpt_range=Range(dim=0, start=0, end=2),
            dst_name="qkv_proj",
            dst_range=Range(dim=0, start=10, end=12),
        ),
    ]

    inferred = infer_mapped_target_entries(
        plan=plan,
        canonical_index=index,
        view_narrows={},
    )

    assert len(inferred) == 1
    assert inferred[0].name == "qkv_proj"
    assert inferred[0].shape == (12, 4)
    assert inferred[0].stride == (4, 1)
