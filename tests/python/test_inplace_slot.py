#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import concurrent.futures
import importlib
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
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.proto.operation.v1 import operation_pb2
from tensorcast.types import VramRegionHandle

artifact_mod = importlib.import_module("tensorcast.api.store.artifact")
inplace_slot_mod = importlib.import_module("tensorcast.api.store.inplace_slot")
owned_binding_slot_mod = importlib.import_module(
    "tensorcast.api.store.owned_binding_slot"
)


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


class FakeSlotClient:
    def __init__(self, index_bytes: bytes) -> None:
        self._index_bytes = index_bytes
        self.materialize_calls: list[dict[str, Any]] = []
        self.create_binding_calls: list[dict[str, Any]] = []
        self.create_calls: list[dict[str, Any]] = []
        self.commit_calls: list[dict[str, Any]] = []
        self.begin_update_calls: list[dict[str, Any]] = []
        self.seal_calls: list[dict[str, Any]] = []
        self.refill_calls: list[dict[str, Any]] = []
        self.close_calls: list[str] = []
        self.publish_calls: list[dict[str, Any]] = []
        self.start_publish_calls: list[dict[str, Any]] = []
        self.retire_calls: list[dict[str, Any]] = []
        self.register_calls: list[dict[str, Any]] = []
        self.unregister_calls: list[str] = []
        self.last_get_operation_ref: operation_pb2.OperationRef | None = None
        self.last_wait_operation_ref: operation_pb2.OperationRef | None = None
        self.publish_failures = 0
        self.materialize_failures = 0
        self.omit_commit_current_value = False
        self._token_counter = 0
        self._region_counter = 0
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
    ) -> VramRegionHandle:
        self._region_counter += 1
        self.register_calls.append(
            {
                "device_id": device_id,
                "size_bytes": size_bytes,
                "ttl_ms": ttl_ms,
                "handle": cuda_ipc_handle,
                "region_name": region_name,
            }
        )
        return VramRegionHandle(
            region_id=f"region:slot:{self._region_counter}",
            ttl_ms=ttl_ms,
        )

    def unregister_vram_region(
        self, region_id: str, *, force: bool | None = None
    ) -> bool:
        self.unregister_calls.append(region_id)
        return True

    def materialize_into_target_v2(self, **kwargs: Any) -> Any:
        self.materialize_calls.append(kwargs)
        if self.materialize_failures > 0:
            self.materialize_failures -= 1
            raise ArtifactError(
                "materialize failed after bytes changed",
                status_code="DATA_LOSS",
                retryable=False,
            )
        self._token_counter += 1
        token = f"token-{self._token_counter}".encode("utf-8")
        selection = common_pb2.ArtifactSelection()
        selection.CopyFrom(kwargs["selection"])
        return types.SimpleNamespace(
            status=store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_ALLOCATED,
            target_publication_token=token,
            resolved_selection=selection,
        )

    def create_binding(self, **kwargs: Any) -> Any:
        self.create_binding_calls.append(kwargs)
        self._binding_counter += 1
        binding_id = f"client-slot-{self._binding_counter}"
        self._binding_layout_ids[binding_id] = str(kwargs["binding_layout_id"])
        current_value = None
        if kwargs.get("initial_selection") is not None:
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

    def start_publish_target_replica(self, **kwargs: Any) -> Any:
        self.start_publish_calls.append(kwargs)
        operation_id = str(kwargs.get("operation_id") or "publish-op-1")
        operation = operation_pb2.OperationRef(
            operation_id=operation_id,
            kind="publish_target_replica",
            target_artifact_id="artifact-1",
            authority_scope_kind="workflow_owner",
            authority_scope_id="publication-1",
            attachment_kind="target_publication",
            recovery_class="ephemeral_process_local",
        )
        return types.SimpleNamespace(operation=operation)

    def get_operation(
        self,
        operation_id: str,
        *,
        operation_ref: operation_pb2.OperationRef | None = None,
        timeout_s: float = 10.0,
    ) -> operation_pb2.GetOperationResponse:
        del timeout_s
        self.last_get_operation_ref = operation_pb2.OperationRef()
        if operation_ref is not None:
            self.last_get_operation_ref.CopyFrom(operation_ref)
        response = operation_pb2.GetOperationResponse()
        response.ref.CopyFrom(operation_ref or operation_pb2.OperationRef())
        if not response.ref.operation_id:
            response.ref.operation_id = operation_id
        response.status.state = operation_pb2.OPERATION_STATE_SUCCESS
        payload = store_daemon_pb2.PublishTargetReplicaResponse(
            lease_id="lease-1",
            replica_id="replica-1",
        )
        response.status.result.Pack(payload)
        return response

    def wait_operation(
        self,
        operation_id: str,
        *,
        operation_ref: operation_pb2.OperationRef | None = None,
        timeout_ms: int,
        timeout_s: float,
    ) -> operation_pb2.GetOperationResponse:
        del timeout_ms, timeout_s
        self.last_wait_operation_ref = operation_pb2.OperationRef()
        if operation_ref is not None:
            self.last_wait_operation_ref.CopyFrom(operation_ref)
        return self.get_operation(operation_id, operation_ref=operation_ref)

    def retire_published_replica(self, **kwargs: Any) -> Any:
        self.retire_calls.append(kwargs)
        return types.SimpleNamespace(drained=True, removed=True)

    def create_owned_binding(self, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        self._token_counter += 1
        self._binding_counter += 1
        binding_id = f"owned-slot-{self._binding_counter}"
        self._binding_layout_ids[binding_id] = str(kwargs["binding_layout_id"])
        selection = common_pb2.ArtifactSelection()
        selection.CopyFrom(kwargs["source_selection"])
        self._binding_selections[binding_id] = selection
        return types.SimpleNamespace(
            binding_id=binding_id,
            artifact_id=str(selection.artifact_id),
            target_index_bytes=bytes(kwargs["target_index_bytes"]),
            resolved_selection=selection,
            target_publication_token=f"token-{self._token_counter}".encode("utf-8"),
            current_value=self._make_binding_value(
                binding_id=binding_id,
                selection=selection,
            ),
        )

    def commit_binding_artifact(self, **kwargs: Any) -> Any:
        self.commit_calls.append(kwargs)
        binding_id = str(kwargs["binding_id"])
        selection = common_pb2.ArtifactSelection()
        selection.CopyFrom(kwargs["selection"])
        self._binding_selections[binding_id] = selection
        if self.omit_commit_current_value:
            return types.SimpleNamespace()
        return types.SimpleNamespace(
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
            target_publication_token=f"token-{self._token_counter}".encode("utf-8"),
            current_value=self._make_binding_value(
                binding_id=binding_id,
                selection=selection,
            ),
        )

    def close_owned_binding(self, *, binding_id: str, **_kwargs: Any) -> Any:
        self.close_calls.append(str(binding_id))
        return types.SimpleNamespace(closed=True)


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
        expires_at=time.monotonic(),
    )
    runtime.cache_artifact_index(entry)


def _patch_owned_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(artifact_mod, "device_uuid_for", lambda device_id: "gpu-0")
    monkeypatch.setattr(inplace_slot_mod, "device_uuid_for", lambda device_id: "gpu-0")
    monkeypatch.setattr(
        artifact_mod,
        "restore_owned_binding_tensors",
        lambda *, response, runtime, device_id: {
            entry.name: torch.empty_strided(
                size=tuple(int(v) for v in entry.shape),
                stride=tuple(int(v) for v in entry.stride),
                dtype=entry.dtype,
                device=torch.device("cuda", int(device_id)),
            )
            for entry in canonical_index_from_bytes(
                bytes(response.target_index_bytes)
            ).entries
        },
    )
    monkeypatch.setattr(
        owned_binding_slot_mod, "device_uuid_for", lambda device_id: "gpu-0"
    )


def test_binding_swap_preserves_data_ptr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _skip_if_no_cuda()
    index_bytes = _make_index_bytes()
    client = FakeSlotClient(index_bytes)
    runtime = FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    _cache_index(runtime, "artifact-1", index_bytes)
    _cache_index(runtime, "artifact-2", index_bytes)
    monkeypatch.setattr(
        store_mod, "get_cuda_memory_handle", lambda *args, **kwargs: b"fake-handle"
    )
    monkeypatch.setattr(
        store_mod,
        "get_cuda_memory_handle_with_offset",
        lambda *args, **kwargs: (b"fake-handle", 0),
    )
    _patch_owned_binding(monkeypatch)

    artifact1 = store.artifact(artifact_id="artifact-1")
    artifact2 = store.artifact(artifact_id="artifact-2")

    binding = artifact1.bind(device="cuda:0", packing="byte_space")
    ptrs = {name: tensor.data_ptr() for name, tensor in binding.tensors.items()}

    binding.swap(artifact2, publish=False)

    assert binding.tensors["alpha"].data_ptr() == ptrs["alpha"]
    assert binding.tensors["beta"].data_ptr() == ptrs["beta"]


def test_binding_swap_reuses_view_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _skip_if_no_cuda()
    index_bytes = _make_index_bytes()
    client = FakeSlotClient(index_bytes)
    runtime = FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    _cache_index(runtime, "mi2:artifact-1", index_bytes)
    _cache_index(runtime, "mi2:artifact-2", index_bytes)
    monkeypatch.setattr(
        store_mod, "get_cuda_memory_handle", lambda *args, **kwargs: b"fake-handle"
    )
    _patch_owned_binding(monkeypatch)

    artifact1 = store.artifact(artifact_id="mi2:artifact-1")
    artifact2 = store.artifact(artifact_id="mi2:artifact-2")

    slices = {"alpha": (slice(0, 2),)}
    artifact_view = artifact1.view(slices=slices)

    binding = artifact_view.bind(device="cuda:0", packing="byte_space")
    ptrs = {name: tensor.data_ptr() for name, tensor in binding.tensors.items()}

    binding.swap(artifact2, publish=False)

    assert binding.artifact_id == "mi2:artifact-2"
    assert binding.tensors["alpha"].data_ptr() == ptrs["alpha"]
    assert binding.tensors["beta"].data_ptr() == ptrs["beta"]


def test_publish_failure_keeps_binding_clean_and_retry(
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
    monkeypatch.setattr(
        store_mod, "get_cuda_memory_handle", lambda *args, **kwargs: b"fake-handle"
    )
    _patch_owned_binding(monkeypatch)

    artifact1 = store.artifact(artifact_id="artifact-1")
    artifact2 = store.artifact(artifact_id="artifact-2")

    binding = artifact1.bind(device="cuda:0", packing="byte_space")

    with pytest.raises(ArtifactError):
        binding.swap(artifact2, publish=True)

    assert binding._slot.dirty is False
    assert binding._slot.published_lease_id is None

    binding.publish_replica()
    assert binding._slot.published_lease_id == "lease-1"
    assert len(client.publish_calls) == 2


def test_binding_publish_operation_uses_operation_ref_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _skip_if_no_cuda()
    index_bytes = _make_index_bytes()
    client = FakeSlotClient(index_bytes)
    runtime = FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    _cache_index(runtime, "artifact-1", index_bytes)
    monkeypatch.setattr(
        store_mod, "get_cuda_memory_handle", lambda *args, **kwargs: b"fake-handle"
    )
    _patch_owned_binding(monkeypatch)

    artifact = store.artifact(artifact_id="artifact-1")
    binding = artifact.bind(device="cuda:0", packing="byte_space")

    op = binding._slot.publish_replica_operation()
    status = op.status()

    assert status.state == "success"
    assert len(client.start_publish_calls) == 1
    assert client.last_get_operation_ref is not None
    assert client.last_get_operation_ref.kind == "publish_target_replica"
    assert client.last_get_operation_ref.authority_scope_kind == "workflow_owner"
    assert client.last_get_operation_ref.attachment_kind == "target_publication"

    assert op.wait() is None
    assert binding._slot.published_lease_id == "lease-1"
    assert binding._slot.published_replica_id == "replica-1"


def test_binding_begin_update_rejects_published_replica(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _skip_if_no_cuda()
    index_bytes = _make_index_bytes()
    client = FakeSlotClient(index_bytes)
    runtime = FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    _cache_index(runtime, "artifact-1", index_bytes)
    monkeypatch.setattr(
        store_mod, "get_cuda_memory_handle", lambda *args, **kwargs: b"fake-handle"
    )
    _patch_owned_binding(monkeypatch)

    artifact = store.artifact(artifact_id="artifact-1")
    binding = artifact.bind(device="cuda:0", packing="byte_space")
    binding._slot._published_lease_id = "lease-1"

    with pytest.raises(ArtifactError) as excinfo:
        binding.begin_update()

    assert excinfo.value.status_code == "FAILED_PRECONDITION"
    assert "call retire() first" in str(excinfo.value)
    assert client.retire_calls == []
    assert client.begin_update_calls == []


def test_bind_into_failed_materialize_clears_current_value_and_marks_dirty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _skip_if_no_cuda()
    index_bytes = _make_index_bytes()
    client = FakeSlotClient(index_bytes)
    runtime = FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    _cache_index(runtime, "artifact-1", index_bytes)
    _cache_index(runtime, "artifact-2", index_bytes)
    monkeypatch.setattr(
        store_mod, "get_cuda_memory_handle", lambda *args, **kwargs: b"fake-handle"
    )
    monkeypatch.setattr(
        store_mod,
        "get_cuda_memory_handle_with_offset",
        lambda *args, **kwargs: (b"fake-handle", 0),
    )
    _patch_owned_binding(monkeypatch)

    artifact1 = store.artifact(artifact_id="artifact-1")
    artifact2 = store.artifact(artifact_id="artifact-2")
    target_tensors = {
        "alpha": torch.empty((4,), dtype=torch.float32, device="cuda:0"),
        "beta": torch.empty((4,), dtype=torch.float32, device="cuda:0"),
    }
    binding = artifact1.bind_into(target_tensors, packing="byte_space")
    client.materialize_failures = 2

    with pytest.raises(ArtifactError) as excinfo:
        binding.swap(artifact2)

    assert excinfo.value.status_code == "DATA_LOSS"
    assert binding.current_value is None
    assert binding.artifact_id is None
    assert binding.selection is None
    assert binding._slot.dirty is True


def test_bind_into_missing_commit_current_value_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _skip_if_no_cuda()
    index_bytes = _make_index_bytes()
    client = FakeSlotClient(index_bytes)
    runtime = FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    _cache_index(runtime, "artifact-1", index_bytes)
    _cache_index(runtime, "artifact-2", index_bytes)
    monkeypatch.setattr(
        store_mod, "get_cuda_memory_handle", lambda *args, **kwargs: b"fake-handle"
    )
    monkeypatch.setattr(
        store_mod,
        "get_cuda_memory_handle_with_offset",
        lambda *args, **kwargs: (b"fake-handle", 0),
    )
    _patch_owned_binding(monkeypatch)

    artifact1 = store.artifact(artifact_id="artifact-1")
    artifact2 = store.artifact(artifact_id="artifact-2")
    target_tensors = {
        "alpha": torch.empty((4,), dtype=torch.float32, device="cuda:0"),
        "beta": torch.empty((4,), dtype=torch.float32, device="cuda:0"),
    }
    binding = artifact1.bind_into(target_tensors, packing="byte_space")
    client.omit_commit_current_value = True

    with pytest.raises(ArtifactError) as excinfo:
        binding.swap(artifact2)

    assert excinfo.value.status_code == "DATA_LOSS"
    assert binding.current_value is None
    assert binding.artifact_id is None
    assert binding.selection is None
    assert binding._slot.dirty is True


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
