#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import json
import weakref
from typing import cast

import pytest
import torch

from tensorcast.api._materialize import MaterializationPayload, TensorPayloadDescriptor
from tensorcast.api.store import Store
from tensorcast.api.store.artifact import Artifact
from tensorcast.api.store.cache import ArtifactCache
from tensorcast.api.store.types import ArtifactError, FallbackOptions, StoreOptions
from tensorcast.api.store.retry import build_retry_policies


def _build_payload(tensors: dict[str, torch.Tensor]) -> tuple[bytes, MaterializationPayload]:
    descriptors: list[TensorPayloadDescriptor] = []
    index: dict[str, list[object]] = {}
    offset = 0
    for name, tensor in tensors.items():
        size_bytes = int(tensor.element_size() * tensor.numel())
        shape = list(map(int, tensor.shape))
        stride = list(map(int, tensor.stride()))
        descriptors.append(
            TensorPayloadDescriptor(
                name=name,
                dtype=str(tensor.dtype),
                shape=tuple(shape),
                stride=tuple(stride),
                buffer_offset=offset,
                byte_length=size_bytes,
                storage_offset=0,
            )
        )
        index[name] = [offset, size_bytes, shape, stride, str(tensor.dtype), 0]
        offset += size_bytes
    canonical_bytes = json.dumps(index, separators=(",", ":")).encode("utf-8")
    payload = MaterializationPayload(
        artifact_id="aid",
        canonical_index_bytes=canonical_bytes,
        descriptors=tuple(descriptors),
        payload_iter=lambda: iter(()),
        state_dict=tensors,
        replica_uuid="replica-1",
        generation=7,
    )
    return canonical_bytes, payload


class _ClientStub:
    def __init__(self, canonical_index_bytes: bytes) -> None:
        self.canonical_index_bytes = canonical_index_bytes
        self.unloaded: list[tuple[str, str]] = []
        self.get_index_calls = 0

    def get_artifact_index_by_id(self, artifact_id: str) -> bytes:
        self.get_index_calls += 1
        return self.canonical_index_bytes

    def unload_replica(self, replica_uuid: str, *, disk_path: str = "") -> bool:
        self.unloaded.append((replica_uuid, disk_path))
        return True


class _RuntimeStub:
    def __init__(self, client: _ClientStub) -> None:
        self.daemon_endpoint = "daemon"
        self.session_id = "sess"
        self.opts = StoreOptions()
        self.retry_policies = build_retry_policies()
        self._artifact_cache = ArtifactCache(
            daemon_endpoint="daemon", ttl_seconds=10, max_entries=8
        )
        self._key_cache: dict[str, tuple[str | None, str | None]] = {}
        self._client = client

    def ensure_client(self) -> _ClientStub:
        return self._client

    def cache_artifact_index(self, entry) -> None:
        self._artifact_cache.cache_artifact_index(entry)

    def get_artifact_index_cached(self, artifact_id: str):
        return self._artifact_cache.get_artifact_index_cached(artifact_id)

    def invalidate_artifact(self, artifact_id: str | None, *, key=None, reason=None) -> None:
        self._artifact_cache.invalidate_artifact(artifact_id or "", reason=reason)

    def resolve_key_mapping_cached(
        self, *, key: str
    ) -> tuple[str | None, str | None]:
        return self._key_cache.get(key, (None, None))

    def cache_key_mapping(
        self, key: str, *, artifact_id: str | None, disk_path: str | None, ttl_override=None
    ) -> None:
        self._key_cache[key] = (artifact_id, disk_path)


class _PipelineStub:
    def __init__(self, payload: MaterializationPayload) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []
        self.released: list[str] = []

    def materialize_subset(self, **kwargs):
        self.calls.append(kwargs)
        return self.payload, 0

    def _payload_state_dict(self, payload: MaterializationPayload):
        if payload.state_dict is not None:
            return dict(payload.state_dict)
        state: dict[str, torch.Tensor] = {}
        for desc, tensor in payload.payload_iter():
            state[desc.name] = tensor
        return state

    def _release_materialized(self, payload: MaterializationPayload, client: _ClientStub) -> None:
        self.released.append(payload.replica_uuid)
        client.unload_replica(payload.replica_uuid, disk_path=getattr(payload, "disk_path", "") or "")

    def get_into(
        self,
        target: dict[str, torch.Tensor],
        *,
        artifact_id: str | None,
        key: str | None,
        device,
        fallback,
    ) -> None:
        state = self._payload_state_dict(self.payload)
        for name, tensor in state.items():
            if name in target:
                target[name].copy_(tensor)


class _StoreStub:
    def __init__(self, runtime: _RuntimeStub, pipeline: _PipelineStub) -> None:
        self._runtime = runtime
        self._materialization = pipeline
        self.closed = False


def _store_ref(store: _StoreStub) -> weakref.ReferenceType[Store]:
    typed_store = cast(Store, store)
    return weakref.ref(typed_store)


def test_tensor_subset_materialization_and_release():
    canonical_bytes, payload = _build_payload(
        {"foo": torch.ones(1), "bar": torch.zeros(1)}
    )
    runtime = _RuntimeStub(_ClientStub(canonical_bytes))
    pipeline = _PipelineStub(payload)
    store = _StoreStub(runtime, pipeline)
    artifact = Artifact(
        store_ref=_store_ref(store),
        artifact_id="aid",
        canonical_index_bytes=canonical_bytes,
        generation=1,
    )

    result = artifact.tensor_dict(device="cpu", names=["bar"])

    assert set(result.keys()) == {"bar"}
    assert pipeline.calls and pipeline.calls[0]["tensor_names"] == ("bar",)
    assert ("replica-1", "") in runtime._client.unloaded


def test_release_blocks_materialization():
    canonical_bytes, payload = _build_payload({"foo": torch.ones(1)})
    runtime = _RuntimeStub(_ClientStub(canonical_bytes))
    pipeline = _PipelineStub(payload)
    store = _StoreStub(runtime, pipeline)
    artifact = Artifact(
        store_ref=_store_ref(store),
        artifact_id="aid",
        canonical_index_bytes=canonical_bytes,
    )

    artifact.release()
    with pytest.raises(ArtifactError) as excinfo:
        artifact.tensor(name="foo", device="cpu")
    assert excinfo.value.status_code == "FAILED_PRECONDITION"


def test_to_dict_round_trip_preserves_metadata():
    canonical_bytes, payload = _build_payload({"foo": torch.ones(1)})
    runtime = _RuntimeStub(_ClientStub(canonical_bytes))
    pipeline = _PipelineStub(payload)
    store = _StoreStub(runtime, pipeline)
    artifact = Artifact(
        store_ref=_store_ref(store),
        artifact_id="aid",
        canonical_index_bytes=canonical_bytes,
        generation=5,
    )

    serialized = artifact.to_dict()
    restored = Artifact.from_dict(serialized, store=cast(Store, store))

    assert restored.artifact_id == "aid"
    assert restored.tensor_names == ("foo",)
    assert restored.describe().generation == 5
    assert runtime._client.get_index_calls == 0


def test_with_fallback_handles_multiple_identifiers():
    canonical_bytes, payload = _build_payload({"foo": torch.ones(1)})
    runtime = _RuntimeStub(_ClientStub(canonical_bytes))
    runtime.cache_key_mapping("mapped", artifact_id="aid", disk_path=None)
    pipeline = _PipelineStub(payload)
    store = _StoreStub(runtime, pipeline)
    artifact = Artifact(store_ref=_store_ref(store), key="mapped")

    assert artifact.artifact_id == "aid"
    clone = artifact.with_fallback(
        FallbackOptions(disk_path="/tmp/fallback", prefer_disk=True, allow_p2p=False)
    )

    assert clone.artifact_id == "aid"
    assert clone.key == "mapped"
    assert clone.tensor_names == ("foo",)


def test_from_dict_accepts_key_and_artifact_id():
    canonical_bytes, payload = _build_payload({"foo": torch.ones(1)})
    runtime = _RuntimeStub(_ClientStub(canonical_bytes))
    runtime.cache_key_mapping("mapped", artifact_id="aid", disk_path=None)
    pipeline = _PipelineStub(payload)
    store = _StoreStub(runtime, pipeline)
    artifact = Artifact(store_ref=_store_ref(store), key="mapped")
    assert artifact.artifact_id == "aid"

    serialized = artifact.to_dict()
    restored = Artifact.from_dict(serialized, store=cast(Store, store))

    assert restored.artifact_id == "aid"
    assert restored.key == "mapped"
    assert restored.tensor_names == ("foo",)
