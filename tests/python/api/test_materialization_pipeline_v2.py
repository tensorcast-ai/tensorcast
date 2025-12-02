#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import concurrent.futures
import json
import threading
import time
import weakref
from contextlib import contextmanager
from typing import Callable

import pytest
import torch

from tensorcast.api._config import GetArtifactOptions
from tensorcast.api._materialize import (
    MaterializationPayload,
    TensorPayloadDescriptor,
    materialize_artifact_v2,
)
from tensorcast.api.store.batch_context import PrefetchTicket
from tensorcast.api.store.cache import ArtifactCache, ArtifactCacheEntry
from tensorcast.api.store.common import canonical_index_from_bytes
from tensorcast.api.store.materialization import MaterializationPipeline
from tensorcast.api.store.retry import build_retry_policies
from tensorcast.api.store.types import ArtifactError, FallbackOptions, StoreOptions
from tensorcast.api.store.views import ViewOrchestrator
from tensorcast.daemon_ctl import PrefetchRpcError
from tensorcast.proto.daemon.v1 import store_daemon_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2 as store_daemon_v2_pb2


@pytest.fixture(autouse=True)
def _force_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)


def test_materialization_proto_alignment():
    resp = store_daemon_v2_pb2.MaterializeReplicaResponse()
    assert hasattr(resp, "canonical_index_bytes")
    assert hasattr(resp, "view_index_bytes")
    assert hasattr(resp, "generation")
    assert not hasattr(resp, "canonical_index_json")


def test_materialize_by_key_forwards_preferences(monkeypatch):
    captured: dict[str, object] = {}

    class _StubClient:
        def materialize_by_key_v2(self, **kwargs):
            captured.update(kwargs)
            resp = store_daemon_v2_pb2.MaterializeByKeyResponse()
            resp.status = (
                store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_ALLOCATED
            )
            resp.artifact_id = "aid-key"
            resp.used_disk_path = "/tmp/disk"
            resp.canonical_index_bytes = json.dumps(
                {"w": [0, 4, [1], [1], "float32", 0]}
            ).encode("utf-8")
            desc = resp.payloads.add()
            desc.name = "w"
            desc.dtype = "float32"
            desc.shape.append(1)
            desc.stride.append(1)
            desc.byte_length = 4
            desc.buffer_offset = 0
            desc.storage_offset = 0
            resp.mem_handle.cuda_ipc_handle = b"\x00" * 8
            resp.ticket.replica_uuid = kwargs["replica_uuid"]
            resp.ticket.status = resp.status
            return resp

    monkeypatch.setattr("tensorcast.api._materialize.device_uuid_for", lambda _dev: "gpu-0")
    monkeypatch.setattr("tensorcast.api._materialize.resolve_device", lambda _dev: 0)
    monkeypatch.setattr("tensorcast.api._materialize.get_cuda_memory_ptr", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(
        "tensorcast.api._materialize.restore_tensors",
        lambda *_args, **_kwargs: {"w": torch.ones(1)},
    )

    payload = materialize_artifact_v2(
        client=_StubClient(),
        daemon_address="daemon",
        device_id=0,
        artifact_id=None,
        key="by-key-demo",
        preference=store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_PREFER_DISK,
        tensor_names=["w"],
        verify_checksums=False,
        view_subset_hash=b"abc",
        options=GetArtifactOptions(wait_for_completion=True, enable_verification=False),
    )

    assert captured["preference"] == store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_PREFER_DISK
    assert captured["verify_checksums"] is False
    assert captured["tensor_names"] == ["w"]
    assert captured["view_subset_hash"] == b"abc"
    assert payload.artifact_id == "aid-key"
    assert payload.ticket_replica_uuid == captured["replica_uuid"]


class _DummySpan:
    def __init__(self) -> None:
        self.attributes: dict[str, object] = {}
        self.events: list[tuple[str, dict[str, object]]] = []
        self.exceptions: list[Exception] = []

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value

    def set_status(self, status: object) -> None:  # pragma: no cover - noop
        self.attributes["status"] = status

    def add_event(self, name: str, attrs: dict[str, object]) -> None:
        self.events.append((name, attrs))

    def record_exception(self, exc: Exception) -> None:
        self.exceptions.append(exc)


class _FakeClient:
    def __init__(self) -> None:
        self.unloaded: list[str] = []
        self.resolve_calls: list[tuple[str, bool]] = []

    def unload_replica(self, replica_uuid: str, *, disk_path: str = "") -> bool:
        self.unloaded.append(f"{replica_uuid}:{disk_path}")
        return True

    def resolve_artifact_from_disk_v2(
        self, *, disk_path: str, verify_checksums: bool = True
    ):
        self.resolve_calls.append((disk_path, bool(verify_checksums)))

        class _Resp:
            pass

        resp = _Resp()
        resp.artifact_id = "aid"
        resp.disk_path = disk_path
        resp.canonical_index_bytes = json.dumps({}).encode("utf-8")
        resp.generation = 0
        return resp


class _RuntimeStub:
    def __init__(self) -> None:
        self.daemon_endpoint = "daemon"
        self.session_id = "sess"
        self.opts = StoreOptions()
        self.retry_policies = build_retry_policies()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.client = _FakeClient()
        self.futures: list[object] = []
        self._artifact_cache = ArtifactCache(
            daemon_endpoint="daemon", ttl_seconds=10, max_entries=8
        )
        self.closed = False

    def track_future(self, future: concurrent.futures.Future[object]) -> None:
        self.futures.append(future)

    def track_lease(self, lease: object | None) -> None:  # pragma: no cover - noop
        return None

    def ensure_client(self) -> _FakeClient:
        return self.client

    def cache_artifact_index(self, entry: ArtifactCacheEntry) -> None:
        self._artifact_cache.cache_artifact_index(entry)

    def get_artifact_index_by_disk_path(self, disk_path: str):
        return self._artifact_cache.get_artifact_index_by_disk_path(disk_path)

    def cache_key_mapping(self, *_, **__) -> None:  # pragma: no cover - noop
        return None

    def resolve_key_mapping_cached(
        self, *, key: str
    ) -> tuple[str | None, str | None]:  # pragma: no cover - noop
        return None, None

    @contextmanager
    def operation_span(self, *_args, **_kwargs):
        yield _DummySpan()

    def close(self) -> None:
        self.closed = True
        self.executor.shutdown(wait=True)

    def invalidate_artifact(self, *_args, **_kwargs) -> None:  # pragma: no cover - stub
        return None


def _make_payload(
    tensors: dict[str, torch.Tensor],
    *,
    replica_uuid: str = "r1",
    gate: threading.Event | None = None,
    on_iter: Callable[[], None] | None = None,
    generation: int | None = None,
    disk_path: str | None = None,
    source: store_daemon_pb2.MaterializationSource | None = None,
) -> MaterializationPayload:
    descriptors: list[TensorPayloadDescriptor] = []
    index: dict[str, list[object]] = {}
    offset = 0
    for name, tensor in tensors.items():
        size = int(tensor.element_size() * tensor.numel())
        shape = list(tensor.shape)
        stride = list(tensor.stride())
        dtype_str = str(tensor.dtype)
        index[name] = [offset, size, shape, stride, dtype_str, 0]
        descriptors.append(
            TensorPayloadDescriptor(
                name=name,
                dtype=dtype_str,
                shape=tuple(shape),
                stride=tuple(stride),
                buffer_offset=offset,
                byte_length=size,
                storage_offset=0,
            )
        )
        offset += size
    layout_bytes = json.dumps(index, separators=(",", ":")).encode("utf-8")

    def _iter():
        if on_iter:
            on_iter()
        for idx, desc in enumerate(descriptors):
            if gate is not None and idx > 0:
                gate.wait()
            yield desc, tensors[desc.name]

    return MaterializationPayload(
        artifact_id="aid",
        canonical_index_bytes=layout_bytes,
        descriptors=tuple(descriptors),
        payload_iter=_iter,
        state_dict=None,
        replica_uuid=replica_uuid,
        disk_path=disk_path,
        view_index_bytes=None,
        view_data_hash=None,
        source=source,
        device_uuid=None,
        generation=generation,
    )


def test_get_selective_tensors():
    runtime = _RuntimeStub()
    views = ViewOrchestrator(runtime)
    pipeline = MaterializationPipeline(runtime, views)
    captured: dict[str, object] = {"called": False}

    def fake_materialize(**kwargs):
        captured["called"] = True
        return _make_payload({"foo": torch.ones(1), "bar": torch.zeros(1)}, replica_uuid="p1")

    pipeline.set_materialize_fn(fake_materialize)
    result = pipeline.get(artifact_id="aid", tensor_names=["bar"])
    runtime.close()

    assert captured["called"] is True
    assert set(result.keys()) == {"bar"}
    assert torch.equal(result["bar"], torch.zeros(1))


def test_get_into_async_cancel_releases(monkeypatch):
    runtime = _RuntimeStub()
    views = ViewOrchestrator(runtime)
    pipeline = MaterializationPipeline(runtime, views)
    gate = threading.Event()
    release_calls: list[str] = []

    def fake_release(payload, _client):
        release_calls.append(payload.replica_uuid)

    monkeypatch.setattr(pipeline, "_release_materialized", fake_release)

    def fake_materialize(**_kwargs):
        return _make_payload(
            {"a": torch.ones(1), "b": torch.zeros(1)}, replica_uuid="rx", gate=gate
        )

    pipeline.set_materialize_fn(fake_materialize)
    target = {"a": torch.empty(1), "b": torch.empty(1)}
    fut = pipeline.get_into_async(
        target, artifact_id="aid", tensor_names=["a", "b"]
    )

    cancelled = fut.cancel()
    gate.set()
    with pytest.raises(ArtifactError) as excinfo:
        fut.result(timeout=1.0)
    runtime.close()

    assert cancelled is True
    assert excinfo.value.status_code == "CANCELLED"
    assert release_calls.count("rx") >= 1


def test_get_into_streaming_uses_iterator():
    runtime = _RuntimeStub()
    views = ViewOrchestrator(runtime)
    pipeline = MaterializationPipeline(runtime, views)
    iter_calls = {"count": 0}

    def on_iter():
        iter_calls["count"] += 1

    payload = _make_payload({"a": torch.ones(2)}, replica_uuid="iter", on_iter=on_iter)

    pipeline.set_materialize_fn(lambda **_kwargs: payload)

    target = {"a": torch.empty(2)}
    pipeline.get_into(target, artifact_id="aid")
    runtime.close()

    assert iter_calls["count"] == 1
    assert torch.equal(target["a"], torch.ones(2))


def test_release_uses_payload_replica_uuid():
    runtime = _RuntimeStub()
    views = ViewOrchestrator(runtime)
    pipeline = MaterializationPipeline(runtime, views)

    payload = _make_payload({"a": torch.ones(1)}, replica_uuid="rel-123")

    pipeline._release_materialized(payload, runtime.ensure_client())
    runtime.close()

    assert runtime.client.unloaded == ["rel-123:"]


def test_disk_fallback_verify_flag_passed():
    runtime = _RuntimeStub()
    views = ViewOrchestrator(runtime)
    pipeline = MaterializationPipeline(runtime, views)
    captured: dict[str, object] = {}

    def fake_materialize(**kwargs):
        captured["verify_checksums"] = kwargs.get("verify_checksums")
        return _make_payload({"a": torch.ones(1)}, replica_uuid="disk")

    pipeline.set_materialize_fn(fake_materialize)
    fallback = FallbackOptions(disk_path="/tmp/artifact", prefer_disk=True, verify_checksums=False)
    pipeline.get(artifact_id="aid", fallback=fallback)
    runtime.close()

    assert captured["verify_checksums"] is False


def test_disk_path_hint_prefers_disk_without_fallback():
    runtime = _RuntimeStub()
    views = ViewOrchestrator(runtime)
    pipeline = MaterializationPipeline(runtime, views)
    captured: dict[str, object] = {}

    def fake_materialize(**kwargs):
        captured["preference"] = kwargs.get("preference")
        captured["disk_path_hint"] = kwargs.get("disk_path_hint")
        captured["artifact_id"] = kwargs.get("artifact_id")
        return _make_payload({"a": torch.ones(1)}, replica_uuid="disk")

    pipeline.set_materialize_fn(fake_materialize)
    materialized, _ = pipeline.materialize_subset(
        artifact_id=None,
        key=None,
        device=0,
        fallback=None,
        tensor_names=None,
        disk_path_hint="/tmp/artifact",
    )
    runtime.close()

    assert captured["disk_path_hint"] == "/tmp/artifact"
    assert (
        captured["preference"]
        == store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_PREFER_DISK
    )
    assert runtime.client.resolve_calls == [("/tmp/artifact", True)]
    assert materialized.replica_uuid == "disk"


def test_fallback_for_disk_with_key_forces_disk_resolution():
    runtime = _RuntimeStub()
    views = ViewOrchestrator(runtime)
    pipeline = MaterializationPipeline(runtime, views)
    captured: dict[str, object] = {}

    def fake_materialize(**kwargs):
        captured["key"] = kwargs.get("key")
        captured["disk_path_hint"] = kwargs.get("disk_path_hint")
        captured["preference"] = kwargs.get("preference")
        captured["verify_checksums"] = kwargs.get("verify_checksums")
        return _make_payload(
            {"a": torch.ones(1)},
            replica_uuid="disk",
            disk_path=kwargs.get("disk_path_hint"),
        )

    pipeline.set_materialize_fn(fake_materialize)
    fallback = FallbackOptions.for_disk("/tmp/model", verify=False)
    materialized, _ = pipeline.materialize_subset(
        artifact_id=None,
        key="model:v1",
        device=0,
        fallback=fallback,
        tensor_names=None,
    )
    runtime.close()

    assert captured["key"] is None
    assert captured["disk_path_hint"] == "/tmp/model"
    assert (
        captured["preference"]
        == store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_PREFER_DISK
    )
    assert captured["verify_checksums"] is False
    assert runtime.client.resolve_calls == [("/tmp/model", False)]
    assert materialized.replica_uuid == "disk"


def test_local_preference_blocks_p2p_source():
    runtime = _RuntimeStub()
    views = ViewOrchestrator(runtime)
    pipeline = MaterializationPipeline(runtime, views)
    payload = _make_payload(
        {"a": torch.ones(1)},
        replica_uuid="p2p-replica",
        source=store_daemon_pb2.MaterializationSource.MATERIALIZATION_SOURCE_P2P,
    )

    pipeline.set_materialize_fn(lambda **_kwargs: payload)
    with pytest.raises(ArtifactError) as excinfo:
        pipeline.get(artifact_id="aid", fallback=FallbackOptions.local_only())
    runtime.close()

    assert excinfo.value.status_code == "FAILED_PRECONDITION"
    assert runtime.client.unloaded == ["p2p-replica:"]


def test_disk_path_cache_reuses_index_without_resolve():
    runtime = _RuntimeStub()
    views = ViewOrchestrator(runtime)
    pipeline = MaterializationPipeline(runtime, views)
    payload = _make_payload({"a": torch.ones(1)}, generation=7)
    canonical_bytes = payload.canonical_index_bytes
    cache_entry = ArtifactCacheEntry(
        artifact_id="aid",
        canonical_index_bytes=canonical_bytes,
        parsed_index=canonical_index_from_bytes(canonical_bytes),
        generation=7,
        disk_path="/tmp/artifact",
        expires_at=time.monotonic() + 5.0,
    )
    runtime.cache_artifact_index(cache_entry)
    captured: dict[str, object] = {}

    def fake_materialize(**kwargs):
        captured["artifact_id"] = kwargs.get("artifact_id")
        captured["canonical_index_hint"] = kwargs.get("canonical_index_hint")
        captured["generation_hint"] = kwargs.get("generation_hint")
        return payload

    pipeline.set_materialize_fn(fake_materialize)
    materialized, _ = pipeline.materialize_subset(
        artifact_id=None,
        key=None,
        device=0,
        fallback=None,
        tensor_names=None,
        disk_path_hint="/tmp/artifact",
    )
    runtime.close()

    assert captured["artifact_id"] == "aid"
    assert captured["canonical_index_hint"] == canonical_bytes
    assert captured["generation_hint"] == 7
    assert runtime.client.resolve_calls == []
    assert materialized.generation == 7


def test_materialize_subset_preserves_generation():
    runtime = _RuntimeStub()
    views = ViewOrchestrator(runtime)
    pipeline = MaterializationPipeline(runtime, views)
    payload = _make_payload({"a": torch.ones(1), "b": torch.zeros(1)}, generation=5)

    pipeline.set_materialize_fn(lambda **_kwargs: payload)
    materialized, _ = pipeline.materialize_subset(
        artifact_id="aid",
        key=None,
        device=0,
        fallback=None,
        tensor_names=["a"],
    )
    runtime.close()

    assert materialized.generation == 5


def test_prefetch_wait_retries_retryable_rpc_errors():
    runtime = _RuntimeStub()
    ticket = PrefetchTicket(
        replica_uuid="retryable",
        artifact_id="aid",
        device=torch.device("cuda", 0),
        expires_at=None,
        started_at=time.monotonic(),
        view_hash=None,
        runtime_ref=weakref.ref(runtime),
    )
    calls = {"count": 0}

    def fake_query(_ticket):
        calls["count"] += 1
        if calls["count"] == 1:
            raise PrefetchRpcError("temporary", retryable=True, status_code="UNAVAILABLE")
        resp = store_daemon_v2_pb2.QueryReplicaStatusResponse()
        resp.ticket.replica_uuid = "retryable"
        resp.ticket.status = (
            store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_ALLOCATED
        )
        return resp

    runtime.client.query_replica_status = fake_query

    assert ticket.wait(timeout=0.5) is True
    runtime.close()
    assert calls["count"] >= 2


def test_prefetch_wait_stops_on_nonretryable_rpc_errors():
    runtime = _RuntimeStub()
    ticket = PrefetchTicket(
        replica_uuid="fatal",
        artifact_id="aid",
        device=torch.device("cuda", 0),
        expires_at=None,
        started_at=time.monotonic(),
        view_hash=None,
        runtime_ref=weakref.ref(runtime),
    )

    def fake_query(_ticket):
        raise PrefetchRpcError("bad request", retryable=False, status_code="INVALID_ARGUMENT")

    runtime.client.query_replica_status = fake_query

    assert ticket.wait(timeout=0.2) is False
    runtime.close()
