#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import concurrent.futures
import json
import threading
from contextlib import contextmanager
from typing import Callable

import pytest
import torch

from tensorcast.api._config import GetArtifactOptions
from tensorcast.api._materialize import (
    MaterializationPayload,
    TensorPayloadDescriptor,
    _resolve_collective_load_group,
)
from tensorcast.api.context import (
    BroadcastContext,
    CallContext,
    CollectiveLoadGroup,
    TransportSchedulingGroup,
)
from tensorcast.api.store.cache import ArtifactCache, ArtifactCacheEntry
from tensorcast.api.store.materialization import MaterializationPipeline
from tensorcast.api.store.retry import build_retry_policies
from tensorcast.api.store.types import (
    ArtifactError,
    StoreCapabilities,
    StoreOptions,
)
from tensorcast.api.store.views import ViewOrchestrator
from tensorcast.proto.daemon.v2 import store_daemon_pb2


@pytest.fixture(autouse=True)
def _force_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)


@pytest.fixture(autouse=True)
def _patch_validate_targets(monkeypatch):
    from tensorcast.api.store import materialization as mat_mod

    def _pair(*, canonical_index, target, source, device_id, required_names=None):
        pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        names = (
            required_names
            if required_names is not None
            else [entry.name for entry in canonical_index.entries]
        )
        for name in names:
            entry = next(e for e in canonical_index.entries if e.name == name)
            pairs.append((target[entry.name], source[entry.name]))
        return pairs

    monkeypatch.setattr(mat_mod, "validate_targets", _pair)


def test_materialization_proto_alignment():
    resp = store_daemon_pb2.MaterializeReplicaResponse()
    assert hasattr(resp, "canonical_index_bytes")
    assert hasattr(resp, "view_index_bytes")
    assert hasattr(resp, "generation")
    assert not hasattr(resp, "canonical_index_json")


def test_resolve_collective_load_group_from_explicit_context() -> None:
    ctx = CallContext(
        collective=CollectiveLoadGroup(
            group_id="explicit-group",
            world_size=4,
            rank=1,
        )
    )
    group = _resolve_collective_load_group(ctx)
    assert group is not None
    assert group.group_id == "explicit-group"
    assert group.world_size == 4
    assert group.rank == 1


def test_resolve_collective_load_group_rejects_invalid_context() -> None:
    ctx = CallContext(
        collective=CollectiveLoadGroup(group_id="bad", world_size=1, rank=0)
    )
    assert _resolve_collective_load_group(ctx) is None


def test_resolve_collective_load_group_ignores_ambient_gpu_env(monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "5")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    assert _resolve_collective_load_group(None) is None


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

    def get_artifact_index_by_id(self, artifact_id: str) -> bytes:
        # Region-backed get_into may probe artifact indices even when tests do not
        # set up a daemon. Returning an empty canonical index forces the region-backed
        # path to fall back cleanly.
        del artifact_id
        return b"{}"

    def import_artifact_from_path_v2(self, *, path: str, verify_checksums: bool = True):
        self.resolve_calls.append((path, bool(verify_checksums)))

        class _Resp:
            pass

        resp = _Resp()
        resp.artifact_id = "aid"
        resp.canonical_index_bytes = json.dumps({}).encode("utf-8")
        resp.generation = 0
        return resp

    def import_artifact_from_path_stream_v2(
        self, *, path: str, verify_checksums: bool = True
    ):
        resp = self.import_artifact_from_path_v2(
            path=path,
            verify_checksums=verify_checksums,
        )
        final_resp = store_daemon_pb2.ImportArtifactFromPathResponse(
            artifact_id=resp.artifact_id,
            canonical_index_bytes=resp.canonical_index_bytes,
            generation=resp.generation,
        )
        event = store_daemon_pb2.ImportArtifactFromPathStreamEvent(
            seq=1,
            phase=store_daemon_pb2.IMPORT_ARTIFACT_PHASE_DONE,
            done=True,
        )
        event.result.CopyFrom(final_resp)
        yield event


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
        self._capabilities = StoreCapabilities(
            mem_pool_bytes=0,
            tx_slice_bytes=0,
            artifact_chunk_bytes=0,
            server_config=None,
        )

    def track_future(self, future: concurrent.futures.Future[object]) -> None:
        self.futures.append(future)

    def track_lease(self, lease: object | None) -> None:  # pragma: no cover - noop
        return None

    def ensure_client(self) -> _FakeClient:
        return self.client

    def cache_artifact_index(self, entry: ArtifactCacheEntry) -> None:
        self._artifact_cache.cache_artifact_index(entry)

    def get_artifact_index_by_disk_path(
        self, disk_path: str
    ) -> ArtifactCacheEntry | None:
        del disk_path
        return None

    def get_artifact_index_cached(self, artifact_id: str) -> ArtifactCacheEntry | None:
        return self._artifact_cache.get_artifact_index_cached(artifact_id)

    def cache_key_mapping(self, *_, **__) -> None:  # pragma: no cover - noop
        return None

    def resolve_key_mapping_cached(
        self, *, key: str
    ) -> tuple[str | None, str | None]:  # pragma: no cover - noop
        del key
        return None, None

    @property
    def capabilities(self) -> StoreCapabilities:
        return self._capabilities

    @contextmanager
    def operation_span(self, *_args, **_kwargs):
        yield _DummySpan()

    def close(self) -> None:
        self.executor.shutdown(wait=True)


def _make_payload(
    tensors: dict[str, torch.Tensor],
    *,
    replica_uuid: str = "r1",
    gate: threading.Event | None = None,
    on_iter: Callable[[], None] | None = None,
    generation: int | None = None,
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
        disk_path=None,
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
        return _make_payload(
            {"foo": torch.ones(1), "bar": torch.zeros(1)}, replica_uuid="p1"
        )

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
    fut = pipeline.get_into_async(target, artifact_id="aid", tensor_names=["a", "b"])

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
    pipeline.get(
        artifact_id="aid",
        options=GetArtifactOptions(
            source="disk_first",
            verify_checksums=False,
        ),
    )
    runtime.close()

    assert captured["verify_checksums"] is False


def test_disk_fallback_allows_local_replica_source():
    runtime = _RuntimeStub()
    views = ViewOrchestrator(runtime)
    pipeline = MaterializationPipeline(runtime, views)

    def fake_materialize(**_kwargs):
        return _make_payload(
            {"a": torch.ones(1)},
            replica_uuid="local",
            source=store_daemon_pb2.MaterializationSource.MATERIALIZATION_SOURCE_LOCAL_REPLICA,
        )

    pipeline.set_materialize_fn(fake_materialize)
    result = pipeline.get(
        artifact_id="aid",
        options=GetArtifactOptions(
            source="disk_first",
            verify_checksums=False,
        ),
    )
    runtime.close()

    assert torch.equal(result["a"], torch.ones(1))


def test_prefer_disk_sets_source_policy():
    runtime = _RuntimeStub()
    views = ViewOrchestrator(runtime)
    pipeline = MaterializationPipeline(runtime, views)
    captured: dict[str, object] = {}

    def fake_materialize(**kwargs):
        source_policy = kwargs.get("source_policy")
        captured["allow_disk"] = (
            bool(source_policy.allow_disk) if source_policy is not None else None
        )
        captured["artifact_id"] = kwargs.get("artifact_id")
        return _make_payload({"a": torch.ones(1)}, replica_uuid="disk")

    pipeline.set_materialize_fn(fake_materialize)
    materialized, _ = pipeline.materialize_subset(
        artifact_id="aid",
        key=None,
        device=0,
        options=GetArtifactOptions(source="disk_first"),
        tensor_names=None,
    )
    runtime.close()

    assert captured["artifact_id"] == "aid"
    assert captured["allow_disk"] is True
    assert materialized.replica_uuid == "disk"


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
        tensor_names=["a"],
    )
    runtime.close()

    assert materialized.generation == 5


def test_materialize_subset_forwards_transport_hints():
    runtime = _RuntimeStub()
    views = ViewOrchestrator(runtime)
    pipeline = MaterializationPipeline(runtime, views)
    payload = _make_payload({"a": torch.ones(1)}, replica_uuid="transport")
    captured: dict[str, object] = {}
    group = TransportSchedulingGroup(
        group_kind="weight_broadcast",
        group_id="model-a:v42",
        total_parts=16,
        part_id="daemon-1",
        priority=7,
        epoch=42,
    )

    def fake_materialize(**kwargs):
        captured.update(kwargs)
        return payload

    pipeline.set_materialize_fn(fake_materialize)
    materialized, _ = pipeline.materialize_subset(
        artifact_id="aid",
        key=None,
        device=0,
        tensor_names=None,
        transport_request_id="transport-req-1",
        transport_scheduling_group=group,
    )
    runtime.close()

    assert materialized.replica_uuid == "transport"
    assert captured["transport_request_id"] == "transport-req-1"
    assert captured["transport_scheduling_group"] == group


def test_get_forwards_broadcast_context_hint():
    runtime = _RuntimeStub()
    views = ViewOrchestrator(runtime)
    pipeline = MaterializationPipeline(runtime, views)
    payload = _make_payload({"a": torch.ones(1)}, replica_uuid="broadcast")
    captured: dict[str, object] = {}
    ctx = CallContext(
        broadcast=BroadcastContext(
            session_id="broadcast-session-1",
            strict_parent=False,
        )
    )

    def fake_materialize(**kwargs):
        captured.update(kwargs)
        return payload

    pipeline.set_materialize_fn(fake_materialize)
    result = pipeline.get(artifact_id="aid", ctx=ctx)
    runtime.close()

    assert torch.equal(result["a"], torch.ones(1))
    assert captured["broadcast_session_id"] == "broadcast-session-1"
    assert captured["broadcast_strict_parent"] is False


def test_materialize_subset_explicit_broadcast_hint_overrides_context():
    runtime = _RuntimeStub()
    views = ViewOrchestrator(runtime)
    pipeline = MaterializationPipeline(runtime, views)
    payload = _make_payload({"a": torch.ones(1)}, replica_uuid="broadcast")
    captured: dict[str, object] = {}
    ctx = CallContext(
        broadcast=BroadcastContext(
            session_id="ctx-session",
            strict_parent=False,
        )
    )

    def fake_materialize(**kwargs):
        captured.update(kwargs)
        return payload

    pipeline.set_materialize_fn(fake_materialize)
    materialized, _ = pipeline.materialize_subset(
        artifact_id="aid",
        key=None,
        device=0,
        tensor_names=None,
        ctx=ctx,
        broadcast_session_id="explicit-session",
        broadcast_strict_parent=True,
    )
    runtime.close()

    assert materialized.replica_uuid == "broadcast"
    assert captured["broadcast_session_id"] == "explicit-session"
    assert captured["broadcast_strict_parent"] is True
