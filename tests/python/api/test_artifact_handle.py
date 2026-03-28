#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import json
import threading
import time
import weakref
from dataclasses import replace
from typing import Sequence, cast

import pytest
import torch

from tensorcast.api._materialize import MaterializationPayload, TensorPayloadDescriptor
from tensorcast.api.store import Store
from tensorcast.api.store.artifact import Artifact
from tensorcast.api.store.cache import ArtifactCache, ArtifactCacheEntry
from tensorcast.api.store.common import canonical_index_from_bytes
from tensorcast.api.store.retry import build_retry_policies
from tensorcast.api.store.types import ArtifactError, FallbackOptions, StoreOptions
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.types import (
    BuilderMode,
    ServingArtifactManifest,
    ServingRuntimePolicy,
    build_serving_manifest_ref,
)


def _build_payload(
    tensors: dict[str, torch.Tensor],
) -> tuple[bytes, MaterializationPayload]:
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
    def __init__(
        self,
        canonical_index_bytes: bytes,
        *,
        disk_generation: int | None = None,
        disk_artifact_id: str | None = None,
        startup_in_progress_failures: int = 0,
    ) -> None:
        self.canonical_index_bytes = canonical_index_bytes
        self.disk_generation = disk_generation
        self.disk_artifact_id = disk_artifact_id
        self.startup_in_progress_failures = startup_in_progress_failures
        self.unloaded: list[tuple[str, str]] = []
        self.get_index_calls = 0
        self.resolve_calls: list[tuple[str, bool]] = []

    def get_artifact_index_by_id(self, artifact_id: str) -> bytes:
        self.get_index_calls += 1
        return self.canonical_index_bytes

    def import_artifact_from_path_v2(self, *, path: str, verify_checksums: bool = True):
        self.resolve_calls.append((path, bool(verify_checksums)))
        generation = self.disk_generation if self.disk_generation is not None else 0

        class _Resp:
            pass

        resp = _Resp()
        resp.artifact_id = self.disk_artifact_id or "mi2:idx:data"
        resp.canonical_index_bytes = self.canonical_index_bytes
        resp.generation = generation
        return resp

    def import_artifact_from_path_stream_v2(
        self, *, path: str, verify_checksums: bool = True
    ):
        if self.startup_in_progress_failures > 0:
            self.startup_in_progress_failures -= 1
            raise RuntimeError(
                "Local StoreDaemon (daemon) is not available. Msg: "
                "daemon startup still in progress: prewarming"
            )
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

    def invalidate_artifact(
        self, artifact_id: str | None, *, key=None, reason=None
    ) -> None:
        self._artifact_cache.invalidate_artifact(artifact_id or "", reason=reason)

    def resolve_key_mapping_cached(
        self, *, key: str
    ) -> tuple[str | None, str | None]:
        return self._key_cache.get(key, (None, None))

    def cache_key_mapping(
        self,
        key: str,
        *,
        artifact_id: str | None,
        disk_path: str | None = None,
        ttl_override=None,
    ) -> None:
        del ttl_override
        self._key_cache[key] = (artifact_id, disk_path)


class _PipelineStub:
    def __init__(self, payload: MaterializationPayload) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []
        self.get_into_calls: list[dict[str, object]] = []
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

    def _release_materialized(
        self, payload: MaterializationPayload, client: _ClientStub
    ) -> None:
        self.released.append(payload.replica_uuid)
        client.unload_replica(
            payload.replica_uuid, disk_path=getattr(payload, "disk_path", "") or ""
        )

    def get_into(
        self,
        target: dict[str, torch.Tensor],
        *,
        artifact_id: str | None,
        key: str | None,
        device,
        fallback,
        options=None,
        tensor_names: Sequence[str] | None = None,
        view_spec=None,
        view_data_hash=None,
        view_index_hint=None,
        replica_uuid=None,
        ctx=None,
    ) -> None:
        del ctx
        self.get_into_calls.append(
            {
                "artifact_id": artifact_id,
                "key": key,
                "device": device,
                "tensor_names": tuple(tensor_names) if tensor_names else None,
                "view_spec": view_spec,
                "view_data_hash": view_data_hash,
                "view_index_hint": view_index_hint,
                "replica_uuid": replica_uuid,
            }
        )
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

    result = artifact.subset(["bar"]).tensor_dict(device="cpu")

    assert set(result.keys()) == {"bar"}
    assert pipeline.calls and pipeline.calls[0]["tensor_names"] == ("bar",)
    assert pipeline.calls[0]["view_index_hint"]
    assert runtime._client.unloaded == []
    assert pipeline.released == []


def test_subset_derives_view_metadata_eagerly():
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

    derived = artifact.subset(["bar"])

    assert derived._view_metadata is not None
    assert derived._view_metadata.tensor_names == ("bar",)
    assert derived._view_metadata.selected_index is not None
    assert tuple(entry.name for entry in derived._view_metadata.selected_index.entries) == (
        "bar",
    )
    assert derived._view_metadata.view_index_bytes
    assert derived._view_metadata.view_data_hash is None
    assert derived._view_metadata.view_id


def test_selection_reuses_eager_view_metadata():
    canonical_bytes, payload = _build_payload(
        {"foo": torch.ones(2, 3), "bar": torch.zeros(2, 3)}
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

    derived = artifact.view(transpose={"foo": [(0, 1)]})
    assert derived._view_metadata is not None
    assert derived._view_metadata.view_index_bytes
    assert derived._view_metadata.view_id

    selection = derived._build_artifact_selection()

    assert selection.view_id == derived._view_metadata.view_id
    assert derived._view_metadata is not None
    assert derived._view_metadata.view_index_bytes
    assert derived._view_metadata.selected_index is not None


def test_tensor_dict_with_diagnostics_reports_source_and_bytes():
    canonical_bytes, payload = _build_payload(
        {"foo": torch.ones(4, dtype=torch.float32), "bar": torch.zeros(2)}
    )
    payload = replace(
        payload,
        source=store_daemon_pb2.MATERIALIZATION_SOURCE_P2P,
        ticket_replica_uuid="ticket-1",
        ticket_status=store_daemon_pb2.MATERIALIZE_REPLICA_STATUS_ALLOCATED,
    )
    runtime = _RuntimeStub(_ClientStub(canonical_bytes))
    pipeline = _PipelineStub(payload)
    store = _StoreStub(runtime, pipeline)
    artifact = Artifact(
        store_ref=_store_ref(store),
        artifact_id="aid",
        canonical_index_bytes=canonical_bytes,
    )

    result = artifact.tensor_dict_with_diagnostics(device="cpu")

    assert set(result.tensors) == {"foo", "bar"}
    diagnostics = result.diagnostics
    assert diagnostics.source == "p2p"
    assert diagnostics.source_code == int(store_daemon_pb2.MATERIALIZATION_SOURCE_P2P)
    assert diagnostics.tensor_count == 2
    assert diagnostics.total_bytes == sum(
        int(desc.byte_length) for desc in payload.descriptors
    )
    assert diagnostics.replica_uuid == "replica-1"
    assert diagnostics.ticket_replica_uuid == "ticket-1"
    assert diagnostics.ticket_status == "allocated"
    assert diagnostics.materialize_sec >= 0.0
    assert diagnostics.tensor_bind_sec >= 0.0
    assert diagnostics.total_sec >= diagnostics.materialize_sec


def test_bind_coerces_serving_manifest_into_runtime_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_bytes, payload = _build_payload({"foo": torch.ones(1)})
    runtime = _RuntimeStub(_ClientStub(canonical_bytes))
    pipeline = _PipelineStub(payload)
    store = _StoreStub(runtime, pipeline)
    artifact = Artifact(
        store_ref=_store_ref(store),
        artifact_id="aid",
        canonical_index_bytes=canonical_bytes,
    )
    captured: dict[str, object] = {}

    def _fake_bind_owned(self, **kwargs):
        del self
        captured.update(kwargs)
        return "binding"

    monkeypatch.setattr(Artifact, "_bind_owned", _fake_bind_owned)

    manifest = ServingArtifactManifest(
        framework_name="torch",
        adapter_version="adapter-v1",
        serving_abi_version="abi-v1",
        representation_contract_hash="bafkrepresentation",
        serving_build_digest="bafkbuilddigest",
        tensor_schema_hash="bafktensorschema",
        canonical_tensor_count=1,
        serving_manifest_ref=build_serving_manifest_ref("__alt_manifest__.json"),
        builder_mode=BuilderMode.BINDING_FINALIZE,
        build_pipeline_version="pipeline-v1",
    )

    result = artifact.bind(
        device="cuda:0",
        serving_runtime_policy=manifest,
    )

    assert result == "binding"
    assert captured["device"] == torch.device("cuda:0")
    assert captured["serving_runtime_policy"] == ServingRuntimePolicy(
        require_manifest=True,
        serving_manifest_ref="tensor:__alt_manifest__.json",
        expected_representation_contract_hash="bafkrepresentation",
        expected_serving_build_digest="bafkbuilddigest",
    )


def test_tensor_into_materializes_subset_only():
    tensors = {
        "foo": torch.zeros(2, dtype=torch.float32),
        "bar": torch.ones(2, dtype=torch.float32),
    }
    canonical_bytes, payload = _build_payload(tensors)
    runtime = _RuntimeStub(_ClientStub(canonical_bytes))
    pipeline = _PipelineStub(payload)
    store = _StoreStub(runtime, pipeline)
    artifact = Artifact(
        store_ref=_store_ref(store),
        artifact_id="aid",
        canonical_index_bytes=canonical_bytes,
    )

    target = torch.zeros(2, dtype=torch.float32)
    artifact.tensor_into("bar", target, device=torch.device("cpu"))

    assert torch.allclose(target, tensors["bar"])
    assert pipeline.get_into_calls
    assert pipeline.get_into_calls[0]["tensor_names"] == ("bar",)


def test_prefetch_returns_operation():
    tensors = {"foo": torch.ones(1, dtype=torch.float32)}
    canonical_bytes, payload = _build_payload(tensors)
    payload = replace(
        payload,
        ticket_replica_uuid="prefetch-replica",
        ticket_created_at_ts=time.monotonic(),
        ticket_expires_at_ts=time.monotonic() + 5.0,
    )
    runtime = _RuntimeStub(_ClientStub(canonical_bytes))
    pipeline = _PipelineStub(payload)
    store = _StoreStub(runtime, pipeline)
    artifact = Artifact(
        store_ref=_store_ref(store),
        artifact_id="aid",
        canonical_index_bytes=canonical_bytes,
    )

    op = artifact.prefetch(device="cuda:0")

    from tensorcast.proto.daemon.v2 import store_daemon_pb2

    assert op.operation_id == "prefetch-replica"
    assert pipeline.calls
    assert (
        pipeline.calls[0]["lease_mode"]
        == store_daemon_pb2.LeaseMode.LEASE_MODE_NO_LEASE
    )


def test_prefetch_accepts_cpu_device() -> None:
    canonical_bytes, payload = _build_payload({"foo": torch.ones(1)})
    runtime = _RuntimeStub(_ClientStub(canonical_bytes))
    pipeline = _PipelineStub(payload)
    store = _StoreStub(runtime, pipeline)
    artifact = Artifact(
        store_ref=_store_ref(store),
        artifact_id="aid",
        canonical_index_bytes=canonical_bytes,
    )

    op = artifact.prefetch(device="cpu")

    assert op.operation_id == "replica-1"
    assert pipeline.calls
    assert str(pipeline.calls[0]["device"]) == "cpu"
    assert (
        pipeline.calls[0]["lease_mode"]
        == store_daemon_pb2.LeaseMode.LEASE_MODE_NO_LEASE
    )


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
    runtime.cache_key_mapping("mapped", artifact_id="aid")
    pipeline = _PipelineStub(payload)
    store = _StoreStub(runtime, pipeline)
    artifact = Artifact(store_ref=_store_ref(store), key="mapped")

    assert artifact.artifact_id == "aid"
    clone = artifact.with_fallback(FallbackOptions(prefer="disk", allow_p2p=False))

    assert clone.artifact_id == "aid"
    assert clone.key == "mapped"
    assert clone.tensor_names == ("foo",)


def test_describe_uses_cached_generation_without_fetch():
    canonical_bytes, payload = _build_payload({"foo": torch.ones(1)})
    runtime = _RuntimeStub(_ClientStub(canonical_bytes))
    cache_entry = ArtifactCacheEntry(
        artifact_id="aid",
        canonical_index_bytes=canonical_bytes,
        parsed_index=canonical_index_from_bytes(canonical_bytes),
        generation=42,
        expires_at=time.monotonic() + 1.0,
    )
    runtime.cache_artifact_index(cache_entry)
    pipeline = _PipelineStub(payload)
    store = _StoreStub(runtime, pipeline)
    artifact = Artifact(store_ref=_store_ref(store), artifact_id="aid")

    desc = artifact.describe()

    assert desc.generation == 42
    assert runtime._client.get_index_calls == 0


def test_from_dict_accepts_key_and_artifact_id():
    canonical_bytes, payload = _build_payload({"foo": torch.ones(1)})
    runtime = _RuntimeStub(_ClientStub(canonical_bytes))
    runtime.cache_key_mapping("mapped", artifact_id="aid")
    pipeline = _PipelineStub(payload)
    store = _StoreStub(runtime, pipeline)
    artifact = Artifact(store_ref=_store_ref(store), key="mapped")
    assert artifact.artifact_id == "aid"

    serialized = artifact.to_dict()
    restored = Artifact.from_dict(serialized, store=cast(Store, store))

    assert restored.artifact_id == "aid"
    assert restored.key == "mapped"
    assert restored.tensor_names == ("foo",)


def test_from_disk_resolves_once_and_caches_generation():
    canonical_bytes, payload = _build_payload({"foo": torch.ones(1)})
    client = _ClientStub(
        canonical_bytes, disk_generation=11, disk_artifact_id="mi2:idx:data"
    )
    runtime = _RuntimeStub(client)
    store = Store("daemon", runtime=runtime)

    artifact = store.from_disk("/tmp/artifact")
    desc = artifact.describe()
    repeat = artifact.describe()

    assert desc.artifact_id == "mi2:idx:data"
    assert desc.generation == 11
    assert repeat.generation == 11
    assert client.resolve_calls == [("/tmp/artifact", True)]
    assert client.get_index_calls == 0
    cached = runtime.get_artifact_index_cached("mi2:idx:data")
    assert cached is not None
    assert cached.canonical_index_bytes == canonical_bytes
    assert cached.generation == 11


def test_from_disk_progress_mode_uses_stream_resolution():
    canonical_bytes, _payload = _build_payload({"foo": torch.ones(1)})
    client = _ClientStub(
        canonical_bytes,
        disk_generation=5,
        disk_artifact_id="mi2:idx:data",
    )
    runtime = _RuntimeStub(client)
    store = Store("daemon", runtime=runtime)

    artifact = store.from_disk("/tmp/artifact", show_progress=True)

    assert artifact.artifact_id == "mi2:idx:data"
    assert client.resolve_calls == [("/tmp/artifact", True)]


def test_from_disk_retries_daemon_startup_in_progress(monkeypatch):
    canonical_bytes, _payload = _build_payload({"foo": torch.ones(1)})
    client = _ClientStub(
        canonical_bytes,
        disk_generation=7,
        disk_artifact_id="mi2:idx:data",
        startup_in_progress_failures=2,
    )
    runtime = _RuntimeStub(client)
    store = Store("daemon", runtime=runtime)
    sleeps: list[float] = []

    monkeypatch.setattr(time, "sleep", lambda seconds: sleeps.append(seconds))

    artifact = store.from_disk("/tmp/artifact")

    assert artifact.artifact_id == "mi2:idx:data"
    assert len(sleeps) == 2
    assert client.resolve_calls == [("/tmp/artifact", True)]


def test_ensure_metadata_sets_under_lock(monkeypatch):
    canonical_bytes, payload = _build_payload({"foo": torch.ones(1)})
    client = _ClientStub(canonical_bytes)
    runtime = _RuntimeStub(client)
    pipeline = _PipelineStub(payload)
    store = _StoreStub(runtime, pipeline)
    artifact = Artifact(store_ref=_store_ref(store), artifact_id="aid")

    lock_checked = threading.Event()
    original_set_metadata = Artifact._set_metadata

    def _wrapped(self, *args, **kwargs):
        if not self._lock._is_owned():
            raise AssertionError("metadata updated without holding artifact lock")
        lock_checked.set()
        return original_set_metadata(self, *args, **kwargs)

    monkeypatch.setattr(Artifact, "_set_metadata", _wrapped)

    assert artifact.tensor_names == ("foo",)
    assert lock_checked.is_set()
