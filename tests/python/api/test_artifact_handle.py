#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import json
import threading
import time
import weakref
from dataclasses import replace
from typing import Any, Sequence, cast

import pytest
import torch

from tensorcast.api._config import (
    GetArtifactOptions,
    RegionBackedMode,
    RetrievalPolicy,
    RetrievalPreference,
)
from tensorcast.api._materialize import MaterializationPayload, TensorPayloadDescriptor
from tensorcast.api.store import ArtifactRealizationSpec, Store
from tensorcast.api.store.artifact import Artifact
from tensorcast.api.store.cache import ArtifactCache, ArtifactCacheEntry
from tensorcast.api.store.common import canonical_index_from_bytes
from tensorcast.api.store.mapped_binding import CopyPlanEntry, Range
from tensorcast.api.store.materialization import GetIntoResult
from tensorcast.api.store.retry import build_retry_policies
from tensorcast.api.store.types import ArtifactError, StoreOptions
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.types import (
    BuilderMode,
    RuntimeArtifactManifest,
    RuntimeArtifactPolicy,
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
        mounted_generation: int | None = None,
        mounted_artifact_id: str | None = None,
        trusted_content_artifact_id: str | None = None,
        startup_in_progress_failures: int = 0,
    ) -> None:
        self.canonical_index_bytes = canonical_index_bytes
        self.disk_generation = disk_generation
        self.disk_artifact_id = disk_artifact_id
        self.mounted_generation = mounted_generation
        self.mounted_artifact_id = mounted_artifact_id
        self.trusted_content_artifact_id = trusted_content_artifact_id
        self.startup_in_progress_failures = startup_in_progress_failures
        self.unloaded: list[tuple[str, str]] = []
        self.get_index_calls = 0
        self.import_calls: list[tuple[str, bool]] = []
        self.resolve_public_disk_calls: list[tuple[str, bool]] = []
        self.promote_mounted_source_calls: list[tuple[str, bool, float | None]] = []
        self.publish_calls: list[tuple[str, str]] = []

    def get_artifact_index_by_id(self, artifact_id: str) -> bytes:
        self.get_index_calls += 1
        return self.canonical_index_bytes

    def import_artifact_from_path(self, *, path: str, verify_checksums: bool = True):
        self.import_calls.append((path, bool(verify_checksums)))
        generation = self.disk_generation if self.disk_generation is not None else 0

        class _Resp:
            pass

        resp = _Resp()
        resp.artifact_id = self.disk_artifact_id or "mi2:idx:data"
        resp.canonical_index_bytes = self.canonical_index_bytes
        resp.generation = generation
        return resp

    def import_artifact_from_path_stream(
        self, *, path: str, verify_checksums: bool = True
    ):
        if self.startup_in_progress_failures > 0:
            self.startup_in_progress_failures -= 1
            raise RuntimeError(
                "Local StoreDaemon (daemon) is not available. Msg: "
                "daemon startup still in progress: prewarming"
            )
        resp = self.import_artifact_from_path(
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

    def resolve_public_disk_source(self, *, path: str, verify_checksums: bool = True):
        self.resolve_public_disk_calls.append((path, bool(verify_checksums)))
        if self.startup_in_progress_failures > 0:
            self.startup_in_progress_failures -= 1
            raise RuntimeError(
                "Local StoreDaemon (daemon) is not available. Msg: "
                "daemon startup still in progress: prewarming"
            )
        mounted_generation = (
            self.mounted_generation
            if self.mounted_generation is not None
            else self.disk_generation
        )
        mounted_artifact_id = (
            self.mounted_artifact_id or "msa1:test-session~policy~safetensors~deadbeef"
        )
        trusted_content_artifact_id = (
            self.trusted_content_artifact_id or self.disk_artifact_id or ""
        )
        return store_daemon_pb2.ResolvePublicDiskSourceResponse(
            source=store_daemon_pb2.PublicDiskSourceHandle(
                path=path,
                canonical_index_bytes=self.canonical_index_bytes,
                artifact_id=mounted_artifact_id,
                generation=mounted_generation or 0,
                verify_checksums=bool(verify_checksums),
                trusted_content_artifact_id=trusted_content_artifact_id,
            )
        )

    def publish_replica_key(self, *, key: str, descriptor) -> bool:
        self.publish_calls.append((key, str(descriptor.artifact_id)))
        return True

    def promote_mounted_source_artifact(
        self,
        *,
        artifact_id: str,
        verify_checksums: bool = True,
        timeout_s: float | None = None,
    ):
        self.promote_mounted_source_calls.append(
            (
                artifact_id,
                bool(verify_checksums),
                float(timeout_s) if timeout_s is not None else None,
            )
        )
        return store_daemon_pb2.PromoteMountedSourceArtifactResponse(
            artifact_id=self.disk_artifact_id or "mi2:idx:data",
            canonical_index_bytes=self.canonical_index_bytes,
            generation=self.disk_generation or 0,
            source_artifact_id=artifact_id,
        )

    def unload_replica(self, replica_uuid: str, *, disk_path: str = "") -> bool:
        self.unloaded.append((replica_uuid, disk_path))
        return True


class _RuntimeStub:
    def __init__(self, client: _ClientStub) -> None:
        self.daemon_endpoint = "daemon"
        self.session_id = "sess"
        self.closed = False
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

    def resolve_key_mapping_cached(self, *, key: str) -> tuple[str | None, str | None]:
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
    def __init__(
        self,
        payload: MaterializationPayload,
        *,
        get_into_result: GetIntoResult | None = None,
    ) -> None:
        self.payload = payload
        self.get_into_result = get_into_result
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
        options=None,
        tensor_names: Sequence[str] | None = None,
        view_spec=None,
        view_data_hash=None,
        view_index_hint=None,
        replica_uuid=None,
        ctx=None,
    ) -> GetIntoResult | None:
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
        return self.get_into_result


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

    projection = cast(Any, result)
    tensor_owner = cast(Any, result["bar"])._tensorcast_realization_owner
    assert tensor_owner is projection._tensorcast_realization_owner

    tensor_owner.close()
    projection.close()
    assert pipeline.released == ["replica-1"]
    assert runtime._client.unloaded == [("replica-1", "")]


def test_tensor_dict_with_diagnostics_release_is_idempotent():
    canonical_bytes, payload = _build_payload({"foo": torch.ones(1)})
    runtime = _RuntimeStub(_ClientStub(canonical_bytes))
    pipeline = _PipelineStub(payload)
    store = _StoreStub(runtime, pipeline)
    artifact = Artifact(
        store_ref=_store_ref(store),
        artifact_id="aid",
        canonical_index_bytes=canonical_bytes,
        generation=1,
    )

    result = artifact.tensor_dict_with_diagnostics(device="cpu")

    assert set(result.tensors) == {"foo"}
    assert runtime._client.unloaded == []
    assert pipeline.released == []

    result.release()
    result.release()
    cast(Any, result.tensors).close()
    assert pipeline.released == ["replica-1"]
    assert runtime._client.unloaded == [("replica-1", "")]


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
    assert tuple(
        entry.name for entry in derived._view_metadata.selected_index.entries
    ) == ("bar",)
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

    selection = derived._resolve_realization_selection().proto

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


def test_realize_emits_report_shaped_profile_event(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("TENSORCAST_PROFILE_DIR", str(tmp_path))
    canonical_bytes, payload = _build_payload({"foo": torch.ones(1)})
    runtime = _RuntimeStub(_ClientStub(canonical_bytes))
    pipeline = _PipelineStub(payload)
    store = _StoreStub(runtime, pipeline)
    artifact = Artifact(
        store_ref=_store_ref(store),
        artifact_id="aid",
        canonical_index_bytes=canonical_bytes,
    )

    _ = artifact.tensor_dict(device="cpu")

    records = [
        json.loads(line)
        for path in tmp_path.glob("tensorcast_pid*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    stages = [str(record.get("stage") or "") for record in records]
    assert "artifact.tensor_dict_with_diagnostics" not in stages
    events = [record for record in records if record.get("stage") == "artifact.realize"]
    assert len(events) == 1
    event = events[0]
    assert event["target_kind"] == "tensor_dict"
    assert event["artifact_id"] == "aid"
    assert event["operation_backend"] == "daemon_materialization"
    assert event["envelope_backing_kind"] == "daemon_temporary_replica"
    assert event["envelope_export_kind"] == "cpu_memfd"
    assert event["target_plan_kind"] == "tensor_dict"
    assert event["strategy_fallback_policy"] == "fail_closed"
    assert event["lifecycle_capability"] == "tensor_dict"
    assert event["publishable"] is False
    assert event["source_selection_digest"]


def test_caller_tensors_realization_reports_temporary_copy_costs() -> None:
    canonical_bytes, payload = _build_payload({"foo": torch.ones(1)})
    runtime = _RuntimeStub(_ClientStub(canonical_bytes))
    pipeline = _PipelineStub(
        payload,
        get_into_result=GetIntoResult(
            used_region_backed=False,
            source="disk",
            source_code=int(store_daemon_pb2.MATERIALIZATION_SOURCE_DISK),
            replica_uuid="replica-1",
            total_bytes=4,
            fallback_reason_buckets={"layout_mismatch": 1},
        ),
    )
    store = _StoreStub(runtime, pipeline)
    artifact = Artifact(
        store_ref=_store_ref(store),
        artifact_id="aid",
        canonical_index_bytes=canonical_bytes,
    )
    target = {"foo": torch.zeros(1)}
    options = GetArtifactOptions(
        source=RetrievalPolicy(
            preference=RetrievalPreference.PREFER_DISK,
            allow_p2p=False,
            allow_disk=True,
        ),
        wait_for_shared_disk_ms=250,
        verify_checksums=False,
        enable_verification=False,
        region_backed_mode=RegionBackedMode.DISABLE,
        export_policy="force",
        transport_hold_ms=77,
    )

    handle = artifact.realize(
        ArtifactRealizationSpec.caller_tensors(
            target=target,
            device="cpu",
            options=options,
        )
    )

    handle.complete()
    assert torch.equal(target["foo"], torch.ones(1))
    assert pipeline.get_into_calls
    report = handle.report
    assert report.target_kind == "caller_tensors"
    assert report.operation_backend == "daemon_materialize_into_target"
    assert report.source == "disk"
    assert report.target_plan is not None
    assert report.target_plan.target_layout_digest
    assert report.envelope.direct_write_bytes == 0
    assert report.envelope.copy_bytes == 4
    assert report.envelope.copy_count == 1
    assert report.envelope.temporary_replica_bytes == 4
    assert report.envelope.fallback_reason_buckets == {"layout_mismatch": 1}
    assert report.envelope.release_policy == (
        "unregister_target_region",
        "unload_temporary_replica_on_fallback",
    )
    assert report.strategy_plan is not None
    assert report.strategy_plan.fallback_policy == "generic_fallback"
    assert report.strategy_plan.source_policy == {
        "preference": "prefer_disk",
        "allow_p2p": False,
        "allow_disk": True,
        "wait_for_shared_disk_ms": 250,
        "verify_checksums": False,
        "enable_verification": False,
        "region_backed_mode": "disable",
        "export_policy": "force",
        "wait_for_completion": True,
        "transport_hold_ms": 77,
        "lease_mode": None,
        "topology_collective_policy": None,
        "source_locality": "auto",
        "source_sharing_domain": None,
    }
    assert report.representation_admission is not None
    assert report.representation_admission.representation_contract == "caller_tensors"
    assert report.lifecycle_plan is not None
    assert report.lifecycle_plan.capability == "caller_tensors"
    assert report.lifecycle_plan.release_policy == report.envelope.release_policy


def test_caller_tensors_realization_reports_cuda_direct_write_costs() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA direct-write cost accounting requires CUDA")
    canonical_bytes, payload = _build_payload({"foo": torch.ones(1)})
    runtime = _RuntimeStub(_ClientStub(canonical_bytes))
    pipeline = _PipelineStub(
        payload,
        get_into_result=GetIntoResult(used_region_backed=True),
    )
    store = _StoreStub(runtime, pipeline)
    artifact = Artifact(
        store_ref=_store_ref(store),
        artifact_id="aid",
        canonical_index_bytes=canonical_bytes,
    )
    target = {"foo": torch.zeros(1, device="cuda:0")}

    handle = artifact.realize(
        ArtifactRealizationSpec.caller_tensors(
            target=target,
            device="cuda:0",
        )
    )

    handle.complete()
    assert torch.equal(target["foo"].cpu(), torch.ones(1))
    report = handle.report
    assert report.envelope.export_kind == "registered_vram_region_direct_write"
    assert report.envelope.direct_write_bytes == 4
    assert report.envelope.copy_bytes == 0
    assert report.envelope.copy_count == 0
    assert report.envelope.temporary_replica_bytes == 0
    assert report.envelope.release_policy == ("unregister_target_region",)


def test_caller_tensors_realization_reports_cuda_fallback_copy_costs() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA fallback copy cost accounting requires CUDA")
    canonical_bytes, payload = _build_payload({"foo": torch.ones(1)})
    runtime = _RuntimeStub(_ClientStub(canonical_bytes))
    pipeline = _PipelineStub(
        payload,
        get_into_result=GetIntoResult(
            used_region_backed=False,
            source="disk",
            source_code=int(store_daemon_pb2.MATERIALIZATION_SOURCE_DISK),
            replica_uuid="replica-1",
            total_bytes=4,
            fallback_reason_buckets={"layout_mismatch": 1},
        ),
    )
    store = _StoreStub(runtime, pipeline)
    artifact = Artifact(
        store_ref=_store_ref(store),
        artifact_id="aid",
        canonical_index_bytes=canonical_bytes,
    )
    target = {"foo": torch.zeros(1, device="cuda:0")}

    handle = artifact.realize(
        ArtifactRealizationSpec.caller_tensors(
            target=target,
            device="cuda:0",
        )
    )

    handle.complete()
    assert torch.equal(target["foo"].cpu(), torch.ones(1))
    report = handle.report
    assert report.source == "disk"
    assert report.envelope.export_kind == "temporary_copy"
    assert report.envelope.direct_write_bytes == 0
    assert report.envelope.copy_bytes == 4
    assert report.envelope.copy_count == 1
    assert report.envelope.temporary_replica_bytes == 4
    assert report.envelope.fallback_reason_buckets == {"layout_mismatch": 1}
    assert report.envelope.release_policy == (
        "unregister_target_region",
        "unload_temporary_replica_on_fallback",
    )
    assert report.strategy_plan is not None
    assert report.strategy_plan.fallback_policy == "generic_fallback"


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
    fake_binding = type(
        "_BindingStub",
        (),
        {
            "binding_id": "binding",
            "binding_layout_id": "bl1:test",
            "current_value": None,
            "staged_value": None,
            "last_materialization_diagnostics": {
                "source": "disk",
                "total_bytes": 4,
                "retry_reason_buckets": {},
            },
            "last_execution_diagnostics": None,
            "last_source_bound_plan_diagnostics": None,
        },
    )()

    def _fake_bind_owned(self, **kwargs):
        del self
        captured.update(kwargs)
        return fake_binding

    monkeypatch.setattr(Artifact, "_bind_owned", _fake_bind_owned)

    manifest = RuntimeArtifactManifest(
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
        runtime_artifact_policy=manifest,
    )

    assert result is fake_binding
    assert captured["device"] == torch.device("cuda:0")
    assert captured["runtime_artifact_policy"] == RuntimeArtifactPolicy(
        require_manifest=True,
        serving_manifest_ref="tensor:__alt_manifest__.json",
        expected_representation_contract_hash="bafkrepresentation",
        expected_serving_build_digest="bafkbuilddigest",
    )


def test_realization_spec_uses_only_runtime_artifact_policy_name() -> None:
    neutral_policy = RuntimeArtifactPolicy(
        serving_manifest_ref="tensor:manifest-a.json",
    )

    with pytest.raises(TypeError, match="serving_runtime_policy"):
        ArtifactRealizationSpec.binding(
            device="cuda:0",
            runtime_artifact_policy=neutral_policy,
            serving_runtime_policy=neutral_policy,
        )

    spec = ArtifactRealizationSpec.binding(
        device="cuda:0",
        runtime_artifact_policy=neutral_policy,
    )
    assert spec.runtime_artifact_policy is neutral_policy
    assert not hasattr(spec, "serving_runtime_policy")


def test_tensor_dict_and_adopted_binding_share_source_selection_with_separate_target_digests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_bytes, payload = _build_payload({"foo": torch.ones(2)})
    runtime = _RuntimeStub(_ClientStub(canonical_bytes))
    pipeline = _PipelineStub(payload)
    store = _StoreStub(runtime, pipeline)
    artifact = Artifact(
        store_ref=_store_ref(store),
        artifact_id="aid",
        canonical_index_bytes=canonical_bytes,
        generation=7,
    )
    target = {"dst": torch.zeros(2)}
    copy_plan = (
        CopyPlanEntry(
            ckpt_name="foo",
            ckpt_range=None,
            dst_name="dst",
            dst_range=Range(dim=0, start=0, end=2),
        ),
    )
    captured: dict[str, object] = {}
    binding_value = type(
        "_BindingValueStub",
        (),
        {
            "binding_id": "binding",
            "binding_layout_id": "bl1:test",
            "binding_value_id": "value-1",
            "seal_generation": 1,
            "source_artifact_id": "aid",
            "is_artifact_backed": True,
            "verification_state": 0,
            "is_published": False,
        },
    )()
    fake_binding = type(
        "_AdoptedBindingStub",
        (),
        {
            "binding_id": "binding",
            "binding_layout_id": "bl1:test",
            "current_value": binding_value,
            "staged_value": None,
            "last_materialization_diagnostics": {
                "source": "disk",
                "total_bytes": 8,
                "retry_reason_buckets": {},
            },
            "last_execution_diagnostics": None,
            "last_source_bound_plan_diagnostics": None,
        },
    )()

    def _fake_execute_bind_into(self, target_tensors, **kwargs):
        del self
        captured["target_tensors"] = target_tensors
        captured.update(kwargs)
        return fake_binding

    monkeypatch.setattr(Artifact, "_execute_bind_into", _fake_execute_bind_into)

    tensor_handle = artifact.realize(ArtifactRealizationSpec.tensor_dict(device="cpu"))
    adopted_handle = artifact.realize(
        ArtifactRealizationSpec.adopted_binding(
            target=target,
            mapping=copy_plan,
            packing="byte_space",
        )
    )

    tensor_report = tensor_handle.report
    adopted_report = adopted_handle.report
    assert adopted_handle.binding() is fake_binding
    assert captured["mapping"] == copy_plan
    assert (
        adopted_report.source_selection_digest == tensor_report.source_selection_digest
    )
    assert adopted_report.target_layout_digest
    assert adopted_report.copy_plan_digest
    assert adopted_report.target_layout_digest != adopted_report.source_selection_digest
    assert adopted_report.copy_plan_digest != adopted_report.target_layout_digest
    assert str(adopted_report.copy_plan_digest).startswith("mapped:v1:")
    assert adopted_report.target_plan is not None
    assert adopted_report.target_plan.target_layout_digest == (
        adopted_report.target_layout_digest
    )
    assert (
        adopted_report.target_plan.copy_plan_digest == adopted_report.copy_plan_digest
    )

    tensor_handle.close()
    adopted_handle.close()


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


def test_subset_clone_handles_multiple_identifiers():
    canonical_bytes, payload = _build_payload({"foo": torch.ones(1)})
    runtime = _RuntimeStub(_ClientStub(canonical_bytes))
    runtime.cache_key_mapping("mapped", artifact_id="aid")
    pipeline = _PipelineStub(payload)
    store = _StoreStub(runtime, pipeline)
    artifact = Artifact(store_ref=_store_ref(store), key="mapped")

    assert artifact.artifact_id == "aid"
    clone = artifact.subset(["foo"])

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


def test_artifact_ref_accepts_msa1_prefix():
    canonical_bytes, _payload = _build_payload({"foo": torch.ones(1)})
    runtime = _RuntimeStub(_ClientStub(canonical_bytes))
    store = Store("daemon", runtime=cast(Any, runtime))

    artifact = store.artifact("msa1:test-session~policy~safetensors~deadbeef")

    assert artifact.artifact_id == "msa1:test-session~policy~safetensors~deadbeef"
    assert artifact.key is None


def test_from_disk_resolves_once_and_caches_generation():
    canonical_bytes, payload = _build_payload({"foo": torch.ones(1)})
    client = _ClientStub(
        canonical_bytes,
        disk_generation=11,
        mounted_generation=11,
        disk_artifact_id="mi2:idx:data",
        mounted_artifact_id="msa1:test-session~policy~safetensors~deadbeef",
    )
    runtime = _RuntimeStub(client)
    store = Store("daemon", runtime=cast(Any, runtime))

    artifact = store.from_disk("/tmp/artifact")
    desc = artifact.describe()
    repeat = artifact.describe()

    assert desc.artifact_id == "msa1:test-session~policy~safetensors~deadbeef"
    assert desc.generation == 11
    assert repeat.generation == 11
    assert client.resolve_public_disk_calls == [("/tmp/artifact", True)]
    assert client.import_calls == []
    assert client.get_index_calls == 0
    cached = runtime.get_artifact_index_cached(
        "msa1:test-session~policy~safetensors~deadbeef"
    )
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
    store = Store("daemon", runtime=cast(Any, runtime))

    artifact = store.from_disk("/tmp/artifact", show_progress=True)

    assert artifact.artifact_id == "mi2:idx:data"
    assert client.import_calls == [("/tmp/artifact", True)]
    assert client.resolve_public_disk_calls == []


def test_from_disk_default_stays_fast_path_even_when_stderr_is_tty(monkeypatch):
    canonical_bytes, _payload = _build_payload({"foo": torch.ones(1)})
    client = _ClientStub(
        canonical_bytes,
        disk_generation=5,
        disk_artifact_id="mi2:idx:data",
        mounted_artifact_id="msa1:test-session~policy~safetensors~deadbeef",
    )
    runtime = _RuntimeStub(client)
    store = Store("daemon", runtime=cast(Any, runtime))

    monkeypatch.setattr("sys.stderr.isatty", lambda: True)

    artifact = store.from_disk("/tmp/artifact")

    assert artifact.artifact_id == "msa1:test-session~policy~safetensors~deadbeef"
    assert client.resolve_public_disk_calls == [("/tmp/artifact", True)]
    assert client.import_calls == []


def test_from_disk_retries_daemon_startup_in_progress(monkeypatch):
    canonical_bytes, _payload = _build_payload({"foo": torch.ones(1)})
    client = _ClientStub(
        canonical_bytes,
        disk_generation=7,
        disk_artifact_id="mi2:idx:data",
        mounted_artifact_id="msa1:test-session~policy~safetensors~retry",
        startup_in_progress_failures=2,
    )
    runtime = _RuntimeStub(client)
    store = Store("daemon", runtime=cast(Any, runtime))
    sleeps: list[float] = []

    monkeypatch.setattr(time, "sleep", lambda seconds: sleeps.append(seconds))

    artifact = store.from_disk("/tmp/artifact")

    assert artifact.artifact_id == "msa1:test-session~policy~safetensors~retry"
    assert len(sleeps) == 2
    assert client.resolve_public_disk_calls == [
        ("/tmp/artifact", True),
        ("/tmp/artifact", True),
        ("/tmp/artifact", True),
    ]
    assert client.import_calls == []


def test_store_promote_mounted_source_returns_mi2_artifact():
    canonical_bytes, _payload = _build_payload({"foo": torch.ones(1)})
    client = _ClientStub(
        canonical_bytes,
        disk_generation=13,
        disk_artifact_id="mi2:idx:promoted",
        mounted_artifact_id="msa1:test-session~policy~safetensors~source",
    )
    runtime = _RuntimeStub(client)
    store = Store("daemon", runtime=cast(Any, runtime))

    artifact = store.promote_mounted_source(
        "msa1:test-session~policy~safetensors~source"
    )

    assert artifact.artifact_id == "mi2:idx:promoted"
    assert client.promote_mounted_source_calls == [
        ("msa1:test-session~policy~safetensors~source", True, None)
    ]


def test_mounted_source_realize_promotes_and_reports_identity():
    canonical_bytes, _payload = _build_payload({"foo": torch.ones(1)})
    client = _ClientStub(
        canonical_bytes,
        disk_generation=17,
        disk_artifact_id="mi2:idx:promoted",
        mounted_artifact_id="msa1:test-session~policy~safetensors~source",
    )
    runtime = _RuntimeStub(client)
    store = Store("daemon", runtime=cast(Any, runtime))
    source = store.artifact("msa1:test-session~policy~safetensors~source")

    handle = source.realize(
        ArtifactRealizationSpec.mounted_source(
            verify_checksums=False,
            timeout_s=60.0,
        )
    )
    promoted = handle.promote()

    assert promoted.artifact_id == "mi2:idx:promoted"
    assert client.promote_mounted_source_calls == [
        ("msa1:test-session~policy~safetensors~source", False, 60.0)
    ]
    assert handle.report.target_kind == "mounted_source"
    assert handle.report.artifact_profile == "mounted_source"
    assert handle.report.authority_scope == "daemon_local_mounted_source"
    assert handle.report.operation_backend == "daemon_mounted_source_promotion"
    assert handle.report.mounted_source is not None
    assert (
        handle.report.mounted_source.source_artifact_id
        == "msa1:test-session~policy~safetensors~source"
    )
    assert handle.report.mounted_source.promoted_artifact_id == "mi2:idx:promoted"
    assert handle.report.mounted_source.promoted_artifact_profile == "durable_artifact"
    assert handle.report.mounted_source.generation == 17
    assert promoted.key is None


def test_mounted_source_realize_rejects_non_msa1_subject():
    canonical_bytes, _payload = _build_payload({"foo": torch.ones(1)})
    client = _ClientStub(
        canonical_bytes,
        disk_generation=17,
        disk_artifact_id="mi2:idx:promoted",
        mounted_artifact_id="msa1:test-session~policy~safetensors~source",
    )
    runtime = _RuntimeStub(client)
    store = Store("daemon", runtime=cast(Any, runtime))
    source = store.artifact("mi2:idx:ordinary")

    with pytest.raises(ArtifactError) as excinfo:
        source.realize(ArtifactRealizationSpec.mounted_source())

    assert excinfo.value.status_code == "INVALID_ARGUMENT"
    assert client.promote_mounted_source_calls == []


def test_mounted_source_realize_rejects_key_mapping_activation():
    canonical_bytes, payload = _build_payload({"foo": torch.ones(1)})
    client = _ClientStub(
        canonical_bytes,
        disk_generation=17,
        disk_artifact_id="mi2:idx:promoted",
        mounted_artifact_id="msa1:test-session~policy~safetensors~source",
    )
    runtime = _RuntimeStub(client)
    runtime.cache_key_mapping(
        "mounted-source-key",
        artifact_id="msa1:test-session~policy~safetensors~source",
    )
    store = _StoreStub(runtime, _PipelineStub(payload))
    source = Artifact(store_ref=_store_ref(store), key="mounted-source-key")

    with pytest.raises(ArtifactError) as excinfo:
        source.realize(ArtifactRealizationSpec.mounted_source())

    assert excinfo.value.status_code == "INVALID_ARGUMENT"
    assert "explicit msa1 artifact id" in str(excinfo.value)
    assert client.promote_mounted_source_calls == []


def test_store_promote_mounted_source_accepts_artifact_handle():
    canonical_bytes, _payload = _build_payload({"foo": torch.ones(1)})
    client = _ClientStub(
        canonical_bytes,
        disk_generation=14,
        disk_artifact_id="mi2:idx:promoted",
        mounted_artifact_id="msa1:test-session~policy~safetensors~source",
    )
    runtime = _RuntimeStub(client)
    store = Store("daemon", runtime=cast(Any, runtime))
    source = store.from_disk("/tmp/artifact")

    promoted = store.promote_mounted_source(source, verify_checksums=False)

    assert promoted.artifact_id == "mi2:idx:promoted"
    assert client.promote_mounted_source_calls == [
        ("msa1:test-session~policy~safetensors~source", False, None)
    ]


def test_store_promote_mounted_source_passes_timeout_override():
    canonical_bytes, _payload = _build_payload({"foo": torch.ones(1)})
    client = _ClientStub(
        canonical_bytes,
        disk_generation=15,
        disk_artifact_id="mi2:idx:promoted",
        mounted_artifact_id="msa1:test-session~policy~safetensors~source",
    )
    runtime = _RuntimeStub(client)
    store = Store("daemon", runtime=cast(Any, runtime))

    promoted = store.promote_mounted_source(
        "msa1:test-session~policy~safetensors~source",
        timeout_s=180.0,
    )

    assert promoted.artifact_id == "mi2:idx:promoted"
    assert client.promote_mounted_source_calls == [
        ("msa1:test-session~policy~safetensors~source", True, 180.0)
    ]


def test_import_from_disk_uses_import_stream_and_publishes_key():
    canonical_bytes, _payload = _build_payload({"foo": torch.ones(1)})
    client = _ClientStub(
        canonical_bytes,
        disk_generation=9,
        disk_artifact_id="mi2:idx:data",
    )
    runtime = _RuntimeStub(client)
    store = Store("daemon", runtime=cast(Any, runtime))

    artifact = store.import_from_disk("/tmp/artifact", key="model:key")

    assert artifact.artifact_id == "mi2:idx:data"
    assert artifact.key == "model:key"
    assert client.import_calls == [("/tmp/artifact", True)]
    assert client.resolve_public_disk_calls == []
    assert client.publish_calls == [("model:key", "mi2:idx:data")]


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
