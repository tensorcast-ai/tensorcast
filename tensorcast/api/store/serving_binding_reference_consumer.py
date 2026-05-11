#  Copyright (c) 2026, TensorCast Team.

"""Reference serving-binding consumer helpers for examples and E2E tests."""

from __future__ import annotations

import hashlib
import json
import os

from google.protobuf.any_pb2 import Any
from pydantic import BaseModel, ConfigDict, model_validator

from tensorcast.api.store.serving_binding_spec_cache import (
    ServingBindingSpecCacheRecord,
    read_matching_resolved_spec_cache_entry,
    write_resolved_spec_cache_entry,
)
from tensorcast.daemon_ctl import DaemonCtl
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.proto.operation.v1 import operation_pb2
from tensorcast.types import (
    BindingValueRef,
    BlobRef,
    PrefetchedServingBinding,
    PrefetchRetentionPolicy,
    ServingBindingMemberRef,
    ServingBindingResolvedLayout,
    ServingBindingResolvedSpecCacheEntry,
    ServingBindingSourceRef,
    ServingBindingSourceReuseDecision,
    ServingBindingTarget,
    ServingTopologyRef,
)

REFERENCE_RUNTIME = "tensorcast-reference"
_TARGET_LAYOUT_BLOB = "target_layout"
_TARGET_INDEX_BLOB = "target_index"


class ReferenceServingTensorSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = "alpha"
    size_bytes: int = 4
    dtype: str = "torch.uint8"
    shape: tuple[int, ...] = (4,)
    stride: tuple[int, ...] = (1,)

    @model_validator(mode="after")
    def _validate_spec(self) -> "ReferenceServingTensorSpec":
        if not self.name:
            raise ValueError("name must not be empty")
        if int(self.size_bytes) <= 0:
            raise ValueError("size_bytes must be positive")
        if not self.dtype:
            raise ValueError("dtype must not be empty")
        if not self.shape:
            raise ValueError("shape must not be empty")
        if not self.stride:
            raise ValueError("stride must not be empty")
        return self


class ReferenceServingResolvedSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    cache_entry: ServingBindingResolvedSpecCacheEntry
    target: ServingBindingTarget
    blobs: dict[str, bytes]


class ReferenceServingAcquireResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    binding_value_ref: BindingValueRef
    lease_token: bytes
    has_cuda_ipc_handle: bool
    has_cpu_memfd_handle: bool


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def build_reference_tensor_index_bytes(
    tensor: ReferenceServingTensorSpec,
) -> bytes:
    payload = {
        tensor.name: [
            0,
            int(tensor.size_bytes),
            list(tensor.shape),
            list(tensor.stride),
            tensor.dtype,
            0,
        ]
    }
    return _canonical_json_bytes(payload)


def build_reference_target_layout(
    tensor: ReferenceServingTensorSpec,
    *,
    device_id: int = 0,
) -> store_daemon_pb2.TargetLayout:
    layout = store_daemon_pb2.TargetLayout(
        layout_kind=store_daemon_pb2.TargetLayout.LAYOUT_KIND_TENSOR_TABLE,
        index_kind=store_daemon_pb2.TargetLayout.INDEX_KIND_CANONICAL_UNSPECIFIED,
        tensor_spec_kind=store_daemon_pb2.TargetLayout.TENSOR_SPEC_KIND_OFFSETS,
    )
    layout.logical_layout_hash = _sha256_bytes(
        build_reference_tensor_index_bytes(tensor)
    ).encode("ascii")
    storage = layout.storages.add()
    storage.storage_id = "storage-0"
    storage.device_id = int(device_id)
    storage.storage_length = int(tensor.size_bytes)
    offset = layout.offsets.add()
    offset.name = tensor.name
    offset.storage_id = storage.storage_id
    offset.storage_offset = 0
    offset.logical_length = int(tensor.size_bytes)
    return layout


def _blob_ref(path: str, payload: bytes) -> BlobRef:
    return BlobRef(
        path=path,
        sha256=_sha256_bytes(payload),
        size_bytes=len(payload),
    )


def build_reference_resolved_spec(
    *,
    source_artifact_id: str,
    artifact_selection_digest: str,
    device_uuid: str,
    tensor: ReferenceServingTensorSpec | None = None,
    runtime: str = REFERENCE_RUNTIME,
    topology: ServingTopologyRef | None = None,
    member: ServingBindingMemberRef | None = None,
    source_schema_hash: str = "reference-source-schema",
    model_config_digest: str = "reference-model-config",
    serving_build_digest: str = "reference-serving-build",
    representation_contract_hash: str = "reference-representation-contract",
    binding_layout_id: str = "reference-layout-0",
) -> ReferenceServingResolvedSpec:
    if not source_artifact_id:
        raise ValueError("source_artifact_id is required")
    if not artifact_selection_digest:
        raise ValueError("artifact_selection_digest is required")
    if not device_uuid:
        raise ValueError("device_uuid is required")
    resolved_tensor = tensor or ReferenceServingTensorSpec()
    resolved_topology = topology or ServingTopologyRef(
        schema_topology_digest="reference-topology"
    )
    resolved_member = member or ServingBindingMemberRef(
        member_id="member-0",
        member_index=0,
        member_count=1,
        group_id="reference-group",
    )
    source = ServingBindingSourceRef(
        source_kind="checkpoint_artifact",
        artifact_selection_digest=artifact_selection_digest,
        source_artifact_ref=source_artifact_id,
        source_schema_hash=source_schema_hash,
    )
    source_reuse = ServingBindingSourceReuseDecision(
        mode="checkpoint_to_serving",
        representation_contract_hash=representation_contract_hash,
    )
    target_layout = build_reference_target_layout(resolved_tensor)
    target_layout_bytes = target_layout.SerializeToString(deterministic=True)
    target_index_bytes = build_reference_tensor_index_bytes(resolved_tensor)
    layout_hash = _sha256_bytes(target_layout_bytes)
    tensor_schema_hash = _sha256_bytes(target_index_bytes)
    draft_entry = ServingBindingResolvedSpecCacheEntry(
        schema_version=1,
        cache_key_digest="placeholder",
        spec_digest="placeholder",
        runtime=runtime,
        source=source,
        source_reuse=source_reuse,
        topology=resolved_topology,
        member=resolved_member,
        source_schema_hash=source_schema_hash,
        model_config_digest=model_config_digest,
        serving_build_digest=serving_build_digest,
        binding_layout_id=binding_layout_id,
        target_layout_hash=layout_hash,
        tensor_schema_hash=tensor_schema_hash,
        blob_refs={
            _TARGET_LAYOUT_BLOB: _blob_ref("target_layout.bin", target_layout_bytes),
            _TARGET_INDEX_BLOB: _blob_ref("target_index.json", target_index_bytes),
        },
    )
    entry_with_key = draft_entry.model_copy(
        update={"cache_key_digest": draft_entry.computed_cache_key_digest()}
    )
    entry = entry_with_key.model_copy(
        update={"spec_digest": entry_with_key.computed_spec_digest()}
    )
    resolved_layout = ServingBindingResolvedLayout(
        binding_layout_id=binding_layout_id,
        source=source,
        source_reuse=source_reuse,
        topology=resolved_topology,
        member=resolved_member,
        target_layout=target_layout_bytes,
        target_index_bytes=target_index_bytes,
        target_layout_hash=layout_hash,
        tensor_schema_hash=tensor_schema_hash,
        spec_digest=entry.spec_digest,
        source_schema_hash=source_schema_hash,
    )
    target = ServingBindingTarget(
        runtime=runtime,
        device="cuda:0",
        device_uuid=device_uuid,
        source=source,
        topology=resolved_topology,
        member=resolved_member,
        model_config_digest=model_config_digest,
        serving_build_digest=serving_build_digest,
        resolved_layout=resolved_layout,
    )
    return ReferenceServingResolvedSpec(
        cache_entry=entry,
        target=target,
        blobs={
            _TARGET_LAYOUT_BLOB: target_layout_bytes,
            _TARGET_INDEX_BLOB: target_index_bytes,
        },
    )


def write_reference_resolved_spec_cache_entry(
    cache_root: str | os.PathLike[str],
    *,
    resolved_spec: ReferenceServingResolvedSpec,
) -> ServingBindingSpecCacheRecord:
    write_resolved_spec_cache_entry(
        cache_root,
        entry=resolved_spec.cache_entry,
        blobs=resolved_spec.blobs,
    )
    return read_matching_resolved_spec_cache_entry(
        cache_root,
        expected_entry=resolved_spec.cache_entry,
    )


def target_from_reference_cache_record(
    record: ServingBindingSpecCacheRecord,
    *,
    device_uuid: str,
    device: str = "cuda:0",
) -> ServingBindingTarget:
    target_layout = record.blobs.get(_TARGET_LAYOUT_BLOB)
    target_index = record.blobs.get(_TARGET_INDEX_BLOB)
    if target_layout is None:
        raise ValueError("reference cache record is missing target_layout blob")
    if target_index is None:
        raise ValueError("reference cache record is missing target_index blob")
    entry = record.entry
    resolved_layout = ServingBindingResolvedLayout(
        binding_layout_id=entry.binding_layout_id,
        source=entry.source,
        source_reuse=entry.source_reuse,
        topology=entry.topology,
        member=entry.member,
        target_layout=target_layout,
        target_index_bytes=target_index,
        target_layout_hash=entry.target_layout_hash,
        tensor_schema_hash=entry.tensor_schema_hash,
        spec_digest=entry.spec_digest,
        source_schema_hash=entry.source_schema_hash,
    )
    return ServingBindingTarget(
        runtime=entry.runtime,
        device=device,
        device_uuid=device_uuid,
        source=entry.source,
        topology=entry.topology,
        member=entry.member,
        model_config_digest=entry.model_config_digest,
        load_config_digest=entry.load_config_digest,
        serving_build_digest=entry.serving_build_digest,
        resolved_layout=resolved_layout,
    )


def unpack_prefetched_serving_binding(
    result_any: Any,
) -> PrefetchedServingBinding:
    proto = operation_pb2.PrefetchServingBindingResult()
    if not result_any.Unpack(proto):
        raise ValueError("operation result is not PrefetchServingBindingResult")
    return PrefetchedServingBinding.from_proto(proto)


def prefetch_reference_binding(
    client: DaemonCtl,
    *,
    source_artifact_id: str,
    target: ServingBindingTarget,
    retention_policy: PrefetchRetentionPolicy | None = None,
    operation_id: str | None = None,
    timeout_s: float = 30.0,
) -> PrefetchedServingBinding:
    selection = common_pb2.ArtifactSelection(artifact_id=source_artifact_id)
    response = client.prefetch_serving_binding(
        source_selection=selection,
        target=target,
        requested_readiness="serving_local_ready",
        retention_policy=retention_policy,
        operation_id=operation_id,
        timeout_s=timeout_s,
    )
    if response.status.state != operation_pb2.OPERATION_STATE_SUCCESS:
        message = response.status.message or "PrefetchServingBinding did not succeed"
        if response.status.HasField("error"):
            message = response.status.error.message or message
        raise RuntimeError(message)
    return unpack_prefetched_serving_binding(response.status.result)


def acquire_reference_binding(
    client: DaemonCtl,
    *,
    prefetched: PrefetchedServingBinding,
    target: ServingBindingTarget,
    caller_pid: int | None = None,
    timeout_s: float = 30.0,
) -> ReferenceServingAcquireResult:
    response = acquire_reference_binding_response(
        client,
        prefetched=prefetched,
        target=target,
        caller_pid=caller_pid,
        timeout_s=timeout_s,
    )
    binding_value_ref = prefetched.binding_value_ref
    if response.HasField("current_value"):
        binding_value_ref = BindingValueRef(
            binding_id=str(response.current_value.binding_id),
            binding_layout_id=str(response.current_value.binding_layout_id),
            binding_value_id=str(response.current_value.binding_value_id),
            seal_generation=int(response.current_value.seal_generation),
        )
    return ReferenceServingAcquireResult(
        binding_value_ref=binding_value_ref,
        lease_token=bytes(response.mem_handle.lease_token),
        has_cuda_ipc_handle=response.mem_handle.HasField("cuda_ipc_handle"),
        has_cpu_memfd_handle=response.mem_handle.HasField("cpu_memfd"),
    )


def acquire_reference_binding_response(
    client: DaemonCtl,
    *,
    prefetched: PrefetchedServingBinding,
    target: ServingBindingTarget,
    caller_pid: int | None = None,
    timeout_s: float = 30.0,
) -> store_daemon_pb2.AcquireBindingValueResponse:
    return client.acquire_binding_value(
        binding_value_ref=prefetched.binding_value_ref,
        reservation_capability=prefetched.reservation_capability,
        expected_device_uuid=prefetched.device_uuid,
        expected_target_layout_hash=target.resolved_layout.target_layout_hash,
        expected_tensor_schema_hash=target.resolved_layout.tensor_schema_hash,
        expected_serving_build_digest=target.serving_build_digest,
        expected_daemon_id=prefetched.daemon_id,
        expected_daemon_session_id=prefetched.daemon_session_id,
        expected_member=prefetched.member,
        local_serving_ref=prefetched.local_serving_ref,
        caller_pid=caller_pid or os.getpid(),
        timeout_s=timeout_s,
    )


def release_reference_acquire(
    client: DaemonCtl,
    *,
    acquire_result: ReferenceServingAcquireResult,
    timeout_s: float = 5.0,
) -> None:
    if acquire_result.lease_token:
        client.release_placement_lease(
            lease_token=acquire_result.lease_token,
            timeout_s=timeout_s,
        )


__all__ = [
    "REFERENCE_RUNTIME",
    "ReferenceServingAcquireResult",
    "ReferenceServingResolvedSpec",
    "ReferenceServingTensorSpec",
    "acquire_reference_binding",
    "acquire_reference_binding_response",
    "build_reference_resolved_spec",
    "build_reference_target_layout",
    "build_reference_tensor_index_bytes",
    "prefetch_reference_binding",
    "release_reference_acquire",
    "target_from_reference_cache_record",
    "unpack_prefetched_serving_binding",
    "write_reference_resolved_spec_cache_entry",
]
