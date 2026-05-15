#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

from google.protobuf.any_pb2 import Any

from tensorcast.api.context import GroupRealization
from tensorcast.api.store.serving_binding_reference_consumer import (
    REFERENCE_RUNTIME,
    ReferenceServingAcquireResult,
    ReferenceServingTensorSpec,
    acquire_reference_binding,
    build_reference_resolved_spec,
    prefetch_reference_binding,
    release_reference_acquire,
    target_from_reference_cache_record,
    write_reference_resolved_spec_cache_entry,
)
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.proto.operation.v1 import operation_pb2
from tensorcast.types import (
    BindingReservationCapability,
    BindingValueRef,
    BindingValueVerificationState,
    GroupRealizationAcquireRef,
    PrefetchedServingBinding,
)


class _FakeDaemonClient:
    def __init__(self, prefetched: PrefetchedServingBinding) -> None:
        self.prefetched = prefetched
        self.prefetch_calls: list[dict[str, object]] = []
        self.acquire_calls: list[dict[str, object]] = []
        self.released_tokens: list[bytes] = []

    def prefetch_serving_binding(self, **kwargs: object):
        self.prefetch_calls.append(kwargs)
        result_any = Any()
        result_any.Pack(self.prefetched.to_proto())
        return store_daemon_pb2.PrefetchServingBindingResponse(
            status=operation_pb2.OperationStatus(
                state=operation_pb2.OPERATION_STATE_SUCCESS,
                result=result_any,
            )
        )

    def acquire_binding_value(self, **kwargs: object):
        self.acquire_calls.append(kwargs)
        mem_handle = store_daemon_pb2.MemCopyHandle(
            cuda_ipc_handle=b"fake-ipc-handle",
            lease_token=b"lease-token",
        )
        response = store_daemon_pb2.AcquireBindingValueResponse(mem_handle=mem_handle)
        if self.prefetched.staged_value:
            response.acquired_staged_value = True
            target = response.acquired_value
        else:
            target = response.current_value
        target.binding_id = self.prefetched.binding_value_ref.binding_id
        target.binding_layout_id = self.prefetched.binding_value_ref.binding_layout_id
        target.binding_value_id = self.prefetched.binding_value_ref.binding_value_id
        target.seal_generation = self.prefetched.binding_value_ref.seal_generation
        return response

    def release_placement_lease(self, **kwargs: object):
        self.released_tokens.append(bytes(kwargs["lease_token"]))
        return store_daemon_pb2.ReleasePlacementLeaseResponse()


def _prefetched() -> PrefetchedServingBinding:
    member = build_reference_resolved_spec(
        source_artifact_id="mi2:source",
        artifact_selection_digest="selection",
        device_uuid="gpu-0",
    ).target.member
    ref = BindingValueRef(
        binding_id="binding-1",
        binding_layout_id="reference-layout-0",
        binding_value_id="value-1",
        seal_generation=1,
    )
    capability = BindingReservationCapability(
        capability_id="capability-1",
        binding_value_ref=ref,
        daemon_id="daemon-1",
        daemon_session_id="session-1",
        device_uuid="gpu-0",
        member=member,
        reservation_bytes=4,
        scope_digest="scope",
    )
    return PrefetchedServingBinding(
        local_serving_ref="binding-local:binding-1:value-1",
        binding_value_ref=ref,
        daemon_id="daemon-1",
        daemon_session_id="session-1",
        device_uuid="gpu-0",
        member=member,
        reservation_bytes=4,
        reservation_capability=capability,
        readiness="serving_local_ready",
        verification_state=BindingValueVerificationState.LOCAL_ONLY,
    )


def _staged_prefetched() -> PrefetchedServingBinding:
    base = _prefetched()
    ref = BindingValueRef(
        binding_id=base.binding_value_ref.binding_id,
        binding_layout_id=base.binding_value_ref.binding_layout_id,
        binding_value_id="staged-value-1",
        seal_generation=0,
    )
    capability = base.reservation_capability.model_copy(
        update={
            "capability_id": "capability-staged-1",
            "binding_value_ref": ref,
        }
    )
    return base.model_copy(
        update={
            "local_serving_ref": "binding-local:binding-1:staged-value-1",
            "binding_value_ref": ref,
            "reservation_capability": capability,
            "staged_value": True,
            "group_realization_acquire": GroupRealizationAcquireRef(
                transaction_id="txn-1",
                version_set_id="version-set-1",
                part_id="member-0",
                staging_token="stage-1",
            ),
        }
    )


def test_reference_consumer_writes_cache_and_rebuilds_target(tmp_path) -> None:
    resolved = build_reference_resolved_spec(
        source_artifact_id="mi2:source",
        artifact_selection_digest="selection",
        device_uuid="gpu-0",
        tensor=ReferenceServingTensorSpec(name="weight", size_bytes=16, shape=(4,)),
    )

    record = write_reference_resolved_spec_cache_entry(
        tmp_path,
        resolved_spec=resolved,
    )
    target = target_from_reference_cache_record(record, device_uuid="gpu-0")

    assert record.entry.runtime == REFERENCE_RUNTIME
    assert target.runtime == REFERENCE_RUNTIME
    assert target.device_uuid == "gpu-0"
    assert target.resolved_layout.target_layout == resolved.blobs["target_layout"]
    assert target.resolved_layout.target_index_bytes == resolved.blobs["target_index"]


def test_reference_consumer_prefetch_acquire_and_release_lifecycle(tmp_path) -> None:
    resolved = build_reference_resolved_spec(
        source_artifact_id="mi2:source",
        artifact_selection_digest="selection",
        device_uuid="gpu-0",
    )
    record = write_reference_resolved_spec_cache_entry(
        tmp_path,
        resolved_spec=resolved,
    )
    target = target_from_reference_cache_record(record, device_uuid="gpu-0")
    fake_client = _FakeDaemonClient(_prefetched())

    prefetched = prefetch_reference_binding(
        fake_client,
        source_artifact_id="mi2:source",
        target=target,
        operation_id="op-1",
    )
    acquired = acquire_reference_binding(
        fake_client,
        prefetched=prefetched,
        target=target,
        caller_pid=1234,
    )
    release_reference_acquire(fake_client, acquire_result=acquired)

    assert isinstance(acquired, ReferenceServingAcquireResult)
    assert acquired.has_cuda_ipc_handle is True
    assert acquired.lease_token == b"lease-token"
    assert fake_client.released_tokens == [b"lease-token"]
    assert fake_client.prefetch_calls[0]["operation_id"] == "op-1"
    assert fake_client.acquire_calls[0]["caller_pid"] == 1234


def test_reference_consumer_passes_same_selection_group_realization_to_prefetch(
    tmp_path,
) -> None:
    resolved = build_reference_resolved_spec(
        source_artifact_id="mi2:source",
        artifact_selection_digest="selection",
        device_uuid="gpu-0",
    )
    record = write_reference_resolved_spec_cache_entry(
        tmp_path,
        resolved_spec=resolved,
    )
    target = target_from_reference_cache_record(record, device_uuid="gpu-0")
    fake_client = _FakeDaemonClient(_staged_prefetched())
    group = GroupRealization(
        group_kind="serving_prefetch",
        group_id="group-1",
        epoch=1,
        total_parts=1,
        part_id="member-0",
        required_part_ids=("member-0",),
        require_staged_publish=True,
    )

    prefetched = prefetch_reference_binding(
        fake_client,
        source_artifact_id="mi2:source",
        target=target,
        operation_id="op-1",
        group_realization=group,
    )

    assert prefetched.staged_value is True
    assert fake_client.prefetch_calls[0]["group_realization"] == group


def test_reference_consumer_passes_group_realization_acquire_for_staged_value(
    tmp_path,
) -> None:
    resolved = build_reference_resolved_spec(
        source_artifact_id="mi2:source",
        artifact_selection_digest="selection",
        device_uuid="gpu-0",
    )
    record = write_reference_resolved_spec_cache_entry(
        tmp_path,
        resolved_spec=resolved,
    )
    target = target_from_reference_cache_record(record, device_uuid="gpu-0")
    fake_client = _FakeDaemonClient(_staged_prefetched())

    prefetched = prefetch_reference_binding(
        fake_client,
        source_artifact_id="mi2:source",
        target=target,
        operation_id="op-1",
    )
    acquired = acquire_reference_binding(
        fake_client,
        prefetched=prefetched,
        target=target,
        caller_pid=1234,
    )

    assert acquired.binding_value_ref.seal_generation == 0
    assert fake_client.acquire_calls[0]["group_realization_acquire"] == (
        prefetched.group_realization_acquire
    )
