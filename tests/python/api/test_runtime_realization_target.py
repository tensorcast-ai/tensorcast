#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import inspect

import pytest

from tensorcast.api.store.artifact import Artifact
from tensorcast.types import (
    BindingReservationCapability,
    BindingValueRef,
    BindingValueVerificationState,
    GroupRealizationAcquireRef,
    PrefetchHandoff,
    PrefetchHandoffMemberFailure,
    PrefetchHandoffSet,
    PrefetchRetentionPolicy,
    RealizationTarget,
    RealizationTargetSet,
    RuntimeBindingMemberRef,
    RuntimeBindingResolvedLayout,
    RuntimeBindingSourceMemberRef,
    RuntimeBindingSourceRef,
    RuntimeBindingSourceReuseDecision,
    RuntimeTopologyRef,
    plan_runtime_binding_source_reuse,
)


def _topology() -> RuntimeTopologyRef:
    return RuntimeTopologyRef(
        schema_topology_digest="topology-schema",
        admission_topology_digest="topology-admission",
    )


def _member() -> RuntimeBindingMemberRef:
    return RuntimeBindingMemberRef(
        member_id="member-0",
        member_index=0,
        member_count=1,
        group_id="group-1",
    )


def _checkpoint_source() -> RuntimeBindingSourceRef:
    return RuntimeBindingSourceRef(
        source_kind="checkpoint_artifact",
        artifact_selection_digest="selection-digest",
        source_artifact_ref="mi2:checkpoint",
        source_schema_hash="source-schema",
    )


def _resolved_layout(
    *,
    source: RuntimeBindingSourceRef | None = None,
    source_reuse: RuntimeBindingSourceReuseDecision | None = None,
    topology: RuntimeTopologyRef | None = None,
    member: RuntimeBindingMemberRef | None = None,
) -> RuntimeBindingResolvedLayout:
    return RuntimeBindingResolvedLayout(
        binding_layout_id="layout-1",
        source=source or _checkpoint_source(),
        source_reuse=source_reuse
        or RuntimeBindingSourceReuseDecision(
            mode="checkpoint_to_runtime",
            representation_contract_hash="repr-contract",
        ),
        topology=topology or _topology(),
        member=member or _member(),
        target_layout=b"target-layout",
        target_index_bytes=b"target-index",
        target_layout_hash="target-layout-hash",
        tensor_schema_hash="tensor-schema",
        spec_digest="spec-digest",
        source_schema_hash="source-schema",
        copy_plan_bytes=b"copy-plan",
        dst_specs_bytes=b"dst-specs",
    )


def _target(
    *,
    source: RuntimeBindingSourceRef | None = None,
    source_reuse: RuntimeBindingSourceReuseDecision | None = None,
    topology: RuntimeTopologyRef | None = None,
    member: RuntimeBindingMemberRef | None = None,
    device: str = "cuda:0",
    device_uuid: str = "GPU-0",
    target_layout: bytes = b"target-layout",
    target_index_bytes: bytes = b"target-index",
    target_layout_hash: str = "target-layout-hash",
) -> RealizationTarget:
    resolved_source = source or _checkpoint_source()
    resolved_topology = topology or _topology()
    resolved_member = member or _member()
    resolved_layout = _resolved_layout(
        source=resolved_source,
        source_reuse=source_reuse,
        topology=resolved_topology,
        member=resolved_member,
    ).model_copy(
        update={
            "target_layout": target_layout,
            "target_index_bytes": target_index_bytes,
            "target_layout_hash": target_layout_hash,
        }
    )
    return RealizationTarget(
        runtime="vllm",
        device=device,
        device_uuid=device_uuid,
        source=resolved_source,
        topology=resolved_topology,
        member=resolved_member,
        model_config_digest="model-config",
        runtime_build_digest="serving-build",
        resolved_layout=resolved_layout,
    )


def test_prefetch_signature_keeps_device_and_adds_target() -> None:
    params = inspect.signature(Artifact.prefetch).parameters

    assert "device" in params
    assert params["device"].default is None
    assert "target" in params
    assert "readiness" in params
    assert "retention" in params


def test_runtime_target_proto_roundtrip_includes_source() -> None:
    target = _target()

    roundtripped = RealizationTarget.from_proto(target.to_proto())

    assert roundtripped == target
    assert roundtripped.source.source_kind == "checkpoint_artifact"
    assert roundtripped.resolved_layout.source_reuse.mode == "checkpoint_to_runtime"


def test_runtime_target_set_requires_shared_source() -> None:
    target = _target()
    runtime_set = RealizationTargetSet(
        runtime="vllm",
        source=target.source,
        topology=target.topology,
        group_id="group-1",
        members=(target,),
    )

    assert RealizationTargetSet.from_proto(runtime_set.to_proto()) == runtime_set


def test_runtime_target_set_allows_distinct_member_device_and_layout_specs() -> None:
    topology = _topology()
    source = _checkpoint_source()
    member_0 = RuntimeBindingMemberRef(
        member_id="member-0",
        member_index=0,
        member_count=2,
        group_id="group-1",
    )
    member_1 = RuntimeBindingMemberRef(
        member_id="member-1",
        member_index=1,
        member_count=2,
        group_id="group-1",
    )
    target_0 = _target(
        source=source,
        topology=topology,
        member=member_0,
        device="cuda:0",
        device_uuid="GPU-0",
        target_layout=b"target-layout-member-0",
        target_index_bytes=b"target-index-member-0",
        target_layout_hash="target-layout-hash-member-0",
    )
    target_1 = _target(
        source=source,
        topology=topology,
        member=member_1,
        device="cuda:1",
        device_uuid="GPU-1",
        target_layout=b"target-layout-member-1",
        target_index_bytes=b"target-index-member-1",
        target_layout_hash="target-layout-hash-member-1",
    )

    runtime_set = RealizationTargetSet(
        runtime="vllm",
        source=source,
        topology=topology,
        group_id="group-1",
        members=(target_0, target_1),
    )
    roundtripped = RealizationTargetSet.from_proto(runtime_set.to_proto())

    assert roundtripped == runtime_set
    assert {member.device_uuid for member in roundtripped.members} == {
        "GPU-0",
        "GPU-1",
    }
    assert {
        member.resolved_layout.target_layout_hash for member in roundtripped.members
    } == {"target-layout-hash-member-0", "target-layout-hash-member-1"}
    assert all(member.topology == topology for member in roundtripped.members)


def test_direct_runtime_member_copy_requires_matching_member_schema_and_layout() -> (
    None
):
    topology = _topology()
    member = _member()
    source = RuntimeBindingSourceRef(
        source_kind="runtime_artifact_set",
        artifact_selection_digest="selection-digest",
        source_schema_hash="source-schema",
        representation_contract_hash="repr-contract",
        runtime_build_digest="serving-build",
        tensor_schema_hash="tensor-schema",
        topology=topology,
        members=(
            RuntimeBindingSourceMemberRef(
                member=member,
                artifact_ref="mi2:serving-member",
                tensor_schema_hash="tensor-schema",
                target_layout_hash="target-layout-hash",
            ),
        ),
    )
    reuse = RuntimeBindingSourceReuseDecision(
        mode="runtime_direct_member_copy",
        representation_contract_hash="repr-contract",
    )

    target = _target(source=source, source_reuse=reuse)

    assert target.resolved_layout.source_reuse.mode == "runtime_direct_member_copy"


def test_direct_runtime_member_copy_planner_admits_only_matching_source() -> None:
    topology = _topology()
    member = _member()
    source = RuntimeBindingSourceRef(
        source_kind="runtime_artifact_set",
        artifact_selection_digest="selection-digest",
        source_schema_hash="source-schema",
        representation_contract_hash="repr-contract",
        runtime_build_digest="serving-build",
        tensor_schema_hash="tensor-schema",
        topology=topology,
        members=(
            RuntimeBindingSourceMemberRef(
                member=member,
                artifact_ref="mi2:serving-member",
                tensor_schema_hash="tensor-schema",
                target_layout_hash="target-layout-hash",
            ),
        ),
    )

    decision = plan_runtime_binding_source_reuse(
        source=source,
        topology=topology,
        member=member,
        tensor_schema_hash="tensor-schema",
        target_layout_hash="target-layout-hash",
        representation_contract_hash="repr-contract",
    )

    assert decision.mode == "runtime_direct_member_copy"


def test_direct_runtime_member_copy_planner_returns_transform_for_topology_mismatch() -> (
    None
):
    source = RuntimeBindingSourceRef(
        source_kind="runtime_artifact_set",
        artifact_selection_digest="selection-digest",
        source_schema_hash="source-schema",
        representation_contract_hash="repr-contract",
        runtime_build_digest="serving-build",
        tensor_schema_hash="tensor-schema",
        topology=RuntimeTopologyRef(schema_topology_digest="different-topology"),
        members=(
            RuntimeBindingSourceMemberRef(
                member=_member(),
                artifact_ref="mi2:serving-member",
                tensor_schema_hash="tensor-schema",
                target_layout_hash="target-layout-hash",
            ),
        ),
    )
    decision = plan_runtime_binding_source_reuse(
        source=source,
        topology=_topology(),
        member=_member(),
        tensor_schema_hash="tensor-schema",
        target_layout_hash="target-layout-hash",
        representation_contract_hash="repr-contract",
    )

    assert decision.mode == "runtime_transform_required"
    assert "topology" in (decision.reason or "")


def test_transform_required_decision_is_serializable_but_not_direct_copy() -> None:
    decision = RuntimeBindingSourceReuseDecision(
        mode="runtime_transform_required",
        work_plan_hash="work-plan",
    )

    assert RuntimeBindingSourceReuseDecision.from_proto(decision.to_proto()) == decision


def test_prefetch_handoff_result_proto_roundtrip() -> None:
    member = _member()
    binding_ref = BindingValueRef(
        binding_id="binding-1",
        binding_layout_id="layout-1",
        binding_value_id="value-1",
        seal_generation=1,
    )
    capability = BindingReservationCapability(
        capability_id="capability-1",
        binding_value_ref=binding_ref,
        daemon_id="daemon-1",
        daemon_session_id="session-1",
        device_uuid="GPU-0",
        member=member,
        reservation_bytes=1024,
        scope_digest="scope",
        expires_at_ms=1234,
    )
    result = PrefetchHandoff(
        local_serving_ref="binding-local:binding-1:value-1",
        binding_value_ref=binding_ref,
        daemon_id="daemon-1",
        daemon_session_id="session-1",
        device_uuid="GPU-0",
        member=member,
        reservation_bytes=1024,
        reservation_capability=capability,
        readiness="runtime_local_ready",
        verification_state=BindingValueVerificationState.LOCAL_ONLY,
        serving_artifact_id=None,
        expires_at_ms=1234,
    )

    assert PrefetchHandoff.from_proto(result.to_proto()) == result


def test_prefetch_handoff_staged_result_proto_roundtrip() -> None:
    member = _member()
    binding_ref = BindingValueRef(
        binding_id="binding-1",
        binding_layout_id="layout-1",
        binding_value_id="staged-value-1",
        seal_generation=0,
    )
    capability = BindingReservationCapability(
        capability_id="capability-1",
        binding_value_ref=binding_ref,
        daemon_id="daemon-1",
        daemon_session_id="session-1",
        device_uuid="GPU-0",
        member=member,
        reservation_bytes=1024,
        scope_digest="scope",
    )
    result = PrefetchHandoff(
        local_serving_ref="binding-local:binding-1:staged-value-1",
        binding_value_ref=binding_ref,
        daemon_id="daemon-1",
        daemon_session_id="session-1",
        device_uuid="GPU-0",
        member=member,
        reservation_bytes=1024,
        reservation_capability=capability,
        readiness="runtime_local_ready",
        verification_state=BindingValueVerificationState.LOCAL_ONLY,
        staged_value=True,
        group_realization_acquire=GroupRealizationAcquireRef(
            transaction_id="txn-1",
            version_set_id="version-set-1",
            part_id="member-0",
            staging_token="stage-1",
            wait_for_publish=True,
            wait_timeout_ms=250,
        ),
    )

    assert PrefetchHandoff.from_proto(result.to_proto()) == result


def test_prefetch_handoff_set_partial_diagnostics_roundtrip() -> None:
    member = _member()
    binding_ref = BindingValueRef(
        binding_id="binding-1",
        binding_layout_id="layout-1",
        binding_value_id="value-1",
        seal_generation=1,
    )
    capability = BindingReservationCapability(
        capability_id="capability-1",
        binding_value_ref=binding_ref,
        daemon_id="daemon-1",
        daemon_session_id="session-1",
        device_uuid="GPU-0",
        member=member,
        reservation_bytes=1024,
        scope_digest="scope",
    )
    success = PrefetchHandoff(
        local_serving_ref="binding-local:binding-1:value-1",
        binding_value_ref=binding_ref,
        daemon_id="daemon-1",
        daemon_session_id="session-1",
        device_uuid="GPU-0",
        member=member,
        reservation_bytes=1024,
        reservation_capability=capability,
        readiness="runtime_local_ready",
        verification_state=BindingValueVerificationState.LOCAL_ONLY,
    )
    failed_member = RuntimeBindingMemberRef(
        member_id="member-1",
        member_index=1,
        member_count=2,
        group_id="group-1",
    )
    failure = PrefetchHandoffMemberFailure(
        member=failed_member,
        code="FAILED_PRECONDITION",
        message="resolved spec mismatch",
        phase="cache_validation",
        cache_key_digest="cache-key",
        spec_digest="spec",
    )
    result = PrefetchHandoffSet(
        runtime="vllm",
        topology=_topology(),
        group_id="group-1",
        members=(success,),
        readiness="runtime_local_ready",
        member_failures=(failure,),
        partial=True,
    )

    assert PrefetchHandoffSet.from_proto(result.to_proto()) == result


def test_prefetch_handoff_set_rejects_overlap_between_success_and_failure() -> (
    None
):
    member = _member()
    binding_ref = BindingValueRef(
        binding_id="binding-1",
        binding_layout_id="layout-1",
        binding_value_id="value-1",
        seal_generation=1,
    )
    capability = BindingReservationCapability(
        capability_id="capability-1",
        binding_value_ref=binding_ref,
        daemon_id="daemon-1",
        daemon_session_id="session-1",
        device_uuid="GPU-0",
        member=member,
        reservation_bytes=1024,
        scope_digest="scope",
    )
    success = PrefetchHandoff(
        binding_value_ref=binding_ref,
        daemon_id="daemon-1",
        daemon_session_id="session-1",
        device_uuid="GPU-0",
        member=member,
        reservation_bytes=1024,
        reservation_capability=capability,
        readiness="runtime_local_ready",
        verification_state=BindingValueVerificationState.LOCAL_ONLY,
    )
    failure = PrefetchHandoffMemberFailure(
        member=member,
        code="FAILED_PRECONDITION",
        message="same member failed",
    )

    with pytest.raises(ValueError, match="both success and failure"):
        PrefetchHandoffSet(
            runtime="vllm",
            topology=_topology(),
            group_id="group-1",
            members=(success,),
            readiness="runtime_local_ready",
            member_failures=(failure,),
            partial=True,
        )


def test_retention_policy_rejects_negative_duration() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        PrefetchRetentionPolicy(expire_if_unacquired_after_ms=-1)
