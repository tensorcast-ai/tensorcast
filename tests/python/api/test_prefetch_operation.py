#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import hashlib
import importlib
import json
import uuid
import weakref
from typing import Any, cast

import pytest

import tensorcast as tc
from tensorcast.api._materialize import MaterializationPayload
from tensorcast.api.operation import OperationTimeoutError
from tensorcast.api.store.artifact import Artifact
from tensorcast.api.store.realization_kernel import (
    ArtifactRealizationReport,
    ArtifactRealizationSpec,
)
from tensorcast.common.selection_identity import (
    compute_selection_hash,
)
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.proto.operation.v1 import operation_pb2
from tensorcast.types import (
    BindingReservationCapability,
    BindingValueRef,
    BindingValueVerificationState,
    GroupRealizationAcquireRef,
    PrefetchHandoff,
    PrefetchHandoffSet,
    RealizationTarget,
    RealizationTargetSet,
    RuntimeBindingMemberRef,
    RuntimeBindingResolvedLayout,
    RuntimeBindingSourceMemberRef,
    RuntimeBindingSourceRef,
    RuntimeBindingSourceReuseDecision,
    RuntimeTopologyRef,
)


def _profile_records(tmp_path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for path in tmp_path.glob("tensorcast_pid*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


class _Client:
    def __init__(self) -> None:
        self.prefetch_binding_calls: list[dict[str, object]] = []
        self.query_replica_state = store_daemon_pb2.REPLICA_OPERATION_STATE_SUCCESS
        self.wait_replica_state = store_daemon_pb2.REPLICA_OPERATION_STATE_SUCCESS
        self.release_replica_calls: list[str] = []
        self.release_replica_result = True

    def get_artifact_index_by_id(self, artifact_id: str) -> bytes:
        del artifact_id
        return b"{}"

    def query_replica_status(self, ticket: store_daemon_pb2.ReplicaTicket):
        resp = store_daemon_pb2.QueryReplicaStatusResponse()
        resp.ticket.replica_uuid = ticket.replica_uuid
        resp.status.state = self.query_replica_state
        return resp

    def wait_replica_status(
        self, ticket: store_daemon_pb2.ReplicaTicket, *, timeout_ms: int | None
    ):
        del timeout_ms
        resp = store_daemon_pb2.WaitReplicaStatusResponse()
        resp.ticket.replica_uuid = ticket.replica_uuid
        resp.status.state = self.wait_replica_state
        return resp

    def release_replica(self, ticket: store_daemon_pb2.ReplicaTicket):
        self.release_replica_calls.append(str(ticket.replica_uuid))
        return store_daemon_pb2.ReleaseReplicaResponse(
            released=self.release_replica_result
        )

    def _prefetched_binding(
        self,
        target: RealizationTarget,
        *,
        readiness: object,
        staged_value: bool = False,
        wait_for_publish: bool = False,
    ) -> PrefetchHandoff:
        device_uuid = str(target.device_uuid or "GPU-0")
        suffix = target.member.member_index + 1
        value_id = f"staged-value-{suffix}" if staged_value else f"value-{suffix}"
        binding_ref = BindingValueRef(
            binding_id=f"binding-{suffix}",
            binding_layout_id=target.resolved_layout.binding_layout_id,
            binding_value_id=value_id,
            seal_generation=1,
        )
        capability = BindingReservationCapability(
            capability_id=f"capability-{suffix}",
            binding_value_ref=binding_ref,
            daemon_id="daemon-1",
            daemon_session_id="sess",
            device_uuid=device_uuid,
            member=target.member,
            reservation_bytes=1024 * suffix,
            scope_digest=f"scope-digest-{suffix}",
            expires_at_ms=1234,
        )
        return PrefetchHandoff(
            local_serving_ref=f"binding-local:binding-{suffix}:{value_id}",
            binding_value_ref=binding_ref,
            daemon_id="daemon-1",
            daemon_session_id="sess",
            device_uuid=device_uuid,
            member=target.member,
            reservation_bytes=1024 * suffix,
            reservation_capability=capability,
            readiness=cast(Any, readiness),
            verification_state=BindingValueVerificationState.LOCAL_ONLY,
            serving_artifact_id="mi2:serving",
            expires_at_ms=1234,
            staged_value=staged_value,
            group_realization_acquire=(
                GroupRealizationAcquireRef(
                    transaction_id="txn-1",
                    version_set_id="version-set-1",
                    part_id=target.member.member_id,
                    staging_token=f"stage-{suffix}",
                    wait_for_publish=wait_for_publish,
                    wait_timeout_ms=2500,
                )
                if staged_value
                else None
            ),
        )

    def prefetch_serving_binding(self, **kwargs):
        self.prefetch_binding_calls.append(kwargs)
        target = cast(RealizationTarget | RealizationTargetSet, kwargs["target"])
        operation_id = str(kwargs.get("operation_id") or "prefetch-binding-op")
        readiness = kwargs["requested_readiness"]
        if isinstance(target, RealizationTargetSet):
            staged_members = target.source.source_kind == "runtime_artifact_set"
            result = PrefetchHandoffSet(
                runtime=target.runtime,
                topology=target.topology,
                group_id=target.group_id,
                members=tuple(
                    self._prefetched_binding(
                        member,
                        readiness=readiness,
                        staged_value=staged_members,
                        wait_for_publish=staged_members,
                    )
                    for member in target.members
                ),
                readiness=cast(Any, readiness),
                expires_at_ms=1234,
            )
        else:
            result = self._prefetched_binding(target, readiness=readiness)
        response = store_daemon_pb2.PrefetchServingBindingResponse()
        response.operation_ref.operation_id = operation_id
        response.status.state = operation_pb2.OPERATION_STATE_SUCCESS
        response.status.result.Pack(result.to_proto())
        return response


class _Runtime:
    daemon_endpoint = "daemon"
    daemon_id = "daemon-1"
    session_id = "sess"
    closed = False

    def __init__(self) -> None:
        self._client = _Client()
        self.cached: list[Any] = []
        self.invalidated: list[tuple[str, str]] = []

    def ensure_client(self) -> _Client:
        return self._client

    def get_artifact_index_cached(self, artifact_id: str):  # noqa: ANN001, ARG002
        return None

    def invalidate_artifact(self, artifact_id: str, *, reason: str) -> None:
        self.invalidated.append((str(artifact_id), str(reason)))

    def cache_artifact_index(self, entry: object) -> None:
        self.cached.append(entry)


class _Pipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def materialize_subset(self, **kwargs):
        self.calls.append(kwargs)
        replica_uuid = str(kwargs.get("replica_uuid") or "")
        payload = MaterializationPayload(
            artifact_id=str(kwargs.get("artifact_id") or ""),
            canonical_index_bytes=b"{}",
            descriptors=(),
            payload_iter=lambda: iter(()),
            replica_uuid=replica_uuid,
            ticket_replica_uuid=replica_uuid,
        )
        return payload, 0


class _Store:
    def __init__(self) -> None:
        self._runtime = _Runtime()
        self._materialization = _Pipeline()
        self.closed = False


def _store_ref(store: _Store) -> Any:
    return cast(Any, weakref.ref(store))


def _realization_target(
    *,
    topology: RuntimeTopologyRef | None = None,
    source: RuntimeBindingSourceRef | None = None,
    member_id: str = "member-0",
    member_index: int = 0,
    member_count: int = 1,
    device: str = "cuda:0",
    device_uuid: str = "GPU-0",
    binding_layout_id: str = "layout-1",
    target_layout: bytes = b"target-layout",
    target_index_bytes: bytes = b"target-index",
    target_layout_hash: str = "target-layout-hash",
    spec_digest: str = "spec-digest",
) -> RealizationTarget:
    topology = topology or RuntimeTopologyRef(schema_topology_digest="topology-schema")
    member = RuntimeBindingMemberRef(
        member_id=member_id,
        member_index=member_index,
        member_count=member_count,
        group_id="group-1",
    )
    source = source or RuntimeBindingSourceRef(
        source_kind="checkpoint_artifact",
        artifact_selection_digest="selection-digest",
        source_artifact_ref="mi2:source",
        source_schema_hash="source-schema",
    )
    source_reuse = RuntimeBindingSourceReuseDecision(
        mode="checkpoint_to_runtime",
        representation_contract_hash="repr-contract",
    )
    resolved_layout = RuntimeBindingResolvedLayout(
        binding_layout_id=binding_layout_id,
        source=source,
        source_reuse=source_reuse,
        topology=topology,
        member=member,
        target_layout=target_layout,
        target_index_bytes=target_index_bytes,
        target_layout_hash=target_layout_hash,
        tensor_schema_hash="tensor-schema",
        spec_digest=spec_digest,
        source_schema_hash="source-schema",
        copy_plan_bytes=b"copy-plan",
        dst_specs_bytes=b"dst-specs",
    )
    return RealizationTarget(
        runtime="vllm",
        device=device,
        device_uuid=device_uuid,
        source=source,
        topology=topology,
        member=member,
        model_config_digest="model-config",
        runtime_build_digest="serving-build",
        resolved_layout=resolved_layout,
    )


def _realization_target_set() -> RealizationTargetSet:
    topology = RuntimeTopologyRef(schema_topology_digest="topology-schema")
    source = RuntimeBindingSourceRef(
        source_kind="checkpoint_artifact",
        artifact_selection_digest="selection-digest",
        source_artifact_ref="mi2:source",
        source_schema_hash="source-schema",
    )
    target_0 = _realization_target(
        topology=topology,
        source=source,
        member_id="member-0",
        member_index=0,
        member_count=2,
        device="cuda:0",
        device_uuid="GPU-0",
        binding_layout_id="layout-0",
        target_layout=b"target-layout-0",
        target_index_bytes=b"target-index-0",
        target_layout_hash="target-layout-hash-0",
        spec_digest="spec-digest-0",
    )
    target_1 = _realization_target(
        topology=topology,
        source=source,
        member_id="member-1",
        member_index=1,
        member_count=2,
        device="cuda:1",
        device_uuid="GPU-1",
        binding_layout_id="layout-1",
        target_layout=b"target-layout-1",
        target_index_bytes=b"target-index-1",
        target_layout_hash="target-layout-hash-1",
        spec_digest="spec-digest-1",
    )
    return RealizationTargetSet(
        runtime="vllm",
        source=source,
        topology=topology,
        group_id="group-1",
        members=(target_0, target_1),
    )


def _serving_artifact_realization_target_set() -> RealizationTargetSet:
    topology = RuntimeTopologyRef(schema_topology_digest="topology-schema")
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
    source = RuntimeBindingSourceRef(
        source_kind="runtime_artifact_set",
        artifact_selection_digest="artifact-set-selection",
        source_schema_hash="source-schema",
        topology=topology,
        members=(
            RuntimeBindingSourceMemberRef(
                member=member_0,
                artifact_ref="mi2:serving-member-0",
            ),
            RuntimeBindingSourceMemberRef(
                member=member_1,
                artifact_ref="mi2:serving-member-1",
            ),
        ),
    )
    target_0 = _realization_target(
        topology=topology,
        source=source,
        member_id=member_0.member_id,
        member_index=member_0.member_index,
        member_count=member_0.member_count,
        device="cuda:0",
        device_uuid="GPU-0",
        binding_layout_id="layout-0",
        target_layout=b"target-layout-0",
        target_index_bytes=b"target-index-0",
        target_layout_hash="target-layout-hash-0",
        spec_digest="spec-digest-0",
    )
    target_1 = _realization_target(
        topology=topology,
        source=source,
        member_id=member_1.member_id,
        member_index=member_1.member_index,
        member_count=member_1.member_count,
        device="cuda:1",
        device_uuid="GPU-1",
        binding_layout_id="layout-1",
        target_layout=b"target-layout-1",
        target_index_bytes=b"target-index-1",
        target_layout_hash="target-layout-hash-1",
        spec_digest="spec-digest-1",
    )
    return RealizationTargetSet(
        runtime="vllm",
        source=source,
        topology=topology,
        group_id="group-1",
        members=(target_0, target_1),
    )


def test_prefetch_uses_deterministic_operation_id(monkeypatch) -> None:
    artifact_mod = importlib.import_module("tensorcast.api.store.artifact")
    monkeypatch.setattr(artifact_mod, "device_uuid_for",
                        lambda device_id: f"GPU-{device_id}")
    store = _Store()
    artifact = Artifact(store_ref=_store_ref(store), artifact_id="aid")
    ctx = tc.context(request_id="req-1", idempotency_key="idem-1")

    op = artifact.prefetch(device="cuda:0", ctx=ctx)

    daemon_id = store._runtime.daemon_id
    selection_hash = compute_selection_hash(
        view_id="",
        view_subset_hash=None,
    ).hex()
    logical_layout_hash = (
        artifact._resolve_realization_selection().proto.logical_layout_hash.hex()
    )
    device_uuid = "GPU-0"
    action_fingerprint = (
        f"prefetch|daemon={daemon_id}|artifact=aid|layout={logical_layout_hash}"
        f"|selection={selection_hash}|device=0|device_uuid={device_uuid}|lease=NO_LEASE|v2"
    )
    ns = uuid.uuid5(uuid.NAMESPACE_DNS, "tensorcast.op.v1")
    idempotency_key_hex = hashlib.sha256(
        ctx.idempotency_key.encode("utf-8")
    ).hexdigest()
    expected = str(uuid.uuid5(ns, f"{idempotency_key_hex}|{action_fingerprint}"))

    assert store._materialization.calls
    assert store._materialization.calls[0]["replica_uuid"] == expected
    assert (
        store._materialization.calls[0]["lease_mode"]
        == store_daemon_pb2.LeaseMode.LEASE_MODE_NO_LEASE
    )
    assert op.operation_id == expected

    replica = op.result(timeout_s=1.0)
    assert replica.operation_id == expected
    assert replica.daemon_id == daemon_id
    assert replica.report is not None
    assert replica.report.target_kind == "retained_replica"
    assert replica.report.operation_id == expected
    assert replica.report.operation_backend == "daemon_materialization"
    assert replica.report.publishability is not None
    assert replica.report.publishability.publishable is False
    assert replica.report.envelope.backing_kind == "daemon_retained_replica"
    assert replica.report.envelope.retained_bytes == 0
    assert replica.report.strategy_plan is not None
    assert replica.report.strategy_plan.fallback_policy == "fail_closed"
    assert replica.report.representation_admission is not None
    assert (
        replica.report.representation_admission.representation_contract
        == "retained_replica"
    )
    assert replica.report.lifecycle_plan is not None
    assert replica.report.lifecycle_plan.capability == "retained_replica"


def test_retained_binding_prefetch_uses_deterministic_operation_id() -> None:
    store = _Store()
    artifact = Artifact(store_ref=_store_ref(store), artifact_id="aid")
    target = _realization_target()
    ctx = tc.context(request_id="req-1", idempotency_key="idem-1")

    op = artifact.realize_async(
        ArtifactRealizationSpec.retained_binding(target=target),
        ctx=ctx,
    )

    daemon_id = store._runtime.daemon_id
    selection = artifact._resolve_realization_selection().proto
    target_bytes = target.to_proto().SerializeToString(deterministic=True)
    action_fingerprint = hashlib.sha256(
        b"prefetch_serving_binding|"
        + selection.SerializeToString(deterministic=True)
        + b"|"
        + target_bytes
        + b"|readiness=runtime_local_ready|daemon="
        + daemon_id.encode("utf-8")
    ).hexdigest()
    ns = uuid.uuid5(uuid.NAMESPACE_DNS, "tensorcast.op.v1")
    idempotency_key_hex = hashlib.sha256(
        ctx.idempotency_key.encode("utf-8")
    ).hexdigest()
    expected = str(uuid.uuid5(ns, f"{idempotency_key_hex}|{action_fingerprint}"))

    assert store._runtime.ensure_client().prefetch_binding_calls
    call = store._runtime.ensure_client().prefetch_binding_calls[0]
    assert call["operation_id"] == expected
    assert op.operation_id == expected

    result = op.result(timeout_s=1.0)
    assert isinstance(result, PrefetchHandoff)
    assert result.report is not None
    assert result.report.operation_id == expected


def test_prefetch_without_ctx_generates_operation_id() -> None:
    store = _Store()
    artifact = Artifact(store_ref=_store_ref(store), artifact_id="aid")

    op = artifact.prefetch(device="cuda:0")

    assert store._materialization.calls
    replica_uuid = str(store._materialization.calls[0]["replica_uuid"] or "")
    assert replica_uuid
    assert op.operation_id == replica_uuid


def test_realize_async_retained_replica_preserves_prefetch_operation_contract() -> None:
    store = _Store()
    artifact = Artifact(store_ref=_store_ref(store), artifact_id="aid")

    op = artifact.realize_async(ArtifactRealizationSpec.retained_replica(device="cpu"))

    assert store._materialization.calls
    call = store._materialization.calls[0]
    assert str(call["device"]) == "cpu"
    assert call["options"].wait_for_completion is False
    assert call["lease_mode"] == store_daemon_pb2.LeaseMode.LEASE_MODE_NO_LEASE
    assert op.operation_id


def test_realize_async_retained_replica_operation_status_wait_and_cancel() -> None:
    store = _Store()
    artifact = Artifact(store_ref=_store_ref(store), artifact_id="aid")
    client = store._runtime.ensure_client()
    client.query_replica_state = store_daemon_pb2.REPLICA_OPERATION_STATE_RUNNING

    op = artifact.realize_async(ArtifactRealizationSpec.retained_replica(device="cpu"))

    assert op.status().state == "running"
    result = op.result(timeout_s=1.0)
    assert result.operation_id == op.operation_id
    assert op.cancel() is True
    assert client.release_replica_calls == [op.operation_id]


def test_realize_async_retained_replica_degraded_wait_raises_timeout() -> None:
    store = _Store()
    artifact = Artifact(store_ref=_store_ref(store), artifact_id="aid")
    client = store._runtime.ensure_client()
    client.wait_replica_state = store_daemon_pb2.REPLICA_OPERATION_STATE_DEGRADED

    op = artifact.realize_async(ArtifactRealizationSpec.retained_replica(device="cpu"))

    with pytest.raises(OperationTimeoutError, match="Operation wait timeout expired"):
        op.result(timeout_s=0.1)


def test_realize_async_prefetch_targets_emit_report_shaped_profile_events(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("TENSORCAST_PROFILE_DIR", str(tmp_path))
    store = _Store()
    artifact = Artifact(store_ref=_store_ref(store), artifact_id="aid")

    replica_op = artifact.realize_async(
        ArtifactRealizationSpec.retained_replica(device="cpu")
    )
    _ = replica_op.result(timeout_s=1.0)
    retained_op = artifact.realize_async(
        ArtifactRealizationSpec.retained_binding(target=_realization_target())
    )
    _ = retained_op.result(timeout_s=1.0)
    target_set_op = artifact.realize_async(
        ArtifactRealizationSpec.target_set(target=_realization_target_set())
    )
    _ = target_set_op.result(timeout_s=1.0)

    events = [
        record
        for record in _profile_records(tmp_path)
        if record.get("stage") == "artifact.realize"
    ]
    by_kind = {str(event["target_kind"]): event for event in events}
    assert set(by_kind) == {"retained_replica", "retained_binding", "target_set"}
    assert by_kind["retained_replica"]["operation_backend"] == "daemon_materialization"
    assert by_kind["retained_binding"]["operation_backend"] == (
        "daemon_prefetch_serving_binding"
    )
    assert by_kind["target_set"]["operation_backend"] == (
        "daemon_prefetch_serving_binding_set"
    )
    assert by_kind["retained_binding"]["envelope_backing_kind"] == (
        "daemon_retained_binding"
    )
    assert by_kind["target_set"]["target_plan_member_count"] == 2
    assert by_kind["target_set"]["lifecycle_acquire_claim_count"] == 2
    assert by_kind["target_set"]["publishable"] is False


def test_realize_async_retained_binding_completed_operation_status_and_cancel() -> None:
    store = _Store()
    artifact = Artifact(store_ref=_store_ref(store), artifact_id="aid")
    target = _realization_target()

    op = artifact.realize_async(ArtifactRealizationSpec.retained_binding(target=target))

    assert op.status().state == "success"
    assert op.done() is True
    assert op.cancel() is False
    result = op.result(timeout_s=1.0)
    assert isinstance(result, PrefetchHandoff)


def test_realize_async_retained_binding_propagates_call_context_timeout() -> None:
    store = _Store()
    artifact = Artifact(store_ref=_store_ref(store), artifact_id="aid")
    target = _realization_target()

    op = artifact.realize_async(
        ArtifactRealizationSpec.retained_binding(target=target),
        ctx=tc.CallContext(deadline_ms=123456),
    )
    result = op.result(timeout_s=1.0)

    assert isinstance(result, PrefetchHandoff)
    calls = store._runtime.ensure_client().prefetch_binding_calls
    assert calls
    assert calls[0]["timeout_s"] == pytest.approx(123.456)


def test_realize_async_retained_binding_attaches_report_to_result() -> None:
    store = _Store()
    artifact = Artifact(store_ref=_store_ref(store), artifact_id="aid")
    target = _realization_target()

    op = artifact.realize_async(ArtifactRealizationSpec.retained_binding(target=target))
    result = op.result(timeout_s=1.0)

    assert isinstance(result, PrefetchHandoff)
    assert store._runtime.ensure_client().prefetch_binding_calls
    assert result.report is not None
    report = cast(ArtifactRealizationReport, result.report)
    assert report.target_kind == "retained_binding"
    assert report.operation_id == op.operation_id
    assert report.operation_backend == "daemon_prefetch_serving_binding"
    assert report.target_layout_digest == "target-layout-hash"
    assert report.copy_plan_digest == "spec-digest"
    assert report.envelope.backing_kind == "daemon_retained_binding"
    assert report.envelope.retained_bytes == 1024
    assert report.strategy_plan is not None
    assert report.strategy_plan.fallback_policy == "fail_closed"
    assert report.representation_admission is not None
    assert report.representation_admission.representation_contract == (
        "retained_binding"
    )
    assert report.lifecycle_plan is not None
    assert report.lifecycle_plan.capability == "retained_binding"
    assert report.lifecycle_plan.acquire_claim_count == 1
    assert report.lifecycle_plan.acquire_claim_ids == ("capability-1",)
    assert report.publishability is not None
    assert report.publishability.publishable is False
    assert len(report.retained_bindings) == 1
    retained = report.retained_bindings[0]
    assert retained.binding_layout_id == "layout-1"
    assert retained.binding_value_id == "value-1"
    assert retained.reservation_bytes == 1024
    assert retained.readiness == "runtime_local_ready"
    assert retained.verification_state == "local_only"


def test_realize_async_retained_binding_set_attaches_target_set_report() -> None:
    store = _Store()
    artifact = Artifact(store_ref=_store_ref(store), artifact_id="aid")
    target = _realization_target_set()

    op = artifact.realize_async(ArtifactRealizationSpec.target_set(target=target))
    result = op.result(timeout_s=1.0)

    assert isinstance(result, PrefetchHandoffSet)
    assert store._runtime.ensure_client().prefetch_binding_calls
    assert result.report is not None
    report = cast(ArtifactRealizationReport, result.report)
    assert report.target_kind == "target_set"
    assert report.operation_id == op.operation_id
    assert report.operation_backend == "daemon_prefetch_serving_binding_set"
    assert report.envelope.backing_kind == "daemon_retained_binding_set"
    assert report.envelope.export_kind == "binding_reservation_set"
    assert report.envelope.projection_kind == "target_set"
    assert report.envelope.retained_bytes == 3072
    assert len(report.retained_bindings) == 2
    assert report.target_plan is not None
    assert report.target_plan.member_count == 2
    assert report.strategy_plan is not None
    assert report.strategy_plan.source_selection_mode == "same_selection"
    assert report.strategy_plan.source_coordination == "same_daemon_session"
    assert report.strategy_plan.collective_policy == "collective_first_candidate"
    assert report.strategy_plan.group_barriers == ("member_readiness",)
    assert report.lifecycle_plan is not None
    assert report.lifecycle_plan.capability == "target_set"
    assert report.lifecycle_plan.acquire_claim_count == 2
    assert report.lifecycle_plan.acquire_claim_ids == (
        "capability-1",
        "capability-2",
    )
    assert report.lifecycle_plan.release_policy == ("release_binding_reservations",)
    assert report.representation_admission is not None
    assert report.representation_admission.representation_contract == (
        "target_set_binding_reservation"
    )
    assert report.target_set is not None
    assert report.publishability is not None
    assert report.publishability.publishable is False
    assert report.target_set.group_id == "group-1"
    assert report.target_set.runtime == "vllm"
    assert report.target_set.member_count == 2
    assert report.target_set.successful_member_count == 2
    assert report.target_set.source_selection_mode == "same_selection"
    assert report.target_set.total_reservation_bytes == 3072
    assert report.target_set.same_daemon_session is True
    assert [member.member_id for member in report.target_set.members] == [
        "member-0",
        "member-1",
    ]
    assert [member.target_layout_digest for member in report.target_set.members] == [
        "target-layout-hash-0",
        "target-layout-hash-1",
    ]
    assert all(member.report is report for member in result.members)


def test_retained_binding_realization_rejects_target_set_bypass() -> None:
    store = _Store()
    artifact = Artifact(store_ref=_store_ref(store), artifact_id="aid")

    with pytest.raises(
        tc.ArtifactError,
        match="RealizationTargetSet requires target_set realization",
    ):
        artifact.realize_async(
            ArtifactRealizationSpec.retained_binding(target=_realization_target_set())
        )


def test_realize_async_target_set_per_part_selection_reports_group_lifecycle() -> None:
    store = _Store()
    artifact = Artifact(store_ref=_store_ref(store), artifact_id="aid")
    target = _serving_artifact_realization_target_set()

    op = artifact.realize_async(ArtifactRealizationSpec.target_set(target=target))
    result = op.result(timeout_s=1.0)

    assert isinstance(result, PrefetchHandoffSet)
    assert result.report is not None
    report = cast(ArtifactRealizationReport, result.report)
    assert report.target_kind == "target_set"
    assert report.envelope.release_policy == (
        "release_binding_reservations",
        "release_group_staged_acquire",
    )
    assert report.target_set is not None
    assert report.target_set.source_kind == "runtime_artifact_set"
    assert report.target_set.source_selection_mode == "per_part_selection"
    assert report.target_set.publish_barrier is True
    assert report.target_set.group_realization_transaction_ids == ("txn-1",)
    assert report.target_set.group_realization_version_set_ids == ("version-set-1",)
    assert [member.source_artifact_ref for member in report.target_set.members] == [
        "mi2:serving-member-0",
        "mi2:serving-member-1",
    ]
    assert [member.copy_plan_digest for member in report.target_set.members] == [
        "spec-digest-0",
        "spec-digest-1",
    ]
    assert all(member.staged_value for member in report.target_set.members)
    assert report.strategy_plan is not None
    assert report.strategy_plan.source_selection_mode == "per_part_selection"
    assert report.strategy_plan.group_barriers == (
        "member_readiness",
        "group_acquire",
        "staged_values",
        "publish_barrier",
    )
    assert report.lifecycle_plan is not None
    assert report.lifecycle_plan.staged_value_count == 2
    assert report.lifecycle_plan.acquire_claim_ids == (
        "capability-1",
        "capability-2",
    )
    assert report.lifecycle_plan.member_release_policies["member-0"] == (
        "release_binding_reservation",
        "release_group_staged_acquire",
    )


def test_prefetch_target_set_uses_target_set_realization_spec() -> None:
    store = _Store()
    artifact = Artifact(store_ref=_store_ref(store), artifact_id="aid")
    target = _realization_target_set()

    op = artifact.prefetch(target=target)
    result = op.result(timeout_s=1.0)

    assert isinstance(result, PrefetchHandoffSet)
    assert store._runtime.ensure_client().prefetch_binding_calls[0]["target"] == target
    report = cast(ArtifactRealizationReport, result.report)
    assert report.target_kind == "target_set"
    assert report.target_set is not None
    assert report.target_set.member_count == 2


def test_prefetch_target_set_allows_per_member_source_reuse_contract() -> None:
    store = _Store()
    artifact = Artifact(store_ref=_store_ref(store), artifact_id="aid")
    target = _realization_target_set()
    member_1 = target.members[1]
    member_1_layout = member_1.resolved_layout.model_copy(
        update={
            "source_reuse": RuntimeBindingSourceReuseDecision(
                mode="checkpoint_to_runtime",
                representation_contract_hash="repr-contract-member-1",
            )
        }
    )
    member_1 = member_1.model_copy(update={"resolved_layout": member_1_layout})
    target = target.model_copy(
        update={"members": (target.members[0], member_1)}
    )

    op = artifact.prefetch(target=target)
    result = op.result(timeout_s=1.0)

    assert isinstance(result, PrefetchHandoffSet)
    assert store._runtime.ensure_client().prefetch_binding_calls[0]["target"] == target
