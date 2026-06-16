#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tensorcast.api._config import (
    CollectiveLoadGroup,
    ExecutionTopologyContext,
    GetArtifactOptions,
    RetrievalPolicy,
    RetrievalPreference,
)
from tensorcast.api._materialize import MaterializationPayload, TensorPayloadDescriptor
from tensorcast.artifact_runtime import lifecycle as tc_lifecycle
from tensorcast.api.store.common import canonical_index_to_bytes
from tensorcast.api.store.realization_kernel import (
    ArtifactRealizationHandle,
    ArtifactRealizationReport,
    ArtifactRealizationSpec,
    RealizationResourceEnvelope,
    RealizationRetainedBindingReport,
    RealizationStrategyPlan,
    RealizationTargetKind,
    RealizationTargetPlan,
    artifact_realization_profile_payload,
    artifact_realization_report_to_dict,
    binding_materialization_diagnostics_from_response,
    envelope_for_binding,
    envelope_for_caller_tensors,
    envelope_for_mounted_source,
    envelope_for_publication,
    envelope_for_retained_binding,
    envelope_for_runtime_attachment,
    envelope_for_target_set,
    envelope_for_tensor_dict,
    lifecycle_plan_for_envelope,
    materialization_source_label,
    model_runtime_report_for,
    mounted_source_target_digest,
    publishability_report_for,
    release_contract_for,
    report_for_binding_realization,
    report_for_mounted_source,
    report_for_publication,
    report_for_runtime_attachment,
    representation_admission_for_target,
    resolve_artifact_selection,
    retained_binding_lifecycle_plan_for,
    retained_binding_reports_for,
    risk_labels_for_target,
    strategy_plan_for_execution,
    target_set_lifecycle_plan_for,
    target_set_report_for_retained_bindings,
    target_set_representation_admission_for,
    target_set_strategy_plan_for,
)
from tensorcast.api.store.types import (
    ArtifactError,
    CanonicalIndex,
    CanonicalIndexEntry,
)
from tensorcast.proto.common.v1 import capability_token_pb2, common_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.types import (
    BindingReservationCapability,
    BindingValueRef,
    GroupRealizationAcquireRef,
    RuntimeBindingMemberRef,
)


def _canonical_index_bytes() -> bytes:
    entries = (
        CanonicalIndexEntry(
            name="a",
            dtype=torch.float32,
            shape=(2,),
            stride=(1,),
            storage_offset=0,
            segment_offset=0,
            size_bytes=8,
        ),
        CanonicalIndexEntry(
            name="b",
            dtype=torch.float32,
            shape=(4,),
            stride=(1,),
            storage_offset=0,
            segment_offset=8,
            size_bytes=16,
        ),
    )
    return canonical_index_to_bytes(
        CanonicalIndex(entries=entries, total_size_bytes=24, avbs_hash="")
    )


def _execution_topology_options() -> GetArtifactOptions:
    return GetArtifactOptions(
        source=RetrievalPolicy(
            preference=RetrievalPreference.PREFER_DISK,
            allow_p2p=False,
            allow_disk=True,
        ),
        execution_topology=ExecutionTopologyContext(
            collective_group=CollectiveLoadGroup(
                group_id="same-host-tp-load",
                world_size=8,
                rank=3,
            ),
            collective_policy="collective_first",
            source_locality="shared_source",
            source_sharing_domain="node-a",
        ),
    )


def test_materialization_source_label_uses_realization_report_vocabulary() -> None:
    assert (
        materialization_source_label(
            store_daemon_pb2.MATERIALIZATION_SOURCE_LOCAL_REPLICA
        )
        == "local_replica"
    )
    assert materialization_source_label(
        store_daemon_pb2.MATERIALIZATION_SOURCE_P2P
    ) == ("p2p")
    assert materialization_source_label(
        store_daemon_pb2.MATERIALIZATION_SOURCE_DISK
    ) == ("disk")


def test_resolve_artifact_selection_subset_digest_is_stable() -> None:
    index_bytes = _canonical_index_bytes()

    first = resolve_artifact_selection(
        artifact_id="mi2:test:artifact",
        canonical_index_bytes=index_bytes,
        tensor_names=("b",),
        generation_hint=7,
    )
    second = resolve_artifact_selection(
        artifact_id="mi2:test:artifact",
        canonical_index_bytes=index_bytes,
        tensor_names=("b",),
        generation_hint=7,
    )

    assert first.source_selection_digest == second.source_selection_digest
    assert first.proto.artifact_id == "mi2:test:artifact"
    assert tuple(first.proto.tensor_names) == ("b",)
    assert first.generation_hint == 7
    assert first.artifact_profile == "durable_artifact"
    assert first.authority_scope == "daemon_mediated_durable"


def test_resolve_artifact_selection_uses_daemon_key_and_index_resolution() -> None:
    index_bytes = _canonical_index_bytes()

    def key_resolver(key: str) -> tuple[str, int]:
        assert key == "model/demo"
        return "mi2:test:resolved", 42

    def canonical_index_resolver(artifact_id: str) -> bytes:
        assert artifact_id == "mi2:test:resolved"
        return index_bytes

    selection = resolve_artifact_selection(
        artifact_id=None,
        key="model/demo",
        key_resolver=key_resolver,
        canonical_index_resolver=canonical_index_resolver,
    )

    assert selection.artifact_id == "mi2:test:resolved"
    assert selection.key == "model/demo"
    assert selection.generation_hint == 42
    assert selection.canonical_index_bytes == index_bytes
    assert selection.diagnostics["resolved_from_key"] is True
    assert selection.diagnostics["canonical_index_source"] == "daemon"


def test_resolve_artifact_selection_rejects_mixed_id_and_key() -> None:
    with pytest.raises(ArtifactError) as excinfo:
        resolve_artifact_selection(
            artifact_id="mi2:test:artifact",
            key="model/demo",
            canonical_index_bytes=_canonical_index_bytes(),
        )

    assert excinfo.value.status_code == "INVALID_ARGUMENT"


def test_resolve_artifact_selection_digest_tracks_generation_profile_and_scope() -> (
    None
):
    index_bytes = _canonical_index_bytes()
    base = resolve_artifact_selection(
        artifact_id="mi2:test:artifact",
        canonical_index_bytes=index_bytes,
        generation_hint=1,
    )
    newer_generation = resolve_artifact_selection(
        artifact_id="mi2:test:artifact",
        canonical_index_bytes=index_bytes,
        generation_hint=2,
    )
    custom_authority = resolve_artifact_selection(
        artifact_id="mi2:test:artifact",
        canonical_index_bytes=index_bytes,
        generation_hint=1,
        artifact_profile="runtime_local_ready",
        authority_scope="daemon_retained_binding",
    )

    assert newer_generation.source_selection_digest != base.source_selection_digest
    assert custom_authority.source_selection_digest != base.source_selection_digest
    assert custom_authority.artifact_profile == "runtime_local_ready"
    assert custom_authority.authority_scope == "daemon_retained_binding"


def test_resolve_artifact_selection_keeps_target_plan_identity_separate() -> None:
    selection = resolve_artifact_selection(
        artifact_id="mi2:test:artifact",
        canonical_index_bytes=_canonical_index_bytes(),
        tensor_names=("a",),
        generation_hint=7,
    )
    direct_target = RealizationTargetPlan(
        kind="caller_tensors",
        target_layout_digest="target-layout:direct",
        copy_plan_digest=None,
    )
    mapped_target = RealizationTargetPlan(
        kind="binding_adopted",
        target_layout_digest="target-layout:mapped",
        copy_plan_digest="copy-plan:mapped",
        binding_layout_id="binding-layout:mapped",
    )

    assert selection.source_selection_digest
    assert direct_target.target_layout_digest != mapped_target.target_layout_digest
    assert direct_target.copy_plan_digest != mapped_target.copy_plan_digest
    assert selection.source_selection_digest != direct_target.target_layout_digest
    assert selection.source_selection_digest != mapped_target.target_layout_digest
    assert not hasattr(selection, "target_layout_digest")
    assert not hasattr(selection, "copy_plan_digest")


def test_resolve_artifact_selection_view_uses_selected_index_identity() -> None:
    index_bytes = _canonical_index_bytes()
    view_spec = common_pb2.ViewSpec()
    op = view_spec.tensors["b"].ops.add()
    op.narrow.dim = 0
    op.narrow.start = 1
    op.narrow.length = 2

    full = resolve_artifact_selection(
        artifact_id="mi2:test:artifact",
        canonical_index_bytes=index_bytes,
    )
    viewed = resolve_artifact_selection(
        artifact_id="mi2:test:artifact",
        canonical_index_bytes=index_bytes,
        view_spec=view_spec,
        tensor_names=("b",),
    )

    assert viewed.proto.HasField("view_spec")
    assert viewed.view_id
    assert tuple(viewed.proto.tensor_names) == ("b",)
    assert viewed.selected_index_bytes != full.selected_index_bytes
    assert viewed.logical_layout_hash != full.logical_layout_hash
    assert viewed.source_selection_digest != full.source_selection_digest
    assert viewed.diagnostics["selected_index_source"] == "computed"


def test_resolve_artifact_selection_accepts_attested_mounted_source() -> None:
    selection = resolve_artifact_selection(
        artifact_id="msa1:test-session~policy~safetensors~deadbeef",
        canonical_index_bytes=_canonical_index_bytes(),
        generation_hint=11,
    )

    assert selection.artifact_profile == "mounted_source"
    assert selection.authority_scope == "daemon_local_mounted_source"
    assert selection.generation_hint == 11
    assert selection.proto.artifact_id.startswith("msa1:")


def test_mounted_source_selection_requires_attested_index() -> None:
    with pytest.raises(ArtifactError) as excinfo:
        resolve_artifact_selection(artifact_id="msa1:local-source")

    assert excinfo.value.status_code == "FAILED_PRECONDITION"


def test_mounted_source_selection_rejects_key_activation() -> None:
    def key_resolver(key: str) -> tuple[str, int]:
        assert key == "disk/bootstrap"
        return "msa1:test-session~policy~safetensors~source", 3

    with pytest.raises(ArtifactError) as excinfo:
        resolve_artifact_selection(
            artifact_id=None,
            key="disk/bootstrap",
            key_resolver=key_resolver,
            canonical_index_bytes=_canonical_index_bytes(),
        )

    assert excinfo.value.status_code == "FAILED_PRECONDITION"
    assert "explicit msa1 artifact ids" in str(excinfo.value)


def test_mounted_source_selection_rejects_global_store_index_routing() -> None:
    index_fetches: list[str] = []

    def canonical_index_resolver(artifact_id: str) -> bytes:
        index_fetches.append(artifact_id)
        return _canonical_index_bytes()

    with pytest.raises(ArtifactError) as excinfo:
        resolve_artifact_selection(
            artifact_id="msa1:test-session~policy~safetensors~source",
            canonical_index_resolver=canonical_index_resolver,
        )

    assert excinfo.value.status_code == "FAILED_PRECONDITION"
    assert index_fetches == []


def test_mounted_source_selection_rejects_durable_authority_override() -> None:
    with pytest.raises(ArtifactError) as excinfo:
        resolve_artifact_selection(
            artifact_id="msa1:test-session~policy~safetensors~source",
            canonical_index_bytes=_canonical_index_bytes(),
            authority_scope="daemon_mediated_durable",
        )

    assert excinfo.value.status_code == "FAILED_PRECONDITION"
    assert "daemon-local mounted-source authority" in str(excinfo.value)


def test_mounted_source_report_captures_promotion_identity() -> None:
    source_artifact_id = "msa1:test-session~policy~safetensors~source"
    promoted_artifact_id = "mi2:idx:data"
    canonical_index_bytes = _canonical_index_bytes()
    selection = resolve_artifact_selection(
        artifact_id=source_artifact_id,
        canonical_index_bytes=canonical_index_bytes,
        generation_hint=5,
    )
    target_plan = RealizationTargetPlan(
        kind="mounted_source",
        target_layout_digest=mounted_source_target_digest(
            source_artifact_id=source_artifact_id,
            promoted_artifact_id=promoted_artifact_id,
            canonical_index_bytes=canonical_index_bytes,
        ),
    )
    envelope = envelope_for_mounted_source(
        canonical_index_bytes=canonical_index_bytes,
    )

    envelope.validate_for_target(target_plan)
    report = report_for_mounted_source(
        selection=selection,
        promoted_artifact_id=promoted_artifact_id,
        generation=9,
        canonical_index_bytes=canonical_index_bytes,
        verify_checksums=False,
        target_plan=target_plan,
        envelope=envelope,
    )

    assert envelope.backing_kind == "daemon_mounted_source"
    assert envelope.projection_kind == "artifact_identity"
    assert envelope.release_policy == ("drop_promotion_handle",)
    assert report.target_kind == "mounted_source"
    assert report.operation_backend == "daemon_mounted_source_promotion"
    assert report.mounted_source is not None
    assert report.mounted_source.source_artifact_id == source_artifact_id
    assert report.mounted_source.promoted_artifact_id == promoted_artifact_id
    assert report.mounted_source.verify_checksums is False
    assert report.mounted_source.generation == 9
    assert report.mounted_source.promoted_artifact_profile == "durable_artifact"
    assert report.representation_admission is not None
    assert report.representation_admission.representation_contract == "mounted_source"
    assert report.lifecycle_plan is not None
    assert report.lifecycle_plan.capability == "mounted_source"
    assert report.lifecycle_plan.release_policy == ("drop_promotion_handle",)
    assert "mounted_source" in report.risk_labels


def test_resolve_artifact_selection_accepts_mapped_source_view_hint() -> None:
    index_bytes = _canonical_index_bytes()
    subset = resolve_artifact_selection(
        artifact_id="mi2:test:artifact",
        canonical_index_bytes=index_bytes,
        tensor_names=("a",),
    )

    mapped = resolve_artifact_selection(
        artifact_id="mi2:test:artifact",
        canonical_index_bytes=index_bytes,
        view_id="mapped:v1:target-layout",
        view_index_hint=subset.selected_index_bytes,
        allow_view_id_without_spec=True,
    )

    assert mapped.view_id == "mapped:v1:target-layout"
    assert mapped.selected_index_bytes == subset.selected_index_bytes
    assert mapped.logical_layout_hash == subset.logical_layout_hash
    assert mapped.source_selection_digest != subset.source_selection_digest
    assert mapped.diagnostics["selected_index_source"] == "hint"


def test_group_member_same_and_per_part_selection_identity() -> None:
    index_bytes = _canonical_index_bytes()
    rank0 = resolve_artifact_selection(
        artifact_id="mi2:test:artifact",
        canonical_index_bytes=index_bytes,
        tensor_names=("a",),
    )
    rank1 = resolve_artifact_selection(
        artifact_id="mi2:test:artifact",
        canonical_index_bytes=index_bytes,
        tensor_names=("a",),
    )
    per_part = resolve_artifact_selection(
        artifact_id="mi2:test:artifact",
        canonical_index_bytes=index_bytes,
        tensor_names=("b",),
    )

    assert rank0.source_selection_digest == rank1.source_selection_digest
    assert rank0.selection_hash == rank1.selection_hash
    assert rank0.source_selection_digest != per_part.source_selection_digest
    assert rank0.selection_hash != per_part.selection_hash


def test_caller_target_envelope_requires_target_layout_digest() -> None:
    envelope = RealizationResourceEnvelope(
        backing_kind="caller_region",
        export_kind="direct_write",
        projection_kind="completion",
        owner_kind="caller",
        release_policy=("unregister_target_region",),
        mutability_contract="caller_mutable",
        release_strictness="strict",
        export_lifetime_kind="none",
    )

    with pytest.raises(ArtifactError) as excinfo:
        envelope.validate_for_target(RealizationTargetPlan(kind="caller_tensors"))

    assert excinfo.value.status_code == "FAILED_PRECONDITION"


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        ("backing_kind", ""),
        ("export_kind", ""),
        ("projection_kind", ""),
        ("owner_kind", ""),
        ("release_policy", ()),
        ("mutability_contract", ""),
        ("release_strictness", ""),
        ("export_lifetime_kind", ""),
    ),
)
def test_resource_envelope_requires_shared_risk_fields(
    field_name: str,
    replacement: object,
) -> None:
    envelope = envelope_for_tensor_dict(
        {"a": torch.zeros(2, dtype=torch.float32)},
        source="disk",
    )
    invalid_envelope = replace(envelope, **{field_name: replacement})

    with pytest.raises(ArtifactError) as excinfo:
        invalid_envelope.validate_for_target(
            RealizationTargetPlan(kind="tensor_dict", device="cpu")
        )

    assert excinfo.value.status_code == "FAILED_PRECONDITION"
    assert field_name in str(excinfo.value)


def test_caller_tensor_envelope_accounts_cpu_copy_costs() -> None:
    target = {"a": torch.zeros(2, dtype=torch.float32)}

    envelope = envelope_for_caller_tensors(target)
    envelope.validate_for_target(
        RealizationTargetPlan(
            kind="caller_tensors",
            target_layout_digest="target-digest",
        )
    )

    assert envelope.backing_kind == "caller_region_or_temporary_replica"
    assert envelope.export_kind == "temporary_copy"
    assert envelope.copy_bytes == 8
    assert envelope.copy_count == 1
    assert envelope.temporary_replica_bytes == 8
    assert envelope.direct_write_bytes == 0


class _FakeCudaTensor:
    is_cuda = True

    def __init__(self, *, nbytes: int) -> None:
        self._nbytes = nbytes

    def element_size(self) -> int:
        return 4

    def numel(self) -> int:
        return self._nbytes // self.element_size()


def test_resource_envelope_adapter_matrix_covers_implementation_objects() -> None:
    payload_tensor = torch.zeros(2, dtype=torch.float32)
    descriptor = TensorPayloadDescriptor(
        name="a",
        dtype="float32",
        shape=(2,),
        stride=(1,),
        buffer_offset=0,
        byte_length=8,
        storage_offset=0,
    )
    materialization_payload = MaterializationPayload(
        artifact_id="mi2:test:payload",
        canonical_index_bytes=_canonical_index_bytes(),
        descriptors=(descriptor,),
        payload_iter=lambda: iter(((descriptor, payload_tensor),)),
        replica_uuid="replica-payload",
        state_dict={"a": payload_tensor},
        retry_reason_buckets={"disk_retry": 1},
    )
    assert materialization_payload.state_dict is not None
    payload_envelope = envelope_for_tensor_dict(
        materialization_payload.state_dict,
        source="disk",
        retry_reason_buckets=materialization_payload.retry_reason_buckets,
    )

    region_backed_target = SimpleNamespace(
        tensors={"a": _FakeCudaTensor(nbytes=64)},
        fallback_reason_buckets={"no_fallback": 0},
    )
    region_envelope = envelope_for_caller_tensors(
        region_backed_target.tensors,
        fallback_reason_buckets=region_backed_target.fallback_reason_buckets,
    )

    owned_binding_slot = SimpleNamespace(
        binding_id="binding-1",
        binding_layout_id="layout-1",
        current_value=SimpleNamespace(
            binding_value_id="value-1",
            seal_generation=1,
            is_artifact_backed=True,
        ),
        last_materialization_diagnostics={
            "total_bytes": 96,
            "retry_reason_buckets": {"disk_retry": 2},
        },
        last_execution_diagnostics=SimpleNamespace(
            actual_collective_committed_bytes=16,
            actual_local_typed_bytes=32,
            fallback_bytes=8,
            residual_bytes=4,
            actual_generic_backend_bytes=12,
            collective_peak_temporary_bytes=20,
        ),
        last_source_bound_plan_diagnostics=SimpleNamespace(
            planner_reject_reason_buckets={"not_collective": 3},
        ),
    )
    owned_binding_envelope = envelope_for_binding(
        owned_binding_slot,
        target_kind="binding_owned",
    )

    def retained_state(
        member_index: int,
        *,
        reservation_bytes: int,
        staged_value: bool,
    ) -> SimpleNamespace:
        binding_value_ref = BindingValueRef(
            binding_id=f"binding-{member_index}",
            binding_layout_id=f"layout-{member_index}",
            binding_value_id=f"value-{member_index}",
            seal_generation=1,
        )
        member = RuntimeBindingMemberRef(
            member_id=f"member-{member_index}",
            member_index=member_index,
            member_count=2,
            group_id="group-1",
        )
        capability = BindingReservationCapability(
            capability_id=f"capability-{member_index}",
            binding_value_ref=binding_value_ref,
            daemon_id="daemon-1",
            daemon_session_id="session-1",
            device_uuid=f"GPU-{member_index}",
            member=member,
            reservation_bytes=reservation_bytes,
            scope_digest=f"scope-{member_index}",
            expires_at_ms=4_102_444_800_000,
        )
        return SimpleNamespace(
            binding_value_ref=binding_value_ref,
            reservation_capability=capability,
            member=member,
            daemon_id="daemon-1",
            daemon_session_id="session-1",
            device_uuid=f"GPU-{member_index}",
            reservation_bytes=reservation_bytes,
            readiness="runtime_local_ready",
            verification_state="local_only",
            staged_value=staged_value,
            group_realization_acquire=GroupRealizationAcquireRef(
                transaction_id="tx-1",
                version_set_id="vs-1",
                part_id=f"part-{member_index}",
                staging_token=f"staging-{member_index}",
                wait_for_publish=True,
                wait_timeout_ms=250,
            ),
        )

    retained_acquire_state = retained_state(
        0,
        reservation_bytes=4096,
        staged_value=True,
    )
    retained_reports = retained_binding_reports_for(retained_acquire_state)
    assert retained_reports[0].reservation_capability_expires_at_ms == (
        4_102_444_800_000
    )
    retained_envelope = envelope_for_retained_binding(retained_reports)

    runtime_attachment_state = SimpleNamespace(
        tensors={"a": torch.zeros(1, dtype=torch.float32)},
        retained=True,
        reservation_bytes=2048,
        fallback_reason_buckets={"lease_refresh": 1},
    )
    runtime_attachment_envelope = envelope_for_runtime_attachment(
        runtime_attachment_state.tensors,
        retained=runtime_attachment_state.retained,
        reservation_bytes=runtime_attachment_state.reservation_bytes,
        fallback_reason_buckets=runtime_attachment_state.fallback_reason_buckets,
    )

    group_realization_state = SimpleNamespace(
        members=(
            retained_acquire_state,
            retained_state(1, reservation_bytes=2048, staged_value=False),
        )
    )
    group_reports = retained_binding_reports_for(group_realization_state)
    target_set_envelope = envelope_for_target_set(group_reports)

    publication_state = SimpleNamespace(
        projection=SimpleNamespace(
            byte_space_kind="canonical",
            lease_id="lease-1",
        ),
        binding=SimpleNamespace(size_bytes=512),
    )
    publication_envelope = envelope_for_publication(
        projection=publication_state.projection,
        binding=publication_state.binding,
    )

    mounted_source_envelope = envelope_for_mounted_source(
        canonical_index_bytes=b"canonical-index",
    )

    matrix: dict[str, tuple[RealizationResourceEnvelope, dict[str, object]]] = {
        "MaterializationPayload": (
            payload_envelope,
            {
                "backing_kind": "daemon_temporary_replica",
                "export_kind": "cpu_memfd",
                "projection_kind": "tensor_dict",
                "owner_kind": "tensor_projection_owner",
                "release_policy": (
                    "release_export_token",
                    "unload_temporary_replica",
                ),
                "temporary_replica_bytes": 8,
                "mmap_bytes": 8,
                "cpu_memfd_fd_count": 1,
                "fallback_reason_buckets": {"disk_retry": 1},
            },
        ),
        "region-backed MaterializeIntoTarget": (
            region_envelope,
            {
                "backing_kind": "caller_region_or_temporary_replica",
                "export_kind": "registered_vram_region_direct_write",
                "projection_kind": "completion",
                "owner_kind": "caller",
                "release_policy": ("unregister_target_region",),
                "direct_write_bytes": 64,
                "copy_bytes": 0,
                "fallback_reason_buckets": {"no_fallback": 0},
            },
        ),
        "OwnedBindingSlot": (
            owned_binding_envelope,
            {
                "backing_kind": "daemon_binding_value",
                "export_kind": "binding_restore",
                "projection_kind": "binding",
                "owner_kind": "binding_slot",
                "release_policy": ("close_binding", "retire_binding_value"),
                "direct_write_bytes": 48,
                "copy_bytes": 24,
                "copy_count": 3,
                "temporary_replica_bytes": 20,
                "retained_bytes": 96,
                "fallback_reason_buckets": {
                    "disk_retry": 2,
                    "not_collective": 3,
                },
            },
        ),
        "retained acquire state": (
            retained_envelope,
            {
                "backing_kind": "daemon_retained_binding",
                "export_kind": "binding_reservation",
                "projection_kind": "prefetch_handoff",
                "owner_kind": "binding_reservation_capability",
                "release_policy": (
                    "release_binding_reservation",
                    "release_group_staged_acquire",
                ),
                "retained_bytes": 4096,
            },
        ),
        "runtime attachment state": (
            runtime_attachment_envelope,
            {
                "backing_kind": "daemon_retained_binding",
                "export_kind": "fresh_retained_acquire_export",
                "projection_kind": "runtime_attachment",
                "owner_kind": "runtime_attachment",
                "release_policy": (
                    "close_runtime_attachment",
                    "release_placement_lease",
                ),
                "retained_bytes": 2048,
                "fallback_reason_buckets": {"lease_refresh": 1},
            },
        ),
        "group realization state": (
            target_set_envelope,
            {
                "backing_kind": "daemon_retained_binding_set",
                "export_kind": "binding_reservation_set",
                "projection_kind": "target_set",
                "owner_kind": "binding_reservation_capability_set",
                "release_policy": (
                    "release_binding_reservations",
                    "release_group_staged_acquire",
                ),
                "retained_bytes": 6144,
            },
        ),
        "publication state": (
            publication_envelope,
            {
                "backing_kind": "daemon_published_replica",
                "export_kind": "canonical_publication_lease",
                "projection_kind": "published_replica",
                "owner_kind": "runtime_publication",
                "release_policy": (
                    "retire_published_replica",
                    "release_publication_lease",
                ),
                "retained_bytes": 512,
            },
        ),
        "mounted source state": (
            mounted_source_envelope,
            {
                "backing_kind": "daemon_mounted_source",
                "export_kind": "metadata_attestation",
                "projection_kind": "artifact_identity",
                "owner_kind": "daemon_control_plane",
                "release_policy": ("drop_promotion_handle",),
                "retained_bytes": len(b"canonical-index"),
            },
        ),
    }
    for case, (envelope, expected_fields) in matrix.items():
        assert envelope.mutability_contract, case
        assert envelope.release_strictness == "strict", case
        assert envelope.export_lifetime_kind, case
        for field_name, expected_value in expected_fields.items():
            assert getattr(envelope, field_name) == expected_value, case


def _minimal_realization_report(
    envelope: RealizationResourceEnvelope,
    *,
    target_kind: RealizationTargetKind = "tensor_dict",
    source_selection_digest: str = "selection-digest",
    target_plan: RealizationTargetPlan | None = None,
    strategy_plan: RealizationStrategyPlan | None = None,
) -> ArtifactRealizationReport:
    return ArtifactRealizationReport(
        target_kind=target_kind,
        source_selection_digest=source_selection_digest,
        target_layout_digest=target_plan.target_layout_digest
        if target_plan is not None
        else None,
        copy_plan_digest=target_plan.copy_plan_digest
        if target_plan is not None
        else None,
        artifact_id="mi2:test:artifact",
        view_id="",
        artifact_profile="durable_artifact",
        authority_scope="daemon_mediated_durable",
        generation_hint=None,
        envelope=envelope,
        target_plan=target_plan,
        strategy_plan=strategy_plan,
    )


def test_realization_handle_rejects_missing_source_authority() -> None:
    envelope = envelope_for_tensor_dict({"a": torch.zeros(2)}, source="disk")
    report = _minimal_realization_report(
        envelope,
        source_selection_digest="",
    )

    with pytest.raises(ArtifactError) as excinfo:
        ArtifactRealizationHandle(target_kind="tensor_dict", report=report)

    assert excinfo.value.status_code == "FAILED_PRECONDITION"


def test_realization_handle_rejects_missing_release_strictness() -> None:
    envelope = replace(
        envelope_for_tensor_dict({"a": torch.zeros(2)}, source="disk"),
        release_strictness="",
    )
    report = _minimal_realization_report(envelope)

    with pytest.raises(ArtifactError) as excinfo:
        ArtifactRealizationHandle(target_kind="tensor_dict", report=report)

    assert excinfo.value.status_code == "FAILED_PRECONDITION"


def test_realization_handle_rejects_missing_mutability_contract() -> None:
    envelope = replace(
        envelope_for_tensor_dict({"a": torch.zeros(2)}, source="disk"),
        mutability_contract="",
    )
    report = _minimal_realization_report(envelope)

    with pytest.raises(ArtifactError) as excinfo:
        ArtifactRealizationHandle(target_kind="tensor_dict", report=report)

    assert excinfo.value.status_code == "FAILED_PRECONDITION"


def test_realization_handle_rejects_target_plan_admission_gaps() -> None:
    envelope = envelope_for_caller_tensors({"a": torch.zeros(2)})
    report = _minimal_realization_report(
        envelope,
        target_kind="caller_tensors",
        target_plan=RealizationTargetPlan(kind="caller_tensors"),
    )

    with pytest.raises(ArtifactError) as excinfo:
        ArtifactRealizationHandle(target_kind="caller_tensors", report=report)

    assert excinfo.value.status_code == "FAILED_PRECONDITION"


def test_realization_handle_rejects_mapped_target_without_layout_digest() -> None:
    envelope = RealizationResourceEnvelope(
        backing_kind="caller_region",
        export_kind="registered_region_direct_write",
        projection_kind="binding",
        owner_kind="caller_pid",
        release_policy=("release_external_target_storage_lease",),
        mutability_contract="caller_mutable_binding_controlled",
        release_strictness="strict",
        export_lifetime_kind="request_scoped",
    )
    report = _minimal_realization_report(
        envelope,
        target_kind="binding_adopted",
        target_plan=RealizationTargetPlan(
            kind="binding_adopted",
            binding_layout_id="binding-layout-1",
        ),
    )

    with pytest.raises(ArtifactError) as excinfo:
        ArtifactRealizationHandle(target_kind="binding_adopted", report=report)

    assert excinfo.value.status_code == "FAILED_PRECONDITION"


def test_realization_handle_rejects_observed_fallback_without_policy() -> None:
    envelope = replace(
        envelope_for_tensor_dict(
            {"a": torch.zeros(2)},
            source="disk",
            retry_reason_buckets={"disk_retry": 1},
        ),
        fallback_reason_buckets={"generic_fallback": 8},
    )
    report = _minimal_realization_report(
        envelope,
        strategy_plan=RealizationStrategyPlan(fallback_policy=""),
    )

    with pytest.raises(ArtifactError) as excinfo:
        ArtifactRealizationHandle(target_kind="tensor_dict", report=report)

    assert excinfo.value.status_code == "FAILED_PRECONDITION"


def test_tensor_dict_projection_retains_realization_owner() -> None:
    tensors = {"a": torch.zeros(2)}
    envelope = envelope_for_tensor_dict(tensors, source="disk")
    report = ArtifactRealizationReport(
        target_kind="tensor_dict",
        source_selection_digest="digest",
        target_layout_digest=None,
        copy_plan_digest=None,
        artifact_id="mi2:test:artifact",
        view_id="",
        artifact_profile="durable_artifact",
        authority_scope="daemon_mediated_durable",
        generation_hint=None,
        envelope=envelope,
    )
    handle = ArtifactRealizationHandle(
        target_kind="tensor_dict",
        report=report,
        tensor_dict_value=tensors,
    )

    projection = handle.tensor_dict()

    assert projection["a"] is tensors["a"]
    assert projection.report is report
    assert projection["a"]._tensorcast_realization_owner is handle
    assert handle.release_contract.release_policy == envelope.release_policy


def test_tensor_dict_handle_rejects_binding_lifecycle_capabilities() -> None:
    tensor = torch.zeros(2)
    envelope = envelope_for_tensor_dict({"a": tensor}, source="disk")
    report = _minimal_realization_report(envelope)
    handle = ArtifactRealizationHandle(
        target_kind="tensor_dict",
        report=report,
        tensor_dict_value={"a": tensor},
    )

    projection = handle.tensor_dict()
    assert projection["a"] is tensor
    assert projection.report.target_kind == "tensor_dict"

    unsupported: tuple[tuple[str, Callable[[], object]], ...] = (
        ("binding", handle.binding),
        ("prefetch_handoff", handle.prefetch_handoff),
        ("complete", handle.complete),
        ("attach", handle.attach),
        ("publish_replica", handle.publish_replica),
        ("promote", handle.promote),
    )
    for action, call in unsupported:
        with pytest.raises(ArtifactError) as excinfo:
            call()
        assert excinfo.value.status_code == "FAILED_PRECONDITION"
        assert action in str(excinfo.value)


def test_cpu_tensor_dict_envelope_reports_private_copy_mutability() -> None:
    envelope = envelope_for_tensor_dict({"a": torch.zeros(2)}, source="disk")

    assert envelope.export_kind == "cpu_memfd"
    assert envelope.mutability_contract == "read_mostly_private_copy"


def test_release_contract_runs_actions_once_from_projection_close() -> None:
    calls: list[str] = []
    tensors = {"a": torch.zeros(2)}
    envelope = envelope_for_tensor_dict(tensors, source="disk")
    report = ArtifactRealizationReport(
        target_kind="tensor_dict",
        source_selection_digest="digest",
        target_layout_digest=None,
        copy_plan_digest=None,
        artifact_id="mi2:test:artifact",
        view_id="",
        artifact_profile="durable_artifact",
        authority_scope="daemon_mediated_durable",
        generation_hint=None,
        envelope=envelope,
    )
    handle = ArtifactRealizationHandle(
        target_kind="tensor_dict",
        report=report,
        tensor_dict_value=tensors,
        release_contract=release_contract_for(
            envelope,
            lambda: calls.append("release_export_token"),
            lambda: calls.append("unload_temporary_replica"),
        ),
    )

    projection = handle.tensor_dict()
    projection.close()
    handle.close()

    assert calls == ["release_export_token", "unload_temporary_replica"]
    assert handle.release_contract.released is True


@pytest.mark.parametrize(
    "case",
    [
        "raw_tensor_escape",
        "caller_target_cleanup",
        "owned_binding_cleanup",
        "retained_acquire_cleanup",
        "runtime_workflow_cleanup",
    ],
)
def test_release_contract_lifecycle_matrix_runs_policy_actions_once(
    case: str,
) -> None:
    calls: list[str] = []
    tensors = {"a": torch.zeros(2)}
    if case == "raw_tensor_escape":
        target_kind = "tensor_dict"
        envelope = envelope_for_tensor_dict(tensors, source="disk")
        tensor_dict_value = tensors
        expected_policy = ("release_export_token", "unload_temporary_replica")
    elif case == "caller_target_cleanup":
        target_kind = "caller_tensors"
        envelope = envelope_for_caller_tensors(tensors)
        tensor_dict_value = None
        expected_policy = (
            "unregister_target_region",
            "unload_temporary_replica_on_fallback",
        )
    elif case == "owned_binding_cleanup":
        target_kind = "binding_owned"
        binding = SimpleNamespace(
            last_materialization_diagnostics={"total_bytes": 8},
            last_execution_diagnostics=None,
            last_source_bound_plan_diagnostics=None,
        )
        envelope = envelope_for_binding(binding, target_kind="binding_owned")
        tensor_dict_value = None
        expected_policy = ("close_binding", "retire_binding_value")
    elif case == "retained_acquire_cleanup":
        target_kind = "runtime_attachment"
        envelope = envelope_for_runtime_attachment(
            tensors,
            retained=True,
            reservation_bytes=4096,
        )
        tensor_dict_value = None
        expected_policy = ("close_runtime_attachment", "release_placement_lease")
    else:
        target_kind = "runtime_attachment"
        envelope = envelope_for_runtime_attachment(tensors, retained=False)
        tensor_dict_value = None
        expected_policy = ("close_runtime_attachment", "close_binding")
    assert envelope.release_policy == expected_policy
    report = ArtifactRealizationReport(
        target_kind=target_kind,
        source_selection_digest="digest",
        target_layout_digest=None,
        copy_plan_digest=None,
        artifact_id="mi2:test:artifact",
        view_id="",
        artifact_profile="durable_artifact",
        authority_scope="daemon_mediated_durable",
        generation_hint=None,
        envelope=envelope,
    )
    actions = tuple(
        lambda action=action: calls.append(action) for action in envelope.release_policy
    )
    handle = ArtifactRealizationHandle(
        target_kind=target_kind,
        report=report,
        tensor_dict_value=tensor_dict_value,
        release_contract=release_contract_for(envelope, *actions),
    )

    if case == "raw_tensor_escape":
        raw_tensor = handle.tensor_dict()["a"]
        owner = raw_tensor._tensorcast_realization_owner
        assert owner is handle
        owner.close()
    else:
        handle.close()
    handle.close()

    assert calls == list(envelope.release_policy)
    assert handle.release_contract.released is True


def _tensor_dict_projection_for_mutability_test(
    tensors: dict[str, torch.Tensor],
) -> Any:
    envelope = envelope_for_tensor_dict(tensors, source="disk")
    report = ArtifactRealizationReport(
        target_kind="tensor_dict",
        source_selection_digest="digest",
        target_layout_digest=None,
        copy_plan_digest=None,
        artifact_id="mi2:test:artifact",
        view_id="",
        artifact_profile="durable_artifact",
        authority_scope="daemon_mediated_durable",
        generation_hint=None,
        envelope=envelope,
    )
    return ArtifactRealizationHandle(
        target_kind="tensor_dict",
        report=report,
        tensor_dict_value=tensors,
    ).tensor_dict()


def _projection_ior(projection: Any) -> None:
    projection.__ior__({"b": torch.ones(1)})


@pytest.mark.parametrize(
    "mutation",
    [
        lambda projection: projection.__setitem__("b", torch.ones(1)),
        lambda projection: projection.__delitem__("a"),
        lambda projection: projection.clear(),
        lambda projection: projection.pop("a"),
        lambda projection: projection.popitem(),
        lambda projection: projection.setdefault("b", torch.ones(1)),
        lambda projection: projection.update({"b": torch.ones(1)}),
        _projection_ior,
    ],
)
def test_tensor_dict_projection_rejects_mapping_mutations(
    mutation: Callable[[Any], None],
) -> None:
    tensors = {"a": torch.zeros(2)}
    projection = _tensor_dict_projection_for_mutability_test(tensors)

    with pytest.raises(ArtifactError) as excinfo:
        mutation(projection)

    assert excinfo.value.status_code == "FAILED_PRECONDITION"
    assert "read-only" in str(excinfo.value)
    assert projection["a"] is tensors["a"]


class _FakeBinding:
    def __init__(self) -> None:
        self.published = False
        self.promoted = False
        self.closed = False

    def publish_replica(self, *, marker: str) -> str:
        self.published = True
        return f"published:{marker}"

    def promote_current_value(self, *, binding_value_id: str) -> str:
        self.promoted = True
        return f"promoted:{binding_value_id}"

    def close(self) -> None:
        self.closed = True


def test_binding_handle_facade_delegates_publication_promotion_and_close() -> None:
    binding = _FakeBinding()
    envelope = RealizationResourceEnvelope(
        backing_kind="daemon_binding_value",
        export_kind="binding_restore",
        projection_kind="binding",
        owner_kind="binding_slot",
        release_policy=("close_binding",),
        mutability_contract="binding_controlled",
        release_strictness="strict",
        export_lifetime_kind="token_backed",
    )
    report = ArtifactRealizationReport(
        target_kind="binding_owned",
        source_selection_digest="digest",
        target_layout_digest=None,
        copy_plan_digest=None,
        artifact_id="mi2:test:artifact",
        view_id="",
        artifact_profile="durable_artifact",
        authority_scope="daemon_mediated_durable",
        generation_hint=None,
        envelope=envelope,
    )
    handle = ArtifactRealizationHandle(
        target_kind="binding_owned",
        report=report,
        binding_value=binding,
        close_fn=binding.close,
    )

    assert handle.binding() is binding
    assert handle.publish_replica(marker="x") == "published:x"
    assert handle.promote(binding_value_id="v1") == "promoted:v1"
    handle.close()
    handle.close()

    assert binding.published is True
    assert binding.promoted is True
    assert binding.closed is True
    assert handle.release_contract.release_policy == envelope.release_policy
    assert handle.release_contract.released is True


def test_model_runtime_spec_and_attach_facade_delegate_explicit_adapter() -> None:
    spec = ArtifactRealizationSpec.model_runtime(
        framework="vllm",
        device="cuda:0",
        topology={"tp": 2},
        member={"rank": 1},
        adapter_version="adapter-v1",
        runtime_abi_version="abi-v1",
    )
    envelope = RealizationResourceEnvelope(
        backing_kind="daemon_binding_value",
        export_kind="runtime_attachment",
        projection_kind="model_runtime",
        owner_kind="runtime_attachment",
        release_policy=("close_runtime_attachment",),
        mutability_contract="runtime_adapter_owned",
        release_strictness="strict",
        export_lifetime_kind="runtime_attachment",
    )
    report = ArtifactRealizationReport(
        target_kind="model_runtime",
        source_selection_digest="digest",
        target_layout_digest="runtime-layout",
        copy_plan_digest=None,
        artifact_id="mi2:test:artifact",
        view_id="",
        artifact_profile="durable_artifact",
        authority_scope="daemon_mediated_durable",
        generation_hint=None,
        envelope=envelope,
    )
    handle = ArtifactRealizationHandle(
        target_kind="model_runtime",
        report=report,
        attach_fn=lambda *, adapter: f"attached:{adapter}",
    )

    assert spec.target_kind == "model_runtime"
    assert spec.framework == "vllm"
    assert spec.topology == {"tp": 2}
    assert spec.member == {"rank": 1}
    assert spec.adapter_version == "adapter-v1"
    assert spec.runtime_abi_version == "abi-v1"
    assert handle.attach(adapter="runtime-adapter") == "attached:runtime-adapter"


def test_model_runtime_report_wraps_runtime_attachment_report() -> None:
    spec = ArtifactRealizationSpec.model_runtime(
        framework="vllm",
        device="cuda:0",
        topology={"tp": 2},
        member={"rank": 1},
        adapter_version="adapter-v1",
        runtime_abi_version="abi-v1",
    )
    selection = resolve_artifact_selection(
        artifact_id="mi2:test:serving",
        canonical_index_bytes=_canonical_index_bytes(),
        tensor_names=("a",),
        artifact_profile="runtime_artifact",
        authority_scope="daemon_mediated_runtime_attachment",
    )
    target_plan = RealizationTargetPlan(
        kind="runtime_attachment",
        device="cuda:0",
        target_layout_digest="binding-layout:layout-1",
        binding_layout_id="layout-1",
    )
    envelope = envelope_for_runtime_attachment(
        {"a": torch.zeros(2, dtype=torch.float32)},
        retained=False,
    )
    runtime_report = report_for_runtime_attachment(
        selection=selection,
        target_plan=target_plan,
        envelope=envelope,
        risk_labels=("runtime_attachment",),
    )

    report = model_runtime_report_for(
        spec=spec,
        runtime_attachment_report=runtime_report,
    )

    assert report.target_kind == "model_runtime"
    assert report.operation_backend == "model_runtime_attachment"
    assert report.envelope.projection_kind == "model_runtime"
    assert report.source_selection_digest == runtime_report.source_selection_digest
    assert report.model_runtime is not None
    assert report.model_runtime.framework == "vllm"
    assert report.model_runtime.device == "cuda:0"
    assert report.model_runtime.adapter_version == "adapter-v1"
    assert report.model_runtime.runtime_attachment_target_kind == "runtime_attachment"
    assert report.representation_admission is not None
    assert report.representation_admission.representation_contract == "model_runtime"
    assert report.lifecycle_plan is not None
    assert report.lifecycle_plan.capability == "model_runtime"
    assert report.lifecycle_plan.release_policy == envelope.release_policy
    assert "model_runtime" in report.risk_labels
    assert "framework:vllm" in report.risk_labels


def test_model_runtime_spec_requires_framework() -> None:
    with pytest.raises(ArtifactError) as excinfo:
        ArtifactRealizationSpec.model_runtime(framework="")

    assert excinfo.value.status_code == "INVALID_ARGUMENT"


def test_publication_spec_and_handle_facade_own_release_contract() -> None:
    released: list[str] = []
    binding_value_ref = SimpleNamespace(binding_value_id="value-1")
    projection = SimpleNamespace(
        state="published",
        artifact_ref="mi2:test:serving",
        operation_id="publish-op-1",
        replica_id="replica-1",
        lease_id="lease-1",
        device_uuid="GPU-0",
        owner_pid=123,
        binding_layout_id="layout-1",
        generation="generation-1",
        reason=None,
        byte_space_kind="cuda",
        byte_space_id="0",
        binding_value_ref=binding_value_ref,
    )
    binding = SimpleNamespace(
        binding_id="binding-1",
        binding_layout_id="layout-1",
        current_value=SimpleNamespace(
            binding_id="binding-1",
            binding_layout_id="layout-1",
            binding_value_id="value-1",
            seal_generation=1,
            is_artifact_backed=True,
            is_published=True,
            serving_artifact_id="mi2:test:serving",
        ),
        publish_replica=lambda: None,
        size_bytes=1024,
    )
    spec = ArtifactRealizationSpec._publication(target=projection, timeout_s=5)
    target_plan = RealizationTargetPlan(
        kind=spec.target_kind,
        target_layout_digest="layout-1",
        binding_layout_id="layout-1",
    )
    envelope = envelope_for_publication(projection=projection, binding=binding)
    report = report_for_publication(
        artifact_id="mi2:test:serving",
        source_selection_digest="publication-generation",
        target_plan=target_plan,
        envelope=envelope,
        projection=projection,
        binding_handle=binding,
    )
    contract = release_contract_for(
        envelope,
        lambda: released.append("release_publication"),
    )
    handle = ArtifactRealizationHandle(
        target_kind=spec.target_kind,
        report=report,
        binding_value=binding,
        release_contract=contract,
    )

    assert spec.target_kind == "publication"
    assert spec.publish is True
    assert spec.target is projection
    assert spec.timeout_s == 5
    assert report.target_kind == "publication"
    assert report.publication is not None
    assert report.publication.lease_id == "lease-1"
    assert report.publishability is not None
    assert report.publishability.publish_requested is True
    assert report.representation_admission is not None
    assert report.representation_admission.representation_contract == "publication"
    assert report.lifecycle_plan is not None
    assert report.lifecycle_plan.capability == "publication"
    assert report.lifecycle_plan.publishable is True
    assert report.lifecycle_plan.release_policy == envelope.release_policy
    assert handle.binding() is binding
    handle.close()
    handle.close()

    assert released == ["release_publication"]
    assert handle.release_contract.released is True


class _FakeCurrentBindingValue:
    binding_id = "binding-1"
    binding_layout_id = "bl1:test"
    binding_value_id = "value-1"
    seal_generation = 3
    source_artifact_id = "mi2:test:artifact"
    is_artifact_backed = True
    verification_state = 7
    verification_job_id = "verify-1"
    source_artifact_ref = "mi2:test:artifact"
    local_serving_ref = None
    serving_artifact_id = "mi2:test:serving"
    verification_failure_reason = None
    is_published = True


class _FakeExecutionDiagnostics:
    collective_requested = True
    collective_acknowledged = True
    collective_used = True
    dominant_executor = "SourceBoundMixedExecutor"
    direct_write_supported = True
    actual_local_typed_bytes = 64
    actual_collective_committed_bytes = 32
    actual_generic_backend_bytes = 8
    fallback_bytes = 8
    residual_bytes = 4
    collective_unique_source_bytes = 96
    collective_peer_transfer_bytes = 48
    collective_peak_temporary_bytes = 16
    collective_batch_count = 2
    collective_dedup_saving_bytes = 32
    collective_skip_reason = "partial_generic_residual"


class _FakeSourceBoundPlanDiagnostics:
    execution_plan_kind = "collective_first_mixed"
    planned_collective_candidate_bytes = 128
    planned_collective_admitted_bytes = 96
    planned_local_typed_bytes = 64
    planned_non_admitted_typed_bytes = 12
    planned_generic_residual_bytes = 4
    collective_lowered_bytes = 32
    planner_reject_reason_buckets = {"not_collective": 2}
    planner_version = "source_bound_collective_first.v4"
    plan_hash = "mh:test-plan"
    estimated_collective_peak_temporary_bytes = 24
    estimated_collective_batch_bytes = 16
    estimated_collective_dedup_saving_bytes = 64


class _FakeReportedBinding:
    binding_id = "binding-1"
    binding_layout_id = "bl1:test"
    current_value = _FakeCurrentBindingValue()
    staged_value = None
    last_execution_diagnostics = _FakeExecutionDiagnostics()
    last_source_bound_plan_diagnostics = _FakeSourceBoundPlanDiagnostics()
    last_materialization_diagnostics = {
        "source": "p2p",
        "operation_id": "op-1",
        "total_bytes": 128,
        "retry_reason_buckets": {"p2p_unavailable": 1},
    }

    def publish_replica(self) -> None:
        return None


def test_binding_materialization_diagnostics_use_realization_kernel_helper() -> None:
    layout = SimpleNamespace(
        target_index_bytes=b'{"tensor":[0,4096,[1024],[1],"torch.float32",0]}'
    )

    diagnostics = binding_materialization_diagnostics_from_response(
        SimpleNamespace(
            source=store_daemon_pb2.MATERIALIZATION_SOURCE_P2P,
            selected_source_replica_id="replica-source-1",
            selected_source_transport_id="transport-1",
        ),
        layout=layout,
    )

    assert diagnostics == {
        "source": "p2p",
        "source_code": int(store_daemon_pb2.MATERIALIZATION_SOURCE_P2P),
        "total_bytes": 4096,
        "retry_attempts": 1,
        "retry_reason_buckets": {},
        "replica_id": "replica-source-1",
        "transport_id": "transport-1",
    }


def test_binding_envelope_and_report_capture_identity_diagnostics() -> None:
    binding = _FakeReportedBinding()
    selection = resolve_artifact_selection(
        artifact_id="mi2:test:artifact",
        canonical_index_bytes=_canonical_index_bytes(),
    )
    target_plan = RealizationTargetPlan(
        kind="binding_owned",
        device="cuda:0",
        target_layout_digest="binding-layout:bl1:test",
        binding_layout_id="bl1:test",
    )

    envelope = envelope_for_binding(
        binding,
        target_kind="binding_owned",
        publish_requested=True,
    )
    envelope.validate_for_target(target_plan)
    report = report_for_binding_realization(
        target_kind="binding_owned",
        selection=selection,
        target_plan=target_plan,
        binding=binding,
        envelope=envelope,
        publish_requested=True,
        risk_labels=("authority", "identity", "lifecycle"),
    )

    assert envelope.retained_bytes == 128
    assert envelope.direct_write_bytes == 96
    assert envelope.copy_bytes == 20
    assert envelope.copy_count == 3
    assert envelope.temporary_replica_bytes == 16
    assert envelope.fallback_reason_buckets == {
        "not_collective": 2,
        "p2p_unavailable": 1,
    }
    assert envelope.release_policy == (
        "close_binding",
        "retire_binding_value",
        "release_publish_lease",
    )
    assert report.source == "p2p"
    assert report.operation_id == "op-1"
    assert report.operation_backend == "daemon_binding"
    assert report.binding is not None
    assert report.binding.binding_id == "binding-1"
    assert report.binding.binding_layout_id == "bl1:test"
    assert report.binding.binding_value_id == "value-1"
    assert report.binding.value_state == "current"
    assert report.view_subset_hash == selection.view_subset_hash.hex()
    assert report.logical_layout_hash == selection.logical_layout_hash.hex()
    assert report.selection_hash == selection.selection_hash.hex()
    assert report.binding.publication_eligible is True
    assert report.binding.publish_requested is True
    assert report.binding.published is True
    assert report.publishability is not None
    assert report.publishability.publishable is True
    assert report.publishability.publish_requested is True
    assert report.publishability.publication_eligible is True
    assert report.publishability.published is True
    assert report.strategy_plan is not None
    assert report.strategy_plan.fallback_policy == "generic_fallback"
    assert report.strategy_plan.collective_policy == "collective_used"
    assert report.representation_admission is not None
    assert report.representation_admission.representation_contract == "binding_layout"
    assert report.representation_admission.transform_required is False
    assert report.lifecycle_plan is not None
    assert report.lifecycle_plan.capability == "binding_owned"
    assert report.lifecycle_plan.publishable is True
    assert report.lifecycle_plan.release_policy == envelope.release_policy
    assert report.execution_commit is not None
    assert report.execution_commit.actual_executor_path == "mixed_collective"
    assert report.execution_commit.dominant_executor == "SourceBoundMixedExecutor"
    assert report.execution_commit.source == "p2p"
    assert report.execution_commit.requested_bytes == 128
    assert report.execution_commit.committed_bytes == 104
    assert report.execution_commit.direct_write_bytes == 96
    assert report.execution_commit.fallback_bytes == 8
    assert report.execution_commit.residual_bytes == 4
    assert report.execution_commit.collective_peer_transfer_bytes == 48
    assert report.execution_commit.collective_peak_temporary_bytes == 16
    assert report.execution_commit.collective_batch_count == 2
    assert report.execution_commit.collective_dedup_saving_bytes == 32
    assert report.execution_commit.collective_skip_reason == "partial_generic_residual"
    assert report.execution_commit.direct_write_supported is True
    assert report.execution_commit.collective_requested is True
    assert report.execution_commit.collective_acknowledged is True
    assert report.execution_commit.collective_used is True
    assert report.execution_commit.execution_plan_kind == "collective_first_mixed"
    assert report.execution_commit.plan_hash == "mh:test-plan"
    assert report.execution_commit.lane_allocation_bytes == {
        "collective_candidate": 128,
        "collective_admitted": 96,
        "local_typed": 64,
        "non_admitted_typed": 12,
        "generic_residual": 4,
        "collective_lowered": 32,
    }
    assert report.execution_commit.committed_range_bytes == {
        "collective": 32,
        "local_typed": 64,
        "generic_backend": 8,
    }
    assert report.execution_commit.residual_fallback_range_bytes == {
        "fallback": 8,
        "residual": 4,
        "planned_generic_residual": 4,
    }
    assert report.execution_commit.planner_reject_reason_buckets == {
        "not_collective": 2
    }
    report_dict = artifact_realization_report_to_dict(report)
    execution_dict = report_dict["execution_commit"]
    assert isinstance(execution_dict, dict)
    assert execution_dict["actual_executor_path"] == "mixed_collective"
    assert execution_dict["lane_allocation_bytes"] == {
        "collective_candidate": 128,
        "collective_admitted": 96,
        "local_typed": 64,
        "non_admitted_typed": 12,
        "generic_residual": 4,
        "collective_lowered": 32,
    }
    assert execution_dict["committed_range_bytes"] == {
        "collective": 32,
        "local_typed": 64,
        "generic_backend": 8,
    }
    assert execution_dict["residual_fallback_range_bytes"] == {
        "fallback": 8,
        "residual": 4,
        "planned_generic_residual": 4,
    }
    assert execution_dict["planner_reject_reason_buckets"] == {"not_collective": 2}

    profile_payload = artifact_realization_profile_payload(report)
    assert profile_payload["logical_layout_hash"] == selection.logical_layout_hash.hex()
    assert profile_payload["selection_hash"] == selection.selection_hash.hex()
    assert profile_payload["execution_actual_executor_path"] == "mixed_collective"
    assert profile_payload["execution_residual_bytes"] == 4
    assert profile_payload["execution_plan_kind"] == "collective_first_mixed"
    assert profile_payload["execution_lane_allocation_bytes"] == {
        "collective_candidate": 128,
        "collective_admitted": 96,
        "local_typed": 64,
        "non_admitted_typed": 12,
        "generic_residual": 4,
        "collective_lowered": 32,
    }
    assert profile_payload["execution_committed_range_bytes"] == {
        "collective": 32,
        "local_typed": 64,
        "generic_backend": 8,
    }
    assert profile_payload["execution_residual_fallback_range_bytes"] == {
        "fallback": 8,
        "residual": 4,
        "planned_generic_residual": 4,
    }
    assert profile_payload["execution_planner_reject_reason_buckets"] == {
        "not_collective": 2
    }
    assert report.execution_diagnostics is binding.last_execution_diagnostics
    assert (
        report.materialization_diagnostics == binding.last_materialization_diagnostics
    )
    assert (
        report.source_bound_plan_diagnostics
        is binding.last_source_bound_plan_diagnostics
    )


def test_binding_target_envelope_requires_binding_layout_id() -> None:
    envelope = envelope_for_binding(
        _FakeReportedBinding(),
        target_kind="binding_owned",
    )

    with pytest.raises(ArtifactError) as excinfo:
        envelope.validate_for_target(RealizationTargetPlan(kind="binding_owned"))

    assert excinfo.value.status_code == "FAILED_PRECONDITION"


def test_local_ready_pending_verification_report_records_admission_state() -> None:
    pending_verification = store_daemon_pb2.BINDING_VALUE_VERIFICATION_STATE_PENDING
    binding = SimpleNamespace(
        binding_id="binding-local-ready",
        binding_layout_id="layout-local-ready",
        current_value=SimpleNamespace(
            binding_id="binding-local-ready",
            binding_layout_id="layout-local-ready",
            binding_value_id="value-local-ready",
            seal_generation=2,
            source_artifact_id="mi2:test:local-ready-source",
            is_artifact_backed=True,
            verification_state=pending_verification,
            verification_job_id="verify-pending",
            source_artifact_ref="mi2:test:local-ready-source",
            local_serving_ref="local-ready:binding-local-ready:value-local-ready",
            serving_artifact_id=None,
            verification_failure_reason=None,
            is_published=False,
        ),
        staged_value=None,
        last_execution_diagnostics=None,
        last_source_bound_plan_diagnostics=None,
        last_materialization_diagnostics={
            "source": "local_ready",
            "operation_id": "op-local-ready",
            "total_bytes": 32,
        },
    )
    selection = resolve_artifact_selection(
        artifact_id="mi2:test:local-ready-source",
        canonical_index_bytes=_canonical_index_bytes(),
        artifact_profile="runtime_local_ready",
        authority_scope="daemon_mediated_local_ready_runtime_attachment",
    )
    target_plan = RealizationTargetPlan(
        kind="binding_owned",
        target_layout_digest="binding-layout:layout-local-ready",
        binding_layout_id="layout-local-ready",
    )
    envelope = envelope_for_binding(binding, target_kind="binding_owned")

    envelope.validate_for_target(target_plan)
    report = report_for_binding_realization(
        target_kind="binding_owned",
        selection=selection,
        target_plan=target_plan,
        binding=binding,
        envelope=envelope,
        risk_labels=("local_ready", "pending_verification"),
    )

    assert report.artifact_profile == "runtime_local_ready"
    assert report.authority_scope == "daemon_mediated_local_ready_runtime_attachment"
    assert report.binding is not None
    assert report.binding.verification_state == pending_verification
    assert report.binding.verification_job_id == "verify-pending"
    assert report.binding.local_serving_ref == (
        "local-ready:binding-local-ready:value-local-ready"
    )
    assert report.source == "local_ready"
    assert report.operation_id == "op-local-ready"
    assert "local_ready" in report.risk_labels
    assert "pending_verification" in report.risk_labels


def test_runtime_attachment_envelope_and_report_capture_release_contract() -> None:
    tensors = {"a": torch.zeros(2, dtype=torch.float32)}
    selection = resolve_artifact_selection(
        artifact_id="mi2:test:serving",
        canonical_index_bytes=_canonical_index_bytes(),
        tensor_names=("a",),
        artifact_profile="runtime_artifact",
        authority_scope="daemon_mediated_runtime_attachment",
    )
    target_plan = RealizationTargetPlan(
        kind="runtime_attachment",
        device="cuda:0",
        target_layout_digest="binding-layout:bl1:test",
        binding_layout_id="bl1:test",
    )

    envelope = envelope_for_runtime_attachment(tensors, retained=False)
    envelope.validate_for_target(target_plan)
    report = report_for_runtime_attachment(
        selection=selection,
        target_plan=target_plan,
        envelope=envelope,
        materialization_diagnostics={"source": "disk", "total_bytes": 128},
        execution_diagnostics=_FakeExecutionDiagnostics(),
        source_bound_plan_diagnostics=_FakeSourceBoundPlanDiagnostics(),
        risk_labels=("runtime_attachment",),
    )

    assert envelope.backing_kind == "daemon_binding_value"
    assert envelope.projection_kind == "runtime_attachment"
    assert envelope.owner_kind == "runtime_attachment"
    assert envelope.release_policy == (
        "close_runtime_attachment",
        "close_binding",
    )
    assert report.target_kind == "runtime_attachment"
    assert report.artifact_id == "mi2:test:serving"
    assert report.target_layout_digest == "binding-layout:bl1:test"
    assert report.operation_backend == "runtime_attachment"
    assert report.publishability is not None
    assert report.publishability.publishable is False
    assert report.strategy_plan is not None
    assert report.strategy_plan.fallback_policy == "generic_fallback"
    assert report.strategy_plan.collective_policy == "collective_used"
    assert report.representation_admission is not None
    assert (
        report.representation_admission.representation_contract == "runtime_attachment"
    )
    assert report.lifecycle_plan is not None
    assert report.lifecycle_plan.capability == "runtime_attachment"
    assert report.lifecycle_plan.release_policy == envelope.release_policy
    assert report.execution_commit is not None
    assert report.execution_commit.source == "disk"
    assert report.execution_commit.actual_executor_path == "mixed_collective"
    assert report.execution_commit.planner_reject_reason_buckets == {
        "not_collective": 2
    }
    assert report.materialization_diagnostics == {
        "source": "disk",
        "total_bytes": 128,
    }


def test_runtime_attachment_resolved_report_projects_execution_topology_options() -> (
    None
):
    tensors = {"a": torch.zeros(2, dtype=torch.float32)}
    options = _execution_topology_options()

    report = tc_lifecycle._runtime_attachment_report_for_resolved(
        resolved=SimpleNamespace(artifact_ref="mi2:test:serving"),
        tensors=tensors,
        binding_handle=SimpleNamespace(binding_layout_id="bl1:test"),
        target_device="cuda:0",
        tensor_schema_hash="schema-test",
        execution_diagnostics=_FakeExecutionDiagnostics(),
        materialization_diagnostics={"source": "disk", "total_bytes": 128},
        options=options,
    )

    assert report.target_kind == "runtime_attachment"
    assert report.operation_backend == "runtime_attachment"
    assert report.strategy_plan is not None
    assert report.strategy_plan.source_policy["preference"] == "prefer_disk"
    assert report.strategy_plan.source_policy["allow_p2p"] is False
    assert report.strategy_plan.source_policy["allow_disk"] is True
    assert report.strategy_plan.source_policy["topology_collective_policy"] == (
        "collective_first"
    )
    assert report.strategy_plan.source_policy["source_locality"] == (
        "shared_source"
    )
    assert report.strategy_plan.source_policy["source_sharing_domain"] == "node-a"
    assert report.strategy_plan.collective_policy == "collective_used"


def test_runtime_attachment_envelope_requires_target_layout_digest() -> None:
    envelope = envelope_for_runtime_attachment(
        {"a": torch.zeros(1, dtype=torch.float32)},
        retained=True,
        reservation_bytes=4,
    )

    with pytest.raises(ArtifactError) as excinfo:
        envelope.validate_for_target(RealizationTargetPlan(kind="runtime_attachment"))

    assert excinfo.value.status_code == "FAILED_PRECONDITION"


def _target_set_retained_member(member_index: int) -> RealizationRetainedBindingReport:
    suffix = str(member_index)
    return RealizationRetainedBindingReport(
        binding_id=f"binding-{suffix}",
        binding_layout_id=f"layout-{suffix}",
        binding_value_id=f"value-{suffix}",
        seal_generation=1,
        local_serving_ref=f"local-{suffix}",
        daemon_id="daemon-1",
        daemon_session_id="sess-1",
        device_uuid=f"GPU-{suffix}",
        member_id=f"member-{suffix}",
        member_index=member_index,
        member_count=2,
        member_group_id="group-1",
        reservation_bytes=1024,
        reservation_capability_id=f"cap-{suffix}",
        reservation_scope_digest=f"scope-{suffix}",
        readiness="runtime_local_ready",
        verification_state="local_only",
    )


def test_retained_binding_report_captures_capability_expiry() -> None:
    binding_ref = SimpleNamespace(
        binding_id="binding-0",
        binding_layout_id="layout-0",
        binding_value_id="value-0",
        seal_generation=1,
    )
    member = SimpleNamespace(
        member_id="member-0",
        member_index=0,
        member_count=1,
        group_id="group-1",
    )
    result = SimpleNamespace(
        binding_value_ref=binding_ref,
        reservation_capability=SimpleNamespace(
            capability_id="cap-0",
            scope_digest="scope-0",
            expires_at_ms=4_102_444_800_000,
        ),
        member=member,
        daemon_id="daemon-1",
        daemon_session_id="session-1",
        device_uuid="GPU-0",
        reservation_bytes=4096,
        readiness="runtime_local_ready",
        verification_state="local_only",
        expires_at_ms=4_102_444_800_000,
    )

    (report,) = retained_binding_reports_for(result)

    assert report.reservation_capability_id == "cap-0"
    assert report.reservation_capability_expires_at_ms == 4_102_444_800_000
    assert report.expires_at_ms == 4_102_444_800_000


def test_target_set_report_groups_retained_member_facts() -> None:
    retained = (
        RealizationRetainedBindingReport(
            binding_id="binding-0",
            binding_layout_id="layout-0",
            binding_value_id="value-0",
            seal_generation=1,
            local_serving_ref="local-0",
            daemon_id="daemon-1",
            daemon_session_id="sess-1",
            device_uuid="GPU-0",
            member_id="member-0",
            member_index=0,
            member_count=2,
            member_group_id="group-1",
            reservation_bytes=1024,
            reservation_capability_id="cap-0",
            reservation_scope_digest="scope-0",
            readiness="runtime_local_ready",
            verification_state="local_only",
            staged_value=True,
            group_realization_transaction_id="txn-1",
            group_realization_version_set_id="vs-1",
            group_realization_part_id="part-0",
        ),
        RealizationRetainedBindingReport(
            binding_id="binding-1",
            binding_layout_id="layout-1",
            binding_value_id="value-1",
            seal_generation=1,
            local_serving_ref="local-1",
            daemon_id="daemon-1",
            daemon_session_id="sess-1",
            device_uuid="GPU-1",
            member_id="member-1",
            member_index=1,
            member_count=2,
            member_group_id="group-1",
            reservation_bytes=2048,
            reservation_capability_id="cap-1",
            reservation_scope_digest="scope-1",
            readiness="runtime_local_ready",
            verification_state="local_only",
            staged_value=True,
            group_realization_transaction_id="txn-1",
            group_realization_version_set_id="vs-1",
            group_realization_part_id="part-1",
        ),
    )
    source = SimpleNamespace(
        source_kind="checkpoint_artifact",
        artifact_selection_digest="source-selection",
        source_artifact_ref="mi2:source",
    )
    target = SimpleNamespace(
        runtime="vllm",
        group_id="group-1",
        source=source,
        topology=SimpleNamespace(schema_topology_digest="topology-digest"),
        members=(
            SimpleNamespace(
                member=SimpleNamespace(member_id="member-0"),
                source=source,
                resolved_layout=SimpleNamespace(target_layout_hash="target-layout-0"),
            ),
            SimpleNamespace(
                member=SimpleNamespace(member_id="member-1"),
                source=source,
                resolved_layout=SimpleNamespace(target_layout_hash="target-layout-1"),
            ),
        ),
    )
    result = SimpleNamespace(
        runtime="vllm",
        group_id="group-1",
        topology=target.topology,
        readiness="runtime_local_ready",
        partial=False,
        member_failures=(),
    )

    envelope = envelope_for_target_set(retained)
    envelope.validate_for_target(
        RealizationTargetPlan(
            kind="target_set",
            target_layout_digest="group-target-layout",
            member_count=2,
        )
    )
    report = target_set_report_for_retained_bindings(
        retained,
        result=result,
        target=target,
        source_selection_digest="fallback-selection",
    )

    assert envelope.backing_kind == "daemon_retained_binding_set"
    assert envelope.export_kind == "binding_reservation_set"
    assert envelope.projection_kind == "target_set"
    assert envelope.retained_bytes == 3072
    assert envelope.release_policy == (
        "release_binding_reservations",
        "release_group_staged_acquire",
    )
    assert report.group_id == "group-1"
    assert report.runtime == "vllm"
    assert report.source_selection_mode == "same_selection"
    assert report.member_count == 2
    assert report.successful_member_count == 2
    assert report.ready_member_count == 2
    assert report.staged_member_count == 2
    assert report.same_daemon_session is True
    assert report.group_realization_transaction_ids == ("txn-1",)
    assert report.group_realization_version_set_ids == ("vs-1",)
    assert [member.target_layout_digest for member in report.members] == [
        "target-layout-0",
        "target-layout-1",
    ]
    assert [member.source_artifact_ref for member in report.members] == [
        "mi2:source",
        "mi2:source",
    ]


def test_reports_share_core_realization_fields_across_targets() -> None:
    selection = resolve_artifact_selection(
        artifact_id="mi2:test:artifact",
        canonical_index_bytes=_canonical_index_bytes(),
        tensor_names=("a",),
    )
    tensors = {"a": torch.zeros(2, dtype=torch.float32)}
    tensor_target = RealizationTargetPlan(kind="tensor_dict", device="cpu")
    tensor_envelope = envelope_for_tensor_dict(tensors, source="disk")
    tensor_publishability = publishability_report_for()
    tensor_report = ArtifactRealizationReport(
        target_kind="tensor_dict",
        source_selection_digest=selection.source_selection_digest,
        target_layout_digest=tensor_target.target_layout_digest,
        copy_plan_digest=tensor_target.copy_plan_digest,
        artifact_id=selection.artifact_id,
        view_id=selection.view_id,
        artifact_profile=selection.artifact_profile,
        authority_scope=selection.authority_scope,
        generation_hint=selection.generation_hint,
        envelope=tensor_envelope,
        target_plan=tensor_target,
        strategy_plan=strategy_plan_for_execution(envelope=tensor_envelope),
        representation_admission=representation_admission_for_target(tensor_target),
        lifecycle_plan=lifecycle_plan_for_envelope(
            tensor_target,
            tensor_envelope,
            publishability=tensor_publishability,
        ),
        operation_backend="daemon_materialization",
        risk_labels=risk_labels_for_target(
            tensor_target,
            tensor_envelope,
            source_selection_digest=selection.source_selection_digest,
        ),
        publishability=tensor_publishability,
    )

    binding = _FakeReportedBinding()
    binding_target = RealizationTargetPlan(
        kind="binding_owned",
        device="cuda:0",
        target_layout_digest="binding-layout:bl1:test",
        binding_layout_id="bl1:test",
    )
    binding_envelope = envelope_for_binding(binding, target_kind="binding_owned")
    binding_report = report_for_binding_realization(
        target_kind="binding_owned",
        selection=selection,
        target_plan=binding_target,
        binding=binding,
        envelope=binding_envelope,
    )

    retained_member = replace(
        _target_set_retained_member(0),
        reservation_capability_expires_at_ms=4_102_444_800_000,
    )
    retained_target = RealizationTargetPlan(
        kind="retained_binding",
        target_layout_digest="retained-layout",
        copy_plan_digest="retained-copy-plan",
        member_count=1,
    )
    retained_envelope = envelope_for_retained_binding((retained_member,))
    retained_report = ArtifactRealizationReport(
        target_kind="retained_binding",
        source_selection_digest=selection.source_selection_digest,
        target_layout_digest=retained_target.target_layout_digest,
        copy_plan_digest=retained_target.copy_plan_digest,
        artifact_id=selection.artifact_id,
        view_id=selection.view_id,
        artifact_profile=selection.artifact_profile,
        authority_scope=selection.authority_scope,
        generation_hint=selection.generation_hint,
        envelope=retained_envelope,
        target_plan=retained_target,
        strategy_plan=strategy_plan_for_execution(envelope=retained_envelope),
        representation_admission=representation_admission_for_target(retained_target),
        lifecycle_plan=retained_binding_lifecycle_plan_for(
            (retained_member,),
            envelope=retained_envelope,
        ),
        operation_backend="daemon_prefetch_serving_binding",
        risk_labels=risk_labels_for_target(
            retained_target,
            retained_envelope,
            source_selection_digest=selection.source_selection_digest,
        ),
        retained_bindings=(retained_member,),
        publishability=publishability_report_for(),
    )

    runtime_target = RealizationTargetPlan(
        kind="runtime_attachment",
        device="cuda:0",
        target_layout_digest="runtime-layout",
        binding_layout_id="runtime-binding-layout",
    )
    runtime_envelope = envelope_for_runtime_attachment(tensors, retained=False)
    runtime_report = report_for_runtime_attachment(
        selection=selection,
        target_plan=runtime_target,
        envelope=runtime_envelope,
        materialization_diagnostics={"source": "disk", "total_bytes": 8},
    )

    target_set_member = replace(
        retained_member,
        staged_value=True,
        group_realization_transaction_id="txn-1",
        group_realization_version_set_id="vs-1",
    )
    target = SimpleNamespace(
        runtime="vllm",
        group_id="group-1",
        source=SimpleNamespace(
            source_kind="checkpoint_artifact",
            artifact_selection_digest=selection.source_selection_digest,
            source_artifact_ref=selection.artifact_id,
        ),
        topology=SimpleNamespace(schema_topology_digest="topology-digest"),
        members=(
            SimpleNamespace(
                member=SimpleNamespace(member_id="member-0"),
                resolved_layout=SimpleNamespace(target_layout_hash="target-layout-0"),
            ),
        ),
    )
    target_set_target = RealizationTargetPlan(
        kind="target_set",
        target_layout_digest="group-layout",
        member_count=1,
    )
    target_set_envelope = envelope_for_target_set((target_set_member,))
    target_set_details = target_set_report_for_retained_bindings(
        (target_set_member,),
        target=target,
        source_selection_digest=selection.source_selection_digest,
    )
    target_set_report = ArtifactRealizationReport(
        target_kind="target_set",
        source_selection_digest=selection.source_selection_digest,
        target_layout_digest=target_set_target.target_layout_digest,
        copy_plan_digest=target_set_target.copy_plan_digest,
        artifact_id=selection.artifact_id,
        view_id=selection.view_id,
        artifact_profile=selection.artifact_profile,
        authority_scope=selection.authority_scope,
        generation_hint=selection.generation_hint,
        envelope=target_set_envelope,
        target_plan=target_set_target,
        strategy_plan=target_set_strategy_plan_for(target_set_details, target=target),
        representation_admission=target_set_representation_admission_for(
            target_set_details
        ),
        lifecycle_plan=target_set_lifecycle_plan_for(
            target_set_details,
            envelope=target_set_envelope,
        ),
        operation_id="target-set-op",
        operation_backend="daemon_prefetch_serving_binding_set",
        risk_labels=risk_labels_for_target(
            target_set_target,
            target_set_envelope,
            source_selection_digest=selection.source_selection_digest,
        ),
        retained_bindings=(target_set_member,),
        target_set=target_set_details,
        publishability=publishability_report_for(),
    )

    reports = (
        tensor_report,
        binding_report,
        retained_report,
        runtime_report,
        target_set_report,
    )
    for report in reports:
        assert report.source_selection_digest
        assert report.artifact_id == selection.artifact_id
        assert report.artifact_profile == selection.artifact_profile
        assert report.authority_scope == selection.authority_scope
        assert report.operation_backend
        assert report.target_plan is not None
        assert report.target_plan.kind == report.target_kind
        assert report.representation_admission is not None
        assert report.lifecycle_plan is not None
        assert report.lifecycle_plan.capability == report.target_kind
        assert report.publishability is not None
        assert report.envelope.backing_kind
        assert report.envelope.export_kind
        assert report.envelope.projection_kind
        assert report.envelope.owner_kind
        assert report.envelope.release_policy
        assert report.envelope.release_strictness == "strict"
        assert report.envelope.export_lifetime_kind
        assert report.envelope.mutability_contract
        assert "authority" in report.risk_labels
        assert "identity" in report.risk_labels
        assert "lifecycle" in report.risk_labels
        report.validate_for_handle(report.target_kind)

    retained_profile = artifact_realization_profile_payload(retained_report)
    assert retained_profile["retained_binding_count"] == 1
    assert retained_profile["retained_binding_reservation_bytes"] == 1024
    assert retained_profile["retained_binding_capability_ids"] == ("cap-0",)
    assert retained_profile["retained_binding_capability_expires_at_ms"] == (
        4_102_444_800_000,
    )
    assert retained_profile["retained_binding_readiness"] == ("runtime_local_ready",)
    assert retained_profile["retained_binding_verification_states"] == ("local_only",)


def test_target_set_strategy_and_lifecycle_plans_capture_group_barriers() -> None:
    retained = (
        replace(
            _target_set_retained_member(0),
            staged_value=True,
            group_realization_transaction_id="txn-1",
            group_realization_version_set_id="vs-1",
            group_realization_part_id="part-0",
            group_realization_staging_token="stage-0",
            group_realization_wait_for_publish=True,
            group_realization_wait_timeout_ms=2500,
        ),
        replace(
            _target_set_retained_member(1),
            staged_value=True,
            group_realization_transaction_id="txn-1",
            group_realization_version_set_id="vs-1",
            group_realization_part_id="part-1",
            group_realization_staging_token="stage-1",
        ),
    )
    source = SimpleNamespace(
        source_kind="checkpoint_artifact",
        artifact_selection_digest="source-selection",
        source_artifact_ref="mi2:source",
    )
    target = SimpleNamespace(
        runtime="vllm",
        group_id="group-1",
        source=source,
        topology=SimpleNamespace(
            schema_topology_digest="topology-digest",
            admission_topology_digest="placement-digest",
        ),
        members=(
            SimpleNamespace(
                runtime="vllm",
                device="cuda:0",
                device_uuid="GPU-0",
                member=SimpleNamespace(
                    member_id="member-0",
                    member_index=0,
                    member_count=2,
                    group_id="group-1",
                ),
                model_config_digest="model-config",
                runtime_build_digest="serving-build",
                source=source,
                resolved_layout=SimpleNamespace(
                    target_layout_hash="target-layout-0",
                    spec_digest="spec-digest-0",
                ),
            ),
            SimpleNamespace(
                runtime="vllm",
                device="cuda:1",
                device_uuid="GPU-1",
                member=SimpleNamespace(
                    member_id="member-1",
                    member_index=1,
                    member_count=2,
                    group_id="group-1",
                ),
                model_config_digest="model-config",
                runtime_build_digest="serving-build",
                source=source,
                resolved_layout=SimpleNamespace(
                    target_layout_hash="target-layout-1",
                    spec_digest="spec-digest-1",
                ),
            ),
        ),
    )
    target_set_report = target_set_report_for_retained_bindings(
        retained,
        target=target,
        source_selection_digest="fallback-selection",
    )
    envelope = envelope_for_target_set(retained)

    strategy_plan = target_set_strategy_plan_for(
        target_set_report,
        target=target,
    )
    lifecycle_plan = target_set_lifecycle_plan_for(
        target_set_report,
        envelope=envelope,
    )

    assert target_set_report.publish_barrier is True
    assert target_set_report.acquire_claim_ids == ("cap-0", "cap-1")
    assert target_set_report.members[0].device == "cuda:0"
    assert target_set_report.members[0].copy_plan_digest == "spec-digest-0"
    assert target_set_report.members[0].runtime_profile_digest is not None
    assert target_set_report.members[0].placement_digest is not None
    assert strategy_plan.source_selection_mode == "same_selection"
    assert strategy_plan.source_coordination == "same_daemon_session"
    assert strategy_plan.collective_policy == "collective_first_candidate"
    assert strategy_plan.group_barriers == (
        "member_readiness",
        "group_acquire",
        "staged_values",
        "publish_barrier",
    )
    assert lifecycle_plan.release_policy == envelope.release_policy
    assert lifecycle_plan.staged_value_count == 2
    assert lifecycle_plan.acquire_claim_count == 2
    assert lifecycle_plan.acquire_claim_ids == ("cap-0", "cap-1")
    assert lifecycle_plan.publish_barrier is True
    assert lifecycle_plan.member_release_policies["member-0"] == (
        "release_binding_reservation",
        "release_group_staged_acquire",
    )


def test_target_set_report_marks_serving_artifact_set_as_per_part() -> None:
    retained = (_target_set_retained_member(0), _target_set_retained_member(1))
    target = SimpleNamespace(
        runtime="vllm",
        group_id="group-1",
        source=SimpleNamespace(
            source_kind="runtime_artifact_set",
            artifact_selection_digest="artifact-set-selection",
            source_artifact_ref=None,
            members=(
                SimpleNamespace(
                    member=SimpleNamespace(member_id="member-0"),
                    artifact_ref="mi2:serving-member-0",
                ),
                SimpleNamespace(
                    member=SimpleNamespace(member_id="member-1"),
                    artifact_ref="mi2:serving-member-1",
                ),
            ),
        ),
        topology=SimpleNamespace(schema_topology_digest="topology-digest"),
        members=(
            SimpleNamespace(
                member=SimpleNamespace(member_id="member-0"),
                resolved_layout=SimpleNamespace(target_layout_hash="target-layout-0"),
            ),
            SimpleNamespace(
                member=SimpleNamespace(member_id="member-1"),
                resolved_layout=SimpleNamespace(target_layout_hash="target-layout-1"),
            ),
        ),
    )

    report = target_set_report_for_retained_bindings(
        retained,
        target=target,
        source_selection_digest="fallback-selection",
    )

    assert report.source_kind == "runtime_artifact_set"
    assert report.source_selection_mode == "per_part_selection"
    assert [member.source_artifact_ref for member in report.members] == [
        "mi2:serving-member-0",
        "mi2:serving-member-1",
    ]


def test_target_set_envelope_requires_group_layout_and_members() -> None:
    with pytest.raises(ArtifactError) as excinfo:
        envelope_for_target_set(())

    assert excinfo.value.status_code == "DATA_LOSS"


def test_target_set_report_dict_includes_strategy_and_lifecycle_plans() -> None:
    retained = (_target_set_retained_member(0), _target_set_retained_member(1))
    source = SimpleNamespace(
        source_kind="checkpoint_artifact",
        artifact_selection_digest="source-selection",
        source_artifact_ref="mi2:source",
    )
    target = SimpleNamespace(
        runtime="vllm",
        group_id="group-1",
        source=source,
        topology=SimpleNamespace(schema_topology_digest="topology-digest"),
        members=(
            SimpleNamespace(
                member=SimpleNamespace(member_id="member-0"),
                source=source,
                resolved_layout=SimpleNamespace(target_layout_hash="target-layout-0"),
            ),
            SimpleNamespace(
                member=SimpleNamespace(member_id="member-1"),
                source=source,
                resolved_layout=SimpleNamespace(target_layout_hash="target-layout-1"),
            ),
        ),
    )
    target_set_report = target_set_report_for_retained_bindings(
        retained,
        target=target,
        source_selection_digest="selection-digest",
    )
    envelope = envelope_for_target_set(retained)
    report = ArtifactRealizationReport(
        target_kind="target_set",
        source_selection_digest="selection-digest",
        target_layout_digest="group-target-layout",
        copy_plan_digest="copy-plan",
        artifact_id="mi2:test:artifact",
        view_id="",
        artifact_profile="durable_artifact",
        authority_scope="daemon_mediated_durable",
        generation_hint=None,
        envelope=envelope,
        strategy_plan=target_set_strategy_plan_for(target_set_report, target=target),
        lifecycle_plan=target_set_lifecycle_plan_for(
            target_set_report,
            envelope=envelope,
        ),
        target_set=target_set_report,
    )

    payload = artifact_realization_report_to_dict(report)

    assert isinstance(payload["strategy_plan"], dict)
    assert payload["strategy_plan"]["source_selection_mode"] == "same_selection"
    assert isinstance(payload["lifecycle_plan"], dict)
    assert payload["lifecycle_plan"]["acquire_claim_ids"] == ("cap-0", "cap-1")


def test_risk_labels_are_derived_from_target_plan_and_envelope() -> None:
    envelope = envelope_for_target_set((_target_set_retained_member(0),))
    target_plan = RealizationTargetPlan(
        kind="target_set",
        target_layout_digest="group-target-layout",
        member_count=2,
    )

    labels = risk_labels_for_target(
        target_plan,
        envelope,
        source_selection_digest="selection-digest",
        extra=("authority", "custom-risk"),
    )

    assert labels == (
        "authority",
        "identity",
        "lifecycle",
        "lease_strength",
        "mutability",
        "async_continuation",
        "target_set",
        "custom-risk",
    )


_RISK_CLOSURE_MATRIX: tuple[dict[str, str], ...] = (
    {
        "risk": "Selection resolver becomes too broad.",
        "admission_field": "artifact_id/key exclusivity, view_id, generation_hint",
        "envelope_field": "target_layout_digest remains target-plan owned",
        "report_field": "source_selection_digest",
        "guardrail_test": "test_resolve_artifact_selection_keeps_target_plan_identity_separate",
        "blocking_condition": "target layout or copy-plan policy moves into selection",
    },
    {
        "risk": "SDK direct Global Store access survives behind helper APIs.",
        "admission_field": "authority_scope",
        "envelope_field": "owner_kind",
        "report_field": "authority_scope",
        "guardrail_test": "test_sdk_api_paths_do_not_open_global_store_channels",
        "blocking_condition": "SDK artifact realization opens Global Store channels",
    },
    {
        "risk": "`PublicDiskSourceHandle` becomes a permanent source authority.",
        "admission_field": "artifact_profile=mounted_source",
        "envelope_field": "backing_kind=mounted_source_metadata",
        "report_field": "mounted_source.source_artifact_id",
        "guardrail_test": "test_mounted_source_realize_rejects_non_msa1_subject",
        "blocking_condition": "mounted source executes without msa1 identity",
    },
    {
        "risk": "Mapped target layout is confused with source selection.",
        "admission_field": "target_layout_digest",
        "envelope_field": "projection_kind",
        "report_field": "copy_plan_digest",
        "guardrail_test": "test_resolve_artifact_selection_accepts_mapped_source_view_hint",
        "blocking_condition": "mapped/adopted target reports reuse selection digest as layout",
    },
    {
        "risk": "TensorDict accidentally inherits binding lifecycle.",
        "admission_field": "target_kind=tensor_dict",
        "envelope_field": "projection_kind=tensor_dict",
        "report_field": "publishability.reason",
        "guardrail_test": "test_tensor_dict_handle_rejects_binding_lifecycle_capabilities",
        "blocking_condition": "TensorDict handle can publish, promote, or retain",
    },
    {
        "risk": "TensorDict projections release daemon payloads too early or leak them.",
        "admission_field": "release_strictness",
        "envelope_field": "release_policy",
        "report_field": "envelope.release_policy",
        "guardrail_test": "test_tensor_subset_materialization_and_release",
        "blocking_condition": "projection close does not unload daemon payload exactly once",
    },
    {
        "risk": "Resource lifecycle remains path-specific under a unified API.",
        "admission_field": "release_strictness",
        "envelope_field": "release_policy",
        "report_field": "lifecycle_plan.capability",
        "guardrail_test": "test_release_contract_lifecycle_matrix_runs_policy_actions_once",
        "blocking_condition": "cleanup action exists outside a release contract",
    },
    {
        "risk": "Handle-lease mint failure silently weakens export lifetime.",
        "admission_field": "export_lifetime_kind",
        "envelope_field": "export_kind",
        "report_field": "envelope.export_lifetime_kind",
        "guardrail_test": "test_cpu_memfd_materialization_fails_before_tensor_restore_without_export_authority",
        "blocking_condition": "CPU memfd or CUDA IPC export succeeds without token authority",
    },
    {
        "risk": "CPU TensorDict mutability stays ambiguous.",
        "admission_field": "mutability_contract",
        "envelope_field": "mutability_contract",
        "report_field": "envelope.mutability_contract",
        "guardrail_test": "test_tensor_dict_projection_rejects_mapping_mutations",
        "blocking_condition": "TensorDict mapping mutation succeeds",
    },
    {
        "risk": "`get_into` hides expensive fallback copies.",
        "admission_field": "fallback_policy",
        "envelope_field": "fallback_reason_buckets",
        "report_field": "copy_bytes",
        "guardrail_test": "test_get_into_returns_fallback_result_and_unloads",
        "blocking_condition": "temporary-payload fallback has no report bucket",
    },
    {
        "risk": "Prefetch grows a second continuation model.",
        "admission_field": "operation_id",
        "envelope_field": "projection_kind=prefetch_handoff",
        "report_field": "operation_backend",
        "guardrail_test": "test_realize_async_retained_replica_operation_status_wait_and_cancel",
        "blocking_condition": "prefetch bypasses Operation status/wait/cancel",
    },
    {
        "risk": "Binding paths bypass strategy planning.",
        "admission_field": "fallback_policy",
        "envelope_field": "direct_write_bytes",
        "report_field": "strategy_plan.fallback_policy",
        "guardrail_test": "test_binding_envelope_and_report_capture_identity_diagnostics",
        "blocking_condition": "binding materialization report lacks strategy facts",
    },
    {
        "risk": "Tensor-aware strategy loses lane/residual visibility.",
        "admission_field": "execution_plan_kind",
        "envelope_field": "temporary_replica_bytes",
        "report_field": "execution_commit.lane_allocation_bytes",
        "guardrail_test": "test_binding_envelope_and_report_capture_identity_diagnostics",
        "blocking_condition": "mixed execution omits lane, residual, or reject buckets",
    },
    {
        "risk": "TP grows special-case orchestration.",
        "admission_field": "target_set.source_selection_mode",
        "envelope_field": "projection_kind=target_set",
        "report_field": "target_set.members",
        "guardrail_test": "test_group_member_same_and_per_part_selection_identity",
        "blocking_condition": "TP path adds non-target-set realization state",
    },
    {
        "risk": "RPC cleanup is attempted too early.",
        "admission_field": "controller plan validation",
        "envelope_field": "resource_authorities",
        "report_field": "controller plan spans",
        "guardrail_test": "daemon controller realization plan tests",
        "blocking_condition": "proto cleanup lands before shared controller path",
    },
    {
        "risk": "Target-state behavior regresses while compatibility code is deleted.",
        "admission_field": "scenario acceptance coverage",
        "envelope_field": "runtime_attachment release_policy",
        "report_field": "model_runtime.runtime_attachment_target_kind",
        "guardrail_test": "serving integration/runtime publication scenarios",
        "blocking_condition": "compatibility code deletion lacks runtime scenario coverage",
    },
)


def test_risk_closure_matrix_has_unique_risks_and_enforcement_fields() -> None:
    required_fields = (
        "admission_field",
        "envelope_field",
        "report_field",
        "guardrail_test",
        "blocking_condition",
    )
    matrix_by_risk = {entry["risk"]: entry for entry in _RISK_CLOSURE_MATRIX}

    assert len(matrix_by_risk) == len(_RISK_CLOSURE_MATRIX)
    for risk, entry in matrix_by_risk.items():
        for field in required_fields:
            assert entry[field], f"{risk} missing {field}"


def test_sdk_realization_paths_do_not_import_selection_builder() -> None:
    checked = (
        Path("tensorcast/api/store/__init__.py"),
        Path("tensorcast/api/_materialize.py"),
        Path("tensorcast/api/store/artifact.py"),
        Path("tensorcast/api/store/materialization.py"),
        Path("tensorcast/api/store/inplace_slot.py"),
        Path("tensorcast/api/store/runtime_realization_reference_consumer.py"),
        Path("tensorcast/api/plan/plan.py"),
    )
    offenders: list[str] = []
    for path in checked:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != "tensorcast.common.selection_contract":
                continue
            if any(alias.name == "build_artifact_selection" for alias in node.names):
                offenders.append(str(path))

    assert offenders == []


def test_sdk_lowering_does_not_call_removed_artifact_selection_adapter() -> None:
    offenders = [
        str(path)
        for path in sorted(Path("tensorcast/api").rglob("*.py"))
        if "_build_artifact_selection(" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_artifact_no_longer_exposes_removed_selection_adapters() -> None:
    from tensorcast.api.store.artifact import Artifact

    assert not hasattr(Artifact, "_build_artifact_selection")
    assert not hasattr(Artifact, "_build_owner_source_selection")
    assert not hasattr(Artifact, "_resolve_owner_source_selection")


def test_public_artifact_entrypoints_call_realization_facade() -> None:
    path = Path("tensorcast/api/store/artifact.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    artifact_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Artifact"
    )
    expected = {
        "tensor_dict",
        "tensor_dict_with_diagnostics",
        "tensor_dict_into",
        "tensor_into",
        "bind",
        "bind_into",
    }
    methods = {
        node.name: node
        for node in artifact_class.body
        if isinstance(node, ast.FunctionDef) and node.name in expected
    }

    assert set(methods) == expected
    for name, method in methods.items():
        calls_realize = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "realize"
            for node in ast.walk(method)
        )
        assert calls_realize, f"{name} does not call a realization facade"


def test_sdk_lowering_does_not_call_removed_owner_source_selection_adapter() -> None:
    offenders = [
        str(path)
        for path in sorted(Path("tensorcast/api").rglob("*.py"))
        if any(
            removed_call in path.read_text(encoding="utf-8")
            for removed_call in (
                "._build_owner_source_selection(",
                "._resolve_owner_source_selection(",
            )
        )
    ]

    assert offenders == []


def test_realization_reports_do_not_expose_legacy_diagnostics_field() -> None:
    offenders = [
        str(path)
        for path in sorted(Path("tensorcast/api").rglob("*.py"))
        if "legacy_diagnostics" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_client_binding_rollbacks_log_cleanup_failures_instead_of_suppressing() -> None:
    path = Path("tensorcast/api/store/artifact.py")
    source = path.read_text(encoding="utf-8")
    assert "contextlib.suppress(Exception)" not in source

    tree = ast.parse(source)
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_close_client_binding_best_effort"
    )
    helper_source = ast.get_source_segment(source, helper) or ""
    assert "close_owned_binding" in helper_source
    assert "logger.exception" in helper_source


def test_realization_lifecycle_code_does_not_silently_suppress_broad_exceptions() -> (
    None
):
    guarded_paths = (
        Path("tensorcast/api/_register.py"),
        Path("tensorcast/api/store/__init__.py"),
        Path("tensorcast/api/store/artifact.py"),
        Path("tensorcast/api/store/materialization.py"),
        Path("tensorcast/api/store/inplace_slot.py"),
        Path("tensorcast/api/store/owned_binding_slot.py"),
        Path("tensorcast/api/store/async_ops.py"),
        Path("tensorcast/api/store/registration.py"),
        Path("tensorcast/api/store/realization_kernel.py"),
        Path("tensorcast/api/store/runtime.py"),
        Path("tensorcast/global_store/cluster_runtime_rpc.py"),
        Path("tensorcast/global_store/db_utils.py"),
        Path("tensorcast/global_store/repositories/base.py"),
        Path("tensorcast/global_store/rpc/replica_registration_rpc_handler.py"),
        Path("tensorcast/global_store/rpc/transport_rpc_handler.py"),
        Path("tensorcast/global_store/services/instance_service.py"),
        Path("tensorcast/artifact_runtime/binding/retained.py"),
        Path("tensorcast/artifact_runtime/lifecycle.py"),
        Path("tensorcast/artifact_runtime/recipe/local_ready.py"),
        Path("tensorcast/artifact_runtime/recipe/build.py"),
    )
    offenders = [
        str(path)
        for path in guarded_paths
        if "suppress(Exception)" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_target_region_registration_uses_offset_aware_cuda_ipc_export_only() -> None:
    guarded_paths = (
        Path("tensorcast/api/store/__init__.py"),
        Path("tensorcast/api/store/materialization.py"),
    )
    retired_export = "get_cuda_memory_" + "handle("
    offenders = [
        str(path)
        for path in guarded_paths
        if retired_export in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_core_cuda_ipc_export_no_longer_exposes_non_offset_handle_path() -> None:
    guarded_paths = (
        Path("core/checkpoint/checkpoint.h"),
        Path("core/checkpoint/checkpoint.cc"),
        Path("tensorcast/csrc/checkpoint_py.cc"),
        Path("tensorcast/_C.pyi"),
        Path("tensorcast/_c_ext.py"),
    )
    retired_export = "get_cuda_memory_" + "handle("
    offenders = [
        str(path)
        for path in guarded_paths
        if retired_export in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_mounted_source_config_no_longer_exposes_absolute_fallback_mode() -> None:
    guarded_paths = (
        Path("proto/tensorcast/config/v1/daemon_config.proto"),
        Path("daemon/state/daemon_options.h"),
        Path("daemon/service/controllers/disk_artifact_service.cc"),
        Path("daemon/app/server_main.cc"),
        Path("tensorcast/daemon_runtime_config.py"),
        Path("core/common/config/daemon_config_io.cc"),
    )
    retired_tokens = (
        "UNMATCHED_PATH_MODE_ALLOW_ABSOLUTE_FALLBACK",
        "kAllowAbsoluteFallback",
        "used_absolute_fallback",
        "unmatched_path_mode()",
        "unmatched_path_mode =",
    )
    offenders = [
        f"{path}:{token}"
        for path in guarded_paths
        for token in retired_tokens
        if token in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_daemon_canonical_index_loading_uses_explicit_authority_not_disk_fallback() -> (
    None
):
    guarded_paths = (
        Path("daemon/service/controllers/materialization_index_source_utils.h"),
        Path("daemon/service/controllers/materialization_index_source_utils.cc"),
        Path("daemon/service/controllers/owned_binding_service.cc"),
        Path("daemon/service/controllers/replica_materialization_service.cc"),
        Path("daemon/service/controllers/target_materialization_service.cc"),
    )
    retired_tokens = (
        "load_canonical_index_with_disk_fallback",
        "prefer_disk_index",
        "fallback_artifact_id",
        "allow_local_import_fallback",
        "local_import_fallback",
    )
    offenders = [
        f"{path}:{token}"
        for path in guarded_paths
        for token in retired_tokens
        if token in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_tensor_dict_projection_fails_closed_when_owner_attachment_fails() -> None:
    class AttrlessTensor:
        __slots__ = ()

    tensor = AttrlessTensor()
    envelope = envelope_for_tensor_dict({"a": tensor}, source="disk")
    report = ArtifactRealizationReport(
        target_kind="tensor_dict",
        source_selection_digest="digest",
        target_layout_digest=None,
        copy_plan_digest=None,
        artifact_id="mi2:test:artifact",
        view_id="",
        artifact_profile="durable_artifact",
        authority_scope="daemon_mediated_durable",
        generation_hint=None,
        envelope=envelope,
    )
    handle = ArtifactRealizationHandle(
        target_kind="tensor_dict",
        report=report,
        tensor_dict_value={"a": tensor},
    )

    with pytest.raises(ArtifactError) as excinfo:
        handle.tensor_dict()

    assert excinfo.value.status_code == "FAILED_PRECONDITION"


def test_sdk_materialization_client_surface_has_no_retired_v2_aliases() -> None:
    retired_snake = (
        "materialize_artifact",
        "materialize_by_artifact_id",
        "materialize_into_target",
        "import_artifact_from_path",
        "import_artifact_from_path_stream",
    )
    retired_span_prefixes = (
        "MaterializeArtifact",
        "MaterializeReplica",
        "MaterializeIntoTarget",
        "ImportArtifactFromPath",
        "ImportArtifactFromPathStream",
    )
    retired_names = tuple(f"{name}_v2" for name in retired_snake) + tuple(
        f"{name}V2" for name in retired_span_prefixes
    )
    offenders: list[str] = []
    for root in (Path("tensorcast"), Path("tests/python")):
        for path in sorted(root.rglob("*.py")):
            if path == Path("tests/python/api/test_realization_kernel.py"):
                continue
            source = path.read_text(encoding="utf-8")
            offenders.extend(
                f"{path}:{name}" for name in retired_names if name in source
            )

    assert offenders == []


def test_target_publication_legacy_capability_proto_is_removed() -> None:
    assert not hasattr(capability_token_pb2, "TargetPublicationScope")
    assert not hasattr(
        capability_token_pb2,
        "CAPABILITY_AUDIENCE_TARGET_PUBLICATION",
    )


def test_daemon_loaded_replica_status_surface_has_no_retired_v2_name() -> None:
    checked_roots = (
        Path("proto/tensorcast/daemon/v2"),
        Path("daemon/service"),
        Path("daemon/state"),
    )
    offenders = [
        str(path)
        for root in checked_roots
        for path in sorted(root.rglob("*"))
        if path.suffix in {".proto", ".cc", ".h"}
        and "GetLoadedReplicasV2" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_realization_kernel_paths_do_not_open_global_store_channels() -> None:
    checked = (
        Path("tensorcast/api/store/__init__.py"),
        Path("tensorcast/api/_materialize.py"),
        Path("tensorcast/api/store/artifact.py"),
        Path("tensorcast/api/store/materialization.py"),
        Path("tensorcast/api/store/inplace_slot.py"),
        Path("tensorcast/api/store/realization_kernel.py"),
    )
    forbidden = (
        "GlobalStoreCompositeStub",
        "tensorcast.global_store",
        "global_store_pb2",
        "global_store_pb2_grpc",
        "tensorcast.proto.global_store",
        "grpc.insecure_channel",
        "grpc.secure_channel",
        "grpc.aio.insecure_channel",
        "grpc.aio.secure_channel",
    )
    offenders = [
        f"{path}:{token}"
        for path in checked
        for token in forbidden
        if token in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_sdk_api_paths_do_not_open_global_store_channels() -> None:
    forbidden = (
        "GlobalStoreCompositeStub",
        "tensorcast.global_store",
        "global_store_pb2",
        "global_store_pb2_grpc",
        "tensorcast.proto.global_store",
        "grpc.insecure_channel",
        "grpc.secure_channel",
        "grpc.aio.insecure_channel",
        "grpc.aio.secure_channel",
    )
    offenders = [
        f"{path}:{token}"
        for path in sorted(Path("tensorcast/api").rglob("*.py"))
        for token in forbidden
        if token in path.read_text(encoding="utf-8")
    ]

    assert offenders == []
