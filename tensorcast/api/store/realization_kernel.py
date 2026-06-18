#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal, NoReturn, TypedDict

from tensorcast.api.store.common import canonical_index_from_bytes
from tensorcast.api.store.types import ArtifactError
from tensorcast.common.selection_contract import (
    build_artifact_selection,
    compute_selected_index_bytes,
)
from tensorcast.profile_utils import emit_tensorcast_profile_event
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.types import ExecutionDiagnostics, SourceBoundPlanDiagnostics

logger = logging.getLogger(__name__)

RealizationTargetKind = Literal[
    "tensor_dict",
    "caller_tensors",
    "binding_owned",
    "binding_adopted",
    "retained_replica",
    "retained_binding",
    "runtime_attachment",
    "model_runtime",
    "publication",
    "target_set",
    "mounted_source",
]
BindingRealizationTargetKind = Literal["binding_owned", "binding_adopted"]


def _sha256_hex(*parts: bytes) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


def _coerce_mapping_resolution(result: object) -> tuple[str, int | None]:
    if isinstance(result, tuple):
        artifact_id = str(result[0] or "")
        generation = (
            int(result[1]) if len(result) > 1 and result[1] is not None else None
        )
        return artifact_id, generation
    artifact_id = str(getattr(result, "artifact_id", "") or "")
    generation_raw = getattr(result, "generation", None)
    generation = int(generation_raw) if generation_raw is not None else None
    return artifact_id, generation


def _artifact_profile_for(artifact_id: str) -> str:
    if artifact_id.startswith("msa1:"):
        return "mounted_source"
    if artifact_id.startswith("cgid:"):
        return "byte_artifact"
    return "durable_artifact"


def _authority_scope_for(artifact_id: str) -> str:
    if artifact_id.startswith("msa1:"):
        return "daemon_local_mounted_source"
    return "daemon_mediated_durable"


def materialization_source_label(source: object) -> str | None:
    if not isinstance(source, (int, str, bytes, bytearray)):
        return None
    try:
        source_code = int(source)
    except (TypeError, ValueError):
        return None
    if source_code == int(store_daemon_pb2.MATERIALIZATION_SOURCE_P2P):
        return "p2p"
    if source_code == int(store_daemon_pb2.MATERIALIZATION_SOURCE_DISK):
        return "disk"
    if source_code == int(store_daemon_pb2.MATERIALIZATION_SOURCE_LOCAL_REPLICA):
        return "local_replica"
    return None


def _layout_total_bytes(layout: object) -> int:
    target_index_bytes = getattr(layout, "target_index_bytes", b"")
    if not isinstance(target_index_bytes, bytes):
        return 0
    try:
        return int(canonical_index_from_bytes(target_index_bytes).total_size_bytes)
    except Exception:  # noqa: BLE001
        return 0


def binding_materialization_diagnostics_from_response(
    response: object,
    *,
    layout: object,
) -> Mapping[str, object] | None:
    source = getattr(
        response,
        "source",
        store_daemon_pb2.MATERIALIZATION_SOURCE_UNSPECIFIED,
    )
    source_label = materialization_source_label(source)
    if source_label is None:
        return None
    try:
        source_code = int(source)
    except (TypeError, ValueError):
        source_code = 0
    diagnostics: dict[str, object] = {
        "source": source_label,
        "source_code": source_code,
        "total_bytes": _layout_total_bytes(layout),
        "retry_attempts": 1,
        "retry_reason_buckets": {},
    }
    replica_id = str(getattr(response, "selected_source_replica_id", "") or "")
    if replica_id:
        diagnostics["replica_id"] = replica_id
    transport_id = str(getattr(response, "selected_source_transport_id", "") or "")
    if transport_id:
        diagnostics["transport_id"] = transport_id
    return diagnostics


def execution_diagnostics_from_response(
    response: object,
    *,
    field_name: str = "execution_diagnostics",
) -> ExecutionDiagnostics | None:
    diagnostics_proto = getattr(response, field_name, None)
    if diagnostics_proto is None:
        return None
    has_field = getattr(response, "HasField", None)
    if callable(has_field):
        try:
            if not has_field(field_name):
                return None
        except ValueError:
            pass
    return ExecutionDiagnostics.from_proto(diagnostics_proto)


def source_bound_plan_diagnostics_from_response(
    response: object,
    *,
    field_name: str = "source_bound_plan_diagnostics",
) -> SourceBoundPlanDiagnostics | None:
    diagnostics_proto = getattr(response, field_name, None)
    if diagnostics_proto is None:
        return None
    has_field = getattr(response, "HasField", None)
    if callable(has_field):
        try:
            if not has_field(field_name):
                return None
        except ValueError:
            pass
    return SourceBoundPlanDiagnostics.from_proto(diagnostics_proto)


@dataclass(frozen=True, slots=True)
class ResolvedArtifactSelection:
    proto: common_pb2.ArtifactSelection
    artifact_id: str
    key: str | None
    canonical_index_bytes: bytes
    selected_index_bytes: bytes
    tensor_names: tuple[str, ...]
    view_id: str
    view_subset_hash: bytes
    logical_layout_hash: bytes
    selection_hash: bytes
    generation_hint: int | None
    artifact_profile: str
    authority_scope: str
    source_selection_digest: str
    diagnostics: Mapping[str, object] = field(default_factory=dict)


class SelectionReportFields(TypedDict):
    view_subset_hash: str
    logical_layout_hash: str
    selection_hash: str


def selection_report_fields(
    selection: ResolvedArtifactSelection,
) -> SelectionReportFields:
    return {
        "view_subset_hash": selection.view_subset_hash.hex(),
        "logical_layout_hash": selection.logical_layout_hash.hex(),
        "selection_hash": selection.selection_hash.hex(),
    }


def resolve_artifact_selection(
    *,
    artifact_id: str | None,
    key: str | None = None,
    canonical_index_bytes: bytes | None = None,
    view_spec: common_pb2.ViewSpec | None = None,
    view_id: str | None = None,
    tensor_names: Sequence[str] | None = None,
    view_subset_hash: bytes | None = None,
    view_index_hint: bytes | None = None,
    generation_hint: int | None = None,
    key_resolver: Callable[[str], object] | None = None,
    canonical_index_resolver: Callable[[str], bytes] | None = None,
    allow_view_id_without_spec: bool = False,
    artifact_profile: str | None = None,
    authority_scope: str | None = None,
) -> ResolvedArtifactSelection:
    if artifact_id and key:
        raise ArtifactError(
            "Specify either artifact_id or key, not both",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    if not artifact_id and not key:
        raise ArtifactError(
            "Either artifact_id or key is required",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )

    diagnostics: dict[str, object] = {}
    resolved_artifact_id = str(artifact_id or "")
    resolved_generation = generation_hint
    if not resolved_artifact_id:
        if key_resolver is None or key is None:
            raise ArtifactError(
                "key resolution requires a daemon key_resolver",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        mapped_id, mapped_generation = _coerce_mapping_resolution(key_resolver(key))
        if not mapped_id:
            raise ArtifactError(
                f"Artifact key '{key}' is not mapped",
                status_code="NOT_FOUND",
                retryable=False,
            )
        resolved_artifact_id = mapped_id
        if resolved_generation is None:
            resolved_generation = mapped_generation
        diagnostics["resolved_from_key"] = True

    canonical_bytes = bytes(canonical_index_bytes or b"")
    profile = artifact_profile or _artifact_profile_for(resolved_artifact_id)
    scope = authority_scope or _authority_scope_for(resolved_artifact_id)
    if resolved_artifact_id.startswith("msa1:"):
        if diagnostics.get("resolved_from_key") is True:
            raise ArtifactError(
                "mounted-source artifacts require explicit msa1 artifact ids, not durable key activation",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if profile != "mounted_source" or scope != "daemon_local_mounted_source":
            raise ArtifactError(
                "mounted-source artifacts require daemon-local mounted-source authority",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if not canonical_bytes:
            raise ArtifactError(
                "mounted-source artifact realization requires daemon-attested canonical index bytes",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
    if not canonical_bytes and canonical_index_resolver is not None:
        canonical_bytes = bytes(canonical_index_resolver(resolved_artifact_id) or b"")
        diagnostics["canonical_index_source"] = "daemon"
    elif canonical_bytes:
        diagnostics["canonical_index_source"] = "hint"
    else:
        diagnostics["canonical_index_source"] = "empty"

    ordered_names = tuple(str(name) for name in (tensor_names or ()))
    has_subset = bool(ordered_names)
    has_transform = bool(view_spec is not None and view_spec.tensors)
    resolved_view_index = bytes(view_index_hint or b"")
    if (has_subset or has_transform) and not resolved_view_index:
        try:
            resolved_view_index = compute_selected_index_bytes(
                canonical_index_bytes=canonical_bytes,
                view_spec=view_spec if has_transform else None,
                tensor_names=ordered_names if has_subset else None,
            )
            diagnostics["selected_index_source"] = "computed"
        except ValueError as exc:
            raise ArtifactError(
                str(exc),
                status_code="FAILED_PRECONDITION",
                retryable=False,
            ) from exc
    elif resolved_view_index:
        diagnostics["selected_index_source"] = "hint"
    else:
        diagnostics["selected_index_source"] = "canonical"

    try:
        proto = build_artifact_selection(
            artifact_id=resolved_artifact_id,
            canonical_index_bytes=canonical_bytes,
            layout_index_bytes=resolved_view_index or None,
            view_spec=view_spec,
            tensor_names=ordered_names,
            view_subset_hash=view_subset_hash,
            view_id=view_id,
            allow_view_id_without_spec=allow_view_id_without_spec,
        )
    except ValueError as exc:
        raise ArtifactError(
            str(exc),
            status_code="FAILED_PRECONDITION",
            retryable=False,
        ) from exc

    selected_index_bytes = resolved_view_index or canonical_bytes
    proto_bytes = proto.SerializeToString(deterministic=True)
    source_selection_digest = _sha256_hex(
        proto_bytes,
        selected_index_bytes,
        profile.encode("utf-8"),
        scope.encode("utf-8"),
        str(resolved_generation if resolved_generation is not None else "").encode(
            "utf-8"
        ),
    )
    return ResolvedArtifactSelection(
        proto=proto,
        artifact_id=resolved_artifact_id,
        key=str(key) if key else None,
        canonical_index_bytes=canonical_bytes,
        selected_index_bytes=selected_index_bytes,
        tensor_names=ordered_names,
        view_id=str(proto.view_id or ""),
        view_subset_hash=bytes(proto.view_subset_hash or b""),
        logical_layout_hash=bytes(proto.logical_layout_hash),
        selection_hash=bytes(proto.selection_hash),
        generation_hint=resolved_generation,
        artifact_profile=profile,
        authority_scope=scope,
        source_selection_digest=source_selection_digest,
        diagnostics=diagnostics,
    )


@dataclass(frozen=True, slots=True)
class RealizationTargetPlan:
    kind: RealizationTargetKind
    device: object | None = None
    target_layout_digest: str | None = None
    binding_layout_id: str | None = None
    mapped_view_id: str | None = None
    copy_plan_digest: str | None = None
    member_count: int | None = None


@dataclass(frozen=True, slots=True)
class RealizationStrategyPlan:
    source_policy: object | None = None
    fallback_policy: str = "fail_closed"
    retry_policy: object | None = None
    deadline_ms: int | None = None
    collective_policy: str = "disabled"
    source_selection_mode: str | None = None
    source_coordination: str | None = None
    same_daemon_session: bool | None = None
    member_count: int | None = None
    group_barriers: tuple[str, ...] = ()
    publish_barrier: bool = False
    group_realization_transaction_ids: tuple[str, ...] = ()
    group_realization_version_set_ids: tuple[str, ...] = ()
    member_source_selection_digests: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RepresentationAdmissionPlan:
    representation_contract: str = "identity"
    transform_required: bool = False
    transform_plan_digest: str | None = None


@dataclass(frozen=True, slots=True)
class RealizationLifecyclePlan:
    capability: str
    publishable: bool = False
    retained: bool = False
    export_lifetime_kind: str = "token_backed"
    release_strictness: str = "strict"
    mutability_contract: str = "read_only"
    release_policy: tuple[str, ...] = ()
    staged_value_count: int = 0
    acquire_claim_count: int = 0
    acquire_claim_ids: tuple[str, ...] = ()
    publish_barrier: bool = False
    group_realization_transaction_ids: tuple[str, ...] = ()
    group_realization_version_set_ids: tuple[str, ...] = ()
    member_release_policies: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RealizationResourceEnvelope:
    backing_kind: str
    export_kind: str
    projection_kind: str
    owner_kind: str
    release_policy: tuple[str, ...]
    mutability_contract: str
    release_strictness: str
    export_lifetime_kind: str
    direct_write_bytes: int = 0
    copy_bytes: int = 0
    copy_count: int = 0
    mmap_bytes: int = 0
    cuda_ipc_open_count: int = 0
    cpu_memfd_fd_count: int = 0
    temporary_replica_bytes: int = 0
    retained_bytes: int = 0
    fallback_reason_buckets: Mapping[str, int] = field(default_factory=dict)

    def validate_for_target(self, target: RealizationTargetPlan) -> None:
        required_text_fields = (
            "backing_kind",
            "export_kind",
            "projection_kind",
            "owner_kind",
            "mutability_contract",
            "release_strictness",
            "export_lifetime_kind",
        )
        for field_name in required_text_fields:
            if not getattr(self, field_name):
                raise ArtifactError(
                    f"artifact realization resource envelope requires {field_name}",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
        if not self.release_policy:
            raise ArtifactError(
                "artifact realization resource envelope requires release_policy",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if target.kind == "tensor_dict":
            if (
                self.export_kind in {"cuda_ipc", "cpu_memfd"}
                and self.export_lifetime_kind != "token_backed"
            ):
                raise ArtifactError(
                    "tensor_dict process-visible exports require token-backed lifetime",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            if not self.mutability_contract:
                raise ArtifactError(
                    "tensor_dict realization requires a mutability contract",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
        if (
            target.kind in {"binding_owned", "binding_adopted", "caller_tensors"}
            and not target.target_layout_digest
        ):
            raise ArtifactError(
                f"{target.kind} realization requires target_layout_digest",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if (
            target.kind in {"binding_owned", "binding_adopted"}
            and not target.binding_layout_id
        ):
            raise ArtifactError(
                f"{target.kind} realization requires binding_layout_id",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if target.kind == "retained_replica" and target.device is None:
            raise ArtifactError(
                "retained_replica realization requires device",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if target.kind == "retained_binding" and not target.target_layout_digest:
            raise ArtifactError(
                "retained_binding realization requires target_layout_digest",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if target.kind == "runtime_attachment" and not target.target_layout_digest:
            raise ArtifactError(
                "runtime_attachment realization requires target_layout_digest",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if target.kind == "publication" and not target.target_layout_digest:
            raise ArtifactError(
                "publication realization requires target_layout_digest",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if target.kind == "target_set":
            if not target.target_layout_digest:
                raise ArtifactError(
                    "target_set realization requires target_layout_digest",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            if target.member_count is None or target.member_count <= 0:
                raise ArtifactError(
                    "target_set realization requires a positive member_count",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
        if target.kind == "mounted_source" and not target.target_layout_digest:
            raise ArtifactError(
                "mounted_source realization requires promoted target digest",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )


@dataclass(frozen=True, slots=True)
class RealizationBindingReport:
    binding_id: str
    binding_layout_id: str
    binding_value_id: str | None
    seal_generation: int | None
    value_state: str
    source_artifact_id: str | None
    is_artifact_backed: bool
    verification_state: int | None = None
    verification_job_id: str | None = None
    source_artifact_ref: str | None = None
    local_serving_ref: str | None = None
    serving_artifact_id: str | None = None
    verification_failure_reason: str | None = None
    published: bool = False
    publication_eligible: bool = False
    publish_requested: bool = False
    acquired: bool = False


@dataclass(frozen=True, slots=True)
class RealizationRetainedBindingReport:
    binding_id: str
    binding_layout_id: str
    binding_value_id: str
    seal_generation: int
    local_serving_ref: str | None
    daemon_id: str
    daemon_session_id: str
    device_uuid: str
    member_id: str
    member_index: int
    member_count: int
    member_group_id: str | None
    reservation_bytes: int
    reservation_capability_id: str
    reservation_scope_digest: str
    readiness: str
    verification_state: str
    reservation_capability_expires_at_ms: int | None = None
    serving_artifact_id: str | None = None
    expires_at_ms: int | None = None
    staged_value: bool = False
    group_realization_transaction_id: str | None = None
    group_realization_version_set_id: str | None = None
    group_realization_part_id: str | None = None
    group_realization_staging_token: str | None = None
    group_realization_wait_for_publish: bool = False
    group_realization_wait_timeout_ms: int | None = None


@dataclass(frozen=True, slots=True)
class RealizationTargetSetMemberReport:
    member_id: str
    member_index: int
    member_count: int
    member_group_id: str | None
    device_uuid: str
    device: str | None
    binding_layout_id: str
    binding_value_id: str
    target_layout_digest: str | None
    copy_plan_digest: str | None
    source_selection_digest: str | None
    source_artifact_ref: str | None
    runtime_profile_digest: str | None
    placement_digest: str | None
    readiness: str
    verification_state: str
    reservation_bytes: int
    reservation_capability_id: str
    reservation_scope_digest: str
    staged_value: bool
    group_realization_transaction_id: str | None = None
    group_realization_version_set_id: str | None = None
    group_realization_part_id: str | None = None
    group_realization_wait_for_publish: bool = False
    group_realization_wait_timeout_ms: int | None = None


@dataclass(frozen=True, slots=True)
class RealizationTargetSetReport:
    group_id: str | None
    runtime: str | None
    topology_digest: str | None
    source_kind: str | None
    source_artifact_ref: str | None
    source_selection_mode: str
    member_count: int
    successful_member_count: int
    failed_member_count: int
    partial: bool
    readiness: str | None
    ready_member_count: int
    staged_member_count: int
    total_reservation_bytes: int
    same_daemon_session: bool
    group_realization_transaction_ids: tuple[str, ...]
    group_realization_version_set_ids: tuple[str, ...]
    acquire_claim_ids: tuple[str, ...]
    publish_barrier: bool
    members: tuple[RealizationTargetSetMemberReport, ...]
    failure_code_buckets: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RealizationPublicationReport:
    state: str
    artifact_ref: str | None
    operation_id: str | None
    replica_id: str | None
    lease_id: str | None
    device_uuid: str | None
    owner_pid: int | None
    binding_layout_id: str | None
    generation: str | None
    reason: str | None
    byte_space_kind: str | None
    byte_space_id: str | None
    binding_value_id: str | None


@dataclass(frozen=True, slots=True)
class RealizationMountedSourceReport:
    source_artifact_id: str
    promoted_artifact_id: str
    verify_checksums: bool
    generation: int | None
    canonical_index_bytes_len: int
    promoted_artifact_profile: str
    promoted_authority_scope: str


@dataclass(frozen=True, slots=True)
class RealizationModelRuntimeReport:
    framework: str
    device: str | None
    adapter_version: str | None
    runtime_abi_version: str | None
    topology_digest: str | None
    member_digest: str | None
    runtime_attachment_target_kind: str


@dataclass(frozen=True, slots=True)
class RealizationPublishabilityReport:
    publishable: bool
    publish_requested: bool = False
    publication_eligible: bool = False
    published: bool = False
    publication_state: str | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class RealizationExecutionCommitReport:
    actual_executor_path: str = "unknown"
    dominant_executor: str | None = None
    source: str | None = None
    requested_bytes: int = 0
    committed_bytes: int = 0
    direct_write_bytes: int = 0
    fallback_bytes: int = 0
    residual_bytes: int = 0
    actual_collective_committed_bytes: int = 0
    actual_local_typed_bytes: int = 0
    actual_generic_backend_bytes: int = 0
    collective_unique_source_bytes: int = 0
    collective_peer_transfer_bytes: int = 0
    collective_peak_temporary_bytes: int = 0
    collective_batch_count: int = 0
    collective_dedup_saving_bytes: int = 0
    collective_skip_reason: str | None = None
    direct_write_supported: bool = False
    collective_requested: bool = False
    collective_acknowledged: bool = False
    collective_used: bool = False
    execution_plan_kind: str | None = None
    plan_hash: str | None = None
    planner_version: str | None = None
    lane_allocation_bytes: Mapping[str, int] = field(default_factory=dict)
    committed_range_bytes: Mapping[str, int] = field(default_factory=dict)
    residual_fallback_range_bytes: Mapping[str, int] = field(default_factory=dict)
    planner_reject_reason_buckets: Mapping[str, int] = field(default_factory=dict)
    estimated_collective_peak_temporary_bytes: int = 0
    estimated_collective_batch_bytes: int = 0
    estimated_collective_dedup_saving_bytes: int = 0


@dataclass(frozen=True, slots=True)
class ArtifactRealizationReport:
    target_kind: RealizationTargetKind
    source_selection_digest: str
    target_layout_digest: str | None
    copy_plan_digest: str | None
    artifact_id: str
    view_id: str
    artifact_profile: str
    authority_scope: str
    generation_hint: int | None
    envelope: RealizationResourceEnvelope
    target_plan: RealizationTargetPlan | None = None
    strategy_plan: RealizationStrategyPlan | None = None
    representation_admission: RepresentationAdmissionPlan | None = None
    lifecycle_plan: RealizationLifecyclePlan | None = None
    materialize_sec: float | None = None
    tensor_bind_sec: float | None = None
    total_sec: float | None = None
    runtime_attach_sec: float | None = None
    runtime_finalize_sec: float | None = None
    source: str | None = None
    operation_id: str | None = None
    operation_backend: str | None = None
    risk_labels: tuple[str, ...] = ()
    materialization_diagnostics: object | None = None
    binding: RealizationBindingReport | None = None
    retained_bindings: tuple[RealizationRetainedBindingReport, ...] = ()
    target_set: RealizationTargetSetReport | None = None
    publication: RealizationPublicationReport | None = None
    mounted_source: RealizationMountedSourceReport | None = None
    model_runtime: RealizationModelRuntimeReport | None = None
    publishability: RealizationPublishabilityReport | None = None
    execution_commit: RealizationExecutionCommitReport | None = None
    execution_diagnostics: object | None = None
    source_bound_plan_diagnostics: object | None = None
    view_subset_hash: str = ""
    logical_layout_hash: str = ""
    selection_hash: str = ""

    def validate_for_handle(self, target_kind: RealizationTargetKind) -> None:
        if self.target_kind != target_kind:
            raise ArtifactError(
                "artifact realization handle target_kind does not match report",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if not self.source_selection_digest:
            raise ArtifactError(
                "artifact realization report requires source_selection_digest",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if not self.envelope.release_strictness:
            raise ArtifactError(
                "artifact realization envelope requires release_strictness",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if not self.envelope.mutability_contract:
            raise ArtifactError(
                "artifact realization envelope requires mutability_contract",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if self.target_plan is not None:
            self.envelope.validate_for_target(self.target_plan)
        if (
            self.envelope.fallback_reason_buckets
            and self.strategy_plan is not None
            and not self.strategy_plan.fallback_policy
        ):
            raise ArtifactError(
                "artifact realization fallback diagnostics require fallback_policy",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )


def artifact_realization_profile_payload(
    report: ArtifactRealizationReport,
) -> dict[str, Any]:
    envelope = report.envelope
    payload: dict[str, Any] = {
        "target_kind": report.target_kind,
        "artifact_id": report.artifact_id,
        "view_id": report.view_id,
        "artifact_profile": report.artifact_profile,
        "authority_scope": report.authority_scope,
        "source_selection_digest": report.source_selection_digest,
        "view_subset_hash": report.view_subset_hash,
        "logical_layout_hash": report.logical_layout_hash,
        "selection_hash": report.selection_hash,
        "target_layout_digest": report.target_layout_digest,
        "copy_plan_digest": report.copy_plan_digest,
        "operation_backend": report.operation_backend,
        "operation_id": report.operation_id,
        "source": report.source,
        "risk_labels": report.risk_labels,
        "materialize_sec": report.materialize_sec,
        "tensor_bind_sec": report.tensor_bind_sec,
        "total_sec": report.total_sec,
        "runtime_attach_sec": report.runtime_attach_sec,
        "runtime_finalize_sec": report.runtime_finalize_sec,
        "envelope_backing_kind": envelope.backing_kind,
        "envelope_export_kind": envelope.export_kind,
        "envelope_projection_kind": envelope.projection_kind,
        "envelope_owner_kind": envelope.owner_kind,
        "envelope_release_policy": envelope.release_policy,
        "envelope_release_strictness": envelope.release_strictness,
        "envelope_export_lifetime_kind": envelope.export_lifetime_kind,
        "envelope_mutability_contract": envelope.mutability_contract,
        "direct_write_bytes": envelope.direct_write_bytes,
        "copy_bytes": envelope.copy_bytes,
        "copy_count": envelope.copy_count,
        "mmap_bytes": envelope.mmap_bytes,
        "cuda_ipc_open_count": envelope.cuda_ipc_open_count,
        "cpu_memfd_fd_count": envelope.cpu_memfd_fd_count,
        "temporary_replica_bytes": envelope.temporary_replica_bytes,
        "retained_bytes": envelope.retained_bytes,
        "fallback_reason_buckets": dict(envelope.fallback_reason_buckets),
    }
    if report.retained_bindings:
        payload["retained_binding_count"] = len(report.retained_bindings)
        payload["retained_binding_reservation_bytes"] = sum(
            int(retained.reservation_bytes) for retained in report.retained_bindings
        )
        payload["retained_binding_capability_ids"] = tuple(
            retained.reservation_capability_id for retained in report.retained_bindings
        )
        payload["retained_binding_capability_expires_at_ms"] = tuple(
            retained.reservation_capability_expires_at_ms
            for retained in report.retained_bindings
        )
        payload["retained_binding_readiness"] = tuple(
            retained.readiness for retained in report.retained_bindings
        )
        payload["retained_binding_verification_states"] = tuple(
            retained.verification_state for retained in report.retained_bindings
        )
    if report.target_plan is not None:
        payload["target_plan_kind"] = report.target_plan.kind
        payload["target_plan_member_count"] = report.target_plan.member_count
    if report.strategy_plan is not None:
        payload["strategy_source_selection_mode"] = (
            report.strategy_plan.source_selection_mode
        )
        payload["strategy_source_coordination"] = (
            report.strategy_plan.source_coordination
        )
        payload["strategy_collective_policy"] = report.strategy_plan.collective_policy
        payload["strategy_fallback_policy"] = report.strategy_plan.fallback_policy
    if report.lifecycle_plan is not None:
        payload["lifecycle_capability"] = report.lifecycle_plan.capability
        payload["lifecycle_staged_value_count"] = (
            report.lifecycle_plan.staged_value_count
        )
        payload["lifecycle_acquire_claim_count"] = (
            report.lifecycle_plan.acquire_claim_count
        )
        payload["lifecycle_publish_barrier"] = report.lifecycle_plan.publish_barrier
    if report.execution_commit is not None:
        payload["execution_actual_executor_path"] = (
            report.execution_commit.actual_executor_path
        )
        payload["execution_committed_bytes"] = report.execution_commit.committed_bytes
        payload["execution_fallback_bytes"] = report.execution_commit.fallback_bytes
        payload["execution_residual_bytes"] = report.execution_commit.residual_bytes
        payload["execution_plan_kind"] = report.execution_commit.execution_plan_kind
        payload["execution_plan_hash"] = report.execution_commit.plan_hash
        payload["execution_lane_allocation_bytes"] = dict(
            report.execution_commit.lane_allocation_bytes
        )
        payload["execution_committed_range_bytes"] = dict(
            report.execution_commit.committed_range_bytes
        )
        payload["execution_residual_fallback_range_bytes"] = dict(
            report.execution_commit.residual_fallback_range_bytes
        )
        payload["execution_planner_reject_reason_buckets"] = dict(
            report.execution_commit.planner_reject_reason_buckets
        )
    if report.publishability is not None:
        payload["publishable"] = report.publishability.publishable
        payload["publish_requested"] = report.publishability.publish_requested
        payload["published"] = report.publishability.published
    return payload


def emit_artifact_realization_profile_event(
    report: ArtifactRealizationReport,
) -> None:
    emit_tensorcast_profile_event(
        "tensorcast",
        "artifact.realize",
        logger=logger,
        payload=artifact_realization_profile_payload(report),
    )


def _tensor_nbytes(tensor: object) -> int:
    try:
        return int(tensor.element_size()) * int(tensor.numel())
    except (AttributeError, TypeError, ValueError):
        return 0


def _tensor_is_cuda(tensor: object) -> bool:
    try:
        return bool(tensor.is_cuda)
    except (AttributeError, TypeError, ValueError):
        return False


def _safe_attr(value: object | None, name: str) -> object | None:
    if value is None:
        return None
    try:
        return getattr(value, name)
    except AttributeError:
        return None


def _optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _optional_int(value: object | None) -> int | None:
    if value is None:
        return None
    value_any: Any = value
    with contextlib.suppress(TypeError, ValueError):
        return int(value_any)
    return None


def _literal_value(value: object | None) -> str | None:
    if value is None:
        return None
    inner = _safe_attr(value, "value")
    return _optional_str(inner if inner is not None else value)


def _object_digest(label: str, value: object | None) -> str | None:
    if value is None:
        return None
    payload = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return _sha256_hex(label.encode("utf-8"), payload)


def _mapping_copy(value: object | None) -> dict[str, object] | None:
    if isinstance(value, Mapping):
        return dict(value)
    return None


def _mapping_int(mapping: Mapping[str, object] | None, key: str) -> int:
    if mapping is None:
        return 0
    return _optional_int(mapping.get(key)) or 0


def _execution_int(execution_diagnostics: object | None, name: str) -> int:
    return _optional_int(_safe_attr(execution_diagnostics, name)) or 0


def _plan_int(source_bound_plan_diagnostics: object | None, name: str) -> int:
    return _optional_int(_safe_attr(source_bound_plan_diagnostics, name)) or 0


def _attr_int_mapping(value: object | None, name: str) -> dict[str, int]:
    mapping = _safe_attr(value, name)
    if not isinstance(mapping, Mapping):
        return {}
    result: dict[str, int] = {}
    for key, count in mapping.items():
        count_int = _optional_int(count)
        if count_int is not None:
            result[str(key)] = count_int
    return result


def _actual_executor_path_for(
    execution_diagnostics: object | None,
    *,
    collective_bytes: int,
    local_typed_bytes: int,
    generic_bytes: int,
    fallback_bytes: int,
    residual_bytes: int,
) -> str:
    fallback_present = generic_bytes > 0 or fallback_bytes > 0 or residual_bytes > 0
    if collective_bytes > 0 and (local_typed_bytes > 0 or fallback_present):
        return "mixed_collective"
    if collective_bytes > 0:
        return "collective"
    if local_typed_bytes > 0 and fallback_present:
        return "mixed_local_generic"
    if local_typed_bytes > 0:
        return "local_typed"
    if fallback_present:
        return "generic_fallback"
    return _optional_str(_safe_attr(execution_diagnostics, "dominant_executor")) or (
        "unknown"
    )


def execution_commit_report_for(
    *,
    execution_diagnostics: object | None,
    source_bound_plan_diagnostics: object | None = None,
    source: str | None = None,
    requested_bytes: int = 0,
) -> RealizationExecutionCommitReport | None:
    if execution_diagnostics is None and source_bound_plan_diagnostics is None:
        return None

    collective_bytes = _execution_int(
        execution_diagnostics,
        "actual_collective_committed_bytes",
    )
    local_typed_bytes = _execution_int(
        execution_diagnostics,
        "actual_local_typed_bytes",
    )
    generic_bytes = _execution_int(
        execution_diagnostics,
        "actual_generic_backend_bytes",
    )
    fallback_bytes = _execution_int(execution_diagnostics, "fallback_bytes")
    residual_bytes = _execution_int(execution_diagnostics, "residual_bytes")
    planned_generic_residual_bytes = _plan_int(
        source_bound_plan_diagnostics,
        "planned_generic_residual_bytes",
    )
    return RealizationExecutionCommitReport(
        actual_executor_path=_actual_executor_path_for(
            execution_diagnostics,
            collective_bytes=collective_bytes,
            local_typed_bytes=local_typed_bytes,
            generic_bytes=generic_bytes,
            fallback_bytes=fallback_bytes,
            residual_bytes=residual_bytes,
        ),
        dominant_executor=_optional_str(
            _safe_attr(execution_diagnostics, "dominant_executor")
        ),
        source=source,
        requested_bytes=max(0, int(requested_bytes)),
        committed_bytes=collective_bytes + local_typed_bytes + generic_bytes,
        direct_write_bytes=collective_bytes + local_typed_bytes,
        fallback_bytes=fallback_bytes,
        residual_bytes=residual_bytes,
        actual_collective_committed_bytes=collective_bytes,
        actual_local_typed_bytes=local_typed_bytes,
        actual_generic_backend_bytes=generic_bytes,
        collective_unique_source_bytes=_execution_int(
            execution_diagnostics,
            "collective_unique_source_bytes",
        ),
        collective_peer_transfer_bytes=_execution_int(
            execution_diagnostics,
            "collective_peer_transfer_bytes",
        ),
        collective_peak_temporary_bytes=_execution_int(
            execution_diagnostics,
            "collective_peak_temporary_bytes",
        ),
        collective_batch_count=_execution_int(
            execution_diagnostics,
            "collective_batch_count",
        ),
        collective_dedup_saving_bytes=_execution_int(
            execution_diagnostics,
            "collective_dedup_saving_bytes",
        ),
        collective_skip_reason=_optional_str(
            _safe_attr(execution_diagnostics, "collective_skip_reason")
        ),
        direct_write_supported=bool(
            _safe_attr(execution_diagnostics, "direct_write_supported")
        ),
        collective_requested=bool(
            _safe_attr(execution_diagnostics, "collective_requested")
        ),
        collective_acknowledged=bool(
            _safe_attr(execution_diagnostics, "collective_acknowledged")
        ),
        collective_used=bool(_safe_attr(execution_diagnostics, "collective_used")),
        execution_plan_kind=_optional_str(
            _safe_attr(source_bound_plan_diagnostics, "execution_plan_kind")
        ),
        plan_hash=_optional_str(_safe_attr(source_bound_plan_diagnostics, "plan_hash")),
        planner_version=_optional_str(
            _safe_attr(source_bound_plan_diagnostics, "planner_version")
        ),
        lane_allocation_bytes={
            "collective_candidate": _plan_int(
                source_bound_plan_diagnostics,
                "planned_collective_candidate_bytes",
            ),
            "collective_admitted": _plan_int(
                source_bound_plan_diagnostics,
                "planned_collective_admitted_bytes",
            ),
            "local_typed": _plan_int(
                source_bound_plan_diagnostics,
                "planned_local_typed_bytes",
            ),
            "non_admitted_typed": _plan_int(
                source_bound_plan_diagnostics,
                "planned_non_admitted_typed_bytes",
            ),
            "generic_residual": planned_generic_residual_bytes,
            "collective_lowered": _plan_int(
                source_bound_plan_diagnostics,
                "collective_lowered_bytes",
            ),
        },
        committed_range_bytes={
            "collective": collective_bytes,
            "local_typed": local_typed_bytes,
            "generic_backend": generic_bytes,
        },
        residual_fallback_range_bytes={
            "fallback": fallback_bytes,
            "residual": residual_bytes,
            "planned_generic_residual": planned_generic_residual_bytes,
        },
        planner_reject_reason_buckets=_attr_int_mapping(
            source_bound_plan_diagnostics,
            "planner_reject_reason_buckets",
        ),
        estimated_collective_peak_temporary_bytes=_plan_int(
            source_bound_plan_diagnostics,
            "estimated_collective_peak_temporary_bytes",
        ),
        estimated_collective_batch_bytes=_plan_int(
            source_bound_plan_diagnostics,
            "estimated_collective_batch_bytes",
        ),
        estimated_collective_dedup_saving_bytes=_plan_int(
            source_bound_plan_diagnostics,
            "estimated_collective_dedup_saving_bytes",
        ),
    )


def _has_positive_bucket(mapping: Mapping[str, int]) -> bool:
    return any(int(value) > 0 for value in mapping.values())


def _strategy_source_policy_for_options(
    options: object | None,
    *,
    lease_mode: object | None = None,
) -> dict[str, object]:
    retrieval = _safe_attr(options, "source")
    execution_topology = _safe_attr(options, "execution_topology")
    return {
        "preference": _literal_value(_safe_attr(retrieval, "preference")) or "auto",
        "allow_p2p": (
            bool(_safe_attr(retrieval, "allow_p2p"))
            if _safe_attr(retrieval, "allow_p2p") is not None
            else True
        ),
        "allow_disk": (
            bool(_safe_attr(retrieval, "allow_disk"))
            if _safe_attr(retrieval, "allow_disk") is not None
            else True
        ),
        "wait_for_shared_disk_ms": _optional_int(
            _safe_attr(options, "wait_for_shared_disk_ms")
        )
        or 0,
        "verify_checksums": (
            bool(_safe_attr(options, "verify_checksums"))
            if _safe_attr(options, "verify_checksums") is not None
            else True
        ),
        "enable_verification": (
            bool(_safe_attr(options, "enable_verification"))
            if _safe_attr(options, "enable_verification") is not None
            else True
        ),
        "region_backed_mode": _literal_value(_safe_attr(options, "region_backed_mode"))
        or "auto",
        "export_policy": _optional_str(_safe_attr(options, "export_policy")) or "never",
        "wait_for_completion": (
            bool(_safe_attr(options, "wait_for_completion"))
            if _safe_attr(options, "wait_for_completion") is not None
            else True
        ),
        "transport_hold_ms": _optional_int(_safe_attr(options, "transport_hold_ms")),
        "lease_mode": _literal_value(lease_mode),
        "topology_collective_policy": _literal_value(
            _safe_attr(execution_topology, "collective_policy")
        ),
        "source_locality": _literal_value(
            _safe_attr(execution_topology, "source_locality")
        )
        or "auto",
        "source_sharing_domain": _optional_str(
            _safe_attr(execution_topology, "source_sharing_domain")
        ),
    }


def strategy_plan_for_execution(
    *,
    envelope: RealizationResourceEnvelope,
    execution_commit: RealizationExecutionCommitReport | None = None,
    options: object | None = None,
    ctx: object | None = None,
    lease_mode: object | None = None,
) -> RealizationStrategyPlan:
    has_execution_fallback = False
    collective_policy = "disabled"
    if execution_commit is not None:
        has_execution_fallback = (
            execution_commit.fallback_bytes > 0
            or execution_commit.residual_bytes > 0
            or execution_commit.actual_generic_backend_bytes > 0
            or _has_positive_bucket(execution_commit.residual_fallback_range_bytes)
        )
        if execution_commit.collective_used:
            collective_policy = "collective_used"
        elif execution_commit.collective_acknowledged:
            collective_policy = "collective_acknowledged"
        elif execution_commit.collective_requested:
            collective_policy = "collective_requested"

    fallback_policy = (
        "generic_fallback"
        if (
            has_execution_fallback
            or _has_positive_bucket(envelope.fallback_reason_buckets)
        )
        else "fail_closed"
    )
    return RealizationStrategyPlan(
        source_policy=_strategy_source_policy_for_options(
            options,
            lease_mode=lease_mode,
        ),
        fallback_policy=fallback_policy,
        retry_policy={
            "fallback_reason_buckets": dict(envelope.fallback_reason_buckets),
        },
        deadline_ms=_optional_int(_safe_attr(ctx, "deadline_ms")),
        collective_policy=collective_policy,
    )


def _total_tensor_bytes(tensors: Mapping[str, object] | None) -> int:
    if tensors is None:
        return 0
    return sum(_tensor_nbytes(tensor) for tensor in tensors.values())


def envelope_for_tensor_dict(
    tensors: Mapping[str, object],
    *,
    source: str | None,
    retry_reason_buckets: Mapping[str, int] | None = None,
) -> RealizationResourceEnvelope:
    total_bytes = sum(_tensor_nbytes(tensor) for tensor in tensors.values())
    has_cuda = any(_tensor_is_cuda(tensor) for tensor in tensors.values())
    export_kind = "cuda_ipc" if has_cuda else "cpu_memfd"
    return RealizationResourceEnvelope(
        backing_kind="daemon_temporary_replica",
        export_kind=export_kind,
        projection_kind="tensor_dict",
        owner_kind="tensor_projection_owner",
        release_policy=("release_export_token", "unload_temporary_replica"),
        mutability_contract=(
            "read_mostly_private_copy" if not has_cuda else "read_mostly"
        ),
        release_strictness="strict",
        export_lifetime_kind="token_backed",
        mmap_bytes=0 if has_cuda else total_bytes,
        cuda_ipc_open_count=len(tensors) if has_cuda else 0,
        cpu_memfd_fd_count=0 if has_cuda else 1 if tensors else 0,
        temporary_replica_bytes=total_bytes,
        fallback_reason_buckets=dict(retry_reason_buckets or {}),
    )


def envelope_for_caller_tensors(
    tensors: Mapping[str, object],
    *,
    used_region_backed: bool | None = None,
    actual_total_bytes: int | None = None,
    fallback_reason_buckets: Mapping[str, int] | None = None,
) -> RealizationResourceEnvelope:
    planned_total_bytes = _total_tensor_bytes(tensors)
    materialized_total_bytes = (
        planned_total_bytes if actual_total_bytes is None else int(actual_total_bytes)
    )
    has_cuda = any(_tensor_is_cuda(tensor) for tensor in tensors.values())
    direct_write = has_cuda and used_region_backed is not False
    movement_bytes = planned_total_bytes if direct_write else materialized_total_bytes
    release_policy: tuple[str, ...] = ("unregister_target_region",)
    if not direct_write:
        release_policy = (
            "unregister_target_region",
            "unload_temporary_replica_on_fallback",
        )
    return RealizationResourceEnvelope(
        backing_kind="caller_region_or_temporary_replica",
        export_kind=(
            "registered_vram_region_direct_write" if direct_write else "temporary_copy"
        ),
        projection_kind="completion",
        owner_kind="caller",
        release_policy=release_policy,
        mutability_contract="caller_mutable",
        release_strictness="strict",
        export_lifetime_kind="none",
        direct_write_bytes=movement_bytes if direct_write else 0,
        copy_bytes=0 if direct_write else movement_bytes,
        copy_count=0 if direct_write else len(tensors),
        temporary_replica_bytes=0 if direct_write else movement_bytes,
        fallback_reason_buckets=dict(fallback_reason_buckets or {}),
    )


def envelope_for_target_region_registration(
    tensors: Mapping[str, object] | None = None,
    *,
    total_bytes: int | None = None,
) -> RealizationResourceEnvelope:
    region_bytes = (
        _total_tensor_bytes(tensors) if total_bytes is None else int(total_bytes)
    )
    return RealizationResourceEnvelope(
        backing_kind="caller_registered_region",
        export_kind="registered_vram_region",
        projection_kind="target_region_registration",
        owner_kind="caller",
        release_policy=("unregister_target_region",),
        mutability_contract="caller_mutable",
        release_strictness="strict",
        export_lifetime_kind="none",
        direct_write_bytes=max(0, region_bytes),
    )


def envelope_for_retained_replica(
    *,
    total_bytes: int,
    device_kind: str,
    retry_reason_buckets: Mapping[str, int] | None = None,
) -> RealizationResourceEnvelope:
    return RealizationResourceEnvelope(
        backing_kind="daemon_retained_replica",
        export_kind=f"{device_kind}_retained_replica",
        projection_kind="prefetch_handoff",
        owner_kind="daemon_retention_ticket",
        release_policy=("release_replica_ticket",),
        mutability_contract="read_only",
        release_strictness="strict",
        export_lifetime_kind="daemon_retained",
        retained_bytes=max(0, int(total_bytes)),
        fallback_reason_buckets=dict(retry_reason_buckets or {}),
    )


def retained_binding_reports_for(
    result: object,
) -> tuple[RealizationRetainedBindingReport, ...]:
    members = _safe_attr(result, "members")
    if isinstance(members, Sequence) and not isinstance(members, (str, bytes)):
        return tuple(_retained_binding_report_for_member(member) for member in members)
    return (_retained_binding_report_for_member(result),)


def envelope_for_retained_binding(
    retained_bindings: Sequence[RealizationRetainedBindingReport],
    *,
    target_set: bool = False,
) -> RealizationResourceEnvelope:
    reservation_bytes = sum(
        int(report.reservation_bytes) for report in retained_bindings
    )
    release_policy = ["release_binding_reservation"]
    if any(report.staged_value for report in retained_bindings):
        release_policy.append("release_group_staged_acquire")
    return RealizationResourceEnvelope(
        backing_kind=(
            "daemon_retained_binding_set" if target_set else "daemon_retained_binding"
        ),
        export_kind="binding_reservation",
        projection_kind="prefetch_handoff",
        owner_kind="binding_reservation_capability",
        release_policy=tuple(release_policy),
        mutability_contract="binding_controlled_read_only",
        release_strictness="strict",
        export_lifetime_kind="daemon_retained",
        retained_bytes=max(0, int(reservation_bytes)),
    )


def envelope_for_target_set(
    retained_bindings: Sequence[RealizationRetainedBindingReport],
) -> RealizationResourceEnvelope:
    retained = tuple(retained_bindings)
    if not retained:
        raise ArtifactError(
            "target_set envelope requires at least one retained binding member",
            status_code="DATA_LOSS",
            retryable=False,
        )
    reservation_bytes = sum(int(report.reservation_bytes) for report in retained)
    release_policy = ["release_binding_reservations"]
    if any(report.staged_value for report in retained):
        release_policy.append("release_group_staged_acquire")
    return RealizationResourceEnvelope(
        backing_kind="daemon_retained_binding_set",
        export_kind="binding_reservation_set",
        projection_kind="target_set",
        owner_kind="binding_reservation_capability_set",
        release_policy=tuple(release_policy),
        mutability_contract="binding_controlled_read_only",
        release_strictness="strict",
        export_lifetime_kind="daemon_retained",
        retained_bytes=max(0, int(reservation_bytes)),
    )


def envelope_for_runtime_attachment(
    tensors: Mapping[str, object],
    *,
    retained: bool,
    reservation_bytes: int = 0,
    fallback_reason_buckets: Mapping[str, int] | None = None,
) -> RealizationResourceEnvelope:
    total_bytes = _total_tensor_bytes(tensors)
    has_cuda = any(_tensor_is_cuda(tensor) for tensor in tensors.values())
    retained_bytes = max(0, int(reservation_bytes))
    if retained and retained_bytes == 0:
        retained_bytes = total_bytes
    return RealizationResourceEnvelope(
        backing_kind=(
            "daemon_retained_binding" if retained else "daemon_binding_value"
        ),
        export_kind=(
            "fresh_retained_acquire_export" if retained else "binding_restore_export"
        ),
        projection_kind="runtime_attachment",
        owner_kind="runtime_attachment",
        release_policy=(
            ("close_runtime_attachment", "release_placement_lease")
            if retained
            else ("close_runtime_attachment", "close_binding")
        ),
        mutability_contract="runtime_adapter_owned",
        release_strictness="strict",
        export_lifetime_kind="runtime_attachment",
        cuda_ipc_open_count=len(tensors) if has_cuda else 0,
        cpu_memfd_fd_count=0 if has_cuda else 1 if tensors else 0,
        retained_bytes=retained_bytes,
        fallback_reason_buckets=dict(fallback_reason_buckets or {}),
    )


def envelope_for_publication(
    *,
    projection: object,
    binding: object | None = None,
) -> RealizationResourceEnvelope:
    byte_space_kind = _optional_str(_safe_attr(projection, "byte_space_kind"))
    retained_bytes = _optional_int(_safe_attr(binding, "size_bytes")) or 0
    if retained_bytes == 0:
        current_value = _safe_attr(binding, "current_value")
        retained_bytes = _optional_int(_safe_attr(current_value, "size_bytes")) or 0
    release_policy = ["retire_published_replica"]
    if _safe_attr(projection, "lease_id") is not None:
        release_policy.append("release_publication_lease")
    return RealizationResourceEnvelope(
        backing_kind="daemon_published_replica",
        export_kind=(
            f"{byte_space_kind}_publication_lease"
            if byte_space_kind
            else "publication_lease"
        ),
        projection_kind="published_replica",
        owner_kind="runtime_publication",
        release_policy=tuple(release_policy),
        mutability_contract="published_read_only",
        release_strictness="strict",
        export_lifetime_kind="publication_lease",
        retained_bytes=max(0, int(retained_bytes)),
    )


def envelope_for_mounted_source(
    *,
    canonical_index_bytes: bytes,
) -> RealizationResourceEnvelope:
    return RealizationResourceEnvelope(
        backing_kind="daemon_mounted_source",
        export_kind="metadata_attestation",
        projection_kind="artifact_identity",
        owner_kind="daemon_control_plane",
        release_policy=("drop_promotion_handle",),
        mutability_contract="read_only_attested_source",
        release_strictness="strict",
        export_lifetime_kind="daemon_attested",
        retained_bytes=max(0, len(canonical_index_bytes)),
    )


def mounted_source_target_digest(
    *,
    source_artifact_id: str,
    promoted_artifact_id: str,
    canonical_index_bytes: bytes,
) -> str:
    return _sha256_hex(
        b"mounted-source-promotion",
        str(source_artifact_id).encode("utf-8"),
        str(promoted_artifact_id).encode("utf-8"),
        canonical_index_bytes,
    )


def risk_labels_for_target(
    target_plan: RealizationTargetPlan,
    envelope: RealizationResourceEnvelope,
    *,
    source_selection_digest: str | None = None,
    publishable: bool = False,
    extra: Sequence[str] = (),
) -> tuple[str, ...]:
    labels: list[str] = []

    def add(label: str) -> None:
        if label and label not in labels:
            labels.append(label)

    if source_selection_digest:
        add("authority")
        add("identity")
    if (
        target_plan.target_layout_digest
        or target_plan.binding_layout_id
        or target_plan.mapped_view_id
    ):
        add("identity")
    if envelope.release_policy:
        add("lifecycle")
    if envelope.export_lifetime_kind in {
        "token_backed",
        "daemon_retained",
        "runtime_attachment",
        "publication_lease",
    }:
        add("lease_strength")
    if envelope.mutability_contract:
        add("mutability")
    if (
        envelope.direct_write_bytes
        or envelope.copy_bytes
        or envelope.copy_count
        or envelope.mmap_bytes
        or envelope.cuda_ipc_open_count
        or envelope.cpu_memfd_fd_count
        or envelope.temporary_replica_bytes
        or envelope.fallback_reason_buckets
    ):
        add("hidden_movement_cost")
    if target_plan.kind in {"retained_replica", "retained_binding", "target_set"}:
        add("async_continuation")
    if target_plan.kind == "target_set" or (target_plan.member_count or 0) > 1:
        add("target_set")
    if (
        target_plan.kind == "publication"
        or publishable
        or any("publish" in policy for policy in envelope.release_policy)
    ):
        add("publication")
    for label in extra:
        add(str(label))
    return tuple(labels)


def publishability_report_for(
    *,
    binding: RealizationBindingReport | None = None,
    publication: RealizationPublicationReport | None = None,
    publish_requested: bool = False,
) -> RealizationPublishabilityReport:
    if publication is not None:
        published = bool(publication.replica_id or publication.artifact_ref)
        return RealizationPublishabilityReport(
            publishable=True,
            publish_requested=True,
            publication_eligible=True,
            published=published,
            publication_state=publication.state or None,
            reason=publication.reason,
        )
    if binding is not None:
        publishable = (
            binding.publication_eligible
            or binding.published
            or bool(binding.publish_requested)
            or bool(publish_requested)
        )
        return RealizationPublishabilityReport(
            publishable=publishable,
            publish_requested=bool(binding.publish_requested or publish_requested),
            publication_eligible=bool(binding.publication_eligible),
            published=bool(binding.published),
            publication_state="published" if binding.published else None,
            reason=None if publishable else "not_publication_eligible",
        )
    return RealizationPublishabilityReport(
        publishable=False,
        publish_requested=bool(publish_requested),
        publication_eligible=False,
        published=False,
        publication_state=None,
        reason="not_publishable_target",
    )


def _retained_binding_report_for_member(
    result: object,
) -> RealizationRetainedBindingReport:
    binding_ref = _safe_attr(result, "binding_value_ref")
    capability = _safe_attr(result, "reservation_capability")
    member = _safe_attr(result, "member")
    binding_id = _optional_str(_safe_attr(binding_ref, "binding_id"))
    binding_layout_id = _optional_str(_safe_attr(binding_ref, "binding_layout_id"))
    binding_value_id = _optional_str(_safe_attr(binding_ref, "binding_value_id"))
    seal_generation = _optional_int(_safe_attr(binding_ref, "seal_generation"))
    if (
        not binding_id
        or not binding_layout_id
        or not binding_value_id
        or seal_generation is None
    ):
        raise ArtifactError(
            "retained binding report requires a complete binding value reference",
            status_code="DATA_LOSS",
            retryable=False,
        )
    daemon_id = _optional_str(_safe_attr(result, "daemon_id"))
    daemon_session_id = _optional_str(_safe_attr(result, "daemon_session_id"))
    device_uuid = _optional_str(_safe_attr(result, "device_uuid"))
    member_id = _optional_str(_safe_attr(member, "member_id"))
    member_index = _optional_int(_safe_attr(member, "member_index"))
    member_count = _optional_int(_safe_attr(member, "member_count"))
    reservation_bytes = _optional_int(_safe_attr(result, "reservation_bytes"))
    capability_id = _optional_str(_safe_attr(capability, "capability_id"))
    scope_digest = _optional_str(_safe_attr(capability, "scope_digest"))
    readiness = _literal_value(_safe_attr(result, "readiness"))
    verification_state = _literal_value(_safe_attr(result, "verification_state"))
    if (
        not daemon_id
        or not daemon_session_id
        or not device_uuid
        or not member_id
        or member_index is None
        or member_count is None
        or reservation_bytes is None
        or not capability_id
        or not scope_digest
        or not readiness
        or not verification_state
    ):
        raise ArtifactError(
            "retained binding report requires daemon, member, reservation, readiness, and verification facts",
            status_code="DATA_LOSS",
            retryable=False,
        )
    group_acquire = _safe_attr(result, "group_realization_acquire")
    return RealizationRetainedBindingReport(
        binding_id=binding_id,
        binding_layout_id=binding_layout_id,
        binding_value_id=binding_value_id,
        seal_generation=seal_generation,
        local_serving_ref=_optional_str(_safe_attr(result, "local_serving_ref")),
        daemon_id=daemon_id,
        daemon_session_id=daemon_session_id,
        device_uuid=device_uuid,
        member_id=member_id,
        member_index=member_index,
        member_count=member_count,
        member_group_id=_optional_str(_safe_attr(member, "group_id")),
        reservation_bytes=reservation_bytes,
        reservation_capability_id=capability_id,
        reservation_scope_digest=scope_digest,
        reservation_capability_expires_at_ms=_optional_int(
            _safe_attr(capability, "expires_at_ms")
        ),
        readiness=readiness,
        verification_state=verification_state,
        serving_artifact_id=_optional_str(_safe_attr(result, "serving_artifact_id")),
        expires_at_ms=_optional_int(_safe_attr(result, "expires_at_ms")),
        staged_value=bool(_safe_attr(result, "staged_value")),
        group_realization_transaction_id=_optional_str(
            _safe_attr(group_acquire, "transaction_id")
        ),
        group_realization_version_set_id=_optional_str(
            _safe_attr(group_acquire, "version_set_id")
        ),
        group_realization_part_id=_optional_str(_safe_attr(group_acquire, "part_id")),
        group_realization_staging_token=_optional_str(
            _safe_attr(group_acquire, "staging_token")
        ),
        group_realization_wait_for_publish=bool(
            _safe_attr(group_acquire, "wait_for_publish")
        ),
        group_realization_wait_timeout_ms=_optional_int(
            _safe_attr(group_acquire, "wait_timeout_ms")
        ),
    )


def _target_members_by_member_id(target: object | None) -> dict[str, object]:
    members = _safe_attr(target, "members")
    if not isinstance(members, Sequence) or isinstance(members, (str, bytes)):
        return {}
    keyed: dict[str, object] = {}
    for target_member in members:
        member = _safe_attr(target_member, "member")
        member_id = _optional_str(_safe_attr(member, "member_id"))
        if member_id:
            keyed[member_id] = target_member
    return keyed


def _target_member_layout_digest(target_member: object | None) -> str | None:
    resolved_layout = _safe_attr(target_member, "resolved_layout")
    layout_hash = _optional_str(_safe_attr(resolved_layout, "target_layout_hash"))
    if layout_hash:
        return layout_hash
    layout_bytes = _safe_attr(resolved_layout, "target_layout")
    if isinstance(layout_bytes, bytes) and layout_bytes:
        return _sha256_hex(b"target-set-member-layout", layout_bytes)
    return None


def _target_member_copy_plan_digest(target_member: object | None) -> str | None:
    resolved_layout = _safe_attr(target_member, "resolved_layout")
    spec_digest = _optional_str(_safe_attr(resolved_layout, "spec_digest"))
    if spec_digest:
        return spec_digest
    copy_plan_bytes = _safe_attr(resolved_layout, "copy_plan_bytes")
    if isinstance(copy_plan_bytes, bytes) and copy_plan_bytes:
        return _sha256_hex(b"target-set-member-copy-plan", copy_plan_bytes)
    return None


def _target_member_runtime_profile_digest(
    target_member: object | None,
    target: object | None,
) -> str | None:
    payload = {
        "runtime": _optional_str(_safe_attr(target_member, "runtime"))
        or _optional_str(_safe_attr(target, "runtime")),
        "model_config_digest": _optional_str(
            _safe_attr(target_member, "model_config_digest")
        ),
        "load_config_digest": _optional_str(
            _safe_attr(target_member, "load_config_digest")
        ),
        "runtime_build_digest": _optional_str(
            _safe_attr(target_member, "runtime_build_digest")
        ),
    }
    if not any(payload.values()):
        return None
    return _object_digest("target-set-runtime-profile", payload)


def _target_member_placement_digest(
    target_member: object | None,
    target: object | None,
) -> str | None:
    member = _safe_attr(target_member, "member")
    topology = _safe_attr(target_member, "topology") or _safe_attr(target, "topology")
    payload = {
        "device": _optional_str(_safe_attr(target_member, "device")),
        "device_uuid": _optional_str(_safe_attr(target_member, "device_uuid")),
        "member_id": _optional_str(_safe_attr(member, "member_id")),
        "member_index": _optional_int(_safe_attr(member, "member_index")),
        "member_count": _optional_int(_safe_attr(member, "member_count")),
        "group_id": _optional_str(_safe_attr(member, "group_id")),
        "schema_topology_digest": _optional_str(
            _safe_attr(topology, "schema_topology_digest")
        ),
        "admission_topology_digest": _optional_str(
            _safe_attr(topology, "admission_topology_digest")
        ),
    }
    if not any(value is not None for value in payload.values()):
        return None
    return _object_digest("target-set-placement", payload)


def _target_member_source(
    target_member: object | None, target: object | None
) -> object:
    member_source = _safe_attr(target_member, "source")
    if member_source is not None:
        return member_source
    return _safe_attr(target, "source")


def _source_members_by_member_id(source: object | None) -> dict[str, object]:
    members = _safe_attr(source, "members")
    if not isinstance(members, Sequence) or isinstance(members, (str, bytes)):
        return {}
    keyed: dict[str, object] = {}
    for source_member in members:
        member = _safe_attr(source_member, "member")
        member_id = _optional_str(_safe_attr(member, "member_id"))
        if member_id:
            keyed[member_id] = source_member
    return keyed


def _source_selection_mode(
    digests: Sequence[str | None],
    *,
    source: object | None,
) -> str:
    source_kind = _literal_value(_safe_attr(source, "source_kind"))
    if source_kind == "runtime_artifact_set":
        source_members = _source_members_by_member_id(source)
        artifact_refs = {
            artifact_ref
            for source_member in source_members.values()
            if (
                artifact_ref := _optional_str(_safe_attr(source_member, "artifact_ref"))
            )
        }
        if len(artifact_refs) > 1:
            return "per_part_selection"
    non_empty = {digest for digest in digests if digest}
    if len(non_empty) <= 1:
        return "same_selection"
    return "per_part_selection"


def _count_by_value(values: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _failure_code_buckets(result: object | None) -> dict[str, int]:
    failures = _safe_attr(result, "member_failures")
    if not isinstance(failures, Sequence) or isinstance(failures, (str, bytes)):
        return {}
    codes = [
        code
        for failure in failures
        if (code := _optional_str(_safe_attr(failure, "code"))) is not None
    ]
    return _count_by_value(codes)


def _target_set_publish_barrier(
    retained: Sequence[RealizationRetainedBindingReport],
    *,
    target: object | None,
    result: object | None,
) -> bool:
    if any(report.group_realization_wait_for_publish for report in retained):
        return True
    for owner in (result, target):
        if bool(_safe_attr(owner, "publish_barrier")) or bool(
            _safe_attr(owner, "wait_for_publish")
        ):
            return True
    return False


def target_set_report_for_retained_bindings(
    retained_bindings: Sequence[RealizationRetainedBindingReport],
    *,
    result: object | None = None,
    target: object | None = None,
    source_selection_digest: str | None = None,
) -> RealizationTargetSetReport:
    retained = tuple(retained_bindings)
    if not retained:
        raise ArtifactError(
            "target_set report requires at least one retained binding member",
            status_code="DATA_LOSS",
            retryable=False,
        )
    target_members = _target_members_by_member_id(target)
    group_source = _safe_attr(target, "source")
    source_members = _source_members_by_member_id(group_source)
    member_reports: list[RealizationTargetSetMemberReport] = []
    member_source_digests: list[str | None] = []
    for report in retained:
        target_member = target_members.get(report.member_id)
        source_member = source_members.get(report.member_id)
        source = _target_member_source(target_member, target)
        member_source_digest = (
            _optional_str(_safe_attr(source, "artifact_selection_digest"))
            or source_selection_digest
        )
        member_source_digests.append(member_source_digest)
        member_reports.append(
            RealizationTargetSetMemberReport(
                member_id=report.member_id,
                member_index=report.member_index,
                member_count=report.member_count,
                member_group_id=report.member_group_id,
                device_uuid=report.device_uuid,
                device=_optional_str(_safe_attr(target_member, "device")),
                binding_layout_id=report.binding_layout_id,
                binding_value_id=report.binding_value_id,
                target_layout_digest=_target_member_layout_digest(target_member),
                copy_plan_digest=_target_member_copy_plan_digest(target_member),
                source_selection_digest=member_source_digest,
                source_artifact_ref=_optional_str(
                    _safe_attr(source_member, "artifact_ref")
                )
                or _optional_str(_safe_attr(source, "source_artifact_ref")),
                runtime_profile_digest=_target_member_runtime_profile_digest(
                    target_member,
                    target,
                ),
                placement_digest=_target_member_placement_digest(
                    target_member,
                    target,
                ),
                readiness=report.readiness,
                verification_state=report.verification_state,
                reservation_bytes=report.reservation_bytes,
                reservation_capability_id=report.reservation_capability_id,
                reservation_scope_digest=report.reservation_scope_digest,
                staged_value=report.staged_value,
                group_realization_transaction_id=(
                    report.group_realization_transaction_id
                ),
                group_realization_version_set_id=(
                    report.group_realization_version_set_id
                ),
                group_realization_part_id=report.group_realization_part_id,
                group_realization_wait_for_publish=(
                    report.group_realization_wait_for_publish
                ),
                group_realization_wait_timeout_ms=(
                    report.group_realization_wait_timeout_ms
                ),
            )
        )

    failures = _safe_attr(result, "member_failures")
    failed_member_count = (
        len(failures)
        if isinstance(failures, Sequence) and not isinstance(failures, (str, bytes))
        else 0
    )
    target_member_count = len(target_members)
    declared_member_count = max(
        target_member_count,
        max(report.member_count for report in retained),
        len(retained) + failed_member_count,
    )
    topology = _safe_attr(target, "topology") or _safe_attr(result, "topology")
    readiness = _literal_value(_safe_attr(result, "readiness"))
    if readiness is None:
        readiness_values = {report.readiness for report in retained}
        readiness = next(iter(readiness_values)) if len(readiness_values) == 1 else None
    transaction_ids = tuple(
        sorted(
            {
                report.group_realization_transaction_id
                for report in retained
                if report.group_realization_transaction_id
            }
        )
    )
    version_set_ids = tuple(
        sorted(
            {
                report.group_realization_version_set_id
                for report in retained
                if report.group_realization_version_set_id
            }
        )
    )
    daemon_sessions = {
        (report.daemon_id, report.daemon_session_id) for report in retained
    }
    publish_barrier = _target_set_publish_barrier(
        retained,
        target=target,
        result=result,
    )
    return RealizationTargetSetReport(
        group_id=_optional_str(_safe_attr(result, "group_id"))
        or _optional_str(_safe_attr(target, "group_id"))
        or retained[0].member_group_id,
        runtime=_optional_str(_safe_attr(result, "runtime"))
        or _optional_str(_safe_attr(target, "runtime")),
        topology_digest=_optional_str(_safe_attr(topology, "schema_topology_digest"))
        or _optional_str(_safe_attr(topology, "admission_topology_digest")),
        source_kind=_literal_value(_safe_attr(group_source, "source_kind")),
        source_artifact_ref=_optional_str(
            _safe_attr(group_source, "source_artifact_ref")
        ),
        source_selection_mode=_source_selection_mode(
            member_source_digests,
            source=group_source,
        ),
        member_count=declared_member_count,
        successful_member_count=len(retained),
        failed_member_count=failed_member_count,
        partial=bool(_safe_attr(result, "partial"))
        or failed_member_count > 0
        or len(retained) < declared_member_count,
        readiness=readiness,
        ready_member_count=sum(
            1
            for report in retained
            if report.readiness in {"runtime_local_ready", "runtime_published_ready"}
        ),
        staged_member_count=sum(1 for report in retained if report.staged_value),
        total_reservation_bytes=sum(report.reservation_bytes for report in retained),
        same_daemon_session=len(daemon_sessions) == 1,
        group_realization_transaction_ids=transaction_ids,
        group_realization_version_set_ids=version_set_ids,
        acquire_claim_ids=tuple(
            report.reservation_capability_id for report in retained
        ),
        publish_barrier=publish_barrier,
        members=tuple(member_reports),
        failure_code_buckets=_failure_code_buckets(result),
    )


def _target_set_collective_policy(
    report: RealizationTargetSetReport,
    *,
    target: object | None,
    result: object | None,
) -> str:
    for owner in (result, target, _safe_attr(target, "source")):
        value = _literal_value(_safe_attr(owner, "collective_policy"))
        if value:
            return value
    if report.member_count > 1 and report.source_kind == "checkpoint_artifact":
        return "collective_first_candidate"
    return "disabled"


def _target_set_group_barriers(
    report: RealizationTargetSetReport,
) -> tuple[str, ...]:
    barriers: list[str] = []
    if report.member_count > 1:
        barriers.append("member_readiness")
    if (
        report.group_realization_transaction_ids
        or report.group_realization_version_set_ids
    ):
        barriers.append("group_acquire")
    if report.staged_member_count > 0:
        barriers.append("staged_values")
    if report.publish_barrier:
        barriers.append("publish_barrier")
    if report.partial:
        barriers.append("partial_failure_accounting")
    return tuple(barriers)


def target_set_strategy_plan_for(
    report: RealizationTargetSetReport,
    *,
    target: object | None = None,
    result: object | None = None,
) -> RealizationStrategyPlan:
    member_digests = tuple(
        digest
        for member in report.members
        if (digest := member.source_selection_digest)
    )
    return RealizationStrategyPlan(
        source_policy={
            "source_kind": report.source_kind,
            "source_artifact_ref": report.source_artifact_ref,
        },
        collective_policy=_target_set_collective_policy(
            report,
            target=target,
            result=result,
        ),
        source_selection_mode=report.source_selection_mode,
        source_coordination=(
            "same_daemon_session"
            if report.same_daemon_session
            else "cross_daemon_session"
        ),
        same_daemon_session=report.same_daemon_session,
        member_count=report.member_count,
        group_barriers=_target_set_group_barriers(report),
        publish_barrier=report.publish_barrier,
        group_realization_transaction_ids=report.group_realization_transaction_ids,
        group_realization_version_set_ids=report.group_realization_version_set_ids,
        member_source_selection_digests=member_digests,
    )


def _member_release_policy(
    member: RealizationTargetSetMemberReport | RealizationRetainedBindingReport,
    envelope: RealizationResourceEnvelope,
) -> tuple[str, ...]:
    policies = ["release_binding_reservation"]
    if (
        member.staged_value
        and "release_group_staged_acquire" in envelope.release_policy
    ):
        policies.append("release_group_staged_acquire")
    return tuple(policies)


def target_set_lifecycle_plan_for(
    report: RealizationTargetSetReport,
    *,
    envelope: RealizationResourceEnvelope,
) -> RealizationLifecyclePlan:
    return RealizationLifecyclePlan(
        capability="target_set",
        retained=True,
        export_lifetime_kind=envelope.export_lifetime_kind,
        release_strictness=envelope.release_strictness,
        mutability_contract=envelope.mutability_contract,
        release_policy=envelope.release_policy,
        staged_value_count=report.staged_member_count,
        acquire_claim_count=len(report.acquire_claim_ids),
        acquire_claim_ids=report.acquire_claim_ids,
        publish_barrier=report.publish_barrier,
        group_realization_transaction_ids=report.group_realization_transaction_ids,
        group_realization_version_set_ids=report.group_realization_version_set_ids,
        member_release_policies={
            member.member_id: _member_release_policy(member, envelope)
            for member in report.members
        },
    )


def target_set_representation_admission_for(
    report: RealizationTargetSetReport,
) -> RepresentationAdmissionPlan:
    return RepresentationAdmissionPlan(
        representation_contract="target_set_binding_reservation",
        transform_required=report.source_kind == "checkpoint_artifact",
        transform_plan_digest=(
            _object_digest(
                "target-set-representation",
                {
                    "topology_digest": report.topology_digest,
                    "source_selection_mode": report.source_selection_mode,
                    "member_layouts": [
                        member.target_layout_digest for member in report.members
                    ],
                },
            )
            if report.members
            else None
        ),
    )


def representation_admission_for_target(
    target_plan: RealizationTargetPlan,
) -> RepresentationAdmissionPlan:
    transform_plan_digest = target_plan.copy_plan_digest or target_plan.mapped_view_id
    representation_contract = (
        "runtime_attachment"
        if target_plan.kind == "runtime_attachment"
        else "binding_layout"
        if target_plan.kind in {"binding_owned", "binding_adopted"}
        else target_plan.kind
    )
    return RepresentationAdmissionPlan(
        representation_contract=representation_contract,
        transform_required=bool(transform_plan_digest),
        transform_plan_digest=transform_plan_digest,
    )


def lifecycle_plan_for_envelope(
    target_plan: RealizationTargetPlan,
    envelope: RealizationResourceEnvelope,
    *,
    publishability: RealizationPublishabilityReport | None = None,
) -> RealizationLifecyclePlan:
    return RealizationLifecyclePlan(
        capability=target_plan.kind,
        publishable=bool(publishability.publishable) if publishability else False,
        retained=(
            envelope.retained_bytes > 0
            or envelope.export_lifetime_kind
            in {"daemon_retained", "runtime_attachment", "publication_lease"}
        ),
        export_lifetime_kind=envelope.export_lifetime_kind,
        release_strictness=envelope.release_strictness,
        mutability_contract=envelope.mutability_contract,
        release_policy=envelope.release_policy,
    )


def retained_binding_lifecycle_plan_for(
    retained_bindings: Sequence[RealizationRetainedBindingReport],
    *,
    envelope: RealizationResourceEnvelope,
) -> RealizationLifecyclePlan:
    retained = tuple(retained_bindings)
    return RealizationLifecyclePlan(
        capability="retained_binding",
        retained=True,
        export_lifetime_kind=envelope.export_lifetime_kind,
        release_strictness=envelope.release_strictness,
        mutability_contract=envelope.mutability_contract,
        release_policy=envelope.release_policy,
        staged_value_count=sum(1 for report in retained if report.staged_value),
        acquire_claim_count=len(retained),
        acquire_claim_ids=tuple(
            report.reservation_capability_id for report in retained
        ),
        publish_barrier=any(
            report.group_realization_wait_for_publish for report in retained
        ),
        group_realization_transaction_ids=tuple(
            sorted(
                {
                    report.group_realization_transaction_id
                    for report in retained
                    if report.group_realization_transaction_id
                }
            )
        ),
        group_realization_version_set_ids=tuple(
            sorted(
                {
                    report.group_realization_version_set_id
                    for report in retained
                    if report.group_realization_version_set_id
                }
            )
        ),
        member_release_policies={
            report.member_id: _member_release_policy(report, envelope)
            for report in retained
        },
    )


def binding_report_for(
    binding: object,
    *,
    publish_requested: bool = False,
) -> RealizationBindingReport:
    current_value = _safe_attr(binding, "current_value")
    staged_value = (
        None if current_value is not None else _safe_attr(binding, "staged_value")
    )
    binding_value = current_value if current_value is not None else staged_value
    value_state = (
        "current"
        if current_value is not None
        else "staged"
        if staged_value is not None
        else "none"
    )
    binding_id = _optional_str(_safe_attr(binding, "binding_id")) or _optional_str(
        _safe_attr(binding_value, "binding_id")
    )
    binding_layout_id = _optional_str(
        _safe_attr(binding, "binding_layout_id")
    ) or _optional_str(_safe_attr(binding_value, "binding_layout_id"))
    if not binding_id or not binding_layout_id:
        raise ArtifactError(
            "binding realization report requires binding_id and binding_layout_id",
            status_code="DATA_LOSS",
            retryable=False,
        )
    is_artifact_backed = bool(_safe_attr(binding_value, "is_artifact_backed"))
    publish_replica = _safe_attr(binding, "publish_replica")
    published = bool(_safe_attr(current_value, "is_published"))
    publication_eligible = (
        current_value is not None and is_artifact_backed and callable(publish_replica)
    )
    return RealizationBindingReport(
        binding_id=binding_id,
        binding_layout_id=binding_layout_id,
        binding_value_id=_optional_str(_safe_attr(binding_value, "binding_value_id")),
        seal_generation=_optional_int(_safe_attr(binding_value, "seal_generation")),
        value_state=value_state,
        source_artifact_id=_optional_str(
            _safe_attr(binding_value, "source_artifact_id")
        ),
        is_artifact_backed=is_artifact_backed,
        verification_state=_optional_int(
            _safe_attr(binding_value, "verification_state")
        ),
        verification_job_id=_optional_str(
            _safe_attr(binding_value, "verification_job_id")
        ),
        source_artifact_ref=_optional_str(
            _safe_attr(binding_value, "source_artifact_ref")
        ),
        local_serving_ref=_optional_str(_safe_attr(binding_value, "local_serving_ref")),
        serving_artifact_id=_optional_str(
            _safe_attr(binding_value, "serving_artifact_id")
        ),
        verification_failure_reason=_optional_str(
            _safe_attr(binding_value, "verification_failure_reason")
        ),
        published=published,
        publication_eligible=publication_eligible,
        publish_requested=bool(publish_requested),
        acquired=bool(_safe_attr(binding_value, "acquired")),
    )


def publication_report_for(projection: object) -> RealizationPublicationReport:
    binding_value_ref = _safe_attr(projection, "binding_value_ref")
    return RealizationPublicationReport(
        state=str(_safe_attr(projection, "state") or ""),
        artifact_ref=_optional_str(_safe_attr(projection, "artifact_ref")),
        operation_id=_optional_str(_safe_attr(projection, "operation_id")),
        replica_id=_optional_str(_safe_attr(projection, "replica_id")),
        lease_id=_optional_str(_safe_attr(projection, "lease_id")),
        device_uuid=_optional_str(_safe_attr(projection, "device_uuid")),
        owner_pid=_optional_int(_safe_attr(projection, "owner_pid")),
        binding_layout_id=_optional_str(_safe_attr(projection, "binding_layout_id")),
        generation=_optional_str(_safe_attr(projection, "generation")),
        reason=_optional_str(_safe_attr(projection, "reason")),
        byte_space_kind=_optional_str(_safe_attr(projection, "byte_space_kind")),
        byte_space_id=_optional_str(_safe_attr(projection, "byte_space_id")),
        binding_value_id=_optional_str(
            _safe_attr(binding_value_ref, "binding_value_id")
        ),
    )


def mounted_source_report_for(
    *,
    source_artifact_id: str,
    promoted_artifact_id: str,
    verify_checksums: bool,
    generation: int | None,
    canonical_index_bytes: bytes,
) -> RealizationMountedSourceReport:
    return RealizationMountedSourceReport(
        source_artifact_id=str(source_artifact_id),
        promoted_artifact_id=str(promoted_artifact_id),
        verify_checksums=bool(verify_checksums),
        generation=generation,
        canonical_index_bytes_len=len(canonical_index_bytes),
        promoted_artifact_profile=_artifact_profile_for(promoted_artifact_id),
        promoted_authority_scope=_authority_scope_for(promoted_artifact_id),
    )


def envelope_for_binding(
    binding: object,
    *,
    target_kind: BindingRealizationTargetKind,
    target_tensors: Mapping[str, object] | None = None,
    publish_requested: bool = False,
) -> RealizationResourceEnvelope:
    materialization_diagnostics = _mapping_copy(
        _safe_attr(binding, "last_materialization_diagnostics")
    )
    execution_diagnostics = _safe_attr(binding, "last_execution_diagnostics")
    total_bytes = _mapping_int(materialization_diagnostics, "total_bytes")
    if total_bytes == 0:
        tensors = target_tensors
        if tensors is None:
            candidate = _safe_attr(binding, "tensors")
            tensors = candidate if isinstance(candidate, Mapping) else None
        total_bytes = _total_tensor_bytes(tensors)

    direct_write_bytes = _execution_int(
        execution_diagnostics, "actual_local_typed_bytes"
    ) + _execution_int(execution_diagnostics, "actual_collective_committed_bytes")
    copy_components = (
        _execution_int(execution_diagnostics, "fallback_bytes"),
        _execution_int(execution_diagnostics, "residual_bytes"),
        _execution_int(execution_diagnostics, "actual_generic_backend_bytes"),
    )
    copy_bytes = sum(copy_components)
    copy_count = sum(1 for component in copy_components if component > 0)
    if direct_write_bytes == 0 and target_kind == "binding_adopted":
        direct_write_bytes = total_bytes

    source_bound_plan = _safe_attr(binding, "last_source_bound_plan_diagnostics")
    fallback_buckets: dict[str, int] = {}
    retry_buckets = (
        materialization_diagnostics.get("retry_reason_buckets")
        if materialization_diagnostics is not None
        else None
    )
    if isinstance(retry_buckets, Mapping):
        fallback_buckets.update(
            {
                str(reason): int(count)
                for reason, count in retry_buckets.items()
                if _optional_int(count) is not None
            }
        )
    planner_buckets = _safe_attr(source_bound_plan, "planner_reject_reason_buckets")
    if isinstance(planner_buckets, Mapping):
        for reason, count in planner_buckets.items():
            reason_key = str(reason)
            fallback_buckets[reason_key] = fallback_buckets.get(reason_key, 0) + (
                _optional_int(count) or 0
            )

    release_policy = (
        ("close_binding", "retire_binding_value", "release_publish_lease")
        if publish_requested
        else ("close_binding", "retire_binding_value")
    )
    temporary_replica_bytes = _execution_int(
        execution_diagnostics,
        "collective_peak_temporary_bytes",
    )
    if target_kind == "binding_owned":
        return RealizationResourceEnvelope(
            backing_kind="daemon_binding_value",
            export_kind="binding_restore",
            projection_kind="binding",
            owner_kind="binding_slot",
            release_policy=release_policy,
            mutability_contract="binding_controlled",
            release_strictness="strict",
            export_lifetime_kind="token_backed",
            direct_write_bytes=direct_write_bytes,
            copy_bytes=copy_bytes,
            copy_count=copy_count,
            temporary_replica_bytes=temporary_replica_bytes,
            retained_bytes=total_bytes,
            fallback_reason_buckets=fallback_buckets,
        )
    return RealizationResourceEnvelope(
        backing_kind="caller_region_binding_value",
        export_kind="registered_region_direct_write",
        projection_kind="binding",
        owner_kind="binding_slot_over_caller_memory",
        release_policy=("close_binding", "unregister_target_region"),
        mutability_contract="caller_mutable_binding_controlled",
        release_strictness="strict",
        export_lifetime_kind="token_backed",
        direct_write_bytes=direct_write_bytes,
        copy_bytes=copy_bytes,
        copy_count=copy_count,
        temporary_replica_bytes=temporary_replica_bytes,
        fallback_reason_buckets=fallback_buckets,
    )


def report_for_binding_realization(
    *,
    target_kind: BindingRealizationTargetKind,
    selection: ResolvedArtifactSelection,
    target_plan: RealizationTargetPlan,
    binding: object,
    envelope: RealizationResourceEnvelope,
    publish_requested: bool = False,
    risk_labels: tuple[str, ...] = (),
    options: object | None = None,
    ctx: object | None = None,
) -> ArtifactRealizationReport:
    materialization_diagnostics = _mapping_copy(
        _safe_attr(binding, "last_materialization_diagnostics")
    )
    execution_diagnostics = _safe_attr(binding, "last_execution_diagnostics")
    source_bound_plan_diagnostics = _safe_attr(
        binding,
        "last_source_bound_plan_diagnostics",
    )
    source = None
    operation_id = None
    if materialization_diagnostics is not None:
        source = _optional_str(materialization_diagnostics.get("source"))
        operation_id = _optional_str(materialization_diagnostics.get("operation_id"))
    binding_report = binding_report_for(
        binding,
        publish_requested=publish_requested,
    )
    execution_commit = execution_commit_report_for(
        execution_diagnostics=execution_diagnostics,
        source_bound_plan_diagnostics=source_bound_plan_diagnostics,
        source=source,
        requested_bytes=_mapping_int(materialization_diagnostics, "total_bytes"),
    )
    publishability = publishability_report_for(
        binding=binding_report,
        publish_requested=publish_requested,
    )
    return ArtifactRealizationReport(
        target_kind=target_kind,
        source_selection_digest=selection.source_selection_digest,
        target_layout_digest=target_plan.target_layout_digest,
        copy_plan_digest=target_plan.copy_plan_digest,
        artifact_id=selection.artifact_id,
        view_id=selection.view_id,
        artifact_profile=selection.artifact_profile,
        authority_scope=selection.authority_scope,
        generation_hint=selection.generation_hint,
        **selection_report_fields(selection),
        envelope=envelope,
        target_plan=target_plan,
        strategy_plan=strategy_plan_for_execution(
            envelope=envelope,
            execution_commit=execution_commit,
            options=options,
            ctx=ctx,
        ),
        representation_admission=representation_admission_for_target(target_plan),
        lifecycle_plan=lifecycle_plan_for_envelope(
            target_plan,
            envelope,
            publishability=publishability,
        ),
        source=source,
        operation_id=operation_id,
        operation_backend="daemon_binding",
        risk_labels=risk_labels_for_target(
            target_plan,
            envelope,
            source_selection_digest=selection.source_selection_digest,
            publishable=publish_requested,
            extra=risk_labels,
        ),
        materialization_diagnostics=materialization_diagnostics,
        binding=binding_report,
        publishability=publishability,
        execution_commit=execution_commit,
        execution_diagnostics=execution_diagnostics,
        source_bound_plan_diagnostics=source_bound_plan_diagnostics,
    )


def report_for_runtime_attachment(
    *,
    selection: ResolvedArtifactSelection,
    target_plan: RealizationTargetPlan,
    envelope: RealizationResourceEnvelope,
    binding_handle: object | None = None,
    retained_authority: object | None = None,
    materialization_diagnostics: object | None = None,
    execution_diagnostics: object | None = None,
    source_bound_plan_diagnostics: object | None = None,
    source: str | None = None,
    operation_id: str | None = None,
    risk_labels: tuple[str, ...] = (),
    options: object | None = None,
    ctx: object | None = None,
) -> ArtifactRealizationReport:
    binding_report = None
    if binding_handle is not None:
        with contextlib.suppress(ArtifactError):
            binding_report = binding_report_for(binding_handle)
    retained_reports: tuple[RealizationRetainedBindingReport, ...] = ()
    if retained_authority is not None:
        retained_reports = retained_binding_reports_for(retained_authority)
    materialization_mapping = _mapping_copy(materialization_diagnostics)
    if materialization_mapping is not None:
        source = source or _optional_str(materialization_mapping.get("source"))
        operation_id = operation_id or _optional_str(
            materialization_mapping.get("operation_id")
        )
    execution_commit = execution_commit_report_for(
        execution_diagnostics=execution_diagnostics,
        source_bound_plan_diagnostics=source_bound_plan_diagnostics,
        source=source,
        requested_bytes=_mapping_int(materialization_mapping, "total_bytes"),
    )
    publishability = publishability_report_for(binding=binding_report)
    return ArtifactRealizationReport(
        target_kind="runtime_attachment",
        source_selection_digest=selection.source_selection_digest,
        target_layout_digest=target_plan.target_layout_digest,
        copy_plan_digest=target_plan.copy_plan_digest,
        artifact_id=selection.artifact_id,
        view_id=selection.view_id,
        artifact_profile=selection.artifact_profile,
        authority_scope=selection.authority_scope,
        generation_hint=selection.generation_hint,
        **selection_report_fields(selection),
        envelope=envelope,
        target_plan=target_plan,
        strategy_plan=strategy_plan_for_execution(
            envelope=envelope,
            execution_commit=execution_commit,
            options=options,
            ctx=ctx,
        ),
        representation_admission=representation_admission_for_target(target_plan),
        lifecycle_plan=lifecycle_plan_for_envelope(
            target_plan,
            envelope,
            publishability=publishability,
        ),
        source=source,
        operation_id=operation_id,
        operation_backend="runtime_attachment",
        risk_labels=risk_labels_for_target(
            target_plan,
            envelope,
            source_selection_digest=selection.source_selection_digest,
            extra=risk_labels,
        ),
        materialization_diagnostics=materialization_diagnostics,
        binding=binding_report,
        retained_bindings=retained_reports,
        publishability=publishability,
        execution_commit=execution_commit,
        execution_diagnostics=execution_diagnostics,
        source_bound_plan_diagnostics=source_bound_plan_diagnostics,
    )


def model_runtime_report_for(
    *,
    spec: "ArtifactRealizationSpec",
    runtime_attachment_report: ArtifactRealizationReport,
) -> ArtifactRealizationReport:
    envelope = replace(
        runtime_attachment_report.envelope,
        projection_kind="model_runtime",
    )
    target_plan = RealizationTargetPlan(
        kind="model_runtime",
        device=spec.device,
        target_layout_digest=runtime_attachment_report.target_layout_digest,
        binding_layout_id=runtime_attachment_report.binding.binding_layout_id
        if runtime_attachment_report.binding is not None
        else None,
        copy_plan_digest=runtime_attachment_report.copy_plan_digest,
    )
    model_runtime_report = RealizationModelRuntimeReport(
        framework=str(spec.framework or ""),
        device=_optional_str(spec.device),
        adapter_version=_optional_str(spec.adapter_version),
        runtime_abi_version=_optional_str(spec.runtime_abi_version),
        topology_digest=_object_digest("model-runtime-topology", spec.topology),
        member_digest=_object_digest("model-runtime-member", spec.member),
        runtime_attachment_target_kind=runtime_attachment_report.target_kind,
    )
    return replace(
        runtime_attachment_report,
        target_kind="model_runtime",
        envelope=envelope,
        operation_backend="model_runtime_attachment",
        representation_admission=representation_admission_for_target(target_plan),
        lifecycle_plan=lifecycle_plan_for_envelope(
            target_plan,
            envelope,
            publishability=publishability_report_for(),
        ),
        risk_labels=risk_labels_for_target(
            target_plan,
            envelope,
            source_selection_digest=runtime_attachment_report.source_selection_digest,
            extra=(
                *runtime_attachment_report.risk_labels,
                "model_runtime",
                f"framework:{model_runtime_report.framework}",
            ),
        ),
        target_plan=target_plan,
        model_runtime=model_runtime_report,
        publishability=publishability_report_for(),
    )


def report_for_publication(
    *,
    artifact_id: str,
    source_selection_digest: str,
    target_plan: RealizationTargetPlan,
    envelope: RealizationResourceEnvelope,
    projection: object,
    binding_handle: object | None = None,
    risk_labels: tuple[str, ...] = (),
) -> ArtifactRealizationReport:
    binding_report = None
    if binding_handle is not None:
        with contextlib.suppress(ArtifactError):
            binding_report = binding_report_for(binding_handle, publish_requested=True)
    publication_report = publication_report_for(projection)
    publishability = publishability_report_for(
        binding=binding_report,
        publication=publication_report,
        publish_requested=True,
    )
    return ArtifactRealizationReport(
        target_kind="publication",
        source_selection_digest=source_selection_digest,
        target_layout_digest=target_plan.target_layout_digest,
        copy_plan_digest=target_plan.copy_plan_digest,
        artifact_id=str(artifact_id),
        view_id="",
        artifact_profile="runtime_artifact",
        authority_scope="daemon_publication",
        generation_hint=None,
        envelope=envelope,
        target_plan=target_plan,
        representation_admission=representation_admission_for_target(target_plan),
        lifecycle_plan=lifecycle_plan_for_envelope(
            target_plan,
            envelope,
            publishability=publishability,
        ),
        operation_id=_optional_str(_safe_attr(projection, "operation_id")),
        operation_backend="runtime_publication",
        risk_labels=risk_labels_for_target(
            target_plan,
            envelope,
            source_selection_digest=source_selection_digest,
            publishable=True,
            extra=risk_labels,
        ),
        binding=binding_report,
        publication=publication_report,
        publishability=publishability,
    )


def report_for_mounted_source(
    *,
    selection: ResolvedArtifactSelection,
    promoted_artifact_id: str,
    generation: int | None,
    canonical_index_bytes: bytes,
    verify_checksums: bool,
    target_plan: RealizationTargetPlan,
    envelope: RealizationResourceEnvelope,
    risk_labels: tuple[str, ...] = (),
) -> ArtifactRealizationReport:
    mounted_source_report = mounted_source_report_for(
        source_artifact_id=selection.artifact_id,
        promoted_artifact_id=promoted_artifact_id,
        verify_checksums=verify_checksums,
        generation=generation,
        canonical_index_bytes=canonical_index_bytes,
    )
    return ArtifactRealizationReport(
        target_kind="mounted_source",
        source_selection_digest=selection.source_selection_digest,
        target_layout_digest=target_plan.target_layout_digest,
        copy_plan_digest=target_plan.copy_plan_digest,
        artifact_id=selection.artifact_id,
        view_id=selection.view_id,
        artifact_profile=selection.artifact_profile,
        authority_scope=selection.authority_scope,
        generation_hint=selection.generation_hint,
        **selection_report_fields(selection),
        envelope=envelope,
        target_plan=target_plan,
        representation_admission=representation_admission_for_target(target_plan),
        lifecycle_plan=lifecycle_plan_for_envelope(target_plan, envelope),
        operation_backend="daemon_mounted_source_promotion",
        risk_labels=risk_labels_for_target(
            target_plan,
            envelope,
            source_selection_digest=selection.source_selection_digest,
            extra=("mounted_source", *risk_labels),
        ),
        mounted_source=mounted_source_report,
        publishability=publishability_report_for(),
    )


def report_for_target_set(
    *,
    selection: ResolvedArtifactSelection,
    target_plan: RealizationTargetPlan,
    target: object | None,
    result: object,
    envelope: RealizationResourceEnvelope,
    operation_id: str | None = None,
    risk_labels: tuple[str, ...] = (),
) -> ArtifactRealizationReport:
    retained_bindings = retained_binding_reports_for(result)
    target_set_report = target_set_report_for_retained_bindings(
        retained_bindings,
        result=result,
        target=target,
        source_selection_digest=selection.source_selection_digest,
    )
    return ArtifactRealizationReport(
        target_kind="target_set",
        source_selection_digest=selection.source_selection_digest,
        target_layout_digest=target_plan.target_layout_digest,
        copy_plan_digest=target_plan.copy_plan_digest,
        artifact_id=selection.artifact_id,
        view_id=selection.view_id,
        artifact_profile=selection.artifact_profile,
        authority_scope=selection.authority_scope,
        generation_hint=selection.generation_hint,
        **selection_report_fields(selection),
        envelope=envelope,
        target_plan=target_plan,
        strategy_plan=target_set_strategy_plan_for(
            target_set_report,
            target=target,
            result=result,
        ),
        representation_admission=target_set_representation_admission_for(
            target_set_report
        ),
        lifecycle_plan=target_set_lifecycle_plan_for(
            target_set_report,
            envelope=envelope,
        ),
        operation_id=operation_id,
        operation_backend="daemon_prefetch_serving_binding_set",
        risk_labels=risk_labels_for_target(
            target_plan,
            envelope,
            source_selection_digest=selection.source_selection_digest,
            extra=risk_labels,
        ),
        retained_bindings=retained_bindings,
        publishability=publishability_report_for(),
        target_set=target_set_report,
    )


def artifact_realization_report_to_dict(
    report: ArtifactRealizationReport,
) -> dict[str, object]:
    return asdict(report)


@dataclass(slots=True)
class RealizationReleaseContract:
    release_policy: tuple[str, ...]
    release_strictness: str
    actions: tuple[Callable[[], None], ...] = ()
    _released: bool = field(default=False, init=False, repr=False)

    @property
    def released(self) -> bool:
        return self._released

    def release(self) -> None:
        if self._released:
            return
        try:
            for action in self.actions:
                action()
        except Exception as exc:
            if self.release_strictness == "strict":
                raise ArtifactError(
                    "artifact realization release contract failed",
                    status_code="INTERNAL",
                    retryable=False,
                ) from exc
            return
        self._released = True


def release_contract_for(
    envelope: RealizationResourceEnvelope,
    *actions: Callable[[], None] | None,
) -> RealizationReleaseContract:
    return RealizationReleaseContract(
        release_policy=tuple(envelope.release_policy),
        release_strictness=str(envelope.release_strictness),
        actions=tuple(action for action in actions if action is not None),
    )


class TensorDictProjection(dict[str, Any]):
    def __init__(
        self,
        tensors: Mapping[str, Any],
        *,
        owner: "ArtifactRealizationHandle",
    ) -> None:
        super().__init__(tensors)
        self._tensorcast_realization_owner = owner
        for tensor in self.values():
            try:
                tensor._tensorcast_realization_owner = owner
            except (AttributeError, TypeError) as exc:
                raise ArtifactError(
                    "failed to attach artifact realization owner to tensor projection",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                ) from exc

    @property
    def report(self) -> ArtifactRealizationReport:
        return self._tensorcast_realization_owner.report

    def close(self) -> None:
        self._tensorcast_realization_owner.close()

    def __setitem__(self, key: str, value: Any) -> None:
        self._reject_mutation()

    def __delitem__(self, key: str) -> None:
        self._reject_mutation()

    def clear(self) -> None:
        self._reject_mutation()

    def pop(self, key: str, default: Any = None) -> Any:
        self._reject_mutation()

    def popitem(self) -> tuple[str, Any]:
        self._reject_mutation()

    def setdefault(self, key: str, default: Any = None) -> Any:
        self._reject_mutation()

    def update(self, *args: Any, **kwargs: Any) -> None:
        self._reject_mutation()

    def __ior__(  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: object, /
    ) -> "TensorDictProjection":
        self._reject_mutation()

    def _reject_mutation(self) -> NoReturn:
        raise ArtifactError(
            "TensorDict projection is read-only; materialize into caller tensors for mutable writes",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )


class ArtifactRealizationHandle:
    def __init__(
        self,
        *,
        target_kind: RealizationTargetKind,
        report: ArtifactRealizationReport,
        tensor_dict_value: Mapping[str, Any] | None = None,
        binding_value: Any | None = None,
        prefetch_value: Any | None = None,
        attachment_value: Any | None = None,
        promote_fn: Callable[..., Any] | None = None,
        attach_fn: Callable[..., Any] | None = None,
        release_contract: RealizationReleaseContract | None = None,
        close_fn: Callable[[], None] | None = None,
    ) -> None:
        report.validate_for_handle(target_kind)
        self._target_kind = target_kind
        self._report = report
        self._tensor_dict_value = tensor_dict_value
        self._tensor_dict_projection: TensorDictProjection | None = None
        self._binding_value = binding_value
        self._prefetch_value = prefetch_value
        self._attachment_value = attachment_value
        self._promote_fn = promote_fn
        self._attach_fn = attach_fn
        self._release_contract = release_contract or release_contract_for(
            report.envelope,
            close_fn,
        )
        self._closed = False
        self.created_at = time.time()

    @property
    def report(self) -> ArtifactRealizationReport:
        return self._report

    @property
    def release_contract(self) -> RealizationReleaseContract:
        return self._release_contract

    def tensor_dict(self) -> TensorDictProjection:
        tensor_dict_value = self._tensor_dict_value
        if tensor_dict_value is None:
            self._unsupported("tensor_dict")
        if self._tensor_dict_projection is None:
            self._tensor_dict_projection = TensorDictProjection(
                tensor_dict_value,
                owner=self,
            )
        return self._tensor_dict_projection

    def binding(self) -> Any:
        if self._binding_value is None:
            self._unsupported("binding")
        return self._binding_value

    def prefetch_handoff(self) -> Any:
        if self._prefetch_value is None:
            self._unsupported("prefetch_handoff")
        return self._prefetch_value

    def complete(self) -> None:
        if self._target_kind != "caller_tensors":
            self._unsupported("complete")
        return None

    def attach(self, *args: object, **kwargs: object) -> Any:
        if self._attach_fn is None:
            if args or kwargs or self._attachment_value is None:
                self._unsupported("attach")
            return self._attachment_value
        if self._attachment_value is not None and not args and not kwargs:
            return self._attachment_value
        return self._attach_fn(*args, **kwargs)

    def attachment(self) -> Any:
        if self._attachment_value is not None:
            return self._attachment_value
        if self._attach_fn is None:
            self._unsupported("attach")
        return self._attach_fn()

    def publish_replica(self, *args: object, **kwargs: object) -> Any:
        binding_value = self._binding_value
        publish = getattr(binding_value, "publish_replica", None)
        if not callable(publish):
            self._unsupported("publish_replica")
        return publish(*args, **kwargs)

    def promote(self, *args: object, **kwargs: object) -> Any:
        if self._promote_fn is not None:
            return self._promote_fn(*args, **kwargs)
        binding_value = self._binding_value
        promote_current = getattr(binding_value, "promote_current_value", None)
        if not callable(promote_current):
            self._unsupported("promote")
        return promote_current(*args, **kwargs)

    def close(self) -> None:
        if self._closed:
            return
        self._release_contract.release()
        self._closed = True

    def _unsupported(self, action: str) -> NoReturn:
        raise ArtifactError(
            f"{action} is not supported for {self._target_kind} realization",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )


@dataclass(frozen=True, slots=True)
class ArtifactRealizationSpec:
    target_kind: RealizationTargetKind
    device: object | None = None
    options: object | None = None
    target: object | None = None
    mapping: object | None = None
    packing: str = "byte_space"
    capacity_bytes: int | None = None
    publish: bool = False
    runtime_artifact_policy: object | None = None
    readiness: object | None = None
    retention: object | None = None
    verify_checksums: bool = True
    timeout_s: float | None = None
    framework: str | None = None
    topology: object | None = None
    member: object | None = None
    adapter_version: str | None = None
    runtime_abi_version: str | None = None

    @classmethod
    def tensor_dict(
        cls,
        *,
        device: object,
        options: object | None = None,
    ) -> "ArtifactRealizationSpec":
        return cls(target_kind="tensor_dict", device=device, options=options)

    @classmethod
    def caller_tensors(
        cls,
        *,
        target: object,
        device: object | None = None,
        options: object | None = None,
    ) -> "ArtifactRealizationSpec":
        return cls(
            target_kind="caller_tensors",
            device=device,
            options=options,
            target=target,
        )

    @classmethod
    def binding(
        cls,
        *,
        device: object,
        mapping: object | None = None,
        packing: str = "byte_space",
        options: object | None = None,
        capacity_bytes: int | None = None,
        publish: bool = False,
        runtime_artifact_policy: object | None = None,
    ) -> "ArtifactRealizationSpec":
        return cls(
            target_kind="binding_owned",
            device=device,
            mapping=mapping,
            packing=packing,
            options=options,
            capacity_bytes=capacity_bytes,
            publish=publish,
            runtime_artifact_policy=runtime_artifact_policy,
        )

    @classmethod
    def adopted_binding(
        cls,
        *,
        target: object,
        mapping: object | None = None,
        packing: str = "byte_space",
        options: object | None = None,
        publish: bool = False,
        runtime_artifact_policy: object | None = None,
    ) -> "ArtifactRealizationSpec":
        return cls(
            target_kind="binding_adopted",
            target=target,
            mapping=mapping,
            packing=packing,
            options=options,
            publish=publish,
            runtime_artifact_policy=runtime_artifact_policy,
        )

    @classmethod
    def retained_replica(
        cls,
        *,
        device: object,
        options: object | None = None,
        retention: object | None = None,
    ) -> "ArtifactRealizationSpec":
        return cls(
            target_kind="retained_replica",
            device=device,
            options=options,
            retention=retention,
        )

    @classmethod
    def retained_binding(
        cls,
        *,
        target: object,
        readiness: object | None = None,
        retention: object | None = None,
        options: object | None = None,
    ) -> "ArtifactRealizationSpec":
        return cls(
            target_kind="retained_binding",
            target=target,
            readiness=readiness,
            retention=retention,
            options=options,
        )

    @classmethod
    def target_set(
        cls,
        *,
        target: object,
        readiness: object | None = None,
        retention: object | None = None,
        options: object | None = None,
    ) -> "ArtifactRealizationSpec":
        return cls(
            target_kind="target_set",
            target=target,
            readiness=readiness,
            retention=retention,
            options=options,
        )

    @classmethod
    def mounted_source(
        cls,
        *,
        verify_checksums: bool = True,
        timeout_s: float | None = None,
    ) -> "ArtifactRealizationSpec":
        return cls(
            target_kind="mounted_source",
            verify_checksums=bool(verify_checksums),
            timeout_s=timeout_s,
        )

    @classmethod
    def model_runtime(
        cls,
        *,
        framework: str,
        device: object | None = None,
        topology: object | None = None,
        member: object | None = None,
        adapter_version: str | None = None,
        runtime_abi_version: str | None = None,
        options: object | None = None,
        runtime_artifact_policy: object | None = None,
    ) -> "ArtifactRealizationSpec":
        if not str(framework or "").strip():
            raise ArtifactError(
                "model_runtime realization requires framework",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        return cls(
            target_kind="model_runtime",
            device=device,
            options=options,
            framework=str(framework),
            topology=topology,
            member=member,
            adapter_version=adapter_version,
            runtime_abi_version=runtime_abi_version,
            runtime_artifact_policy=runtime_artifact_policy,
        )

    @classmethod
    def _publication(
        cls,
        *,
        target: object,
        options: object | None = None,
        timeout_s: float | None = None,
    ) -> "ArtifactRealizationSpec":
        return cls(
            target_kind="publication",
            target=target,
            options=options,
            publish=True,
            timeout_s=timeout_s,
        )


__all__ = [
    "ArtifactRealizationHandle",
    "ArtifactRealizationReport",
    "ArtifactRealizationSpec",
    "RealizationBindingReport",
    "RealizationExecutionCommitReport",
    "RealizationLifecyclePlan",
    "RealizationModelRuntimeReport",
    "RealizationMountedSourceReport",
    "RealizationPublicationReport",
    "RealizationPublishabilityReport",
    "RealizationReleaseContract",
    "RealizationRetainedBindingReport",
    "RealizationResourceEnvelope",
    "RealizationStrategyPlan",
    "RealizationTargetSetMemberReport",
    "RealizationTargetSetReport",
    "RealizationTargetPlan",
    "RepresentationAdmissionPlan",
    "ResolvedArtifactSelection",
    "TensorDictProjection",
    "artifact_realization_profile_payload",
    "artifact_realization_report_to_dict",
    "binding_materialization_diagnostics_from_response",
    "binding_report_for",
    "emit_artifact_realization_profile_event",
    "envelope_for_binding",
    "envelope_for_caller_tensors",
    "envelope_for_mounted_source",
    "envelope_for_target_region_registration",
    "envelope_for_retained_binding",
    "envelope_for_retained_replica",
    "envelope_for_runtime_attachment",
    "envelope_for_publication",
    "envelope_for_target_set",
    "envelope_for_tensor_dict",
    "execution_commit_report_for",
    "execution_diagnostics_from_response",
    "lifecycle_plan_for_envelope",
    "materialization_source_label",
    "mounted_source_report_for",
    "mounted_source_target_digest",
    "model_runtime_report_for",
    "publishability_report_for",
    "publication_report_for",
    "release_contract_for",
    "report_for_binding_realization",
    "report_for_mounted_source",
    "report_for_publication",
    "report_for_runtime_attachment",
    "report_for_target_set",
    "representation_admission_for_target",
    "retained_binding_lifecycle_plan_for",
    "risk_labels_for_target",
    "retained_binding_reports_for",
    "resolve_artifact_selection",
    "source_bound_plan_diagnostics_from_response",
    "strategy_plan_for_execution",
    "target_set_lifecycle_plan_for",
    "target_set_representation_admission_for",
    "target_set_report_for_retained_bindings",
    "target_set_strategy_plan_for",
]
