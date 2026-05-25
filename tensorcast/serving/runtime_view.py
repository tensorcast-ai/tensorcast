#  Copyright (c) 2026, TensorCast Team.
"""Framework-neutral runtime endpoint view models and aggregation helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

RUNTIME_ENDPOINT_PROJECTION_SCHEMA_VERSION = 1
WEIGHT_VERSION_PROJECTION_SCHEMA_VERSION = 1
RELOAD_RESPONSE_PROJECTION_SCHEMA_VERSION = 1
PUBLISHED_REPLICA_PROJECTION_SCHEMA_VERSION = 1
SOURCE_SELECTION_PROJECTION_SCHEMA_VERSION = 1

_PENDING_PUBLICATION_STATES = {"publishing", "retiring"}


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _optional_text(value: Any) -> str | None:
    return _optional_str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, Iterable):
        return []
    return [str(item) for item in value]


def _diagnostic_value(
    diagnostics: Any,
    name: str,
    default: object | None = None,
) -> object | None:
    if isinstance(diagnostics, Mapping):
        return diagnostics.get(name, default)
    return getattr(diagnostics, name, default)


def _serving_realization_report(
    diagnostics: Mapping[str, object],
) -> Mapping[str, object] | None:
    value = diagnostics.get("serving_realization_report")
    if isinstance(value, Mapping):
        return value
    return None


def _artifact_realization_report(
    diagnostics: Mapping[str, object],
) -> Mapping[str, object] | None:
    value = diagnostics.get("artifact_realization_report")
    if isinstance(value, Mapping):
        return value
    return None


def _nested_mapping(
    value: Mapping[str, object] | None,
    key: str,
) -> Mapping[str, object] | None:
    if value is None:
        return None
    nested = value.get(key)
    if isinstance(nested, Mapping):
        return nested
    return None


def _nested_value(
    value: Mapping[str, object] | None,
    *path: str,
) -> object | None:
    current: object | None = value
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


@dataclass(frozen=True)
class BindingValueRefProjection:
    binding_id: str
    binding_layout_id: str
    binding_value_id: str
    seal_generation: int

    @classmethod
    def from_value(cls, value: object) -> "BindingValueRefProjection | None":
        if value is None:
            return None
        if isinstance(value, Mapping):
            return cls(
                binding_id=str(value.get("binding_id", "") or ""),
                binding_layout_id=str(value.get("binding_layout_id", "") or ""),
                binding_value_id=str(value.get("binding_value_id", "") or ""),
                seal_generation=int(value.get("seal_generation", 0) or 0),
            )
        return cls(
            binding_id=str(getattr(value, "binding_id", "") or ""),
            binding_layout_id=str(getattr(value, "binding_layout_id", "") or ""),
            binding_value_id=str(getattr(value, "binding_value_id", "") or ""),
            seal_generation=int(getattr(value, "seal_generation", 0) or 0),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "binding_id": self.binding_id,
            "binding_layout_id": self.binding_layout_id,
            "binding_value_id": self.binding_value_id,
            "seal_generation": self.seal_generation,
        }


@dataclass(frozen=True)
class SourceBoundContractProjection:
    fields: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return dict(self.fields)


@dataclass(frozen=True)
class MaterializationDiagnosticsProjection:
    fields: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return dict(self.fields)


@dataclass(frozen=True)
class ReloadRequestProjection:
    artifact_locator: Mapping[str, object] | None = None
    policy: Mapping[str, object] | None = None
    requested_at: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        if self.artifact_locator is not None:
            payload["artifact_locator"] = dict(self.artifact_locator)
        if self.policy is not None:
            payload["policy"] = dict(self.policy)
        if self.requested_at is not None:
            payload["requested_at"] = self.requested_at
        return payload


@dataclass(frozen=True)
class PublishedReplicaProjection:
    state: str
    operation_id: str | None = None
    replica_id: str | None = None
    lease_id: str | None = None
    artifact_ref: str | None = None
    device_uuid: str | None = None
    owner_pid: int | None = None
    byte_space_kind: str | None = None
    byte_space_id: str | None = None
    binding_layout_id: str | None = None
    binding_value_ref: BindingValueRefProjection | None = None
    generation: str | None = None
    reason: str | None = None
    schema_version: int = PUBLISHED_REPLICA_PROJECTION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "state": self.state,
        }
        optional: dict[str, object | None] = {
            "operation_id": self.operation_id,
            "replica_id": self.replica_id,
            "lease_id": self.lease_id,
            "artifact_ref": self.artifact_ref,
            "device_uuid": self.device_uuid,
            "owner_pid": self.owner_pid,
            "byte_space_kind": self.byte_space_kind,
            "byte_space_id": self.byte_space_id,
            "binding_layout_id": self.binding_layout_id,
            "generation": self.generation,
            "reason": self.reason,
        }
        payload.update(
            {key: value for key, value in optional.items() if value is not None}
        )
        if self.binding_value_ref is not None:
            payload["binding_value_ref"] = self.binding_value_ref.to_dict()
        return payload


@dataclass(frozen=True)
class SourceSelectionProjection:
    selected_source_kind: str
    selected_replica_id: str | None = None
    selected_producer_worker_id: str | None = None
    selected_byte_space_kind: str | None = None
    selected_byte_space_id: str | None = None
    p2p_bytes: int = 0
    fallback_bytes: int = 0
    disk_bytes: int = 0
    reselection_attempts: int = 0
    reject_reason_bucket: str | None = None
    fallback_reason_bucket: str | None = None
    schema_version: int = SOURCE_SELECTION_PROJECTION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "selected_source_kind": self.selected_source_kind,
            "p2p_bytes": self.p2p_bytes,
            "fallback_bytes": self.fallback_bytes,
            "disk_bytes": self.disk_bytes,
            "reselection_attempts": self.reselection_attempts,
        }
        optional: dict[str, object | None] = {
            "selected_replica_id": self.selected_replica_id,
            "selected_producer_worker_id": self.selected_producer_worker_id,
            "selected_byte_space_kind": self.selected_byte_space_kind,
            "selected_byte_space_id": self.selected_byte_space_id,
            "reject_reason_bucket": self.reject_reason_bucket,
            "fallback_reason_bucket": self.fallback_reason_bucket,
        }
        payload.update(
            {key: value for key, value in optional.items() if value is not None}
        )
        return payload


def _source_bound_projection_from_diagnostics(
    diagnostics: Mapping[str, object],
) -> SourceBoundContractProjection | None:
    report = _serving_realization_report(diagnostics)
    source_contract = _nested_mapping(report, "source_bound_contract")
    if source_contract is not None:
        return SourceBoundContractProjection(dict(source_contract))
    fields = {
        "version": (
            _optional_int(diagnostics.get("source_bound_contract_version")) or 0
        ),
        "capability_flags": _optional_string_list(
            diagnostics.get("source_bound_capability_flags")
        ),
        "ready": bool(diagnostics.get("source_bound_contract_ready") or False),
        "path": _optional_text(diagnostics.get("source_bound_contract_path")),
    }
    if not any(value for value in fields.values()):
        return None
    return SourceBoundContractProjection(fields)


def _materialization_projection_from_fields(
    *,
    prefix: str,
    diagnostics: Mapping[str, object],
) -> MaterializationDiagnosticsProjection | None:
    fields: dict[str, object] = {}
    if prefix == "realize":
        report = _serving_realization_report(diagnostics)
        realization = _nested_mapping(report, "realization")
        execution = _nested_mapping(realization, "execution")
        plan = _nested_mapping(realization, "plan")
        if execution is not None:
            fields.update(dict(execution))
        if plan is not None:
            fields.update(dict(plan))
    diagnostics_prefix = f"{prefix}_"
    for key, value in diagnostics.items():
        if key.startswith(diagnostics_prefix) and value is not None:
            fields[key[len(diagnostics_prefix) :]] = value
    if not fields:
        return None
    return MaterializationDiagnosticsProjection(fields)


def _reload_request_projection_from_diagnostics(
    diagnostics: Mapping[str, object],
) -> ReloadRequestProjection | None:
    value = diagnostics.get("reload_request")
    if value is None:
        return None
    if isinstance(value, ReloadRequestProjection):
        return value
    if not isinstance(value, Mapping):
        return None
    artifact_locator = value.get("artifact_locator")
    policy = value.get("policy")
    return ReloadRequestProjection(
        artifact_locator=(
            dict(artifact_locator) if isinstance(artifact_locator, Mapping) else None
        ),
        policy=dict(policy) if isinstance(policy, Mapping) else None,
        requested_at=_optional_text(value.get("requested_at")),
    )


def _published_replica_projection_from_value(
    value: object | None,
) -> PublishedReplicaProjection | None:
    if value is None:
        return None
    if isinstance(value, PublishedReplicaProjection):
        return value
    if not isinstance(value, Mapping):
        return None
    binding_value_ref = BindingValueRefProjection.from_value(
        value.get("binding_value_ref")
    )
    owner_pid = _optional_int(value.get("owner_pid"))
    return PublishedReplicaProjection(
        state=str(value.get("state") or ""),
        operation_id=_optional_text(value.get("operation_id")),
        replica_id=_optional_text(value.get("replica_id")),
        lease_id=_optional_text(value.get("lease_id")),
        artifact_ref=_optional_text(value.get("artifact_ref")),
        device_uuid=_optional_text(value.get("device_uuid")),
        owner_pid=owner_pid,
        byte_space_kind=_optional_text(value.get("byte_space_kind")),
        byte_space_id=_optional_text(value.get("byte_space_id")),
        binding_layout_id=_optional_text(value.get("binding_layout_id")),
        binding_value_ref=binding_value_ref,
        generation=_optional_text(value.get("generation")),
        reason=_optional_text(value.get("reason")),
    )


def _dominant_reason_bucket(value: object | None) -> str | None:
    if not isinstance(value, Mapping):
        return None
    candidates: list[tuple[int, str]] = []
    for key, count in value.items():
        name = _optional_text(key)
        weight = _optional_int(count) or 0
        if name is not None and weight > 0:
            candidates.append((weight, name))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


def source_selection_projection_from_materialization_diagnostics(
    diagnostics: Any | None,
) -> SourceSelectionProjection | None:
    """Project store materialization diagnostics into the runtime endpoint DTO."""

    if diagnostics is None:
        return None
    source = _optional_text(_diagnostic_value(diagnostics, "source"))
    if source is None:
        return None
    total_bytes = _optional_int(_diagnostic_value(diagnostics, "total_bytes")) or 0
    reselection_attempts = max(
        0,
        (_optional_int(_diagnostic_value(diagnostics, "retry_attempts")) or 1) - 1,
    )
    reason_bucket = _dominant_reason_bucket(
        _diagnostic_value(diagnostics, "retry_reason_buckets")
    )
    replica_id = _optional_text(
        _diagnostic_value(diagnostics, "ticket_replica_uuid")
    ) or _optional_text(_diagnostic_value(diagnostics, "replica_uuid"))
    if replica_id is None:
        replica_id = _optional_text(_diagnostic_value(diagnostics, "replica_id"))
    if source == "p2p":
        return SourceSelectionProjection(
            selected_source_kind="published_memory_replica",
            selected_replica_id=replica_id,
            p2p_bytes=total_bytes,
            reselection_attempts=reselection_attempts,
            reject_reason_bucket=reason_bucket,
        )
    if source == "local_replica":
        return SourceSelectionProjection(
            selected_source_kind="local_memory_replica",
            selected_replica_id=replica_id,
            reselection_attempts=reselection_attempts,
            reject_reason_bucket=reason_bucket,
        )
    if source == "disk":
        return SourceSelectionProjection(
            selected_source_kind="canonical_fallback",
            fallback_bytes=total_bytes,
            disk_bytes=total_bytes,
            reselection_attempts=reselection_attempts,
            fallback_reason_bucket=reason_bucket,
        )
    return None


def source_selection_projection_from_execution_diagnostics(
    diagnostics: Any | None,
) -> SourceSelectionProjection | None:
    """Summarize daemon execution diagnostics as low-cardinality source choice."""

    if diagnostics is None:
        return None
    collective_bytes = (
        _optional_int(
            _diagnostic_value(diagnostics, "actual_collective_committed_bytes")
        )
        or 0
    )
    peer_transfer_bytes = (
        _optional_int(_diagnostic_value(diagnostics, "collective_peer_transfer_bytes"))
        or 0
    )
    local_typed_bytes = (
        _optional_int(_diagnostic_value(diagnostics, "actual_local_typed_bytes")) or 0
    )
    generic_bytes = (
        _optional_int(_diagnostic_value(diagnostics, "actual_generic_backend_bytes"))
        or 0
    )
    fallback_bytes = (
        _optional_int(_diagnostic_value(diagnostics, "fallback_bytes")) or 0
    )
    residual_bytes = (
        _optional_int(_diagnostic_value(diagnostics, "residual_bytes")) or 0
    )
    skip_reason = _optional_text(
        _diagnostic_value(diagnostics, "collective_skip_reason")
    )
    if collective_bytes or peer_transfer_bytes:
        return SourceSelectionProjection(
            selected_source_kind="published_memory_replica",
            p2p_bytes=peer_transfer_bytes or collective_bytes,
            fallback_bytes=fallback_bytes,
            fallback_reason_bucket=skip_reason if fallback_bytes else None,
        )
    if local_typed_bytes:
        return SourceSelectionProjection(
            selected_source_kind="local_memory_replica",
            fallback_bytes=fallback_bytes,
            fallback_reason_bucket=skip_reason if fallback_bytes else None,
        )
    if fallback_bytes or residual_bytes or generic_bytes:
        return SourceSelectionProjection(
            selected_source_kind="canonical_fallback",
            fallback_bytes=fallback_bytes or residual_bytes or generic_bytes,
            fallback_reason_bucket=skip_reason,
        )
    return None


def source_selection_projection_from_artifact_realization_report(
    report: Mapping[str, object] | None,
) -> SourceSelectionProjection | None:
    """Project source choice from the unified artifact realization report."""

    if report is None:
        return None
    materialization = _diagnostic_value(report, "materialization_diagnostics")
    projection = source_selection_projection_from_materialization_diagnostics(
        materialization
    )
    if projection is not None:
        return projection

    execution_commit = _nested_mapping(report, "execution_commit")
    projection = source_selection_projection_from_execution_diagnostics(
        execution_commit
    )
    if projection is not None:
        return projection

    envelope = _nested_mapping(report, "envelope")
    strategy_plan = _nested_mapping(report, "strategy_plan")
    if envelope is None and strategy_plan is None:
        return None

    source = _optional_text(report.get("source"))
    fallback_policy = (
        None
        if strategy_plan is None
        else _optional_text(strategy_plan.get("fallback_policy"))
    )
    fallback_reason_bucket = (
        None
        if envelope is None
        else _dominant_reason_bucket(envelope.get("fallback_reason_buckets"))
    )
    copy_bytes = (
        0 if envelope is None else (_optional_int(envelope.get("copy_bytes")) or 0)
    )
    temporary_bytes = (
        0
        if envelope is None
        else (_optional_int(envelope.get("temporary_replica_bytes")) or 0)
    )
    retained_bytes = (
        0 if envelope is None else (_optional_int(envelope.get("retained_bytes")) or 0)
    )
    direct_write_bytes = (
        0
        if envelope is None
        else (_optional_int(envelope.get("direct_write_bytes")) or 0)
    )
    total_bytes = max(copy_bytes, temporary_bytes, retained_bytes, direct_write_bytes)

    if source in {"p2p", "local_replica", "disk"}:
        return source_selection_projection_from_materialization_diagnostics(
            {
                "source": source,
                "total_bytes": total_bytes,
                "retry_attempts": 1,
                "retry_reason_buckets": (
                    {}
                    if fallback_reason_bucket is None
                    else {fallback_reason_bucket: 1}
                ),
            }
        )

    if (
        copy_bytes > 0
        or temporary_bytes > 0
        or fallback_reason_bucket is not None
        or fallback_policy == "generic_fallback"
    ):
        return SourceSelectionProjection(
            selected_source_kind="canonical_fallback",
            fallback_bytes=max(copy_bytes, temporary_bytes),
            fallback_reason_bucket=fallback_reason_bucket or fallback_policy,
        )

    return None


def source_selection_projection_from_runtime_diagnostics(
    diagnostics: Mapping[str, object] | None,
) -> SourceSelectionProjection | None:
    """Project runtime diagnostics into source selection from source facts."""

    if diagnostics is None:
        return None
    materialization = source_selection_projection_from_materialization_diagnostics(
        diagnostics.get("materialization")
    )
    if materialization is not None:
        return materialization
    execution = source_selection_projection_from_execution_diagnostics(
        diagnostics.get("execution")
    )
    if execution is not None:
        return execution
    report = _serving_realization_report(diagnostics)
    realization = _nested_mapping(report, "realization")
    report_execution = _nested_mapping(realization, "execution")
    serving_projection = source_selection_projection_from_execution_diagnostics(
        report_execution
    )
    if serving_projection is not None:
        return serving_projection
    return source_selection_projection_from_artifact_realization_report(
        _artifact_realization_report(diagnostics)
    )


@dataclass(frozen=True)
class WeightVersionProjection:
    source_artifact_ref: str | None
    serving_artifact_ref: str | None
    serving_version_key: str | None
    serving_manifest_ref: str | None
    representation_contract_hash: str
    serving_build_digest: str | None
    tensor_schema_hash: str
    readiness: str
    family: str
    tp_rank: int | None
    tp_world_size: int | None
    binding_layout_id: str | None
    local_serving_ref: str | None
    binding_value_ref: BindingValueRefProjection | None
    verification_state: str
    verification_job_id: str | None
    source_bound_contract: SourceBoundContractProjection | None = None
    realize_diagnostics: MaterializationDiagnosticsProjection | None = None
    publish_diagnostics: MaterializationDiagnosticsProjection | None = None
    published_replica: PublishedReplicaProjection | None = None
    source_selection: SourceSelectionProjection | None = None
    reload_request: ReloadRequestProjection | None = None
    schema_version: int = WEIGHT_VERSION_PROJECTION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "source_artifact_ref": self.source_artifact_ref,
            "serving_artifact_ref": self.serving_artifact_ref,
            "serving_version_key": self.serving_version_key,
            "serving_manifest_ref": self.serving_manifest_ref,
            "representation_contract_hash": self.representation_contract_hash,
            "serving_build_digest": self.serving_build_digest,
            "tensor_schema_hash": self.tensor_schema_hash,
            "readiness": self.readiness,
            "family": self.family,
            "tp_rank": self.tp_rank,
            "tp_world_size": self.tp_world_size,
            "binding_layout_id": self.binding_layout_id,
            "local_serving_ref": self.local_serving_ref,
            "binding_value_ref": (
                None
                if self.binding_value_ref is None
                else self.binding_value_ref.to_dict()
            ),
            "verification_state": self.verification_state,
            "verification_job_id": self.verification_job_id,
        }
        optional = {
            "source_bound_contract": self.source_bound_contract,
            "realize_diagnostics": self.realize_diagnostics,
            "publish_diagnostics": self.publish_diagnostics,
            "published_replica": self.published_replica,
            "source_selection": self.source_selection,
            "reload_request": self.reload_request,
        }
        for key, value in optional.items():
            if value is not None:
                payload[key] = value.to_dict()
        return payload


@dataclass(frozen=True)
class ReloadResponseProjection:
    serving_artifact_ref: str | None
    representation_contract_hash: str
    serving_build_digest: str | None
    readiness: str
    schema_version: int = RELOAD_RESPONSE_PROJECTION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "serving_artifact_ref": self.serving_artifact_ref,
            "representation_contract_hash": self.representation_contract_hash,
            "serving_build_digest": self.serving_build_digest,
            "readiness": self.readiness,
        }


@dataclass(frozen=True)
class RuntimeEndpointProjection:
    weight_version: WeightVersionProjection
    reload_response: ReloadResponseProjection | None = None
    schema_version: int = RUNTIME_ENDPOINT_PROJECTION_SCHEMA_VERSION

    def to_weight_version_payload(self) -> dict[str, object]:
        return self.weight_version.to_dict()

    def to_reload_response_payload(self) -> dict[str, object] | None:
        if self.reload_response is None:
            return None
        return self.reload_response.to_dict()


@dataclass(frozen=True)
class RuntimeWorkerView:
    readiness: str
    serving_artifact_ref: str | None
    source_artifact_ref: str | None
    representation_contract_hash: str
    serving_build_digest: str | None
    tensor_schema_hash: str
    local_serving_ref: str | None
    binding_value_ref: BindingValueRefProjection | None
    verification_state: str
    verification_job_id: str | None
    endpoint: RuntimeEndpointProjection
    diagnostics: Mapping[str, object]

    @classmethod
    def from_runtime_view(
        cls,
        view: Any,
        *,
        family: str = "",
        tp_rank: int | None = None,
        tp_world_size: int | None = None,
        include_reload_response: bool = False,
    ) -> "RuntimeWorkerView":
        diagnostics = dict(getattr(view, "diagnostics", None) or {})
        report = _serving_realization_report(diagnostics)
        binding_value_ref = BindingValueRefProjection.from_value(
            getattr(view, "binding_value_ref", None)
        )
        serving_build_digest = _optional_text(
            _nested_value(report, "serving_build_digest")
        ) or _optional_text(diagnostics.get("serving_build_digest"))
        verification_state = str(
            _nested_value(
                report,
                "realization",
                "binding_value",
                "verification_state",
            )
            or diagnostics.get("verification_state")
            or "verified"
        )
        verification_job_id = _optional_text(
            _nested_value(
                report,
                "realization",
                "binding_value",
                "verification_job_id",
            )
        ) or _optional_text(diagnostics.get("verification_job_id"))
        source_bound_projection = _source_bound_projection_from_diagnostics(diagnostics)
        realize_projection = _materialization_projection_from_fields(
            prefix="realize",
            diagnostics=diagnostics,
        )
        publish_projection = _materialization_projection_from_fields(
            prefix="publish",
            diagnostics=diagnostics,
        )
        reload_projection = _reload_request_projection_from_diagnostics(diagnostics)
        published_replica = _published_replica_projection_from_value(
            diagnostics.get("published_replica")
        )
        source_selection = source_selection_projection_from_runtime_diagnostics(
            diagnostics
        )
        report_tp_rank = _optional_int(_nested_value(report, "tp_rank"))
        diagnostics_tp_rank = _optional_int(diagnostics.get("tp_rank"))
        report_tp_world_size = _optional_int(_nested_value(report, "tp_world_size"))
        diagnostics_tp_world_size = _optional_int(diagnostics.get("tp_world_size"))
        weight_version = WeightVersionProjection(
            source_artifact_ref=getattr(view, "source_artifact_ref", None),
            serving_artifact_ref=getattr(view, "serving_artifact_ref", None),
            serving_version_key=_optional_text(diagnostics.get("serving_version_key")),
            serving_manifest_ref=_optional_text(
                _nested_value(report, "serving_manifest_ref")
            )
            or _optional_text(diagnostics.get("serving_manifest_ref")),
            representation_contract_hash=getattr(
                view, "representation_contract_hash", ""
            ),
            serving_build_digest=serving_build_digest,
            tensor_schema_hash=getattr(view, "tensor_schema_hash", ""),
            readiness=getattr(view, "readiness", ""),
            family=family
            or str(_nested_value(report, "family") or diagnostics.get("family") or ""),
            tp_rank=(
                tp_rank
                if tp_rank is not None
                else (
                    report_tp_rank
                    if report_tp_rank is not None
                    else diagnostics_tp_rank
                )
            ),
            tp_world_size=(
                tp_world_size
                if tp_world_size is not None
                else (
                    report_tp_world_size
                    if report_tp_world_size is not None
                    else diagnostics_tp_world_size
                )
            ),
            binding_layout_id=_optional_text(
                _nested_value(report, "realization", "binding_layout_id")
            )
            or _optional_text(diagnostics.get("binding_layout_id"))
            or (
                None
                if binding_value_ref is None
                else _optional_text(binding_value_ref.binding_layout_id)
            ),
            local_serving_ref=getattr(view, "local_serving_ref", None),
            binding_value_ref=binding_value_ref,
            verification_state=verification_state,
            verification_job_id=verification_job_id,
            source_bound_contract=source_bound_projection,
            realize_diagnostics=realize_projection,
            publish_diagnostics=publish_projection,
            published_replica=published_replica,
            source_selection=source_selection,
            reload_request=reload_projection,
        )
        reload_response = None
        if include_reload_response:
            reload_response = ReloadResponseProjection(
                serving_artifact_ref=getattr(view, "serving_artifact_ref", None),
                representation_contract_hash=getattr(
                    view, "representation_contract_hash", ""
                ),
                serving_build_digest=serving_build_digest,
                readiness=getattr(view, "readiness", ""),
            )
        return cls(
            readiness=getattr(view, "readiness", ""),
            serving_artifact_ref=getattr(view, "serving_artifact_ref", None),
            source_artifact_ref=getattr(view, "source_artifact_ref", None),
            representation_contract_hash=getattr(
                view, "representation_contract_hash", ""
            ),
            serving_build_digest=serving_build_digest,
            tensor_schema_hash=getattr(view, "tensor_schema_hash", ""),
            local_serving_ref=getattr(view, "local_serving_ref", None),
            binding_value_ref=binding_value_ref,
            verification_state=verification_state,
            verification_job_id=verification_job_id,
            endpoint=RuntimeEndpointProjection(
                weight_version=weight_version,
                reload_response=reload_response,
            ),
            diagnostics=diagnostics,
        )


def _worker_publication_state(payload: Mapping[str, Any]) -> str:
    projection = payload.get("published_replica")
    if not isinstance(projection, Mapping):
        return "unpublished"
    state = str(projection.get("state") or "").strip().lower()
    return state or "unpublished"


def publication_aggregate(
    worker_payloads: Iterable[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Aggregate per-worker published replica projections.

    The aggregate is intentionally conservative: a single published worker is
    ``partial`` until every required worker reports ``published``.
    """

    payloads = [dict(payload) for payload in worker_payloads]
    if not any("published_replica" in payload for payload in payloads):
        return None

    states = [_worker_publication_state(payload) for payload in payloads]
    required_workers = len(states)
    published_workers = sum(state == "published" for state in states)
    failed_workers = sum(state == "failed" for state in states)
    stale_workers = sum(state == "stale" for state in states)
    pending_workers = sum(state in _PENDING_PUBLICATION_STATES for state in states)
    disabled_workers = sum(state == "disabled" for state in states)

    if failed_workers:
        aggregate_state = "failed"
    elif stale_workers:
        aggregate_state = "stale"
    elif published_workers == required_workers:
        aggregate_state = "published"
    elif published_workers:
        aggregate_state = "partial"
    elif pending_workers:
        aggregate_state = "publishing"
    elif disabled_workers == required_workers:
        aggregate_state = "disabled"
    else:
        aggregate_state = "unpublished"

    return {
        "schema_version": 1,
        "state": aggregate_state,
        "mode": "runtime_view",
        "published_workers": published_workers,
        "required_workers": required_workers,
        "failed_workers": failed_workers,
        "pending_workers": pending_workers,
        "stale_workers": stale_workers,
    }


def aggregate_runtime_view_outputs(
    outputs: Iterable[Any],
    *,
    response_name: str,
) -> dict[str, Any] | None:
    """Aggregate runtime endpoint payloads returned by multiple workers."""

    worker_payloads: list[dict[str, Any]] = []
    for payload in outputs:
        if payload is None:
            continue
        if not isinstance(payload, dict):
            raise RuntimeError(f"{response_name} worker response must be a dict")
        worker_payloads.append(dict(payload))
    if not worker_payloads:
        return None

    result = dict(worker_payloads[0])
    aggregate = publication_aggregate(worker_payloads)
    if aggregate is not None:
        result["publication_aggregate"] = aggregate
    return result


__all__ = [
    "BindingValueRefProjection",
    "MaterializationDiagnosticsProjection",
    "PUBLISHED_REPLICA_PROJECTION_SCHEMA_VERSION",
    "PublishedReplicaProjection",
    "RELOAD_RESPONSE_PROJECTION_SCHEMA_VERSION",
    "RUNTIME_ENDPOINT_PROJECTION_SCHEMA_VERSION",
    "ReloadRequestProjection",
    "ReloadResponseProjection",
    "RuntimeEndpointProjection",
    "RuntimeWorkerView",
    "SOURCE_SELECTION_PROJECTION_SCHEMA_VERSION",
    "SourceBoundContractProjection",
    "SourceSelectionProjection",
    "WEIGHT_VERSION_PROJECTION_SCHEMA_VERSION",
    "WeightVersionProjection",
    "aggregate_runtime_view_outputs",
    "publication_aggregate",
    "source_selection_projection_from_artifact_realization_report",
    "source_selection_projection_from_execution_diagnostics",
    "source_selection_projection_from_materialization_diagnostics",
    "source_selection_projection_from_runtime_diagnostics",
]
