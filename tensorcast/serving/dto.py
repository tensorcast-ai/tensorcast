#  Copyright (c) 2026, TensorCast Team.

"""Serving artifact runtime DTOs shared by framework integrations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tensorcast.serving.policy import ServingSelector
from tensorcast.types import (
    BindingValueRef,
    ServingBindingMemberRef,
    ServingTopologyRef,
)


def _normalize_manifest_ref_payload(data: Any) -> Any:
    if not isinstance(data, Mapping):
        return data
    payload = dict(data)
    manifest_ref = payload.pop("manifest_ref", None)
    if manifest_ref is not None:
        existing = payload.get("serving_manifest_ref")
        if existing is not None and existing != manifest_ref:
            raise ValueError("manifest_ref and serving_manifest_ref must match")
        payload["serving_manifest_ref"] = manifest_ref
    return payload


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value)
    return normalized or None


def _model_dump_or_none(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        payload = dump(mode="python")
        if isinstance(payload, Mapping):
            return {str(key): payload[key] for key in payload}
        raise TypeError(f"Cannot serialize {type(value)!r} as a mapping")
    if isinstance(value, Mapping):
        return {str(key): value[key] for key in value}
    raise TypeError(f"Cannot serialize {type(value)!r} as a mapping")


def _text(payload: Mapping[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return str(value)
    return default


def _optional_bool(payload: Mapping[str, Any], key: str) -> bool | None:
    if key not in payload:
        return None
    value = payload.get(key)
    return None if value is None else bool(value)


def _optional_int(payload: Mapping[str, Any], key: str) -> int | None:
    if key not in payload:
        return None
    value = payload.get(key)
    return None if value is None else int(value)


class BootstrapSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_artifact_ref: str
    serving_artifact_ref: str | None = None
    serving_manifest_ref: str
    representation_contract_hash: str
    serving_build_digest: str
    binding_value_ref: BindingValueRef | None = None
    readiness: str = "serving_published_ready"
    binding_layout_id: str | None = None
    local_serving_ref: str | None = None
    verification_state: str = "verified"
    verification_job_id: str | None = None
    source_bound_contract_version: int = 0
    source_bound_capability_flags: tuple[str, ...] = ()
    source_bound_contract_ready: bool = False
    source_bound_contract_path: str | None = None
    realize_collective_policy: str | None = None
    realize_collective_requested: bool | None = None
    realize_collective_acknowledged: bool | None = None
    realize_collective_used: bool | None = None
    realize_collective_failure_class: str | None = None
    realize_dominant_executor: str | None = None
    realize_direct_write_supported: bool | None = None
    realize_fallback_bytes: int | None = None
    realize_residual_bytes: int | None = None
    realize_actual_collective_committed_bytes: int | None = None
    realize_actual_local_typed_bytes: int | None = None
    realize_actual_generic_backend_bytes: int | None = None
    realize_collective_skip_reason: str | None = None
    realize_hash_rounds: int | None = None
    realize_hash_backend: str | None = None
    realize_hash_bytes: int | None = None
    realize_hash_wall_time_ms: int | None = None
    realize_hash_identity_forming: bool | None = None
    realize_hash_location: str | None = None
    realize_identity_mint_strategy: str | None = None
    realize_execution_plan_kind: str | None = None
    realize_planned_collective_candidate_bytes: int | None = None
    realize_planned_collective_admitted_bytes: int | None = None
    realize_planned_local_typed_bytes: int | None = None
    realize_planned_non_admitted_typed_bytes: int | None = None
    realize_planned_generic_residual_bytes: int | None = None
    realize_compatibility_lowered_bytes: int | None = None
    realize_planner_reject_reason_buckets: dict[str, int] | None = None
    realize_planner_version: str | None = None
    realize_plan_hash: str | None = None
    realize_estimated_collective_peak_temporary_bytes: int | None = None
    realize_estimated_collective_batch_bytes: int | None = None
    realize_estimated_collective_dedup_saving_bytes: int | None = None
    publish_collective_policy: str | None = None
    publish_collective_requested: bool | None = None
    publish_collective_acknowledged: bool | None = None
    publish_collective_used: bool | None = None
    publish_collective_failure_class: str | None = None
    publish_dominant_executor: str | None = None
    publish_direct_write_supported: bool | None = None
    publish_fallback_bytes: int | None = None
    publish_residual_bytes: int | None = None
    publish_actual_collective_committed_bytes: int | None = None
    publish_actual_local_typed_bytes: int | None = None
    publish_actual_generic_backend_bytes: int | None = None
    publish_collective_skip_reason: str | None = None
    publish_hash_rounds: int | None = None
    publish_hash_backend: str | None = None
    publish_hash_bytes: int | None = None
    publish_hash_wall_time_ms: int | None = None
    publish_hash_identity_forming: bool | None = None
    publish_hash_location: str | None = None
    publish_identity_mint_strategy: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_input(cls, data: Any) -> Any:
        return _normalize_manifest_ref_payload(data)

    @property
    def manifest_ref(self) -> str:
        return self.serving_manifest_ref

    def to_dict(self) -> dict[str, Any]:
        return {
            "bootstrap_source_artifact_ref": self.source_artifact_ref,
            "bootstrap_serving_artifact_ref": self.serving_artifact_ref,
            "bootstrap_serving_manifest_ref": self.serving_manifest_ref,
            "bootstrap_representation_contract_hash": self.representation_contract_hash,
            "bootstrap_serving_build_digest": self.serving_build_digest,
            "bootstrap_binding_value_ref": _model_dump_or_none(self.binding_value_ref),
            "bootstrap_readiness": self.readiness,
            "bootstrap_binding_layout_id": self.binding_layout_id,
            "bootstrap_local_serving_ref": self.local_serving_ref,
            "bootstrap_verification_state": self.verification_state,
            "bootstrap_verification_job_id": self.verification_job_id,
            "bootstrap_source_bound_contract_version": self.source_bound_contract_version,
            "bootstrap_source_bound_capability_flags": list(
                self.source_bound_capability_flags
            ),
            "bootstrap_source_bound_contract_ready": self.source_bound_contract_ready,
            "bootstrap_source_bound_contract_path": self.source_bound_contract_path,
            "bootstrap_realize_collective_policy": self.realize_collective_policy,
            "bootstrap_realize_collective_requested": self.realize_collective_requested,
            "bootstrap_realize_collective_acknowledged": self.realize_collective_acknowledged,
            "bootstrap_realize_collective_used": self.realize_collective_used,
            "bootstrap_realize_collective_failure_class": self.realize_collective_failure_class,
            "bootstrap_realize_dominant_executor": self.realize_dominant_executor,
            "bootstrap_realize_direct_write_supported": self.realize_direct_write_supported,
            "bootstrap_realize_fallback_bytes": self.realize_fallback_bytes,
            "bootstrap_realize_residual_bytes": self.realize_residual_bytes,
            "bootstrap_realize_actual_collective_committed_bytes": self.realize_actual_collective_committed_bytes,
            "bootstrap_realize_actual_local_typed_bytes": self.realize_actual_local_typed_bytes,
            "bootstrap_realize_actual_generic_backend_bytes": self.realize_actual_generic_backend_bytes,
            "bootstrap_realize_collective_skip_reason": self.realize_collective_skip_reason,
            "bootstrap_realize_hash_rounds": self.realize_hash_rounds,
            "bootstrap_realize_hash_backend": self.realize_hash_backend,
            "bootstrap_realize_hash_bytes": self.realize_hash_bytes,
            "bootstrap_realize_hash_wall_time_ms": self.realize_hash_wall_time_ms,
            "bootstrap_realize_hash_identity_forming": self.realize_hash_identity_forming,
            "bootstrap_realize_hash_location": self.realize_hash_location,
            "bootstrap_realize_identity_mint_strategy": self.realize_identity_mint_strategy,
            "bootstrap_realize_execution_plan_kind": self.realize_execution_plan_kind,
            "bootstrap_realize_planned_collective_candidate_bytes": self.realize_planned_collective_candidate_bytes,
            "bootstrap_realize_planned_collective_admitted_bytes": self.realize_planned_collective_admitted_bytes,
            "bootstrap_realize_planned_local_typed_bytes": self.realize_planned_local_typed_bytes,
            "bootstrap_realize_planned_non_admitted_typed_bytes": self.realize_planned_non_admitted_typed_bytes,
            "bootstrap_realize_planned_generic_residual_bytes": self.realize_planned_generic_residual_bytes,
            "bootstrap_realize_compatibility_lowered_bytes": self.realize_compatibility_lowered_bytes,
            "bootstrap_realize_planner_reject_reason_buckets": self.realize_planner_reject_reason_buckets,
            "bootstrap_realize_planner_version": self.realize_planner_version,
            "bootstrap_realize_plan_hash": self.realize_plan_hash,
            "bootstrap_realize_estimated_collective_peak_temporary_bytes": self.realize_estimated_collective_peak_temporary_bytes,
            "bootstrap_realize_estimated_collective_batch_bytes": self.realize_estimated_collective_batch_bytes,
            "bootstrap_realize_estimated_collective_dedup_saving_bytes": self.realize_estimated_collective_dedup_saving_bytes,
            "bootstrap_publish_collective_policy": self.publish_collective_policy,
            "bootstrap_publish_collective_requested": self.publish_collective_requested,
            "bootstrap_publish_collective_acknowledged": self.publish_collective_acknowledged,
            "bootstrap_publish_collective_used": self.publish_collective_used,
            "bootstrap_publish_collective_failure_class": self.publish_collective_failure_class,
            "bootstrap_publish_dominant_executor": self.publish_dominant_executor,
            "bootstrap_publish_direct_write_supported": self.publish_direct_write_supported,
            "bootstrap_publish_fallback_bytes": self.publish_fallback_bytes,
            "bootstrap_publish_residual_bytes": self.publish_residual_bytes,
            "bootstrap_publish_actual_collective_committed_bytes": self.publish_actual_collective_committed_bytes,
            "bootstrap_publish_actual_local_typed_bytes": self.publish_actual_local_typed_bytes,
            "bootstrap_publish_actual_generic_backend_bytes": self.publish_actual_generic_backend_bytes,
            "bootstrap_publish_collective_skip_reason": self.publish_collective_skip_reason,
            "bootstrap_publish_hash_rounds": self.publish_hash_rounds,
            "bootstrap_publish_hash_backend": self.publish_hash_backend,
            "bootstrap_publish_hash_bytes": self.publish_hash_bytes,
            "bootstrap_publish_hash_wall_time_ms": self.publish_hash_wall_time_ms,
            "bootstrap_publish_hash_identity_forming": self.publish_hash_identity_forming,
            "bootstrap_publish_hash_location": self.publish_hash_location,
            "bootstrap_publish_identity_mint_strategy": self.publish_identity_mint_strategy,
            "bootstrap_rank_local_artifact_ids_present": True,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BootstrapSummary:
        raw_flags = payload.get("bootstrap_source_bound_capability_flags")
        if raw_flags is None or isinstance(raw_flags, int):
            capability_flags: tuple[str, ...] = ()
        elif isinstance(raw_flags, (str, bytes)):
            capability_flags = (str(raw_flags),)
        else:
            capability_flags = tuple(str(item) for item in raw_flags)

        buckets = payload.get("bootstrap_realize_planner_reject_reason_buckets")
        reject_buckets = (
            None
            if buckets is None
            else {str(key): int(value) for key, value in dict(buckets).items()}
        )

        return cls(
            source_artifact_ref=_text(payload, "bootstrap_source_artifact_ref"),
            serving_artifact_ref=_optional_text(
                payload.get("bootstrap_serving_artifact_ref")
            ),
            serving_manifest_ref=_text(payload, "bootstrap_serving_manifest_ref"),
            representation_contract_hash=_text(
                payload,
                "bootstrap_representation_contract_hash",
            ),
            serving_build_digest=_text(payload, "bootstrap_serving_build_digest"),
            binding_value_ref=(
                None
                if payload.get("bootstrap_binding_value_ref") is None
                else BindingValueRef.model_validate(
                    payload.get("bootstrap_binding_value_ref")
                )
            ),
            readiness=_text(
                payload,
                "bootstrap_readiness",
                default="serving_published_ready",
            ),
            binding_layout_id=_optional_text(
                payload.get("bootstrap_binding_layout_id")
            ),
            local_serving_ref=_optional_text(
                payload.get("bootstrap_local_serving_ref")
            ),
            verification_state=_text(
                payload,
                "bootstrap_verification_state",
                default="verified",
            ),
            verification_job_id=_optional_text(
                payload.get("bootstrap_verification_job_id")
            ),
            source_bound_contract_version=_optional_int(
                payload, "bootstrap_source_bound_contract_version"
            )
            or 0,
            source_bound_capability_flags=capability_flags,
            source_bound_contract_ready=(
                _optional_bool(payload, "bootstrap_source_bound_contract_ready")
                or False
            ),
            source_bound_contract_path=_optional_text(
                payload.get("bootstrap_source_bound_contract_path")
            ),
            realize_collective_policy=_optional_text(
                payload.get("bootstrap_realize_collective_policy")
            ),
            realize_collective_requested=_optional_bool(
                payload, "bootstrap_realize_collective_requested"
            ),
            realize_collective_acknowledged=_optional_bool(
                payload, "bootstrap_realize_collective_acknowledged"
            ),
            realize_collective_used=_optional_bool(
                payload, "bootstrap_realize_collective_used"
            ),
            realize_collective_failure_class=_optional_text(
                payload.get("bootstrap_realize_collective_failure_class")
            ),
            realize_dominant_executor=_optional_text(
                payload.get("bootstrap_realize_dominant_executor")
            ),
            realize_direct_write_supported=_optional_bool(
                payload, "bootstrap_realize_direct_write_supported"
            ),
            realize_fallback_bytes=_optional_int(
                payload, "bootstrap_realize_fallback_bytes"
            ),
            realize_residual_bytes=_optional_int(
                payload, "bootstrap_realize_residual_bytes"
            ),
            realize_actual_collective_committed_bytes=_optional_int(
                payload, "bootstrap_realize_actual_collective_committed_bytes"
            ),
            realize_actual_local_typed_bytes=_optional_int(
                payload, "bootstrap_realize_actual_local_typed_bytes"
            ),
            realize_actual_generic_backend_bytes=_optional_int(
                payload, "bootstrap_realize_actual_generic_backend_bytes"
            ),
            realize_collective_skip_reason=_optional_text(
                payload.get("bootstrap_realize_collective_skip_reason")
            ),
            realize_hash_rounds=_optional_int(payload, "bootstrap_realize_hash_rounds"),
            realize_hash_backend=_optional_text(
                payload.get("bootstrap_realize_hash_backend")
            ),
            realize_hash_bytes=_optional_int(payload, "bootstrap_realize_hash_bytes"),
            realize_hash_wall_time_ms=_optional_int(
                payload, "bootstrap_realize_hash_wall_time_ms"
            ),
            realize_hash_identity_forming=_optional_bool(
                payload, "bootstrap_realize_hash_identity_forming"
            ),
            realize_hash_location=_optional_text(
                payload.get("bootstrap_realize_hash_location")
            ),
            realize_identity_mint_strategy=_optional_text(
                payload.get("bootstrap_realize_identity_mint_strategy")
            ),
            realize_execution_plan_kind=_optional_text(
                payload.get("bootstrap_realize_execution_plan_kind")
            ),
            realize_planned_collective_candidate_bytes=_optional_int(
                payload, "bootstrap_realize_planned_collective_candidate_bytes"
            ),
            realize_planned_collective_admitted_bytes=_optional_int(
                payload, "bootstrap_realize_planned_collective_admitted_bytes"
            ),
            realize_planned_local_typed_bytes=_optional_int(
                payload, "bootstrap_realize_planned_local_typed_bytes"
            ),
            realize_planned_non_admitted_typed_bytes=_optional_int(
                payload, "bootstrap_realize_planned_non_admitted_typed_bytes"
            ),
            realize_planned_generic_residual_bytes=_optional_int(
                payload, "bootstrap_realize_planned_generic_residual_bytes"
            ),
            realize_compatibility_lowered_bytes=_optional_int(
                payload, "bootstrap_realize_compatibility_lowered_bytes"
            ),
            realize_planner_reject_reason_buckets=reject_buckets,
            realize_planner_version=_optional_text(
                payload.get("bootstrap_realize_planner_version")
            ),
            realize_plan_hash=_optional_text(
                payload.get("bootstrap_realize_plan_hash")
            ),
            realize_estimated_collective_peak_temporary_bytes=_optional_int(
                payload, "bootstrap_realize_estimated_collective_peak_temporary_bytes"
            ),
            realize_estimated_collective_batch_bytes=_optional_int(
                payload, "bootstrap_realize_estimated_collective_batch_bytes"
            ),
            realize_estimated_collective_dedup_saving_bytes=_optional_int(
                payload, "bootstrap_realize_estimated_collective_dedup_saving_bytes"
            ),
            publish_collective_policy=_optional_text(
                payload.get("bootstrap_publish_collective_policy")
            ),
            publish_collective_requested=_optional_bool(
                payload, "bootstrap_publish_collective_requested"
            ),
            publish_collective_acknowledged=_optional_bool(
                payload, "bootstrap_publish_collective_acknowledged"
            ),
            publish_collective_used=_optional_bool(
                payload, "bootstrap_publish_collective_used"
            ),
            publish_collective_failure_class=_optional_text(
                payload.get("bootstrap_publish_collective_failure_class")
            ),
            publish_dominant_executor=_optional_text(
                payload.get("bootstrap_publish_dominant_executor")
            ),
            publish_direct_write_supported=_optional_bool(
                payload, "bootstrap_publish_direct_write_supported"
            ),
            publish_fallback_bytes=_optional_int(
                payload, "bootstrap_publish_fallback_bytes"
            ),
            publish_residual_bytes=_optional_int(
                payload, "bootstrap_publish_residual_bytes"
            ),
            publish_actual_collective_committed_bytes=_optional_int(
                payload, "bootstrap_publish_actual_collective_committed_bytes"
            ),
            publish_actual_local_typed_bytes=_optional_int(
                payload, "bootstrap_publish_actual_local_typed_bytes"
            ),
            publish_actual_generic_backend_bytes=_optional_int(
                payload, "bootstrap_publish_actual_generic_backend_bytes"
            ),
            publish_collective_skip_reason=_optional_text(
                payload.get("bootstrap_publish_collective_skip_reason")
            ),
            publish_hash_rounds=_optional_int(payload, "bootstrap_publish_hash_rounds"),
            publish_hash_backend=_optional_text(
                payload.get("bootstrap_publish_hash_backend")
            ),
            publish_hash_bytes=_optional_int(payload, "bootstrap_publish_hash_bytes"),
            publish_hash_wall_time_ms=_optional_int(
                payload, "bootstrap_publish_hash_wall_time_ms"
            ),
            publish_hash_identity_forming=_optional_bool(
                payload, "bootstrap_publish_hash_identity_forming"
            ),
            publish_hash_location=_optional_text(
                payload.get("bootstrap_publish_hash_location")
            ),
            publish_identity_mint_strategy=_optional_text(
                payload.get("bootstrap_publish_identity_mint_strategy")
            ),
        )


class PreparedServingArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_artifact_ref: str
    serving_artifact_ref: str | None = None
    serving_manifest_ref: str
    representation_contract_hash: str
    serving_build_digest: str
    binding_value_ref: BindingValueRef | None = None
    readiness: str = "serving_published_ready"
    family: str
    tensor_schema_hash: str
    serving_version_key: str | None = None
    binding_layout_id: str | None = None
    local_serving_ref: str | None = None
    verification_state: str = "verified"
    verification_job_id: str | None = None
    tp_rank: int = 0
    tp_world_size: int = 1
    selector: ServingSelector | None = None
    bootstrap_summary: BootstrapSummary | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_input(cls, data: Any) -> Any:
        return _normalize_manifest_ref_payload(data)

    @property
    def manifest_ref(self) -> str:
        return self.serving_manifest_ref

    def to_reload_request(self) -> dict[str, Any]:
        if self.selector is not None:
            selector = self.selector.model_dump(mode="python")
        elif self.serving_version_key is not None:
            selector = {
                "kind": "version_key",
                "value": self.serving_version_key,
            }
        elif self.serving_artifact_ref is not None:
            selector = {
                "kind": "artifact_ref",
                "value": self.serving_artifact_ref,
            }
        else:
            raise RuntimeError(
                "TensorCast local-ready serving result does not reference a "
                "durable serving artifact and cannot be used as a reload "
                "request"
            )
        return {
            "selector": selector,
            "policy": {
                "mode": "pinned",
                "manifest_ref": self.serving_manifest_ref,
                "representation_contract_hash": self.representation_contract_hash,
                "serving_build_digest": self.serving_build_digest,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source_artifact_ref": self.source_artifact_ref,
            "serving_artifact_ref": self.serving_artifact_ref,
            "serving_version_key": self.serving_version_key,
            "serving_manifest_ref": self.serving_manifest_ref,
            "representation_contract_hash": self.representation_contract_hash,
            "serving_build_digest": self.serving_build_digest,
            "binding_value_ref": _model_dump_or_none(self.binding_value_ref),
            "readiness": self.readiness,
            "family": self.family,
            "tensor_schema_hash": self.tensor_schema_hash,
            "binding_layout_id": self.binding_layout_id,
            "local_serving_ref": self.local_serving_ref,
            "verification_state": self.verification_state,
            "verification_job_id": self.verification_job_id,
            "tp_rank": self.tp_rank,
            "tp_world_size": self.tp_world_size,
        }
        try:
            payload["reload_request"] = self.to_reload_request()
        except RuntimeError:
            if (
                self.selector is not None
                or self.serving_version_key is not None
                or self.serving_artifact_ref is not None
            ):
                raise
            payload["reload_request"] = None
        if self.bootstrap_summary is not None:
            payload["bootstrap_summary"] = self.bootstrap_summary.to_dict()
        return payload

    def to_bootstrap_summary(self) -> BootstrapSummary:
        if self.bootstrap_summary is not None:
            return self.bootstrap_summary
        return BootstrapSummary(
            source_artifact_ref=self.source_artifact_ref,
            serving_artifact_ref=self.serving_artifact_ref,
            serving_manifest_ref=self.serving_manifest_ref,
            representation_contract_hash=self.representation_contract_hash,
            serving_build_digest=self.serving_build_digest,
            binding_value_ref=self.binding_value_ref,
            readiness=self.readiness,
            binding_layout_id=self.binding_layout_id,
            local_serving_ref=self.local_serving_ref,
            verification_state=self.verification_state,
            verification_job_id=self.verification_job_id,
        )


class FamilyReadiness(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    family: str
    model_types: tuple[str, ...] = ()
    architectures: tuple[str, ...] = ()
    process_after_load_class: Any | None = None
    post_bind_finalize_class: Any | None = None
    support_level: Any | None = None
    publication_modes: tuple[str, ...] = ()
    runtime_bind_swap_allowed: bool = False
    notes: str = ""


class RuntimeTensorView(BaseModel):
    """Framework-neutral tensor identity view without live tensor payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int = 0
    element_size: int | None = None


class ServingPlacement(BaseModel):
    """Stable runtime placement identity shared with framework integrations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    topology: ServingTopologyRef
    member: ServingBindingMemberRef
    framework_payload: dict[str, Any]
    identity_payload: dict[str, Any]

    def stable_identity_payload(self) -> dict[str, Any]:
        return {
            "topology": self.topology.model_dump(mode="python"),
            "member": self.member.model_dump(mode="python"),
            "framework_payload": self.framework_payload,
            "identity_payload": self.identity_payload,
        }


class FrameworkIntegrationContext(BaseModel):
    """Serializable framework identity facts used by core-owned facades."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    framework_name: str
    framework_version: str
    adapter_version: str
    serving_abi_version: str
    placement: ServingPlacement | None = None
    source_identity: dict[str, Any] = Field(default_factory=dict)

    def stable_identity_payload(self) -> dict[str, Any]:
        return {
            "framework_name": self.framework_name,
            "framework_version": self.framework_version,
            "adapter_version": self.adapter_version,
            "serving_abi_version": self.serving_abi_version,
            "placement": (
                None
                if self.placement is None
                else self.placement.stable_identity_payload()
            ),
            "source_identity": dict(self.source_identity),
        }
