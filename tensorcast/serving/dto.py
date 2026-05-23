#  Copyright (c) 2026, TensorCast Team.

"""Serving artifact runtime DTOs shared by framework integrations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tensorcast.serving.policy import ServingArtifactLocator
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


class ServingBindingValue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_artifact_ref: str
    binding_value_ref: BindingValueRef | None = None
    readiness: str
    tensor_schema_hash: str
    serving_manifest_ref: str
    serving_build_digest: str
    family: str
    binding_layout_id: str | None = None
    local_serving_ref: str | None = None
    verification_state: str = "verified"
    verification_job_id: str | None = None
    tp_rank: int = 0
    tp_world_size: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_artifact_ref": self.source_artifact_ref,
            "binding_value_ref": _model_dump_or_none(self.binding_value_ref),
            "readiness": self.readiness,
            "tensor_schema_hash": self.tensor_schema_hash,
            "serving_manifest_ref": self.serving_manifest_ref,
            "serving_build_digest": self.serving_build_digest,
            "family": self.family,
            "binding_layout_id": self.binding_layout_id,
            "local_serving_ref": self.local_serving_ref,
            "verification_state": self.verification_state,
            "verification_job_id": self.verification_job_id,
            "tp_rank": self.tp_rank,
            "tp_world_size": self.tp_world_size,
        }


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
    artifact_locator: ServingArtifactLocator | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_input(cls, data: Any) -> Any:
        return _normalize_manifest_ref_payload(data)

    @property
    def manifest_ref(self) -> str:
        return self.serving_manifest_ref

    def to_binding_value(self) -> ServingBindingValue:
        return ServingBindingValue(
            source_artifact_ref=self.source_artifact_ref,
            binding_value_ref=self.binding_value_ref,
            readiness=self.readiness,
            tensor_schema_hash=self.tensor_schema_hash,
            serving_manifest_ref=self.serving_manifest_ref,
            serving_build_digest=self.serving_build_digest,
            family=self.family,
            binding_layout_id=self.binding_layout_id,
            local_serving_ref=self.local_serving_ref,
            verification_state=self.verification_state,
            verification_job_id=self.verification_job_id,
            tp_rank=self.tp_rank,
            tp_world_size=self.tp_world_size,
        )

    def to_reload_request(self) -> dict[str, Any]:
        if self.artifact_locator is not None:
            artifact_locator = self.artifact_locator.model_dump(mode="python")
        elif self.serving_version_key is not None:
            artifact_locator = {
                "kind": "version_key",
                "value": self.serving_version_key,
            }
        elif self.serving_artifact_ref is not None:
            artifact_locator = {
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
            "artifact_locator": artifact_locator,
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
                self.artifact_locator is not None
                or self.serving_version_key is not None
                or self.serving_artifact_ref is not None
            ):
                raise
            payload["reload_request"] = None
        return payload


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
