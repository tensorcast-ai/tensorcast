#  Copyright (c) 2026, TensorCast Team.

"""Typed retained realization authority models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tensorcast.types import (
    BindingReservationCapability,
    BindingValueRef,
    GroupRealizationAcquireRef,
    RuntimeBindingMemberRef,
)

_READINESS_STATES = {
    "runtime_reserved",
    "runtime_local_ready",
    "runtime_published_ready",
}


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _normalize_enum(value: Any, *, allowed: set[str], field_name: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in allowed:
        raise ValueError(
            f"{field_name} must be one of {sorted(allowed)}, got: {value!r}"
        )
    return normalized


class RetainedRealizationExpectedDigests(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    target_layout_hash: str
    tensor_schema_hash: str
    runtime_build_digest: str
    resolved_spec_digest: str

    @field_validator(
        "target_layout_hash",
        "tensor_schema_hash",
        "runtime_build_digest",
        "resolved_spec_digest",
        mode="before",
    )
    @classmethod
    def _normalize_required_text(cls, value: Any) -> str:
        normalized = _normalize_optional_text(value)
        if normalized is None:
            raise ValueError("expected digest fields must be non-empty")
        return normalized


class RetainedRealizationAuthority(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    group_id: str
    member_ref: dict[str, Any]
    daemon_id: str
    daemon_session_id: str
    device_uuid: str
    binding_value_ref: dict[str, Any]
    reservation_capability: dict[str, Any]
    group_realization_acquire: dict[str, Any] | None = None
    local_serving_ref: str | None = None
    readiness: str
    verification_state: str = "local_only"
    serving_artifact_id: str | None = None
    trusted_reservation_bytes: int = Field(ge=0)
    expected: RetainedRealizationExpectedDigests

    @field_validator(
        "group_id",
        "daemon_id",
        "daemon_session_id",
        "device_uuid",
        mode="before",
    )
    @classmethod
    def _normalize_required_text(cls, value: Any) -> str:
        normalized = _normalize_optional_text(value)
        if normalized is None:
            raise ValueError("retained binding authority text fields required")
        return normalized

    @field_validator(
        "local_serving_ref",
        "verification_state",
        "serving_artifact_id",
        mode="before",
    )
    @classmethod
    def _normalize_optional_fields(cls, value: Any) -> Any:
        return _normalize_optional_text(value)

    @field_validator("readiness", mode="before")
    @classmethod
    def _normalize_readiness(cls, value: Any) -> str:
        return _normalize_enum(
            value,
            allowed=_READINESS_STATES,
            field_name="retained_binding_acquire.authority.readiness",
        )

    @model_validator(mode="after")
    def _validate_published_ready(self) -> RetainedRealizationAuthority:
        if self.readiness == "runtime_published_ready" and not self.serving_artifact_id:
            raise ValueError(
                "retained_binding_acquire.authority.serving_artifact_id is required when "
                "readiness='runtime_published_ready'"
            )
        return self


@dataclass(frozen=True)
class ParsedRetainedRealizationAuthority:
    group_id: str
    local_serving_ref: str | None
    binding_value_ref: BindingValueRef
    reservation_capability: BindingReservationCapability
    daemon_id: str
    daemon_session_id: str
    device_uuid: str
    member: RuntimeBindingMemberRef
    reservation_bytes: int
    expected: RetainedRealizationExpectedDigests
    readiness: str
    verification_state: str
    serving_artifact_id: str | None = None
    group_realization_acquire: GroupRealizationAcquireRef | None = None


__all__ = [
    "ParsedRetainedRealizationAuthority",
    "RetainedRealizationAuthority",
    "RetainedRealizationExpectedDigests",
]
