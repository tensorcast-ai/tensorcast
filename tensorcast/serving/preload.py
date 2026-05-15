#  Copyright (c) 2026, TensorCast Team.

"""External preloaded serving binding authority schema."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_PRELOAD_MODES = {"disabled", "external"}
_READINESS_STATES = {
    "serving_reserved",
    "serving_local_ready",
    "serving_published_ready",
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


class ExternalPreloadExpectedDigests(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    target_layout_hash: str
    tensor_schema_hash: str
    serving_build_digest: str
    resolved_spec_digest: str

    @field_validator(
        "target_layout_hash",
        "tensor_schema_hash",
        "serving_build_digest",
        "resolved_spec_digest",
        mode="before",
    )
    @classmethod
    def _normalize_required_text(cls, value: Any) -> str:
        normalized = _normalize_optional_text(value)
        if normalized is None:
            raise ValueError("expected digest fields must be non-empty")
        return normalized


class ExternalPreloadAuthority(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    group_id: str
    member_ref: dict[str, Any]
    daemon_id: str
    daemon_session_id: str
    device_uuid: str
    binding_value_ref: dict[str, Any]
    reservation_capability: dict[str, Any]
    local_serving_ref: str | None = None
    readiness: str
    verification_state: str = "local_only"
    serving_artifact_id: str | None = None
    trusted_reservation_bytes: int = Field(ge=0)
    expected: ExternalPreloadExpectedDigests

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
            raise ValueError("external preload authority text fields required")
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
            field_name="preload.authority.readiness",
        )

    @model_validator(mode="after")
    def _validate_published_ready(self) -> ExternalPreloadAuthority:
        if self.readiness == "serving_published_ready" and not self.serving_artifact_id:
            raise ValueError(
                "preload.authority.serving_artifact_id is required when "
                "readiness='serving_published_ready'"
            )
        return self


class PreloadSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: str = "disabled"
    authority: ExternalPreloadAuthority | None = None

    @field_validator("mode", mode="before")
    @classmethod
    def _normalize_mode(cls, value: Any) -> str:
        if value is None:
            return "disabled"
        return _normalize_enum(
            value,
            allowed=_PRELOAD_MODES,
            field_name="preload.mode",
        )

    @model_validator(mode="after")
    def _validate_authority(self) -> PreloadSettings:
        if self.mode == "external" and self.authority is None:
            raise ValueError(
                "preload.authority is required when preload.mode='external'"
            )
        if self.mode != "external" and self.authority is not None:
            raise ValueError(
                "preload.authority is only valid when preload.mode='external'"
            )
        return self
