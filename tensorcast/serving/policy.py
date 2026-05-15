#  Copyright (c) 2026, TensorCast Team.

"""Serving artifact selection and runtime policy schema."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

import tensorcast as tc

_SELECTOR_KINDS = {"version_key", "artifact_ref"}
_POLICY_MODES = {"from_manifest", "pinned"}


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


class ServingSelector(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str
    value: str

    @field_validator("kind", mode="before")
    @classmethod
    def _normalize_kind(cls, value: Any) -> str:
        return _normalize_enum(
            value,
            allowed=_SELECTOR_KINDS,
            field_name="serving.selector.kind",
        )

    @field_validator("value", mode="before")
    @classmethod
    def _normalize_value(cls, value: Any) -> str:
        normalized = _normalize_optional_text(value)
        if normalized is None:
            raise ValueError("serving.selector.value is required")
        return normalized

    def resolve_artifact_ref(self) -> str:
        if self.kind == "artifact_ref":
            return self.value

        from tensorcast.api.store import get_runtime_context

        artifact_id, _disk_path = get_runtime_context().resolve_key_mapping_cached(
            key=self.value
        )
        if not artifact_id:
            raise ValueError(
                "serving.selector version_key did not resolve to a serving "
                f"artifact: {self.value!r}"
            )
        return artifact_id


class ServingPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: str = "from_manifest"
    manifest_ref: str | None = None
    representation_contract_hash: str | None = None
    serving_build_digest: str | None = None

    @field_validator("mode", mode="before")
    @classmethod
    def _normalize_mode(cls, value: Any) -> str:
        if value is None:
            return "from_manifest"
        return _normalize_enum(
            value,
            allowed=_POLICY_MODES,
            field_name="serving.policy.mode",
        )

    @field_validator(
        "manifest_ref",
        "representation_contract_hash",
        "serving_build_digest",
        mode="before",
    )
    @classmethod
    def _normalize_optional_fields(cls, value: Any) -> Any:
        return _normalize_optional_text(value)

    @model_validator(mode="after")
    def _validate_pinned_policy(self) -> ServingPolicy:
        if self.mode != "pinned":
            return self
        missing = [
            name
            for name, value in (
                ("manifest_ref", self.manifest_ref),
                (
                    "representation_contract_hash",
                    self.representation_contract_hash,
                ),
                ("serving_build_digest", self.serving_build_digest),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                f"serving.policy.mode='pinned' requires {', '.join(missing)}"
            )
        return self

    def to_runtime_policy(self) -> Any | None:
        if self.mode == "from_manifest":
            return None
        return tc.ServingRuntimePolicy(
            require_manifest=True,
            serving_manifest_ref=self.manifest_ref,
            expected_representation_contract_hash=(self.representation_contract_hash),
            expected_serving_build_digest=self.serving_build_digest,
        )
