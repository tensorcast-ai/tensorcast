#  Copyright (c) 2026, TensorCast Team.

"""Serving artifact selection and runtime policy schema."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

import tensorcast as tc

_SELECTOR_KINDS = {"version_key", "artifact_ref"}
_POLICY_MODES = {"from_manifest", "pinned"}
SERVING_SELECTOR_SCHEMA_VERSION = 1
SERVING_POLICY_SCHEMA_VERSION = 1


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
    schema_version: int = SERVING_SELECTOR_SCHEMA_VERSION

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

    @classmethod
    def artifact_ref(cls, artifact_ref: str) -> ServingSelector:
        return cls(kind="artifact_ref", value=str(artifact_ref))

    @classmethod
    def version_key(cls, version_key: str) -> ServingSelector:
        return cls(kind="version_key", value=str(version_key))

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
    schema_version: int = SERVING_POLICY_SCHEMA_VERSION

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


def normalize_serving_reload_request_payload(
    *,
    selector: ServingSelector | Mapping[str, Any],
    policy: ServingPolicy | Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize public reload selector/policy data to the stable wire shape."""

    parsed_selector = (
        selector
        if isinstance(selector, ServingSelector)
        else ServingSelector.model_validate(selector)
    )
    parsed_policy = (
        policy
        if isinstance(policy, ServingPolicy)
        else ServingPolicy.model_validate(policy or {"mode": "from_manifest"})
    )
    selector_payload = {
        "kind": parsed_selector.kind,
        "value": parsed_selector.value,
    }
    policy_payload: dict[str, Any] = {"mode": parsed_policy.mode}
    if parsed_policy.manifest_ref is not None:
        policy_payload["manifest_ref"] = parsed_policy.manifest_ref
    if parsed_policy.representation_contract_hash is not None:
        policy_payload["representation_contract_hash"] = (
            parsed_policy.representation_contract_hash
        )
    if parsed_policy.serving_build_digest is not None:
        policy_payload["serving_build_digest"] = parsed_policy.serving_build_digest
    return selector_payload, policy_payload


def merge_serving_reload_extra_config(
    extra: Mapping[str, Any] | None,
    *,
    selector: ServingSelector | Mapping[str, Any],
    policy: ServingPolicy | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return model_loader_extra_config with a normalized serving reload request."""

    normalized_selector, normalized_policy = normalize_serving_reload_request_payload(
        selector=selector,
        policy=policy,
    )
    merged_extra = dict(extra or {})
    serving = dict(merged_extra.get("serving", {}))
    serving["selector"] = normalized_selector
    serving["policy"] = normalized_policy
    merged_extra["serving"] = serving
    return merged_extra
