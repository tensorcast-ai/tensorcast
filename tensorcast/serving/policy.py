#  Copyright (c) 2026, TensorCast Team.

"""Serving artifact locator and runtime policy schema."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

import tensorcast as tc

_ARTIFACT_LOCATOR_KINDS = {"version_key", "artifact_ref", "ranked_version_key"}
_POLICY_MODES = {"from_manifest", "pinned"}
SERVING_ARTIFACT_LOCATOR_SCHEMA_VERSION = 1
SERVING_POLICY_SCHEMA_VERSION = 1
RANKED_VERSION_KEY_MEMBER_SEGMENT = "members"


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


def _member_id_from_ref(member: Any) -> str:
    if member is None:
        raise ValueError(
            "ranked_version_key artifact locator resolution requires a serving member"
        )
    if isinstance(member, Mapping):
        member_id = member.get("member_id")
    else:
        member_id = getattr(member, "member_id", None)
    normalized = _normalize_optional_text(member_id)
    if normalized is None:
        raise ValueError(
            "ranked_version_key artifact locator resolution requires member.member_id"
        )
    return normalized


def _member_from_placement(placement: Any | None) -> Any | None:
    if placement is None:
        return None
    if isinstance(placement, Mapping):
        return placement.get("member")
    return getattr(placement, "member", None)


def ranked_version_key_for_member(version_key: str, member: Any) -> str:
    base_key = _normalize_optional_text(version_key)
    if base_key is None:
        raise ValueError("ranked_version_key base value is required")
    member_id = quote(_member_id_from_ref(member), safe=":._-")
    return f"{base_key.rstrip('/')}/{RANKED_VERSION_KEY_MEMBER_SEGMENT}/{member_id}"


class ServingArtifactLocator(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str
    value: str
    schema_version: int = SERVING_ARTIFACT_LOCATOR_SCHEMA_VERSION

    @field_validator("kind", mode="before")
    @classmethod
    def _normalize_kind(cls, value: Any) -> str:
        return _normalize_enum(
            value,
            allowed=_ARTIFACT_LOCATOR_KINDS,
            field_name="serving.artifact_locator.kind",
        )

    @field_validator("value", mode="before")
    @classmethod
    def _normalize_value(cls, value: Any) -> str:
        normalized = _normalize_optional_text(value)
        if normalized is None:
            raise ValueError("serving.artifact_locator.value is required")
        return normalized

    @classmethod
    def artifact_ref(cls, artifact_ref: str) -> ServingArtifactLocator:
        return cls(kind="artifact_ref", value=str(artifact_ref))

    @classmethod
    def version_key(cls, version_key: str) -> ServingArtifactLocator:
        return cls(kind="version_key", value=str(version_key))

    @classmethod
    def ranked_version_key(cls, version_key: str) -> ServingArtifactLocator:
        return cls(kind="ranked_version_key", value=str(version_key))

    def resolve_version_key(
        self,
        *,
        member: Any | None = None,
        placement: Any | None = None,
    ) -> str:
        if self.kind == "artifact_ref":
            return self.value
        if self.kind == "ranked_version_key":
            if member is None:
                member = _member_from_placement(placement)
            return ranked_version_key_for_member(self.value, member)
        return self.value

    def resolve_artifact_ref(
        self,
        *,
        member: Any | None = None,
        placement: Any | None = None,
    ) -> str:
        if self.kind == "artifact_ref":
            return self.value

        from tensorcast.api.store import get_runtime_context

        artifact_id, _disk_path = get_runtime_context().resolve_key_mapping_cached(
            key=self.resolve_version_key(member=member, placement=placement)
        )
        if not artifact_id:
            raise ValueError(
                "serving artifact locator version key did not resolve to a serving "
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
    artifact_locator: ServingArtifactLocator | Mapping[str, Any],
    policy: ServingPolicy | Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize public reload locator/policy data to the stable wire shape."""

    parsed_locator = (
        artifact_locator
        if isinstance(artifact_locator, ServingArtifactLocator)
        else ServingArtifactLocator.model_validate(artifact_locator)
    )
    parsed_policy = (
        policy
        if isinstance(policy, ServingPolicy)
        else ServingPolicy.model_validate(policy or {"mode": "from_manifest"})
    )
    locator_payload = {
        "kind": parsed_locator.kind,
        "value": parsed_locator.value,
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
    return locator_payload, policy_payload


def merge_serving_reload_extra_config(
    extra: Mapping[str, Any] | None,
    *,
    artifact_locator: ServingArtifactLocator | Mapping[str, Any],
    policy: ServingPolicy | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return model_loader_extra_config with a normalized serving reload request."""

    normalized_locator, normalized_policy = normalize_serving_reload_request_payload(
        artifact_locator=artifact_locator,
        policy=policy,
    )
    merged_extra = dict(extra or {})
    serving = dict(merged_extra.get("serving", {}))
    serving["artifact_locator"] = normalized_locator
    serving["policy"] = normalized_policy
    merged_extra["serving"] = serving
    return merged_extra
