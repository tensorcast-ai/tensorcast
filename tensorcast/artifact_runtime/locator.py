#  Copyright (c) 2026, TensorCast Team.

"""Artifact runtime locator schema and resolution helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, field_validator

_ARTIFACT_LOCATOR_KINDS = {"version_key", "artifact_ref", "ranked_version_key"}
ARTIFACT_LOCATOR_SCHEMA_VERSION = 1
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
            "ranked_version_key artifact locator resolution requires a member"
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


class ArtifactLocator(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str
    value: str
    schema_version: int = ARTIFACT_LOCATOR_SCHEMA_VERSION

    @field_validator("kind", mode="before")
    @classmethod
    def _normalize_kind(cls, value: Any) -> str:
        return _normalize_enum(
            value,
            allowed=_ARTIFACT_LOCATOR_KINDS,
            field_name="artifact_locator.kind",
        )

    @field_validator("value", mode="before")
    @classmethod
    def _normalize_value(cls, value: Any) -> str:
        normalized = _normalize_optional_text(value)
        if normalized is None:
            raise ValueError("artifact_locator.value is required")
        return normalized

    @classmethod
    def artifact_ref(cls, artifact_ref: str) -> ArtifactLocator:
        return cls(kind="artifact_ref", value=str(artifact_ref))

    @classmethod
    def version_key(cls, version_key: str) -> ArtifactLocator:
        return cls(kind="version_key", value=str(version_key))

    @classmethod
    def ranked_version_key(cls, version_key: str) -> ArtifactLocator:
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

        resolved_mapping = get_runtime_context().resolve_key_mapping_cached(
            key=self.resolve_version_key(member=member, placement=placement)
        )
        artifact_id = (
            resolved_mapping[0]
            if isinstance(resolved_mapping, tuple)
            else getattr(resolved_mapping, "artifact_id", None)
        )
        if not artifact_id:
            raise ValueError(
                "artifact locator version key did not resolve to an artifact: "
                f"{self.value!r}"
            )
        return artifact_id


__all__ = [
    "ARTIFACT_LOCATOR_SCHEMA_VERSION",
    "ArtifactLocator",
    "RANKED_VERSION_KEY_MEMBER_SEGMENT",
    "ranked_version_key_for_member",
]
