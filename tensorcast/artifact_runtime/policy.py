#  Copyright (c) 2026, TensorCast Team.

"""Artifact runtime policy schema and reload request helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from tensorcast.artifact_runtime.locator import ArtifactLocator
from tensorcast.types import RuntimeArtifactPolicy

_POLICY_MODES = {"from_manifest", "pinned"}
RUNTIME_POLICY_SCHEMA_VERSION = 1


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


class RuntimePolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: str = "from_manifest"
    manifest_ref: str | None = None
    representation_contract_hash: str | None = None
    serving_build_digest: str | None = None
    schema_version: int = RUNTIME_POLICY_SCHEMA_VERSION

    @field_validator("mode", mode="before")
    @classmethod
    def _normalize_mode(cls, value: Any) -> str:
        if value is None:
            return "from_manifest"
        return _normalize_enum(
            value,
            allowed=_POLICY_MODES,
            field_name="runtime.policy.mode",
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
    def _validate_pinned_policy(self) -> RuntimePolicy:
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
                f"runtime.policy.mode='pinned' requires {', '.join(missing)}"
            )
        return self

    def to_runtime_policy(self) -> RuntimeArtifactPolicy | None:
        if self.mode == "from_manifest":
            return None
        return RuntimeArtifactPolicy(
            require_manifest=True,
            serving_manifest_ref=self.manifest_ref,
            expected_representation_contract_hash=(self.representation_contract_hash),
            expected_serving_build_digest=self.serving_build_digest,
        )


def normalize_runtime_reload_request_payload(
    *,
    artifact_locator: ArtifactLocator | Mapping[str, Any],
    policy: RuntimePolicy | Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize runtime reload locator/policy data to the stable wire shape."""

    parsed_locator = (
        artifact_locator
        if isinstance(artifact_locator, ArtifactLocator)
        else ArtifactLocator.model_validate(artifact_locator)
    )
    parsed_policy = (
        policy
        if isinstance(policy, RuntimePolicy)
        else RuntimePolicy.model_validate(policy or {"mode": "from_manifest"})
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


def merge_runtime_reload_extra_config(
    extra: Mapping[str, Any] | None,
    *,
    artifact_locator: ArtifactLocator | Mapping[str, Any],
    policy: RuntimePolicy | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return model-loader config with a normalized runtime reload request."""

    normalized_locator, normalized_policy = normalize_runtime_reload_request_payload(
        artifact_locator=artifact_locator,
        policy=policy,
    )
    merged_extra = dict(extra or {})
    if "serving" in merged_extra:
        raise ValueError(
            "TensorCast runtime reload config section 'serving' was removed; "
            "use 'runtime_artifact'"
        )
    runtime_artifact = dict(merged_extra.get("runtime_artifact", {}))
    runtime_artifact["artifact_locator"] = normalized_locator
    runtime_artifact["policy"] = normalized_policy
    merged_extra["runtime_artifact"] = runtime_artifact
    return merged_extra


__all__ = [
    "RUNTIME_POLICY_SCHEMA_VERSION",
    "RuntimePolicy",
    "merge_runtime_reload_extra_config",
    "normalize_runtime_reload_request_payload",
]
