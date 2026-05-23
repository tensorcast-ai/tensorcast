#  Copyright (c) 2026, TensorCast Team.

"""Public TensorCast serving artifact runtime configuration schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tensorcast.serving.policy import ServingArtifactLocator, ServingPolicy
from tensorcast.serving.retained_binding import RetainedBindingAcquireSettings
from tensorcast.serving.runtime_config import RuntimeSettings

_BOOTSTRAP_MODES = {"disabled", "auto", "required"}
_COLLECTIVE_MODES = {"auto", "required", "disabled"}
_REPLICA_PUBLICATION_MODES = {"disabled", "optional", "required"}
_REPLICA_PUBLICATION_TRIGGERS = {"after_vllm_ready"}
_TOP_LEVEL_KEYS = {
    "runtime",
    "serving",
    "bootstrap",
    "materialization",
    "retained_binding_acquire",
    "diagnostics",
    "replica_publication",
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


class ServingSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_locator: ServingArtifactLocator | None = None
    policy: ServingPolicy = ServingPolicy()


class BootstrapSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: str = "auto"
    cache_dir: str | None = None
    verify_source_checksums: bool = True

    @field_validator("mode", mode="before")
    @classmethod
    def _normalize_mode(cls, value: Any) -> str:
        if value is None:
            return "auto"
        return _normalize_enum(
            value,
            allowed=_BOOTSTRAP_MODES,
            field_name="bootstrap.mode",
        )

    @field_validator("cache_dir", mode="before")
    @classmethod
    def _normalize_optional_fields(cls, value: Any) -> Any:
        return _normalize_optional_text(value)


class MaterializationSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    collective: str = "auto"

    @field_validator("collective", mode="before")
    @classmethod
    def _normalize_collective(cls, value: Any) -> str:
        if value is None:
            return "auto"
        return _normalize_enum(
            value,
            allowed=_COLLECTIVE_MODES,
            field_name="materialization.collective",
        )

    def collective_policy_value(self) -> str:
        return {
            "auto": "collective_first",
            "required": "require_collective",
            "disabled": "disable_collective",
        }[self.collective]


class DiagnosticsSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    debug_path: str | None = None
    verify_tensors: bool = False

    @field_validator("debug_path", mode="before")
    @classmethod
    def _normalize_debug_path(cls, value: Any) -> Any:
        return _normalize_optional_text(value)


class ReplicaPublicationPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: str = "disabled"
    trigger: str = "after_vllm_ready"
    async_publish: bool = True
    timeout_s: float = 30.0
    ttl_ms: int | None = None
    drain_timeout_s: float = 30.0

    @field_validator("mode", mode="before")
    @classmethod
    def _normalize_mode(cls, value: Any) -> str:
        if value is None:
            return "disabled"
        return _normalize_enum(
            value,
            allowed=_REPLICA_PUBLICATION_MODES,
            field_name="replica_publication.mode",
        )

    @field_validator("trigger", mode="before")
    @classmethod
    def _normalize_trigger(cls, value: Any) -> str:
        if value is None:
            return "after_vllm_ready"
        return _normalize_enum(
            value,
            allowed=_REPLICA_PUBLICATION_TRIGGERS,
            field_name="replica_publication.trigger",
        )

    @field_validator("async_publish")
    @classmethod
    def _validate_async_publish(cls, value: bool) -> bool:
        if not value:
            raise ValueError("replica_publication.async_publish=false is not supported")
        return value

    @field_validator("timeout_s", "drain_timeout_s")
    @classmethod
    def _validate_positive_timeout(cls, value: float) -> float:
        normalized = float(value)
        if normalized <= 0:
            raise ValueError("replica_publication timeouts must be positive")
        return normalized

    @field_validator("ttl_ms")
    @classmethod
    def _reject_ttl(cls, value: int | None) -> int | None:
        if value is not None:
            raise ValueError("replica_publication.ttl_ms is not supported yet")
        return value


class ServingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    runtime: RuntimeSettings = RuntimeSettings()
    serving: ServingSettings = ServingSettings()
    bootstrap: BootstrapSettings = BootstrapSettings()
    materialization: MaterializationSettings = MaterializationSettings()
    retained_binding_acquire: RetainedBindingAcquireSettings = Field(
        default_factory=RetainedBindingAcquireSettings,
    )
    diagnostics: DiagnosticsSettings = DiagnosticsSettings()
    replica_publication: ReplicaPublicationPolicy = ReplicaPublicationPolicy()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> ServingConfig:
        payload: Mapping[str, Any] = {} if data is None else data
        if not isinstance(payload, Mapping):
            raise ValueError("model_loader_extra_config must be a mapping")
        unknown = {str(key) for key in payload if str(key) not in _TOP_LEVEL_KEYS}
        if unknown:
            raise ValueError(
                "Unexpected TensorCast serving config keys in "
                "model_loader_extra_config: "
                f"{sorted(unknown)}"
            )
        serving = payload.get("serving")
        if isinstance(serving, Mapping) and "selector" in serving:
            raise ValueError(
                "serving.selector is not supported; use serving.artifact_locator"
            )
        return cls.model_validate(dict(payload))

    def to_mapping(self) -> dict[str, Any]:
        return self.model_dump(mode="python")


class ServingStartPlanError(ValueError):
    """Startup configuration cannot be lowered into one serving plan."""


@dataclass(frozen=True)
class ServingStartPlan:
    """Typed serving startup intent selected before runtime allocation."""

    kind: str = field(init=False)


@dataclass(frozen=True)
class ArtifactBindStartPlan(ServingStartPlan):
    """Bind a durable serving artifact selected by an artifact locator."""

    artifact_locator: ServingArtifactLocator
    policy: ServingPolicy
    kind: str = field(default="artifact_bind", init=False)


@dataclass(frozen=True)
class SourceBootstrapToBindingStartPlan(ServingStartPlan):
    """Bootstrap a source artifact into a daemon-owned binding value."""

    source_selector: Any
    bootstrap_policy: BootstrapSettings
    kind: str = field(default="source_bootstrap_to_binding", init=False)


@dataclass(frozen=True)
class RetainedBindingAcquireStartPlan(ServingStartPlan):
    """Acquire a retained binding authority prepared by artifact prefetch."""

    authority: Any
    kind: str = field(default="retained_binding_acquire", init=False)


def _candidate_rejection_reasons(
    *,
    has_retained_authority: bool,
    has_artifact_locator: bool,
    has_source_selector: bool,
    bootstrap_mode: str,
) -> dict[str, str]:
    source_reason = (
        "bootstrap.mode is disabled"
        if bootstrap_mode == "disabled"
        else "source selector is unavailable"
    )
    return {
        "retained_binding_acquire": (
            "selected"
            if has_retained_authority
            else "retained_binding_acquire.mode is not external"
        ),
        "artifact_bind": (
            "selected" if has_artifact_locator else "serving artifact locator missing"
        ),
        "source_bootstrap_to_binding": (
            "selected"
            if has_source_selector and bootstrap_mode in {"auto", "required"}
            else source_reason
        ),
    }


def _format_rejection_reasons(reasons: Mapping[str, str]) -> str:
    return "; ".join(f"{name}: {reason}" for name, reason in reasons.items())


def plan_serving_start(
    *,
    config: ServingConfig,
    source_selector: Any | None,
    expected_member: Any | None = None,
) -> ServingStartPlan:
    """Classify serving startup into exactly one canonical start plan."""

    retained_requested = config.retained_binding_acquire.mode == "external"
    artifact_locator = config.serving.artifact_locator
    has_artifact_locator = artifact_locator is not None
    bootstrap_mode = config.bootstrap.mode
    has_source_selector = source_selector is not None

    if retained_requested and has_artifact_locator:
        raise ServingStartPlanError(
            "TensorCast serving config cannot request both retained binding "
            "acquire and durable serving artifact bind"
        )
    if bootstrap_mode == "required" and (retained_requested or has_artifact_locator):
        raise ServingStartPlanError(
            "TensorCast bootstrap.mode='required' is mutually exclusive with "
            "retained binding acquire and durable serving artifact bind"
        )
    if bootstrap_mode == "disabled" and not (
        retained_requested or has_artifact_locator
    ):
        raise ServingStartPlanError(
            "TensorCast bootstrap.mode='disabled' requires retained binding "
            "authority or durable serving artifact locator"
        )

    if retained_requested:
        from tensorcast.serving.retained_binding import (
            parse_retained_serving_binding_authority,
        )

        return RetainedBindingAcquireStartPlan(
            authority=parse_retained_serving_binding_authority(
                config,
                expected_member=expected_member,
            )
        )
    if artifact_locator is not None:
        return ArtifactBindStartPlan(
            artifact_locator=artifact_locator,
            policy=config.serving.policy,
        )
    if bootstrap_mode in {"auto", "required"} and source_selector is not None:
        return SourceBootstrapToBindingStartPlan(
            source_selector=source_selector,
            bootstrap_policy=config.bootstrap,
        )

    reasons = _candidate_rejection_reasons(
        has_retained_authority=retained_requested,
        has_artifact_locator=has_artifact_locator,
        has_source_selector=has_source_selector,
        bootstrap_mode=bootstrap_mode,
    )
    raise ServingStartPlanError(
        "TensorCast serving config did not resolve to one startup plan; "
        f"rejected candidates: {_format_rejection_reasons(reasons)}"
    )
