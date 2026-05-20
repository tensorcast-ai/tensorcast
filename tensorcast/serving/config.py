#  Copyright (c) 2026, TensorCast Team.

"""Public TensorCast serving artifact runtime configuration schema."""

from __future__ import annotations

from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, field_validator

from tensorcast.serving.policy import ServingPolicy, ServingSelector
from tensorcast.serving.preload import PreloadSettings
from tensorcast.serving.runtime import RuntimeSettings

_BOOTSTRAP_MODES = {"disabled", "auto", "required"}
_COLLECTIVE_MODES = {"auto", "required", "disabled"}
_REPLICA_PUBLICATION_MODES = {"disabled", "optional", "required"}
_REPLICA_PUBLICATION_TRIGGERS = {"after_vllm_ready"}
_TOP_LEVEL_KEYS = {
    "runtime",
    "serving",
    "bootstrap",
    "materialization",
    "preload",
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

    selector: ServingSelector | None = None
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
    preload: PreloadSettings = PreloadSettings()
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
        return cls.model_validate(dict(payload))

    def to_mapping(self) -> dict[str, Any]:
        return self.model_dump(mode="python")
