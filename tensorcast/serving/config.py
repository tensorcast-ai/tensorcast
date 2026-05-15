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
_TOP_LEVEL_KEYS = {
    "runtime",
    "serving",
    "bootstrap",
    "materialization",
    "preload",
    "diagnostics",
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
    publication_name: str | None = None

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

    @field_validator("cache_dir", "publication_name", mode="before")
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


class ServingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    runtime: RuntimeSettings = RuntimeSettings()
    serving: ServingSettings = ServingSettings()
    bootstrap: BootstrapSettings = BootstrapSettings()
    materialization: MaterializationSettings = MaterializationSettings()
    preload: PreloadSettings = PreloadSettings()
    diagnostics: DiagnosticsSettings = DiagnosticsSettings()

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
