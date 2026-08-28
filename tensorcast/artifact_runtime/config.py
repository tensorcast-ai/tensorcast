#  Copyright (c) 2026, TensorCast Team.

"""Artifact runtime configuration schema and startup planning."""

from __future__ import annotations

import importlib.resources
import re
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tensorcast.artifact_runtime.locator import ArtifactLocator
from tensorcast.artifact_runtime.policy import RuntimePolicy
from tensorcast.retained_realization_authority import (
    ParsedRetainedRealizationAuthority,
    RetainedRealizationAuthority,
)

_INIT_LOCK = Lock()
_INIT_KWARGS: dict[str, Any] | None = None
_DEFAULT_GLOBAL_STORE_ADDRESS = "127.0.0.1:50051"
DEFAULT_RUNTIME_PROFILE = "serving_single_node"

_RUNTIME_MODES = {"auto", "connect", "create"}
_GLOBAL_STORE_MODES = {"auto", "connect", "start", "none"}
_BOOTSTRAP_MODES = {"disabled", "auto", "required"}
_COLLECTIVE_MODES = {
    "auto",
    "required",
    "disabled",
    "collective_first",
    "require_collective",
    "disable_collective",
}
_RETAINED_BINDING_ACQUIRE_MODES = {"disabled", "external"}
_REPLICA_PUBLICATION_MODES = {"disabled", "optional", "required"}
_REPLICA_PUBLICATION_TRIGGERS = {"after_vllm_ready"}
_PROFILE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_TOP_LEVEL_KEYS = {
    "runtime",
    "runtime_artifact",
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


def _validate_existing_file(path: str, *, field_name: str) -> str:
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        raise ValueError(f"{field_name} must point to an existing file, got: {path!r}")
    return str(candidate)


def _default_resource_path(package: str, name: str) -> str | None:
    try:
        resource = importlib.resources.files(package).joinpath(name)
    except (FileNotFoundError, ModuleNotFoundError):
        return None
    path = Path(str(resource))
    return str(path) if path.is_file() else None


def _normalize_profile_name(value: Any) -> str | None:
    normalized = _normalize_optional_text(value)
    if normalized is None:
        return None
    if not _PROFILE_NAME_PATTERN.fullmatch(normalized):
        raise ValueError(
            "runtime.profile must contain only letters, digits, '.', '_', or '-'"
        )
    return normalized


def _profile_resource_path(profile: str, filename: str) -> str:
    profile_name = _normalize_profile_name(profile)
    if profile_name is None:
        raise ValueError("runtime.profile must be non-empty")
    try:
        resource = (
            importlib.resources.files("tensorcast")
            .joinpath("config")
            .joinpath("profiles")
            .joinpath(profile_name)
            .joinpath(filename)
        )
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        raise ValueError(
            f"Unknown TensorCast runtime config profile: {profile_name!r}"
        ) from exc
    path = Path(str(resource))
    if not path.is_file():
        raise ValueError(f"Unknown TensorCast runtime config profile: {profile_name!r}")
    return str(path)


@dataclass(frozen=True)
class RuntimeConfigProfile:
    name: str
    daemon_config_path: str
    global_store_config_path: str


def resolve_runtime_config_profile(profile: str) -> RuntimeConfigProfile:
    profile_name = _normalize_profile_name(profile)
    if profile_name is None:
        raise ValueError("runtime.profile must be non-empty")
    return RuntimeConfigProfile(
        name=profile_name,
        daemon_config_path=_profile_resource_path(
            profile_name, "store_daemon_config.yaml"
        ),
        global_store_config_path=_profile_resource_path(
            profile_name, "global_store_config.yaml"
        ),
    )


class RuntimeDaemonSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    address: str | None = None
    config_path: str | None = None
    show_logs: bool = False

    @field_validator("address", "config_path", mode="before")
    @classmethod
    def _normalize_optional_fields(cls, value: Any) -> Any:
        return _normalize_optional_text(value)


class RuntimeGlobalStoreSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: str = "auto"
    address: str | None = None
    config_path: str | None = None

    @field_validator("mode", mode="before")
    @classmethod
    def _normalize_mode(cls, value: Any) -> str:
        if value is None:
            return "auto"
        return _normalize_enum(
            value,
            allowed=_GLOBAL_STORE_MODES,
            field_name="runtime.global_store.mode",
        )

    @field_validator("address", "config_path", mode="before")
    @classmethod
    def _normalize_optional_fields(cls, value: Any) -> Any:
        return _normalize_optional_text(value)

    def resolved_mode(self, runtime_mode: str) -> str:
        if self.mode != "auto":
            return self.mode
        if self.address is not None:
            return "connect"
        if self.config_path is not None:
            return "start"
        if runtime_mode in {"create", "auto"}:
            return "start"
        return "none"


class RuntimeSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    profile: str | None = DEFAULT_RUNTIME_PROFILE
    mode: str = "auto"
    daemon: RuntimeDaemonSettings = RuntimeDaemonSettings()
    global_store: RuntimeGlobalStoreSettings = RuntimeGlobalStoreSettings()

    @field_validator("profile", mode="before")
    @classmethod
    def _normalize_profile(cls, value: Any) -> str | None:
        return _normalize_profile_name(value)

    @field_validator("mode", mode="before")
    @classmethod
    def _normalize_mode(cls, value: Any) -> str:
        if value is None:
            return "auto"
        return _normalize_enum(
            value,
            allowed=_RUNTIME_MODES,
            field_name="runtime.mode",
        )

    @staticmethod
    def _default_daemon_config_path() -> str | None:
        return _default_resource_path("tensorcast", "daemon_config.yaml")

    @staticmethod
    def _default_global_store_config_path() -> str | None:
        return _default_resource_path("tensorcast", "global_store_config.yaml")

    def to_init_kwargs(
        self,
        *,
        default_daemon_config_path: str | None = None,
        default_global_store_config_path: str | None = None,
    ) -> dict[str, Any]:
        profile = (
            resolve_runtime_config_profile(self.profile)
            if self.profile is not None
            else None
        )
        kwargs: dict[str, Any] = {
            "mode": self.mode,
            "show_daemon_logs": self.daemon.show_logs,
        }
        if self.daemon.address is not None:
            kwargs["address"] = self.daemon.address

        daemon_config_path = self.daemon.config_path
        if daemon_config_path is None and self.mode in {"create", "auto"}:
            daemon_config_path = (
                profile.daemon_config_path
                if profile is not None
                else default_daemon_config_path or self._default_daemon_config_path()
            )
        if daemon_config_path is not None:
            kwargs["daemon_config_path"] = _validate_existing_file(
                daemon_config_path,
                field_name="runtime.daemon.config_path",
            )
        elif self.mode in {"create", "auto"}:
            raise ValueError(
                "runtime.mode requires a daemon config file for create/auto; "
                "set runtime.profile or runtime.daemon.config_path"
            )

        global_store_mode = self.global_store.resolved_mode(self.mode)
        if global_store_mode != "none":
            kwargs["global_store_mode"] = global_store_mode
        if global_store_mode == "connect":
            kwargs["global_store_address"] = (
                self.global_store.address or _DEFAULT_GLOBAL_STORE_ADDRESS
            )
        elif global_store_mode == "start":
            global_store_config_path = self.global_store.config_path
            if global_store_config_path is None:
                global_store_config_path = (
                    profile.global_store_config_path
                    if profile is not None
                    else default_global_store_config_path
                    or self._default_global_store_config_path()
                )
            if global_store_config_path is not None:
                kwargs["global_store_config_path"] = _validate_existing_file(
                    global_store_config_path,
                    field_name="runtime.global_store.config_path",
                )
            else:
                raise ValueError(
                    "runtime.global_store.mode='start' requires a Global "
                    "Store config file; set runtime.profile or "
                    "runtime.global_store.config_path"
                )

        return kwargs

    def ensure_initialized(
        self,
        *,
        default_daemon_config_path: str | None = None,
        default_global_store_config_path: str | None = None,
    ) -> None:
        init_kwargs = self.to_init_kwargs(
            default_daemon_config_path=default_daemon_config_path,
            default_global_store_config_path=default_global_store_config_path,
        )
        import tensorcast as tc

        with _INIT_LOCK:
            global _INIT_KWARGS
            if tc.is_initialized():
                if _INIT_KWARGS is None:
                    raise RuntimeError(
                        "TensorCast runtime was already initialized outside "
                        "tensorcast.artifact_runtime.config.RuntimeSettings."
                    )
                if init_kwargs != _INIT_KWARGS:
                    raise RuntimeError(
                        "TensorCast runtime already initialized with different "
                        "settings. Existing="
                        f"{_INIT_KWARGS}, requested={init_kwargs}"
                    )
                return
            tc.init(**init_kwargs)
            _INIT_KWARGS = dict(init_kwargs)


class RuntimeArtifactSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_locator: ArtifactLocator | None = None
    policy: RuntimePolicy = RuntimePolicy()


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
            "auto": "auto",
            "required": "require_collective",
            "disabled": "disable_collective",
            "collective_first": "collective_first",
            "require_collective": "require_collective",
            "disable_collective": "disable_collective",
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


def _retained_authority(value: Any) -> RetainedRealizationAuthority | None:
    if value is None:
        return None

    if isinstance(value, RetainedRealizationAuthority):
        return value
    return RetainedRealizationAuthority.model_validate(value)


class RetainedBindingAcquireSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: str = "disabled"
    authority: RetainedRealizationAuthority | None = None
    authorities: tuple[RetainedRealizationAuthority, ...] = ()

    @field_validator("mode", mode="before")
    @classmethod
    def _normalize_mode(cls, value: Any) -> str:
        if value is None:
            return "disabled"
        return _normalize_enum(
            value,
            allowed=_RETAINED_BINDING_ACQUIRE_MODES,
            field_name="retained_binding_acquire.mode",
        )

    @field_validator("authority", mode="before")
    @classmethod
    def _validate_authority_value(
        cls,
        value: Any,
    ) -> RetainedRealizationAuthority | None:
        return _retained_authority(value)

    @field_validator("authorities", mode="before")
    @classmethod
    def _validate_authorities_value(
        cls,
        value: Any,
    ) -> tuple[RetainedRealizationAuthority, ...]:
        if value is None:
            return ()
        return tuple(
            authority
            for authority in (_retained_authority(item) for item in value)
            if authority is not None
        )

    @field_validator("authorities")
    @classmethod
    def _validate_authorities(
        cls,
        value: tuple[RetainedRealizationAuthority, ...],
    ) -> tuple[RetainedRealizationAuthority, ...]:
        return value

    @model_validator(mode="after")
    def _validate_authority(self) -> RetainedBindingAcquireSettings:
        has_authority = self.authority is not None
        has_authorities = bool(self.authorities)
        if self.mode == "external" and not (has_authority or has_authorities):
            raise ValueError(
                "retained_binding_acquire.authority or "
                "retained_binding_acquire.authorities is required when "
                "retained_binding_acquire.mode='external'"
            )
        if self.mode == "external" and has_authority and has_authorities:
            raise ValueError(
                "retained_binding_acquire.authority and "
                "retained_binding_acquire.authorities are mutually exclusive"
            )
        if self.mode != "external" and (has_authority or has_authorities):
            raise ValueError(
                "retained_binding_acquire.authority and "
                "retained_binding_acquire.authorities are only valid when "
                "retained_binding_acquire.mode='external'"
            )
        return self


class TensorCastRuntimeConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    runtime: RuntimeSettings = RuntimeSettings()
    runtime_artifact: RuntimeArtifactSettings = RuntimeArtifactSettings()
    bootstrap: BootstrapSettings = BootstrapSettings()
    materialization: MaterializationSettings = MaterializationSettings()
    retained_binding_acquire: RetainedBindingAcquireSettings = Field(
        default_factory=RetainedBindingAcquireSettings,
    )
    diagnostics: DiagnosticsSettings = DiagnosticsSettings()
    replica_publication: ReplicaPublicationPolicy = ReplicaPublicationPolicy()

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any] | None,
    ) -> TensorCastRuntimeConfig:
        payload: Mapping[str, Any] = {} if data is None else data
        if not isinstance(payload, Mapping):
            raise ValueError("model_loader_extra_config must be a mapping")
        if "serving" in payload:
            raise ValueError(
                "TensorCast runtime config section 'serving' was removed; "
                "use 'runtime_artifact'"
            )
        unknown = {str(key) for key in payload if str(key) not in _TOP_LEVEL_KEYS}
        if unknown:
            raise ValueError(
                "Unexpected TensorCast runtime config keys in "
                "model_loader_extra_config: "
                f"{sorted(unknown)}"
            )
        runtime_artifact = payload.get("runtime_artifact")
        if isinstance(runtime_artifact, Mapping) and "selector" in runtime_artifact:
            raise ValueError(
                "runtime_artifact.selector is not supported; "
                "use runtime_artifact.artifact_locator"
            )
        return cls.model_validate(dict(payload))

    def to_mapping(self) -> dict[str, Any]:
        return self.model_dump(mode="python")


class RuntimeStartPlanError(ValueError):
    """Startup configuration cannot be lowered into one runtime plan."""


@dataclass(frozen=True)
class RuntimeStartPlan:
    """Typed artifact runtime startup intent selected before allocation."""

    kind: str = field(init=False)


@dataclass(frozen=True)
class RuntimeArtifactBindStartPlan(RuntimeStartPlan):
    """Bind a durable runtime artifact selected by an artifact locator."""

    artifact_locator: ArtifactLocator
    policy: RuntimePolicy
    kind: str = field(default="artifact_bind", init=False)


@dataclass(frozen=True)
class RuntimeSourceBootstrapStartPlan(RuntimeStartPlan):
    """Bootstrap a source artifact into a daemon-owned binding value."""

    source_selector: Any
    bootstrap_policy: BootstrapSettings
    kind: str = field(default="source_bootstrap_to_binding", init=False)


@dataclass(frozen=True)
class RuntimeRetainedRealizationStartPlan(RuntimeStartPlan):
    """Acquire a retained binding authority prepared by artifact prefetch."""

    authority: ParsedRetainedRealizationAuthority
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
            "selected" if has_artifact_locator else "runtime artifact locator missing"
        ),
        "source_bootstrap_to_binding": (
            "selected"
            if has_source_selector and bootstrap_mode in {"auto", "required"}
            else source_reason
        ),
    }


def _format_rejection_reasons(reasons: Mapping[str, str]) -> str:
    return "; ".join(f"{name}: {reason}" for name, reason in reasons.items())


def plan_runtime_start(
    *,
    config: TensorCastRuntimeConfig,
    source_selector: Any | None,
    expected_member: Any | None = None,
) -> RuntimeStartPlan:
    """Classify artifact runtime startup into exactly one canonical start plan."""

    retained_requested = config.retained_binding_acquire.mode == "external"
    artifact_locator = config.runtime_artifact.artifact_locator
    has_artifact_locator = artifact_locator is not None
    bootstrap_mode = config.bootstrap.mode
    has_source_selector = source_selector is not None

    if retained_requested and has_artifact_locator:
        raise RuntimeStartPlanError(
            "TensorCast runtime config cannot request both retained binding "
            "acquire and durable runtime artifact bind"
        )
    if bootstrap_mode == "required" and (retained_requested or has_artifact_locator):
        raise RuntimeStartPlanError(
            "TensorCast bootstrap.mode='required' is mutually exclusive with "
            "retained binding acquire and durable runtime artifact bind"
        )
    if bootstrap_mode == "disabled" and not (
        retained_requested or has_artifact_locator
    ):
        raise RuntimeStartPlanError(
            "TensorCast bootstrap.mode='disabled' requires retained binding "
            "authority or durable runtime artifact locator"
        )

    if retained_requested:
        from tensorcast.retained_realization import parse_retained_realization_authority

        return RuntimeRetainedRealizationStartPlan(
            authority=parse_retained_realization_authority(
                config,
                expected_member=expected_member,
            )
        )
    if artifact_locator is not None:
        return RuntimeArtifactBindStartPlan(
            artifact_locator=artifact_locator,
            policy=config.runtime_artifact.policy,
        )
    if bootstrap_mode in {"auto", "required"} and source_selector is not None:
        return RuntimeSourceBootstrapStartPlan(
            source_selector=source_selector,
            bootstrap_policy=config.bootstrap,
        )

    reasons = _candidate_rejection_reasons(
        has_retained_authority=retained_requested,
        has_artifact_locator=has_artifact_locator,
        has_source_selector=has_source_selector,
        bootstrap_mode=bootstrap_mode,
    )
    raise RuntimeStartPlanError(
        "TensorCast runtime config did not resolve to one startup plan; "
        f"rejected candidates: {_format_rejection_reasons(reasons)}"
    )


RuntimeArtifactLocator = ArtifactLocator
RuntimeBootstrapSettings = BootstrapSettings
RuntimeDiagnosticsSettings = DiagnosticsSettings
RuntimeMaterializationSettings = MaterializationSettings
RuntimeReplicaPublicationPolicy = ReplicaPublicationPolicy


__all__ = [
    "DEFAULT_RUNTIME_PROFILE",
    "ArtifactLocator",
    "BootstrapSettings",
    "DiagnosticsSettings",
    "MaterializationSettings",
    "ReplicaPublicationPolicy",
    "RetainedBindingAcquireSettings",
    "RuntimeArtifactSettings",
    "RuntimeArtifactBindStartPlan",
    "RuntimeArtifactLocator",
    "RuntimeBootstrapSettings",
    "RuntimeConfigProfile",
    "RuntimeDaemonSettings",
    "RuntimeDiagnosticsSettings",
    "RuntimeGlobalStoreSettings",
    "RuntimeMaterializationSettings",
    "RuntimePolicy",
    "RuntimeReplicaPublicationPolicy",
    "RuntimeRetainedRealizationStartPlan",
    "RuntimeSettings",
    "RuntimeSourceBootstrapStartPlan",
    "RuntimeStartPlan",
    "RuntimeStartPlanError",
    "TensorCastRuntimeConfig",
    "plan_runtime_start",
    "resolve_runtime_config_profile",
]
