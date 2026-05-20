#  Copyright (c) 2026, TensorCast Team.

"""Runtime settings for TensorCast serving artifact consumers."""

from __future__ import annotations

import importlib.resources
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

import tensorcast as tc

_INIT_LOCK = Lock()
_INIT_KWARGS: dict[str, Any] | None = None
_DEFAULT_GLOBAL_STORE_ADDRESS = "127.0.0.1:50051"
DEFAULT_RUNTIME_PROFILE = "serving_single_node"

_RUNTIME_MODES = {"auto", "connect", "create"}
_GLOBAL_STORE_MODES = {"auto", "connect", "start", "none"}
_PROFILE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


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
        with _INIT_LOCK:
            global _INIT_KWARGS
            if tc.is_initialized():
                if _INIT_KWARGS is None:
                    raise RuntimeError(
                        "TensorCast runtime was already initialized outside "
                        "tensorcast.serving.RuntimeSettings."
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


_PUBLIC_RUNTIME_EXPORTS = {
    "AdmissionRejectedError",
    "AttachFinalizeError",
    "AuthorityValidationError",
    "BootstrapPolicy",
    "CapabilityMissingError",
    "ConfigConflictError",
    "ExistingServingArtifact",
    "LocalSourceBootstrap",
    "OwnershipTransferError",
    "PlacementAdmissionError",
    "PolicyMismatchError",
    "PublishedReplicaProjection",
    "PublicationRequiredError",
    "ReplicaPublicationError",
    "ReloadResponseProjection",
    "RequestContext",
    "RetainedBindingAcquire",
    "RetainedBindingAuthority",
    "RuntimeAttachment",
    "RuntimeEndpointProjection",
    "RuntimeSwapError",
    "RuntimeWorkerView",
    "SchemaMismatchError",
    "SelectorResolutionError",
    "ServingArtifactSelector",
    "ServingConfig",
    "ServingIntegrationError",
    "ServingPolicy",
    "ServingRuntimeSession",
    "ReplicaPublicationPolicy",
    "SourceSelectionProjection",
    "source_selection_projection_from_execution_diagnostics",
    "source_selection_projection_from_materialization_diagnostics",
    "SourceProviderError",
    "SourceSelector",
    "TensorCastServingRuntimeError",
    "WeightVersionProjection",
    "merge_serving_reload_extra_config",
    "normalize_serving_reload_request_payload",
}

_PUBLIC_READINESS_EXPORTS = {
    "ReadinessInventoryAdmissionPolicy",
    "coerce_finalize_class",
    "coerce_serving_support_level",
    "is_binding_finalize_publication_allowlisted",
    "is_pure_transform_publication_allowlisted",
    "is_runtime_bind_swap_allowlisted",
    "readiness_family",
    "readiness_post_bind_finalize_class",
    "readiness_process_after_load_class",
    "readiness_publication_modes",
    "readiness_support_level",
    "serving_support_level_at_least",
    "serving_support_level_display_name",
}

_PUBLIC_STATE_EXPORTS = {
    "ModelAttributeNames",
    "ModelAttributeRuntimeState",
    "OneShotRuntimeHook",
    "RuntimeAttachmentRecord",
    "RuntimeAttachmentStore",
    "attachment_generation_key",
}

_PUBLIC_RUNTIME_VIEW_EXPORTS = {
    "aggregate_runtime_view_outputs",
    "publication_aggregate",
}


def __getattr__(name: str) -> object:
    """Lazily expose the public serving runtime API.

    ``ServingConfig`` imports ``RuntimeSettings`` from this module, while the
    runtime session implementation imports this module for daemon settings.
    Lazy exports avoid a circular import while making
    ``tensorcast.serving.runtime`` the framework-facing import path.
    """

    if name == "ServingPolicy":
        from tensorcast.serving.policy import ServingPolicy

        return ServingPolicy
    if name == "ServingConfig":
        from tensorcast.serving.config import ServingConfig

        return ServingConfig
    if name in _PUBLIC_RUNTIME_EXPORTS:
        from tensorcast.serving import integration as tc_integration

        return getattr(tc_integration, name)
    if name in _PUBLIC_READINESS_EXPORTS:
        from tensorcast.serving import readiness as tc_readiness

        return getattr(tc_readiness, name)
    if name in _PUBLIC_STATE_EXPORTS:
        from tensorcast.serving import state as tc_state

        return getattr(tc_state, name)
    if name in _PUBLIC_RUNTIME_VIEW_EXPORTS:
        from tensorcast.serving import runtime_view as tc_runtime_view

        return getattr(tc_runtime_view, name)
    raise AttributeError(
        f"module 'tensorcast.serving.runtime' has no attribute {name!r}"
    )


__all__ = [  # noqa: F822 - names are resolved lazily by __getattr__.
    "AdmissionRejectedError",
    "AttachFinalizeError",
    "AuthorityValidationError",
    "BootstrapPolicy",
    "CapabilityMissingError",
    "ConfigConflictError",
    "DEFAULT_RUNTIME_PROFILE",
    "ExistingServingArtifact",
    "LocalSourceBootstrap",
    "OwnershipTransferError",
    "PlacementAdmissionError",
    "PolicyMismatchError",
    "PublishedReplicaProjection",
    "PublicationRequiredError",
    "ReplicaPublicationError",
    "ReloadResponseProjection",
    "RequestContext",
    "RetainedBindingAcquire",
    "RetainedBindingAuthority",
    "RuntimeConfigProfile",
    "RuntimeDaemonSettings",
    "RuntimeAttachment",
    "RuntimeEndpointProjection",
    "RuntimeGlobalStoreSettings",
    "ModelAttributeNames",
    "ModelAttributeRuntimeState",
    "OneShotRuntimeHook",
    "RuntimeAttachmentRecord",
    "RuntimeAttachmentStore",
    "RuntimeSettings",
    "RuntimeSwapError",
    "RuntimeWorkerView",
    "SchemaMismatchError",
    "SelectorResolutionError",
    "ServingArtifactSelector",
    "ServingConfig",
    "ServingIntegrationError",
    "ServingPolicy",
    "ServingRuntimeSession",
    "ReplicaPublicationPolicy",
    "SourceSelectionProjection",
    "source_selection_projection_from_execution_diagnostics",
    "source_selection_projection_from_materialization_diagnostics",
    "SourceProviderError",
    "SourceSelector",
    "TensorCastServingRuntimeError",
    "WeightVersionProjection",
    "ReadinessInventoryAdmissionPolicy",
    "attachment_generation_key",
    "aggregate_runtime_view_outputs",
    "coerce_finalize_class",
    "coerce_serving_support_level",
    "is_binding_finalize_publication_allowlisted",
    "is_pure_transform_publication_allowlisted",
    "is_runtime_bind_swap_allowlisted",
    "merge_serving_reload_extra_config",
    "normalize_serving_reload_request_payload",
    "readiness_family",
    "readiness_post_bind_finalize_class",
    "readiness_process_after_load_class",
    "readiness_publication_modes",
    "readiness_support_level",
    "resolve_runtime_config_profile",
    "serving_support_level_at_least",
    "serving_support_level_display_name",
    "publication_aggregate",
]
