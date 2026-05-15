#  Copyright (c) 2026, TensorCast Team.

"""Runtime settings for TensorCast serving artifact consumers."""

from __future__ import annotations

import importlib.resources
from pathlib import Path
from threading import Lock
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

import tensorcast as tc

_INIT_LOCK = Lock()
_INIT_KWARGS: dict[str, Any] | None = None
_DEFAULT_GLOBAL_STORE_ADDRESS = "127.0.0.1:50051"

_RUNTIME_MODES = {"auto", "connect", "create"}
_GLOBAL_STORE_MODES = {"auto", "connect", "start", "none"}


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


def _default_resource_path(package: str, name: str) -> str | None:
    try:
        resource = importlib.resources.files(package).joinpath(name)
    except (FileNotFoundError, ModuleNotFoundError):
        return None
    path = Path(str(resource))
    return str(path) if path.is_file() else None


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

    mode: str = "auto"
    daemon: RuntimeDaemonSettings = RuntimeDaemonSettings()
    global_store: RuntimeGlobalStoreSettings = RuntimeGlobalStoreSettings()

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
        kwargs: dict[str, Any] = {
            "mode": self.mode,
            "show_daemon_logs": self.daemon.show_logs,
        }
        if self.daemon.address is not None:
            kwargs["address"] = self.daemon.address

        daemon_config_path = self.daemon.config_path
        if daemon_config_path is None and self.mode in {"create", "auto"}:
            daemon_config_path = (
                default_daemon_config_path or self._default_daemon_config_path()
            )
        if daemon_config_path is not None:
            kwargs["daemon_config_path"] = daemon_config_path

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
                    default_global_store_config_path
                    or self._default_global_store_config_path()
                )
            if global_store_config_path is not None:
                kwargs["global_store_config_path"] = global_store_config_path

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
