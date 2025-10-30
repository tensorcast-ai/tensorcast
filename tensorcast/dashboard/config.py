#  Copyright (c) 2025, TensorCast Team.

"""Configuration helpers for the dashboard backend."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping


def _parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    lowered = value.strip().lower()
    return lowered in {"1", "true", "yes", "y", "on"}


def _parse_uint(value: str | None, *, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value, 10)
    except ValueError as exc:  # noqa: PERF203 - explicit exception conveys intent
        raise ValueError(f"Invalid integer value: {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"Integer value must be non-negative: {parsed}")
    return parsed


def _parse_float(value: str | None, *, default: float) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:  # noqa: PERF203
        raise ValueError(f"Invalid float value: {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"Timeout must be positive: {parsed}")
    return parsed


def _parse_csv(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    entries = [item.strip() for item in value.split(",") if item.strip()]
    return tuple(entries)


@dataclass(frozen=True)
class GrafanaConfig:
    """Optional Grafana embedding configuration."""

    host: str
    dashboard_uid: str
    panel_ids: tuple[str, ...]
    auth: str | None = None


@dataclass(frozen=True)
class DashboardSettings:
    """Runtime settings for the dashboard backend."""

    gs_endpoint: str
    gs_secure: bool
    gs_ca_cert: Path | None
    request_timeout_sec: float
    host: str
    port: int
    base_path: str
    cors_allowed_origins: tuple[str, ...]
    grafana: GrafanaConfig | None

    @property
    def root_path(self) -> str:
        if not self.base_path or self.base_path == "/":
            return ""
        # Ensure FastAPI receives a root path that starts with a single slash
        normalized = self.base_path.strip()
        if not normalized.startswith("/"):
            normalized = "/" + normalized
        return normalized.rstrip("/")

    @property
    def has_cors(self) -> bool:
        return bool(self.cors_allowed_origins)


def _read_env(environ: Mapping[str, str]) -> DashboardSettings:
    gs_endpoint = environ.get("TENSORCAST_GS_ADDR")
    if not gs_endpoint:
        raise ValueError("TENSORCAST_GS_ADDR environment variable is required")

    gs_secure = _parse_bool(environ.get("TENSORCAST_GS_SECURE"), default=False)
    ca_cert = environ.get("TENSORCAST_GS_CA_CERT")
    gs_ca_cert = Path(ca_cert).expanduser() if ca_cert else None

    request_timeout_sec = _parse_float(
        environ.get("TENSORCAST_GS_TIMEOUT_SEC"), default=5.0
    )

    host = environ.get("HOST", "0.0.0.0")
    port = _parse_uint(environ.get("PORT"), default=8080)
    base_path = environ.get("BASE_PATH", "")
    cors_allowed_origins = _parse_csv(environ.get("CORS_ALLOWED_ORIGINS"))

    grafana_host = environ.get("GRAFANA_HOST")
    grafana = None
    if grafana_host:
        panel_ids = _parse_csv(environ.get("GRAFANA_PANEL_IDS"))
        grafana_dashboard_uid = environ.get("GRAFANA_DASHBOARD_UID")
        if not grafana_dashboard_uid:
            raise ValueError(
                "GRAFANA_DASHBOARD_UID is required when GRAFANA_HOST is set"
            )
        grafana = GrafanaConfig(
            host=grafana_host,
            dashboard_uid=grafana_dashboard_uid,
            panel_ids=panel_ids,
            auth=environ.get("GRAFANA_AUTH"),
        )

    return DashboardSettings(
        gs_endpoint=gs_endpoint,
        gs_secure=gs_secure,
        gs_ca_cert=gs_ca_cert,
        request_timeout_sec=request_timeout_sec,
        host=host,
        port=port,
        base_path=base_path,
        cors_allowed_origins=cors_allowed_origins,
        grafana=grafana,
    )


@lru_cache(maxsize=1)
def get_settings() -> DashboardSettings:
    """Load settings from process environment (cached)."""

    return _read_env(os.environ)
