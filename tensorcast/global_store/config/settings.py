#  Copyright (c) 2025, TensorCast Team.

"""Global Store configuration settings (file-based only)."""

import json
from pathlib import Path
from typing import Any, Optional

import yaml
from google.protobuf import json_format as _pb_json
from pydantic import BaseModel

from tensorcast.proto.config.v1 import (
    global_store_config_pb2 as gsc_pb2,
)

# GlobalStoreConfig now leverages Pydantic for validation / immutability


class GlobalStoreConfig(BaseModel):
    """Configuration for Global Store service."""

    # Database settings
    # Optional path to a persistent on-disk DuckDB database. When ``None`` a
    # purely in-memory (ephemeral) database will be used. The CLI and daemon
    # will create a temporary on-disk database if persistence is required, so
    # using ``None`` is the safest default for unit-tests which should not
    # touch the real filesystem.
    db_file: Optional[Path] = None

    # Worker management settings
    heartbeat_timeout_ms: int = 30000  # 30 seconds
    cleanup_interval_ms: int = 60000  # 1 minute
    default_heartbeat_interval_ms: int = 5000  # 5 seconds

    # Server settings
    port: int = 50051
    max_workers: int = 10

    # Performance settings
    transport_wait_retry_interval_ms: int = 200

    # Maintenance settings
    optimize_interval_ms: int = 3_600_000  # 1 hour

    # Metrics settings
    metrics_port: int = 8000

    # Web UI settings
    ui_port: int = 9000
    ui_host: str = "0.0.0.0"
    ui_static_dir: Optional[str] = None
    ui_enabled: bool = True
    ui_cors_origins: Optional[str] = (
        None  # Comma-separated list of allowed CORS origins for the Web UI
    )
    # Path for Web UI process log file
    ui_log_file: Path = Path("/tmp/global-store-webui.log")

    class Config:
        # Match previous @dataclass(frozen=True) behaviour (immutability)
        frozen = True

    # Environment-based loader removed in final scheme

    @classmethod
    def from_file(cls, path: str) -> "GlobalStoreConfig":
        """Create config from a YAML or JSON file validated against Proto schema.

        This loader parses YAML/JSON into tensorcast.config.v1.GlobalStoreConfig
        (strict unknown-key rejection), then maps the fields into the
        Pydantic model used by the Python service.
        """
        if path.endswith(".yaml") or path.endswith(".yml"):
            with open(path, "r", encoding="utf-8") as f:
                data: Any = yaml.safe_load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

        # Parse into proto (strict)
        pb = gsc_pb2.GlobalStoreConfig()
        _pb_json.ParseDict(data, pb, ignore_unknown_fields=False)

        # Map to Pydantic
        # Database
        db_file = Path(pb.database.db_file) if pb.database.db_file else None
        # Server
        port = int(pb.server.listen.port) if pb.server.HasField("listen") else 50051
        # Preserve default when field is absent; accept explicit 0 when provided
        server_section = data.get("server", {}) if isinstance(data, dict) else {}
        has_max_workers = isinstance(server_section, dict) and (
            "max_workers" in server_section
        )
        max_workers = int(pb.server.max_workers) if has_max_workers else 10

        # Worker policy durations are in seconds+nanos
        def _dur_ms(dur) -> int:
            # google.protobuf.Duration parsed from JSON strings "Xs" yields
            # (seconds, nanos). Convert to milliseconds.
            return int(dur.seconds * 1000 + dur.nanos // 1_000_000)

        heartbeat_timeout_ms = (
            _dur_ms(pb.worker_policy.heartbeat_timeout)
            if pb.worker_policy.HasField("heartbeat_timeout")
            else 30000
        )
        cleanup_interval_ms = (
            _dur_ms(pb.worker_policy.cleanup_interval)
            if pb.worker_policy.HasField("cleanup_interval")
            else 60000
        )
        default_hb_ms = (
            _dur_ms(pb.worker_policy.default_heartbeat_interval)
            if pb.worker_policy.HasField("default_heartbeat_interval")
            else 5000
        )
        # Clamp negatives to 0
        heartbeat_timeout_ms = max(0, heartbeat_timeout_ms)
        cleanup_interval_ms = max(0, cleanup_interval_ms)
        default_hb_ms = max(0, default_hb_ms)

        # Web UI
        ui_enabled = pb.web_ui.enabled
        ui_host = pb.web_ui.host or "0.0.0.0"
        ui_port = int(pb.web_ui.port) if pb.web_ui.port > 0 else 9000
        # Convert CORS list to comma-separated for existing code
        ui_cors_origins = (
            ",".join(pb.web_ui.cors_allowed_origins)
            if pb.web_ui.cors_allowed_origins
            else None
        )

        return cls(
            db_file=db_file,
            heartbeat_timeout_ms=heartbeat_timeout_ms,
            cleanup_interval_ms=cleanup_interval_ms,
            default_heartbeat_interval_ms=default_hb_ms,
            port=port,
            max_workers=max_workers,
            transport_wait_retry_interval_ms=200,
            optimize_interval_ms=3_600_000,
            # Metrics port: keep default unless overridden elsewhere
            metrics_port=8000,
            ui_port=ui_port,
            ui_host=ui_host,
            ui_static_dir=None,
            ui_enabled=ui_enabled,
            ui_cors_origins=ui_cors_origins,
            ui_log_file=Path("/tmp/global-store-webui.log"),
        )

    @staticmethod
    def load_proto_from_file(path: str) -> gsc_pb2.GlobalStoreConfig:
        """Load and return the GlobalStoreConfig proto from YAML/JSON (strict)."""
        if path.endswith(".yaml") or path.endswith(".yml"):
            with open(path, "r", encoding="utf-8") as f:
                data: Any = yaml.safe_load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        pb = gsc_pb2.GlobalStoreConfig()
        _pb_json.ParseDict(data, pb, ignore_unknown_fields=False)
        return pb


# Singleton config instance
_config: Optional[GlobalStoreConfig] = None


def get_config() -> GlobalStoreConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        raise RuntimeError(
            "GlobalStoreConfig not initialized. Call set_config() first."
        )
    return _config


def set_config(cfg: GlobalStoreConfig) -> None:
    global _config
    _config = cfg
