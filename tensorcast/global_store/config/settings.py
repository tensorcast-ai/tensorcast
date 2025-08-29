#  Copyright (c) 2025, TensorCast Team.

"""Global Store configuration settings."""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


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

    @classmethod
    def from_env(cls) -> "GlobalStoreConfig":
        """Create config from environment variables."""

        # Helper function to safely parse integer values
        def safe_int(value: str, default: int) -> int:
            try:
                return int(value.strip())
            except (ValueError, AttributeError):
                return default

        # Helper function to safely parse positive integers
        def safe_positive_int(value: str, default: int) -> int:
            try:
                parsed = int(value.strip())
                return max(0, parsed)  # Clamp to non-negative
            except (ValueError, AttributeError):
                return default

        # Helper function to safely parse path values
        def safe_path(value: str) -> Optional[Path]:
            if not value or not value.strip():
                return None
            return Path(value.strip())

        # Get values from environment with fallbacks
        db_file = safe_path(os.getenv("GLOBAL_STORE_DB_PATH", ""))

        # Keep ``db_file`` as ``None`` when the user has not supplied an explicit
        # value.  The daemon will take care of choosing a suitable on-disk
        # location when persistence is required.  This behaviour also makes it
        # easier to unit-test the configuration layer because it avoids
        # accidental writes to the host filesystem.

        heartbeat_timeout_ms = safe_positive_int(
            os.getenv("GLOBAL_STORE_HEARTBEAT_TIMEOUT_MS", ""), 30000
        )
        cleanup_interval_ms = safe_positive_int(
            os.getenv("GLOBAL_STORE_CLEANUP_INTERVAL_MS", ""), 60000
        )
        default_heartbeat_interval_ms = safe_positive_int(
            os.getenv("GLOBAL_STORE_HEARTBEAT_INTERVAL_MS", ""), 5000
        )

        port = safe_int(os.getenv("GLOBAL_STORE_PORT", ""), 50051)
        max_workers = safe_int(os.getenv("GLOBAL_STORE_MAX_WORKERS", ""), 10)

        optimize_interval_ms = safe_positive_int(
            os.getenv("GLOBAL_STORE_OPTIMIZE_INTERVAL_MS", ""), 3_600_000
        )
        metrics_port = safe_int(os.getenv("GLOBAL_STORE_METRICS_PORT", ""), 8000)

        # Web UI settings
        ui_port = safe_int(os.getenv("GLOBAL_STORE_UI_PORT", ""), 9000)
        ui_host = os.getenv("GLOBAL_STORE_UI_HOST", "0.0.0.0")
        ui_static_dir = os.getenv("GLOBAL_STORE_UI_STATIC_DIR")
        ui_cors_origins = os.getenv("GLOBAL_STORE_UI_CORS_ORIGINS")
        ui_enabled = os.getenv("GLOBAL_STORE_UI_ENABLED", "true").lower() in (
            "true",
            "1",
            "yes",
        )
        ui_log_file = safe_path(
            os.getenv("GLOBAL_STORE_UI_LOG_FILE", "/tmp/global-store-webui.log")
        ) or Path("/tmp/global-store-webui.log")

        return cls(
            db_file=db_file,
            heartbeat_timeout_ms=heartbeat_timeout_ms,
            cleanup_interval_ms=cleanup_interval_ms,
            default_heartbeat_interval_ms=default_heartbeat_interval_ms,
            port=port,
            max_workers=max_workers,
            transport_wait_retry_interval_ms=200,  # Use default
            optimize_interval_ms=optimize_interval_ms,
            metrics_port=metrics_port,
            ui_port=ui_port,
            ui_host=ui_host,
            ui_static_dir=ui_static_dir,
            ui_enabled=ui_enabled,
            ui_cors_origins=ui_cors_origins,
            ui_log_file=ui_log_file,
        )


# Singleton config instance
_config: Optional[GlobalStoreConfig] = None


def get_config() -> GlobalStoreConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = GlobalStoreConfig.from_env()
    return _config
