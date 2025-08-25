#  Copyright (c) 2025, StepCast Team.

#
# ---------------------------------------------------------------------------- #

"""Simplified configuration management for **StoreDaemon**.

Goals of this redesign:
1. 1-to-1 mapping between YAML structure and Python objects – no implicit
   flattening/merging.
2. Reject unknown keys early to avoid silent typos (`extra = "forbid"`).
3. Support human-friendly byte units via `pydantic.ByteSize`.
4. Provide convenient `load()` / `dump()` helpers for YAML round-tripping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import (
    BaseModel,
    ByteSize,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# ---------------------------------------------------------------------------
# Section: artifact storage
# ---------------------------------------------------------------------------


class ServerConfig(BaseModel):
    """gRPC server and memory settings."""

    host: str = Field(default="0.0.0.0", description="Bind address for the gRPC server")
    port: int = Field(
        default=50052, ge=1, le=65535, description="gRPC port exposed to clients"
    )
    storage_path: Path = Field(
        default=Path("/tmp/models"),
        description="Directory used to store artifact files",
    )
    num_threads: int = Field(default=10, ge=1, description="Worker thread pool size")
    chunk_size: ByteSize = Field(
        default=ByteSize(128 * 1024 * 1024),
        description="Maximum chunk size sent over the wire",
    )
    mem_pool_size: ByteSize = Field(
        default=ByteSize(8 * 1024 * 1024 * 1024),
        description="Pinned memory pool size for staging data",
    )
    enable_p2p_engine: bool = Field(
        default=True, description="Enable the high-performance communication engine"
    )
    enable_p2p_access: bool = Field(
        default=True,
        description="Require explicit registration before a artifact can be loaded",
    )
    enable_rdma: bool = Field(
        default=False, description="Enable RDMA transport if supported"
    )
    pinned_memory_timeout_ms: int = Field(
        default=30_000,
        ge=0,
        description="Timeout when allocating from the pinned pool (ms)",
    )

    model_config = ConfigDict(extra="forbid")

    # -------------------------------------
    # Validators
    # -------------------------------------

    @field_validator("storage_path", mode="after")
    @classmethod
    def _expand_path(cls, v: Path) -> Path:  # noqa: D401  (imperative mood)
        """Expand user symbols ("~") and resolve to an absolute path."""
        return v.expanduser().resolve()


class NetworkConfig(BaseModel):
    """Ports for data, metrics and health-checks."""

    p2p_port: int = Field(
        default=9090, ge=1, le=65535, description="P2P data-plane port"
    )
    metrics_port: int = Field(
        default=9091, ge=1, le=65535, description="Prometheus scrape port"
    )
    health_check_port: int | None = Field(
        default=8080, ge=1, le=65535, description="HTTP /health and /ready probe port"
    )

    model_config = ConfigDict(extra="forbid")


class LifecycleConfig(BaseModel):
    """Artifact eviction & health-check parameters."""

    gpu_memory_limit_fraction: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="GPU memory utilisation threshold that triggers eviction",
    )
    global_cache_fraction: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Target fraction of GPU memory reserved as global cache",
    )
    proc_check_interval_s: float = Field(
        default=5.0, ge=0.1, description="Process health-check interval (seconds)"
    )
    eviction_check_interval_s: float = Field(
        default=30.0,
        ge=0.1,
        description="Eviction policy evaluation interval (seconds)",
    )

    model_config = ConfigDict(extra="forbid")

    # -------------------------------------
    # Cross-field validation
    # -------------------------------------

    @model_validator(mode="after")
    def _validate_fractions(self) -> "LifecycleConfig":  # noqa: D401
        # Note: global_cache_fraction and gpu_memory_limit_fraction are independent
        # - global_cache_fraction: fraction of GPU memory reserved as global cache
        # - gpu_memory_limit_fraction: memory utilization threshold that triggers eviction
        # They do not need to sum to 1.0 as they represent different aspects of memory management
        return self


class ShutdownConfig(BaseModel):
    """Graceful termination settings."""

    grace_period_ms: int = Field(
        default=30_000, ge=0, description="Time allowed for graceful shutdown (ms)"
    )

    model_config = ConfigDict(extra="forbid")


class HighAvailabilityConfig(BaseModel):
    """Optional high-availability features."""

    enabled: bool = Field(default=True, description="Enable high-availability mode")
    heartbeat_interval_ms: int = Field(
        default=5_000, ge=100, description="Heartbeat interval when registered (ms)"
    )
    registration_retry_delay_ms: int = Field(
        default=1_000,
        ge=100,
        description="Delay between retries when Global Store registration fails (ms)",
    )
    max_retries: int = Field(
        default=10, ge=-1, description="Maximum registration retries (-1 for infinite)"
    )

    # Connection retry configuration (backwards compatibility)
    initial_delay_ms: int = Field(
        default=1_000, ge=100, description="Initial retry delay (ms)"
    )
    max_delay_ms: int = Field(
        default=60_000, ge=1000, description="Maximum retry delay (ms)"
    )
    exponential_base: float = Field(
        default=2.0, ge=1.0, description="Exponential backoff multiplier"
    )

    # State sync configuration (backwards compatibility)
    periodic_sync_interval_ms: int = Field(
        default=600_000, ge=10_000, description="Periodic state sync interval (ms)"
    )

    # Thread monitoring configuration (backwards compatibility)
    thread_monitor_interval_ms: int = Field(
        default=10_000, ge=1000, description="Thread monitoring interval (ms)"
    )
    auto_restart_threads: bool = Field(
        default=True, description="Auto-restart failed threads"
    )

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class StoreDaemonConfig(BaseModel):
    """Root configuration artifact – mirrors the YAML structure 1-for-1."""

    server: ServerConfig = Field(default_factory=lambda: ServerConfig())
    network: NetworkConfig = Field(default_factory=lambda: NetworkConfig())
    lifecycle: LifecycleConfig = Field(default_factory=lambda: LifecycleConfig())
    shutdown: ShutdownConfig = Field(default_factory=lambda: ShutdownConfig())
    high_availability: HighAvailabilityConfig = Field(
        default_factory=lambda: HighAvailabilityConfig()
    )

    # Optional global-store settings (kept at root level for clarity)
    global_store_address: str | None = Field(
        default=None, description="`host:port` of the Global Store service, if used"
    )

    model_config = ConfigDict(extra="forbid")

    # ------------------------------------------------------------------
    # YAML helpers
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "StoreDaemonConfig":
        """Load configuration from a YAML file."""

        cfg_path = Path(path).expanduser().resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")

        with open(cfg_path, "r", encoding="utf-8") as fp:
            raw: Any = yaml.safe_load(fp) or {}

        return cls.model_validate(raw)

    def dump(self, path: str | Path) -> None:
        """Write configuration back to YAML (pretty-printed)."""

        dump_path = Path(path).expanduser().resolve()
        with open(dump_path, "w", encoding="utf-8") as fp:
            yaml.safe_dump(
                self.model_dump(mode="python", by_alias=False),
                fp,
                sort_keys=False,
                default_flow_style=False,
            )
