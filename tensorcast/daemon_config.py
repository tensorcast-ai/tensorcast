#  Copyright (c) 2025, TensorCast Team.

"""
Typed configuration models for the StoreDaemon (used to launch the C++ daemon).

This module is a lightweight, import-friendly home for the configuration types
that were previously under `tensorcast.store_daemon`. It intentionally contains
only data models and YAML helpers — no Python daemon logic — so that the Python
CLI, daemon manager, and clients can construct and validate configuration
without depending on the legacy Python StoreDaemon implementation.
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


class StagerSettings(BaseModel):
    stage_cpu_for_rdma: bool = Field(default=True)
    stage_chunk_mb_cpu: int = Field(default=4, ge=1)
    stage_chunk_mb_gpu: int = Field(default=16, ge=1)
    buffers_per_flow: int = Field(default=4, ge=1)

    model_config = ConfigDict(extra="forbid")


class RdmaSettings(BaseModel):
    outstanding_wr: int = Field(default=64, ge=1)
    ack_ttl_ms: int = Field(default=30_000, ge=0)

    model_config = ConfigDict(extra="forbid")


class PoolSettings(BaseModel):
    preregister_mr: bool = Field(default=True)
    pool_size_bytes: int = Field(default=8 * 1024 * 1024 * 1024, ge=0)
    chunk_bytes: int = Field(default=64 * 1024 * 1024, ge=4096)

    model_config = ConfigDict(extra="forbid")


class TransportSettings(BaseModel):
    tcp_conn_count: int = Field(default=8, ge=1)

    model_config = ConfigDict(extra="forbid")


class AffinitySettings(BaseModel):
    enable: bool = Field(default=False)

    model_config = ConfigDict(extra="forbid")


class SimpleNumaNode(BaseModel):
    id: int = Field(default=0)
    nics: list[str] = Field(default_factory=list)
    gpus: list[int] = Field(default_factory=list)
    is_default: bool = Field(default=False)

    model_config = ConfigDict(extra="forbid")


class SimpleNumaSettings(BaseModel):
    enable: bool = Field(default=False)
    nodes: list[SimpleNumaNode] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class CommunicatorSettings(BaseModel):
    enable_rdma: bool = Field(default=False)
    stager: StagerSettings = Field(default_factory=lambda: StagerSettings())
    rdma: RdmaSettings = Field(default_factory=lambda: RdmaSettings())
    pool: PoolSettings = Field(default_factory=lambda: PoolSettings())
    transport: TransportSettings = Field(default_factory=lambda: TransportSettings())
    affinity: AffinitySettings = Field(default_factory=lambda: AffinitySettings())
    simple_numa: SimpleNumaSettings = Field(
        default_factory=lambda: SimpleNumaSettings()
    )

    model_config = ConfigDict(extra="forbid")


class ServerConfig(BaseModel):
    """gRPC server and memory settings."""

    host: str = Field(default="0.0.0.0", description="Bind address for the gRPC server")
    port: int = Field(default=50052, ge=1, le=65535, description="gRPC port")
    storage_path: Path = Field(
        default=Path("/tmp/models"), description="Artifacts directory"
    )
    num_threads: int = Field(default=10, ge=1, description="Worker thread pool size")
    chunk_size: ByteSize = Field(
        default=ByteSize(128 * 1024 * 1024), description="Max chunk size"
    )
    mem_pool_size: ByteSize = Field(
        default=ByteSize(8 * 1024 * 1024 * 1024), description="Pinned memory pool size"
    )
    enable_p2p_engine: bool = Field(
        default=True, description="Enable P2P communication engine"
    )
    enable_p2p_access: bool = Field(
        default=True, description="Require explicit registration for load"
    )
    enable_rdma: bool = Field(
        default=False, description="Enable RDMA transport if supported"
    )
    pinned_memory_timeout_ms: int = Field(
        default=30_000, ge=0, description="Pinned pool allocation timeout (ms)"
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("storage_path", mode="after")
    @classmethod
    def _expand_path(cls, v: Path) -> Path:
        return v.expanduser().resolve()


class NetworkConfig(BaseModel):
    """Ports for data, metrics and health-checks."""

    p2p_port: int = Field(default=9090, ge=1, le=65535)
    metrics_port: int = Field(default=9091, ge=1, le=65535)
    health_check_port: int | None = Field(default=8080, ge=1, le=65535)

    model_config = ConfigDict(extra="forbid")


class LifecycleConfig(BaseModel):
    gpu_memory_limit_fraction: float = Field(default=0.75, ge=0.0, le=1.0)
    global_cache_fraction: float = Field(default=0.20, ge=0.0, le=1.0)
    proc_check_interval_s: float = Field(default=5.0, ge=0.1)
    eviction_check_interval_s: float = Field(default=30.0, ge=0.1)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_fractions(self) -> "LifecycleConfig":
        # These represent different aspects and do not need to sum to 1.0
        return self


class ShutdownConfig(BaseModel):
    grace_period_ms: int = Field(default=30_000, ge=0)

    model_config = ConfigDict(extra="forbid")


class HighAvailabilityConfig(BaseModel):
    enabled: bool = Field(default=True)
    heartbeat_interval_ms: int = Field(default=5_000, ge=100)
    registration_retry_delay_ms: int = Field(default=1_000, ge=100)
    max_retries: int = Field(default=10, ge=-1)

    # Back-compat knobs
    initial_delay_ms: int = Field(default=1_000, ge=100)
    max_delay_ms: int = Field(default=60_000, ge=1000)
    exponential_base: float = Field(default=2.0, ge=1.0)
    periodic_sync_interval_ms: int = Field(default=600_000, ge=10_000)
    thread_monitor_interval_ms: int = Field(default=10_000, ge=1000)
    auto_restart_threads: bool = Field(default=True)

    model_config = ConfigDict(extra="forbid")


class StoreDaemonConfig(BaseModel):
    """Root configuration artifact – mirrors the YAML structure 1-for-1."""

    server: ServerConfig = Field(default_factory=lambda: ServerConfig())
    network: NetworkConfig = Field(default_factory=lambda: NetworkConfig())
    lifecycle: LifecycleConfig = Field(default_factory=lambda: LifecycleConfig())
    shutdown: ShutdownConfig = Field(default_factory=lambda: ShutdownConfig())
    high_availability: HighAvailabilityConfig = Field(
        default_factory=lambda: HighAvailabilityConfig()
    )

    # Optional communicator configuration used by the C++ engine
    communicator: CommunicatorSettings | None = Field(default=None)

    # Optional global-store settings (kept at root level for clarity)
    global_store_address: str | None = Field(
        default=None, description="host:port of Global Store service"
    )

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def load(cls, path: str | Path) -> "StoreDaemonConfig":
        cfg_path = Path(path).expanduser().resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as fp:
            raw: Any = yaml.safe_load(fp) or {}
        return cls.model_validate(raw)

    def dump(self, path: str | Path) -> None:
        dump_path = Path(path).expanduser().resolve()
        with open(dump_path, "w", encoding="utf-8") as fp:
            yaml.safe_dump(
                self.model_dump(mode="python", by_alias=False),
                fp,
                sort_keys=False,
                default_flow_style=False,
            )
