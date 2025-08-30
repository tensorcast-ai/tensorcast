#  Copyright (c) 2025, TensorCast Team.

"""Typed communicator settings (Python mirror of proto/communicator_config.proto).

This module defines Pydantic models that mirror the communicator configuration
schema used by the C++ engine and the protobuf definition. It is not yet wired
into the StoreDaemon config, but can be imported and composed there.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


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
    simple_numa: SimpleNumaSettings = Field(default_factory=lambda: SimpleNumaSettings())

    model_config = ConfigDict(extra="forbid")
