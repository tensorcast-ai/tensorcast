#  Copyright (c) 2025, StepCast Team.

"""Pydantic models for Web UI API responses."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """Memory type enumeration."""

    GPU = "GPU"
    RAM = "RAM"
    DISK = "DISK"


class ConnectionStatus(str, Enum):
    """Connection status enumeration."""

    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    DISCONNECTED = "DISCONNECTED"


class WorkerStatus(str, Enum):
    """Worker health status."""

    HEALTHY = "healthy"  # < 5s
    WARNING = "warning"  # 5-15s
    CRITICAL = "critical"  # > 15s
    DEAD = "dead"  # > timeout


# API Response Models
class WorkerOut(BaseModel):
    """Worker information output model."""

    worker_id: str
    node_id: str
    node_address: str
    grpc_port: int
    p2p_port: int
    mem_pool_total_size: int
    mem_pool_available_size: int
    accepting_new_requests: bool
    registered_at: datetime
    last_heartbeat: datetime
    status: WorkerStatus = Field(description="Health status based on heartbeat")
    replica_count: int = 0


class ReplicaOut(BaseModel):
    """Model replica output model."""

    replica_id: str
    model_id: str
    node_id: str
    node_address: str
    node_port: int
    memory_size: int
    memory_type: MemoryType
    device_id: int
    max_concurrency: int
    is_available: bool
    worker_id: str | None
    created_at: datetime
    updated_at: datetime
    current_requests: int = 0
    worker_accepting: bool = True


class ModelSummary(BaseModel):
    """Model summary information."""

    model_id: str
    total_replicas: int
    available_replicas: int
    gpu_replicas: int
    ram_replicas: int
    disk_replicas: int
    total_memory_size: int
    avg_load_ratio: float


class NodeSummary(BaseModel):
    """Node aggregated information."""

    node_id: str
    total_replicas: int
    total_memory: int
    gpu_memory: int
    ram_memory: int
    disk_memory: int
    active_workers: int


class TransportOut(BaseModel):
    """Transport history output."""

    transport_id: str
    replica_id: str
    model_id: str
    source_node_id: str
    source_address: str
    source_port: int
    created_at: datetime
    completed_at: datetime | None = None
    status: str = "in_progress"
    wait_duration_seconds: float | None = None


class GlobalMetrics(BaseModel):
    """Global store metrics summary."""

    total_workers: int
    active_workers: int
    total_replicas: int
    available_replicas: int
    total_models: int
    active_transports: int
    total_memory_bytes: int
    available_memory_bytes: int


# API Request Models
class ListWorkersRequest(BaseModel):
    """List workers request parameters."""

    include_unavailable: bool = False
    page: int = Field(1, ge=1)
    page_size: int = Field(50, ge=1, le=1000)


class ListReplicasRequest(BaseModel):
    """List replicas request parameters."""

    model_id: str | None = None
    node_id: str | None = None
    memory_type: MemoryType | None = None
    worker_id: str | None = None
    page: int = Field(1, ge=1)
    page_size: int = Field(100, ge=1, le=1000)


class ListTransportsRequest(BaseModel):
    """List transports request parameters."""

    status: str | None = None
    model_id: str | None = None
    page: int = Field(1, ge=1)
    page_size: int = Field(50, ge=1, le=1000)


# WebSocket Messages
class WebSocketMessage(BaseModel):
    """Base WebSocket message."""

    topic: str
    payload: dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WebSocketSubscription(BaseModel):
    """WebSocket subscription request."""

    action: str = "subscribe"
    topics: list[str] = ["heartbeat", "replica_update", "transport"]


# API Response Envelope
@dataclass
class PaginationMeta:
    """Pagination metadata."""

    page: int
    page_size: int
    total_count: int
    total_pages: int


class ApiResponse(BaseModel):
    """Standard API response envelope."""

    data: Any
    meta: dict[str, Any] | None = None
