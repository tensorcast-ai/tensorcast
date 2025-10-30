#  Copyright (c) 2025, TensorCast Team.

"""Pydantic response models and conversion helpers for the dashboard API."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable

from google.protobuf.timestamp_pb2 import Timestamp
from pydantic import BaseModel, Field

from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.global_store.v1 import global_store_pb2


def _timestamp_to_datetime(timestamp: Timestamp | None) -> datetime | None:
    if timestamp is None:
        return None
    try:
        if (
            getattr(timestamp, "seconds", 0) == 0
            and getattr(timestamp, "nanos", 0) == 0
        ):
            return None
        return timestamp.ToDatetime().astimezone(timezone.utc)
    except Exception:  # noqa: BLE001 - defensive: timestamp may be unset
        return None


def _memory_type_to_str(memory_type: int) -> str:
    mapping = {
        common_pb2.MemoryType.MEMORY_TYPE_RAM: "RAM",
        common_pb2.MemoryType.MEMORY_TYPE_GPU: "GPU",
        common_pb2.MemoryType.MEMORY_TYPE_DISK: "DISK",
    }
    return mapping.get(common_pb2.MemoryType(memory_type), "UNSPECIFIED")


def _connection_status_to_str(status: int) -> str:
    mapping = {
        global_store_pb2.ConnectionStatus.CONNECTION_STATUS_CONNECTED: "CONNECTED",
        global_store_pb2.ConnectionStatus.CONNECTION_STATUS_DISCONNECTED: "DISCONNECTED",
        global_store_pb2.ConnectionStatus.CONNECTION_STATUS_RECONNECTING: "RECONNECTING",
    }
    return mapping.get(global_store_pb2.ConnectionStatus(status), "UNSPECIFIED")


def _chunk_state_to_str(state: int) -> str:
    mapping = {
        global_store_pb2.ChunkState.CHUNK_STATE_LOCKED_TX: "LOCKED_TX",
        global_store_pb2.ChunkState.CHUNK_STATE_COPIED_GPU: "COPIED_GPU",
        global_store_pb2.ChunkState.CHUNK_STATE_COLD: "COLD",
        global_store_pb2.ChunkState.CHUNK_STATE_EVICTED: "EVICTED",
        global_store_pb2.ChunkState.CHUNK_STATE_HOT: "HOT",
    }
    return mapping.get(global_store_pb2.ChunkState(state), "UNSPECIFIED")


class HealthStatus(str, Enum):
    OK = "OK"
    ERROR = "ERROR"


class HealthResponse(BaseModel):
    status: HealthStatus

    @classmethod
    def from_proto(
        cls, response: global_store_pb2.HealthCheckResponse
    ) -> "HealthResponse":
        if response.status == global_store_pb2.Status.STATUS_OK:
            return cls(status=HealthStatus.OK)
        return cls(status=HealthStatus.ERROR)


class WorkerRow(BaseModel):
    worker_id: str
    node_id: str
    node_address: str
    grpc_port: int
    p2p_port: int
    mem_pool_total: int = Field(alias="mem_pool_total")
    mem_pool_available: int = Field(alias="mem_pool_available")
    accepting_new_requests: bool
    last_heartbeat_ts: datetime | None
    state_version: int
    status: str

    model_config = {"populate_by_name": True}

    @classmethod
    def from_proto(
        cls, worker: global_store_pb2.ListActiveWorkersResponse.WorkerInfo
    ) -> "WorkerRow":
        return cls(
            worker_id=worker.worker_id,
            node_id=worker.node_id,
            node_address=worker.node_address,
            grpc_port=worker.grpc_port,
            p2p_port=worker.p2p_port,
            mem_pool_total=worker.mem_pool_total_size,
            mem_pool_available=worker.mem_pool_available_size,
            accepting_new_requests=worker.accepting_new_requests,
            last_heartbeat_ts=_timestamp_to_datetime(worker.last_heartbeat_ts),
            state_version=worker.state_version,
            status=_connection_status_to_str(worker.status),
        )


class WorkersResponse(BaseModel):
    workers: list[WorkerRow]

    @classmethod
    def from_proto(
        cls, response: global_store_pb2.ListActiveWorkersResponse
    ) -> "WorkersResponse":
        return cls(
            workers=[WorkerRow.from_proto(worker) for worker in response.workers]
        )


class PageInfo(BaseModel):
    next_page_token: str | None = None
    total_size: int | None = None

    @classmethod
    def from_proto(cls, page_info: common_pb2.PageInfo | None) -> "PageInfo | None":
        if page_info is None:
            return None
        return cls(
            next_page_token=page_info.next_page_token or None,
            total_size=page_info.total_size
            if page_info.HasField("total_size")
            else None,
        )


class ReplicaEntry(BaseModel):
    artifact_id: str
    node_id: str
    node_address: str
    device_id: int | None
    memory_type: str
    bytes: int
    state: str | None = None
    created_ts: datetime | None = None

    @classmethod
    def from_proto(
        cls, record: global_store_pb2.ArtifactReplicaRecord
    ) -> "ReplicaEntry":
        memory_info = record.memory_info
        return cls(
            artifact_id=record.artifact_id,
            node_id=memory_info.node_id,
            node_address=memory_info.node_address,
            device_id=memory_info.device_id,
            memory_type=_memory_type_to_str(memory_info.memory_type),
            bytes=memory_info.memory_size,
            state=None,
            created_ts=_timestamp_to_datetime(memory_info.creation_ts),
        )


class ReplicasResponse(BaseModel):
    replicas: list[ReplicaEntry]
    page_info: PageInfo | None = None

    @classmethod
    def from_proto(
        cls, response: global_store_pb2.ListReplicasV2Response
    ) -> "ReplicasResponse":
        page_info = (
            PageInfo.from_proto(response.page_info)
            if response.HasField("page_info")
            else None
        )
        return cls(
            replicas=[ReplicaEntry.from_proto(record) for record in response.replicas],
            page_info=page_info,
        )


class ArtifactReplica(BaseModel):
    node_id: str
    node_address: str
    device_id: int | None
    memory_type: str
    bytes: int
    created_ts: datetime | None

    @classmethod
    def from_memory_info(cls, info: common_pb2.MemoryInfo) -> "ArtifactReplica":
        return cls(
            node_id=info.node_id,
            node_address=info.node_address,
            device_id=info.device_id,
            memory_type=_memory_type_to_str(info.memory_type),
            bytes=info.memory_size,
            created_ts=_timestamp_to_datetime(info.creation_ts),
        )


class LeafEntry(BaseModel):
    index: int
    digest_b64: str

    @classmethod
    def from_proto(cls, leaf: global_store_pb2.Leaf) -> "LeafEntry":
        digest = base64.b64encode(leaf.digest).decode("ascii") if leaf.digest else ""
        return cls(index=leaf.leaf_idx, digest_b64=digest)


class CoverageRange(BaseModel):
    offset: int
    length: int

    @classmethod
    def from_proto(cls, rng: global_store_pb2.Range) -> "CoverageRange":
        return cls(offset=rng.off, length=rng.len)


class PartialCoverageEntry(BaseModel):
    space_kind: str
    space_id: str
    missing: list[CoverageRange]

    @classmethod
    def from_proto(
        cls, detail: global_store_pb2.PartialCoverageDetail
    ) -> "PartialCoverageEntry":
        mapping = {
            global_store_pb2.ByteSpaceKind.BYTE_SPACE_KIND_CANONICAL: "CANONICAL",
            global_store_pb2.ByteSpaceKind.BYTE_SPACE_KIND_VARIANT: "VARIANT",
        }
        space_kind = mapping.get(detail.space_kind, "UNSPECIFIED")
        return cls(
            space_kind=space_kind,
            space_id=detail.space_id,
            missing=[CoverageRange.from_proto(rng) for rng in detail.missing_ranges],
        )


class ViewMetaEntry(BaseModel):
    view_spec_json: str
    view_size: int
    view_data_hash: str
    verified_at: datetime | None

    @classmethod
    def from_proto(cls, meta: global_store_pb2.ViewMeta) -> "ViewMetaEntry":
        return cls(
            view_spec_json=meta.view_spec_json,
            view_size=meta.view_size,
            view_data_hash=meta.view_data_hash,
            verified_at=_timestamp_to_datetime(meta.verified_at),
        )


class ArtifactDetailResponse(BaseModel):
    artifact_id: str
    replicas: list[ArtifactReplica] = Field(default_factory=list)
    leaves: list[LeafEntry] = Field(default_factory=list)
    partial_coverage: list[PartialCoverageEntry] = Field(default_factory=list)
    view_meta: ViewMetaEntry | None = None

    @classmethod
    def from_proto(
        cls, artifact_id: str, response: global_store_pb2.GetArtifactInfoByIdResponse
    ) -> "ArtifactDetailResponse":
        replicas = [
            ArtifactReplica.from_memory_info(info) for info in response.replicas
        ]
        leaves = [LeafEntry.from_proto(leaf) for leaf in response.leaves]
        coverage = [
            PartialCoverageEntry.from_proto(detail)
            for detail in response.partial_coverage
        ]
        view_meta = (
            ViewMetaEntry.from_proto(response.view_meta)
            if response.HasField("view_meta")
            else None
        )
        return cls(
            artifact_id=artifact_id,
            replicas=replicas,
            leaves=leaves,
            partial_coverage=coverage,
            view_meta=view_meta,
        )


class ChunkLocationEntry(BaseModel):
    index: int
    node_id: str
    node_address: str
    p2p_port: int
    state: str
    node_load_ratio: float
    device_uuid: str
    replica: int

    @classmethod
    def from_proto(
        cls, location: global_store_pb2.ChunkLocation
    ) -> "ChunkLocationEntry":
        return cls(
            index=location.chunk_idx,
            node_id=location.node_id,
            node_address=location.node_address,
            p2p_port=location.p2p_port,
            state=_chunk_state_to_str(location.state),
            node_load_ratio=location.node_load_ratio,
            device_uuid=location.device_uuid,
            replica=location.replica,
        )


class ChunkLocationsResponse(BaseModel):
    chunks: list[ChunkLocationEntry]

    @classmethod
    def from_proto(
        cls, response: global_store_pb2.QueryChunkLocationsResponse
    ) -> "ChunkLocationsResponse":
        return cls(
            chunks=[ChunkLocationEntry.from_proto(loc) for loc in response.locations]
        )


class ErrorDetail(BaseModel):
    code: str
    http_status: int
    message: str
    details: dict[str, object] = Field(default_factory=dict)
    trace_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


def iter_panel_ids(panel_ids: Iterable[str]) -> list[str]:
    return list(panel_ids)
