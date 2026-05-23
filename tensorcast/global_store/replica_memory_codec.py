#  Copyright (c) 2025-2026, TensorCast Team.

"""Shared converters between Global Store Replica models and MemoryInfo proto."""

from __future__ import annotations

from datetime import datetime
from typing import Callable
from uuid import UUID

from google.protobuf import timestamp_pb2

from tensorcast.global_store.exceptions import ValidationError
from tensorcast.global_store.models import (
    ByteSpaceKind,
    ByteSpaceRef,
    ExportState,
    MemoryType,
    Replica,
)
from tensorcast.proto.common.v1 import common_pb2

TransportMetadata = tuple[bool, ExportState, int, list[str], list[int], str | None]


def export_state_from_proto(
    state: common_pb2.ReplicaTransportMetadata.ExportState,
) -> ExportState:
    if state == common_pb2.ReplicaTransportMetadata.EXPORT_STATE_EXPORTABLE:
        return ExportState.EXPORTABLE
    if state == common_pb2.ReplicaTransportMetadata.EXPORT_STATE_DRAINING:
        return ExportState.DRAINING
    return ExportState.PRESENCE_ONLY


def export_state_to_proto(
    state: ExportState,
) -> common_pb2.ReplicaTransportMetadata.ExportState:
    if state is ExportState.EXPORTABLE:
        return common_pb2.ReplicaTransportMetadata.EXPORT_STATE_EXPORTABLE
    if state is ExportState.DRAINING:
        return common_pb2.ReplicaTransportMetadata.EXPORT_STATE_DRAINING
    return common_pb2.ReplicaTransportMetadata.EXPORT_STATE_PRESENCE_ONLY


def parse_transport_metadata(mem_info: common_pb2.MemoryInfo) -> TransportMetadata:
    transport_authoritative = mem_info.HasField("transport")
    export_state = ExportState.PRESENCE_ONLY
    export_generation = 0
    remote_keys: list[str] = []
    buffer_sizes: list[int] = []
    verification_json: str | None = None

    if not transport_authoritative:
        return (
            transport_authoritative,
            export_state,
            export_generation,
            remote_keys,
            buffer_sizes,
            verification_json,
        )

    transport = mem_info.transport
    export_state = export_state_from_proto(transport.export_state)
    export_generation = int(transport.export_generation or 0)
    remote_keys = list(transport.remote_memory_keys)
    buffer_sizes = [int(size) for size in transport.buffer_sizes]
    verification_json = (
        transport.verification_json if transport.verification_json else None
    )

    if export_state is ExportState.PRESENCE_ONLY:
        return (
            transport_authoritative,
            export_state,
            export_generation,
            [],
            [],
            None,
        )

    keys_required = export_state is ExportState.EXPORTABLE
    if keys_required and not remote_keys:
        export_state = ExportState.PRESENCE_ONLY
        remote_keys = []
        buffer_sizes = []
        verification_json = None

    if remote_keys or buffer_sizes:
        keys_valid = True
        if len(remote_keys) != len(buffer_sizes):
            keys_valid = False
        else:
            total = 0
            for size in buffer_sizes:
                if size <= 0:
                    keys_valid = False
                    break
                total += int(size)
            if keys_valid and mem_info.memory_size > 0:
                keys_valid = total == mem_info.memory_size
        if not keys_valid:
            remote_keys = []
            buffer_sizes = []
            export_state = ExportState.PRESENCE_ONLY
            verification_json = None

    return (
        transport_authoritative,
        export_state,
        export_generation,
        remote_keys,
        buffer_sizes,
        verification_json,
    )


def memory_type_to_proto(memory_type: MemoryType) -> common_pb2.MemoryType:
    if memory_type == MemoryType.GPU:
        return common_pb2.MemoryType.MEMORY_TYPE_GPU
    if memory_type == MemoryType.RAM:
        return common_pb2.MemoryType.MEMORY_TYPE_RAM
    return common_pb2.MemoryType.MEMORY_TYPE_DISK


def memory_type_from_proto(memory_type: common_pb2.MemoryType) -> MemoryType:
    if memory_type == common_pb2.MemoryType.MEMORY_TYPE_GPU:
        return MemoryType.GPU
    if memory_type == common_pb2.MemoryType.MEMORY_TYPE_RAM:
        return MemoryType.RAM
    return MemoryType.DISK


def byte_space_from_memory_info(
    mem_info: common_pb2.MemoryInfo,
    *,
    require_view_id: bool,
) -> ByteSpaceRef:
    if not mem_info.HasField("byte_space"):
        return ByteSpaceRef.canonical()

    if mem_info.byte_space.kind == common_pb2.BYTE_SPACE_KIND_VIEW:
        view_id = mem_info.byte_space.id.strip()
        if not view_id:
            if require_view_id:
                raise ValidationError("byte_space VIEW requires id")
            return ByteSpaceRef.canonical()
        return ByteSpaceRef.view(view_id)

    if mem_info.byte_space.kind in (
        common_pb2.BYTE_SPACE_KIND_CANONICAL,
        common_pb2.BYTE_SPACE_KIND_UNSPECIFIED,
    ):
        return ByteSpaceRef.canonical()

    return ByteSpaceRef.canonical()


def replica_to_memory_info(
    *,
    replica: Replica,
    datetime_to_timestamp: Callable[[datetime | None], timestamp_pb2.Timestamp | None]
    | None = None,
) -> common_pb2.MemoryInfo:
    memory_info = common_pb2.MemoryInfo(
        node_id=replica.node_id,
        node_address=replica.node_address,
        node_port=replica.node_port,
        memory_size=replica.memory_size,
        memory_type=memory_type_to_proto(replica.memory_type),
        device_id=replica.device_id,
        replica_id=str(replica.replica_id),
    )
    transport = memory_info.transport
    transport.export_state = export_state_to_proto(replica.export_state)
    transport.export_generation = int(replica.export_generation or 0)
    transport.remote_memory_keys.extend(replica.remote_memory_keys)
    transport.buffer_sizes.extend([int(size) for size in replica.buffer_sizes])
    if replica.verification_json:
        transport.verification_json = replica.verification_json
    if replica.byte_space.kind == ByteSpaceKind.VIEW:
        memory_info.byte_space.CopyFrom(
            common_pb2.ByteSpaceRef(
                kind=common_pb2.BYTE_SPACE_KIND_VIEW,
                id=replica.byte_space.id or "",
            )
        )
    else:
        memory_info.byte_space.CopyFrom(
            common_pb2.ByteSpaceRef(kind=common_pb2.BYTE_SPACE_KIND_CANONICAL)
        )

    if datetime_to_timestamp is not None:
        creation_proto = datetime_to_timestamp(replica.created_at)
        if creation_proto is not None:
            memory_info.creation_ts.CopyFrom(creation_proto)
        expires_proto = datetime_to_timestamp(replica.expires_at)
        if expires_proto is not None:
            memory_info.expires_at.CopyFrom(expires_proto)

    return memory_info


def memory_info_to_replica(
    *,
    mem_info: common_pb2.MemoryInfo,
    artifact_id: str,
    max_concurrency: int,
    worker_id: str,
    require_view_id: bool,
    replica_id: UUID | None = None,
    current_requests: int = 0,
    is_available: bool = True,
) -> Replica:
    (
        _transport_authoritative,
        export_state,
        export_generation,
        remote_keys,
        buffer_sizes,
        verification_json,
    ) = parse_transport_metadata(mem_info)

    if replica_id is not None:
        return Replica(
            replica_id=replica_id,
            artifact_id=artifact_id,
            byte_space=byte_space_from_memory_info(
                mem_info=mem_info,
                require_view_id=require_view_id,
            ),
            node_id=mem_info.node_id,
            node_address=mem_info.node_address,
            node_port=mem_info.node_port,
            memory_size=mem_info.memory_size,
            memory_type=memory_type_from_proto(mem_info.memory_type),
            device_id=mem_info.device_id,
            max_concurrency=max_concurrency,
            current_requests=current_requests,
            is_available=is_available,
            remote_memory_keys=remote_keys,
            buffer_sizes=buffer_sizes,
            export_state=export_state,
            export_generation=export_generation,
            worker_id=worker_id,
            verification_json=verification_json,
        )
    return Replica(
        artifact_id=artifact_id,
        byte_space=byte_space_from_memory_info(
            mem_info=mem_info,
            require_view_id=require_view_id,
        ),
        node_id=mem_info.node_id,
        node_address=mem_info.node_address,
        node_port=mem_info.node_port,
        memory_size=mem_info.memory_size,
        memory_type=memory_type_from_proto(mem_info.memory_type),
        device_id=mem_info.device_id,
        max_concurrency=max_concurrency,
        current_requests=current_requests,
        is_available=is_available,
        remote_memory_keys=remote_keys,
        buffer_sizes=buffer_sizes,
        export_state=export_state,
        export_generation=export_generation,
        worker_id=worker_id,
        verification_json=verification_json,
    )
