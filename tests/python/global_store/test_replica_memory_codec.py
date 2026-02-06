#  Copyright (c) 2025-2026, TensorCast Team.

"""Unit tests for shared Replica/MemoryInfo conversion helpers."""

from datetime import datetime, timezone

import pytest
from google.protobuf import timestamp_pb2

from tensorcast.global_store.exceptions import ValidationError
from tensorcast.global_store.models import (
    ByteSpaceKind,
    ByteSpaceRef,
    ExportState,
    MemoryType,
    Replica,
)
from tensorcast.global_store.replica_memory_codec import (
    memory_info_to_replica,
    parse_transport_metadata,
    replica_to_memory_info,
)
from tensorcast.proto.common.v1 import common_pb2


def test_parse_transport_metadata_without_transport_field_is_presence_only():
    mem_info = common_pb2.MemoryInfo(memory_size=1024)
    (
        transport_authoritative,
        export_state,
        export_generation,
        remote_keys,
        buffer_sizes,
        verification_json,
    ) = parse_transport_metadata(mem_info)
    assert transport_authoritative is False
    assert export_state == ExportState.PRESENCE_ONLY
    assert export_generation == 0
    assert remote_keys == []
    assert buffer_sizes == []
    assert verification_json is None


def test_parse_transport_metadata_exportable_without_keys_downgrades_to_presence_only():
    mem_info = common_pb2.MemoryInfo(memory_size=1024)
    transport = mem_info.transport
    transport.export_state = common_pb2.ReplicaTransportMetadata.EXPORT_STATE_EXPORTABLE
    transport.export_generation = 5
    (
        transport_authoritative,
        export_state,
        export_generation,
        remote_keys,
        buffer_sizes,
        verification_json,
    ) = parse_transport_metadata(mem_info)
    assert transport_authoritative is True
    assert export_state == ExportState.PRESENCE_ONLY
    assert export_generation == 5
    assert remote_keys == []
    assert buffer_sizes == []
    assert verification_json is None


def test_memory_info_to_replica_requires_view_id_when_strict():
    mem_info = common_pb2.MemoryInfo(
        memory_type=common_pb2.MemoryType.MEMORY_TYPE_GPU,
    )
    mem_info.byte_space.kind = common_pb2.BYTE_SPACE_KIND_VIEW
    mem_info.byte_space.id = ""
    with pytest.raises(ValidationError, match="byte_space VIEW requires id"):
        memory_info_to_replica(
            mem_info=mem_info,
            artifact_id="mi2:index:data",
            max_concurrency=1,
            worker_id="worker-1",
            require_view_id=True,
        )


def test_memory_info_to_replica_allows_blank_view_id_when_lenient():
    mem_info = common_pb2.MemoryInfo(
        memory_type=common_pb2.MemoryType.MEMORY_TYPE_RAM,
    )
    mem_info.byte_space.kind = common_pb2.BYTE_SPACE_KIND_VIEW
    mem_info.byte_space.id = ""
    replica = memory_info_to_replica(
        mem_info=mem_info,
        artifact_id="mi2:index:data",
        max_concurrency=1,
        worker_id="worker-1",
        require_view_id=False,
    )
    assert replica.byte_space.kind == ByteSpaceKind.CANONICAL


def test_replica_to_memory_info_roundtrip_transport_and_view_space():
    created_at = datetime.now(tz=timezone.utc)
    replica = Replica(
        artifact_id="mi2:index:data",
        byte_space=ByteSpaceRef.view("v-1"),
        node_id="n1",
        node_address="127.0.0.1",
        node_port=9000,
        memory_size=1024,
        memory_type=MemoryType.GPU,
        export_state=ExportState.EXPORTABLE,
        export_generation=7,
        remote_memory_keys=["k0"],
        buffer_sizes=[1024],
        verification_json='{"ok":true}',
        created_at=created_at,
    )

    def to_timestamp(value: datetime | None) -> timestamp_pb2.Timestamp | None:
        if value is None:
            return None
        ts = timestamp_pb2.Timestamp()
        ts.FromDatetime(value)
        return ts

    mem_info = replica_to_memory_info(
        replica=replica,
        datetime_to_timestamp=to_timestamp,
    )
    assert mem_info.byte_space.kind == common_pb2.BYTE_SPACE_KIND_VIEW
    assert mem_info.byte_space.id == "v-1"
    assert (
        mem_info.transport.export_state
        == common_pb2.ReplicaTransportMetadata.EXPORT_STATE_EXPORTABLE
    )
    assert list(mem_info.transport.remote_memory_keys) == ["k0"]
    assert list(mem_info.transport.buffer_sizes) == [1024]
    assert mem_info.HasField("creation_ts")
