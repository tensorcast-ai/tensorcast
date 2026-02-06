#  Copyright (c) 2025-2026, TensorCast Team.

"""Transport and artifact index RPC edge-case coverage for GlobalStoreServicer."""

import grpc

from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.global_store.v1 import global_store_pb2


def test_get_artifact_index_requires_tensor_index_key(servicer, test_context):
    response = servicer.GetArtifactIndex(
        global_store_pb2.GetArtifactIndexRequest(),
        test_context,
    )
    assert response.status == global_store_pb2.Status.STATUS_ERROR
    assert test_context.code == grpc.StatusCode.INVALID_ARGUMENT
    assert "tensor_index_key is required" in (test_context.details or "")


def test_get_artifact_index_not_found(servicer, test_context):
    response = servicer.GetArtifactIndex(
        global_store_pb2.GetArtifactIndexRequest(tensor_index_key="missing-key"),
        test_context,
    )
    assert response.status == global_store_pb2.Status.STATUS_NOT_FOUND


def test_request_transport_view_byte_space_requires_id(servicer, test_context):
    response = servicer.RequestReplicaTransport(
        global_store_pb2.RequestReplicaTransportRequest(
            artifact_id="artifact-x",
            requested_byte_space=common_pb2.ByteSpaceRef(
                kind=common_pb2.BYTE_SPACE_KIND_VIEW,
                id="",
            ),
        ),
        test_context,
    )
    assert response.status == global_store_pb2.Status.STATUS_ERROR
    assert test_context.code == grpc.StatusCode.INVALID_ARGUMENT
    assert "VIEW requires id" in (test_context.details or "")


def test_request_transport_rejects_unknown_byte_space_kind(servicer, test_context):
    response = servicer.RequestReplicaTransport(
        global_store_pb2.RequestReplicaTransportRequest(
            artifact_id="artifact-y",
            requested_byte_space=common_pb2.ByteSpaceRef(
                kind=999,
                id="ignored",
            ),
        ),
        test_context,
    )
    assert response.status == global_store_pb2.Status.STATUS_ERROR
    assert test_context.code == grpc.StatusCode.INVALID_ARGUMENT
    assert "unsupported requested_byte_space kind" in (test_context.details or "")


def test_complete_transport_invalid_uuid_returns_error(servicer, test_context):
    response = servicer.CompleteReplicaTransport(
        global_store_pb2.CompleteReplicaTransportRequest(transport_id="not-a-uuid"),
        test_context,
    )
    assert response.status == global_store_pb2.Status.STATUS_ERROR
    assert test_context.code == grpc.StatusCode.INTERNAL
