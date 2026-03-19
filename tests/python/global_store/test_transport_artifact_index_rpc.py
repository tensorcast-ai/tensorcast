#  Copyright (c) 2025-2026, TensorCast Team.

"""Transport and artifact index RPC edge-case coverage for GlobalStoreServicer."""

import grpc

from tensorcast.global_store.grpc_helpers import (
    index_bytes_to_multibase_sha256,
    multibase_sha256_to_hex,
)
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


def test_upsert_artifact_metadata_persists_artifact_and_index(servicer, test_context):
    index_bytes = b'{"metadata":{"total_size_bytes":4},"tensors":[]}'
    index_mh = index_bytes_to_multibase_sha256(index_bytes)
    assert index_mh is not None
    artifact_id = f"mi2:{index_mh}:data-mh"

    response = servicer.UpsertArtifactMetadata(
        global_store_pb2.UpsertArtifactMetadataRequest(
            descriptor=common_pb2.ArtifactDescriptor(
                artifact_id=artifact_id,
                index_multihash=index_mh,
                data_multihash="data-mh",
                schema_version="v3",
                encoding="json",
                id_kind=common_pb2.ArtifactIdKind.ARTIFACT_ID_KIND_MI2,
            ),
            canonical_index_data=index_bytes,
        ),
        test_context,
    )

    assert response.status == global_store_pb2.Status.STATUS_OK
    row = servicer.artifacts_repo.get(artifact_id)
    assert row is not None
    assert row["index_multihash"] == index_mh
    assert row["data_multihash"] == "data-mh"
    index_key = multibase_sha256_to_hex(index_mh)
    assert index_key is not None
    assert servicer.artifact_indices.get(index_key) == index_bytes


def test_upsert_artifact_metadata_rejects_digest_mismatch(servicer, test_context):
    index_bytes = b'{"metadata":{"total_size_bytes":8},"tensors":[]}'
    wrong_index_mh = index_bytes_to_multibase_sha256(b"wrong-index")
    assert wrong_index_mh is not None
    artifact_id = f"mi2:{wrong_index_mh}:data-mh"

    response = servicer.UpsertArtifactMetadata(
        global_store_pb2.UpsertArtifactMetadataRequest(
            descriptor=common_pb2.ArtifactDescriptor(
                artifact_id=artifact_id,
                index_multihash=wrong_index_mh,
                data_multihash="data-mh",
                schema_version="v3",
                encoding="json",
                id_kind=common_pb2.ArtifactIdKind.ARTIFACT_ID_KIND_MI2,
            ),
            canonical_index_data=index_bytes,
        ),
        test_context,
    )

    assert response.status == global_store_pb2.Status.STATUS_ERROR
    assert test_context.code == grpc.StatusCode.FAILED_PRECONDITION
    assert "digest does not match" in (test_context.details or "")
    assert servicer.artifacts_repo.get(artifact_id) is None


def test_request_transport_view_byte_space_requires_id(servicer, test_context):
    response = servicer.RequestReplicaTransport(
        global_store_pb2.RequestReplicaTransportRequest(
            artifact_id="artifact-x",
            request_id="transport-view-byte-space-missing-id",
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
            request_id="transport-byte-space-unknown-kind",
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
