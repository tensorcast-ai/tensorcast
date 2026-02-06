#  Copyright (c) 2025-2026, TensorCast Team.

"""Artifact query RPC edge-case coverage for GlobalStoreServicer."""

import grpc

from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.global_store.v1 import global_store_pb2


def test_get_artifact_info_requires_byte_space_for_leaves(servicer, test_context):
    response = servicer.GetArtifactInfoById(
        global_store_pb2.GetArtifactInfoByIdRequest(
            artifact_id="mi2:index:data",
            include_leaves=True,
        ),
        test_context,
    )
    assert response.status == global_store_pb2.Status.STATUS_ERROR
    assert test_context.code == grpc.StatusCode.INVALID_ARGUMENT
    assert "requested_byte_space required" in (test_context.details or "")


def test_get_artifact_info_rejects_view_meta_on_canonical_space(servicer, test_context):
    artifact_id = "mi2:index:data"
    servicer.artifacts_repo.upsert_artifact(
        artifact_id=artifact_id,
        index_multihash="index",
        data_multihash="data",
        schema_version="v3",
        encoding="json",
    )
    response = servicer.GetArtifactInfoById(
        global_store_pb2.GetArtifactInfoByIdRequest(
            artifact_id=artifact_id,
            include_view_meta=True,
            requested_byte_space=common_pb2.ByteSpaceRef(
                kind=common_pb2.BYTE_SPACE_KIND_CANONICAL
            ),
        ),
        test_context,
    )
    assert response.status == global_store_pb2.Status.STATUS_ERROR
    assert test_context.code == grpc.StatusCode.INVALID_ARGUMENT
    assert "view metadata is only available" in (test_context.details or "")


def test_get_artifact_info_missing_view_compacts_partial_leaf_ranges(
    servicer, test_context
):
    artifact_id = "mi2:index:data"
    servicer.artifacts_repo.upsert_artifact(
        artifact_id=artifact_id,
        index_multihash="index",
        data_multihash="data",
        schema_version="v3",
        encoding="json",
    )
    request = global_store_pb2.GetArtifactInfoByIdRequest(
        artifact_id=artifact_id,
        include_leaves=True,
        requested_byte_space=common_pb2.ByteSpaceRef(
            kind=common_pb2.BYTE_SPACE_KIND_VIEW,
            id="missing-view",
        ),
    )
    request.leaf_idxs.extend([1, 2, 4])
    response = servicer.GetArtifactInfoById(request, test_context)

    assert response.status == global_store_pb2.Status.STATUS_NOT_FOUND
    assert test_context.code == grpc.StatusCode.NOT_FOUND
    assert len(response.partial_leaf_coverage) == 1
    ranges = response.partial_leaf_coverage[0].missing_leaf_ranges
    assert len(ranges) == 2
    assert ranges[0].start == 1 and ranges[0].count == 2
    assert ranges[1].start == 4 and ranges[1].count == 1
