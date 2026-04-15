#  Copyright (c) 2025-2026, TensorCast Team.

"""Chunk directory and disk location RPC coverage for GlobalStoreServicer."""

import grpc

from tensorcast.proto.global_store.v1 import global_store_pb2


def test_query_chunk_locations_returns_ok_for_missing_artifact(servicer, test_context):
    response = servicer.QueryChunkLocations(
        global_store_pb2.QueryChunkLocationsRequest(artifact_id="missing-artifact"),
        test_context,
    )
    assert response.status == global_store_pb2.Status.STATUS_OK
    assert list(response.locations) == []


def test_batch_update_chunk_states_empty_payload_is_noop(servicer, test_context):
    response = servicer.BatchUpdateChunkStates(
        global_store_pb2.BatchUpdateChunkStatesRequest(
            worker_id="worker-1",
            node_id="node-1",
        ),
        test_context,
    )
    assert response.status == global_store_pb2.Status.STATUS_OK
    assert response.updates_applied == 0


def test_upsert_artifact_disk_location_rejects_cluster_mismatch(servicer, test_context):
    response = servicer.UpsertArtifactDiskLocation(
        global_store_pb2.UpsertArtifactDiskLocationRequest(
            artifact_id="aid-1",
            cluster_id="other-cluster",
            relative_path="models/aid-1",
            kind=global_store_pb2.DISK_LOCATION_KIND_MANAGED,
        ),
        test_context,
    )
    assert response.status == global_store_pb2.Status.STATUS_ERROR
    assert test_context.code == grpc.StatusCode.INVALID_ARGUMENT
    assert "cluster_id does not match server cluster_id" in (test_context.details or "")


def test_upsert_and_list_artifact_disk_location_round_trip(servicer, test_context):
    upsert_response = servicer.UpsertArtifactDiskLocation(
        global_store_pb2.UpsertArtifactDiskLocationRequest(
            artifact_id="aid-2",
            cluster_id=servicer.cluster_id,
            relative_path="models/aid-2",
            kind=global_store_pb2.DISK_LOCATION_KIND_MANAGED,
        ),
        test_context,
    )
    assert upsert_response.status == global_store_pb2.Status.STATUS_OK

    list_context = type(test_context)()
    list_response = servicer.ListArtifactDiskLocations(
        global_store_pb2.ListArtifactDiskLocationsRequest(artifact_id="aid-2"),
        list_context,
    )
    assert list_response.status == global_store_pb2.Status.STATUS_OK
    assert len(list_response.locations) == 1
    location = list_response.locations[0]
    assert location.artifact_id == "aid-2"
    assert location.cluster_id == servicer.cluster_id
    assert location.relative_path == "models/aid-2"
    assert location.kind == global_store_pb2.DISK_LOCATION_KIND_MANAGED


def test_upsert_artifact_disk_location_rejects_msa1(servicer, test_context):
    response = servicer.UpsertArtifactDiskLocation(
        global_store_pb2.UpsertArtifactDiskLocationRequest(
            artifact_id="msa1:test-session~policy~partitioned~deadbeef",
            cluster_id=servicer.cluster_id,
            relative_path="models/msa1",
            kind=global_store_pb2.DISK_LOCATION_KIND_MANAGED,
        ),
        test_context,
    )
    assert response.status == global_store_pb2.Status.STATUS_ERROR
    assert test_context.code == grpc.StatusCode.FAILED_PRECONDITION
    assert "daemon-session-local" in (test_context.details or "")


def test_list_artifact_disk_locations_rejects_msa1(servicer, test_context):
    response = servicer.ListArtifactDiskLocations(
        global_store_pb2.ListArtifactDiskLocationsRequest(
            artifact_id="msa1:test-session~policy~partitioned~deadbeef"
        ),
        test_context,
    )
    assert response.status == global_store_pb2.Status.STATUS_ERROR
    assert test_context.code == grpc.StatusCode.FAILED_PRECONDITION
    assert "daemon-session-local" in (test_context.details or "")


def test_upsert_artifact_disk_location_rejects_unsafe_relative_path(
    servicer,
    test_context,
):
    response = servicer.UpsertArtifactDiskLocation(
        global_store_pb2.UpsertArtifactDiskLocationRequest(
            artifact_id="aid-3",
            cluster_id=servicer.cluster_id,
            relative_path="../escape",
            kind=global_store_pb2.DISK_LOCATION_KIND_MANAGED,
        ),
        test_context,
    )
    assert response.status == global_store_pb2.Status.STATUS_ERROR
    assert test_context.code == grpc.StatusCode.INVALID_ARGUMENT
    assert "safe, relative path" in (test_context.details or "")
