#  Copyright (c) 2025-2026, TensorCast Team.

# Copyright (c) 2025, TensorCast Team.

import grpc

from tensorcast.global_store.exceptions import DatabaseError
from tensorcast.proto.global_store.v1 import global_store_pb2


def _plan_request(shard_id: str = "shard-0") -> global_store_pb2.PlanPlacementRequest:
    return global_store_pb2.PlanPlacementRequest(
        artifact_id="artifact-1",
        placement_policy=global_store_pb2.PLACEMENT_POLICY_REPLICATED,
        source_node_id="test_node_1",
        shards=[
            global_store_pb2.PlacementShard(
                shard_id=shard_id,
                shard_idx=0,
                size_bytes=1024,
                content_digest="digest-0",
                byte_range_start=0,
                byte_range_length=1024,
                chunk_ids=[0, 1],
            )
        ],
    )


def test_plan_placement_writes_plan(servicer, test_context, registered_worker):
    assert registered_worker
    response = servicer.PlanPlacement(_plan_request(), test_context)
    assert response.plan_id
    assert len(response.placements) == 1
    placement = response.placements[0]
    assert placement.shard.shard_id == "shard-0"
    assert placement.targets  # local target is always present
    row = (
        servicer.placement_repository.get_cursor()
        .execute(
            "SELECT policy, shard_count FROM artifact_placements WHERE plan_id = ?",
            [response.plan_id],
        )
        .fetchone()
    )
    assert row is not None
    assert row[0] == "replicated"
    assert row[1] == 1


def test_report_persistence_status_upserts(servicer, test_context, registered_worker):
    assert registered_worker
    plan = servicer.PlanPlacement(_plan_request(), test_context)
    status_response = servicer.ReportPersistenceStatus(
        global_store_pb2.ReportPersistenceStatusRequest(
            task_id="task-1",
            artifact_id="artifact-1",
            plan_id=plan.plan_id,
            state=global_store_pb2.PERSISTENCE_STATE_RUNNING,
            progress=0.5,
            shard_statuses=[
                global_store_pb2.PersistenceShardStatus(
                    shard_id="shard-0",
                    shard_idx=0,
                    state=global_store_pb2.PERSISTENCE_STATE_RUNNING,
                    progress=0.5,
                    targets=[
                        global_store_pb2.PlacementTarget(
                            node_id="test_node_1",
                            target_state=global_store_pb2.PLACEMENT_TARGET_STATE_PENDING,
                        )
                    ],
                )
            ],
        ),
        test_context,
    )
    assert status_response.status == global_store_pb2.Status.STATUS_OK
    status_model = servicer.persistence_status_repository.get_by_task_id("task-1")
    assert status_model is not None
    assert status_model.state == "running"


def test_plan_placement_rejects_unspecified_policy(
    servicer, test_context, registered_worker
):
    assert registered_worker
    request = _plan_request()
    request.placement_policy = global_store_pb2.PLACEMENT_POLICY_UNSPECIFIED

    response = servicer.PlanPlacement(request, test_context)

    assert response.plan_id == ""
    assert test_context.code == grpc.StatusCode.INVALID_ARGUMENT


def test_plan_placement_assigns_default_shard_id(
    servicer, test_context, registered_worker
):
    assert registered_worker
    request = global_store_pb2.PlanPlacementRequest(
        artifact_id="artifact-2",
        placement_policy=global_store_pb2.PLACEMENT_POLICY_LOCAL_ONLY,
        source_node_id="test_node_1",
        shards=[
            global_store_pb2.PlacementShard(
                shard_idx=0,
                size_bytes=2048,
                content_digest="digest-1",
                byte_range_start=0,
                byte_range_length=2048,
            )
        ],
    )

    response = servicer.PlanPlacement(request, test_context)

    assert response.plan_id
    assert response.placements[0].shard.shard_id == "artifact-2:0"


def test_report_persistence_status_rejects_unspecified_state(
    servicer, test_context, registered_worker
):
    assert registered_worker
    plan = servicer.PlanPlacement(_plan_request(), test_context)
    request = global_store_pb2.ReportPersistenceStatusRequest(
        task_id="task-2",
        artifact_id="artifact-1",
        plan_id=plan.plan_id,
        state=global_store_pb2.PERSISTENCE_STATE_UNSPECIFIED,
        progress=0.0,
    )

    response = servicer.ReportPersistenceStatus(request, test_context)

    assert response.status == global_store_pb2.Status.STATUS_ERROR
    assert test_context.code == grpc.StatusCode.INVALID_ARGUMENT


def test_report_persistence_status_tx_conflict_returns_aborted(
    servicer, test_context, registered_worker, monkeypatch
):
    assert registered_worker
    plan = servicer.PlanPlacement(_plan_request(), test_context)

    def _raise_conflict(*args, **kwargs):
        del args, kwargs
        raise DatabaseError(
            "Transaction failed: Failed to commit: write-write conflict on key: "
            '"plan-1, 0, node-1"'
        )

    monkeypatch.setattr(
        servicer.placement_persistence_rpc_handler._placement_service,
        "record_status",
        _raise_conflict,
    )

    response = servicer.ReportPersistenceStatus(
        global_store_pb2.ReportPersistenceStatusRequest(
            task_id="task-conflict-1",
            artifact_id="artifact-1",
            plan_id=plan.plan_id,
            state=global_store_pb2.PERSISTENCE_STATE_RUNNING,
            progress=0.1,
            shard_statuses=[
                global_store_pb2.PersistenceShardStatus(
                    shard_id="shard-0",
                    shard_idx=0,
                    state=global_store_pb2.PERSISTENCE_STATE_RUNNING,
                    progress=0.1,
                    targets=[
                        global_store_pb2.PlacementTarget(
                            node_id="test_node_1",
                            target_state=global_store_pb2.PLACEMENT_TARGET_STATE_PENDING,
                        )
                    ],
                )
            ],
        ),
        test_context,
    )

    assert response.status == global_store_pb2.Status.STATUS_ERROR
    assert test_context.code == grpc.StatusCode.ABORTED
