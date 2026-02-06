#  Copyright (c) 2025-2026, TensorCast Team.

"""Operation RPC coverage for GlobalStoreServicer."""

import grpc

from tensorcast.proto.operation.v1 import operation_pb2


def _acquire_request(
    *,
    operation_id: str,
    owner_id: str,
    ttl_ms: int = 5_000,
) -> operation_pb2.AcquireOperationLeaseRequest:
    return operation_pb2.AcquireOperationLeaseRequest(
        operation_id=operation_id,
        kind="seal",
        target_artifact_id="mi2:test_index:test_data",
        owner_id=owner_id,
        ttl_ms=ttl_ms,
    )


def test_acquire_operation_lease_success(servicer, test_context):
    req = _acquire_request(operation_id="op-lease-1", owner_id="worker-a")
    resp = servicer.AcquireOperationLease(req, test_context)

    assert resp.acquired is True
    assert resp.lease.operation_id == "op-lease-1"
    assert resp.lease.owner_id == "worker-a"
    assert resp.lease.lease_token != ""
    assert resp.lease.lease_generation >= 1
    assert test_context.code is None


def test_acquire_operation_lease_conflict_sets_already_exists(servicer, test_context):
    first = _acquire_request(operation_id="op-lease-2", owner_id="worker-a")
    first_resp = servicer.AcquireOperationLease(first, test_context)
    assert first_resp.acquired is True

    second_context = type(test_context)()
    second = _acquire_request(operation_id="op-lease-2", owner_id="worker-b")
    second_resp = servicer.AcquireOperationLease(second, second_context)

    assert second_resp.acquired is False
    assert second_context.code == grpc.StatusCode.ALREADY_EXISTS
    assert "held by another owner" in (second_context.details or "")


def test_update_operation_then_get_round_trip(servicer, test_context):
    acquire_resp = servicer.AcquireOperationLease(
        _acquire_request(operation_id="op-update-1", owner_id="worker-a"),
        test_context,
    )
    assert acquire_resp.acquired is True

    update_req = operation_pb2.UpdateOperationRequest(
        operation_id="op-update-1",
        lease_generation=acquire_resp.lease.lease_generation,
        status=operation_pb2.OperationStatus(
            state=operation_pb2.OPERATION_STATE_RUNNING,
            message="running",
            progress=0.5,
        ),
    )
    update_context = type(test_context)()
    update_resp = servicer.UpdateOperation(update_req, update_context)
    assert isinstance(update_resp, operation_pb2.UpdateOperationResponse)
    assert update_context.code is None

    get_context = type(test_context)()
    get_resp = servicer.GetOperation(
        operation_pb2.GetOperationRequest(operation_id="op-update-1"),
        get_context,
    )
    assert get_resp.ref.operation_id == "op-update-1"
    assert get_resp.status.state == operation_pb2.OPERATION_STATE_RUNNING
    assert get_resp.status.message == "running"
    assert get_context.code is None


def test_update_operation_with_stale_generation_fails(servicer, test_context):
    acquire_resp = servicer.AcquireOperationLease(
        _acquire_request(operation_id="op-update-2", owner_id="worker-a"),
        test_context,
    )
    assert acquire_resp.acquired is True

    stale_update_req = operation_pb2.UpdateOperationRequest(
        operation_id="op-update-2",
        lease_generation=acquire_resp.lease.lease_generation + 1,
        status=operation_pb2.OperationStatus(
            state=operation_pb2.OPERATION_STATE_RUNNING,
            message="stale-generation",
            progress=0.1,
        ),
    )
    stale_context = type(test_context)()
    _ = servicer.UpdateOperation(stale_update_req, stale_context)
    assert stale_context.code == grpc.StatusCode.FAILED_PRECONDITION
    assert "stale lease_generation" in (stale_context.details or "")
