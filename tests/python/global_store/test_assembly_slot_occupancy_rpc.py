#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from tensorcast.proto.global_store.v1 import global_store_pb2


def _register_active_worker(servicer, test_context, *, daemon_id: str) -> None:
    response = servicer.RegisterWorker(
        global_store_pb2.RegisterWorkerRequest(
            node_id=f"node-{daemon_id}",
            node_address="10.0.0.1",
            grpc_port=50051,
            p2p_port=50052,
            mem_pool_total_size=1024,
            mem_pool_available_size=1024,
            daemon_id=daemon_id,
        ),
        test_context,
    )
    assert response.status == global_store_pb2.Status.STATUS_OK


def _make_occupancy(
    *,
    attempt_id: str = "attempt-1",
    slot_id: str = "view-a",
    binding_value_id: str = "value-1",
    coordinator_operation_id: str = "op-1",
    coordinator_generation: int = 1,
    state: str = "accepted",
) -> global_store_pb2.AssemblySlotOccupancy:
    occupancy = global_store_pb2.AssemblySlotOccupancy(
        attempt_id=attempt_id,
        slot_id=slot_id,
        structural_view_id="view-a",
        binding_id="binding-1",
        binding_value_id=binding_value_id,
        coverage_plan_hash="coverage-1",
        contributor_daemon_id="daemon-1",
        coordinator_operation_id=coordinator_operation_id,
        coordinator_generation=coordinator_generation,
        lease_id="lease-1",
        lease_generation=1,
        state=state,
    )
    occupancy.lease_expires_at.FromDatetime(
        datetime.now(timezone.utc) + timedelta(seconds=30)
    )
    return occupancy


def test_assembly_slot_occupancy_rpc_round_trip(servicer, test_context) -> None:
    _register_active_worker(servicer, test_context, daemon_id="daemon-1")

    upsert_resp = servicer.UpsertAssemblySlotOccupancy(
        global_store_pb2.UpsertAssemblySlotOccupancyRequest(
            occupancy=_make_occupancy()
        ),
        test_context,
    )

    assert upsert_resp.status == global_store_pb2.Status.STATUS_OK
    assert upsert_resp.occupancy.attempt_id == "attempt-1"
    assert upsert_resp.occupancy.slot_id == "view-a"

    get_resp = servicer.GetAssemblySlotOccupancy(
        global_store_pb2.GetAssemblySlotOccupancyRequest(
            attempt_id="attempt-1",
            slot_id="view-a",
        ),
        test_context,
    )
    assert get_resp.status == global_store_pb2.Status.STATUS_OK
    assert get_resp.occupancy.binding_value_id == "value-1"

    list_resp = servicer.ListAssemblySlotOccupancies(
        global_store_pb2.ListAssemblySlotOccupanciesRequest(
            attempt_id="attempt-1",
            states=["accepted"],
        ),
        test_context,
    )
    assert list_resp.status == global_store_pb2.Status.STATUS_OK
    assert [item.slot_id for item in list_resp.occupancies] == ["view-a"]

    update_resp = servicer.UpdateAssemblySlotOccupancyState(
        global_store_pb2.UpdateAssemblySlotOccupancyStateRequest(
            attempt_id="attempt-1",
            slot_id="view-a",
            state="released",
            expected_lease_id="lease-1",
            expected_lease_generation=1,
            current_states=["accepted"],
        ),
        test_context,
    )
    assert update_resp.status == global_store_pb2.Status.STATUS_OK
    assert update_resp.occupancy.state == "released"


def test_assembly_slot_occupancy_rejects_live_conflict(servicer, test_context) -> None:
    _register_active_worker(servicer, test_context, daemon_id="daemon-1")

    first = servicer.UpsertAssemblySlotOccupancy(
        global_store_pb2.UpsertAssemblySlotOccupancyRequest(
            occupancy=_make_occupancy()
        ),
        test_context,
    )
    assert first.status == global_store_pb2.Status.STATUS_OK

    conflict_context = type(test_context)()
    resp = servicer.UpsertAssemblySlotOccupancy(
        global_store_pb2.UpsertAssemblySlotOccupancyRequest(
            occupancy=_make_occupancy(
                binding_value_id="value-2",
                coordinator_operation_id="op-2",
                coordinator_generation=2,
            )
        ),
        conflict_context,
    )

    assert resp.status == global_store_pb2.Status.STATUS_ERROR
    assert conflict_context.code is not None
    assert "already held by a live contributor" in str(conflict_context.details)
