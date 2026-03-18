#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from google.protobuf import timestamp_pb2

from tensorcast.proto.global_store.v1 import global_store_pb2


def _make_contribution() -> global_store_pb2.AssemblyContribution:
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=30)
    return global_store_pb2.AssemblyContribution(
        assembly_id="cgid:assembly-1",
        view_id="view-1",
        binding_id="binding-1",
        binding_value_id="value-1",
        coverage_plan_hash="bcp1:test",
        contributor_daemon_id="daemon-1",
        coordinator_operation_id="seal-op-1",
        coordinator_generation=1,
        lease_id="lease-1",
        lease_generation=1,
        lease_expires_at=_timestamp(expires_at),
        state="accepted",
    )


def _timestamp(value: datetime) -> object:
    ts = timestamp_pb2.Timestamp()
    ts.FromDatetime(value)
    return ts


def _register_active_worker(servicer, test_context, *, daemon_id: str) -> None:
    _ = servicer.RegisterWorker(
        global_store_pb2.RegisterWorkerRequest(
            node_id=f"node:{daemon_id}",
            node_address="192.168.1.10",
            grpc_port=50051,
            p2p_port=50052,
            mem_pool_total_size=1024,
            mem_pool_available_size=1024,
            daemon_id=daemon_id,
        ),
        test_context,
    )


def test_assembly_contribution_rpc_round_trip(servicer, test_context) -> None:
    _register_active_worker(servicer, test_context, daemon_id="daemon-1")
    upsert_resp = servicer.UpsertAssemblyContribution(
        global_store_pb2.UpsertAssemblyContributionRequest(
            contribution=_make_contribution()
        ),
        test_context,
    )

    assert upsert_resp.status == global_store_pb2.Status.STATUS_OK
    assert upsert_resp.contribution.assembly_id == "cgid:assembly-1"
    assert upsert_resp.contribution.view_id == "view-1"

    get_resp = servicer.GetAssemblyContribution(
        global_store_pb2.GetAssemblyContributionRequest(
            assembly_id="cgid:assembly-1",
            view_id="view-1",
        ),
        test_context,
    )
    assert get_resp.status == global_store_pb2.Status.STATUS_OK
    assert get_resp.contribution.binding_value_id == "value-1"

    list_resp = servicer.ListAssemblyContributions(
        global_store_pb2.ListAssemblyContributionsRequest(
            assembly_id="cgid:assembly-1",
            states=["accepted"],
        ),
        test_context,
    )
    assert list_resp.status == global_store_pb2.Status.STATUS_OK
    assert [item.view_id for item in list_resp.contributions] == ["view-1"]

    update_resp = servicer.UpdateAssemblyContributionState(
        global_store_pb2.UpdateAssemblyContributionStateRequest(
            assembly_id="cgid:assembly-1",
            view_id="view-1",
            state="released",
            expected_lease_id="lease-1",
            expected_lease_generation=1,
            current_states=["accepted"],
        ),
        test_context,
    )
    assert update_resp.status == global_store_pb2.Status.STATUS_OK
    assert update_resp.contribution.state == "released"


def test_assembly_contribution_rpc_cas_mismatch_is_not_found(
    servicer, test_context
) -> None:
    _register_active_worker(servicer, test_context, daemon_id="daemon-1")
    _ = servicer.UpsertAssemblyContribution(
        global_store_pb2.UpsertAssemblyContributionRequest(
            contribution=_make_contribution()
        ),
        test_context,
    )

    update_resp = servicer.UpdateAssemblyContributionState(
        global_store_pb2.UpdateAssemblyContributionStateRequest(
            assembly_id="cgid:assembly-1",
            view_id="view-1",
            state="stale",
            expected_lease_id="lease-other",
            expected_lease_generation=1,
            current_states=["accepted"],
        ),
        test_context,
    )

    assert update_resp.status == global_store_pb2.Status.STATUS_NOT_FOUND


def test_assembly_contribution_rpc_requires_expiry_for_accepted_updates(
    servicer, test_context
) -> None:
    resp = servicer.UpdateAssemblyContributionState(
        global_store_pb2.UpdateAssemblyContributionStateRequest(
            assembly_id="cgid:assembly-1",
            view_id="view-1",
            state="accepted",
        ),
        test_context,
    )

    assert resp.status == global_store_pb2.Status.STATUS_ERROR


def test_assembly_contribution_rpc_rejects_live_slot_conflict(
    servicer, test_context
) -> None:
    _register_active_worker(servicer, test_context, daemon_id="daemon-1")
    first = servicer.UpsertAssemblyContribution(
        global_store_pb2.UpsertAssemblyContributionRequest(
            contribution=_make_contribution()
        ),
        test_context,
    )
    assert first.status == global_store_pb2.Status.STATUS_OK

    conflict = _make_contribution()
    conflict.binding_id = "binding-2"
    conflict.binding_value_id = "value-2"
    conflict.coverage_plan_hash = "bcp1:other"
    conflict.coordinator_operation_id = "seal-op-2"
    conflict.coordinator_generation = 2
    conflict.lease_id = "lease-2"

    resp = servicer.UpsertAssemblyContribution(
        global_store_pb2.UpsertAssemblyContributionRequest(contribution=conflict),
        test_context,
    )

    assert resp.status == global_store_pb2.Status.STATUS_ERROR


def test_assembly_contribution_rpc_allows_live_slot_replacement_for_same_coordinator(
    servicer, test_context
) -> None:
    _register_active_worker(servicer, test_context, daemon_id="daemon-1")
    first = servicer.UpsertAssemblyContribution(
        global_store_pb2.UpsertAssemblyContributionRequest(
            contribution=_make_contribution()
        ),
        test_context,
    )
    assert first.status == global_store_pb2.Status.STATUS_OK

    replacement = _make_contribution()
    replacement.binding_id = "binding-2"
    replacement.binding_value_id = "value-2"
    replacement.coverage_plan_hash = "bcp1:replacement"
    replacement.lease_id = "lease-2"

    resp = servicer.UpsertAssemblyContribution(
        global_store_pb2.UpsertAssemblyContributionRequest(contribution=replacement),
        test_context,
    )

    assert resp.status == global_store_pb2.Status.STATUS_OK
    assert resp.contribution.binding_id == "binding-2"
    assert resp.contribution.binding_value_id == "value-2"
    assert resp.contribution.lease_id == "lease-2"
