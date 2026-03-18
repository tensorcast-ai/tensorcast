#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from tensorcast.global_store.models.worker import Worker


def _seed_active_worker(repositories, *, daemon_id: str) -> None:
    repositories["worker"].create(
        Worker(
            worker_id=f"worker:{daemon_id}",
            daemon_id=daemon_id,
            node_id=f"node:{daemon_id}",
            node_address="127.0.0.1",
            grpc_port=50051,
            p2p_port=50052,
            mem_pool_total_size=1024,
            mem_pool_available_size=1024,
        )
    )


def test_assembly_contribution_repository_upsert_and_query(repositories) -> None:
    repo = repositories["assembly_contribution"]
    lease_expires_at = datetime.now(timezone.utc)

    row = repo.upsert(
        assembly_id="cgid:assembly-attempt-1",
        view_id="view-a",
        binding_id="binding-1",
        binding_value_id="value-1",
        coverage_plan_hash="cph-1",
        contributor_daemon_id="daemon-1",
        coordinator_operation_id="op-1",
        coordinator_generation=1,
        lease_id="lease-1",
        lease_generation=7,
        lease_expires_at=lease_expires_at,
    )

    assert row["assembly_id"] == "cgid:assembly-attempt-1"
    assert row["view_id"] == "view-a"
    assert row["binding_id"] == "binding-1"
    assert row["binding_value_id"] == "value-1"
    assert row["state"] == "accepted"

    fetched = repo.get(assembly_id="cgid:assembly-attempt-1", view_id="view-a")
    assert fetched is not None
    assert fetched["lease_id"] == "lease-1"
    assert fetched["lease_generation"] == 7

    by_assembly = repo.list_by_assembly(assembly_id="cgid:assembly-attempt-1")
    assert len(by_assembly) == 1
    assert by_assembly[0]["view_id"] == "view-a"

    by_binding = repo.list_by_binding_value(
        binding_id="binding-1",
        binding_value_id="value-1",
    )
    assert by_binding == [
        {"assembly_id": "cgid:assembly-attempt-1", "view_id": "view-a"}
    ]


def test_assembly_contribution_repository_update_state(repositories) -> None:
    repo = repositories["assembly_contribution"]
    _ = repo.upsert(
        assembly_id="cgid:assembly-attempt-2",
        view_id="view-b",
        binding_id="binding-2",
        binding_value_id="value-2",
        coverage_plan_hash="cph-2",
        contributor_daemon_id="daemon-2",
        coordinator_operation_id="op-2",
        coordinator_generation=2,
        lease_id="lease-2",
        lease_generation=3,
        lease_expires_at=datetime.now(timezone.utc),
    )

    updated = repo.update_state(
        assembly_id="cgid:assembly-attempt-2",
        view_id="view-b",
        state="released",
    )

    assert updated["state"] == "released"
    assert (
        repo.list_by_assembly(
            assembly_id="cgid:assembly-attempt-2",
            states=("released",),
        )[0]["view_id"]
        == "view-b"
    )


def test_assembly_contribution_repository_update_state_if_current(
    repositories,
) -> None:
    repo = repositories["assembly_contribution"]
    original_expiry = datetime.now(timezone.utc) + timedelta(seconds=30)
    _ = repo.upsert(
        assembly_id="cgid:assembly-attempt-3",
        view_id="view-c",
        binding_id="binding-3",
        binding_value_id="value-3",
        coverage_plan_hash="cph-3",
        contributor_daemon_id="daemon-3",
        coordinator_operation_id="op-3",
        coordinator_generation=3,
        lease_id="lease-3",
        lease_generation=4,
        lease_expires_at=original_expiry,
    )

    mismatch = repo.update_state_if_current(
        assembly_id="cgid:assembly-attempt-3",
        view_id="view-c",
        state="stale",
        expected_lease_id="lease-other",
        expected_lease_generation=4,
        current_states=("accepted",),
    )
    assert mismatch is None

    refreshed_expiry = datetime.now(timezone.utc)
    refreshed = repo.update_state_if_current(
        assembly_id="cgid:assembly-attempt-3",
        view_id="view-c",
        state="accepted",
        expected_lease_id="lease-3",
        expected_lease_generation=4,
        current_states=("accepted",),
        lease_expires_at=refreshed_expiry,
    )
    assert refreshed is not None
    assert refreshed["state"] == "accepted"
    assert refreshed["lease_expires_at"] is not None
    assert refreshed["lease_expires_at"] >= refreshed_expiry

    updated = repo.update_state_if_current(
        assembly_id="cgid:assembly-attempt-3",
        view_id="view-c",
        state="released",
        expected_lease_id="lease-3",
        expected_lease_generation=4,
        current_states=("accepted",),
    )
    assert updated is not None
    assert updated["state"] == "released"


def test_assembly_contribution_repository_claim_slot_rejects_live_active_row(
    repositories,
) -> None:
    repo = repositories["assembly_contribution"]
    _seed_active_worker(repositories, daemon_id="daemon-live")
    original_expiry = datetime.now(timezone.utc) + timedelta(seconds=30)
    claimed = repo.claim_slot(
        assembly_id="cgid:assembly-attempt-live",
        view_id="view-live",
        binding_id="binding-1",
        binding_value_id="value-1",
        coverage_plan_hash="cph-live-1",
        contributor_daemon_id="daemon-live",
        coordinator_operation_id="op-live-1",
        coordinator_generation=1,
        lease_id="lease-live-1",
        lease_generation=1,
        lease_expires_at=original_expiry,
    )
    assert claimed is not None

    conflict = repo.claim_slot(
        assembly_id="cgid:assembly-attempt-live",
        view_id="view-live",
        binding_id="binding-2",
        binding_value_id="value-2",
        coverage_plan_hash="cph-live-2",
        contributor_daemon_id="daemon-other",
        coordinator_operation_id="op-live-2",
        coordinator_generation=2,
        lease_id="lease-live-2",
        lease_generation=1,
        lease_expires_at=datetime.now(timezone.utc) + timedelta(seconds=30),
    )

    assert conflict is None
    row = repo.get(
        assembly_id="cgid:assembly-attempt-live",
        view_id="view-live",
    )
    assert row is not None
    assert row["binding_id"] == "binding-1"
    assert row["binding_value_id"] == "value-1"


def test_assembly_contribution_repository_claim_slot_replaces_inactive_row(
    repositories,
) -> None:
    repo = repositories["assembly_contribution"]
    _ = repo.claim_slot(
        assembly_id="cgid:assembly-attempt-stale",
        view_id="view-stale",
        binding_id="binding-old",
        binding_value_id="value-old",
        coverage_plan_hash="cph-stale-old",
        contributor_daemon_id="daemon-missing",
        coordinator_operation_id="op-stale-old",
        coordinator_generation=1,
        lease_id="lease-stale-old",
        lease_generation=1,
        lease_expires_at=datetime.now(timezone.utc) + timedelta(seconds=30),
    )

    replaced = repo.claim_slot(
        assembly_id="cgid:assembly-attempt-stale",
        view_id="view-stale",
        binding_id="binding-new",
        binding_value_id="value-new",
        coverage_plan_hash="cph-stale-new",
        contributor_daemon_id="daemon-live-new",
        coordinator_operation_id="op-stale-new",
        coordinator_generation=2,
        lease_id="lease-stale-new",
        lease_generation=1,
        lease_expires_at=datetime.now(timezone.utc) + timedelta(seconds=30),
    )

    assert replaced is not None
    assert replaced["binding_id"] == "binding-new"
    assert replaced["binding_value_id"] == "value-new"


def test_assembly_contribution_repository_claim_slot_replaces_live_row_for_same_coordinator(
    repositories,
) -> None:
    repo = repositories["assembly_contribution"]
    _seed_active_worker(repositories, daemon_id="daemon-live")
    claimed = repo.claim_slot(
        assembly_id="cgid:assembly-attempt-replace",
        view_id="view-replace",
        binding_id="binding-old",
        binding_value_id="value-old",
        coverage_plan_hash="cph-old",
        contributor_daemon_id="daemon-live",
        coordinator_operation_id="op-replace",
        coordinator_generation=9,
        lease_id="lease-old",
        lease_generation=1,
        lease_expires_at=datetime.now(timezone.utc) + timedelta(seconds=30),
    )
    assert claimed is not None

    replaced = repo.claim_slot(
        assembly_id="cgid:assembly-attempt-replace",
        view_id="view-replace",
        binding_id="binding-new",
        binding_value_id="value-new",
        coverage_plan_hash="cph-new",
        contributor_daemon_id="daemon-live",
        coordinator_operation_id="op-replace",
        coordinator_generation=9,
        lease_id="lease-new",
        lease_generation=1,
        lease_expires_at=datetime.now(timezone.utc) + timedelta(seconds=30),
    )

    assert replaced is not None
    assert replaced["binding_id"] == "binding-new"
    assert replaced["binding_value_id"] == "value-new"
    assert replaced["lease_id"] == "lease-new"
