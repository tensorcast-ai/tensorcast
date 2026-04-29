from __future__ import annotations

from uuid import UUID

import pytest

from tensorcast.global_store.models import (
    BroadcastEdge,
    BroadcastEdgeState,
    BroadcastSession,
    BroadcastSessionState,
    BroadcastTarget,
    BroadcastTargetState,
    ExportState,
    MemoryType,
    Replica,
    TransportCompletionOutcome,
    Worker,
)
from tensorcast.global_store.services import BroadcastService


def _worker(worker_id: str, daemon_id: str, node_id: str) -> Worker:
    return Worker(
        worker_id=worker_id,
        daemon_id=daemon_id,
        node_id=node_id,
        node_address=f"10.0.0.{node_id[-1]}",
        grpc_port=5000 + int(node_id[-1]),
        p2p_port=6000 + int(node_id[-1]),
        mem_pool_total_size=4096,
        mem_pool_available_size=4096,
        accepting_new_requests=True,
    )


def _exportable_replica(artifact_id: str, worker: Worker) -> Replica:
    return Replica(
        artifact_id=artifact_id,
        node_id=worker.node_id,
        node_address=worker.node_address,
        node_port=worker.p2p_port,
        memory_size=1024,
        memory_type=MemoryType.GPU,
        device_id=0,
        max_concurrency=4,
        current_requests=0,
        is_available=True,
        remote_memory_keys=[f"rk-{worker.worker_id}"],
        buffer_sizes=[1024],
        export_state=ExportState.EXPORTABLE,
        worker_id=worker.worker_id,
    )


def test_create_session_plans_first_layer_by_fanout(repositories):
    worker_repo = repositories["worker"]
    replica_repo = repositories["replica"]
    broadcast_repo = repositories["broadcast"]
    service = BroadcastService(
        broadcast_repository=broadcast_repo,
        replica_repository=replica_repo,
        worker_repository=worker_repo,
    )
    root = _worker("worker-root", "daemon-root", "node1")
    child1 = _worker("worker-child-1", "daemon-child-1", "node2")
    child2 = _worker("worker-child-2", "daemon-child-2", "node3")
    child3 = _worker("worker-child-3", "daemon-child-3", "node4")
    for worker in (root, child1, child2, child3):
        worker_repo.create(worker)
        assert worker_repo.update_heartbeat(worker.worker_id, 4096, True)
    root_replica = replica_repo.create(_exportable_replica("mi2:model-a", root))

    session = service.create_session(
        session_id="session-a",
        artifact_id="mi2:model-a",
        requested_view_id=None,
        epoch=42,
        fanout=2,
        target_daemon_ids=["daemon-child-1", "daemon-child-2", "daemon-child-3"],
        root_replica_id=str(root_replica.replica_id),
        strict_parent=True,
        max_attempts=3,
    )

    assert session.state is BroadcastSessionState.ACTIVE
    targets = broadcast_repo.list_targets("session-a")
    assert len(targets) == 3
    assigned = [t for t in targets if t.state is BroadcastTargetState.ASSIGNED]
    pending = [t for t in targets if t.state is BroadcastTargetState.PENDING]
    assert len(assigned) == 2
    assert len(pending) == 1
    edges = [
        broadcast_repo.find_active_edge_for_child("session-a", t.target_worker_id)
        for t in assigned
    ]
    assert all(edge is not None for edge in edges)
    assert all(edge.state is BroadcastEdgeState.PLANNED for edge in edges if edge)
    assert all(edge.parent_replica_id == root_replica.replica_id for edge in edges if edge)
    assert len(service.list_edges("session-a")) == 2


def test_create_session_duplicate_explicit_root_returns_existing_without_counter_change(
    repositories,
):
    worker_repo = repositories["worker"]
    replica_repo = repositories["replica"]
    broadcast_repo = repositories["broadcast"]
    service = BroadcastService(
        broadcast_repository=broadcast_repo,
        replica_repository=replica_repo,
        worker_repository=worker_repo,
    )
    root = _worker("worker-root-explicit", "daemon-root-explicit", "node1")
    child = _worker("worker-child-explicit", "daemon-child-explicit", "node2")
    for worker in (root, child):
        worker_repo.create(worker)
        assert worker_repo.update_heartbeat(worker.worker_id, 4096, True)
    root_replica = replica_repo.create(_exportable_replica("mi2:model-explicit", root))

    first = service.create_session(
        session_id="session-explicit",
        artifact_id="mi2:model-explicit",
        requested_view_id=None,
        epoch=1,
        fanout=1,
        target_daemon_ids=["daemon-child-explicit"],
        root_replica_id=str(root_replica.replica_id),
        strict_parent=True,
        max_attempts=3,
    )
    assert replica_repo.get_current_requests(root_replica.replica_id) == 0

    second = service.create_session(
        session_id="session-explicit",
        artifact_id="mi2:model-explicit",
        requested_view_id=None,
        epoch=1,
        fanout=1,
        target_daemon_ids=["daemon-child-explicit"],
        root_replica_id=str(root_replica.replica_id),
        strict_parent=True,
        max_attempts=3,
    )

    assert second.session_id == first.session_id
    assert replica_repo.get_current_requests(root_replica.replica_id) == 0
    assert len(broadcast_repo.list_targets("session-explicit")) == 1
    assert len(service.list_edges("session-explicit")) == 1


def test_create_session_duplicate_auto_root_returns_existing_without_counter_change(
    repositories,
):
    worker_repo = repositories["worker"]
    replica_repo = repositories["replica"]
    broadcast_repo = repositories["broadcast"]
    service = BroadcastService(
        broadcast_repository=broadcast_repo,
        replica_repository=replica_repo,
        worker_repository=worker_repo,
    )
    root = _worker("worker-root-auto", "daemon-root-auto", "node1")
    child = _worker("worker-child-auto", "daemon-child-auto", "node2")
    for worker in (root, child):
        worker_repo.create(worker)
        assert worker_repo.update_heartbeat(worker.worker_id, 4096, True)
    root_replica = replica_repo.create(_exportable_replica("mi2:model-auto", root))

    first = service.create_session(
        session_id="session-auto",
        artifact_id="mi2:model-auto",
        requested_view_id=None,
        epoch=1,
        fanout=1,
        target_daemon_ids=["daemon-child-auto"],
        root_replica_id="",
        strict_parent=True,
        max_attempts=3,
    )
    assert replica_repo.get_current_requests(root_replica.replica_id) == 0

    second = service.create_session(
        session_id="session-auto",
        artifact_id="mi2:model-auto",
        requested_view_id=None,
        epoch=1,
        fanout=1,
        target_daemon_ids=["daemon-child-auto"],
        root_replica_id="",
        strict_parent=True,
        max_attempts=3,
    )

    assert second.session_id == first.session_id
    assert replica_repo.get_current_requests(root_replica.replica_id) == 0
    assert len(broadcast_repo.list_targets("session-auto")) == 1
    assert len(service.list_edges("session-auto")) == 1


def test_create_session_auto_root_failure_releases_counter_and_rolls_back(
    repositories,
    monkeypatch,
):
    worker_repo = repositories["worker"]
    replica_repo = repositories["replica"]
    broadcast_repo = repositories["broadcast"]
    service = BroadcastService(
        broadcast_repository=broadcast_repo,
        replica_repository=replica_repo,
        worker_repository=worker_repo,
    )
    root = _worker("worker-root-failure", "daemon-root-failure", "node1")
    child = _worker("worker-child-failure", "daemon-child-failure", "node2")
    for worker in (root, child):
        worker_repo.create(worker)
        assert worker_repo.update_heartbeat(worker.worker_id, 4096, True)
    root_replica = replica_repo.create(_exportable_replica("mi2:model-failure", root))

    def fail_planning(*args, **kwargs):
        raise RuntimeError("forced planning failure")

    monkeypatch.setattr(service, "_plan_more_edges", fail_planning)

    with pytest.raises(RuntimeError, match="forced planning failure"):
        service.create_session(
            session_id="session-failure",
            artifact_id="mi2:model-failure",
            requested_view_id=None,
            epoch=1,
            fanout=1,
            target_daemon_ids=["daemon-child-failure"],
            root_replica_id=None,
            strict_parent=True,
            max_attempts=3,
        )

    assert replica_repo.get_current_requests(root_replica.replica_id) == 0
    assert broadcast_repo.find_session("session-failure") is None
    assert broadcast_repo.list_targets("session-failure") == []


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"session_id": ""}, "session_id is required"),
        ({"artifact_id": ""}, "artifact_id is required"),
        ({"epoch": -1}, "epoch must be >= 0"),
    ],
)
def test_create_session_validates_required_inputs(repositories, overrides, message):
    worker_repo = repositories["worker"]
    replica_repo = repositories["replica"]
    broadcast_repo = repositories["broadcast"]
    service = BroadcastService(
        broadcast_repository=broadcast_repo,
        replica_repository=replica_repo,
        worker_repository=worker_repo,
    )
    root = _worker("worker-root-validation", "daemon-root-validation", "node1")
    child = _worker("worker-child-validation", "daemon-child-validation", "node2")
    for worker in (root, child):
        worker_repo.create(worker)
        assert worker_repo.update_heartbeat(worker.worker_id, 4096, True)
    root_replica = replica_repo.create(
        _exportable_replica("mi2:model-validation", root)
    )

    kwargs = {
        "session_id": "session-validation",
        "artifact_id": "mi2:model-validation",
        "requested_view_id": None,
        "epoch": 1,
        "fanout": 1,
        "target_daemon_ids": ["daemon-child-validation"],
        "root_replica_id": str(root_replica.replica_id),
        "strict_parent": True,
        "max_attempts": 3,
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        service.create_session(**kwargs)
    assert broadcast_repo.find_session("session-validation") is None


@pytest.mark.parametrize(
    ("target_state", "assigned_edge_id"),
    [
        (BroadcastTargetState.COMPLETED, "edge-stale"),
        (BroadcastTargetState.ASSIGNED, "edge-new"),
    ],
)
def test_complete_transport_edge_stale_failure_does_not_requeue_target(
    repositories,
    target_state,
    assigned_edge_id,
):
    replica_repo = repositories["replica"]
    worker_repo = repositories["worker"]
    broadcast_repo = repositories["broadcast"]
    service = BroadcastService(
        broadcast_repository=broadcast_repo,
        replica_repository=replica_repo,
        worker_repository=worker_repo,
    )
    root_replica_id = UUID("00000000-0000-0000-0000-000000000001")
    completed_replica_id = UUID("00000000-0000-0000-0000-000000000002")
    broadcast_repo.create_session(
        BroadcastSession(
            session_id="session-stale-failure",
            artifact_id="mi2:stale-failure",
            requested_view_id=None,
            epoch=1,
            fanout=1,
            max_attempts=3,
            strict_parent=True,
            state=BroadcastSessionState.ACTIVE,
            root_replica_id=root_replica_id,
        )
    )
    broadcast_repo.upsert_target(
        BroadcastTarget(
            session_id="session-stale-failure",
            target_worker_id="worker-child",
            target_daemon_id="daemon-child",
            state=target_state,
            level=1,
            attempt=2,
            assigned_edge_id=assigned_edge_id,
            completed_replica_id=(
                completed_replica_id
                if target_state is BroadcastTargetState.COMPLETED
                else None
            ),
        )
    )
    broadcast_repo.create_edge(
        BroadcastEdge(
            edge_id="edge-stale",
            session_id="session-stale-failure",
            parent_worker_id="worker-root",
            parent_replica_id=root_replica_id,
            child_worker_id="worker-child",
            level=1,
            attempt=1,
            state=BroadcastEdgeState.PLANNED,
        )
    )
    if assigned_edge_id == "edge-new":
        with broadcast_repo.transaction() as tx:
            tx.execute(
                """
                INSERT INTO broadcast_edges (
                    edge_id, session_id, parent_worker_id, parent_replica_id,
                    child_worker_id, level, attempt, state
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    "edge-new",
                    "session-stale-failure",
                    "worker-root",
                    str(root_replica_id),
                    "worker-child",
                    1,
                    2,
                    BroadcastEdgeState.PLANNED.value,
                ],
            )

    with broadcast_repo.transaction() as tx:
        service.complete_transport_edge(
            session_id="session-stale-failure",
            edge_id="edge-stale",
            transport_outcome=TransportCompletionOutcome.FAILED,
            outcome_detail="stale completion",
            cursor=tx,
        )

    target = broadcast_repo.find_target("session-stale-failure", "worker-child")
    edges = broadcast_repo.list_edges("session-stale-failure")
    session = broadcast_repo.find_session("session-stale-failure")
    assert target is not None
    assert session is not None
    assert target.state is target_state
    assert target.assigned_edge_id == assigned_edge_id
    assert target.completed_replica_id == (
        completed_replica_id if target_state is BroadcastTargetState.COMPLETED else None
    )
    assert {edge.edge_id for edge in edges} == (
        {"edge-stale", "edge-new"} if assigned_edge_id == "edge-new" else {"edge-stale"}
    )
    assert session.state is BroadcastSessionState.ACTIVE


def test_complete_transport_edge_success_noops_after_session_cancelled(repositories):
    worker_repo = repositories["worker"]
    replica_repo = repositories["replica"]
    broadcast_repo = repositories["broadcast"]
    service = BroadcastService(
        broadcast_repository=broadcast_repo,
        replica_repository=replica_repo,
        worker_repository=worker_repo,
    )
    root = _worker("worker-root-cancel", "daemon-root-cancel", "node1")
    child1 = _worker("worker-child-cancel-1", "daemon-child-cancel-1", "node2")
    child2 = _worker("worker-child-cancel-2", "daemon-child-cancel-2", "node3")
    for worker in (root, child1, child2):
        worker_repo.create(worker)
        assert worker_repo.update_heartbeat(worker.worker_id, 4096, True)
    root_replica = replica_repo.create(_exportable_replica("mi2:model-cancel", root))

    service.create_session(
        session_id="session-cancel-inflight",
        artifact_id="mi2:model-cancel",
        requested_view_id=None,
        epoch=1,
        fanout=1,
        target_daemon_ids=[
            "daemon-child-cancel-1",
            "daemon-child-cancel-2",
        ],
        root_replica_id=str(root_replica.replica_id),
        strict_parent=True,
        max_attempts=3,
    )
    assigned_before_claim = [
        target
        for target in broadcast_repo.list_targets("session-cancel-inflight")
        if target.state is BroadcastTargetState.ASSIGNED
    ]
    assert len(assigned_before_claim) == 1
    claimed_worker_id = assigned_before_claim[0].target_worker_id

    with broadcast_repo.transaction() as tx:
        _, edge = service.claim_transport_edge(
            session_id="session-cancel-inflight",
            artifact_id="mi2:model-cancel",
            requested_view_id=None,
            requester_worker_id=claimed_worker_id,
            request_id="request-cancel-inflight",
            heartbeat_timeout_seconds=30.0,
            cursor=tx,
        )

    assert service.cancel_session("session-cancel-inflight")
    completed_child = child1 if child1.worker_id == claimed_worker_id else child2
    replica_repo.create(_exportable_replica("mi2:model-cancel", completed_child))

    with broadcast_repo.transaction() as tx:
        service.complete_transport_edge(
            session_id="session-cancel-inflight",
            edge_id=edge.edge_id,
            transport_outcome=TransportCompletionOutcome.SUCCESS,
            outcome_detail=None,
            cursor=tx,
        )

    session = broadcast_repo.find_session("session-cancel-inflight")
    edge_after = broadcast_repo.find_edge(edge.edge_id)
    targets = broadcast_repo.list_targets("session-cancel-inflight")
    edges = broadcast_repo.list_edges("session-cancel-inflight")

    assert session is not None
    assert edge_after is not None
    assert session.state is BroadcastSessionState.CANCELLED
    assert edge_after.state is BroadcastEdgeState.MATERIALIZING
    assert edge_after.completed_at is None
    assert len(edges) == 1
    materializing_targets = [
        target
        for target in targets
        if target.state is BroadcastTargetState.MATERIALIZING
    ]
    pending_targets = [
        target for target in targets if target.state is BroadcastTargetState.PENDING
    ]
    assert len(materializing_targets) == 1
    assert materializing_targets[0].target_worker_id == claimed_worker_id
    assert materializing_targets[0].assigned_edge_id == edge.edge_id
    assert len(pending_targets) == 1
    assert pending_targets[0].assigned_edge_id is None
