from __future__ import annotations

from uuid import UUID

from tensorcast.global_store.models import (
    BroadcastEdge,
    BroadcastEdgeState,
    BroadcastSession,
    BroadcastSessionState,
    BroadcastTarget,
    BroadcastTargetState,
)
from tensorcast.global_store.repositories import BroadcastRepository


def test_broadcast_repository_creates_session_targets_and_edges(db_connection):
    repo = BroadcastRepository(db_connection)
    session = BroadcastSession(
        session_id="session-a",
        artifact_id="mi2:test",
        requested_view_id=None,
        epoch=42,
        fanout=2,
        max_attempts=3,
        strict_parent=True,
        state=BroadcastSessionState.ACTIVE,
        root_replica_id=UUID("00000000-0000-0000-0000-000000000001"),
    )
    repo.create_session(session)
    repo.upsert_target(
        BroadcastTarget(
            session_id="session-a",
            target_worker_id="worker-child-1",
            target_daemon_id="daemon-child-1",
            state=BroadcastTargetState.PENDING,
        )
    )
    repo.create_edge(
        BroadcastEdge(
            edge_id="edge-1",
            session_id="session-a",
            parent_worker_id="worker-root",
            parent_replica_id=UUID("00000000-0000-0000-0000-000000000001"),
            child_worker_id="worker-child-1",
            level=1,
            attempt=1,
            state=BroadcastEdgeState.PLANNED,
        )
    )

    loaded = repo.find_session("session-a")
    assert loaded is not None
    assert loaded.artifact_id == "mi2:test"
    assert loaded.epoch == 42
    assert loaded.state is BroadcastSessionState.ACTIVE

    target = repo.find_target("session-a", "worker-child-1")
    assert target is not None
    assert target.target_daemon_id == "daemon-child-1"
    assert target.state is BroadcastTargetState.PENDING

    edge = repo.find_active_edge_for_child("session-a", "worker-child-1")
    assert edge is not None
    assert edge.parent_worker_id == "worker-root"
    assert edge.state is BroadcastEdgeState.PLANNED


def test_broadcast_repository_prevents_two_active_edges_for_child(db_connection):
    repo = BroadcastRepository(db_connection)
    repo.create_session(
        BroadcastSession(
            session_id="session-a",
            artifact_id="mi2:test",
            requested_view_id=None,
            epoch=1,
            fanout=1,
            max_attempts=3,
            strict_parent=True,
            state=BroadcastSessionState.ACTIVE,
            root_replica_id=UUID("00000000-0000-0000-0000-000000000001"),
        )
    )
    first = BroadcastEdge(
        edge_id="edge-1",
        session_id="session-a",
        parent_worker_id="worker-root",
        parent_replica_id=UUID("00000000-0000-0000-0000-000000000001"),
        child_worker_id="worker-child",
        level=1,
        attempt=1,
        state=BroadcastEdgeState.PLANNED,
    )
    repo.create_edge(first)

    try:
        repo.create_edge(
            BroadcastEdge(
                edge_id="edge-2",
                session_id="session-a",
                parent_worker_id="worker-root",
                parent_replica_id=UUID("00000000-0000-0000-0000-000000000001"),
                child_worker_id="worker-child",
                level=1,
                attempt=2,
                state=BroadcastEdgeState.ASSIGNED,
            )
        )
    except Exception as exc:  # noqa: BLE001
        assert "active" in str(exc).lower() or "constraint" in str(exc).lower()
    else:
        raise AssertionError("expected active edge uniqueness to reject duplicate child")


def test_broadcast_repository_marks_edge_completed_and_target_completed(db_connection):
    repo = BroadcastRepository(db_connection)
    repo.create_session(
        BroadcastSession(
            session_id="session-a",
            artifact_id="mi2:test",
            requested_view_id=None,
            epoch=1,
            fanout=1,
            max_attempts=3,
            strict_parent=True,
            state=BroadcastSessionState.ACTIVE,
            root_replica_id=UUID("00000000-0000-0000-0000-000000000001"),
        )
    )
    repo.upsert_target(
        BroadcastTarget(
            session_id="session-a",
            target_worker_id="worker-child",
            target_daemon_id="daemon-child",
            state=BroadcastTargetState.MATERIALIZING,
            assigned_edge_id="edge-1",
        )
    )
    repo.create_edge(
        BroadcastEdge(
            edge_id="edge-1",
            session_id="session-a",
            parent_worker_id="worker-root",
            parent_replica_id=UUID("00000000-0000-0000-0000-000000000001"),
            child_worker_id="worker-child",
            level=1,
            attempt=1,
            state=BroadcastEdgeState.MATERIALIZING,
            transport_request_id="transport-request-1",
        )
    )

    completed_replica_id = UUID("00000000-0000-0000-0000-000000000002")
    assert repo.mark_edge_completed(
        edge_id="edge-1",
        completed_replica_id=completed_replica_id,
    )
    edge = repo.find_edge("edge-1")
    target = repo.find_target("session-a", "worker-child")
    assert edge is not None
    assert target is not None
    assert edge.state is BroadcastEdgeState.COMPLETED
    assert target.state is BroadcastTargetState.COMPLETED
    assert target.completed_replica_id == completed_replica_id
