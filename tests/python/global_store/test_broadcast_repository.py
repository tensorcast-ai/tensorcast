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
)
from tensorcast.global_store.repositories import BroadcastRepository


def test_broadcast_models_default_to_initial_states():
    session = BroadcastSession(
        session_id="session-a",
        artifact_id="mi2:test",
        requested_view_id=None,
        epoch=1,
        fanout=2,
        max_attempts=3,
        strict_parent=True,
    )
    target = BroadcastTarget(
        session_id="session-a",
        target_worker_id="worker-child",
        target_daemon_id="daemon-child",
    )
    edge = BroadcastEdge(
        edge_id="edge-1",
        session_id="session-a",
        parent_worker_id="worker-root",
        parent_replica_id=UUID("00000000-0000-0000-0000-000000000001"),
        child_worker_id="worker-child",
        level=1,
    )

    assert session.state is BroadcastSessionState.PLANNING
    assert target.state is BroadcastTargetState.PENDING
    assert edge.attempt == 1
    assert edge.state is BroadcastEdgeState.PLANNED


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

    with pytest.raises(ValueError, match="active"):
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


def test_broadcast_repository_prevents_materializing_edge_when_child_has_active_edge(
    db_connection,
):
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
    repo.create_edge(
        BroadcastEdge(
            edge_id="edge-active",
            session_id="session-a",
            parent_worker_id="worker-root",
            parent_replica_id=UUID("00000000-0000-0000-0000-000000000001"),
            child_worker_id="worker-child",
            level=1,
            attempt=1,
            state=BroadcastEdgeState.PLANNED,
        )
    )
    repo.upsert_target(
        BroadcastTarget(
            session_id="session-a",
            target_worker_id="worker-child",
            target_daemon_id="daemon-child",
            state=BroadcastTargetState.ASSIGNED,
            assigned_edge_id="edge-contender",
        )
    )
    db_connection.execute(
        """
        INSERT INTO broadcast_edges (
            edge_id, session_id, parent_worker_id, parent_replica_id,
            child_worker_id, level, attempt, state
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "edge-contender",
            "session-a",
            "worker-root",
            "00000000-0000-0000-0000-000000000001",
            "worker-child",
            1,
            2,
            "planned",
        ],
    )

    with pytest.raises(ValueError, match="active"):
        repo.mark_edge_materializing(
            edge_id="edge-contender",
            transport_request_id="transport-request-1",
        )


def test_broadcast_repository_does_not_materialize_terminal_edge(db_connection):
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
            state=BroadcastTargetState.FAILED,
            assigned_edge_id="edge-failed",
        )
    )
    repo.create_edge(
        BroadcastEdge(
            edge_id="edge-failed",
            session_id="session-a",
            parent_worker_id="worker-root",
            parent_replica_id=UUID("00000000-0000-0000-0000-000000000001"),
            child_worker_id="worker-child",
            level=1,
            attempt=1,
            state=BroadcastEdgeState.FAILED,
        )
    )

    assert not repo.mark_edge_materializing(
        edge_id="edge-failed",
        transport_request_id="transport-request-1",
    )
    edge = repo.find_edge("edge-failed")
    target = repo.find_target("session-a", "worker-child")
    assert edge is not None
    assert target is not None
    assert edge.state is BroadcastEdgeState.FAILED
    assert edge.transport_request_id is None
    assert target.state is BroadcastTargetState.FAILED


def test_broadcast_repository_leaves_edge_unchanged_when_target_missing(
    db_connection,
):
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
        )
    )

    assert not repo.mark_edge_completed(
        edge_id="edge-1",
        completed_replica_id=UUID("00000000-0000-0000-0000-000000000002"),
    )
    edge = repo.find_edge("edge-1")
    assert edge is not None
    assert edge.state is BroadcastEdgeState.MATERIALIZING
    assert edge.completed_at is None


@pytest.mark.parametrize("transition", ["failed", "completed"])
def test_broadcast_repository_stale_edge_transition_does_not_clobber_newer_target(
    db_connection,
    transition,
):
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
            state=BroadcastTargetState.ASSIGNED,
            assigned_edge_id="edge-new",
        )
    )
    repo.create_edge(
        BroadcastEdge(
            edge_id="edge-old",
            session_id="session-a",
            parent_worker_id="worker-root",
            parent_replica_id=UUID("00000000-0000-0000-0000-000000000001"),
            child_worker_id="worker-child",
            level=1,
            attempt=1,
            state=BroadcastEdgeState.MATERIALIZING,
        )
    )

    if transition == "failed":
        changed = repo.mark_edge_failed(edge_id="edge-old", reason="source unavailable")
    else:
        changed = repo.mark_edge_completed(
            edge_id="edge-old",
            completed_replica_id=UUID("00000000-0000-0000-0000-000000000002"),
        )

    edge = repo.find_edge("edge-old")
    target = repo.find_target("session-a", "worker-child")
    assert not changed
    assert edge is not None
    assert target is not None
    assert edge.state is BroadcastEdgeState.MATERIALIZING
    assert edge.completed_at is None
    assert target.state is BroadcastTargetState.ASSIGNED
    assert target.assigned_edge_id == "edge-new"


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
