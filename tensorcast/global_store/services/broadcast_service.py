#  Copyright (c) 2026, TensorCast Team.

"""Broadcast session planning service."""

from __future__ import annotations

from uuid import UUID, uuid4

from tensorcast.global_store.models import (
    BroadcastEdge,
    BroadcastEdgeState,
    BroadcastSession,
    BroadcastSessionState,
    BroadcastTarget,
    BroadcastTargetState,
    Replica,
    Worker,
)
from tensorcast.global_store.repositories import (
    BroadcastRepository,
    ReplicaRepository,
    WorkerRepository,
)


class BroadcastService:
    """Coordinates broadcast session topology state."""

    _ROOT_HEARTBEAT_TIMEOUT_SECONDS = 5.0

    def __init__(
        self,
        *,
        broadcast_repository: BroadcastRepository,
        replica_repository: ReplicaRepository,
        worker_repository: WorkerRepository,
    ) -> None:
        self._broadcast_repository = broadcast_repository
        self._replica_repository = replica_repository
        self._worker_repository = worker_repository

    def create_session(
        self,
        *,
        session_id: str,
        artifact_id: str,
        requested_view_id: str | None,
        epoch: int,
        fanout: int,
        target_daemon_ids: list[str] | tuple[str, ...],
        root_replica_id: str | None,
        strict_parent: bool,
        max_attempts: int,
        target_worker_ids: list[str] | tuple[str, ...] | None = None,
    ) -> BroadcastSession:
        """Create a broadcast session and reserve the first planned edges."""
        if fanout <= 0:
            raise ValueError("fanout must be > 0")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be > 0")

        targets = self._resolve_targets(
            target_worker_ids=target_worker_ids or (),
            target_daemon_ids=target_daemon_ids,
        )
        root_replica, selected_root = self._resolve_root_replica(
            artifact_id=artifact_id,
            requested_view_id=requested_view_id,
            root_replica_id=root_replica_id,
        )

        session = BroadcastSession(
            session_id=session_id,
            artifact_id=artifact_id,
            requested_view_id=requested_view_id,
            epoch=int(epoch),
            fanout=int(fanout),
            max_attempts=int(max_attempts),
            strict_parent=bool(strict_parent),
            state=BroadcastSessionState.ACTIVE,
            root_replica_id=root_replica.replica_id,
        )
        self._broadcast_repository.create_session(session)

        for worker in targets:
            self._broadcast_repository.upsert_target(
                BroadcastTarget(
                    session_id=session.session_id,
                    target_worker_id=worker.worker_id,
                    target_daemon_id=worker.daemon_id,
                    state=BroadcastTargetState.PENDING,
                )
            )

        self._plan_more_edges(session)

        if selected_root:
            self._replica_repository.decrement_requests(root_replica.replica_id)

        return session

    def get_session(self, session_id: str) -> BroadcastSession | None:
        """Return a broadcast session by ID."""
        return self._broadcast_repository.find_session(session_id)

    def list_edges(self, session_id: str) -> list[BroadcastEdge]:
        """List broadcast edges for a session."""
        cursor = self._broadcast_repository.get_cursor()
        try:
            query = cursor.execute(
                f"""
                SELECT {self._broadcast_repository._EDGE_PROJECTION}
                FROM broadcast_edges
                WHERE session_id = ?
                ORDER BY level ASC, created_at ASC, edge_id ASC
                """,
                [session_id],
            )
            rows = query.fetchall()
            assert query.description is not None
            columns = [desc[0] for desc in query.description]
            return [
                self._broadcast_repository._row_to_edge(row, columns) for row in rows
            ]
        finally:
            cursor.close()

    def cancel_session(self, session_id: str) -> bool:
        """Mark a broadcast session cancelled."""
        return self._broadcast_repository.update_session_state(
            session_id,
            BroadcastSessionState.CANCELLED,
        )

    def _resolve_targets(
        self,
        *,
        target_worker_ids: list[str] | tuple[str, ...],
        target_daemon_ids: list[str] | tuple[str, ...],
    ) -> list[Worker]:
        targets: dict[str, Worker] = {}
        for worker_id in target_worker_ids:
            worker = self._worker_repository.find_by_id(
                worker_id,
                include_inactive=False,
            )
            if worker is None:
                raise ValueError(f"target worker not found: {worker_id}")
            targets[worker.worker_id] = worker

        for daemon_id in target_daemon_ids:
            worker = self._worker_repository.find_by_daemon_id(
                daemon_id,
                include_inactive=False,
            )
            if worker is None:
                raise ValueError(f"target daemon not found: {daemon_id}")
            targets[worker.worker_id] = worker

        if not targets:
            raise ValueError("at least one broadcast target is required")
        return list(targets.values())

    def _resolve_root_replica(
        self,
        *,
        artifact_id: str,
        requested_view_id: str | None,
        root_replica_id: str | None,
    ) -> tuple[Replica, bool]:
        normalized_root_id = (root_replica_id or "").strip()
        if normalized_root_id:
            replica = self._replica_repository.find_by_id(
                UUID(normalized_root_id),
                artifact_id,
            )
            if replica is None:
                raise ValueError(f"root replica not found: {normalized_root_id}")
            return replica, False

        result = self._replica_repository.find_available_for_transport(
            artifact_id=artifact_id,
            heartbeat_timeout_seconds=self._ROOT_HEARTBEAT_TIMEOUT_SECONDS,
            view_id=requested_view_id,
        )
        if result.replica is None:
            raise ValueError(f"no available root replica for artifact: {artifact_id}")
        return result.replica, True

    def _plan_more_edges(self, session: BroadcastSession) -> list[BroadcastEdge]:
        pending_targets = self._broadcast_repository.list_targets_by_state(
            session.session_id,
            BroadcastTargetState.PENDING,
            limit=max(0, int(session.fanout)),
        )
        if not pending_targets or session.root_replica_id is None:
            return []

        active_edges_count = self._count_active_edges(session.session_id)
        capacity = max(0, int(session.fanout) - active_edges_count)
        if capacity <= 0:
            return []

        parent_pool = self._parent_pool(session)
        if not parent_pool:
            return []

        planned: list[BroadcastEdge] = []
        for target, parent in zip(pending_targets[:capacity], parent_pool * capacity):
            parent_replica, parent_level = parent
            edge = BroadcastEdge(
                edge_id=str(uuid4()),
                session_id=session.session_id,
                parent_worker_id=parent_replica.worker_id or "",
                parent_replica_id=parent_replica.replica_id,
                child_worker_id=target.target_worker_id,
                level=parent_level + 1,
                attempt=target.attempt + 1,
                state=BroadcastEdgeState.PLANNED,
            )
            self._broadcast_repository.create_edge(edge)
            target.state = BroadcastTargetState.ASSIGNED
            target.level = edge.level
            target.attempt = edge.attempt
            target.assigned_edge_id = edge.edge_id
            self._broadcast_repository.upsert_target(target)
            planned.append(edge)
        return planned

    def _count_active_edges(self, session_id: str) -> int:
        cursor = self._broadcast_repository.get_cursor()
        try:
            row = cursor.execute(
                """
                SELECT COUNT(*)
                FROM broadcast_edges
                WHERE session_id = ?
                  AND state IN ('planned', 'assigned', 'materializing')
                """,
                [session_id],
            ).fetchone()
            return int(row[0]) if row is not None else 0
        finally:
            cursor.close()

    def _parent_pool(
        self,
        session: BroadcastSession,
    ) -> list[tuple[Replica, int]]:
        parents: list[tuple[Replica, int]] = []
        if session.root_replica_id is not None:
            root = self._replica_repository.find_by_id(
                session.root_replica_id,
                session.artifact_id,
            )
            if root is not None:
                parents.append((root, 0))

        completed_targets = self._broadcast_repository.list_targets_by_state(
            session.session_id,
            BroadcastTargetState.COMPLETED,
            limit=10_000,
        )
        for target in completed_targets:
            if target.completed_replica_id is None:
                continue
            replica = self._replica_repository.find_by_replica_id(
                target.completed_replica_id
            )
            if replica is not None:
                parents.append((replica, int(target.level or 0)))
        return parents
