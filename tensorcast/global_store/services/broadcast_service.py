#  Copyright (c) 2026, TensorCast Team.

"""Broadcast session planning service."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID, uuid4

from duckdb import DuckDBPyConnection

from tensorcast.global_store.exceptions import (
    DatabaseError,
    NotFoundError,
    ValidationError,
)
from tensorcast.global_store.models import (
    BroadcastEdge,
    BroadcastEdgeState,
    BroadcastSession,
    BroadcastSessionState,
    BroadcastTarget,
    BroadcastTargetState,
    Replica,
    TransportCompletionOutcome,
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
        target_identities: Sequence[tuple[str, str]] | None = None,
    ) -> BroadcastSession:
        """Create a broadcast session and reserve the first planned edges."""
        session_id = str(session_id).strip()
        artifact_id = str(artifact_id).strip()
        if not session_id:
            raise ValueError("session_id is required")
        if not artifact_id:
            raise ValueError("artifact_id is required")
        if epoch < 0:
            raise ValueError("epoch must be >= 0")
        if fanout <= 0:
            raise ValueError("fanout must be > 0")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be > 0")

        existing = self._broadcast_repository.find_session(session_id)
        if existing is not None:
            return existing

        targets = self._resolve_targets(
            target_worker_ids=target_worker_ids or (),
            target_daemon_ids=target_daemon_ids,
            target_identities=target_identities or (),
        )
        root_replica: Replica | None = None
        selected_root = False
        try:
            try:
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
                with self._broadcast_repository.transaction() as tx:
                    existing = self._broadcast_repository.find_session(
                        session_id,
                        cursor=tx,
                    )
                    if existing is not None:
                        return existing
                    self._broadcast_repository.create_session(session, cursor=tx)
                    for worker in targets:
                        self._broadcast_repository.upsert_target(
                            BroadcastTarget(
                                session_id=session.session_id,
                                target_worker_id=worker.worker_id,
                                target_daemon_id=worker.daemon_id,
                                state=BroadcastTargetState.PENDING,
                            ),
                            cursor=tx,
                        )

                    self._plan_more_edges(session, cursor=tx)

                return session
            except DatabaseError as exc:
                if exc.__cause__ is not None:
                    raise exc.__cause__ from exc
                raise
        finally:
            if selected_root and root_replica is not None:
                self._replica_repository.decrement_requests(root_replica.replica_id)

    def get_session(self, session_id: str) -> BroadcastSession | None:
        """Return a broadcast session by ID."""
        return self._broadcast_repository.find_session(session_id)

    def list_edges(self, session_id: str) -> list[BroadcastEdge]:
        """List broadcast edges for a session."""
        return self._broadcast_repository.list_edges(session_id)

    def list_targets(self, session_id: str) -> list[BroadcastTarget]:
        """List broadcast targets for a session."""
        return self._broadcast_repository.list_targets(session_id)

    def cancel_session(self, session_id: str) -> bool:
        """Mark a broadcast session cancelled."""
        return self._broadcast_repository.update_session_state(
            session_id,
            BroadcastSessionState.CANCELLED,
        )

    def claim_transport_edge(
        self,
        *,
        session_id: str,
        artifact_id: str,
        requested_view_id: str | None,
        requester_worker_id: str,
        request_id: str,
        heartbeat_timeout_seconds: float,
        cursor: DuckDBPyConnection,
    ) -> tuple[Replica, BroadcastEdge]:
        """Claim the planned broadcast parent for one target worker transport."""
        session = self._broadcast_repository.find_session(session_id, cursor=cursor)
        if session is None:
            raise NotFoundError(f"broadcast session not found: {session_id}")
        if session.state is not BroadcastSessionState.ACTIVE:
            raise ValidationError(f"broadcast session is not active: {session_id}")
        if session.artifact_id != artifact_id:
            raise ValidationError("broadcast session artifact does not match request")
        if (session.requested_view_id or "") != (requested_view_id or ""):
            raise ValidationError("broadcast session byte space does not match request")

        target = self._broadcast_repository.find_target(
            session.session_id,
            requester_worker_id,
            cursor=cursor,
        )
        if target is None:
            raise NotFoundError(
                f"broadcast target not found for worker: {requester_worker_id}"
            )
        if target.state is BroadcastTargetState.COMPLETED:
            raise ValidationError("broadcast target is already completed")

        edge = self._broadcast_repository.find_active_edge_for_child(
            session.session_id,
            requester_worker_id,
            cursor=cursor,
        )
        if edge is None:
            self._plan_more_edges(session, cursor=cursor)
            edge = self._broadcast_repository.find_active_edge_for_child(
                session.session_id,
                requester_worker_id,
                cursor=cursor,
            )
        if edge is None:
            raise NotFoundError(
                f"no broadcast edge available for worker: {requester_worker_id}"
            )

        selection = self._replica_repository.claim_replica_for_transport(
            replica_id=edge.parent_replica_id,
            artifact_id=artifact_id,
            view_id=requested_view_id,
            heartbeat_timeout_seconds=heartbeat_timeout_seconds,
            cursor=cursor,
        )
        if selection.replica is None:
            raise NotFoundError("broadcast parent replica is not transport eligible")

        materialized = self._broadcast_repository.mark_edge_materializing(
            edge.edge_id,
            request_id,
            cursor=cursor,
        )
        if not materialized:
            self._replica_repository.decrement_requests_with_cursor(
                selection.replica.replica_id,
                cursor,
            )
            raise ValidationError("broadcast edge is no longer claimable")

        claimed_edge = self._broadcast_repository.find_edge(edge.edge_id, cursor=cursor)
        if claimed_edge is None:
            raise NotFoundError(f"broadcast edge not found: {edge.edge_id}")
        return selection.replica, claimed_edge

    def complete_transport_edge(
        self,
        *,
        session_id: str,
        edge_id: str,
        transport_outcome: TransportCompletionOutcome,
        outcome_detail: str | None,
        cursor: DuckDBPyConnection,
    ) -> None:
        """Advance broadcast edge state from a completed transport outcome."""
        session = self._broadcast_repository.find_session(session_id, cursor=cursor)
        if session is None:
            raise NotFoundError(f"broadcast session not found: {session_id}")
        edge = self._broadcast_repository.find_edge(edge_id, cursor=cursor)
        if edge is None:
            raise NotFoundError(f"broadcast edge not found: {edge_id}")

        if transport_outcome is TransportCompletionOutcome.SUCCESS:
            child_replica = self._replica_repository.find_exportable_replica_for_worker(
                artifact_id=session.artifact_id,
                view_id=session.requested_view_id,
                worker_id=edge.child_worker_id,
                heartbeat_timeout_seconds=self._ROOT_HEARTBEAT_TIMEOUT_SECONDS,
                cursor=cursor,
            )
            if child_replica is None:
                self._broadcast_repository.mark_edge_failed(
                    edge.edge_id,
                    "child_replica_not_exportable_after_success",
                    cursor=cursor,
                )
                return
            self._broadcast_repository.mark_edge_completed(
                edge.edge_id,
                child_replica.replica_id,
                cursor=cursor,
            )
            self._plan_more_edges(session, cursor=cursor)
            self._mark_session_complete_if_done(session.session_id, cursor=cursor)
            return

        reason = (
            outcome_detail or transport_outcome.value or "transport_failed"
        ).strip()
        if not reason:
            reason = "transport_failed"
        self._broadcast_repository.mark_edge_failed(
            edge.edge_id,
            reason,
            cursor=cursor,
        )
        if int(edge.attempt) < int(session.max_attempts):
            target = self._broadcast_repository.find_target(
                edge.session_id,
                edge.child_worker_id,
                cursor=cursor,
            )
            if target is not None:
                target.state = BroadcastTargetState.PENDING
                target.assigned_edge_id = None
                target.completed_replica_id = None
                target.completed_at = None
                self._broadcast_repository.upsert_target(target, cursor=cursor)
                self._plan_more_edges(session, cursor=cursor)

    def _mark_session_complete_if_done(self, session_id: str, *, cursor=None) -> None:
        if self._broadcast_repository.count_incomplete_targets(
            session_id,
            cursor=cursor,
        ) == 0:
            self._broadcast_repository.update_session_state(
                session_id,
                BroadcastSessionState.COMPLETED,
                cursor=cursor,
            )

    def _resolve_targets(
        self,
        *,
        target_worker_ids: list[str] | tuple[str, ...],
        target_daemon_ids: list[str] | tuple[str, ...],
        target_identities: Sequence[tuple[str, str]],
    ) -> list[Worker]:
        targets: dict[str, Worker] = {}
        for raw_worker_id, raw_daemon_id in target_identities:
            worker_id = str(raw_worker_id).strip()
            daemon_id = str(raw_daemon_id).strip()
            if not worker_id and not daemon_id:
                continue

            worker_by_id: Worker | None = None
            worker_by_daemon: Worker | None = None
            if worker_id:
                worker_by_id = self._worker_repository.find_by_id(
                    worker_id,
                    include_inactive=False,
                )
                if worker_by_id is None:
                    raise ValueError(f"target worker not found: {worker_id}")
            if daemon_id:
                worker_by_daemon = self._worker_repository.find_by_daemon_id(
                    daemon_id,
                    include_inactive=False,
                )
                if worker_by_daemon is None:
                    raise ValueError(f"target daemon not found: {daemon_id}")
            if (
                worker_by_id is not None
                and worker_by_daemon is not None
                and worker_by_id.worker_id != worker_by_daemon.worker_id
            ):
                raise ValueError(
                    "target worker_id and daemon_id resolve to different workers"
                )
            worker = worker_by_id or worker_by_daemon
            if worker is not None:
                targets[worker.worker_id] = worker

        for worker_id in target_worker_ids:
            worker_id = str(worker_id).strip()
            if not worker_id:
                continue
            worker = self._worker_repository.find_by_id(
                worker_id,
                include_inactive=False,
            )
            if worker is None:
                raise ValueError(f"target worker not found: {worker_id}")
            targets[worker.worker_id] = worker

        for daemon_id in target_daemon_ids:
            daemon_id = str(daemon_id).strip()
            if not daemon_id:
                continue
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

    def _plan_more_edges(
        self,
        session: BroadcastSession,
        *,
        cursor=None,
    ) -> list[BroadcastEdge]:
        pending_targets = self._broadcast_repository.list_targets_by_state(
            session.session_id,
            BroadcastTargetState.PENDING,
            limit=max(0, int(session.fanout)),
            cursor=cursor,
        )
        if not pending_targets or session.root_replica_id is None:
            return []

        active_edges_count = self._count_active_edges(session.session_id, cursor=cursor)
        capacity = max(0, int(session.fanout) - active_edges_count)
        if capacity <= 0:
            return []

        parent_pool = self._parent_pool(session, cursor=cursor)
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
            self._broadcast_repository.create_edge(edge, cursor=cursor)
            target.state = BroadcastTargetState.ASSIGNED
            target.level = edge.level
            target.attempt = edge.attempt
            target.assigned_edge_id = edge.edge_id
            self._broadcast_repository.upsert_target(target, cursor=cursor)
            planned.append(edge)
        return planned

    def _count_active_edges(self, session_id: str, *, cursor=None) -> int:
        owns_cursor = cursor is None
        if owns_cursor:
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
            if owns_cursor:
                cursor.close()

    def _parent_pool(
        self,
        session: BroadcastSession,
        *,
        cursor=None,
    ) -> list[tuple[Replica, int]]:
        parents: list[tuple[Replica, int]] = []
        if session.root_replica_id is not None:
            root = self._replica_repository.find_by_id(
                session.root_replica_id,
                session.artifact_id,
                cursor=cursor,
            )
            if root is not None:
                parents.append((root, 0))

        completed_targets = self._broadcast_repository.list_targets_by_state(
            session.session_id,
            BroadcastTargetState.COMPLETED,
            limit=10_000,
            cursor=cursor,
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
