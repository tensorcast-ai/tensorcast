#  Copyright (c) 2026, TensorCast Team.

"""Repository for broadcast state data access."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from duckdb import DuckDBPyConnection

from tensorcast.global_store.models import (
    BroadcastEdge,
    BroadcastEdgeState,
    BroadcastSession,
    BroadcastSessionState,
    BroadcastTarget,
    BroadcastTargetState,
)
from tensorcast.global_store.repositories.base import BaseRepository


class BroadcastRepository(BaseRepository):
    """Repository for managing persistent tree broadcast state."""

    _ACTIVE_EDGE_STATES = (
        BroadcastEdgeState.PLANNED,
        BroadcastEdgeState.ASSIGNED,
        BroadcastEdgeState.MATERIALIZING,
    )
    _SESSION_PROJECTION = (
        "session_id, artifact_id, requested_view_id, epoch, fanout, max_attempts, "
        "strict_parent, state, root_replica_id, created_at, updated_at, completed_at"
    )
    _TARGET_PROJECTION = (
        "session_id, target_worker_id, target_daemon_id, state, level, attempt, "
        "assigned_edge_id, completed_replica_id, failure_reason, created_at, "
        "updated_at, completed_at"
    )
    _EDGE_PROJECTION = (
        "edge_id, session_id, parent_worker_id, parent_replica_id, child_worker_id, "
        "level, attempt, state, transport_request_id, failure_reason, created_at, "
        "updated_at, completed_at"
    )

    def create_session(
        self,
        session: BroadcastSession,
        cursor: DuckDBPyConnection | None = None,
    ) -> BroadcastSession:
        """Create a broadcast session row."""
        normalized_session_id = self._normalize_required_text(session.session_id)
        normalized_artifact_id = self._normalize_required_text(session.artifact_id)
        owns_cursor = cursor is None
        if owns_cursor:
            cursor = self.get_cursor()
        try:
            cursor.execute(
                """
                INSERT INTO broadcast_sessions (
                    session_id, artifact_id, requested_view_id, epoch, fanout,
                    max_attempts, strict_parent, state, root_replica_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    normalized_session_id,
                    normalized_artifact_id,
                    self._normalize_optional_text(session.requested_view_id),
                    int(session.epoch),
                    int(session.fanout),
                    int(session.max_attempts),
                    bool(session.strict_parent),
                    session.state.value,
                    self._uuid_to_text(session.root_replica_id),
                ],
            )
            session.session_id = normalized_session_id
            session.artifact_id = normalized_artifact_id
            return session
        finally:
            if owns_cursor:
                cursor.close()

    def find_session(
        self,
        session_id: str,
        cursor: DuckDBPyConnection | None = None,
    ) -> BroadcastSession | None:
        """Find a broadcast session by ID."""
        normalized_session_id = self._normalize_required_text(session_id)
        owns_cursor = cursor is None
        if owns_cursor:
            cursor = self.get_cursor()
        try:
            query = cursor.execute(
                f"""
                SELECT {self._SESSION_PROJECTION}
                FROM broadcast_sessions
                WHERE session_id = ?
                """,
                [normalized_session_id],
            )
            row = query.fetchone()
            if row is None:
                return None
            assert query.description is not None
            columns = [desc[0] for desc in query.description]
            return self._row_to_session(row, columns)
        finally:
            if owns_cursor:
                cursor.close()

    def update_session_state(
        self,
        session_id: str,
        state: BroadcastSessionState,
        cursor: DuckDBPyConnection | None = None,
    ) -> bool:
        """Update the lifecycle state for a broadcast session."""
        normalized_session_id = self._normalize_required_text(session_id)
        owns_cursor = cursor is None
        if owns_cursor:
            cursor = self.get_cursor()
        try:
            completed_sql = (
                ", completed_at = CURRENT_TIMESTAMP"
                if state
                in (
                    BroadcastSessionState.COMPLETED,
                    BroadcastSessionState.FAILED,
                    BroadcastSessionState.CANCELLED,
                )
                else ""
            )
            row = cursor.execute(
                f"""
                UPDATE broadcast_sessions
                SET state = ?, updated_at = CURRENT_TIMESTAMP {completed_sql}
                WHERE session_id = ?
                RETURNING session_id
                """,
                [state.value, normalized_session_id],
            ).fetchone()
            return row is not None
        finally:
            if owns_cursor:
                cursor.close()

    def upsert_target(
        self,
        target: BroadcastTarget,
        cursor: DuckDBPyConnection | None = None,
    ) -> BroadcastTarget:
        """Insert or update one broadcast target row."""
        normalized_session_id = self._normalize_required_text(target.session_id)
        normalized_target_worker_id = self._normalize_required_text(
            target.target_worker_id
        )
        owns_cursor = cursor is None
        if owns_cursor:
            cursor = self.get_cursor()
        try:
            updated_at = datetime.now()
            cursor.execute(
                """
                INSERT INTO broadcast_targets (
                    session_id, target_worker_id, target_daemon_id, state, level,
                    attempt, assigned_edge_id, completed_replica_id, failure_reason,
                    completed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (session_id, target_worker_id) DO UPDATE SET
                    target_daemon_id = excluded.target_daemon_id,
                    state = excluded.state,
                    level = excluded.level,
                    attempt = excluded.attempt,
                    assigned_edge_id = excluded.assigned_edge_id,
                    completed_replica_id = excluded.completed_replica_id,
                    failure_reason = excluded.failure_reason,
                    updated_at = ?,
                    completed_at = excluded.completed_at
                """,
                [
                    normalized_session_id,
                    normalized_target_worker_id,
                    self._normalize_optional_text(target.target_daemon_id),
                    target.state.value,
                    self._normalize_optional_int(target.level),
                    int(target.attempt),
                    self._normalize_optional_text(target.assigned_edge_id),
                    self._uuid_to_text(target.completed_replica_id),
                    self._normalize_optional_text(target.failure_reason),
                    target.completed_at,
                    updated_at,
                ],
            )
            target.session_id = normalized_session_id
            target.target_worker_id = normalized_target_worker_id
            return target
        finally:
            if owns_cursor:
                cursor.close()

    def find_target(
        self,
        session_id: str,
        target_worker_id: str,
        cursor: DuckDBPyConnection | None = None,
    ) -> BroadcastTarget | None:
        """Find a broadcast target by session and worker ID."""
        normalized_session_id = self._normalize_required_text(session_id)
        normalized_target_worker_id = self._normalize_required_text(target_worker_id)
        owns_cursor = cursor is None
        if owns_cursor:
            cursor = self.get_cursor()
        try:
            query = cursor.execute(
                f"""
                SELECT {self._TARGET_PROJECTION}
                FROM broadcast_targets
                WHERE session_id = ? AND target_worker_id = ?
                """,
                [normalized_session_id, normalized_target_worker_id],
            )
            row = query.fetchone()
            if row is None:
                return None
            assert query.description is not None
            columns = [desc[0] for desc in query.description]
            return self._row_to_target(row, columns)
        finally:
            if owns_cursor:
                cursor.close()

    def list_targets(
        self,
        session_id: str,
        cursor: DuckDBPyConnection | None = None,
    ) -> list[BroadcastTarget]:
        """List all targets for a broadcast session."""
        normalized_session_id = self._normalize_required_text(session_id)
        owns_cursor = cursor is None
        if owns_cursor:
            cursor = self.get_cursor()
        try:
            query = cursor.execute(
                f"""
                SELECT {self._TARGET_PROJECTION}
                FROM broadcast_targets
                WHERE session_id = ?
                ORDER BY target_worker_id ASC
                """,
                [normalized_session_id],
            )
            rows = query.fetchall()
            assert query.description is not None
            columns = [desc[0] for desc in query.description]
            return [self._row_to_target(row, columns) for row in rows]
        finally:
            if owns_cursor:
                cursor.close()

    def list_targets_by_state(
        self,
        session_id: str,
        state: BroadcastTargetState,
        limit: int,
        cursor: DuckDBPyConnection | None = None,
    ) -> list[BroadcastTarget]:
        """List targets for a session with the requested state."""
        normalized_session_id = self._normalize_required_text(session_id)
        owns_cursor = cursor is None
        if owns_cursor:
            cursor = self.get_cursor()
        try:
            query = cursor.execute(
                f"""
                SELECT {self._TARGET_PROJECTION}
                FROM broadcast_targets
                WHERE session_id = ? AND state = ?
                ORDER BY updated_at ASC, target_worker_id ASC
                LIMIT ?
                """,
                [normalized_session_id, state.value, int(limit)],
            )
            rows = query.fetchall()
            assert query.description is not None
            columns = [desc[0] for desc in query.description]
            return [self._row_to_target(row, columns) for row in rows]
        finally:
            if owns_cursor:
                cursor.close()

    def create_edge(
        self,
        edge: BroadcastEdge,
        cursor: DuckDBPyConnection | None = None,
    ) -> BroadcastEdge:
        """Create a broadcast edge row."""
        if cursor is None:
            conflict: ValueError | None = None
            with self.transaction() as tx:
                try:
                    return self.create_edge(edge, cursor=tx)
                except ValueError as exc:
                    conflict = exc
            if conflict is not None:
                raise conflict
            raise RuntimeError("broadcast edge transaction exited without result")

        normalized_edge_id = self._normalize_required_text(edge.edge_id)
        normalized_session_id = self._normalize_required_text(edge.session_id)
        normalized_child_worker_id = self._normalize_required_text(edge.child_worker_id)

        if edge.state in self._ACTIVE_EDGE_STATES:
            existing = self.find_active_edge_for_child(
                normalized_session_id,
                normalized_child_worker_id,
                cursor=cursor,
            )
            if existing is not None:
                raise ValueError(
                    "active broadcast edge already exists for child "
                    f"{normalized_child_worker_id} in session {normalized_session_id}"
                )

        cursor.execute(
            """
            INSERT INTO broadcast_edges (
                edge_id, session_id, parent_worker_id, parent_replica_id,
                child_worker_id, level, attempt, state, transport_request_id,
                failure_reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                normalized_edge_id,
                normalized_session_id,
                self._normalize_required_text(edge.parent_worker_id),
                self._uuid_to_text(edge.parent_replica_id),
                normalized_child_worker_id,
                int(edge.level),
                int(edge.attempt),
                edge.state.value,
                self._normalize_optional_text(edge.transport_request_id),
                self._normalize_optional_text(edge.failure_reason),
            ],
        )
        edge.edge_id = normalized_edge_id
        edge.session_id = normalized_session_id
        edge.child_worker_id = normalized_child_worker_id
        return edge

    def find_edge(
        self,
        edge_id: str,
        cursor: DuckDBPyConnection | None = None,
    ) -> BroadcastEdge | None:
        """Find a broadcast edge by ID."""
        normalized_edge_id = self._normalize_required_text(edge_id)
        owns_cursor = cursor is None
        if owns_cursor:
            cursor = self.get_cursor()
        try:
            query = cursor.execute(
                f"""
                SELECT {self._EDGE_PROJECTION}
                FROM broadcast_edges
                WHERE edge_id = ?
                """,
                [normalized_edge_id],
            )
            row = query.fetchone()
            if row is None:
                return None
            assert query.description is not None
            columns = [desc[0] for desc in query.description]
            return self._row_to_edge(row, columns)
        finally:
            if owns_cursor:
                cursor.close()

    def list_edges(
        self,
        session_id: str,
        cursor: DuckDBPyConnection | None = None,
    ) -> list[BroadcastEdge]:
        """List all broadcast edges for a session."""
        normalized_session_id = self._normalize_required_text(session_id)
        owns_cursor = cursor is None
        if owns_cursor:
            cursor = self.get_cursor()
        try:
            query = cursor.execute(
                f"""
                SELECT {self._EDGE_PROJECTION}
                FROM broadcast_edges
                WHERE session_id = ?
                ORDER BY level ASC, created_at ASC, edge_id ASC
                """,
                [normalized_session_id],
            )
            rows = query.fetchall()
            assert query.description is not None
            columns = [desc[0] for desc in query.description]
            return [self._row_to_edge(row, columns) for row in rows]
        finally:
            if owns_cursor:
                cursor.close()

    def find_active_edge_for_child(
        self,
        session_id: str,
        child_worker_id: str,
        cursor: DuckDBPyConnection | None = None,
    ) -> BroadcastEdge | None:
        """Find an active edge for a session/child worker pair."""
        normalized_session_id = self._normalize_required_text(session_id)
        normalized_child_worker_id = self._normalize_required_text(child_worker_id)
        owns_cursor = cursor is None
        if owns_cursor:
            cursor = self.get_cursor()
        try:
            query = cursor.execute(
                f"""
                SELECT {self._EDGE_PROJECTION}
                FROM broadcast_edges
                WHERE session_id = ?
                  AND child_worker_id = ?
                  AND state IN ('planned', 'assigned', 'materializing')
                ORDER BY attempt DESC, updated_at DESC
                LIMIT 1
                """,
                [normalized_session_id, normalized_child_worker_id],
            )
            row = query.fetchone()
            if row is None:
                return None
            assert query.description is not None
            columns = [desc[0] for desc in query.description]
            return self._row_to_edge(row, columns)
        finally:
            if owns_cursor:
                cursor.close()

    def mark_edge_materializing(
        self,
        edge_id: str,
        transport_request_id: str,
        cursor: DuckDBPyConnection | None = None,
    ) -> bool:
        """Mark an edge and its target as materializing."""
        if cursor is None:
            conflict: ValueError | None = None
            with self.transaction() as tx:
                try:
                    return self.mark_edge_materializing(
                        edge_id=edge_id,
                        transport_request_id=transport_request_id,
                        cursor=tx,
                    )
                except ValueError as exc:
                    conflict = exc
            if conflict is not None:
                raise conflict
            raise RuntimeError("broadcast edge transaction exited without result")

        normalized_edge_id = self._normalize_required_text(edge_id)
        normalized_transport_request_id = self._normalize_required_text(
            transport_request_id
        )
        edge = self.find_edge(normalized_edge_id, cursor=cursor)
        if edge is None:
            return False
        if edge.state not in (BroadcastEdgeState.PLANNED, BroadcastEdgeState.ASSIGNED):
            return False

        existing = cursor.execute(
            """
            SELECT edge_id
            FROM broadcast_edges
            WHERE session_id = ?
              AND child_worker_id = ?
              AND edge_id != ?
              AND state IN ('planned', 'assigned', 'materializing')
            LIMIT 1
            """,
            [edge.session_id, edge.child_worker_id, edge.edge_id],
        ).fetchone()
        if existing is not None:
            raise ValueError(
                "active broadcast edge already exists for child "
                f"{edge.child_worker_id} in session {edge.session_id}"
            )

        target = self.find_target(edge.session_id, edge.child_worker_id, cursor=cursor)
        if target is None:
            return False
        if (
            target.assigned_edge_id is not None
            and target.assigned_edge_id != edge.edge_id
        ):
            return False

        edge_row = cursor.execute(
            """
            UPDATE broadcast_edges
            SET state = 'materializing',
                transport_request_id = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE edge_id = ?
              AND state IN ('planned', 'assigned')
            RETURNING edge_id
            """,
            [normalized_transport_request_id, normalized_edge_id],
        ).fetchone()
        if edge_row is None:
            return False

        target_row = cursor.execute(
            """
            UPDATE broadcast_targets
            SET state = 'materializing',
                assigned_edge_id = ?,
                level = ?,
                attempt = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE session_id = ?
              AND target_worker_id = ?
              AND (assigned_edge_id IS NULL OR assigned_edge_id = ?)
            RETURNING target_worker_id
            """,
            [
                normalized_edge_id,
                int(edge.level),
                int(edge.attempt),
                edge.session_id,
                edge.child_worker_id,
                normalized_edge_id,
            ],
        ).fetchone()
        return target_row is not None

    def mark_edge_failed(
        self,
        edge_id: str,
        reason: str,
        cursor: DuckDBPyConnection | None = None,
    ) -> bool:
        """Mark an edge and its target as failed."""
        if cursor is None:
            with self.transaction() as tx:
                return self.mark_edge_failed(
                    edge_id=edge_id,
                    reason=reason,
                    cursor=tx,
                )

        normalized_edge_id = self._normalize_required_text(edge_id)
        normalized_reason = self._normalize_required_text(reason)
        edge = self.find_edge(normalized_edge_id, cursor=cursor)
        if edge is None:
            return False
        if edge.state is not BroadcastEdgeState.MATERIALIZING:
            return False
        target = self.find_target(edge.session_id, edge.child_worker_id, cursor=cursor)
        if target is None or target.assigned_edge_id != edge.edge_id:
            return False

        row = cursor.execute(
            """
            UPDATE broadcast_edges
            SET state = 'failed',
                failure_reason = ?,
                updated_at = CURRENT_TIMESTAMP,
                completed_at = CURRENT_TIMESTAMP
            WHERE edge_id = ?
              AND state = 'materializing'
            RETURNING edge_id
            """,
            [normalized_reason, normalized_edge_id],
        ).fetchone()
        if row is None:
            return False

        target_row = cursor.execute(
            """
            UPDATE broadcast_targets
            SET state = 'failed',
                failure_reason = ?,
                assigned_edge_id = ?,
                updated_at = CURRENT_TIMESTAMP,
                completed_at = CURRENT_TIMESTAMP
            WHERE session_id = ?
              AND target_worker_id = ?
              AND assigned_edge_id = ?
            RETURNING target_worker_id
            """,
            [
                normalized_reason,
                normalized_edge_id,
                edge.session_id,
                edge.child_worker_id,
                normalized_edge_id,
            ],
        ).fetchone()
        return target_row is not None

    def mark_edge_completed(
        self,
        edge_id: str,
        completed_replica_id: UUID | None,
        cursor: DuckDBPyConnection | None = None,
    ) -> bool:
        """Mark an edge and its target as completed."""
        if cursor is None:
            with self.transaction() as tx:
                return self.mark_edge_completed(
                    edge_id=edge_id,
                    completed_replica_id=completed_replica_id,
                    cursor=tx,
                )

        normalized_edge_id = self._normalize_required_text(edge_id)
        edge = self.find_edge(normalized_edge_id, cursor=cursor)
        if edge is None:
            return False
        if edge.state is not BroadcastEdgeState.MATERIALIZING:
            return False
        target = self.find_target(edge.session_id, edge.child_worker_id, cursor=cursor)
        if target is None or target.assigned_edge_id != edge.edge_id:
            return False

        row = cursor.execute(
            """
            UPDATE broadcast_edges
            SET state = 'completed',
                updated_at = CURRENT_TIMESTAMP,
                completed_at = CURRENT_TIMESTAMP
            WHERE edge_id = ?
              AND state = 'materializing'
            RETURNING edge_id
            """,
            [normalized_edge_id],
        ).fetchone()
        if row is None:
            return False

        target_row = cursor.execute(
            """
            UPDATE broadcast_targets
            SET state = 'completed',
                completed_replica_id = ?,
                assigned_edge_id = ?,
                updated_at = CURRENT_TIMESTAMP,
                completed_at = CURRENT_TIMESTAMP
            WHERE session_id = ?
              AND target_worker_id = ?
              AND assigned_edge_id = ?
            RETURNING target_worker_id
            """,
            [
                self._uuid_to_text(completed_replica_id),
                normalized_edge_id,
                edge.session_id,
                edge.child_worker_id,
                normalized_edge_id,
            ],
        ).fetchone()
        return target_row is not None

    def count_incomplete_targets(
        self,
        session_id: str,
        cursor: DuckDBPyConnection | None = None,
    ) -> int:
        """Count non-terminal targets for a broadcast session."""
        normalized_session_id = self._normalize_required_text(session_id)
        owns_cursor = cursor is None
        if owns_cursor:
            cursor = self.get_cursor()
        try:
            row = cursor.execute(
                """
                SELECT COUNT(*)
                FROM broadcast_targets
                WHERE session_id = ?
                  AND state NOT IN ('completed', 'failed', 'cancelled')
                """,
                [normalized_session_id],
            ).fetchone()
            return int(row[0]) if row is not None else 0
        finally:
            if owns_cursor:
                cursor.close()

    @staticmethod
    def _normalize_optional_text(value: str | None) -> str | None:
        return BroadcastRepository._normalize_text_token(value)

    @staticmethod
    def _normalize_required_text(value: str | None) -> str:
        normalized = BroadcastRepository._normalize_text_token(value)
        if normalized is None:
            raise ValueError("required broadcast field is missing")
        return normalized

    @staticmethod
    def _normalize_optional_int(value: int | None) -> int | None:
        if value is None:
            return None
        return int(value)

    @staticmethod
    def _normalize_text_token(value: str | None) -> str | None:
        if value is None:
            return None
        stripped = str(value).strip()
        if not stripped:
            return None
        return BroadcastRepository._collapse_exact_double(stripped)

    @staticmethod
    def _collapse_exact_double(value: str) -> str:
        size = len(value)
        if size < 8 or (size % 2) != 0:
            return value
        half = size // 2
        if value[:half] != value[half:]:
            return value
        return value[:half]

    @staticmethod
    def _uuid_to_text(value: UUID | None) -> str | None:
        if value is None:
            return None
        return str(value)

    @staticmethod
    def _uuid_or_none(raw: Any) -> UUID | None:
        if raw is None:
            return None
        if isinstance(raw, UUID):
            return raw
        return UUID(str(raw))

    @staticmethod
    def _coerce_datetime_optional(raw: Any) -> datetime | None:
        if raw is None:
            return None
        if isinstance(raw, datetime):
            return raw
        return datetime.fromisoformat(str(raw))

    @classmethod
    def _row_to_session(
        cls,
        row: tuple[Any, ...],
        columns: list[str],
    ) -> BroadcastSession:
        idx = {column: i for i, column in enumerate(columns)}
        return BroadcastSession(
            session_id=str(row[idx["session_id"]]),
            artifact_id=str(row[idx["artifact_id"]]),
            requested_view_id=cls._normalize_optional_text(
                row[idx["requested_view_id"]]
            ),
            epoch=int(row[idx["epoch"]]),
            fanout=int(row[idx["fanout"]]),
            max_attempts=int(row[idx["max_attempts"]]),
            strict_parent=bool(row[idx["strict_parent"]]),
            state=BroadcastSessionState(str(row[idx["state"]])),
            root_replica_id=cls._uuid_or_none(row[idx["root_replica_id"]]),
            created_at=cls._coerce_datetime_optional(row[idx["created_at"]]),
            updated_at=cls._coerce_datetime_optional(row[idx["updated_at"]]),
            completed_at=cls._coerce_datetime_optional(row[idx["completed_at"]]),
        )

    @classmethod
    def _row_to_target(
        cls,
        row: tuple[Any, ...],
        columns: list[str],
    ) -> BroadcastTarget:
        idx = {column: i for i, column in enumerate(columns)}
        raw_level = row[idx["level"]]
        return BroadcastTarget(
            session_id=str(row[idx["session_id"]]),
            target_worker_id=str(row[idx["target_worker_id"]]),
            target_daemon_id=cls._normalize_optional_text(row[idx["target_daemon_id"]]),
            state=BroadcastTargetState(str(row[idx["state"]])),
            level=int(raw_level) if raw_level is not None else None,
            attempt=int(row[idx["attempt"]]),
            assigned_edge_id=cls._normalize_optional_text(row[idx["assigned_edge_id"]]),
            completed_replica_id=cls._uuid_or_none(row[idx["completed_replica_id"]]),
            failure_reason=cls._normalize_optional_text(row[idx["failure_reason"]]),
            created_at=cls._coerce_datetime_optional(row[idx["created_at"]]),
            updated_at=cls._coerce_datetime_optional(row[idx["updated_at"]]),
            completed_at=cls._coerce_datetime_optional(row[idx["completed_at"]]),
        )

    @classmethod
    def _row_to_edge(
        cls,
        row: tuple[Any, ...],
        columns: list[str],
    ) -> BroadcastEdge:
        idx = {column: i for i, column in enumerate(columns)}
        parent_replica_id = cls._uuid_or_none(row[idx["parent_replica_id"]])
        if parent_replica_id is None:
            raise ValueError("broadcast edge parent_replica_id is missing")
        return BroadcastEdge(
            edge_id=str(row[idx["edge_id"]]),
            session_id=str(row[idx["session_id"]]),
            parent_worker_id=str(row[idx["parent_worker_id"]]),
            parent_replica_id=parent_replica_id,
            child_worker_id=str(row[idx["child_worker_id"]]),
            level=int(row[idx["level"]]),
            attempt=int(row[idx["attempt"]]),
            state=BroadcastEdgeState(str(row[idx["state"]])),
            transport_request_id=cls._normalize_optional_text(
                row[idx["transport_request_id"]]
            ),
            failure_reason=cls._normalize_optional_text(row[idx["failure_reason"]]),
            created_at=cls._coerce_datetime_optional(row[idx["created_at"]]),
            updated_at=cls._coerce_datetime_optional(row[idx["updated_at"]]),
            completed_at=cls._coerce_datetime_optional(row[idx["completed_at"]]),
        )
