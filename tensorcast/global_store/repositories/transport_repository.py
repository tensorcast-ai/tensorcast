#  Copyright (c) 2025-2026, TensorCast Team.

"""Repository for transport data access."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from tensorcast.global_store.models import Transport, TransportCompletionOutcome
from tensorcast.global_store.repositories.base import BaseRepository


@dataclass(frozen=True)
class TransportGroupProgress:
    """Aggregate completion snapshot for one scheduling group."""

    completed_parts: int
    total_parts: int
    last_success_at: datetime | None

    @property
    def completion_ratio(self) -> float:
        if self.total_parts <= 0:
            return 0.0
        return min(1.0, float(self.completed_parts) / float(self.total_parts))


@dataclass(frozen=True)
class TransportWindowRow:
    transport_id: str
    replica_id: str
    artifact_id: str
    status: str
    completion_outcome: str
    request_id: str
    requester_worker_id: str
    group_id: str
    group_kind: str
    group_part_id: str
    group_total_parts: int
    created_at: datetime
    completed_at: datetime | None
    replica_memory_size_bytes: int


class TransportRepository(BaseRepository):
    """Repository for managing transports in the database."""

    _TRANSPORT_PROJECTION = (
        "transport_id, replica_id, artifact_id, requested_view_id, disk_path, "
        "source_node_id, source_address, source_port, request_id, request_fingerprint, "
        "requester_worker_id, group_id, group_kind, group_total_parts, "
        "group_part_id, group_priority, group_epoch, completion_outcome, "
        "completion_detail, created_at, completed_at, status"
    )

    def find_by_id(self, transport_id: UUID, cursor=None) -> Transport | None:
        """Find a transport by ID."""
        owns_cursor = cursor is None
        if owns_cursor:
            cursor = self.get_cursor()
        try:
            query = cursor.execute(
                f"SELECT {self._TRANSPORT_PROJECTION} FROM artifact_transports WHERE transport_id = ?",
                [str(transport_id)],
            )
            row = query.fetchone()
            if row is None:
                return None
            assert query.description is not None
            columns = [desc[0] for desc in query.description]
            return self._row_to_model(row, columns)
        finally:
            if owns_cursor:
                cursor.close()

    def find_by_request_id(self, request_id: str, cursor=None) -> Transport | None:
        """Find a transport by request idempotency key."""
        normalized = request_id.strip()
        if not normalized:
            return None
        owns_cursor = cursor is None
        if owns_cursor:
            cursor = self.get_cursor()
        try:
            query = cursor.execute(
                f"""
                SELECT {self._TRANSPORT_PROJECTION}
                FROM artifact_transports
                WHERE request_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                [normalized],
            )
            row = query.fetchone()
            if row is None:
                return None
            assert query.description is not None
            columns = [desc[0] for desc in query.description]
            return self._row_to_model(row, columns)
        finally:
            if owns_cursor:
                cursor.close()

    def create(self, transport: Transport) -> Transport:
        """Create a new transport record."""
        cursor = self.get_cursor()
        try:
            self.create_with_cursor(transport, cursor)
            return transport
        finally:
            cursor.close()

    def create_with_cursor(self, transport: Transport, cursor) -> Transport:
        """Create a new transport record using an existing cursor."""
        normalized_request_id = self._normalize_request_id(transport.request_id)
        cursor.execute(
            """
            INSERT INTO artifact_transports (
                transport_id, replica_id, artifact_id, requested_view_id, disk_path,
                source_node_id, source_address, source_port,
                request_id, request_fingerprint, requester_worker_id,
                group_id, group_kind, group_total_parts, group_part_id,
                group_priority, group_epoch,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                str(transport.transport_id),
                str(transport.replica_id),
                transport.artifact_id,
                self._normalize_optional_text(transport.requested_view_id),
                transport.disk_path,
                transport.source_node_id,
                transport.source_address,
                int(transport.source_port),
                normalized_request_id,
                self._normalize_optional_text(transport.request_fingerprint),
                self._normalize_optional_text(transport.requester_worker_id),
                self._normalize_optional_text(transport.group_id),
                self._normalize_optional_text(transport.group_kind),
                self._normalize_optional_int(transport.group_total_parts),
                self._normalize_optional_text(transport.group_part_id),
                self._normalize_optional_int(transport.group_priority),
                self._normalize_optional_int(transport.group_epoch),
                "in_progress",
            ],
        )
        transport.request_id = normalized_request_id
        transport.status = "in_progress"
        transport.completion_outcome = TransportCompletionOutcome.UNSPECIFIED
        transport.completion_detail = None
        return transport

    def create_if_absent_with_cursor(
        self, transport: Transport, cursor
    ) -> tuple[Transport, bool]:
        """
        Insert transport by request_id idempotency key if absent.

        Returns (transport_row, created_now).
        """
        normalized_request_id = self._normalize_request_id(transport.request_id)
        if normalized_request_id is None:
            created = self.create_with_cursor(transport, cursor)
            return created, True

        cursor.execute(
            """
            INSERT INTO artifact_transports (
                transport_id, replica_id, artifact_id, requested_view_id, disk_path,
                source_node_id, source_address, source_port,
                request_id, request_fingerprint, requester_worker_id,
                group_id, group_kind, group_total_parts, group_part_id,
                group_priority, group_epoch,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (request_id) DO NOTHING
            """,
            [
                str(transport.transport_id),
                str(transport.replica_id),
                transport.artifact_id,
                self._normalize_optional_text(transport.requested_view_id),
                transport.disk_path,
                transport.source_node_id,
                transport.source_address,
                int(transport.source_port),
                normalized_request_id,
                self._normalize_optional_text(transport.request_fingerprint),
                self._normalize_optional_text(transport.requester_worker_id),
                self._normalize_optional_text(transport.group_id),
                self._normalize_optional_text(transport.group_kind),
                self._normalize_optional_int(transport.group_total_parts),
                self._normalize_optional_text(transport.group_part_id),
                self._normalize_optional_int(transport.group_priority),
                self._normalize_optional_int(transport.group_epoch),
                "in_progress",
            ],
        )
        existing = self.find_by_request_id(normalized_request_id, cursor=cursor)
        if existing is None:
            raise RuntimeError(
                "Transport missing after insert-or-ignore "
                f"request_id={normalized_request_id}"
            )
        created_now = existing.transport_id == transport.transport_id
        if created_now:
            transport.request_id = normalized_request_id
            transport.status = "in_progress"
            transport.completion_outcome = TransportCompletionOutcome.UNSPECIFIED
            transport.completion_detail = None
            return transport, True
        return existing, False

    def delete(self, transport_id: UUID) -> bool:
        """Delete a transport record."""
        cursor = self.get_cursor()
        try:
            return self.delete_with_cursor(transport_id, cursor)
        finally:
            cursor.close()

    def delete_with_cursor(self, transport_id: UUID, cursor) -> bool:
        row = cursor.execute(
            """
            DELETE FROM artifact_transports
            WHERE transport_id = ?
            RETURNING transport_id
            """,
            [str(transport_id)],
        ).fetchone()
        return row is not None

    def update_status(self, transport_id: UUID, status: str, completed_at=None) -> bool:
        """Update transport status and optionally set completed_at."""
        cursor = self.get_cursor()
        try:
            if completed_at is not None:
                row = cursor.execute(
                    """
                    UPDATE artifact_transports
                    SET status = ?, completed_at = ?
                    WHERE transport_id = ?
                    RETURNING transport_id
                    """,
                    [status, completed_at, str(transport_id)],
                ).fetchone()
            else:
                row = cursor.execute(
                    """
                    UPDATE artifact_transports
                    SET status = ?
                    WHERE transport_id = ?
                    RETURNING transport_id
                    """,
                    [status, str(transport_id)],
                ).fetchone()
            return row is not None
        finally:
            cursor.close()

    def complete_if_in_progress(
        self,
        transport_id: UUID,
        completed_at: datetime,
        outcome: TransportCompletionOutcome,
        outcome_detail: str | None = None,
        cursor=None,
    ) -> bool:
        """Atomically mark transport completed only when status is in_progress."""
        owns_cursor = cursor is None
        if owns_cursor:
            cursor = self.get_cursor()
        try:
            row = cursor.execute(
                """
                UPDATE artifact_transports
                SET status = 'completed',
                    completed_at = ?,
                    completion_outcome = ?,
                    completion_detail = ?
                WHERE transport_id = ?
                  AND status = 'in_progress'
                RETURNING transport_id
                """,
                [
                    completed_at,
                    self._normalize_completion_outcome(outcome),
                    self._normalize_optional_text(outcome_detail),
                    str(transport_id),
                ],
            ).fetchone()
            return row is not None
        finally:
            if owns_cursor:
                cursor.close()

    def list_with_filters(
        self,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Transport]:
        """List transports with optional filters and pagination."""
        cursor = self.get_cursor()
        try:
            query = f"SELECT {self._TRANSPORT_PROJECTION} FROM artifact_transports"
            params: list[Any] = []
            if status:
                query += " WHERE status = ?"
                params.append(status)
            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([int(limit), int(offset)])
            result = cursor.execute(query, params)
            rows = result.fetchall()
            assert result.description is not None
            columns = [desc[0] for desc in result.description]
            return [self._row_to_model(row, columns) for row in rows]
        finally:
            cursor.close()

    def get_oldest_in_progress_age_ms(self, replica_id: UUID) -> int | None:
        """Return age (ms) of oldest in-progress transport for replica_id."""
        cursor = self.get_cursor()
        try:
            row = cursor.execute(
                """
                SELECT MIN(created_at) FROM artifact_transports
                WHERE replica_id = ? AND status = 'in_progress'
                """,
                [str(replica_id)],
            ).fetchone()
            if row is None or row[0] is None:
                return None
            created_at = row[0]
            now = (
                datetime.now(tz=created_at.tzinfo)
                if created_at.tzinfo is not None
                else datetime.now()
            )
            return int((now - created_at).total_seconds() * 1000)
        finally:
            cursor.close()

    def count_with_filters(self, status: str | None = None) -> int:
        """Count transports with optional filters."""
        cursor = self.get_cursor()
        try:
            query = "SELECT COUNT(*) FROM artifact_transports"
            params: list[Any] = []
            if status:
                query += " WHERE status = ?"
                params.append(status)
            row = cursor.execute(query, params).fetchone()
            return int(row[0]) if row else 0
        finally:
            cursor.close()

    def list_rows_in_created_window(
        self,
        *,
        started_at: datetime,
        finished_at: datetime,
        limit: int,
    ) -> list[TransportWindowRow]:
        cursor = self.get_cursor()
        try:
            rows = cursor.execute(
                """
                SELECT
                    CAST(t.transport_id AS VARCHAR) AS transport_id,
                    CAST(t.replica_id AS VARCHAR) AS replica_id,
                    COALESCE(t.artifact_id, '') AS artifact_id,
                    COALESCE(t.status, '') AS status,
                    COALESCE(t.completion_outcome, '') AS completion_outcome,
                    COALESCE(t.request_id, '') AS request_id,
                    COALESCE(t.requester_worker_id, '') AS requester_worker_id,
                    COALESCE(t.group_id, '') AS group_id,
                    COALESCE(t.group_kind, '') AS group_kind,
                    COALESCE(t.group_part_id, '') AS group_part_id,
                    COALESCE(t.group_total_parts, 0) AS group_total_parts,
                    t.created_at AS created_at,
                    t.completed_at AS completed_at,
                    COALESCE(r.memory_size, 0) AS replica_memory_size_bytes
                FROM artifact_transports t
                LEFT JOIN artifact_replicas r
                  ON t.replica_id = r.replica_id
                WHERE t.created_at >= ?
                  AND t.created_at <= ?
                ORDER BY t.created_at ASC
                LIMIT ?
                """,
                [started_at, finished_at, int(limit)],
            ).fetchall()
            result: list[TransportWindowRow] = []
            for row in rows:
                created_at = self._coerce_datetime(row[11])
                completed_at = self._coerce_datetime_optional(row[12])
                result.append(
                    TransportWindowRow(
                        transport_id=str(row[0] or ""),
                        replica_id=str(row[1] or ""),
                        artifact_id=str(row[2] or ""),
                        status=str(row[3] or ""),
                        completion_outcome=str(row[4] or ""),
                        request_id=str(row[5] or ""),
                        requester_worker_id=str(row[6] or ""),
                        group_id=str(row[7] or ""),
                        group_kind=str(row[8] or ""),
                        group_part_id=str(row[9] or ""),
                        group_total_parts=int(row[10] or 0),
                        created_at=created_at,
                        completed_at=completed_at,
                        replica_memory_size_bytes=int(row[13] or 0),
                    )
                )
            return result
        finally:
            cursor.close()

    def get_group_progress(
        self,
        *,
        group_kind: str,
        group_id: str,
        group_epoch: int,
        total_parts_hint: int | None = None,
        cursor=None,
    ) -> TransportGroupProgress:
        """Return completion snapshot for one group (SUCCESS outcomes only)."""
        owns_cursor = cursor is None
        if owns_cursor:
            cursor = self.get_cursor()
        try:
            row = cursor.execute(
                """
                SELECT
                    COUNT(DISTINCT CASE
                        WHEN completion_outcome = 'success' AND group_part_id IS NOT NULL
                            THEN group_part_id
                        ELSE NULL
                    END) AS completed_parts,
                    MAX(CASE
                        WHEN completion_outcome = 'success' THEN completed_at
                        ELSE NULL
                    END) AS last_success_at,
                    MAX(group_total_parts) AS observed_total_parts
                FROM artifact_transports
                WHERE group_kind = ?
                  AND group_id = ?
                  AND COALESCE(group_epoch, 0) = ?
                """,
                [group_kind, group_id, int(group_epoch)],
            ).fetchone()
            completed_parts = int(row[0] or 0) if row is not None else 0
            last_success_at = row[1] if row is not None else None
            observed_total_parts = int(row[2] or 0) if row is not None else 0
            total_parts = (
                int(total_parts_hint or 0) if total_parts_hint else observed_total_parts
            )
            return TransportGroupProgress(
                completed_parts=max(0, completed_parts),
                total_parts=max(0, total_parts),
                last_success_at=last_success_at,
            )
        finally:
            if owns_cursor:
                cursor.close()

    @staticmethod
    def _normalize_optional_text(value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped if stripped else None

    @staticmethod
    def _normalize_optional_int(value: int | None) -> int | None:
        if value is None:
            return None
        return int(value)

    @staticmethod
    def _normalize_request_id(request_id: str | None) -> str | None:
        if request_id is None:
            return None
        normalized = request_id.strip()
        return normalized if normalized else None

    @staticmethod
    def _normalize_completion_outcome(outcome: TransportCompletionOutcome) -> str:
        if outcome == TransportCompletionOutcome.UNSPECIFIED:
            return "unspecified"
        if outcome == TransportCompletionOutcome.SUCCESS:
            return "success"
        if outcome == TransportCompletionOutcome.FAILED:
            return "failed"
        if outcome == TransportCompletionOutcome.EXPIRED:
            return "expired"
        if outcome == TransportCompletionOutcome.CANCELLED:
            return "cancelled"
        return "unspecified"

    @staticmethod
    def _coerce_datetime(raw: Any) -> datetime:
        if isinstance(raw, datetime):
            return raw
        if raw is None:
            raise ValueError("created_at is missing")
        return datetime.fromisoformat(str(raw))

    @classmethod
    def _coerce_datetime_optional(cls, raw: Any) -> datetime | None:
        if raw is None:
            return None
        return cls._coerce_datetime(raw)

    def _row_to_model(self, row: tuple[Any, ...], columns: list[str]) -> Transport:
        idx = {column: i for i, column in enumerate(columns)}

        def get(column: str, default=None):
            i = idx.get(column)
            if i is None:
                return default
            value = row[i]
            if value is None:
                return default
            return value

        outcome_raw = str(get("completion_outcome", "unspecified")).strip().lower()
        try:
            outcome = TransportCompletionOutcome(outcome_raw)
        except ValueError:
            outcome = TransportCompletionOutcome.UNSPECIFIED

        return Transport(
            transport_id=UUID(str(get("transport_id"))),
            replica_id=UUID(str(get("replica_id"))),
            artifact_id=str(get("artifact_id", "")),
            requested_view_id=self._normalize_optional_text(get("requested_view_id")),
            disk_path=get("disk_path"),
            source_node_id=str(get("source_node_id", "")),
            source_address=str(get("source_address", "")),
            source_port=int(get("source_port", 0)),
            request_id=self._normalize_request_id(get("request_id")),
            request_fingerprint=self._normalize_optional_text(
                get("request_fingerprint")
            ),
            requester_worker_id=self._normalize_optional_text(
                get("requester_worker_id")
            ),
            group_id=self._normalize_optional_text(get("group_id")),
            group_kind=self._normalize_optional_text(get("group_kind")),
            group_total_parts=self._normalize_optional_int(get("group_total_parts")),
            group_part_id=self._normalize_optional_text(get("group_part_id")),
            group_priority=self._normalize_optional_int(get("group_priority")),
            group_epoch=self._normalize_optional_int(get("group_epoch")),
            completion_outcome=outcome,
            completion_detail=self._normalize_optional_text(get("completion_detail")),
            created_at=get("created_at"),
            completed_at=get("completed_at"),
            status=str(get("status", "in_progress")),
        )
