#  Copyright (c) 2025-2026, TensorCast Team.

"""Repository for pending transport request queue records."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from tensorcast.global_store.models import (
    PendingTransportRequest,
    PendingTransportState,
)
from tensorcast.global_store.repositories.base import BaseRepository


class PendingTransportRequestRepository(BaseRepository):
    """Data access layer for pending_transport_requests table."""

    _PROJECTION = (
        "request_id, request_fingerprint, artifact_id, requested_view_id, "
        "source_node_id, source_address, source_port, requester_worker_id, "
        "group_id, group_kind, group_total_parts, group_part_id, group_priority, "
        "group_epoch, state, deadline_at, created_at, dispatched_at, updated_at"
    )

    def find_by_request_id(
        self, request_id: str, cursor=None
    ) -> PendingTransportRequest | None:
        owns_cursor = cursor is None
        if owns_cursor:
            cursor = self.get_cursor()
        try:
            row = cursor.execute(
                f"SELECT {self._PROJECTION} FROM pending_transport_requests WHERE request_id = ?",
                [request_id],
            ).fetchone()
            if row is None:
                return None
            return self._row_to_model(row)
        finally:
            if owns_cursor:
                cursor.close()

    def create_if_absent_with_cursor(
        self,
        pending_request: PendingTransportRequest,
        cursor,
    ) -> PendingTransportRequest:
        """Insert request if absent; return existing row on duplicate request_id."""
        normalized_request_id = pending_request.request_id.strip()
        cursor.execute(
            """
            INSERT INTO pending_transport_requests (
                request_id,
                request_fingerprint,
                artifact_id,
                requested_view_id,
                source_node_id,
                source_address,
                source_port,
                requester_worker_id,
                group_id,
                group_kind,
                group_total_parts,
                group_part_id,
                group_priority,
                group_epoch,
                state,
                deadline_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'enqueued', ?, ?)
            ON CONFLICT (request_id) DO NOTHING
            """,
            [
                normalized_request_id,
                pending_request.request_fingerprint,
                pending_request.artifact_id,
                pending_request.requested_view_id,
                pending_request.source_node_id,
                pending_request.source_address,
                int(pending_request.source_port),
                pending_request.requester_worker_id,
                pending_request.group_id,
                pending_request.group_kind,
                pending_request.group_total_parts,
                pending_request.group_part_id,
                pending_request.group_priority,
                pending_request.group_epoch,
                pending_request.deadline_at,
                datetime.now(timezone.utc),
            ],
        )
        existing = self.find_by_request_id(normalized_request_id, cursor=cursor)
        if existing is None:
            raise RuntimeError(
                f"Pending request missing after insert-or-ignore request_id={normalized_request_id}"
            )
        return existing

    def list_enqueued(
        self,
        *,
        limit: int,
        cursor=None,
    ) -> list[PendingTransportRequest]:
        owns_cursor = cursor is None
        if owns_cursor:
            cursor = self.get_cursor()
        try:
            rows = cursor.execute(
                f"""
                SELECT {self._PROJECTION}
                FROM pending_transport_requests
                WHERE state = 'enqueued'
                ORDER BY created_at ASC, request_id ASC
                LIMIT ?
                """,
                [int(limit)],
            ).fetchall()
            return [self._row_to_model(row) for row in rows]
        finally:
            if owns_cursor:
                cursor.close()

    def mark_dispatched(self, request_id: str, cursor) -> bool:
        row = cursor.execute(
            """
            UPDATE pending_transport_requests
            SET state = 'dispatched',
                dispatched_at = now(),
                updated_at = now()
            WHERE request_id = ?
              AND state = 'enqueued'
            RETURNING request_id
            """,
            [request_id],
        ).fetchone()
        return row is not None

    def mark_cancelled(self, request_id: str, cursor=None) -> bool:
        owns_cursor = cursor is None
        if owns_cursor:
            cursor = self.get_cursor()
        try:
            row = cursor.execute(
                """
                UPDATE pending_transport_requests
                SET state = 'cancelled',
                    updated_at = now()
                WHERE request_id = ?
                  AND state = 'enqueued'
                RETURNING request_id
                """,
                [request_id],
            ).fetchone()
            return row is not None
        finally:
            if owns_cursor:
                cursor.close()

    def expire_enqueued_deadlines(self, *, now_utc: datetime, cursor=None) -> list[str]:
        owns_cursor = cursor is None
        if owns_cursor:
            cursor = self.get_cursor()
        try:
            rows = cursor.execute(
                """
                UPDATE pending_transport_requests
                SET state = 'expired',
                    updated_at = now()
                WHERE state = 'enqueued'
                  AND deadline_at IS NOT NULL
                  AND deadline_at <= ?
                RETURNING request_id
                """,
                [now_utc],
            ).fetchall()
            return [str(row[0]) for row in rows]
        finally:
            if owns_cursor:
                cursor.close()

    def _row_to_model(self, row: tuple[Any, ...]) -> PendingTransportRequest:
        return PendingTransportRequest(
            request_id=str(row[0]),
            request_fingerprint=str(row[1]),
            artifact_id=str(row[2]),
            requested_view_id=str(row[3]) if row[3] is not None else None,
            source_node_id=str(row[4]),
            source_address=str(row[5]),
            source_port=int(row[6]),
            requester_worker_id=str(row[7]) if row[7] is not None else None,
            group_id=str(row[8]) if row[8] is not None else None,
            group_kind=str(row[9]) if row[9] is not None else None,
            group_total_parts=int(row[10]) if row[10] is not None else None,
            group_part_id=str(row[11]) if row[11] is not None else None,
            group_priority=int(row[12]) if row[12] is not None else None,
            group_epoch=int(row[13]) if row[13] is not None else None,
            state=PendingTransportState(str(row[14])),
            deadline_at=row[15],
            created_at=row[16],
            dispatched_at=row[17],
            updated_at=row[18],
        )
