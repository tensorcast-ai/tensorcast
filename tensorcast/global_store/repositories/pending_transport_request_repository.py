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
from tensorcast.logger import init_logger

logger = init_logger(__name__)

_VALID_PENDING_STATES = {
    PendingTransportState.ENQUEUED.value,
    PendingTransportState.DISPATCHED.value,
    PendingTransportState.CANCELLED.value,
    PendingTransportState.EXPIRED.value,
}


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
        normalized_request_id = self._normalize_request_id(request_id)
        if normalized_request_id is None:
            return None
        owns_cursor = cursor is None
        if owns_cursor:
            cursor = self.get_cursor()
        try:
            row = cursor.execute(
                f"SELECT {self._PROJECTION} FROM pending_transport_requests WHERE request_id = ?",
                [normalized_request_id],
            ).fetchone()
            if row is None:
                return None
            return self._safe_row_to_model(row, cursor)
        finally:
            if owns_cursor:
                cursor.close()

    def create_if_absent_with_cursor(
        self,
        pending_request: PendingTransportRequest,
        cursor,
    ) -> PendingTransportRequest:
        """Insert request if absent; return existing row on duplicate request_id."""
        normalized_request_id = self._normalize_request_id(pending_request.request_id)
        if normalized_request_id is None:
            raise ValueError("request_id is required")
        normalized_request_fingerprint = self._normalize_required_text(
            pending_request.request_fingerprint
        )
        normalized_artifact_id = self._normalize_required_text(
            pending_request.artifact_id
        )
        normalized_requested_view_id = self._normalize_optional_text(
            pending_request.requested_view_id
        )
        normalized_source_node_id = self._normalize_required_text(
            pending_request.source_node_id
        )
        normalized_source_address = self._normalize_required_text(
            pending_request.source_address
        )
        normalized_requester_worker_id = self._normalize_optional_text(
            pending_request.requester_worker_id
        )
        normalized_group_id = self._normalize_optional_text(pending_request.group_id)
        normalized_group_kind = self._normalize_optional_text(
            pending_request.group_kind
        )
        normalized_group_part_id = self._normalize_optional_text(
            pending_request.group_part_id
        )
        existing = self.find_by_request_id(normalized_request_id, cursor=cursor)
        if existing is not None:
            return existing

        try:
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
                """,
                [
                    normalized_request_id,
                    normalized_request_fingerprint,
                    normalized_artifact_id,
                    normalized_requested_view_id,
                    normalized_source_node_id,
                    normalized_source_address,
                    int(pending_request.source_port),
                    normalized_requester_worker_id,
                    normalized_group_id,
                    normalized_group_kind,
                    pending_request.group_total_parts,
                    normalized_group_part_id,
                    pending_request.group_priority,
                    pending_request.group_epoch,
                    pending_request.deadline_at,
                    datetime.now(timezone.utc),
                ],
            )
        except Exception as exc:  # noqa: BLE001
            if self._looks_like_unique_violation(exc):
                existing = self.find_by_request_id(normalized_request_id, cursor=cursor)
                if existing is not None:
                    return existing
            raise

        existing = self.find_by_request_id(normalized_request_id, cursor=cursor)
        if existing is None:
            raise RuntimeError(
                f"Pending request missing after insert request_id={normalized_request_id}"
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
            out: list[PendingTransportRequest] = []
            for row in rows:
                model = self._safe_row_to_model(row, cursor)
                if model is not None:
                    out.append(model)
            return out
        finally:
            if owns_cursor:
                cursor.close()

    def purge_malformed_rows(self, cursor=None) -> int:
        """Delete or heal malformed queue rows left by transient DB corruption."""
        owns_cursor = cursor is None
        if owns_cursor:
            cursor = self.get_cursor()
        try:
            rows = cursor.execute(
                """
                SELECT
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
                    state
                FROM pending_transport_requests
                """
            ).fetchall()
            if not rows:
                return 0
            deleted = 0
            healed = 0
            for row in rows:
                request_id = str(row[0] or "")
                state = str(row[14] or "")
                normalized_request_id = self._normalize_request_id(request_id)
                normalized_request_fingerprint = self._normalize_required_text(
                    str(row[1] or "")
                )
                normalized_artifact_id = self._normalize_required_text(
                    str(row[2] or "")
                )
                normalized_requested_view_id = self._normalize_optional_text(
                    row[3] if row[3] is not None else None
                )
                normalized_source_node_id = self._normalize_required_text(
                    str(row[4] or "")
                )
                normalized_source_address = self._normalize_required_text(
                    str(row[5] or "")
                )
                normalized_requester_worker_id = self._normalize_optional_text(
                    row[7] if row[7] is not None else None
                )
                normalized_group_id = self._normalize_optional_text(
                    row[8] if row[8] is not None else None
                )
                normalized_group_kind = self._normalize_optional_text(
                    row[9] if row[9] is not None else None
                )
                normalized_group_part_id = self._normalize_optional_text(
                    row[11] if row[11] is not None else None
                )

                should_delete = (
                    normalized_request_id is None
                    or state not in _VALID_PENDING_STATES
                    or not normalized_request_fingerprint
                    or not normalized_artifact_id
                    or not normalized_source_node_id
                    or not normalized_source_address
                )
                if should_delete:
                    if request_id:
                        cursor.execute(
                            """
                            DELETE FROM pending_transport_requests
                            WHERE request_id = ?
                            """,
                            [request_id],
                        )
                        deleted += 1
                    continue

                needs_heal = (
                    normalized_request_id != request_id
                    or normalized_request_fingerprint != str(row[1] or "")
                    or normalized_artifact_id != str(row[2] or "")
                    or normalized_requested_view_id != row[3]
                    or normalized_source_node_id != str(row[4] or "")
                    or normalized_source_address != str(row[5] or "")
                    or normalized_requester_worker_id != row[7]
                    or normalized_group_id != row[8]
                    or normalized_group_kind != row[9]
                    or normalized_group_part_id != row[11]
                )
                if not needs_heal:
                    continue
                try:
                    cursor.execute(
                        """
                        UPDATE pending_transport_requests
                        SET request_id = ?,
                            request_fingerprint = ?,
                            artifact_id = ?,
                            requested_view_id = ?,
                            source_node_id = ?,
                            source_address = ?,
                            requester_worker_id = ?,
                            group_id = ?,
                            group_kind = ?,
                            group_part_id = ?,
                            updated_at = now()
                        WHERE request_id = ?
                        """,
                        [
                            normalized_request_id,
                            normalized_request_fingerprint,
                            normalized_artifact_id,
                            normalized_requested_view_id,
                            normalized_source_node_id,
                            normalized_source_address,
                            normalized_requester_worker_id,
                            normalized_group_id,
                            normalized_group_kind,
                            normalized_group_part_id,
                            request_id,
                        ],
                    )
                    healed += 1
                except Exception as exc:  # noqa: BLE001
                    if self._looks_like_unique_violation(exc):
                        cursor.execute(
                            """
                            DELETE FROM pending_transport_requests
                            WHERE request_id = ?
                            """,
                            [request_id],
                        )
                        deleted += 1
                        continue
                    raise
            if deleted > 0 or healed > 0:
                logger.warning(
                    "Reconciled pending transport rows healed=%s deleted=%s",
                    healed,
                    deleted,
                )
            return deleted + healed
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
        request_id = self._normalize_request_id(str(row[0] or ""))
        if request_id is None:
            raise ValueError("invalid pending request_id")
        artifact_id = self._normalize_required_text(str(row[2] or ""))
        state_raw = str(row[14] or "")
        self._validate_raw_row(state_raw=state_raw)
        return PendingTransportRequest(
            request_id=request_id,
            request_fingerprint=self._normalize_required_text(str(row[1] or "")),
            artifact_id=artifact_id,
            requested_view_id=self._normalize_optional_text(
                row[3] if row[3] is not None else None
            ),
            source_node_id=self._normalize_required_text(str(row[4] or "")),
            source_address=self._normalize_required_text(str(row[5] or "")),
            source_port=int(row[6]),
            requester_worker_id=self._normalize_optional_text(
                row[7] if row[7] is not None else None
            ),
            group_id=self._normalize_optional_text(
                row[8] if row[8] is not None else None
            ),
            group_kind=self._normalize_optional_text(
                row[9] if row[9] is not None else None
            ),
            group_total_parts=int(row[10]) if row[10] is not None else None,
            group_part_id=self._normalize_optional_text(
                row[11] if row[11] is not None else None
            ),
            group_priority=int(row[12]) if row[12] is not None else None,
            group_epoch=int(row[13]) if row[13] is not None else None,
            state=PendingTransportState(state_raw),
            deadline_at=row[15],
            created_at=row[16],
            dispatched_at=row[17],
            updated_at=row[18],
        )

    def _safe_row_to_model(
        self, row: tuple[Any, ...], cursor
    ) -> PendingTransportRequest | None:
        try:
            return self._row_to_model(row)
        except ValueError as exc:
            self._purge_malformed_row(row, cursor, exc)
            return None

    def _purge_malformed_row(
        self, row: tuple[Any, ...], cursor, exc: Exception
    ) -> None:
        request_id = str(row[0] or "")
        state_raw = str(row[14] or "")
        artifact_id = str(row[2] or "")
        logger.warning(
            "Dropping malformed pending transport row request_id=%s state=%s artifact_id=%s error=%s",
            request_id,
            state_raw,
            artifact_id,
            exc,
        )
        if not request_id:
            return
        cursor.execute(
            """
            DELETE FROM pending_transport_requests
            WHERE request_id = ?
            """,
            [request_id],
        )

    @staticmethod
    def _looks_like_unique_violation(exc: Exception) -> bool:
        message = str(exc).lower()
        unique_markers = (
            "duplicate key",
            "unique constraint",
            "violates unique",
        )
        return any(marker in message for marker in unique_markers)

    @staticmethod
    def _validate_raw_row(*, state_raw: str) -> None:
        if state_raw not in _VALID_PENDING_STATES:
            raise ValueError(f"invalid pending state: {state_raw}")

    @classmethod
    def _normalize_request_id(cls, request_id: str | None) -> str | None:
        return cls._normalize_text_token(request_id)

    @classmethod
    def _normalize_optional_text(cls, value: str | None) -> str | None:
        return cls._normalize_text_token(value)

    @classmethod
    def _normalize_required_text(cls, value: str | None) -> str:
        normalized = cls._normalize_text_token(value)
        if normalized is None:
            raise ValueError("required pending transport field is missing")
        return normalized

    @classmethod
    def _normalize_text_token(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = str(value).strip()
        if not stripped:
            return None
        return cls._collapse_exact_double(stripped)

    @staticmethod
    def _collapse_exact_double(value: str) -> str:
        size = len(value)
        if size < 8 or (size % 2) != 0:
            return value
        half = size // 2
        if value[:half] != value[half:]:
            return value
        return value[:half]
