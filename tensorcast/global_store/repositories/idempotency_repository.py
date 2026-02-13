#  Copyright (c) 2025-2026, TensorCast Team.

"""Repository for control-plane idempotency records."""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from duckdb import DuckDBPyConnection

from tensorcast.global_store.repositories.base import BaseRepository

IDEMPOTENCY_PENDING_STATUS = "__PENDING__"
IDEMPOTENCY_RETENTION_MS = 24 * 60 * 60 * 1000
_CLEANUP_INTERVAL_SECONDS = 60.0


@dataclass(frozen=True)
class IdempotencyRecord:
    client_request_id: str
    operation_kind: str
    request_fingerprint: str
    response_status: str
    response_proto: bytes
    created_at: datetime | None


class IdempotencyRepository(BaseRepository):
    """Data access for `control_plane_idempotency`."""

    def __init__(self, connection: DuckDBPyConnection):
        super().__init__(connection)
        self._cleanup_lock = threading.Lock()
        self._last_cleanup_at_s = 0.0

    @staticmethod
    def fingerprint_payload(payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _is_unique_constraint_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "duplicate key" in message or "constraint" in message

    @staticmethod
    def _row_to_record(row: tuple) -> IdempotencyRecord:
        return IdempotencyRecord(
            client_request_id=str(row[0]),
            operation_kind=str(row[1]),
            request_fingerprint=str(row[2]),
            response_status=str(row[3]),
            response_proto=bytes(row[4] or b""),
            created_at=row[5],
        )

    def reserve_operation(
        self,
        *,
        client_request_id: str,
        operation_kind: str,
        request_fingerprint: str,
        cursor: DuckDBPyConnection | None = None,
    ) -> bool:
        owns_cursor = cursor is None
        cursor = cursor if cursor is not None else self.get_cursor()
        try:
            self._maybe_cleanup(cursor=cursor)
            try:
                cursor.execute(
                    """
                    INSERT INTO control_plane_idempotency (
                        client_request_id,
                        operation_kind,
                        request_fingerprint,
                        response_status,
                        response_proto
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        client_request_id,
                        operation_kind,
                        request_fingerprint,
                        IDEMPOTENCY_PENDING_STATUS,
                        b"",
                    ],
                )
                return True
            except Exception as exc:  # noqa: BLE001
                if self._is_unique_constraint_error(exc):
                    return False
                raise
        finally:
            if owns_cursor:
                cursor.close()

    def get_record(
        self,
        client_request_id: str,
        *,
        cursor: DuckDBPyConnection | None = None,
    ) -> IdempotencyRecord | None:
        owns_cursor = cursor is None
        cursor = cursor if cursor is not None else self.get_cursor()
        try:
            row = cursor.execute(
                """
                SELECT
                    client_request_id,
                    operation_kind,
                    request_fingerprint,
                    response_status,
                    response_proto,
                    created_at
                FROM control_plane_idempotency
                WHERE client_request_id = ?
                """,
                [client_request_id],
            ).fetchone()
            if row is None:
                return None
            return self._row_to_record(row)
        finally:
            if owns_cursor:
                cursor.close()

    def wait_for_completed_record(
        self,
        *,
        client_request_id: str,
        timeout_ms: int = 2000,
        poll_interval_ms: int = 25,
    ) -> IdempotencyRecord | None:
        deadline = time.monotonic() + max(1, timeout_ms) / 1000.0
        while time.monotonic() < deadline:
            record = self.get_record(client_request_id)
            if (
                record is not None
                and record.response_status != IDEMPOTENCY_PENDING_STATUS
            ):
                return record
            time.sleep(max(1, poll_interval_ms) / 1000.0)
        return None

    def finalize_operation(
        self,
        *,
        client_request_id: str,
        response_status: str,
        response_proto: bytes,
        cursor: DuckDBPyConnection | None = None,
    ) -> None:
        owns_cursor = cursor is None
        cursor = cursor if cursor is not None else self.get_cursor()
        try:
            cursor.execute(
                """
                UPDATE control_plane_idempotency
                SET response_status = ?, response_proto = ?
                WHERE client_request_id = ?
                """,
                [response_status, response_proto, client_request_id],
            )
        finally:
            if owns_cursor:
                cursor.close()

    def _maybe_cleanup(self, *, cursor: DuckDBPyConnection) -> None:
        now_s = time.monotonic()
        if now_s - self._last_cleanup_at_s < _CLEANUP_INTERVAL_SECONDS:
            return
        with self._cleanup_lock:
            now_s = time.monotonic()
            if now_s - self._last_cleanup_at_s < _CLEANUP_INTERVAL_SECONDS:
                return
            self._last_cleanup_at_s = now_s
            threshold = datetime.now(timezone.utc) - timedelta(
                milliseconds=IDEMPOTENCY_RETENTION_MS
            )
            cursor.execute(
                """
                DELETE FROM control_plane_idempotency
                WHERE created_at < ?
                """,
                [threshold],
            )
