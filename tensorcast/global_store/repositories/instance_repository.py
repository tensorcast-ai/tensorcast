#  Copyright (c) 2026, TensorCast Team.

"""Repository for engine instance registry data access."""

import json
import threading
from contextlib import contextmanager, nullcontext

from duckdb import DuckDBPyConnection

from tensorcast.global_store.models import Instance
from tensorcast.global_store.repositories.base import BaseRepository

_INSTANCE_SELECT = """
    SELECT
        instance_id,
        daemon_id,
        worker_id,
        engine,
        signals_endpoint,
        labels_json,
        capability_flags,
        registered_at,
        last_heartbeat,
        inactive_at
    FROM instances
"""


class InstanceRepository(BaseRepository):
    def __init__(self, connection: DuckDBPyConnection):
        super().__init__(connection)
        self._write_lock = threading.RLock()

    def _write_guard(self, cursor: DuckDBPyConnection | None):
        return nullcontext() if cursor is not None else self._write_lock

    @contextmanager
    def transaction(self):
        with self._write_lock, super().transaction() as cursor:
            yield cursor

    def find_by_id(
        self, instance_id: str, *, include_inactive: bool = False
    ) -> Instance | None:
        cursor = self.get_cursor()
        try:
            result = cursor.execute(
                f"{_INSTANCE_SELECT} WHERE instance_id = ?"
                + ("" if include_inactive else " AND inactive_at IS NULL"),
                [instance_id],
            ).fetchone()
            if result:
                return self._row_to_model(result)
            return None
        finally:
            cursor.close()

    def find_by_daemon_id(
        self, daemon_id: str, *, include_inactive: bool = False
    ) -> list[Instance]:
        cursor = self.get_cursor()
        try:
            rows = cursor.execute(
                f"{_INSTANCE_SELECT} WHERE daemon_id = ?"
                + ("" if include_inactive else " AND inactive_at IS NULL"),
                [daemon_id],
            ).fetchall()
            return [self._row_to_model(row) for row in rows]
        finally:
            cursor.close()

    def create(self, instance: Instance) -> Instance:
        with self._write_lock:
            cursor = self.get_cursor()
            try:
                cursor.execute(
                    """
                    INSERT INTO instances (
                        instance_id,
                        daemon_id,
                        worker_id,
                        engine,
                        signals_endpoint,
                        labels_json,
                        capability_flags
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        instance.instance_id,
                        instance.daemon_id,
                        instance.worker_id,
                        instance.engine,
                        instance.signals_endpoint,
                        json.dumps(dict(instance.labels or {})),
                        int(instance.capability_flags),
                    ],
                )
            finally:
                cursor.close()
        return instance

    def update(self, instance: Instance) -> Instance:
        with self._write_lock:
            cursor = self.get_cursor()
            try:
                cursor.execute(
                    """
                    UPDATE instances
                    SET daemon_id = ?,
                        worker_id = ?,
                        engine = ?,
                        signals_endpoint = ?,
                        labels_json = ?,
                        capability_flags = ?,
                        last_heartbeat = CURRENT_TIMESTAMP,
                        inactive_at = NULL
                    WHERE instance_id = ?
                    """,
                    [
                        instance.daemon_id,
                        instance.worker_id,
                        instance.engine,
                        instance.signals_endpoint,
                        json.dumps(dict(instance.labels or {})),
                        int(instance.capability_flags),
                        instance.instance_id,
                    ],
                )
            finally:
                cursor.close()
        return instance

    def heartbeat(self, instance_id: str, worker_id: str | None = None) -> bool:
        with self._write_lock:
            cursor = self.get_cursor()
            try:
                cursor.execute(
                    """
                    UPDATE instances
                    SET last_heartbeat = CURRENT_TIMESTAMP,
                        worker_id = COALESCE(?, worker_id)
                    WHERE instance_id = ? AND inactive_at IS NULL
                    RETURNING instance_id
                    """,
                    [worker_id, instance_id],
                )
                row = cursor.fetchone()
            finally:
                cursor.close()
        return bool(row)

    def update_capability_flags(self, instance_id: str, capability_flags: int) -> bool:
        with self._write_lock:
            cursor = self.get_cursor()
            try:
                cursor.execute(
                    """
                    UPDATE instances
                    SET capability_flags = ?
                    WHERE instance_id = ?
                      AND inactive_at IS NULL
                      AND capability_flags != ?
                    RETURNING instance_id
                    """,
                    [int(capability_flags), instance_id, int(capability_flags)],
                )
                row = cursor.fetchone()
            finally:
                cursor.close()
        return bool(row)

    def mark_inactive(self, instance_id: str) -> bool:
        with self._write_lock:
            cursor = self.get_cursor()
            try:
                cursor.execute(
                    """
                    UPDATE instances
                    SET inactive_at = CURRENT_TIMESTAMP
                    WHERE instance_id = ? AND inactive_at IS NULL
                    RETURNING instance_id
                    """,
                    [instance_id],
                )
                row = cursor.fetchone()
            finally:
                cursor.close()
        return bool(row)

    def list_active(self, *, include_unavailable: bool = False) -> list[Instance]:
        cursor = self.get_cursor()
        try:
            rows = cursor.execute(
                f"{_INSTANCE_SELECT}"
                + ("" if include_unavailable else " WHERE inactive_at IS NULL")
                + " ORDER BY last_heartbeat DESC"
            ).fetchall()
            return [self._row_to_model(row) for row in rows]
        finally:
            cursor.close()

    def cleanup_stale_instances(self, cutoff_seconds: float) -> int:
        with self._write_lock:
            cursor = self.get_cursor()
            try:
                cursor.execute(
                    """
                    UPDATE instances
                    SET inactive_at = CURRENT_TIMESTAMP
                    WHERE inactive_at IS NULL
                      AND EXTRACT(epoch FROM last_heartbeat) < ?
                    """,
                    [cutoff_seconds],
                )
                return int(cursor.rowcount or 0)
            finally:
                cursor.close()

    @staticmethod
    def _row_to_model(row) -> Instance:
        labels_json = row[5] or "{}"
        try:
            labels = json.loads(labels_json)
        except Exception:
            labels = {}
        return Instance(
            instance_id=row[0],
            daemon_id=row[1],
            worker_id=row[2],
            engine=row[3],
            signals_endpoint=row[4] or None,
            labels=labels if isinstance(labels, dict) else {},
            capability_flags=row[6] or 0,
            registered_at=row[7],
            last_heartbeat=row[8],
            inactive_at=row[9],
        )


__all__ = ["InstanceRepository"]
