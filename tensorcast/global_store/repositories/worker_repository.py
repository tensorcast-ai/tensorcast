#  Copyright (c) 2025-2026, TensorCast Team.

"""Repository for worker data access."""

import threading
import time
from contextlib import contextmanager, nullcontext

from duckdb import DuckDBPyConnection

from tensorcast.global_store.config import GlobalStoreConfig, get_config
from tensorcast.global_store.models import Worker, WorkerMemoryTierState
from tensorcast.global_store.repositories.base import BaseRepository
from tensorcast.logger import init_logger

logger = init_logger(__name__)

_WORKER_SELECT = """
    SELECT
        workers.worker_id,
        workers.node_id,
        workers.node_address,
        workers.grpc_port,
        workers.p2p_port,
        workers.mem_pool_total_size,
        workers.mem_pool_available_size,
        workers.accepting_new_requests,
        workers.registered_at,
        workers.last_heartbeat,
        workers.inactive_at,
        COALESCE(workers.state_version, 1) AS state_version,
        COALESCE(workers.state_checksum, '') AS state_checksum,
        mt.stable_total_bytes,
        mt.stable_used_bytes,
        mt.preemptible_total_bytes,
        mt.preemptible_marked_bytes,
        mt.faults_per_sec,
        mt.rehydrate_p99_ns,
        mt.enable_preemptible,
        mt.memory_tier_config_json,
        mt.snapshot_epoch_ns
    FROM workers
    LEFT JOIN node_memory_tier_latest AS mt
        ON mt.node_id = workers.node_id
"""


class WorkerRepository(BaseRepository):
    """Repository for managing workers in the database."""

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
        self, worker_id: str, *, include_inactive: bool = False
    ) -> Worker | None:
        """Find a worker by ID."""
        cursor = self.get_cursor()
        result = cursor.execute(
            f"{_WORKER_SELECT} WHERE workers.worker_id = ?"
            + ("" if include_inactive else " AND workers.inactive_at IS NULL"),
            [worker_id],
        ).fetchone()

        if result:
            return self._row_to_model(result)
        return None

    def find_by_address_port(
        self, node_address: str, grpc_port: int, *, include_inactive: bool = True
    ) -> Worker | None:
        """Find a worker by address and port."""
        cursor = self.get_cursor()
        result = cursor.execute(
            f"""
            {_WORKER_SELECT}
            WHERE workers.node_address = ? AND workers.grpc_port = ?
            {"" if include_inactive else "AND workers.inactive_at IS NULL"}
            """,
            [node_address, grpc_port],
        ).fetchone()

        if result:
            return self._row_to_model(result)
        return None

    def create(self, worker: Worker) -> Worker:
        """Create a new worker."""
        with self._write_lock:
            cursor = self.get_cursor()
            cursor.execute(
                """
                INSERT INTO workers (
                    worker_id, node_id, node_address, grpc_port, p2p_port,
                    mem_pool_total_size, mem_pool_available_size,
                    accepting_new_requests, state_version, state_checksum
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    worker.worker_id,
                    worker.node_id,
                    worker.node_address,
                    worker.grpc_port,
                    worker.p2p_port,
                    worker.mem_pool_total_size,
                    worker.mem_pool_available_size,
                    worker.accepting_new_requests,
                    worker.state_version,
                    worker.state_checksum,
                ],
            )

        return worker

    def create_or_update(self, worker: Worker) -> Worker:
        """Create a new worker or update existing one."""
        existing = self.find_by_id(worker.worker_id, include_inactive=True)
        if existing:
            return self.update(worker)
        else:
            return self.create(worker)

    def update(self, worker: Worker, *, reset_state_tracking: bool = False) -> Worker:
        """Update an existing worker."""
        with self._write_lock:
            cursor = self.get_cursor()
            reset_clause = ""
            if reset_state_tracking:
                reset_clause = (
                    ", state_version = 1, state_checksum = '',"
                    " state_sync_epoch = 0, state_sync_request_id = 0"
                )
            result = cursor.execute(
                f"""
                UPDATE workers
                SET node_id = ?, node_address = ?, grpc_port = ?, p2p_port = ?,
                    mem_pool_total_size = ?, mem_pool_available_size = ?,
                    accepting_new_requests = ?,
                    last_heartbeat = CURRENT_TIMESTAMP,
                    inactive_at = NULL
                    {reset_clause}
                WHERE worker_id = ?
                RETURNING worker_id
                """,
                [
                    worker.node_id,
                    worker.node_address,
                    worker.grpc_port,
                    worker.p2p_port,
                    worker.mem_pool_total_size,
                    worker.mem_pool_available_size,
                    worker.accepting_new_requests,
                    worker.worker_id,
                ],
            )

            if result.fetchone() is None:
                raise ValueError(f"Worker {worker.worker_id} not found")

        return worker

    def update_heartbeat(
        self, worker_id: str, mem_pool_available_size: int, accepting_new_requests: bool
    ) -> bool:
        """Update worker heartbeat and status."""
        with self._write_lock:
            cursor = self.get_cursor()
            result = cursor.execute(
                """
                UPDATE workers
                SET last_heartbeat = CURRENT_TIMESTAMP,
                    mem_pool_available_size = ?,
                    accepting_new_requests = ?
                WHERE worker_id = ?
                  AND inactive_at IS NULL
                RETURNING worker_id
                """,
                [mem_pool_available_size, accepting_new_requests, worker_id],
            )

        return result.fetchone() is not None

    def batch_update_heartbeats(self, updates: list[tuple[str, int, bool]]) -> int:
        """
        Batch update worker heartbeats and statuses **in one SQL statement**.

        Args:
            updates: List of (worker_id, mem_pool_available_size, accepting_new_requests).

        Returns:
            Number of successfully updated workers.
        """
        if not updates:
            return 0

        # Build VALUES clause dynamically –  one tuple per update
        # DuckDB supports `UPDATE … FROM (VALUES …) AS alias(col1,col2,…)` pattern.
        placeholders = ", ".join(["(?, ?, ?)"] * len(updates))
        values_sql = f"VALUES {placeholders}"

        # Flatten parameters for executemany-style substitution
        params: list = []
        for worker_id, mem_pool_available_size, accepting_new_requests in updates:
            params.extend([worker_id, mem_pool_available_size, accepting_new_requests])

        sql = f"""
            WITH upd(worker_id, mem_pool_available_size, accepting_new_requests) AS (
                {values_sql}
            )
            UPDATE workers
            SET last_heartbeat = CURRENT_TIMESTAMP,
                mem_pool_available_size = upd.mem_pool_available_size,
                accepting_new_requests = upd.accepting_new_requests
            FROM upd
            WHERE workers.worker_id = upd.worker_id
              AND workers.inactive_at IS NULL
            RETURNING workers.worker_id
        """

        try:
            with self._write_lock:
                cursor = self.get_cursor()
                result = cursor.execute(sql, params)
                updated_rows = result.fetchall()
                return len(updated_rows)
        except Exception:
            logger.exception("Batch heartbeat update failed")
            return 0

    def delete(self, worker_id: str) -> bool:
        """Delete a worker."""
        with self._write_lock:
            cursor = self.get_cursor()
            result = cursor.execute(
                """
                DELETE FROM workers
                WHERE worker_id = ?
                RETURNING worker_id
                """,
                [worker_id],
            )
            return result.fetchone() is not None

    def mark_inactive_by_timeout(self, timeout_seconds: float) -> list[tuple[str, str]]:
        """
        Mark inactive workers and return their IDs and node IDs.

        Returns:
            List of (worker_id, node_id) tuples
        """
        cutoff_time = time.time() - timeout_seconds
        with self._write_lock:
            cursor = self.get_cursor()
            rows = cursor.execute(
                """
                UPDATE workers
                SET inactive_at = CURRENT_TIMESTAMP,
                    accepting_new_requests = FALSE
                WHERE inactive_at IS NULL
                  AND EXTRACT(epoch FROM last_heartbeat) < ?
                RETURNING worker_id, node_id
                """,
                [cutoff_time],
            ).fetchall()
            return rows

    def delete_inactive(self, timeout_seconds: float) -> list[tuple[str, str]]:
        return self.mark_inactive_by_timeout(timeout_seconds)

    def find_active(self, include_unavailable: bool = False) -> list[Worker]:
        """Find all active workers."""
        cursor = self.get_cursor()
        try:
            config = get_config()
        except RuntimeError:
            config = GlobalStoreConfig()

        query = f"""
            {_WORKER_SELECT}
            WHERE EXTRACT(epoch FROM workers.last_heartbeat) > ?
              AND workers.inactive_at IS NULL
        """
        # Use configured heartbeat timeout instead of hardcoded value
        timeout_seconds = config.heartbeat_timeout_ms / 1000.0
        params = [time.time() - timeout_seconds]

        if not include_unavailable:
            query += " AND workers.accepting_new_requests = TRUE"

        query += " ORDER BY workers.last_heartbeat DESC"

        result = cursor.execute(query, params)
        workers = [self._row_to_model(row) for row in result.fetchall()]
        return workers

    # ========== High Availability Methods ==========

    def list_all_workers(self) -> list[Worker]:
        """List all workers in the database."""
        cursor = self.get_cursor()
        result = cursor.execute(f"{_WORKER_SELECT} ORDER BY workers.registered_at DESC")
        return [self._row_to_model(row) for row in result.fetchall()]

    def get_state_version(
        self, worker_id: str, cursor: DuckDBPyConnection | None = None
    ) -> int:
        """Return the persisted state version for the worker."""
        return self.ensure_state_version(worker_id, cursor)

    def get_state_checksum(
        self, worker_id: str, cursor: DuckDBPyConnection | None = None
    ) -> str:
        """Return the persisted state checksum for the worker."""
        with self._write_guard(cursor):
            cursor = cursor if cursor is not None else self.get_cursor()
            result = cursor.execute(
                """
                SELECT COALESCE(state_checksum, '') FROM workers
                WHERE worker_id = ? AND inactive_at IS NULL
                """,
                [worker_id],
            ).fetchone()
            if not result:
                raise ValueError(f"Worker {worker_id} not found")
            return result[0] or ""

    def ensure_state_version(
        self, worker_id: str, cursor: DuckDBPyConnection | None = None
    ) -> int:
        """Ensure the worker has a non-zero state version."""
        with self._write_guard(cursor):
            cursor = cursor if cursor is not None else self.get_cursor()
            result = cursor.execute(
                """
                SELECT state_version FROM workers
                WHERE worker_id = ? AND inactive_at IS NULL
                """,
                [worker_id],
            ).fetchone()
            if not result:
                raise ValueError(f"Worker {worker_id} not found")
            current_version = int(result[0] or 0)
            if current_version <= 0:
                current_version = 1
                cursor.execute(
                    """
                    UPDATE workers
                    SET state_version = ?
                    WHERE worker_id = ? AND inactive_at IS NULL
                    """,
                    [current_version, worker_id],
                )
            return current_version

    def update_state_version_and_checksum(
        self,
        worker_id: str,
        state_version: int,
        state_checksum: str,
        cursor: DuckDBPyConnection | None = None,
    ) -> None:
        """Update the worker's state version and checksum together."""
        with self._write_guard(cursor):
            cursor = cursor if cursor is not None else self.get_cursor()
            result = cursor.execute(
                """
                UPDATE workers
                SET state_version = ?, state_checksum = ?
                WHERE worker_id = ? AND inactive_at IS NULL
                RETURNING worker_id
                """,
                [state_version, state_checksum, worker_id],
            ).fetchone()
            if result is None:
                raise ValueError(f"Worker {worker_id} not found")

    def update_state_checksum(
        self,
        worker_id: str,
        state_checksum: str,
        cursor: DuckDBPyConnection | None = None,
    ) -> None:
        """Update the worker's state checksum."""
        with self._write_guard(cursor):
            cursor = cursor if cursor is not None else self.get_cursor()
            result = cursor.execute(
                """
                UPDATE workers
                SET state_checksum = ?
                WHERE worker_id = ? AND inactive_at IS NULL
                RETURNING worker_id
                """,
                [state_checksum, worker_id],
            ).fetchone()
            if result is None:
                raise ValueError(f"Worker {worker_id} not found")

    def try_advance_state_sync_token(
        self,
        worker_id: str,
        sync_epoch: int,
        sync_request_id: int,
        cursor: DuckDBPyConnection | None = None,
    ) -> bool:
        """Advance the sync token if the incoming token is newer."""
        with self._write_guard(cursor):
            cursor = cursor if cursor is not None else self.get_cursor()
            result = cursor.execute(
                """
                UPDATE workers
                SET state_sync_epoch = ?, state_sync_request_id = ?
                WHERE worker_id = ?
                  AND inactive_at IS NULL
                  AND (
                    state_sync_epoch < ?
                    OR (state_sync_epoch = ? AND state_sync_request_id < ?)
                  )
                RETURNING worker_id
                """,
                [
                    sync_epoch,
                    sync_request_id,
                    worker_id,
                    sync_epoch,
                    sync_epoch,
                    sync_request_id,
                ],
            ).fetchone()
            if result is not None:
                return True

            exists = cursor.execute(
                """
                SELECT 1 FROM workers
                WHERE worker_id = ? AND inactive_at IS NULL
                """,
                [worker_id],
            ).fetchone()
            if not exists:
                raise ValueError(f"Worker {worker_id} not found")
            return False

    def mark_as_stale(self, worker_id: str) -> bool:
        """Mark a worker as stale (for recovery purposes)."""
        with self._write_lock:
            cursor = self.get_cursor()
            # We could add a 'stale' column, but for now we'll use a very old heartbeat
            result = cursor.execute(
                """
                UPDATE workers
                SET last_heartbeat = TIMESTAMP '1970-01-01 00:00:00'
                WHERE worker_id = ? AND inactive_at IS NULL
                RETURNING worker_id
                """,
                [worker_id],
            )

            updated = result.fetchone() is not None
            if updated:
                logger.debug(f"Marked worker {worker_id} as stale")
            return updated

    def mark_inactive(self, worker_id: str) -> bool:
        """Mark a worker as inactive."""
        with self._write_lock:
            cursor = self.get_cursor()
            result = cursor.execute(
                """
                UPDATE workers
                SET accepting_new_requests = FALSE,
                    inactive_at = CURRENT_TIMESTAMP
                WHERE worker_id = ? AND inactive_at IS NULL
                RETURNING worker_id
                """,
                [worker_id],
            )

            updated = result.fetchone() is not None
            if updated:
                logger.debug(f"Marked worker {worker_id} as inactive")
                return True
            exists = cursor.execute(
                "SELECT 1 FROM workers WHERE worker_id = ?",
                [worker_id],
            ).fetchone()
            return exists is not None

    def get_stale_workers(self, recovery_time: int) -> list[Worker]:
        """Get workers that haven't updated since recovery started."""
        cursor = self.get_cursor()

        result = cursor.execute(
            f"""
            {_WORKER_SELECT}
            WHERE EXTRACT(epoch FROM workers.last_heartbeat) < ?
              AND workers.inactive_at IS NULL
            ORDER BY workers.last_heartbeat DESC
            """,
            [recovery_time],
        )

        return [self._row_to_model(row) for row in result.fetchall()]

    def cleanup_stale_workers(self, recovery_time: int) -> list[tuple[str, str]]:
        """
        Clean up workers that remained stale after recovery period.

        Returns:
            List of (worker_id, node_id) tuples that were cleaned up
        """
        with self._write_lock:
            cursor = self.get_cursor()
            rows = cursor.execute(
                """
                UPDATE workers
                SET inactive_at = CURRENT_TIMESTAMP,
                    accepting_new_requests = FALSE
                WHERE EXTRACT(epoch FROM last_heartbeat) < ?
                  AND inactive_at IS NULL
                RETURNING worker_id, node_id
                """,
                [recovery_time],
            ).fetchall()

        if rows:
            logger.info(f"Marked {len(rows)} stale workers as inactive")
        return rows

    def _row_to_model(self, row: tuple) -> Worker:
        """Convert a database row to Worker object."""
        memory_tier_state = None
        if row[13] is not None:
            memory_tier_state = WorkerMemoryTierState(
                stable_total_bytes=row[13],
                stable_used_bytes=row[14],
                preemptible_total_bytes=row[15],
                preemptible_marked_bytes=row[16],
                faults_per_sec=row[17] or 0.0,
                rehydrate_p99_ns=row[18] or 0,
                enable_preemptible=bool(row[19]),
                memory_tier_config_json=row[20] or "{}",
                snapshot_epoch_ns=row[21] or 0,
            )

        return Worker(
            worker_id=row[0],
            node_id=row[1],
            node_address=row[2],
            grpc_port=row[3],
            p2p_port=row[4],
            mem_pool_total_size=row[5],
            mem_pool_available_size=row[6],
            accepting_new_requests=row[7],
            registered_at=row[8],
            last_heartbeat=row[9],
            inactive_at=row[10],
            state_version=row[11],
            state_checksum=row[12] or "",
            memory_tier_state=memory_tier_state,
        )

    # ---------------------------------------------------------------------
    # Additional helper methods (used by test suite)
    # ---------------------------------------------------------------------

    def find_by_node_id(self, node_id: str) -> Worker | None:
        """Find the most recent worker registered on the given node_id."""
        cursor = self.get_cursor()
        row = cursor.execute(
            f"""
            {_WORKER_SELECT}
            WHERE workers.node_id = ?
            ORDER BY workers.registered_at DESC
            LIMIT 1
            """,
            [node_id],
        ).fetchone()

        if row:
            return self._row_to_model(row)
        return None

    def list_active(self, accepting_only: bool = False) -> list[Worker]:
        """Return active workers, optionally filtering by accepting status."""
        # Map accepting_only flag to existing include_unavailable parameter
        include_unavailable = not accepting_only
        return self.find_active(include_unavailable=include_unavailable)
