#  Copyright (c) 2025-2026, TensorCast Team.

"""Repository for worker data access."""

import time

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

    def find_by_id(self, worker_id: str) -> Worker | None:
        """Find a worker by ID."""
        cursor = self.get_cursor()
        result = cursor.execute(
            f"{_WORKER_SELECT} WHERE workers.worker_id = ?",
            [worker_id],
        ).fetchone()

        if result:
            return self._row_to_model(result)
        return None

    def find_by_address_port(self, node_address: str, grpc_port: int) -> Worker | None:
        """Find a worker by address and port."""
        cursor = self.get_cursor()
        result = cursor.execute(
            f"""
            {_WORKER_SELECT}
            WHERE workers.node_address = ? AND workers.grpc_port = ?
            """,
            [node_address, grpc_port],
        ).fetchone()

        if result:
            return self._row_to_model(result)
        return None

    def create(self, worker: Worker) -> Worker:
        """Create a new worker."""
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
        existing = self.find_by_id(worker.worker_id)
        if existing:
            return self.update(worker)
        else:
            return self.create(worker)

    def update(self, worker: Worker) -> Worker:
        """Update an existing worker."""
        cursor = self.get_cursor()

        result = cursor.execute(
            """
            UPDATE workers
            SET node_id = ?, node_address = ?, grpc_port = ?, p2p_port = ?,
                mem_pool_total_size = ?, mem_pool_available_size = ?,
                accepting_new_requests = ?,
                last_heartbeat = CURRENT_TIMESTAMP
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
        cursor = self.get_cursor()

        result = cursor.execute(
            """
            UPDATE workers
            SET last_heartbeat = CURRENT_TIMESTAMP,
                mem_pool_available_size = ?,
                accepting_new_requests = ?
            WHERE worker_id = ?
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
            RETURNING workers.worker_id
        """

        try:
            cursor = self.get_cursor()
            result = cursor.execute(sql, params)
            updated_rows = result.fetchall()
            return len(updated_rows)
        except Exception:
            logger.exception("Batch heartbeat update failed")
            return 0

    def delete(self, worker_id: str) -> bool:
        """Delete a worker."""
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

    def delete_inactive(self, timeout_seconds: float) -> list[tuple[str, str]]:
        """
        Delete inactive workers and return their IDs and node IDs.

        Returns:
            List of (worker_id, node_id) tuples
        """
        cutoff_time = time.time() - timeout_seconds

        with self.transaction() as cursor:
            rows = cursor.execute(
                """
                SELECT worker_id, node_id FROM workers
                WHERE EXTRACT(epoch FROM last_heartbeat) < ?
                """,
                [cutoff_time],
            ).fetchall()

            if rows:
                cursor.execute(
                    """
                    DELETE FROM workers
                    WHERE EXTRACT(epoch FROM last_heartbeat) < ?
                    """,
                    [cutoff_time],
                )

        return rows

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
        cursor = cursor if cursor is not None else self.get_cursor()
        result = cursor.execute(
            "SELECT COALESCE(state_checksum, '') FROM workers WHERE worker_id = ?",
            [worker_id],
        ).fetchone()
        if not result:
            raise ValueError(f"Worker {worker_id} not found")
        return result[0] or ""

    def ensure_state_version(
        self, worker_id: str, cursor: DuckDBPyConnection | None = None
    ) -> int:
        """Ensure the worker has a non-zero state version."""
        cursor = cursor if cursor is not None else self.get_cursor()
        result = cursor.execute(
            "SELECT state_version FROM workers WHERE worker_id = ?",
            [worker_id],
        ).fetchone()
        if not result:
            raise ValueError(f"Worker {worker_id} not found")
        current_version = int(result[0] or 0)
        if current_version <= 0:
            current_version = 1
            cursor.execute(
                "UPDATE workers SET state_version = ? WHERE worker_id = ?",
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
        cursor = cursor if cursor is not None else self.get_cursor()
        result = cursor.execute(
            """
            UPDATE workers
            SET state_version = ?, state_checksum = ?
            WHERE worker_id = ?
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
        cursor = cursor if cursor is not None else self.get_cursor()
        result = cursor.execute(
            """
            UPDATE workers
            SET state_checksum = ?
            WHERE worker_id = ?
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
        cursor = cursor if cursor is not None else self.get_cursor()
        result = cursor.execute(
            """
            UPDATE workers
            SET state_sync_epoch = ?, state_sync_request_id = ?
            WHERE worker_id = ?
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
            "SELECT 1 FROM workers WHERE worker_id = ?",
            [worker_id],
        ).fetchone()
        if not exists:
            raise ValueError(f"Worker {worker_id} not found")
        return False

    def mark_as_stale(self, worker_id: str) -> bool:
        """Mark a worker as stale (for recovery purposes)."""
        cursor = self.get_cursor()

        # We could add a 'stale' column, but for now we'll use a very old heartbeat
        result = cursor.execute(
            """
            UPDATE workers
            SET last_heartbeat = TIMESTAMP '1970-01-01 00:00:00'
            WHERE worker_id = ?
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
        cursor = self.get_cursor()

        result = cursor.execute(
            """
            UPDATE workers
            SET accepting_new_requests = FALSE,
                last_heartbeat = TIMESTAMP '1970-01-01 00:00:00'
            WHERE worker_id = ?
            RETURNING worker_id
            """,
            [worker_id],
        )

        updated = result.fetchone() is not None
        if updated:
            logger.debug(f"Marked worker {worker_id} as inactive")
        return updated

    def get_stale_workers(self, recovery_time: int) -> list[Worker]:
        """Get workers that haven't updated since recovery started."""
        cursor = self.get_cursor()

        result = cursor.execute(
            f"""
            {_WORKER_SELECT}
            WHERE EXTRACT(epoch FROM workers.last_heartbeat) < ?
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
        with self.transaction() as cursor:
            # Select first to avoid fetching a large RETURNING result set
            rows = cursor.execute(
                """
                SELECT worker_id, node_id FROM workers
                WHERE EXTRACT(epoch FROM last_heartbeat) < ?
                """,
                [recovery_time],
            ).fetchall()

            if rows:
                cursor.execute(
                    """
                    DELETE FROM workers
                    WHERE EXTRACT(epoch FROM last_heartbeat) < ?
                    """,
                    [recovery_time],
                )

        if rows:
            logger.info(f"Cleaned up {len(rows)} stale workers")
        return rows

    def _row_to_model(self, row: tuple) -> Worker:
        """Convert a database row to Worker object."""
        memory_tier_state = None
        if row[12] is not None:
            memory_tier_state = WorkerMemoryTierState(
                stable_total_bytes=row[12],
                stable_used_bytes=row[13],
                preemptible_total_bytes=row[14],
                preemptible_marked_bytes=row[15],
                faults_per_sec=row[16] or 0.0,
                rehydrate_p99_ns=row[17] or 0,
                enable_preemptible=bool(row[18]),
                memory_tier_config_json=row[19] or "{}",
                snapshot_epoch_ns=row[20] or 0,
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
            state_version=row[10],
            state_checksum=row[11] or "",
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
