#  Copyright (c) 2025, StepCast Team.

"""Repository for worker data access."""

import time

from scstore.global_store.config import get_config
from scstore.global_store.models import Worker
from scstore.global_store.repositories.base import BaseRepository
from scstore.logger import init_logger

logger = init_logger(__name__)


class WorkerRepository(BaseRepository):
    """Repository for managing workers in the database."""

    def find_by_id(self, worker_id: str) -> Worker | None:
        """Find a worker by ID."""
        cursor = self.get_cursor()
        result = cursor.execute(
            "SELECT * FROM workers WHERE worker_id = ?", [worker_id]
        ).fetchone()

        if result:
            return self._row_to_model(result)
        return None

    def find_by_address_port(self, node_address: str, grpc_port: int) -> Worker | None:
        """Find a worker by address and port."""
        cursor = self.get_cursor()
        result = cursor.execute(
            """
            SELECT * FROM workers
            WHERE node_address = ? AND grpc_port = ?
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
                mem_pool_total_size, mem_pool_available_size, accepting_new_requests
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
            SET node_id = ?, p2p_port = ?,
                mem_pool_total_size = ?, mem_pool_available_size = ?,
                accepting_new_requests = ?,
                last_heartbeat = CURRENT_TIMESTAMP
            WHERE worker_id = ?
            RETURNING worker_id
            """,
            [
                worker.node_id,
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
        cursor = self.get_cursor()

        cutoff_time = time.time() - timeout_seconds

        result = cursor.execute(
            """
            DELETE FROM workers
            WHERE EXTRACT(epoch FROM last_heartbeat) < ?
            RETURNING worker_id, node_id
            """,
            [cutoff_time],
        )

        return result.fetchall()

    def find_active(self, include_unavailable: bool = False) -> list[Worker]:
        """Find all active workers."""
        cursor = self.get_cursor()
        config = get_config()

        query = """
            SELECT * FROM workers
            WHERE EXTRACT(epoch FROM last_heartbeat) > ?
        """
        # Use configured heartbeat timeout instead of hardcoded value
        timeout_seconds = config.heartbeat_timeout_ms / 1000.0
        params = [time.time() - timeout_seconds]

        if not include_unavailable:
            query += " AND accepting_new_requests = TRUE"

        query += " ORDER BY last_heartbeat DESC"

        result = cursor.execute(query, params)
        workers = [self._row_to_model(row) for row in result.fetchall()]
        return workers

    # ========== High Availability Methods ==========

    def list_all_workers(self) -> list[Worker]:
        """List all workers in the database."""
        cursor = self.get_cursor()
        result = cursor.execute("SELECT * FROM workers ORDER BY registered_at DESC")
        return [self._row_to_model(row) for row in result.fetchall()]

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
            """
            SELECT * FROM workers
            WHERE EXTRACT(epoch FROM last_heartbeat) < ?
            ORDER BY last_heartbeat DESC
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
        cursor = self.get_cursor()

        # Delete workers that haven't heartbeated since before recovery
        result = cursor.execute(
            """
            DELETE FROM workers
            WHERE EXTRACT(epoch FROM last_heartbeat) < ?
            RETURNING worker_id, node_id
            """,
            [recovery_time],
        )

        cleaned_up = result.fetchall()
        if cleaned_up:
            logger.info(f"Cleaned up {len(cleaned_up)} stale workers")
        return cleaned_up

    def _row_to_model(self, row: tuple) -> Worker:
        """Convert a database row to Worker object."""
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
        )

    # ---------------------------------------------------------------------
    # Additional helper methods (used by test suite)
    # ---------------------------------------------------------------------

    def find_by_node_id(self, node_id: str) -> Worker | None:
        """Find the most recent worker registered on the given node_id."""
        cursor = self.get_cursor()
        row = cursor.execute(
            """
            SELECT * FROM workers
            WHERE node_id = ?
            ORDER BY registered_at DESC
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
