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

_CONNECTION_LOCKS_GUARD = threading.RLock()
_CONNECTION_WRITE_LOCKS: dict[int, threading.RLock] = {}


def _shared_connection_write_lock(connection: DuckDBPyConnection) -> threading.RLock:
    key = id(connection)
    with _CONNECTION_LOCKS_GUARD:
        lock = _CONNECTION_WRITE_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _CONNECTION_WRITE_LOCKS[key] = lock
        return lock


_WORKER_SELECT = """
    SELECT
        workers.worker_id,
        workers.daemon_id,
        workers.node_id,
        workers.node_address,
        workers.grpc_port,
        workers.p2p_port,
        workers.mem_pool_total_size,
        COALESCE(worker_liveness.mem_pool_available_size, workers.mem_pool_total_size) AS mem_pool_available_size,
        COALESCE(worker_liveness.accepting_new_requests, TRUE) AS accepting_new_requests,
        COALESCE(worker_liveness.capability_flags, 0) AS capability_flags,
        workers.registered_at,
        worker_liveness.last_heartbeat,
        workers.inactive_at,
        COALESCE(worker_reconcile_state.state_version, 1) AS state_version,
        COALESCE(worker_reconcile_state.state_checksum, '') AS state_checksum,
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
    LEFT JOIN worker_liveness
        ON worker_liveness.worker_id = workers.worker_id
    LEFT JOIN worker_reconcile_state
        ON worker_reconcile_state.worker_id = workers.worker_id
    LEFT JOIN node_memory_tier_latest AS mt
        ON mt.node_id = workers.node_id
"""


class WorkerRepository(BaseRepository):
    """Repository for managing workers in the database."""

    def __init__(self, connection: DuckDBPyConnection):
        super().__init__(connection)
        self._write_lock = _shared_connection_write_lock(connection)

    def _write_guard(self, cursor: DuckDBPyConnection | None):
        return nullcontext() if cursor is not None else self._write_lock

    @contextmanager
    def transaction(self):
        with self._write_lock, super().transaction() as cursor:
            yield cursor

    @staticmethod
    def _is_write_conflict(exc: Exception) -> bool:
        message = str(exc).lower()
        conflict_markers = (
            "write-write conflict",
            "conflict on tuple deletion",
            "conflict on update",
            "failed to commit: write-write conflict on key",
            "transactioncontext error: conflict",
        )
        return any(marker in message for marker in conflict_markers)

    def find_by_id(
        self, worker_id: str, *, include_inactive: bool = False
    ) -> Worker | None:
        """Find a worker by ID."""
        with self._write_lock:
            cursor = self.get_cursor()
            try:
                result = cursor.execute(
                    f"{_WORKER_SELECT} WHERE workers.worker_id = ?"
                    + ("" if include_inactive else " AND workers.inactive_at IS NULL"),
                    [worker_id],
                ).fetchone()

                if result:
                    return self._row_to_model(result)
                return None
            finally:
                cursor.close()

    def find_by_daemon_id(
        self, daemon_id: str, *, include_inactive: bool = False
    ) -> Worker | None:
        """Find a worker by stable daemon_id."""
        if not daemon_id:
            return None
        with self._write_lock:
            cursor = self.get_cursor()
            try:
                result = cursor.execute(
                    f"{_WORKER_SELECT} WHERE workers.daemon_id = ?"
                    + ("" if include_inactive else " AND workers.inactive_at IS NULL"),
                    [daemon_id],
                ).fetchone()

                if result:
                    return self._row_to_model(result)
                return None
            finally:
                cursor.close()

    def find_by_address_port(
        self, node_address: str, grpc_port: int, *, include_inactive: bool = True
    ) -> Worker | None:
        """Find a worker by address and port."""
        with self._write_lock:
            cursor = self.get_cursor()
            try:
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
            finally:
                cursor.close()

    def create(self, worker: Worker) -> Worker:
        """Create a new worker."""
        with self.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO workers (
                    worker_id, daemon_id, node_id, node_address, grpc_port, p2p_port,
                    mem_pool_total_size
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    worker.worker_id,
                    worker.daemon_id,
                    worker.node_id,
                    worker.node_address,
                    worker.grpc_port,
                    worker.p2p_port,
                    worker.mem_pool_total_size,
                ],
            )
            cursor.execute(
                """
                INSERT INTO worker_liveness (
                    worker_id,
                    last_heartbeat,
                    mem_pool_available_size,
                    accepting_new_requests,
                    capability_flags,
                    updated_at
                ) VALUES (?, now(), ?, ?, ?, now())
                """,
                [
                    worker.worker_id,
                    worker.mem_pool_available_size,
                    worker.accepting_new_requests,
                    int(worker.capability_flags),
                ],
            )
            cursor.execute(
                """
                INSERT INTO worker_reconcile_state (
                    worker_id,
                    generation,
                    request_seq,
                    state_version,
                    state_checksum,
                    last_reconcile_result,
                    updated_at
                ) VALUES (?, 1, 0, ?, ?, '', now())
                """,
                [
                    worker.worker_id,
                    max(1, int(worker.state_version)),
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
        with self.transaction() as cursor:
            result = cursor.execute(
                """
                UPDATE workers
                SET daemon_id = ?, node_id = ?, node_address = ?, grpc_port = ?, p2p_port = ?,
                    mem_pool_total_size = ?, inactive_at = NULL
                WHERE worker_id = ?
                RETURNING worker_id
                """,
                [
                    worker.daemon_id,
                    worker.node_id,
                    worker.node_address,
                    worker.grpc_port,
                    worker.p2p_port,
                    worker.mem_pool_total_size,
                    worker.worker_id,
                ],
            )
            if result.fetchone() is None:
                raise ValueError(f"Worker {worker.worker_id} not found")

            liveness_updated = cursor.execute(
                """
                UPDATE worker_liveness
                SET last_heartbeat = GREATEST(last_heartbeat, now()),
                    mem_pool_available_size = ?,
                    accepting_new_requests = ?,
                    capability_flags = ?,
                    updated_at = now()
                WHERE worker_id = ?
                RETURNING worker_id
                """,
                [
                    worker.mem_pool_available_size,
                    worker.accepting_new_requests,
                    int(worker.capability_flags),
                    worker.worker_id,
                ],
            ).fetchone()
            if liveness_updated is None:
                cursor.execute(
                    """
                    INSERT INTO worker_liveness (
                        worker_id,
                        last_heartbeat,
                        mem_pool_available_size,
                        accepting_new_requests,
                        capability_flags,
                        updated_at
                    ) VALUES (?, now(), ?, ?, ?, now())
                    """,
                    [
                        worker.worker_id,
                        worker.mem_pool_available_size,
                        worker.accepting_new_requests,
                        int(worker.capability_flags),
                    ],
                )

            if reset_state_tracking:
                reconcile_updated = cursor.execute(
                    """
                    UPDATE worker_reconcile_state
                    SET generation = generation + 1,
                        request_seq = 0,
                        state_version = 1,
                        state_checksum = '',
                        last_reconcile_result = '',
                        updated_at = now()
                    WHERE worker_id = ?
                    RETURNING worker_id
                    """,
                    [worker.worker_id],
                ).fetchone()
                if reconcile_updated is None:
                    cursor.execute(
                        """
                        INSERT INTO worker_reconcile_state (
                            worker_id,
                            generation,
                            request_seq,
                            state_version,
                            state_checksum,
                            last_reconcile_result,
                            updated_at
                        ) VALUES (?, 1, 0, 1, '', '', now())
                        """,
                        [worker.worker_id],
                    )
            else:
                reconcile_updated = cursor.execute(
                    """
                    UPDATE worker_reconcile_state
                    SET generation = generation + 1,
                        request_seq = 0,
                        updated_at = now()
                    WHERE worker_id = ?
                    RETURNING worker_id
                    """,
                    [worker.worker_id],
                ).fetchone()
                if reconcile_updated is None:
                    cursor.execute(
                        """
                        INSERT INTO worker_reconcile_state (
                            worker_id,
                            generation,
                            request_seq,
                            state_version,
                            state_checksum,
                            last_reconcile_result,
                            updated_at
                        ) VALUES (?, 1, 0, 1, '', '', now())
                        """,
                        [worker.worker_id],
                    )
        return worker

    def update_heartbeat(
        self,
        worker_id: str,
        mem_pool_available_size: int,
        accepting_new_requests: bool,
        capability_flags: int | None = None,
    ) -> bool:
        """Update worker heartbeat and status."""
        with self.transaction() as cursor:
            exists = cursor.execute(
                """
                SELECT 1
                FROM workers
                WHERE workers.worker_id = ?
                  AND workers.inactive_at IS NULL
                """,
                [worker_id],
            ).fetchone()
            if exists is None:
                return False

            updated = cursor.execute(
                """
                UPDATE worker_liveness
                SET last_heartbeat = GREATEST(last_heartbeat, now()),
                    mem_pool_available_size = ?,
                    accepting_new_requests = ?,
                    capability_flags = COALESCE(?, capability_flags, 0),
                    updated_at = now()
                WHERE worker_id = ?
                RETURNING worker_id
                """,
                [
                    mem_pool_available_size,
                    accepting_new_requests,
                    int(capability_flags) if capability_flags is not None else None,
                    worker_id,
                ],
            ).fetchone()
            if updated is None:
                cursor.execute(
                    """
                    INSERT INTO worker_liveness (
                        worker_id,
                        last_heartbeat,
                        mem_pool_available_size,
                        accepting_new_requests,
                        capability_flags,
                        updated_at
                    ) VALUES (?, now(), ?, ?, ?, now())
                    """,
                    [
                        worker_id,
                        mem_pool_available_size,
                        accepting_new_requests,
                        int(capability_flags) if capability_flags is not None else 0,
                    ],
                )
            return True

    def update_capability_flags(self, worker_id: str, capability_flags: int) -> bool:
        """Update capability flags only when changed."""
        with self.transaction() as cursor:
            row = cursor.execute(
                """
                SELECT
                    COALESCE(worker_liveness.last_heartbeat, now()),
                    COALESCE(worker_liveness.mem_pool_available_size, workers.mem_pool_total_size),
                    COALESCE(worker_liveness.accepting_new_requests, TRUE),
                    COALESCE(worker_liveness.capability_flags, 0)
                FROM workers
                LEFT JOIN worker_liveness
                  ON worker_liveness.worker_id = workers.worker_id
                WHERE workers.worker_id = ?
                  AND workers.inactive_at IS NULL
                """,
                [worker_id],
            ).fetchone()
            if row is None:
                return False
            if int(row[3] or 0) == int(capability_flags):
                return False
            updated = cursor.execute(
                """
                UPDATE worker_liveness
                SET capability_flags = ?,
                    updated_at = now()
                WHERE worker_id = ?
                RETURNING worker_id
                """,
                [int(capability_flags), worker_id],
            ).fetchone()
            if updated is None:
                cursor.execute(
                    """
                    INSERT INTO worker_liveness (
                        worker_id,
                        last_heartbeat,
                        mem_pool_available_size,
                        accepting_new_requests,
                        capability_flags,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, now())
                    """,
                    [
                        worker_id,
                        row[0],
                        int(row[1] or 0),
                        bool(row[2]),
                        int(capability_flags),
                    ],
                )
            return True

    def batch_update_heartbeats(
        self, updates: list[tuple[str, int, bool, int | None]]
    ) -> int:
        """
        Batch update worker heartbeats and statuses with update-first semantics.

        Args:
            updates: List of (
                worker_id,
                mem_pool_available_size,
                accepting_new_requests,
                capability_flags,
            ).

        Returns:
            Number of successfully updated workers.
        """
        if not updates:
            return 0

        # Build VALUES clause dynamically –  one tuple per update
        # DuckDB supports `UPDATE … FROM (VALUES …) AS alias(col1,col2,…)` pattern.
        placeholders = ", ".join(["(?, ?, ?, ?)"] * len(updates))
        values_sql = f"VALUES {placeholders}"

        # Flatten parameters for executemany-style substitution
        params: list = []
        for (
            worker_id,
            mem_pool_available_size,
            accepting_new_requests,
            capability_flags,
        ) in updates:
            params.extend(
                [
                    worker_id,
                    mem_pool_available_size,
                    accepting_new_requests,
                    int(capability_flags) if capability_flags is not None else None,
                ]
            )

        update_sql = f"""
            WITH upd(worker_id, mem_pool_available_size, accepting_new_requests, capability_flags) AS (
                {values_sql}
            ),
            active AS (
                SELECT
                    workers.worker_id,
                    upd.mem_pool_available_size,
                    upd.accepting_new_requests,
                    COALESCE(upd.capability_flags, worker_liveness.capability_flags, 0) AS capability_flags
                FROM upd
                JOIN workers
                  ON workers.worker_id = upd.worker_id
                LEFT JOIN worker_liveness
                  ON worker_liveness.worker_id = workers.worker_id
                WHERE workers.inactive_at IS NULL
            )
            UPDATE worker_liveness
            SET last_heartbeat = GREATEST(worker_liveness.last_heartbeat, now()),
                mem_pool_available_size = active.mem_pool_available_size,
                accepting_new_requests = active.accepting_new_requests,
                capability_flags = active.capability_flags,
                updated_at = now()
            FROM active
            WHERE worker_liveness.worker_id = active.worker_id
            RETURNING worker_liveness.worker_id
        """
        insert_sql = f"""
            WITH upd(worker_id, mem_pool_available_size, accepting_new_requests, capability_flags) AS (
                {values_sql}
            ),
            active AS (
                SELECT
                    workers.worker_id,
                    upd.mem_pool_available_size,
                    upd.accepting_new_requests,
                    COALESCE(upd.capability_flags, worker_liveness.capability_flags, 0) AS capability_flags
                FROM upd
                JOIN workers
                  ON workers.worker_id = upd.worker_id
                LEFT JOIN worker_liveness
                  ON worker_liveness.worker_id = workers.worker_id
                WHERE workers.inactive_at IS NULL
            ),
            missing AS (
                SELECT
                    active.worker_id,
                    active.mem_pool_available_size,
                    active.accepting_new_requests,
                    active.capability_flags
                FROM active
                LEFT JOIN worker_liveness wl
                  ON wl.worker_id = active.worker_id
                WHERE wl.worker_id IS NULL
            )
            INSERT INTO worker_liveness (
                worker_id,
                last_heartbeat,
                mem_pool_available_size,
                accepting_new_requests,
                capability_flags,
                updated_at
            )
            SELECT
                missing.worker_id,
                now(),
                missing.mem_pool_available_size,
                missing.accepting_new_requests,
                missing.capability_flags,
                now()
            FROM missing
            RETURNING worker_id
        """

        worker_ids = [worker_id for worker_id, _, _, _ in updates]
        unique_worker_ids = list(dict.fromkeys(worker_ids))
        seen_worker_ids: set[str] = set()
        duplicate_worker_ids: set[str] = set()
        for worker_id in worker_ids:
            if worker_id in seen_worker_ids:
                duplicate_worker_ids.add(worker_id)
                continue
            seen_worker_ids.add(worker_id)

        try:
            with self._write_lock:
                cursor = self.get_cursor()
                try:
                    updated_rows = cursor.execute(update_sql, params).fetchall()
                    inserted_rows = cursor.execute(insert_sql, params).fetchall()
                    return len(updated_rows) + len(inserted_rows)
                finally:
                    cursor.close()
        except Exception as exc:
            is_conflict = self._is_write_conflict(exc)
            logger.exception(
                "Batch heartbeat update failed conflict=%s batch_size=%d unique_workers=%d "
                "duplicate_worker_ids=%d worker_sample=%s",
                is_conflict,
                len(worker_ids),
                len(unique_worker_ids),
                len(duplicate_worker_ids),
                unique_worker_ids[:8],
            )
            return 0

    def delete(self, worker_id: str) -> bool:
        """Delete a worker."""
        with self.transaction() as cursor:
            cursor.execute(
                """
                DELETE FROM worker_liveness
                WHERE worker_id = ?
                """,
                [worker_id],
            )
            cursor.execute(
                """
                DELETE FROM worker_reconcile_state
                WHERE worker_id = ?
                """,
                [worker_id],
            )
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
        try:
            with self.transaction() as cursor:
                rows = cursor.execute(
                    """
                    WITH stale AS (
                        SELECT workers.worker_id, workers.node_id
                        FROM workers
                        JOIN worker_liveness
                          ON worker_liveness.worker_id = workers.worker_id
                        WHERE workers.inactive_at IS NULL
                          AND EXTRACT(epoch FROM worker_liveness.last_heartbeat) < ?
                    )
                    UPDATE workers
                    SET inactive_at = now()
                    WHERE workers.worker_id IN (SELECT worker_id FROM stale)
                    RETURNING workers.worker_id, workers.node_id
                    """,
                    [cutoff_time],
                ).fetchall()
                if rows:
                    worker_ids = [row[0] for row in rows]
                    placeholders = ", ".join(["?"] * len(worker_ids))
                    cursor.execute(
                        f"""
                        UPDATE worker_liveness
                        SET accepting_new_requests = FALSE,
                            updated_at = now()
                        WHERE worker_id IN ({placeholders})
                        """,
                        worker_ids,
                    )
                return rows
        except Exception as exc:
            is_conflict = self._is_write_conflict(exc)
            logger.exception(
                "mark_inactive_by_timeout failed conflict=%s timeout_seconds=%.3f cutoff_ts=%.6f",
                is_conflict,
                timeout_seconds,
                cutoff_time,
            )
            raise

    def list_timeout_candidates(
        self, timeout_seconds: float
    ) -> list[tuple[str, str, str]]:
        """List active workers whose heartbeat timestamp is past timeout cutoff."""
        cutoff_time = time.time() - timeout_seconds
        with self._write_lock:
            cursor = self.get_cursor()
            try:
                rows = cursor.execute(
                    """
                    SELECT workers.worker_id, workers.node_id, COALESCE(workers.daemon_id, '')
                    FROM workers
                    JOIN worker_liveness
                      ON worker_liveness.worker_id = workers.worker_id
                    WHERE workers.inactive_at IS NULL
                      AND EXTRACT(epoch FROM worker_liveness.last_heartbeat) < ?
                    ORDER BY worker_liveness.last_heartbeat ASC
                    """,
                    [cutoff_time],
                ).fetchall()
                return [
                    (str(worker_id), str(node_id), str(daemon_id or "").strip())
                    for worker_id, node_id, daemon_id in rows
                ]
            finally:
                cursor.close()

    def mark_inactive_if_timed_out(
        self, worker_id: str, timeout_seconds: float
    ) -> tuple[str, str] | None:
        """Mark a single worker inactive if it is still timed out."""
        cutoff_time = time.time() - timeout_seconds
        with self.transaction() as cursor:
            row = cursor.execute(
                """
                UPDATE workers
                SET inactive_at = now()
                WHERE workers.worker_id = ?
                  AND workers.inactive_at IS NULL
                  AND EXISTS (
                      SELECT 1
                      FROM worker_liveness
                      WHERE worker_liveness.worker_id = workers.worker_id
                        AND EXTRACT(epoch FROM worker_liveness.last_heartbeat) < ?
                  )
                RETURNING workers.worker_id, workers.node_id
                """,
                [worker_id, cutoff_time],
            ).fetchone()
            if row is None:
                return None
            cursor.execute(
                """
                UPDATE worker_liveness
                SET accepting_new_requests = FALSE,
                    updated_at = now()
                WHERE worker_id = ?
                """,
                [worker_id],
            )
            return str(row[0]), str(row[1])

    def delete_inactive(self, timeout_seconds: float) -> list[tuple[str, str]]:
        return self.mark_inactive_by_timeout(timeout_seconds)

    def find_active(self, include_unavailable: bool = False) -> list[Worker]:
        """Find all active workers."""
        with self._write_lock:
            cursor = self.get_cursor()
            try:
                try:
                    config = get_config()
                except RuntimeError:
                    config = GlobalStoreConfig()

                query = f"""
                    {_WORKER_SELECT}
                    WHERE EXTRACT(epoch FROM worker_liveness.last_heartbeat) > ?
                      AND workers.inactive_at IS NULL
                """
                # Use configured heartbeat timeout instead of hardcoded value
                timeout_seconds = config.heartbeat_timeout_ms / 1000.0
                params = [time.time() - timeout_seconds]

                if not include_unavailable:
                    query += " AND worker_liveness.accepting_new_requests = TRUE"

                query += " ORDER BY worker_liveness.last_heartbeat DESC"

                result = cursor.execute(query, params)
                workers = [self._row_to_model(row) for row in result.fetchall()]
                return workers
            finally:
                cursor.close()

    # ========== High Availability Methods ==========

    def list_all_workers(self) -> list[Worker]:
        """List all workers in the database."""
        with self._write_lock:
            cursor = self.get_cursor()
            try:
                result = cursor.execute(
                    f"{_WORKER_SELECT} ORDER BY workers.registered_at DESC"
                )
                return [self._row_to_model(row) for row in result.fetchall()]
            finally:
                cursor.close()

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
            owns_cursor = cursor is None
            cursor = cursor if cursor is not None else self.get_cursor()
            try:
                result = cursor.execute(
                    """
                    SELECT COALESCE(worker_reconcile_state.state_checksum, '')
                    FROM workers
                    LEFT JOIN worker_reconcile_state
                      ON worker_reconcile_state.worker_id = workers.worker_id
                    WHERE workers.worker_id = ? AND workers.inactive_at IS NULL
                    """,
                    [worker_id],
                ).fetchone()
                if not result:
                    raise ValueError(f"Worker {worker_id} not found")
                return result[0] or ""
            finally:
                if owns_cursor:
                    cursor.close()

    def ensure_state_version(
        self, worker_id: str, cursor: DuckDBPyConnection | None = None
    ) -> int:
        """Ensure the worker has a non-zero state version."""
        if cursor is None:
            with self.transaction() as tx_cursor:
                return self.ensure_state_version(worker_id, tx_cursor)
        with self._write_guard(cursor):
            cursor = cursor if cursor is not None else self.get_cursor()
            result = cursor.execute(
                """
                SELECT
                    workers.worker_id,
                    worker_reconcile_state.worker_id,
                    worker_reconcile_state.state_version,
                    worker_reconcile_state.generation,
                    worker_reconcile_state.request_seq,
                    worker_reconcile_state.state_checksum
                FROM workers
                LEFT JOIN worker_reconcile_state
                  ON worker_reconcile_state.worker_id = workers.worker_id
                WHERE workers.worker_id = ? AND workers.inactive_at IS NULL
                """,
                [worker_id],
            ).fetchone()
            if not result:
                raise ValueError(f"Worker {worker_id} not found")

            reconcile_state_exists = result[1] is not None
            current_version = int(result[2] or 0)

            # Fast path: reconcile state is already initialized and valid.
            # Avoid writing on heartbeat/list paths to reduce write contention.
            if reconcile_state_exists and current_version > 0:
                return current_version

            current_version = max(1, current_version)
            updated = cursor.execute(
                """
                UPDATE worker_reconcile_state
                SET state_version = ?,
                    updated_at = now()
                WHERE worker_id = ?
                RETURNING worker_id
                """,
                [current_version, worker_id],
            ).fetchone()
            if updated is None:
                cursor.execute(
                    """
                    INSERT INTO worker_reconcile_state (
                        worker_id,
                        generation,
                        request_seq,
                        state_version,
                        state_checksum,
                        last_reconcile_result,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, '', now())
                    """,
                    [
                        worker_id,
                        int(result[3] or 1),
                        int(result[4] or 0),
                        current_version,
                        result[5] or "",
                    ],
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
        if cursor is None:
            with self.transaction() as tx_cursor:
                self.update_state_version_and_checksum(
                    worker_id,
                    state_version,
                    state_checksum,
                    cursor=tx_cursor,
                )
            return
        with self._write_guard(cursor):
            cursor = cursor if cursor is not None else self.get_cursor()
            current = cursor.execute(
                """
                SELECT
                    workers.worker_id,
                    COALESCE(worker_reconcile_state.generation, 1),
                    COALESCE(worker_reconcile_state.request_seq, 0)
                FROM workers
                LEFT JOIN worker_reconcile_state
                  ON worker_reconcile_state.worker_id = workers.worker_id
                WHERE workers.worker_id = ? AND workers.inactive_at IS NULL
                """,
                [worker_id],
            ).fetchone()
            if current is None:
                raise ValueError(f"Worker {worker_id} not found")
            updated = cursor.execute(
                """
                UPDATE worker_reconcile_state
                SET state_version = ?,
                    state_checksum = ?,
                    updated_at = now()
                WHERE worker_id = ?
                RETURNING worker_id
                """,
                [int(state_version), state_checksum, worker_id],
            ).fetchone()
            if updated is None:
                cursor.execute(
                    """
                    INSERT INTO worker_reconcile_state (
                        worker_id,
                        generation,
                        request_seq,
                        state_version,
                        state_checksum,
                        last_reconcile_result,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, '', now())
                    """,
                    [
                        worker_id,
                        int(current[1] or 1),
                        int(current[2] or 0),
                        int(state_version),
                        state_checksum,
                    ],
                )

    def update_state_checksum(
        self,
        worker_id: str,
        state_checksum: str,
        cursor: DuckDBPyConnection | None = None,
    ) -> None:
        """Update the worker's state checksum."""
        if cursor is None:
            with self.transaction() as tx_cursor:
                self.update_state_checksum(
                    worker_id,
                    state_checksum,
                    cursor=tx_cursor,
                )
            return
        with self._write_guard(cursor):
            cursor = cursor if cursor is not None else self.get_cursor()
            row = cursor.execute(
                """
                SELECT
                    workers.worker_id,
                    COALESCE(worker_reconcile_state.generation, 1),
                    COALESCE(worker_reconcile_state.request_seq, 0),
                    COALESCE(worker_reconcile_state.state_version, 1)
                FROM workers
                LEFT JOIN worker_reconcile_state
                  ON worker_reconcile_state.worker_id = workers.worker_id
                WHERE workers.worker_id = ? AND workers.inactive_at IS NULL
                """,
                [worker_id],
            ).fetchone()
            if row is None:
                raise ValueError(f"Worker {worker_id} not found")
            updated = cursor.execute(
                """
                UPDATE worker_reconcile_state
                SET state_checksum = ?,
                    updated_at = now()
                WHERE worker_id = ?
                RETURNING worker_id
                """,
                [state_checksum, worker_id],
            ).fetchone()
            if updated is None:
                cursor.execute(
                    """
                    INSERT INTO worker_reconcile_state (
                        worker_id,
                        generation,
                        request_seq,
                        state_version,
                        state_checksum,
                        last_reconcile_result,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, '', now())
                    """,
                    [
                        worker_id,
                        int(row[1] or 1),
                        int(row[2] or 0),
                        int(row[3] or 1),
                        state_checksum,
                    ],
                )

    def get_reconcile_cursor(
        self,
        worker_id: str,
        cursor: DuckDBPyConnection | None = None,
    ) -> tuple[int, int]:
        with self._write_guard(cursor):
            owns_cursor = cursor is None
            cursor = cursor if cursor is not None else self.get_cursor()
            try:
                row = cursor.execute(
                    """
                    SELECT
                        COALESCE(worker_reconcile_state.generation, 1),
                        COALESCE(worker_reconcile_state.request_seq, 0)
                    FROM workers
                    LEFT JOIN worker_reconcile_state
                      ON worker_reconcile_state.worker_id = workers.worker_id
                    WHERE workers.worker_id = ? AND workers.inactive_at IS NULL
                    """,
                    [worker_id],
                ).fetchone()
                if row is None:
                    raise ValueError(f"Worker {worker_id} not found")
                return int(row[0] or 1), int(row[1] or 0)
            finally:
                if owns_cursor:
                    cursor.close()

    def update_reconcile_cursor(
        self,
        worker_id: str,
        *,
        generation: int,
        request_seq: int,
        state_version: int,
        state_checksum: str,
        last_reconcile_result: str,
        cursor: DuckDBPyConnection | None = None,
    ) -> None:
        if cursor is None:
            with self.transaction() as tx_cursor:
                self.update_reconcile_cursor(
                    worker_id,
                    generation=generation,
                    request_seq=request_seq,
                    state_version=state_version,
                    state_checksum=state_checksum,
                    last_reconcile_result=last_reconcile_result,
                    cursor=tx_cursor,
                )
            return
        with self._write_guard(cursor):
            cursor = cursor if cursor is not None else self.get_cursor()
            exists = cursor.execute(
                """
                SELECT 1 FROM workers
                WHERE worker_id = ? AND inactive_at IS NULL
                """,
                [worker_id],
            ).fetchone()
            if exists is None:
                raise ValueError(f"Worker {worker_id} not found")
            try:
                updated = cursor.execute(
                    """
                    UPDATE worker_reconcile_state
                    SET generation = ?,
                        request_seq = ?,
                        state_version = ?,
                        state_checksum = ?,
                        last_reconcile_result = ?,
                        updated_at = now()
                    WHERE worker_id = ?
                    RETURNING worker_id
                    """,
                    [
                        int(generation),
                        int(request_seq),
                        int(state_version),
                        state_checksum,
                        last_reconcile_result,
                        worker_id,
                    ],
                ).fetchone()
                if updated is None:
                    cursor.execute(
                        """
                        INSERT INTO worker_reconcile_state (
                            worker_id,
                            generation,
                            request_seq,
                            state_version,
                            state_checksum,
                            last_reconcile_result,
                            updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, now())
                        """,
                        [
                            worker_id,
                            int(generation),
                            int(request_seq),
                            int(state_version),
                            state_checksum,
                            last_reconcile_result,
                        ],
                    )
            except Exception as exc:
                if "conflict" in str(exc).lower():
                    logger.warning(
                        "update_reconcile_cursor conflict worker_id=%s generation=%s request_seq=%s state_version=%s result=%s",
                        worker_id,
                        int(generation),
                        int(request_seq),
                        int(state_version),
                        last_reconcile_result,
                    )
                raise

    def mark_as_stale(self, worker_id: str) -> bool:
        """Mark a worker as stale (for recovery purposes)."""
        with self.transaction() as cursor:
            result = cursor.execute(
                """
                UPDATE worker_liveness
                SET last_heartbeat = TIMESTAMP '1970-01-01 00:00:00',
                    updated_at = now()
                WHERE worker_id = ?
                  AND EXISTS (
                    SELECT 1 FROM workers
                    WHERE workers.worker_id = worker_liveness.worker_id
                      AND workers.inactive_at IS NULL
                  )
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
        with self.transaction() as cursor:
            result = cursor.execute(
                """
                UPDATE workers
                SET inactive_at = now()
                WHERE worker_id = ? AND inactive_at IS NULL
                RETURNING worker_id
                """,
                [worker_id],
            )

            updated = result.fetchone() is not None
            if updated:
                cursor.execute(
                    """
                    UPDATE worker_liveness
                    SET accepting_new_requests = FALSE,
                        updated_at = now()
                    WHERE worker_id = ?
                    """,
                    [worker_id],
                )
                logger.debug(f"Marked worker {worker_id} as inactive")
                return True
            exists = cursor.execute(
                "SELECT 1 FROM workers WHERE worker_id = ?",
                [worker_id],
            ).fetchone()
            return exists is not None

    def get_stale_workers(self, recovery_time: int) -> list[Worker]:
        """Get workers that haven't updated since recovery started."""
        with self._write_lock:
            cursor = self.get_cursor()
            try:
                result = cursor.execute(
                    f"""
                    {_WORKER_SELECT}
                    WHERE EXTRACT(epoch FROM worker_liveness.last_heartbeat) < ?
                      AND workers.inactive_at IS NULL
                    ORDER BY worker_liveness.last_heartbeat DESC
                    """,
                    [recovery_time],
                )
                return [self._row_to_model(row) for row in result.fetchall()]
            finally:
                cursor.close()

    def cleanup_stale_workers(self, recovery_time: int) -> list[tuple[str, str]]:
        """
        Clean up workers that remained stale after recovery period.

        Returns:
            List of (worker_id, node_id) tuples that were cleaned up
        """
        with self.transaction() as cursor:
            rows = cursor.execute(
                """
                WITH stale AS (
                    SELECT workers.worker_id, workers.node_id
                    FROM workers
                    JOIN worker_liveness
                      ON worker_liveness.worker_id = workers.worker_id
                    WHERE EXTRACT(epoch FROM worker_liveness.last_heartbeat) < ?
                      AND workers.inactive_at IS NULL
                )
                UPDATE workers
                SET inactive_at = now()
                WHERE workers.worker_id IN (SELECT worker_id FROM stale)
                RETURNING workers.worker_id, workers.node_id
                """,
                [recovery_time],
            ).fetchall()
            if rows:
                worker_ids = [row[0] for row in rows]
                placeholders = ", ".join(["?"] * len(worker_ids))
                cursor.execute(
                    f"""
                    UPDATE worker_liveness
                    SET accepting_new_requests = FALSE,
                        updated_at = now()
                    WHERE worker_id IN ({placeholders})
                    """,
                    worker_ids,
                )

        if rows:
            logger.info(f"Marked {len(rows)} stale workers as inactive")
        return rows

    def _row_to_model(self, row: tuple) -> Worker:
        """Convert a database row to Worker object."""
        memory_tier_state = None
        if row[15] is not None:
            memory_tier_state = WorkerMemoryTierState(
                stable_total_bytes=row[15],
                stable_used_bytes=row[16],
                preemptible_total_bytes=row[17],
                preemptible_marked_bytes=row[18],
                faults_per_sec=row[19] or 0.0,
                rehydrate_p99_ns=row[20] or 0,
                enable_preemptible=bool(row[21]),
                memory_tier_config_json=row[22] or "{}",
                snapshot_epoch_ns=row[23] or 0,
            )

        return Worker(
            worker_id=row[0],
            daemon_id=row[1] or "",
            node_id=row[2],
            node_address=row[3],
            grpc_port=row[4],
            p2p_port=row[5],
            mem_pool_total_size=row[6],
            mem_pool_available_size=row[7],
            accepting_new_requests=row[8],
            capability_flags=row[9] or 0,
            registered_at=row[10],
            last_heartbeat=row[11],
            inactive_at=row[12],
            state_version=row[13],
            state_checksum=row[14] or "",
            memory_tier_state=memory_tier_state,
        )

    # ---------------------------------------------------------------------
    # Additional helper methods (used by test suite)
    # ---------------------------------------------------------------------

    def find_by_node_id(self, node_id: str) -> Worker | None:
        """Find the most recent worker registered on the given node_id."""
        with self._write_lock:
            cursor = self.get_cursor()
            try:
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
            finally:
                cursor.close()

    def find_by_worker_id(self, worker_id: str) -> Worker | None:
        """Find an active worker by worker_id."""
        with self._write_lock:
            cursor = self.get_cursor()
            try:
                row = cursor.execute(
                    f"""
                    {_WORKER_SELECT}
                    WHERE workers.worker_id = ? AND workers.inactive_at IS NULL
                    LIMIT 1
                    """,
                    [worker_id],
                ).fetchone()
                if row:
                    return self._row_to_model(row)
                return None
            finally:
                cursor.close()

    def list_active(self, accepting_only: bool = False) -> list[Worker]:
        """Return active workers, optionally filtering by accepting status."""
        # Map accepting_only flag to existing include_unavailable parameter
        include_unavailable = not accepting_only
        return self.find_active(include_unavailable=include_unavailable)
