#  Copyright (c) 2025, TensorCast Team.

"""Repository for artifact replica data access."""

import time
from collections.abc import Sequence
from uuid import UUID

from tensorcast.global_store.exceptions import NotFoundError
from tensorcast.global_store.models import MemoryType, Replica
from tensorcast.global_store.repositories.base import BaseRepository
from tensorcast.logger import init_logger

logger = init_logger(__name__)


def _to_uuid(value: str | UUID) -> UUID:
    """Convert str or UUID to UUID.

    This helper is now branch-free and relies on UUID(str(value)) which
    is safe for both input variants.  See 250720-isinstance-removal.md §11.
    """
    return UUID(str(value))


class ReplicaRepository(BaseRepository):
    """Repository for managing artifact replicas in the database."""

    def has_any_replica(self, artifact_id: str) -> bool:
        """Return True when at least one replica exists for *artifact_id*."""

        cursor = self.get_cursor()
        row = cursor.execute(
            "SELECT 1 FROM artifact_replicas WHERE artifact_id = ? LIMIT 1",
            [artifact_id],
        ).fetchone()
        return row is not None

    def find_by_id(self, replica_id: UUID, artifact_id: str) -> Replica | None:
        """Find a replica by ID and content-addressed artifact_id."""
        cursor = self.get_cursor()
        query = cursor.execute(
            """
            SELECT
                mr.replica_id,
                mr.artifact_id,
                mr.disk_path,
                mr.node_id,
                mr.node_address,
                mr.node_port,
                mr.memory_size,
                mr.memory_type,
                mr.device_id,
                mr.max_concurrency,
                COALESCE(rc.current_requests, 0) AS current_requests,
                mr.is_available,
                mr.remote_memory_keys,
                mr.buffer_sizes,
                mr.verification_json,
                mr.worker_id,
                mr.created_at,
                mr.updated_at
            FROM artifact_replicas mr
            LEFT JOIN replica_counters rc ON rc.replica_id = mr.replica_id
            WHERE mr.replica_id = ? AND mr.artifact_id = ?
            """,
            [str(replica_id), artifact_id],
        )

        row = query.fetchone()
        if row:
            columns = [desc[0] for desc in query.description]
            return self._row_to_model(row, columns)
        return None

    def find_existing(
        self,
        artifact_id: str,
        node_id: str,
        node_address: str,
        node_port: int,
        memory_type: MemoryType,
        device_id: int,
    ) -> Replica | None:
        """Find existing replica with same identifying information."""
        cursor = self.get_cursor()
        query = cursor.execute(
            """
            SELECT
                mr.replica_id,
                mr.artifact_id,
                mr.disk_path,
                mr.node_id,
                mr.node_address,
                mr.node_port,
                mr.memory_size,
                mr.memory_type,
                mr.device_id,
                mr.max_concurrency,
                COALESCE(rc.current_requests, 0) AS current_requests,
                mr.is_available,
                mr.remote_memory_keys,
                mr.buffer_sizes,
                mr.verification_json,
                mr.worker_id,
                mr.created_at,
                mr.updated_at
            FROM artifact_replicas mr
            LEFT JOIN replica_counters rc ON rc.replica_id = mr.replica_id
            WHERE mr.artifact_id = ?
              AND mr.node_id = ?
              AND mr.node_address = ?
              AND mr.node_port = ?
              AND mr.memory_type = ?
              AND mr.device_id = ?
            """,
            [
                artifact_id,
                node_id,
                node_address,
                node_port,
                memory_type.value,
                device_id,
            ],
        )

        row = query.fetchone()
        if row:
            columns = [desc[0] for desc in query.description]
            return self._row_to_model(row, columns)
        return None

    def find_available_for_transport(
        self, artifact_id: str, heartbeat_timeout_seconds: float
    ) -> Replica | None:
        """
        Find best available replica for transport with load balancing.

        Atomically increments current_requests counter.
        """
        cursor = self.get_cursor()
        cutoff = time.time() - heartbeat_timeout_seconds

        # Atomic update on replica_counters to claim a replica
        result = cursor.execute(
            """
            WITH candidate AS (
                SELECT r.replica_id
                FROM artifact_replicas r
                LEFT JOIN replica_counters rc ON rc.replica_id = r.replica_id
                LEFT JOIN workers w ON r.worker_id = w.worker_id
                WHERE r.artifact_id = ?
                  AND COALESCE(rc.current_requests, 0) < r.max_concurrency
                  AND r.is_available = TRUE
                  AND w.accepting_new_requests = TRUE
                  AND EXTRACT(epoch FROM w.last_heartbeat) > ?
                ORDER BY
                    -- 1) Prefer GPU over RAM over DISK (same as before)
                    CASE
                        WHEN r.memory_type = 'GPU' THEN 0
                        WHEN r.memory_type = 'RAM' THEN 1
                        WHEN r.memory_type = 'DISK' THEN 2
                        ELSE 3
                    END,
                    -- 2) Prefer *smaller capacity* replicas first so that
                    --    low-capacity GPUs are filled before larger ones.
                    r.max_concurrency ASC,
                    -- 3) Finally fall back to load-ratio to break ties among
                    --    replicas with identical capacity.
                    (COALESCE(rc.current_requests, 0) * 1.0 / GREATEST(r.max_concurrency, 1)),
                    -- 4) Most recently updated last (older replicas first) to keep
                    --    ordering deterministic when previous keys tie.
                    r.updated_at ASC
                LIMIT 1
            )
            UPDATE replica_counters
            SET current_requests = current_requests + 1,
                last_assigned_at = now()
            WHERE replica_id = (SELECT replica_id FROM candidate)
            RETURNING replica_id
            """,
            [artifact_id, cutoff],
        )

        row = result.fetchone()
        if not row:
            return None

        replica_id = row[0]

        # Fetch full replica data including current_requests
        result = cursor.execute(
            """
            SELECT
                mr.replica_id,
                mr.artifact_id,
                mr.disk_path,
                mr.node_id,
                mr.node_address,
                mr.node_port,
                mr.memory_size,
                mr.memory_type,
                mr.device_id,
                mr.max_concurrency,
                rc.current_requests,
                mr.is_available,
                mr.remote_memory_keys,
                mr.buffer_sizes,
                mr.verification_json,
                mr.worker_id,
                mr.created_at,
                mr.updated_at
            FROM artifact_replicas mr
            JOIN replica_counters rc ON rc.replica_id = mr.replica_id
            WHERE mr.replica_id = ?
            """,
            [str(replica_id)],
        )

        full_row = result.fetchone()
        if full_row:
            columns = [desc[0] for desc in result.description]
            return self._row_to_model(full_row, columns)
        return None

    def create(self, replica: Replica) -> Replica:
        """Create a new replica."""
        cursor = self.get_cursor()

        # Insert into artifact_replicas (without current_requests)
        cursor.execute(
            """
            INSERT INTO artifact_replicas (
                replica_id, artifact_id, node_id, node_address, node_port,
                memory_size, memory_type, device_id, max_concurrency,
                is_available, remote_memory_keys, buffer_sizes, verification_json, worker_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                str(replica.replica_id),
                replica.artifact_id,
                replica.node_id,
                replica.node_address,
                replica.node_port,
                replica.memory_size,
                replica.memory_type.value,
                replica.device_id
                if replica.device_id is not None
                else -1,  # Use -1 for NULL device_id
                replica.max_concurrency,
                replica.is_available,
                list(replica.remote_memory_keys),
                list(replica.buffer_sizes),
                replica.verification_json,
                replica.worker_id,
            ],
        )

        # Ensure corresponding counter record exists (atomic upsert)
        cursor.execute(
            """
            INSERT INTO replica_counters (replica_id, current_requests, last_assigned_at)
            VALUES (?, ?, now())
            ON CONFLICT (replica_id) DO UPDATE SET
                current_requests = EXCLUDED.current_requests,
                last_assigned_at = now()
            """,
            [str(replica.replica_id), replica.current_requests],
        )

        return replica

    def create_or_update(self, replica: Replica) -> Replica:
        """Create a new replica or update existing one."""
        existing = self.find_existing(
            replica.artifact_id,
            replica.node_id,
            replica.node_address,
            replica.node_port,
            replica.memory_type,
            replica.device_id,
        )

        if existing:
            # Update the existing replica with new information
            replica.replica_id = existing.replica_id
            return self.update(replica)
        else:
            return self.create(replica)

    def create_or_update_atomic(self, replica: Replica, cursor) -> Replica:
        """
        Atomically create or update a replica within a transaction.

        This method prevents race conditions by using a transaction-safe
        approach to handle concurrent replica registrations.

        Args:
            replica: Replica to create or update
            cursor: Database cursor within an active transaction

        Returns:
            Created or updated Replica
        """
        # First, try to find existing replica within the transaction
        existing_result = cursor.execute(
            """
            SELECT replica_id FROM artifact_replicas
            WHERE artifact_id = ? AND node_id = ? AND node_address = ?
              AND node_port = ? AND memory_type = ?
              AND COALESCE(device_id, -1) = COALESCE(?, -1)
            """,
            [
                replica.artifact_id,
                replica.node_id,
                replica.node_address,
                replica.node_port,
                replica.memory_type.value,
                replica.device_id,
            ],
        ).fetchone()

        if existing_result:
            # Update existing replica
            existing_replica_id = existing_result[0]
            cursor.execute(
                """
                UPDATE artifact_replicas
                SET
                    updated_at = CURRENT_TIMESTAMP,
                    is_available = ?,
                    max_concurrency = ?,
                    remote_memory_keys = ?,
                    buffer_sizes = ?,
                    verification_json = ?,
                    memory_size = ?,
                    worker_id = ?,
                    expires_at = ?
                WHERE replica_id = ?
                """,
                [
                    replica.is_available,
                    replica.max_concurrency,
                    list(replica.remote_memory_keys),
                    list(replica.buffer_sizes),
                    replica.verification_json,
                    replica.memory_size,
                    replica.worker_id,
                    replica.expires_at,
                    str(existing_replica_id),
                ],
            )
            replica.replica_id = UUID(str(existing_replica_id))
        else:
            # Create new replica
            cursor.execute(
                """
                INSERT INTO artifact_replicas (
                    replica_id, artifact_id, node_id, node_address, node_port,
                    memory_size, memory_type, device_id, max_concurrency,
                    is_available, remote_memory_keys, buffer_sizes, verification_json, worker_id, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    str(replica.replica_id),
                    replica.artifact_id,
                    replica.node_id,
                    replica.node_address,
                    replica.node_port,
                    replica.memory_size,
                    replica.memory_type.value,
                    replica.device_id
                    if replica.device_id is not None
                    else -1,  # Use -1 for NULL device_id
                    replica.max_concurrency,
                    replica.is_available,
                    list(replica.remote_memory_keys),
                    list(replica.buffer_sizes),
                    replica.verification_json,
                    replica.worker_id,
                    replica.expires_at,
                ],
            )

        # Ensure corresponding counter record exists (atomic upsert)
        cursor.execute(
            """
            INSERT INTO replica_counters (replica_id, current_requests, last_assigned_at)
            VALUES (?, ?, now())
            ON CONFLICT (replica_id) DO UPDATE SET
                current_requests = EXCLUDED.current_requests,
                last_assigned_at = now()
            """,
            [str(replica.replica_id), replica.current_requests],
        )

        return replica

    def update(self, replica: Replica) -> Replica:
        """Update an existing replica."""
        cursor = self.get_cursor()

        result = cursor.execute(
            """
            UPDATE artifact_replicas
            SET
                updated_at = CURRENT_TIMESTAMP,
                is_available = ?,
                max_concurrency = ?,
                remote_memory_keys = ?,
                buffer_sizes = ?,
                verification_json = ?,
                memory_size = ?,
                worker_id = ?
            WHERE replica_id = ?
            RETURNING replica_id
            """,
            [
                replica.is_available,
                replica.max_concurrency,
                list(replica.remote_memory_keys),
                list(replica.buffer_sizes),
                replica.verification_json,
                replica.memory_size,
                replica.worker_id,
                str(replica.replica_id),
            ],
        )

        if result.fetchone() is None:
            raise NotFoundError(f"Replica {replica.replica_id} not found")

        return replica

    def update_heartbeat(self, replica_id: UUID, artifact_id: str) -> bool:
        """Update the heartbeat timestamp for a replica."""
        cursor = self.get_cursor()

        result = cursor.execute(
            """
            UPDATE artifact_replicas
            SET updated_at = CURRENT_TIMESTAMP
            WHERE replica_id = ? AND artifact_id = ?
            RETURNING replica_id
            """,
            [str(replica_id), artifact_id],
        )

        return result.fetchone() is not None

    def decrement_requests(self, replica_id: UUID) -> tuple[int, int]:
        """
        Decrement the current_requests counter.

        Returns:
            Tuple of (current_requests, max_concurrency)
        """
        cursor = self.get_cursor()

        # Update counter table
        cnt_row = cursor.execute(
            """
            UPDATE replica_counters
            SET current_requests = GREATEST(0, current_requests - 1)
            WHERE replica_id = ?
            RETURNING current_requests
            """,
            [str(replica_id)],
        ).fetchone()

        if not cnt_row:
            return 0, 0

        current_requests = int(cnt_row[0])

        # Fetch max_concurrency from artifact_replicas
        max_row = cursor.execute(
            """
            SELECT max_concurrency FROM artifact_replicas WHERE replica_id = ?
            """,
            [str(replica_id)],
        ).fetchone()

        max_conc = int(max_row[0]) if max_row else 0

        return current_requests, max_conc

    def delete(self, replica_id: UUID, artifact_id: str | None = None) -> bool:
        """Delete a replica."""
        cursor = self.get_cursor()

        # First delete from replica_counters (due to foreign key constraint)
        cursor.execute(
            """
            DELETE FROM replica_counters
            WHERE replica_id = ?
            """,
            [str(replica_id)],
        )

        # Then delete from artifact_replicas
        if artifact_id:
            result = cursor.execute(
                """
                DELETE FROM artifact_replicas
                WHERE replica_id = ? AND artifact_id = ?
                RETURNING replica_id
                """,
                [str(replica_id), artifact_id],
            )
        else:
            result = cursor.execute(
                """
                DELETE FROM artifact_replicas
                WHERE replica_id = ?
                RETURNING replica_id
                """,
                [str(replica_id)],
            )

        return result.fetchone() is not None

    def find_by_artifact(self, artifact_id: str) -> list[Replica]:
        """Convenience helper to list replicas for a given content-addressed artifact_id."""
        return self.find_by_filters(artifact_id=artifact_id)

    def find_by_filters(
        self,
        artifact_id: str | None = None,
        node_id: str | None = None,
        node_address: str | None = None,
        node_port: int | None = None,
        memory_type: MemoryType | None = None,
        device_id: int | None = None,
    ) -> list[Replica]:
        """Find replicas matching the given filters."""
        cursor = self.get_cursor()

        # Build dynamic query joining with replica_counters
        query = (
            "SELECT "
            "mr.replica_id, mr.artifact_id, mr.disk_path, mr.node_id, mr.node_address, mr.node_port, "
            "mr.memory_size, mr.memory_type, mr.device_id, mr.max_concurrency, "
            "COALESCE(rc.current_requests, 0) AS current_requests, "
            "mr.is_available, mr.remote_memory_keys, mr.buffer_sizes, mr.verification_json, mr.worker_id, "
            "mr.created_at, mr.updated_at "
            "FROM artifact_replicas mr "
            "LEFT JOIN replica_counters rc ON rc.replica_id = mr.replica_id "
            "WHERE 1=1"
        )
        params: list = []

        if artifact_id is not None:
            query += " AND mr.artifact_id = ?"
            params.append(artifact_id)

        if node_id is not None:
            query += " AND mr.node_id = ?"
            params.append(node_id)

        if node_address is not None:
            query += " AND mr.node_address = ?"
            params.append(node_address)

        if node_port is not None:
            query += " AND mr.node_port = ?"
            params.append(node_port)

        if memory_type is not None:
            query += " AND mr.memory_type = ?"
            params.append(memory_type.value)

        if device_id is not None:
            query += " AND mr.device_id = ?"
            params.append(device_id)

        result = cursor.execute(query, params)
        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()
        replicas = [self._row_to_model(row, columns) for row in rows]
        return replicas

    def mark_unavailable_by_worker(self, worker_id: str) -> int:
        """Mark all replicas for a worker as unavailable."""
        cursor = self.get_cursor()

        # First count how many replicas will be affected
        count_result = cursor.execute(
            """
            SELECT COUNT(*) FROM artifact_replicas
            WHERE worker_id = ? AND is_available = TRUE
            """,
            [worker_id],
        ).fetchone()

        affected_count = count_result[0] if count_result else 0

        # Update the replicas
        cursor.execute(
            """
            UPDATE artifact_replicas
            SET is_available = FALSE
            WHERE worker_id = ?
            """,
            [worker_id],
        )

        return affected_count

    # ========== High Availability Methods ==========

    def list_all_replicas(self) -> list[Replica]:
        """List all replicas in the database."""
        cursor = self.get_cursor()

        result = cursor.execute(
            """
            SELECT
                mr.replica_id,
                mr.artifact_id,
                mr.disk_path,
                mr.node_id,
                mr.node_address,
                mr.node_port,
                mr.memory_size,
                mr.memory_type,
                mr.device_id,
                mr.max_concurrency,
                COALESCE(rc.current_requests, 0) AS current_requests,
                mr.is_available,
                mr.remote_memory_keys,
                mr.buffer_sizes,
                mr.verification_json,
                mr.worker_id,
                mr.created_at,
                mr.updated_at
            FROM artifact_replicas mr
            LEFT JOIN replica_counters rc ON rc.replica_id = mr.replica_id
            ORDER BY mr.created_at DESC
            """
        )

        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()
        return [self._row_to_model(row, columns) for row in rows]

    def mark_as_stale(self, replica_id: UUID) -> bool:
        """Mark a replica as stale (for recovery purposes)."""
        cursor = self.get_cursor()

        result = cursor.execute(
            """
            UPDATE artifact_replicas
            SET updated_at = TIMESTAMP '1970-01-01 00:00:00',
                is_available = FALSE
            WHERE replica_id = ?
            RETURNING replica_id
            """,
            [str(replica_id)],
        )

        updated = result.fetchone() is not None
        if updated:
            logger.debug(f"Marked replica {replica_id} as stale")
        return updated

    def mark_unavailable(self, replica_id: UUID) -> bool:
        """Mark a replica as unavailable."""
        cursor = self.get_cursor()

        result = cursor.execute(
            """
            UPDATE artifact_replicas
            SET is_available = FALSE,
                updated_at = CURRENT_TIMESTAMP
            WHERE replica_id = ?
            RETURNING replica_id
            """,
            [str(replica_id)],
        )

        updated = result.fetchone() is not None
        if updated:
            logger.debug(f"Marked replica {replica_id} as unavailable")
        return updated

    def find_orphaned_replicas(self) -> list[Replica]:
        """Find replicas that reference non-existent workers."""
        cursor = self.get_cursor()

        result = cursor.execute(
            """
            SELECT
                mr.replica_id,
                mr.artifact_id,
                mr.disk_path,
                mr.node_id,
                mr.node_address,
                mr.node_port,
                mr.memory_size,
                mr.memory_type,
                mr.device_id,
                mr.max_concurrency,
                COALESCE(rc.current_requests, 0) AS current_requests,
                mr.is_available,
                mr.remote_memory_keys,
                mr.buffer_sizes,
                mr.worker_id,
                mr.created_at,
                mr.updated_at
            FROM artifact_replicas mr
            LEFT JOIN replica_counters rc ON rc.replica_id = mr.replica_id
            LEFT JOIN workers w ON mr.worker_id = w.worker_id
            WHERE mr.worker_id IS NOT NULL AND w.worker_id IS NULL
            """
        )

        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()
        orphaned = [self._row_to_model(row, columns) for row in rows]
        if orphaned:
            logger.warning(f"Found {len(orphaned)} orphaned replicas")
        return orphaned

    def get_replicas_by_worker(self, worker_id: str) -> list[Replica]:
        """Get all replicas belonging to a specific worker."""
        cursor = self.get_cursor()

        result = cursor.execute(
            """
            SELECT
                mr.replica_id,
                mr.artifact_id,
                mr.disk_path,
                mr.node_id,
                mr.node_address,
                mr.node_port,
                mr.memory_size,
                mr.memory_type,
                mr.device_id,
                mr.max_concurrency,
                COALESCE(rc.current_requests, 0) AS current_requests,
                mr.is_available,
                mr.remote_memory_keys,
                mr.buffer_sizes,
                mr.worker_id,
                mr.created_at,
                mr.updated_at
            FROM artifact_replicas mr
            LEFT JOIN replica_counters rc ON rc.replica_id = mr.replica_id
            WHERE mr.worker_id = ?
            ORDER BY mr.created_at DESC
            """,
            [worker_id],
        )

        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()
        return [self._row_to_model(row, columns) for row in rows]

    def update_worker_id(self, replica_id: UUID, new_worker_id: str) -> bool:
        """Update the worker_id for a replica."""
        cursor = self.get_cursor()

        result = cursor.execute(
            """
            UPDATE artifact_replicas
            SET worker_id = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE replica_id = ?
            RETURNING replica_id
            """,
            [new_worker_id, str(replica_id)],
        )

        updated = result.fetchone() is not None
        if updated:
            logger.debug(f"Updated replica {replica_id} worker_id to {new_worker_id}")
        return updated

    def get_stale_replicas(self, recovery_time: int) -> list[Replica]:
        """Get replicas that haven't been updated since recovery started."""
        cursor = self.get_cursor()

        result = cursor.execute(
            """
            SELECT
                mr.replica_id,
                mr.artifact_id,
                mr.disk_path,
                mr.node_id,
                mr.node_address,
                mr.node_port,
                mr.memory_size,
                mr.memory_type,
                mr.device_id,
                mr.max_concurrency,
                COALESCE(rc.current_requests, 0) AS current_requests,
                mr.is_available,
                mr.remote_memory_keys,
                mr.buffer_sizes,
                mr.worker_id,
                mr.created_at,
                mr.updated_at
            FROM artifact_replicas mr
            LEFT JOIN replica_counters rc ON rc.replica_id = mr.replica_id
            WHERE EXTRACT(epoch FROM mr.updated_at) < ?
            ORDER BY mr.updated_at DESC
            """,
            [recovery_time],
        )

        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()
        return [self._row_to_model(row, columns) for row in rows]

    def cleanup_stale_replicas(self, recovery_time: int) -> int:
        """
        Clean up replicas that remained stale after recovery period.

        Returns:
            Number of replicas cleaned up
        """
        cursor = self.get_cursor()

        # Mark stale replicas as unavailable instead of deleting
        result = cursor.execute(
            """
            UPDATE artifact_replicas
            SET is_available = FALSE
            WHERE EXTRACT(epoch FROM updated_at) < ?
            RETURNING replica_id
            """,
            [recovery_time],
        )

        cleaned_up_count = len(result.fetchall())
        if cleaned_up_count > 0:
            logger.info(f"Marked {cleaned_up_count} stale replicas as unavailable")
        return cleaned_up_count

    def _row_to_model(self, row: tuple, columns: Sequence[str]) -> Replica:
        """Convert a database row to Replica object using column metadata."""

        if not columns:
            raise ValueError("Replica row conversion requires column metadata.")

        column_index = {column: idx for idx, column in enumerate(columns)}

        def get(column: str, *, default=None):
            idx = column_index.get(column)
            if idx is None:
                return default
            value = row[idx]
            if value is None and default is not None:
                return default
            return value

        def require(column: str):
            value = get(column)
            if value is None:
                raise ValueError(f"Replica row missing required column: {column}")
            return value

        replica_id = _to_uuid(require("replica_id"))
        artifact_id = require("artifact_id")
        disk_path = get("disk_path")
        node_id = require("node_id")
        node_address = require("node_address")
        node_port = int(require("node_port"))
        memory_size = int(require("memory_size"))
        memory_type = MemoryType(require("memory_type"))
        device_id_value = get("device_id")
        device_id = int(device_id_value) if device_id_value is not None else None
        max_concurrency = int(require("max_concurrency"))
        current_requests_value = get("current_requests", default=0)
        current_requests = int(current_requests_value or 0)
        is_available = bool(require("is_available"))

        remote_memory_keys_value = get("remote_memory_keys", default=())
        remote_memory_keys = (
            list(remote_memory_keys_value) if remote_memory_keys_value else []
        )

        buffer_sizes_value = get("buffer_sizes", default=())
        buffer_sizes = list(buffer_sizes_value) if buffer_sizes_value else []

        verification_json = get("verification_json")
        worker_id = get("worker_id")
        created_at = get("created_at")
        updated_at = get("updated_at")
        expires_at = get("expires_at")

        return Replica(
            replica_id=replica_id,
            artifact_id=artifact_id,
            disk_path=disk_path,
            node_id=node_id,
            node_address=node_address,
            node_port=node_port,
            memory_size=memory_size,
            memory_type=memory_type,
            device_id=device_id,
            max_concurrency=max_concurrency,
            current_requests=current_requests,
            is_available=is_available,
            remote_memory_keys=remote_memory_keys,
            buffer_sizes=buffer_sizes,
            worker_id=worker_id,
            verification_json=verification_json,
            created_at=created_at,
            updated_at=updated_at,
            expires_at=expires_at,
        )

    def find_by_disk_path(self, disk_path: str) -> list[Replica]:
        """Find all replicas registered under a given disk path (legacy flow)."""
        cursor = self.get_cursor()
        result = cursor.execute(
            """
            SELECT
                mr.replica_id,
                mr.artifact_id,
                mr.disk_path,
                mr.node_id,
                mr.node_address,
                mr.node_port,
                mr.memory_size,
                mr.memory_type,
                mr.device_id,
                mr.max_concurrency,
                COALESCE(rc.current_requests, 0) AS current_requests,
                mr.is_available,
                mr.remote_memory_keys,
                mr.buffer_sizes,
                mr.worker_id,
                mr.created_at,
                mr.updated_at
            FROM artifact_replicas mr
            LEFT JOIN replica_counters rc ON rc.replica_id = mr.replica_id
            WHERE mr.disk_path = ?
            """,
            [disk_path],
        )
        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()
        return [self._row_to_model(row, columns) for row in rows]
