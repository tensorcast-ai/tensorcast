#  Copyright (c) 2025-2026, TensorCast Team.

"""Repository for transport data access."""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from tensorcast.global_store.models import Transport
from tensorcast.global_store.repositories.base import BaseRepository
from tensorcast.logger import init_logger

logger = init_logger(__name__)


class TransportRepository(BaseRepository):
    """Repository for managing transports in the database."""

    def find_by_id(self, transport_id: UUID) -> Optional[Transport]:
        """Find a transport by ID."""
        cursor = self.get_cursor()
        try:
            result = cursor.execute(
                "SELECT * FROM artifact_transports WHERE transport_id = ?",
                [str(transport_id)],
            ).fetchone()

            if result:
                return self._row_to_model(result)
            return None
        finally:
            cursor.close()

    def create(self, transport: Transport) -> Transport:
        """Create a new transport record."""
        cursor = self.get_cursor()
        try:
            cursor.execute(
                """
                INSERT INTO artifact_transports (
                    transport_id, replica_id, artifact_id, disk_path,
                    source_node_id, source_address, source_port,
                    status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    str(transport.transport_id),
                    str(transport.replica_id),
                    transport.artifact_id,
                    transport.disk_path,
                    transport.source_node_id,
                    transport.source_address,
                    transport.source_port,
                    "in_progress",
                ],
            )

            return transport
        finally:
            cursor.close()

    def create_with_cursor(self, transport: Transport, cursor) -> Transport:
        """Create a new transport record using an existing cursor."""
        cursor.execute(
            """
            INSERT INTO artifact_transports (
                transport_id, replica_id, artifact_id, disk_path,
                source_node_id, source_address, source_port,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                str(transport.transport_id),
                str(transport.replica_id),
                transport.artifact_id,
                transport.disk_path,
                transport.source_node_id,
                transport.source_address,
                transport.source_port,
                "in_progress",
            ],
        )

        return transport

    def delete(self, transport_id: UUID) -> bool:
        """Delete a transport record."""
        cursor = self.get_cursor()
        try:
            result = cursor.execute(
                """
                DELETE FROM artifact_transports
                WHERE transport_id = ?
                RETURNING transport_id
                """,
                [str(transport_id)],
            )

            return result.fetchone() is not None
        finally:
            cursor.close()

    def update_status(self, transport_id: UUID, status: str, completed_at=None) -> bool:
        """Update transport status and optionally set completed_at."""
        cursor = self.get_cursor()
        try:
            if completed_at:
                result = cursor.execute(
                    """
                    UPDATE artifact_transports
                    SET status = ?, completed_at = ?
                    WHERE transport_id = ?
                    RETURNING transport_id
                    """,
                    [status, completed_at, str(transport_id)],
                )
            else:
                result = cursor.execute(
                    """
                    UPDATE artifact_transports
                    SET status = ?
                    WHERE transport_id = ?
                    RETURNING transport_id
                    """,
                    [status, str(transport_id)],
                )

            return result.fetchone() is not None
        finally:
            cursor.close()

    def complete_if_in_progress(
        self, transport_id: UUID, completed_at: datetime
    ) -> bool:
        """Atomically mark transport completed only when status is in_progress."""
        cursor = self.get_cursor()
        try:
            result = cursor.execute(
                """
                UPDATE artifact_transports
                SET status = 'completed', completed_at = ?
                WHERE transport_id = ?
                  AND status = 'in_progress'
                RETURNING transport_id
                """,
                [completed_at, str(transport_id)],
            )
            return result.fetchone() is not None
        finally:
            cursor.close()

    def list_with_filters(
        self, status: Optional[str] = None, limit: int = 50, offset: int = 0
    ) -> list[Transport]:
        """List transports with optional filters and pagination."""
        cursor = self.get_cursor()
        try:
            query = "SELECT * FROM artifact_transports"
            params = []

            if status:
                query += " WHERE status = ?"
                params.append(status)

            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([str(limit), str(offset)])

            rows = cursor.execute(query, params).fetchall()
            return [self._row_to_model(row) for row in rows]
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

    def count_with_filters(self, status: Optional[str] = None) -> int:
        """Count transports with optional filters."""
        cursor = self.get_cursor()
        try:
            query = "SELECT COUNT(*) FROM artifact_transports"
            params = []

            if status:
                query += " WHERE status = ?"
                params.append(status)

            result = cursor.execute(query, params).fetchone()
            return result[0] if result else 0
        finally:
            cursor.close()

    def _row_to_model(self, row: tuple[Any, ...]) -> Transport:
        """Convert a database row returned by DuckDB into a ``Transport`` object.

        The database can return UUIDs either as ``uuid.UUID`` instances (when the
        DuckDB ``uuid`` extension is enabled) or as plain strings.  Instead of
        branching on the runtime type with ``isinstance`` we normalise the value
        by casting it to ``str`` first and then constructing a ``UUID``.
        This keeps the code branch-free and shifts type checking to static
        analysis tools rather than to runtime conditionals.
        """

        # Always cast to ``str`` first – this works for both ``uuid.UUID`` and
        # already-stringified UUIDs and avoids the need for ``isinstance`` checks.
        transport_id = UUID(str(row[0]))
        replica_id = UUID(str(row[1]))

        return Transport(
            transport_id=transport_id,
            replica_id=replica_id,
            artifact_id=row[2],
            disk_path=row[3],
            source_node_id=row[4],
            source_address=row[5],
            source_port=row[6],
            created_at=row[7],
            completed_at=row[8] if len(row) > 8 else None,
            status=row[9] if len(row) > 9 else "in_progress",
        )
