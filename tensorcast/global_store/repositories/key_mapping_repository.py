#  Copyright (c) 2025, TensorCast Team.

"""Repository for RFC-0014 key → artifact_id mappings.

Schema: key_mappings(key PRIMARY KEY, artifact_id, replica_uuid, daemon_address,
disk_path, ttl_seconds, created_at, updated_at)
"""

from __future__ import annotations

from typing import Any, Optional

from tensorcast.global_store.repositories.base import BaseRepository


class KeyMappingRepository(BaseRepository):
    """Data access for the `key_mappings` table."""

    def upsert(
        self,
        *,
        key: str,
        artifact_id: str,
        replica_uuid: str | None = None,
        daemon_address: str | None = None,
        disk_path: str | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        """Create or update a key mapping.

        Enforces uniqueness of key; last write wins for hints.
        """
        cursor = self.get_cursor()
        cursor.execute("DELETE FROM key_mappings WHERE key = ?", [key])
        cursor.execute(
            """
            INSERT INTO key_mappings (
              key, artifact_id, replica_uuid, daemon_address, disk_path, ttl_seconds, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            [key, artifact_id, replica_uuid, daemon_address, disk_path, ttl_seconds],
        )

    def get(self, key: str) -> Optional[dict[str, Any]]:
        cursor = self.get_cursor()
        row = cursor.execute(
            """
            SELECT key, artifact_id, replica_uuid, daemon_address, disk_path, ttl_seconds, created_at, updated_at
            FROM key_mappings WHERE key = ?
            """,
            [key],
        ).fetchone()
        if not row:
            return None
        return {
            "key": row[0],
            "artifact_id": row[1],
            "replica_uuid": row[2],
            "daemon_address": row[3],
            "disk_path": row[4],
            "ttl_seconds": row[5],
            "created_at": row[6],
            "updated_at": row[7],
        }

    def delete(self, key: str) -> bool:
        cursor = self.get_cursor()
        result = cursor.execute(
            "DELETE FROM key_mappings WHERE key = ? RETURNING key", [key]
        )
        return result.fetchone() is not None
