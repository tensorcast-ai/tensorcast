#  Copyright (c) 2025-2026, TensorCast Team.

"""Repository for key -> artifact_id mappings.

Schema: key_mappings(key PRIMARY KEY, artifact_id, replica_uuid, daemon_address,
ttl_seconds, generation, kind, created_at, updated_at)
"""

from __future__ import annotations

import threading
from typing import Any, Optional

from tensorcast.global_store.repositories.base import BaseRepository


class KeyMappingRepository(BaseRepository):
    """Data access for the `key_mappings` table."""

    def __init__(self, connection):
        super().__init__(connection)
        self._cache_lock = threading.RLock()
        # None means a negative cache entry (key not found).
        self._cache: dict[str, dict[str, Any] | None] = {}

    def _cache_get(self, key: str) -> tuple[bool, dict[str, Any] | None]:
        with self._cache_lock:
            if key not in self._cache:
                return False, None
            cached = self._cache[key]
            if cached is None:
                return True, None
            return True, dict(cached)

    def _cache_set(self, key: str, value: dict[str, Any] | None) -> None:
        with self._cache_lock:
            self._cache[key] = None if value is None else dict(value)

    def _cache_delete(self, key: str) -> None:
        with self._cache_lock:
            self._cache.pop(key, None)

    @staticmethod
    def _row_to_dict(row: tuple[Any, ...]) -> dict[str, Any]:
        return {
            "key": row[0],
            "artifact_id": row[1],
            "replica_uuid": row[2],
            "daemon_address": row[3],
            "ttl_seconds": row[4],
            "generation": row[5],
            "kind": row[6],
            "created_at": row[7],
            "updated_at": row[8],
        }

    def _select_row(self, cursor, key: str) -> dict[str, Any] | None:
        row = cursor.execute(
            """
            SELECT key, artifact_id, replica_uuid, daemon_address, ttl_seconds, generation, kind, created_at, updated_at
            FROM key_mappings WHERE key = ?
            """,
            [key],
        ).fetchone()
        if not row:
            return None
        return self._row_to_dict(row)

    def _refresh_cache_from_cursor(
        self,
        *,
        cursor,
        key: str,
    ) -> dict[str, Any] | None:
        row = self._select_row(cursor, key)
        self._cache_set(key, row)
        return row

    def upsert(
        self,
        *,
        key: str,
        artifact_id: str,
        replica_uuid: str | None = None,
        daemon_address: str | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        """Create or update a key mapping.

        Enforces uniqueness of key; last write wins for hints.
        """
        cursor = self.get_cursor()
        try:
            cursor.execute(
                """
                INSERT INTO key_mappings (
                  key, artifact_id, replica_uuid, daemon_address, ttl_seconds, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, now(), now())
                ON CONFLICT (key) DO UPDATE SET
                  artifact_id = EXCLUDED.artifact_id,
                  replica_uuid = EXCLUDED.replica_uuid,
                  daemon_address = EXCLUDED.daemon_address,
                  ttl_seconds = EXCLUDED.ttl_seconds,
                  updated_at = now()
                """,
                [key, artifact_id, replica_uuid, daemon_address, ttl_seconds],
            )
            self._refresh_cache_from_cursor(cursor=cursor, key=key)
        finally:
            cursor.close()

    def get(self, key: str) -> Optional[dict[str, Any]]:
        hit, cached = self._cache_get(key)
        if hit:
            return cached

        cursor = self.get_cursor()
        try:
            row = self._select_row(cursor, key)
            if row is None:
                self._cache_set(key, None)
                return None
            self._cache_set(key, row)
            return row
        finally:
            cursor.close()

    def swap(
        self,
        *,
        key: str,
        new_artifact_id: str,
        expected_artifact_id: str | None = None,
        expected_generation: int | None = None,
    ) -> dict[str, Any]:
        """Swap a key mapping with optional CAS guardrails."""
        with self.transaction() as cursor:
            row = cursor.execute(
                """
                SELECT key, artifact_id, generation, kind
                FROM key_mappings WHERE key = ?
                """,
                [key],
            ).fetchone()
            if not row:
                if expected_artifact_id or expected_generation is not None:
                    self._cache_set(key, None)
                    return {
                        "ok": False,
                        "artifact_id": None,
                        "generation": None,
                        "kind": None,
                    }
                cursor.execute(
                    """
                    INSERT INTO key_mappings (
                      key, artifact_id, kind, created_at, updated_at
                    ) VALUES (?, ?, 'ALIAS', now(), now())
                    """,
                    [key, new_artifact_id],
                )
                self._refresh_cache_from_cursor(cursor=cursor, key=key)
                return {
                    "ok": True,
                    "artifact_id": new_artifact_id,
                    "generation": 0,
                    "kind": "ALIAS",
                }

            current_artifact_id = row[1]
            current_generation = int(row[2])
            current_kind = row[3]
            if expected_artifact_id and expected_artifact_id != current_artifact_id:
                self._refresh_cache_from_cursor(cursor=cursor, key=key)
                return {
                    "ok": False,
                    "artifact_id": current_artifact_id,
                    "generation": current_generation,
                    "kind": current_kind,
                }
            if (
                expected_generation is not None
                and expected_generation != current_generation
            ):
                self._refresh_cache_from_cursor(cursor=cursor, key=key)
                return {
                    "ok": False,
                    "artifact_id": current_artifact_id,
                    "generation": current_generation,
                    "kind": current_kind,
                }
            if current_artifact_id == new_artifact_id:
                if current_kind != "ALIAS":
                    cursor.execute(
                        """
                    UPDATE key_mappings
                    SET kind = 'ALIAS', updated_at = now()
                    WHERE key = ?
                    """,
                        [key],
                    )
                    current_kind = "ALIAS"
                self._refresh_cache_from_cursor(cursor=cursor, key=key)
                return {
                    "ok": True,
                    "artifact_id": current_artifact_id,
                    "generation": current_generation,
                    "kind": current_kind,
                }
            new_generation = current_generation + 1
            cursor.execute(
                """
                UPDATE key_mappings
                SET artifact_id = ?, generation = ?, kind = 'ALIAS', updated_at = now()
                WHERE key = ?
                """,
                [new_artifact_id, new_generation, key],
            )
            self._refresh_cache_from_cursor(cursor=cursor, key=key)
            return {
                "ok": True,
                "artifact_id": new_artifact_id,
                "generation": new_generation,
                "kind": "ALIAS",
            }

    def delete(self, key: str) -> bool:
        # Select first to avoid depending on RETURNING support
        with self.transaction() as cursor:
            exists = (
                cursor.execute(
                    "SELECT 1 FROM key_mappings WHERE key = ?",
                    [key],
                ).fetchone()
                is not None
            )
            if not exists:
                self._cache_set(key, None)
                return False
            cursor.execute("DELETE FROM key_mappings WHERE key = ?", [key])
            self._cache_set(key, None)
            return True
