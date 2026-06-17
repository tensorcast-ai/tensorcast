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
        result = {
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
        if len(row) >= 13:
            result.update(
                {
                    "target_kind": row[9],
                    "group_version_set_id": row[10],
                    "selection_hash": row[11],
                    "manifest_hash": row[12],
                }
            )
        else:
            result.update(
                {
                    "target_kind": "artifact_selection",
                    "group_version_set_id": None,
                    "selection_hash": None,
                    "manifest_hash": None,
                }
            )
        return result

    def _select_row(self, cursor, key: str) -> dict[str, Any] | None:
        row = cursor.execute(
            """
            SELECT key,
                   artifact_id,
                   replica_uuid,
                   daemon_address,
                   ttl_seconds,
                   generation,
                   kind,
                   created_at,
                   updated_at,
                   target_kind,
                   group_version_set_id,
                   selection_hash,
                   manifest_hash
            FROM key_mappings WHERE key = ?
            """,
            [key],
        ).fetchone()
        if not row:
            return None
        return self._row_to_dict(row)

    def _ensure_artifact_target_history(
        self,
        cursor,
        *,
        namespace: str,
        key: str,
        generation: int,
        artifact_id: str,
        selection_hash: bytes | None = None,
    ) -> None:
        cursor.execute(
            """
            INSERT INTO key_version_targets (
              namespace, key, generation, target_kind, artifact_id, selection_hash
            ) VALUES (?, ?, ?, 'artifact_selection', ?, ?)
            ON CONFLICT (namespace, key, generation) DO NOTHING
            """,
            [namespace, key, int(generation), artifact_id, selection_hash],
        )

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
        selection_hash: bytes | None = None,
    ) -> None:
        """Create a mapping or update non-target metadata.

        Target changes are generation-forming and must go through `swap`.
        """
        with self.transaction() as cursor:
            existing = self._select_row(cursor, key)
            if existing is not None:
                target_kind = str(existing.get("target_kind") or "artifact_selection")
                current_artifact_id = str(existing.get("artifact_id") or "")
                if target_kind != "artifact_selection":
                    raise ValueError(
                        "metadata upsert cannot change a group-version-set key target"
                    )
                if current_artifact_id != artifact_id:
                    raise ValueError(
                        "key target changes must use swap_key_mapping or a target-changing API"
                    )
                cursor.execute(
                    """
                    UPDATE key_mappings
                    SET replica_uuid = ?,
                        daemon_address = ?,
                        ttl_seconds = ?,
                        selection_hash = COALESCE(selection_hash, ?),
                        updated_at = now()
                    WHERE key = ?
                    """,
                    [replica_uuid, daemon_address, ttl_seconds, selection_hash, key],
                )
                self._ensure_artifact_target_history(
                    cursor,
                    namespace="",
                    key=key,
                    generation=int(existing.get("generation", 0) or 0),
                    artifact_id=artifact_id,
                    selection_hash=selection_hash or existing.get("selection_hash"),
                )
                self._refresh_cache_from_cursor(cursor=cursor, key=key)
                return

            cursor.execute(
                """
                INSERT INTO key_mappings (
                  key,
                  artifact_id,
                  replica_uuid,
                  daemon_address,
                  ttl_seconds,
                  generation,
                  kind,
                  target_kind,
                  selection_hash,
                  created_at,
                  updated_at
                ) VALUES (?, ?, ?, ?, ?, 0, 'IMMUTABLE', 'artifact_selection', ?, now(), now())
                """,
                [
                    key,
                    artifact_id,
                    replica_uuid,
                    daemon_address,
                    ttl_seconds,
                    selection_hash,
                ],
            )
            self._ensure_artifact_target_history(
                cursor,
                namespace="",
                key=key,
                generation=0,
                artifact_id=artifact_id,
                selection_hash=selection_hash,
            )
            self._refresh_cache_from_cursor(cursor=cursor, key=key)

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
                SELECT key, artifact_id, generation, kind, target_kind
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
                self._ensure_artifact_target_history(
                    cursor,
                    namespace="",
                    key=key,
                    generation=0,
                    artifact_id=new_artifact_id,
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
            current_target_kind = row[4]
            if current_target_kind != "artifact_selection":
                self._refresh_cache_from_cursor(cursor=cursor, key=key)
                return {
                    "ok": False,
                    "artifact_id": current_artifact_id,
                    "generation": current_generation,
                    "kind": current_kind,
                }
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
                self._ensure_artifact_target_history(
                    cursor,
                    namespace="",
                    key=key,
                    generation=current_generation,
                    artifact_id=current_artifact_id,
                )
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
                SET artifact_id = ?,
                    generation = ?,
                    kind = 'ALIAS',
                    target_kind = 'artifact_selection',
                    group_version_set_id = NULL,
                    selection_hash = NULL,
                    manifest_hash = NULL,
                    updated_at = now()
                WHERE key = ?
                """,
                [new_artifact_id, new_generation, key],
            )
            self._ensure_artifact_target_history(
                cursor,
                namespace="",
                key=key,
                generation=new_generation,
                artifact_id=new_artifact_id,
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

    def backfill_target_history(self) -> int:
        """Backfill `key_version_targets` from existing fast-pointer rows."""
        with self.transaction() as cursor:
            rows = cursor.execute(
                """
                SELECT key, generation, target_kind, artifact_id, group_version_set_id, selection_hash, manifest_hash
                FROM key_mappings
                """
            ).fetchall()
            inserted = 0
            for row in rows:
                key = str(row[0])
                generation = int(row[1] or 0)
                target_kind = str(row[2] or "artifact_selection")
                artifact_id = row[3]
                group_version_set_id = row[4]
                selection_hash = row[5]
                manifest_hash = row[6]
                before = cursor.execute(
                    """
                    SELECT 1 FROM key_version_targets
                    WHERE namespace = '' AND key = ? AND generation = ?
                    """,
                    [key, generation],
                ).fetchone()
                if before is not None:
                    continue
                cursor.execute(
                    """
                    INSERT INTO key_version_targets (
                      namespace,
                      key,
                      generation,
                      target_kind,
                      artifact_id,
                      group_version_set_id,
                      selection_hash,
                      manifest_hash
                    ) VALUES ('', ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        key,
                        generation,
                        target_kind,
                        artifact_id,
                        group_version_set_id,
                        selection_hash,
                        manifest_hash,
                    ],
                )
                inserted += 1
            return inserted

    def get_target_for_generation(
        self,
        *,
        key: str,
        generation: int,
        namespace: str = "",
    ) -> dict[str, Any] | None:
        cursor = self.get_cursor()
        try:
            row = cursor.execute(
                """
                SELECT namespace,
                       key,
                       generation,
                       target_kind,
                       artifact_id,
                       view_id,
                       group_version_set_id,
                       selection_hash,
                       manifest_hash,
                       created_at
                FROM key_version_targets
                WHERE namespace = ? AND key = ? AND generation = ?
                """,
                [namespace, key, int(generation)],
            ).fetchone()
            if row is None:
                return None
            return {
                "namespace": row[0],
                "key": row[1],
                "generation": int(row[2]),
                "target_kind": row[3],
                "artifact_id": row[4],
                "view_id": row[5],
                "group_version_set_id": row[6],
                "selection_hash": row[7],
                "manifest_hash": row[8],
                "created_at": row[9],
            }
        finally:
            cursor.close()

    def get_current_target(
        self,
        *,
        key: str,
        namespace: str = "",
    ) -> dict[str, Any] | None:
        row = self.get(key)
        if row is None:
            return None
        generation = int(row.get("generation", 0) or 0)
        target = self.get_target_for_generation(
            key=key,
            namespace=namespace,
            generation=generation,
        )
        if target is not None:
            return target
        self.backfill_target_history()
        return self.get_target_for_generation(
            key=key,
            namespace=namespace,
            generation=generation,
        )

    def persist_artifact_selection_hash(
        self,
        *,
        key: str,
        generation: int,
        selection_hash: bytes,
        namespace: str = "",
    ) -> None:
        """Fill a migrated artifact-selection target hash after strict resolution."""
        with self.transaction() as cursor:
            cursor.execute(
                """
                UPDATE key_version_targets
                SET selection_hash = ?
                WHERE namespace = ?
                  AND key = ?
                  AND generation = ?
                  AND target_kind = 'artifact_selection'
                  AND selection_hash IS NULL
                """,
                [selection_hash, namespace, key, int(generation)],
            )
            if namespace == "":
                cursor.execute(
                    """
                    UPDATE key_mappings
                    SET selection_hash = COALESCE(selection_hash, ?),
                        updated_at = now()
                    WHERE key = ?
                      AND generation = ?
                      AND target_kind = 'artifact_selection'
                    """,
                    [selection_hash, key, int(generation)],
                )
                self._refresh_cache_from_cursor(cursor=cursor, key=key)

    def set_group_version_set_target(
        self,
        *,
        key: str,
        group_version_set_id: str,
        manifest_hash: bytes,
        expected_generation: int | None = None,
        namespace: str = "",
    ) -> dict[str, Any]:
        """Advance a key to a group-version-set target with a generation bump."""
        with self.transaction() as cursor:
            row = self._select_row(cursor, key)
            if row is None:
                if expected_generation is not None:
                    return {"ok": False, "generation": None}
                generation = 0
                cursor.execute(
                    """
                    INSERT INTO key_mappings (
                      key,
                      artifact_id,
                      generation,
                      kind,
                      target_kind,
                      group_version_set_id,
                      manifest_hash,
                      created_at,
                      updated_at
                    ) VALUES (?, '', 0, 'ALIAS', 'group_version_set', ?, ?, now(), now())
                    """,
                    [key, group_version_set_id, manifest_hash],
                )
            else:
                current_generation = int(row.get("generation", 0) or 0)
                if (
                    expected_generation is not None
                    and expected_generation != current_generation
                ):
                    return {"ok": False, "generation": current_generation}
                same_target = (
                    row.get("target_kind") == "group_version_set"
                    and row.get("group_version_set_id") == group_version_set_id
                )
                generation = (
                    current_generation if same_target else current_generation + 1
                )
                cursor.execute(
                    """
                    UPDATE key_mappings
                    SET artifact_id = '',
                        generation = ?,
                        kind = 'ALIAS',
                        target_kind = 'group_version_set',
                        group_version_set_id = ?,
                        selection_hash = NULL,
                        manifest_hash = ?,
                        updated_at = now()
                    WHERE key = ?
                    """,
                    [generation, group_version_set_id, manifest_hash, key],
                )
            cursor.execute(
                """
                INSERT INTO key_version_targets (
                  namespace,
                  key,
                  generation,
                  target_kind,
                  group_version_set_id,
                  manifest_hash
                ) VALUES (?, ?, ?, 'group_version_set', ?, ?)
                ON CONFLICT (namespace, key, generation) DO NOTHING
                """,
                [
                    namespace,
                    key,
                    int(generation),
                    group_version_set_id,
                    manifest_hash,
                ],
            )
            self._refresh_cache_from_cursor(cursor=cursor, key=key)
            return {
                "ok": True,
                "generation": generation,
                "group_version_set_id": group_version_set_id,
            }

    def audit_target_history_consistency(self) -> list[dict[str, Any]]:
        """Return key rows whose fast pointer does not match history truth."""
        cursor = self.get_cursor()
        try:
            rows = cursor.execute(
                """
                SELECT km.key,
                       km.generation,
                       km.target_kind,
                       km.artifact_id,
                       km.group_version_set_id,
                       km.selection_hash,
                       km.manifest_hash,
                       kvt.target_kind,
                       kvt.artifact_id,
                       kvt.group_version_set_id,
                       kvt.selection_hash,
                       kvt.manifest_hash
                FROM key_mappings km
                LEFT JOIN key_version_targets kvt
                  ON kvt.namespace = ''
                 AND kvt.key = km.key
                 AND kvt.generation = km.generation
                WHERE kvt.key IS NULL
                   OR km.target_kind != kvt.target_kind
                   OR COALESCE(km.artifact_id, '') != COALESCE(kvt.artifact_id, '')
                   OR COALESCE(km.group_version_set_id, '') != COALESCE(kvt.group_version_set_id, '')
                   OR km.selection_hash IS DISTINCT FROM kvt.selection_hash
                   OR km.manifest_hash IS DISTINCT FROM kvt.manifest_hash
                """
            ).fetchall()
            return [
                {
                    "key": row[0],
                    "generation": int(row[1] or 0),
                    "fast_target_kind": row[2],
                    "fast_artifact_id": row[3],
                    "fast_group_version_set_id": row[4],
                    "fast_selection_hash": row[5],
                    "fast_manifest_hash": row[6],
                    "history_target_kind": row[7],
                    "history_artifact_id": row[8],
                    "history_group_version_set_id": row[9],
                    "history_selection_hash": row[10],
                    "history_manifest_hash": row[11],
                }
                for row in rows
            ]
        finally:
            cursor.close()
