#  Copyright (c) 2026, TensorCast Team.

"""Repository for managed disk locations."""

from __future__ import annotations

from typing import Any

from tensorcast.global_store.repositories.base import BaseRepository


class ArtifactDiskLocationRepository(BaseRepository):
    """Data access for the `artifact_disk_locations` table."""

    def upsert(
        self,
        *,
        artifact_id: str,
        cluster_id: str,
        relative_path: str,
        kind: str = "MANAGED",
        is_deleted: bool = False,
    ) -> None:
        cursor = self.get_cursor()
        cursor.execute(
            """
            INSERT INTO artifact_disk_locations AS t (
              artifact_id, cluster_id, relative_path, kind, is_deleted, deleted_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, CASE WHEN ? THEN now() ELSE NULL END, now(), now())
            ON CONFLICT (artifact_id, cluster_id, relative_path) DO UPDATE SET
              kind = EXCLUDED.kind,
              is_deleted = (COALESCE(t.is_deleted, FALSE) OR EXCLUDED.is_deleted),
              deleted_at = CASE
                WHEN COALESCE(t.is_deleted, FALSE) THEN t.deleted_at
                WHEN EXCLUDED.is_deleted THEN now()
                ELSE NULL
              END,
              updated_at = now()
            """,
            [
                artifact_id,
                cluster_id,
                relative_path,
                kind,
                bool(is_deleted),
                bool(is_deleted),
            ],
        )

    def list_by_artifact(
        self, artifact_id: str, *, include_deleted: bool = False
    ) -> list[dict[str, Any]]:
        cursor = self.get_cursor()
        rows = cursor.execute(
            """
            SELECT
              artifact_id, cluster_id, relative_path, kind,
              created_at, updated_at,
              COALESCE(is_deleted, FALSE) AS is_deleted,
              deleted_at
            FROM artifact_disk_locations
            WHERE artifact_id = ?
              AND (? OR COALESCE(is_deleted, FALSE) = FALSE)
            ORDER BY relative_path ASC
            """,
            [artifact_id, bool(include_deleted)],
        ).fetchall()
        return [
            {
                "artifact_id": row[0],
                "cluster_id": row[1],
                "relative_path": row[2],
                "kind": row[3],
                "created_at": row[4],
                "updated_at": row[5],
                "is_deleted": bool(row[6]),
                "deleted_at": row[7],
            }
            for row in rows
        ]
