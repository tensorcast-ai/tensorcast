#  Copyright (c) 2025-2026, TensorCast Team.

"""Repository for view metadata anchored to artifacts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from tensorcast.global_store.repositories.base import BaseRepository
from tensorcast.logger import init_logger

logger = init_logger(__name__)


class ViewRepository(BaseRepository):
    """Data access helper for the `views` table."""

    def upsert(
        self,
        *,
        artifact_id: str,
        view_id: str,
        view_spec_json: str,
        view_size: int,
        view_data_hash: Optional[str],
        verified_at: Optional[datetime],
        canonical_size_bytes: Optional[int] = None,
        canonical_bytes_covered: Optional[int] = None,
        cursor=None,
    ) -> None:
        """Insert or update a view row."""
        params = [
            artifact_id,
            view_id,
            view_spec_json,
            view_size,
            view_data_hash,
            verified_at,
            canonical_size_bytes,
            canonical_bytes_covered,
        ]

        try:
            target = cursor if cursor is not None else self.get_cursor()
            target.execute(
                """
                INSERT INTO views (
                    artifact_id,
                    view_id,
                    view_spec_json,
                    view_size,
                    view_data_hash,
                    verified_at,
                    canonical_size_bytes,
                    canonical_bytes_covered
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (artifact_id, view_id) DO UPDATE SET
                    view_spec_json = EXCLUDED.view_spec_json,
                    view_size = EXCLUDED.view_size,
                    view_data_hash = EXCLUDED.view_data_hash,
                    verified_at = EXCLUDED.verified_at,
                    canonical_size_bytes = EXCLUDED.canonical_size_bytes,
                    canonical_bytes_covered = EXCLUDED.canonical_bytes_covered
                """,
                params,
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "View upsert failed for artifact_id=%s view_id=%s", artifact_id, view_id
            )
            raise

    def get(
        self, *, artifact_id: str, view_id: str, cursor=None
    ) -> Optional[dict[str, Any]]:
        """Fetch view metadata by composite primary key."""
        target = cursor if cursor is not None else self.get_cursor()
        row = target.execute(
            """
            SELECT artifact_id,
                   view_id,
                   view_spec_json,
                   view_size,
                   view_data_hash,
                   verified_at,
                   canonical_size_bytes,
                   canonical_bytes_covered,
                   created_at
            FROM views
            WHERE artifact_id = ? AND view_id = ?
            """,
            [artifact_id, view_id],
        ).fetchone()
        if not row:
            return None
        return {
            "artifact_id": row[0],
            "view_id": row[1],
            "view_spec_json": row[2],
            "view_size": row[3],
            "view_data_hash": row[4],
            "verified_at": row[5],
            "canonical_size_bytes": row[6],
            "canonical_bytes_covered": row[7],
            "created_at": row[8],
        }

    def list_by_artifact(
        self,
        *,
        artifact_id: str,
        limit: int | None = None,
        offset: int = 0,
        cursor=None,
    ) -> tuple[list[dict[str, Any]], int]:
        """List views for an artifact with optional pagination."""
        target = cursor if cursor is not None else self.get_cursor()
        total_row = target.execute(
            "SELECT COUNT(*) FROM views WHERE artifact_id = ?",
            [artifact_id],
        ).fetchone()
        total_size = int(total_row[0]) if total_row else 0

        sql = """
            SELECT artifact_id,
                   view_id,
                   view_spec_json,
                   view_size,
                   view_data_hash,
                   verified_at,
                   canonical_size_bytes,
                   canonical_bytes_covered,
                   created_at
            FROM views
            WHERE artifact_id = ?
            ORDER BY created_at ASC
        """
        params: list[object] = [artifact_id]
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        rows = target.execute(sql, params).fetchall()
        items = [
            {
                "artifact_id": row[0],
                "view_id": row[1],
                "view_spec_json": row[2],
                "view_size": row[3],
                "view_data_hash": row[4],
                "verified_at": row[5],
                "canonical_size_bytes": row[6],
                "canonical_bytes_covered": row[7],
                "created_at": row[8],
            }
            for row in rows
        ]
        return items, total_size
