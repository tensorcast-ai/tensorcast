#  Copyright (c) 2025, TensorCast Team.

"""Repository for variant metadata anchored to canonical artifacts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from tensorcast.global_store.repositories.base import BaseRepository
from tensorcast.logger import init_logger

logger = init_logger(__name__)


class VariantRepository(BaseRepository):
    """Data access helper for the `variants` table."""

    def upsert(
        self,
        *,
        artifact_id: str,
        view_id: str,
        view_spec_json: str,
        view_size: int,
        view_data_hash: Optional[str],
        verified_at: Optional[datetime],
        cursor=None,
    ) -> None:
        """Insert or update a variant row."""
        params = [
            artifact_id,
            view_id,
            view_spec_json,
            view_size,
            view_data_hash,
            verified_at,
        ]

        try:
            target = cursor if cursor is not None else self.get_cursor()
            target.execute(
                """
                INSERT INTO variants (
                    artifact_id,
                    view_id,
                    view_spec_json,
                    view_size,
                    view_data_hash,
                    verified_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (artifact_id, view_id) DO UPDATE SET
                    view_spec_json = EXCLUDED.view_spec_json,
                    view_size = EXCLUDED.view_size,
                    view_data_hash = EXCLUDED.view_data_hash,
                    verified_at = EXCLUDED.verified_at
                """,
                params,
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Variant upsert failed for artifact_id=%s view_id=%s",
                artifact_id,
                view_id,
            )
            raise

    def get(
        self, *, artifact_id: str, view_id: str, cursor=None
    ) -> Optional[dict[str, Any]]:
        """Fetch a variant view by composite primary key."""
        target = cursor if cursor is not None else self.get_cursor()
        row = target.execute(
            """
            SELECT artifact_id,
                   view_id,
                   view_spec_json,
                   view_size,
                   view_data_hash,
                   verified_at,
                   created_at
            FROM variants
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
            "created_at": row[6],
        }
