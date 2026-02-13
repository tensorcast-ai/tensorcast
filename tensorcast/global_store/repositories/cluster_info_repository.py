#  Copyright (c) 2026, TensorCast Team.

"""Repository for cluster metadata (singleton)."""

from __future__ import annotations

import secrets
from datetime import datetime, timezone

from tensorcast.global_store.repositories.base import BaseRepository


class ClusterInfoRepository(BaseRepository):
    """Data access for the `cluster_info` singleton row."""

    @staticmethod
    def _mint_cluster_id(now: datetime) -> str:
        ts = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{ts}_{secrets.token_hex(4)}"

    def get_cluster_id(self) -> str | None:
        cursor = self.get_cursor()
        try:
            row = cursor.execute(
                "SELECT cluster_id FROM cluster_info WHERE singleton_id = 1"
            ).fetchone()
            if not row:
                return None
            return row[0]
        finally:
            cursor.close()

    def get_or_create_cluster_id(self) -> str:
        now = datetime.now(timezone.utc)
        with self.transaction() as cursor:
            row = cursor.execute(
                "SELECT cluster_id FROM cluster_info WHERE singleton_id = 1"
            ).fetchone()
            if row:
                return row[0]
            cluster_id = self._mint_cluster_id(now)
            cursor.execute(
                """
                INSERT INTO cluster_info (singleton_id, cluster_id, created_at)
                VALUES (1, ?, now())
                """,
                [cluster_id],
            )
            return cluster_id
