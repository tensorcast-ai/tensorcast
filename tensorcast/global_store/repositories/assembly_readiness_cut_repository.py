#  Copyright (c) 2026, TensorCast Team.

"""Repository for durable readiness-cut rows."""

from __future__ import annotations

from typing import Any

from tensorcast.global_store.repositories.base import BaseRepository


class AssemblyReadinessCutRepository(BaseRepository):
    """Data access layer for `assembly_readiness_cuts`."""

    def get(self, *, attempt_id: str, cursor=None) -> dict[str, Any] | None:
        target = cursor if cursor is not None else self.get_cursor()
        row = target.execute(
            """
            SELECT attempt_id, readiness_cut_proto, created_at, updated_at
            FROM assembly_readiness_cuts
            WHERE attempt_id = ?
            """,
            [attempt_id],
        ).fetchone()
        if not row:
            return None
        return {
            "attempt_id": str(row[0]),
            "readiness_cut_proto": bytes(row[1]),
            "created_at": row[2],
            "updated_at": row[3],
        }

    def upsert(
        self,
        *,
        attempt_id: str,
        readiness_cut_proto: bytes,
        cursor=None,
    ) -> dict[str, Any]:
        target = cursor if cursor is not None else self.get_cursor()
        target.execute(
            """
            INSERT INTO assembly_readiness_cuts (
                attempt_id,
                readiness_cut_proto
            ) VALUES (?, ?)
            ON CONFLICT (attempt_id) DO UPDATE SET
                readiness_cut_proto = excluded.readiness_cut_proto,
                updated_at = now()
            """,
            [attempt_id, readiness_cut_proto],
        )
        row = self.get(attempt_id=attempt_id, cursor=target)
        if row is None:
            raise ValueError("failed to upsert assembly readiness cut")
        return row


__all__ = ["AssemblyReadinessCutRepository"]
