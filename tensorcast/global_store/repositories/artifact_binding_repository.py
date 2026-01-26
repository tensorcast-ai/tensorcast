#  Copyright (c) 2025-2026, TensorCast Team.

"""Repository for artifact binding metadata (assembly -> sealed)."""

from __future__ import annotations

from typing import Any, Optional

from tensorcast.global_store.repositories.base import BaseRepository


class ArtifactBindingRepository(BaseRepository):
    """Data access layer for the `artifact_bindings` table."""

    def get(self, from_artifact_id: str, cursor=None) -> Optional[dict[str, Any]]:
        target = cursor if cursor is not None else self.get_cursor()
        row = target.execute(
            """
            SELECT from_artifact_id, to_artifact_id, kind, created_at
            FROM artifact_bindings
            WHERE from_artifact_id = ?
            """,
            [from_artifact_id],
        ).fetchone()
        if not row:
            return None
        return {
            "from_artifact_id": row[0],
            "to_artifact_id": row[1],
            "kind": row[2],
            "created_at": row[3],
        }

    def upsert(
        self,
        *,
        from_artifact_id: str,
        to_artifact_id: str,
        kind: str,
        cursor=None,
    ) -> tuple[dict[str, Any], bool]:
        """Create or confirm a binding. Returns (binding_row, created)."""
        target = cursor if cursor is not None else self.get_cursor()

        existing = self.get(from_artifact_id, cursor=target)
        if existing:
            if existing["to_artifact_id"] != to_artifact_id or existing["kind"] != kind:
                raise ValueError("artifact binding conflict")
            return existing, False

        result = target.execute(
            """
            INSERT INTO artifact_bindings (
                from_artifact_id,
                to_artifact_id,
                kind
            ) VALUES (?, ?, ?)
            ON CONFLICT (from_artifact_id) DO NOTHING
            """,
            [from_artifact_id, to_artifact_id, kind],
        )
        created = bool(getattr(result, "rowcount", 0))
        row = self.get(from_artifact_id, cursor=target)
        if row is None:
            raise ValueError("failed to upsert artifact binding")
        return row, created
