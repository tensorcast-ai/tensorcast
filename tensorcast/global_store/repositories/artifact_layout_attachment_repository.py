#  Copyright (c) 2025-2026, TensorCast Team.

"""Repository for immutable (mi2_id, layout_id) attachments."""

from __future__ import annotations

from tensorcast.global_store.repositories.base import BaseRepository


class ArtifactLayoutAttachmentRepository(BaseRepository):
    """Data access layer for `artifact_layout_attachments`."""

    def attach(self, *, mi2_id: str, layout_id: str, cursor=None) -> None:
        target = cursor if cursor is not None else self.get_cursor()
        target.execute(
            """
            INSERT INTO artifact_layout_attachments (mi2_id, layout_id)
            VALUES (?, ?)
            ON CONFLICT (mi2_id, layout_id) DO NOTHING
            """,
            [mi2_id, layout_id],
        )

    def list_by_artifact(self, *, mi2_id: str, cursor=None) -> list[str]:
        target = cursor if cursor is not None else self.get_cursor()
        rows = target.execute(
            """
            SELECT layout_id
            FROM artifact_layout_attachments
            WHERE mi2_id = ?
            ORDER BY created_at ASC
            """,
            [mi2_id],
        ).fetchall()
        return [str(row[0]) for row in rows]
