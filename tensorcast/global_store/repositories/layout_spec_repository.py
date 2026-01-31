#  Copyright (c) 2025-2026, TensorCast Team.

"""Repository for immutable, content-addressed LayoutSpec records."""

from __future__ import annotations

from typing import Any

from tensorcast.global_store.repositories.base import BaseRepository
from tensorcast.logger import init_logger

logger = init_logger(__name__)


class LayoutSpecRepository(BaseRepository):
    """Data access layer for `layout_specs`."""

    def get(self, *, layout_id: str, cursor=None) -> dict[str, Any] | None:
        target = cursor if cursor is not None else self.get_cursor()
        row = target.execute(
            """
            SELECT layout_id, index_multihash, layout_proto, layout_json, created_at
            FROM layout_specs
            WHERE layout_id = ?
            """,
            [layout_id],
        ).fetchone()
        if not row:
            return None
        return {
            "layout_id": str(row[0]),
            "index_multihash": str(row[1]),
            "layout_proto": bytes(row[2]),
            "layout_json": (str(row[3]) if row[3] is not None else None),
            "created_at": row[4],
        }

    def put(
        self,
        *,
        layout_id: str,
        index_multihash: str,
        layout_proto: bytes,
        layout_json: str | None,
        cursor=None,
    ) -> bool:
        """Idempotent insert; returns True if created."""
        target = cursor if cursor is not None else self.get_cursor()
        existing = self.get(layout_id=layout_id, cursor=target)
        if existing is not None:
            if existing["index_multihash"] != index_multihash:
                raise ValueError("layout_id collision with different index_multihash")
            if existing["layout_proto"] != layout_proto:
                raise ValueError("layout_id collision with different payload")
            return False

        target.execute(
            """
            INSERT INTO layout_specs (layout_id, index_multihash, layout_proto, layout_json)
            VALUES (?, ?, ?, ?)
            """,
            [layout_id, index_multihash, layout_proto, layout_json],
        )
        return True
