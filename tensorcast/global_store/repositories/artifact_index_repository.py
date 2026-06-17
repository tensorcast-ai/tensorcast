#  Copyright (c) 2025-2026, TensorCast Team.

"""Repository for deduplicated tensor indices (artifact_indices).

Stores canonical tensor index bytes keyed by SHA-256 hex digest.
"""

from __future__ import annotations

import hashlib
from typing import Optional

from tensorcast.global_store.repositories.base import BaseRepository


class ArtifactIndexRepository(BaseRepository):
    """Data access for `artifact_indices` table holding canonical tensor indices."""

    def upsert_index(
        self,
        *,
        index_data: bytes,
        encoding: str,
        schema_version: str,
        cursor=None,
    ) -> str:
        """Insert or replace canonical index by its SHA-256 key.

        Returns the computed index_key (hex).
        """
        index_key = hashlib.sha256(index_data).hexdigest()
        owns_cursor = cursor is None
        cursor = cursor if cursor is not None else self.get_cursor()
        try:
            cursor.execute(
                """
                INSERT INTO artifact_indices (
                    index_key,
                    schema_version,
                    encoding,
                    size_bytes,
                    index_data
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (index_key) DO NOTHING
                """,
                [index_key, schema_version, encoding, len(index_data), index_data],
            )
        finally:
            if owns_cursor:
                cursor.close()
        return index_key

    def get(self, index_key: str, cursor=None) -> Optional[bytes]:
        """Fetch canonical index bytes by key; returns None if not found."""
        owns_cursor = cursor is None
        cursor = cursor if cursor is not None else self.get_cursor()
        try:
            row = cursor.execute(
                "SELECT index_data FROM artifact_indices WHERE index_key = ?",
                [index_key],
            ).fetchone()
            return row[0] if row else None
        finally:
            if owns_cursor:
                cursor.close()
