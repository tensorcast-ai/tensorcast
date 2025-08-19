#  Copyright (c) 2025, StepCast Team.

"""Repository for deduplicated tensor indices (model_indices).

Stores canonical tensor index bytes keyed by SHA-256 hex digest.
"""

from __future__ import annotations

import hashlib
from typing import Optional

from scstore.global_store.repositories.base import BaseRepository


class ModelIndexRepository(BaseRepository):
    """Data access for `model_indices` table."""

    def upsert_index(
        self,
        *,
        index_data: bytes,
        encoding: str,
        schema_version: str,
    ) -> str:
        """Insert or replace canonical index by its SHA-256 key.

        Returns the computed index_key (hex).
        """
        index_key = hashlib.sha256(index_data).hexdigest()
        cursor = self.get_cursor()
        cursor.execute("DELETE FROM model_indices WHERE index_key = ?", [index_key])
        cursor.execute(
            """
            INSERT INTO model_indices (
                index_key,
                schema_version,
                encoding,
                size_bytes,
                index_data
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [index_key, schema_version, encoding, len(index_data), index_data],
        )
        return index_key

    def get(self, index_key: str) -> Optional[bytes]:
        """Fetch canonical index bytes by key; returns None if not found."""
        cursor = self.get_cursor()
        row = cursor.execute(
            "SELECT index_data FROM model_indices WHERE index_key = ?", [index_key]
        ).fetchone()
        return row[0] if row else None
