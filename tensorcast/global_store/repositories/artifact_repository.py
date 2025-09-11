#  Copyright (c) 2025, TensorCast Team.

"""Repository for content-addressed artifacts metadata (RFC-0007).

This repository manages the `artifacts` table which stores the authoritative
artifact identity components for content-addressed `artifact_id` (mi2 scheme):
`index_multihash`, `data_multihash`, `schema_version`, and `encoding`.

We intentionally keep the API minimal: current callers only need an
idempotent UPSERT by `artifact_id` and a simple getter for debugging.
"""

from __future__ import annotations

from typing import Any, Optional

from tensorcast.global_store.repositories.base import BaseRepository


class ArtifactRepository(BaseRepository):
    """Data access layer for the `artifacts` table."""

    def upsert_artifact(
        self,
        *,
        artifact_id: str,
        index_multihash: str,
        data_multihash: str,
        schema_version: str,
        encoding: str,
        hash_params_json: Optional[str] = None,
    ) -> None:
        """Create or update an artifact descriptor row by primary key `artifact_id`.

        Uses ON CONFLICT DO UPDATE for atomic upsert semantics.
        """
        cursor = self.get_cursor()
        cursor.execute(
            """
            INSERT INTO artifacts (
                artifact_id,
                index_multihash,
                data_multihash,
                schema_version,
                encoding,
                hash_params_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (artifact_id) DO UPDATE SET
                index_multihash = EXCLUDED.index_multihash,
                data_multihash = EXCLUDED.data_multihash,
                schema_version = EXCLUDED.schema_version,
                encoding = EXCLUDED.encoding,
                hash_params_json = EXCLUDED.hash_params_json
            """,
            [
                artifact_id,
                index_multihash,
                data_multihash,
                schema_version,
                encoding,
                hash_params_json,
            ],
        )

    def get(self, artifact_id: str) -> Optional[dict[str, Any]]:
        """Fetch a artifact descriptor by id; returns None if not found."""
        cursor = self.get_cursor()
        row = cursor.execute(
            """
            SELECT artifact_id, index_multihash, data_multihash, schema_version, encoding, hash_params_json, created_at
            FROM artifacts WHERE artifact_id = ?
            """,
            [artifact_id],
        ).fetchone()
        if not row:
            return None
        return {
            "artifact_id": row[0],
            "index_multihash": row[1],
            "data_multihash": row[2],
            "schema_version": row[3],
            "encoding": row[4],
            "hash_params_json": row[5],
            "created_at": row[6],
        }
