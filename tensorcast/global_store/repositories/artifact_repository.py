#  Copyright (c) 2025-2026, TensorCast Team.

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
        index_multihash: str | None,
        data_multihash: str | None,
        schema_version: str,
        encoding: str,
        hash_params_json: Optional[str] = None,
        id_kind: str = "MI2",
        cursor=None,
    ) -> None:
        """Create or update an artifact descriptor row by primary key `artifact_id`.

        Uses immutable-first semantics: first writer wins and later writes are no-op.
        """
        owns_cursor = cursor is None
        cursor = cursor if cursor is not None else self.get_cursor()
        try:
            cursor.execute(
                """
                INSERT INTO artifacts (
                    artifact_id,
                    index_multihash,
                    data_multihash,
                    schema_version,
                    encoding,
                    hash_params_json,
                    id_kind
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (artifact_id) DO NOTHING
                """,
                [
                    artifact_id,
                    index_multihash,
                    data_multihash,
                    schema_version,
                    encoding,
                    hash_params_json,
                    id_kind,
                ],
            )
        finally:
            if owns_cursor:
                cursor.close()

    def get(self, artifact_id: str, cursor=None) -> Optional[dict[str, Any]]:
        """Fetch a artifact descriptor by id; returns None if not found."""
        owns_cursor = cursor is None
        cursor = cursor if cursor is not None else self.get_cursor()
        try:
            row = cursor.execute(
                """
                SELECT artifact_id, index_multihash, data_multihash, schema_version, encoding, hash_params_json, id_kind, created_at
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
                "created_at": row[7],
                "id_kind": row[6],
            }
        finally:
            if owns_cursor:
                cursor.close()
