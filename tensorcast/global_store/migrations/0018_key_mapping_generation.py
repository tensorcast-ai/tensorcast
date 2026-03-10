#  Copyright (c) 2026, TensorCast Team.

"""Migration 0018: add generation + kind to key_mappings.

This migration aligns the DuckDB schema with design 0063 by
versioning key mappings and marking mutable alias keys.
"""

from __future__ import annotations

from duckdb import DuckDBPyConnection

UP_QUERIES: tuple[str, ...] = (
    "ALTER TABLE key_mappings ADD COLUMN generation BIGINT NOT NULL DEFAULT 0;",
    "ALTER TABLE key_mappings ADD COLUMN kind TEXT NOT NULL DEFAULT 'IMMUTABLE';",
)

DOWN_QUERIES: tuple[str, ...] = (
    "ALTER TABLE key_mappings DROP COLUMN IF EXISTS kind;",
    "ALTER TABLE key_mappings DROP COLUMN IF EXISTS generation;",
)


def upgrade(conn: DuckDBPyConnection) -> None:
    """Apply migration 0018."""
    for query in UP_QUERIES:
        conn.execute(query)


def downgrade(conn: DuckDBPyConnection) -> None:
    """Rollback migration 0018."""
    for query in DOWN_QUERIES:
        conn.execute(query)
