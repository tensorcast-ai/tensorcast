#  Copyright (c) 2025, TensorCast Team.

"""Migration 0017: introduce CGID identity kind columns.

This migration aligns the DuckDB schema with design 0017 by
tracking artifact identity kinds and allowing CGID rows to omit
multihash digests while providing replica expiry metadata.
"""

from __future__ import annotations

from duckdb import DuckDBPyConnection

UP_QUERIES: tuple[str, ...] = (
    "ALTER TABLE artifacts ADD COLUMN id_kind TEXT NOT NULL DEFAULT 'MI2';",
    "ALTER TABLE artifacts ALTER COLUMN index_multihash DROP NOT NULL;",
    "ALTER TABLE artifacts ALTER COLUMN data_multihash DROP NOT NULL;",
    "ALTER TABLE artifact_replicas ADD COLUMN expires_at TIMESTAMPTZ NULL;",
)

DOWN_QUERIES: tuple[str, ...] = (
    "ALTER TABLE artifact_replicas DROP COLUMN IF EXISTS expires_at;",
    "ALTER TABLE artifacts DROP COLUMN IF EXISTS id_kind;",
    "ALTER TABLE artifacts ALTER COLUMN index_multihash SET NOT NULL;",
    "ALTER TABLE artifacts ALTER COLUMN data_multihash SET NOT NULL;",
)


def upgrade(conn: DuckDBPyConnection) -> None:
    """Apply migration 0017."""
    for query in UP_QUERIES:
        conn.execute(query)


def downgrade(conn: DuckDBPyConnection) -> None:
    """Rollback migration 0017."""
    for query in DOWN_QUERIES:
        conn.execute(query)
