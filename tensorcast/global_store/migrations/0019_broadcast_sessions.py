#  Copyright (c) 2026, TensorCast Team.

"""Migration 0019: add broadcast session state tables."""

from __future__ import annotations

from duckdb import DuckDBPyConnection

UP_QUERIES: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS broadcast_sessions (
        session_id TEXT PRIMARY KEY,
        artifact_id TEXT NOT NULL,
        requested_view_id TEXT NULL,
        epoch BIGINT NOT NULL,
        fanout INTEGER NOT NULL,
        max_attempts INTEGER NOT NULL DEFAULT 3,
        strict_parent BOOLEAN NOT NULL DEFAULT TRUE,
        state TEXT CHECK (state IN ('planning','active','completed','failed','cancelled')) NOT NULL,
        root_replica_id UUID NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMP WITH TIME ZONE DEFAULT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS broadcast_targets (
        session_id TEXT NOT NULL,
        target_worker_id TEXT NOT NULL,
        target_daemon_id TEXT NULL,
        state TEXT CHECK (state IN ('pending','assigned','materializing','completed','failed','cancelled')) NOT NULL,
        level INTEGER NULL,
        attempt INTEGER NOT NULL DEFAULT 0,
        assigned_edge_id TEXT NULL,
        completed_replica_id UUID NULL,
        failure_reason TEXT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMP WITH TIME ZONE DEFAULT NULL,
        PRIMARY KEY (session_id, target_worker_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS broadcast_edges (
        edge_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        parent_worker_id TEXT NOT NULL,
        parent_replica_id UUID NOT NULL,
        child_worker_id TEXT NOT NULL,
        level INTEGER NOT NULL,
        attempt INTEGER NOT NULL DEFAULT 1,
        state TEXT CHECK (state IN ('planned','assigned','materializing','completed','failed','cancelled')) NOT NULL,
        transport_request_id TEXT NULL,
        failure_reason TEXT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMP WITH TIME ZONE DEFAULT NULL
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_broadcast_targets_session_state ON broadcast_targets(session_id, state, updated_at);",
    "CREATE INDEX IF NOT EXISTS idx_broadcast_edges_session_child_state ON broadcast_edges(session_id, child_worker_id, state);",
    "ALTER TABLE artifact_transports ADD COLUMN IF NOT EXISTS broadcast_session_id TEXT NULL;",
    "ALTER TABLE artifact_transports ADD COLUMN IF NOT EXISTS broadcast_edge_id TEXT NULL;",
    "CREATE INDEX IF NOT EXISTS idx_artifact_transports_broadcast ON artifact_transports(broadcast_session_id, broadcast_edge_id, status);",
)

DOWN_QUERIES: tuple[str, ...] = (
    "DROP INDEX IF EXISTS idx_artifact_transports_broadcast;",
    "ALTER TABLE artifact_transports DROP COLUMN IF EXISTS broadcast_edge_id;",
    "ALTER TABLE artifact_transports DROP COLUMN IF EXISTS broadcast_session_id;",
    "DROP INDEX IF EXISTS idx_broadcast_edges_session_child_state;",
    "DROP INDEX IF EXISTS idx_broadcast_targets_session_state;",
    "DROP TABLE IF EXISTS broadcast_edges;",
    "DROP TABLE IF EXISTS broadcast_targets;",
    "DROP TABLE IF EXISTS broadcast_sessions;",
)


def upgrade(conn: DuckDBPyConnection) -> None:
    """Apply migration 0019."""
    for query in UP_QUERIES:
        conn.execute(query)


def downgrade(conn: DuckDBPyConnection) -> None:
    """Rollback migration 0019."""
    for query in DOWN_QUERIES:
        conn.execute(query)
