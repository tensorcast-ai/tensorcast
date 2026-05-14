#  Copyright (c) 2026, TensorCast Team.

"""Migration 0019: add group version-set realization tables."""

from __future__ import annotations

from duckdb import DuckDBPyConnection

UP_QUERIES: tuple[str, ...] = (
    "ALTER TABLE key_mappings ADD COLUMN target_kind TEXT NOT NULL DEFAULT 'artifact_selection';",
    "ALTER TABLE key_mappings ADD COLUMN group_version_set_id TEXT;",
    "ALTER TABLE key_mappings ADD COLUMN selection_hash BLOB;",
    "ALTER TABLE key_mappings ADD COLUMN manifest_hash BLOB;",
    """
    CREATE TABLE IF NOT EXISTS key_version_targets (
        namespace TEXT NOT NULL DEFAULT '',
        key TEXT NOT NULL,
        generation BIGINT NOT NULL,
        target_kind TEXT NOT NULL CHECK (target_kind IN ('artifact_selection','group_version_set')),
        artifact_id TEXT,
        view_id TEXT,
        group_version_set_id TEXT,
        selection_hash BLOB,
        manifest_hash BLOB,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
        CHECK (
            (
                target_kind = 'artifact_selection'
                AND artifact_id IS NOT NULL
                AND group_version_set_id IS NULL
            )
            OR (
                target_kind = 'group_version_set'
                AND group_version_set_id IS NOT NULL
                AND artifact_id IS NULL
                AND view_id IS NULL
            )
        ),
        PRIMARY KEY (namespace, key, generation)
    );
    """,
    """
    INSERT INTO key_version_targets (
        namespace, key, generation, target_kind, artifact_id, selection_hash, manifest_hash
    )
    SELECT '', key, generation, 'artifact_selection', artifact_id, selection_hash, manifest_hash
    FROM key_mappings
    ON CONFLICT (namespace, key, generation) DO NOTHING;
    """,
    """
    CREATE TABLE IF NOT EXISTS group_version_sets (
        version_set_id TEXT PRIMARY KEY,
        realization_kind TEXT NOT NULL CHECK (realization_kind IN ('same_selection','per_part_selection')),
        namespace TEXT,
        key TEXT,
        key_generation BIGINT,
        total_parts INTEGER NOT NULL,
        manifest_hash BLOB NOT NULL UNIQUE,
        manifest_generation BIGINT NOT NULL DEFAULT 1,
        logical_layout_hash BLOB,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS group_version_set_parts (
        version_set_id TEXT NOT NULL,
        part_id TEXT NOT NULL,
        artifact_id TEXT NOT NULL,
        view_id TEXT,
        requested_byte_space TEXT NOT NULL,
        selection_hash BLOB NOT NULL,
        logical_layout_hash BLOB,
        part_metadata_json TEXT,
        selection_proto BLOB,
        PRIMARY KEY (version_set_id, part_id),
        FOREIGN KEY (version_set_id) REFERENCES group_version_sets(version_set_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS group_realization_transactions (
        transaction_id TEXT PRIMARY KEY,
        group_kind TEXT NOT NULL,
        group_id TEXT NOT NULL,
        epoch BIGINT NOT NULL,
        version_set_id TEXT NOT NULL,
        realization_kind TEXT NOT NULL CHECK (realization_kind IN ('same_selection','per_part_selection')),
        transaction_fingerprint BLOB NOT NULL,
        required_part_ids_json TEXT NOT NULL,
        total_parts INTEGER NOT NULL,
        prepared_count INTEGER NOT NULL DEFAULT 0,
        failed_count INTEGER NOT NULL DEFAULT 0,
        published_count INTEGER NOT NULL DEFAULT 0,
        state TEXT NOT NULL CHECK (
            state IN ('open','resolved','preparing','ready_to_publish','published','aborted','expired')
        ),
        deadline_unix_nanos BIGINT,
        namespace TEXT,
        key TEXT,
        key_generation BIGINT,
        manifest_hash BLOB,
        failure_code TEXT,
        failure_detail TEXT,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
        last_state_change_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (group_kind, group_id, epoch),
        FOREIGN KEY (version_set_id) REFERENCES group_version_sets(version_set_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS group_realization_members (
        transaction_id TEXT NOT NULL,
        part_id TEXT NOT NULL,
        daemon_id TEXT NOT NULL DEFAULT '',
        worker_id TEXT,
        daemon_session_id TEXT,
        materialization_attempt_id TEXT,
        artifact_id TEXT NOT NULL,
        view_id TEXT,
        requested_byte_space TEXT NOT NULL,
        selection_hash BLOB NOT NULL,
        member_fingerprint BLOB,
        state TEXT NOT NULL CHECK (
            state IN ('joined','preparing','prepared','published','failed','cancelled','expired')
        ),
        staged_binding_id TEXT,
        staged_binding_value_id TEXT,
        staging_token TEXT,
        staging_epoch BIGINT,
        expected_previous_seal_generation BIGINT,
        prepared_value_hash BLOB,
        source_replica_id TEXT,
        source_export_generation BIGINT,
        child_transport_request_id TEXT,
        failure_code TEXT,
        failure_detail TEXT,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (transaction_id, part_id)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_key_version_targets_current ON key_version_targets(namespace, key, generation);",
    "CREATE INDEX IF NOT EXISTS idx_key_version_targets_artifact ON key_version_targets(artifact_id, view_id);",
    "CREATE INDEX IF NOT EXISTS idx_key_version_targets_version_set ON key_version_targets(group_version_set_id);",
    "CREATE INDEX IF NOT EXISTS idx_group_version_set_parts_artifact ON group_version_set_parts(artifact_id, view_id, requested_byte_space);",
    "CREATE INDEX IF NOT EXISTS idx_group_realization_transactions_state_deadline ON group_realization_transactions(state, deadline_unix_nanos);",
    "CREATE INDEX IF NOT EXISTS idx_group_realization_transactions_version_set ON group_realization_transactions(version_set_id);",
    "CREATE INDEX IF NOT EXISTS idx_group_realization_members_state ON group_realization_members(transaction_id, state);",
)

DOWN_QUERIES: tuple[str, ...] = (
    "DROP TABLE IF EXISTS group_realization_members;",
    "DROP TABLE IF EXISTS group_realization_transactions;",
    "DROP TABLE IF EXISTS group_version_set_parts;",
    "DROP TABLE IF EXISTS group_version_sets;",
    "DROP TABLE IF EXISTS key_version_targets;",
    "ALTER TABLE key_mappings DROP COLUMN IF EXISTS manifest_hash;",
    "ALTER TABLE key_mappings DROP COLUMN IF EXISTS selection_hash;",
    "ALTER TABLE key_mappings DROP COLUMN IF EXISTS group_version_set_id;",
    "ALTER TABLE key_mappings DROP COLUMN IF EXISTS target_kind;",
)


def upgrade(conn: DuckDBPyConnection) -> None:
    """Apply migration 0019."""
    for query in UP_QUERIES:
        conn.execute(query)


def downgrade(conn: DuckDBPyConnection) -> None:
    """Rollback migration 0019."""
    for query in DOWN_QUERIES:
        conn.execute(query)
