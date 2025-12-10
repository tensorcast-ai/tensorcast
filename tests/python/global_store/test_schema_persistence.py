#  Copyright (c) 2025, TensorCast Team.

# Copyright (c) 2025, TensorCast Team.

"""Schema smoke tests for persistence placement tables (Design-0041)."""

from __future__ import annotations

import json


def _index_names(conn, table: str) -> set[str]:
    rows = conn.execute(
        "SELECT lower(name) FROM sqlite_master WHERE type = 'index' AND lower(tbl_name) = ?",
        [table.lower()],
    ).fetchall()
    return {row[0] for row in rows}


def test_persistence_tables_and_indexes_present(db_connection) -> None:
    tables = {row[0].lower() for row in db_connection.execute("SHOW TABLES").fetchall()}
    expected_tables = {
        "artifact_placements",
        "artifact_placement_shards",
        "artifact_placement_targets",
        "artifact_placement_summary",
        "artifact_persistence_status",
    }
    assert expected_tables.issubset(tables)

    targets_indexes = _index_names(db_connection, "artifact_placement_targets")
    assert {
        "idx_artifact_placement_targets_node",
        "idx_artifact_placement_targets_plan_state",
    }.issubset(targets_indexes)

    shard_indexes = _index_names(db_connection, "artifact_placement_shards")
    assert "idx_artifact_placement_shards_digest" in shard_indexes

    status_indexes = _index_names(db_connection, "artifact_persistence_status")
    assert "idx_artifact_persistence_status_artifact_state" in status_indexes


def test_persistence_tables_accept_inserts(db_connection) -> None:
    cursor = db_connection.cursor()
    cursor.execute(
        """
        INSERT INTO artifact_placements (plan_id, artifact_id, policy, shard_count)
        VALUES (?, ?, ?, ?)
        """,
        ["plan-1", "artifact-1", "replicated", 1],
    )
    cursor.execute(
        """
        INSERT INTO artifact_placement_shards (
            plan_id, shard_idx, shard_id, size_bytes, content_digest,
            byte_range_start, byte_range_length, chunk_ids
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "plan-1",
            0,
            "artifact-1:0",
            1024,
            "digest-0",
            0,
            1024,
            json.dumps([0, 1]),
        ],
    )
    cursor.execute(
        """
        INSERT INTO artifact_placement_targets (
            plan_id, shard_idx, node_id, lease_id, target_state, degraded_reason
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        ["plan-1", 0, "node-1", "lease-1", "pending", None],
    )
    cursor.execute(
        """
        INSERT INTO artifact_persistence_status (
            task_id, plan_id, artifact_id, state, progress, last_error, degraded_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ["task-1", "plan-1", "artifact-1", "running", 0.25, None, None],
    )

    plan_row = cursor.execute(
        "SELECT policy, shard_count FROM artifact_placements WHERE plan_id = ?",
        ["plan-1"],
    ).fetchone()
    assert plan_row == ("replicated", 1)

    shard_row = cursor.execute(
        """
        SELECT shard_id, size_bytes, content_digest, chunk_ids
        FROM artifact_placement_shards
        WHERE plan_id = ? AND shard_idx = 0
        """,
        ["plan-1"],
    ).fetchone()
    assert shard_row is not None
    assert shard_row[0] == "artifact-1:0"
    assert shard_row[1] == 1024
    assert shard_row[2] == "digest-0"
    assert json.loads(shard_row[3]) == [0, 1]

    target_row = cursor.execute(
        """
        SELECT target_state, lease_id FROM artifact_placement_targets
        WHERE plan_id = ? AND shard_idx = 0 AND node_id = ?
        """,
        ["plan-1", "node-1"],
    ).fetchone()
    assert target_row == ("pending", "lease-1")

    status_row = cursor.execute(
        """
        SELECT state, progress FROM artifact_persistence_status
        WHERE task_id = ?
        """,
        ["task-1"],
    ).fetchone()
    assert status_row == ("running", 0.25)
