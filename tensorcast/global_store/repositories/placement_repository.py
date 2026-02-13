#  Copyright (c) 2025-2026, TensorCast Team.

"""Repositories for placement plans and persistence status."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable
from contextlib import contextmanager

from duckdb import DuckDBPyConnection

from tensorcast.global_store.models import (
    PersistenceStatus,
    PlacementPlan,
    PlacementTarget,
)
from tensorcast.global_store.repositories.base import BaseRepository


class ArtifactPlacementRepository(BaseRepository):
    """Persistence helpers for placement plans."""

    def __init__(self, connection: DuckDBPyConnection):
        super().__init__(connection)
        self._write_lock = threading.RLock()

    @contextmanager
    def transaction(self):
        with self._write_lock, super().transaction() as cursor:
            yield cursor

    def upsert_plan(
        self, plan: PlacementPlan, *, summary_json: str | None = None
    ) -> None:
        """Persist placement plan rows atomically.

        Replaces any prior plan for the same artifact_id.
        """
        with self.transaction() as cursor:
            existing = cursor.execute(
                "SELECT plan_id FROM artifact_placements WHERE artifact_id = ?",
                [plan.artifact_id],
            ).fetchall()
            for (plan_id,) in existing:
                cursor.execute(
                    "DELETE FROM artifact_placement_targets WHERE plan_id = ?",
                    [plan_id],
                )
                cursor.execute(
                    "DELETE FROM artifact_placement_shards WHERE plan_id = ?",
                    [plan_id],
                )
                cursor.execute(
                    "DELETE FROM artifact_placement_summary WHERE plan_id = ?",
                    [plan_id],
                )
                cursor.execute(
                    "DELETE FROM artifact_placements WHERE plan_id = ?", [plan_id]
                )

            cursor.execute(
                """
                INSERT INTO artifact_placements (plan_id, artifact_id, policy, shard_count)
                VALUES (?, ?, ?, ?)
                """,
                [plan.plan_id, plan.artifact_id, plan.policy, plan.shard_count],
            )

            if plan.shards:
                cursor.executemany(
                    """
                    INSERT INTO artifact_placement_shards (
                        plan_id, shard_idx, shard_id, size_bytes, content_digest,
                        byte_range_start, byte_range_length, chunk_ids
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            shard.plan_id,
                            shard.shard_idx,
                            shard.shard_id,
                            shard.size_bytes,
                            shard.content_digest,
                            shard.byte_range_start,
                            shard.byte_range_length,
                            json.dumps(list(shard.chunk_ids)),
                        )
                        for shard in plan.shards
                    ],
                )

            if plan.targets:
                cursor.executemany(
                    """
                    INSERT INTO artifact_placement_targets (
                        plan_id, shard_idx, node_id, lease_id, target_state, degraded_reason
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            target.plan_id,
                            target.shard_idx,
                            target.node_id,
                            target.lease_id,
                            target.target_state,
                            target.degraded_reason,
                        )
                        for target in plan.targets
                    ],
                )

            if summary_json is not None:
                cursor.execute(
                    "DELETE FROM artifact_placement_summary WHERE plan_id = ?",
                    [plan.plan_id],
                )
                cursor.execute(
                    """
                    INSERT INTO artifact_placement_summary (plan_id, plan_json)
                    VALUES (?, ?)
                    """,
                    [plan.plan_id, summary_json],
                )

    def update_targets(self, targets: Iterable[PlacementTarget]) -> None:
        """Update target rows for a plan."""
        with self.transaction() as cursor:
            for target in targets:
                updated = cursor.execute(
                    """
                    UPDATE artifact_placement_targets
                    SET target_state = ?, degraded_reason = ?, lease_id = ?
                    WHERE plan_id = ? AND shard_idx = ? AND node_id = ?
                    """,
                    [
                        target.target_state,
                        target.degraded_reason,
                        target.lease_id,
                        target.plan_id,
                        target.shard_idx,
                        target.node_id,
                    ],
                ).rowcount
                if updated == 0:
                    cursor.execute(
                        """
                        INSERT INTO artifact_placement_targets (
                            plan_id, shard_idx, node_id, lease_id, target_state, degraded_reason
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        [
                            target.plan_id,
                            target.shard_idx,
                            target.node_id,
                            target.lease_id,
                            target.target_state,
                            target.degraded_reason,
                        ],
                    )


class ArtifactPersistenceStatusRepository(BaseRepository):
    """Persistence task status storage."""

    def __init__(self, connection: DuckDBPyConnection):
        super().__init__(connection)
        self._write_lock = threading.RLock()

    @contextmanager
    def transaction(self):
        with self._write_lock, super().transaction() as cursor:
            yield cursor

    def upsert(self, status: PersistenceStatus) -> PersistenceStatus:
        with self.transaction() as cursor:
            cursor.execute(
                "DELETE FROM artifact_persistence_status WHERE task_id = ?",
                [status.task_id],
            )
            cursor.execute(
                """
                INSERT INTO artifact_persistence_status (
                    task_id, plan_id, artifact_id, state, progress, last_error, degraded_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    status.task_id,
                    status.plan_id,
                    status.artifact_id,
                    status.state,
                    status.progress,
                    status.last_error,
                    status.degraded_reason,
                ],
            )
        return status

    def get_by_task_id(self, task_id: str) -> PersistenceStatus | None:
        cursor = self.get_cursor()
        try:
            row = cursor.execute(
                """
                SELECT task_id, plan_id, artifact_id, state, progress, last_error, degraded_reason
                FROM artifact_persistence_status
                WHERE task_id = ?
                """,
                [task_id],
            ).fetchone()
            if row is None:
                return None
            return PersistenceStatus(
                task_id=row[0],
                plan_id=row[1],
                artifact_id=row[2],
                state=row[3],
                progress=row[4],
                last_error=row[5],
                degraded_reason=row[6],
            )
        finally:
            cursor.close()

    def get_by_artifact(
        self, artifact_id: str, *, state: str | None = None
    ) -> list[PersistenceStatus]:
        cursor = self.get_cursor()
        try:
            query = """
                SELECT task_id, plan_id, artifact_id, state, progress, last_error, degraded_reason
                FROM artifact_persistence_status
                WHERE artifact_id = ?
            """
            params: list[object] = [artifact_id]
            if state:
                query += " AND state = ?"
                params.append(state)
            rows = cursor.execute(query, params).fetchall()
            return [
                PersistenceStatus(
                    task_id=row[0],
                    plan_id=row[1],
                    artifact_id=row[2],
                    state=row[3],
                    progress=row[4],
                    last_error=row[5],
                    degraded_reason=row[6],
                )
                for row in rows
            ]
        finally:
            cursor.close()
