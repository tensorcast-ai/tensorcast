#  Copyright (c) 2025, TensorCast Team.

# Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import json

import duckdb
import pytest

from tensorcast.global_store.db_utils import init_db
from tensorcast.global_store.models import (
    PersistenceShardStatus,
    PersistenceStatus,
    PlacementShard,
    PlacementTarget,
    Worker,
    WorkerMemoryTierState,
)
from tensorcast.global_store.repositories import (
    ArtifactPersistenceStatusRepository,
    ArtifactPlacementRepository,
)
from tensorcast.global_store.services.placement_service import PlacementService


class _WorkerRepoStub:
    def __init__(self, workers: list[Worker]) -> None:
        self._workers = workers

    def find_active(self, include_unavailable: bool = False):  # pragma: no cover - trivial
        return self._workers


def _service_with_workers(workers: list[Worker]) -> tuple[PlacementService, duckdb.DuckDBPyConnection]:
    conn = duckdb.connect()
    init_db(conn.cursor())
    placement_repo = ArtifactPlacementRepository(conn)
    status_repo = ArtifactPersistenceStatusRepository(conn)
    service = PlacementService(_WorkerRepoStub(workers), placement_repo, status_repo)
    return service, conn


def _shard(size_bytes: int) -> PlacementShard:
    return PlacementShard(
        plan_id="",
        shard_idx=0,
        shard_id="artifact-1:0",
        size_bytes=size_bytes,
        content_digest="digest-0",
        byte_range_start=0,
        byte_range_length=size_bytes,
        chunk_ids=[0, 1],
    )


def test_plan_placement_marks_degraded_without_remote_capacity() -> None:
    shard = _shard(64 * 1024 * 1024)
    remote = Worker(
        node_id="node-remote",
        memory_tier_state=WorkerMemoryTierState(
            stable_total_bytes=32 * 1024 * 1024,
            stable_used_bytes=31 * 1024 * 1024,
        ),
    )
    service, conn = _service_with_workers([remote])

    plan = service.plan_placement(
        artifact_id="artifact-1",
        placement_policy="replicated",
        shards=[shard],
        source_node_id="node-source",
    )

    assert plan.degraded_reason == "insufficient_remote_capacity"
    assert {t.node_id for t in plan.targets} == {"node-source"}
    summary_row = conn.execute(
        "SELECT plan_json FROM artifact_placement_summary WHERE plan_id = ?",
        [plan.plan_id],
    ).fetchone()
    assert summary_row is not None
    summary = json.loads(summary_row[0])
    assert summary[0]["degraded_reason"] == "insufficient_remote_capacity"


def test_plan_placement_prefers_remote_with_stable_headroom() -> None:
    shard = _shard(32 * 1024 * 1024)
    remote_low = Worker(
        node_id="node-low",
        memory_tier_state=WorkerMemoryTierState(
            stable_total_bytes=16 * 1024 * 1024,
            stable_used_bytes=15 * 1024 * 1024,
        ),
    )
    remote_ok = Worker(
        node_id="node-ok",
        memory_tier_state=WorkerMemoryTierState(
            stable_total_bytes=256 * 1024 * 1024,
            stable_used_bytes=32 * 1024 * 1024,
        ),
    )
    service, conn = _service_with_workers([remote_low, remote_ok])

    plan = service.plan_placement(
        artifact_id="artifact-2",
        placement_policy="replicated",
        shards=[shard],
        source_node_id="node-source",
    )

    remote_targets = [t.node_id for t in plan.targets if t.node_id != "node-source"]
    assert remote_targets == ["node-ok"]
    summary_row = conn.execute(
        "SELECT plan_json FROM artifact_placement_summary WHERE plan_id = ?",
        [plan.plan_id],
    ).fetchone()
    nodes = json.loads(summary_row[0])[0]["nodes"]
    assert set(nodes) == {"node-source", "node-ok"}


def test_record_status_updates_targets_and_task_state() -> None:
    shard = _shard(16 * 1024 * 1024)
    remote_ok = Worker(
        node_id="node-ok",
        memory_tier_state=WorkerMemoryTierState(
            stable_total_bytes=128 * 1024 * 1024,
            stable_used_bytes=0,
        ),
    )
    service, conn = _service_with_workers([remote_ok])
    plan = service.plan_placement(
        artifact_id="artifact-3",
        placement_policy="replicated",
        shards=[shard],
        source_node_id="node-source",
    )

    shard_status = PersistenceShardStatus(
        shard_id=plan.shards[0].shard_id,
        shard_idx=0,
        state="running",
        progress=0.5,
        degraded_reason=None,
        last_error=None,
        targets=[
            PlacementTarget(
                plan_id=plan.plan_id,
                shard_idx=0,
                node_id=remote_ok.node_id,
                lease_id="lease-1",
                target_state="copying",
                degraded_reason=None,
            )
        ],
    )
    service.record_status(
        PersistenceStatus(
            task_id="task-1",
            plan_id=plan.plan_id,
            artifact_id="artifact-3",
            state="running",
            progress=0.5,
            last_error=None,
            degraded_reason=None,
        ),
        [shard_status],
    )

    target_row = conn.execute(
        """
        SELECT target_state, lease_id
        FROM artifact_placement_targets
        WHERE plan_id = ? AND shard_idx = 0 AND node_id = ?
        """,
        [plan.plan_id, remote_ok.node_id],
    ).fetchone()
    assert target_row == ("copying", "lease-1")

    status_row = conn.execute(
        """
        SELECT state, progress, degraded_reason
        FROM artifact_persistence_status
        WHERE task_id = ?
        """,
        ["task-1"],
    ).fetchone()
    assert status_row == ("running", pytest.approx(0.5), None)
