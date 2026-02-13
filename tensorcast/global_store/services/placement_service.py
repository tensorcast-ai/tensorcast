#  Copyright (c) 2025-2026, TensorCast Team.

"""Placement planning and persistence status ingestion."""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import replace
from typing import Sequence

from tensorcast.global_store.models import (
    PersistenceShardStatus,
    PersistenceStatus,
    PlacementPlan,
    PlacementShard,
    PlacementTarget,
    Worker,
)
from tensorcast.global_store.repositories import (
    ArtifactPersistenceStatusRepository,
    ArtifactPlacementRepository,
    WorkerRepository,
)
from tensorcast.global_store.repositories.base import is_transient_tx_conflict
from tensorcast.logger import init_logger

logger = init_logger(__name__)


class PlacementService:
    """Coordinate placement planning and status updates."""

    def __init__(
        self,
        worker_repository: WorkerRepository,
        placement_repository: ArtifactPlacementRepository,
        status_repository: ArtifactPersistenceStatusRepository,
    ) -> None:
        self.worker_repository = worker_repository
        self.placement_repository = placement_repository
        self.status_repository = status_repository
        self._lane_guard = threading.Lock()
        self._artifact_lanes: dict[str, threading.RLock] = {}

    def _lane_for_artifact(self, artifact_id: str) -> threading.RLock:
        with self._lane_guard:
            lane = self._artifact_lanes.get(artifact_id)
            if lane is None:
                lane = threading.RLock()
                self._artifact_lanes[artifact_id] = lane
            return lane

    def plan_placement(
        self,
        artifact_id: str,
        placement_policy: str,
        shards: Sequence[PlacementShard],
        *,
        source_node_id: str,
    ) -> PlacementPlan:
        """Plan shard placement using stable memory headroom."""
        with self._lane_for_artifact(artifact_id):
            plan_id = uuid.uuid4().hex
            active_workers = self.worker_repository.find_active(
                include_unavailable=False
            )
            shard_targets: list[PlacementTarget] = []
            shard_models: list[PlacementShard] = []
            degraded_reason: str | None = None

            for shard in shards:
                shard_model = replace(shard, plan_id=plan_id)
                shard_models.append(shard_model)
                # Local target is always present
                shard_targets.append(
                    PlacementTarget(
                        plan_id=plan_id,
                        shard_idx=shard_model.shard_idx,
                        node_id=source_node_id,
                        lease_id=None,
                        target_state="pending",
                        degraded_reason=None,
                    )
                )
                if placement_policy == "local_only":
                    continue

                remote_candidates = [
                    w
                    for w in active_workers
                    if w.node_id != source_node_id
                    and self._has_stable_capacity(w, shard_model.size_bytes)
                ]
                if not remote_candidates:
                    shard_model.degraded_reason = "insufficient_remote_capacity"
                    degraded_reason = degraded_reason or shard_model.degraded_reason
                    logger.warning(
                        "placement.plan.degraded",
                        extra={
                            "tc.artifact.id": artifact_id,
                            "tc.placement.policy": placement_policy,
                            "tc.node.source": source_node_id,
                            "tc.shard.idx": shard_model.shard_idx,
                        },
                    )
                    continue
                remote = remote_candidates[
                    shard_model.shard_idx % len(remote_candidates)
                ]
                shard_targets.append(
                    PlacementTarget(
                        plan_id=plan_id,
                        shard_idx=shard_model.shard_idx,
                        node_id=remote.node_id,
                        lease_id=None,
                        target_state="pending",
                        degraded_reason=None,
                    )
                )

            plan = PlacementPlan(
                plan_id=plan_id,
                artifact_id=artifact_id,
                policy=placement_policy,
                shard_count=len(shard_models),
                shards=shard_models,
                targets=shard_targets,
                degraded_reason=degraded_reason,
            )

            summary_json = json.dumps(
                [
                    {
                        "shard_id": shard.shard_id,
                        "size_bytes": shard.size_bytes,
                        "content_digest": shard.content_digest,
                        "nodes": [
                            target.node_id
                            for target in shard_targets
                            if target.shard_idx == shard.shard_idx
                        ],
                        "lease_ids": [
                            target.lease_id
                            for target in shard_targets
                            if target.shard_idx == shard.shard_idx and target.lease_id
                        ],
                        "degraded_reason": shard.degraded_reason
                        or degraded_reason
                        or "",
                    }
                    for shard in shard_models
                ]
            )
            self.placement_repository.upsert_plan(plan, summary_json=summary_json)
            return plan

    def record_status(
        self,
        status: PersistenceStatus,
        shard_statuses: Sequence[PersistenceShardStatus],
    ) -> None:
        """Persist shard-level updates and overall task status."""
        target_updates: list[PlacementTarget] = []
        for shard_status in shard_statuses:
            target_updates.extend(shard_status.targets)

        max_attempts = 3
        with self._lane_for_artifact(status.artifact_id):
            for attempt in range(1, max_attempts + 1):
                try:
                    if target_updates:
                        self.placement_repository.update_targets(target_updates)
                    self.status_repository.upsert(status)
                    return
                except Exception as exc:  # noqa: BLE001
                    if not is_transient_tx_conflict(exc) or attempt >= max_attempts:
                        raise
                    backoff_s = min(0.2, 0.02 * (2 ** (attempt - 1)))
                    logger.warning(
                        "placement.record_status transient conflict "
                        "artifact_id=%s plan_id=%s task_id=%s attempt=%s/%s "
                        "retry_in_ms=%s error=%s",
                        status.artifact_id,
                        status.plan_id,
                        status.task_id,
                        attempt,
                        max_attempts,
                        int(backoff_s * 1000),
                        exc,
                    )
                    time.sleep(backoff_s)

    @staticmethod
    def _has_stable_capacity(worker: Worker, required_bytes: int) -> bool:
        if required_bytes <= 0:
            return True
        tier = worker.memory_tier_state
        if tier is None:
            return False
        available = tier.stable_total_bytes - tier.stable_used_bytes
        return available > required_bytes
