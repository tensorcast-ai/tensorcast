#  Copyright (c) 2025, TensorCast Team.

"""Repository for memory tier telemetry snapshots."""

from tensorcast.global_store.models.memory_tier import MemoryTierSnapshot
from tensorcast.global_store.repositories.base import BaseRepository
from tensorcast.logger import init_logger

logger = init_logger(__name__)


class MemoryTierSnapshotRepository(BaseRepository):
    """Persistence for memory_tier_snapshots."""

    def insert(self, snapshot: MemoryTierSnapshot) -> MemoryTierSnapshot:
        cursor = self.get_cursor()
        cursor.execute(
            """
            INSERT INTO memory_tier_snapshots (
                node_id, epoch_ns, stable_total_bytes, stable_used_bytes,
                preemptible_total_bytes, preemptible_marked_bytes,
                faults_per_sec, rehydrate_p99_ns, enable_preemptible,
                memory_tier_config_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                snapshot.node_id,
                snapshot.epoch_ns,
                snapshot.stable_total_bytes,
                snapshot.stable_used_bytes,
                snapshot.preemptible_total_bytes,
                snapshot.preemptible_marked_bytes,
                snapshot.faults_per_sec,
                snapshot.rehydrate_p99_ns,
                snapshot.enable_preemptible,
                snapshot.memory_tier_config_json,
            ],
        )
        return snapshot

    def prune_before(self, node_id: str, cutoff_epoch_ns: int) -> None:
        """Delete snapshots older than cutoff for a node."""
        if cutoff_epoch_ns <= 0:
            return
        cursor = self.get_cursor()
        cursor.execute(
            """
            DELETE FROM memory_tier_snapshots
            WHERE node_id = ? AND epoch_ns < ?
            """,
            [node_id, cutoff_epoch_ns],
        )

    def prune_to_limit(self, node_id: str, max_rows: int) -> None:
        """Retain only the most recent N rows for a node."""
        if max_rows <= 0:
            return
        cursor = self.get_cursor()
        cursor.execute(
            """
            DELETE FROM memory_tier_snapshots
            WHERE node_id = ?
              AND epoch_ns < (
                SELECT COALESCE(MIN(epoch_ns), 0) FROM (
                  SELECT epoch_ns FROM memory_tier_snapshots
                  WHERE node_id = ?
                  ORDER BY epoch_ns DESC
                  LIMIT ?
                )
              )
            """,
            [node_id, node_id, max_rows],
        )
