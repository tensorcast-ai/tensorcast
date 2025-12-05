#  Copyright (c) 2025, TensorCast Team.

# Copyright (c) 2025, TensorCast Team.

"""Business logic for memory tier telemetry and leases."""

import time
import uuid
from typing import Iterable

from tensorcast.global_store import metrics
from tensorcast.global_store.models.memory_tier import (
    ChunkRange,
    MemoryTierLease,
    MemoryTierSnapshot,
)
from tensorcast.global_store.repositories.memory_tier_lease_repository import (
    MemoryTierLeaseRepository,
)
from tensorcast.global_store.repositories.memory_tier_snapshot_repository import (
    MemoryTierSnapshotRepository,
)
from tensorcast.logger import init_logger

logger = init_logger(__name__)


class MemoryTierService:
    """Handle memory tier telemetry ingestion and lease lifecycle."""

    def __init__(
        self,
        snapshot_repository: MemoryTierSnapshotRepository,
        lease_repository: MemoryTierLeaseRepository,
        snapshot_retention_ns: int,
        snapshot_max_rows: int,
    ):
        self.snapshot_repository = snapshot_repository
        self.lease_repository = lease_repository
        self.snapshot_retention_ns = max(0, snapshot_retention_ns)
        self.snapshot_max_rows = max(0, snapshot_max_rows)

    def publish_status(self, snapshot: MemoryTierSnapshot) -> None:
        """Persist telemetry snapshot and emit metrics."""
        self.snapshot_repository.insert(snapshot)
        if self.snapshot_retention_ns > 0:
            cutoff = snapshot.epoch_ns - self.snapshot_retention_ns
            self.snapshot_repository.prune_before(snapshot.node_id, cutoff)
        if self.snapshot_max_rows > 0:
            self.snapshot_repository.prune_to_limit(
                snapshot.node_id, self.snapshot_max_rows
            )

        metrics.observe_memory_tier_snapshot(
            snapshot.node_id,
            snapshot.stable_total_bytes,
            snapshot.stable_used_bytes,
            snapshot.preemptible_total_bytes,
            snapshot.preemptible_marked_bytes,
            snapshot.faults_per_sec,
            snapshot.rehydrate_p99_ns,
            snapshot.enable_preemptible,
        )

    def request_lease(
        self,
        node_id: str,
        kind: str,
        artifact_id: str,
        chunk_range: ChunkRange,
        chunk_ids: list[int],
        ledger_version: int,
        bytes_count: int,
        workload_id: str,
        request_id: str,
        issued_at_ns: int | None = None,
    ) -> MemoryTierLease:
        """Create or reuse a lease keyed by request_id."""
        existing = self.lease_repository.find_by_request(node_id, request_id)
        if existing:
            return existing
        lease = MemoryTierLease(
            lease_id=uuid.uuid4().hex,
            node_id=node_id,
            kind=kind,
            artifact_id=artifact_id,
            chunk_range=chunk_range,
            chunk_ids=chunk_ids,
            ledger_version=ledger_version,
            bytes=bytes_count,
            workload_id=workload_id,
            state="pending",
            request_id=request_id,
            issued_at_ns=issued_at_ns or int(time.time_ns()),
            ack_epoch_ns=None,
            expires_at_ns=None,
        )
        return self.lease_repository.create(lease)

    def acknowledge(
        self,
        lease_id: str,
        action: str,
        artifact_id: str,
        chunk_ids: list[int],
        ledger_version: int,
        chunk_range: ChunkRange,
        bytes_count: int,
        request_id: str,
        ack_epoch_ns: int | None,
    ) -> MemoryTierLease | None:
        """Transition lease based on daemon acknowledgement."""
        existing = self.lease_repository.find_by_id(lease_id)
        if not existing:
            return None
        if artifact_id and existing.artifact_id != artifact_id:
            logger.warning(
                "Lease %s artifact mismatch: expected %s got %s",
                lease_id,
                existing.artifact_id,
                artifact_id,
            )
            return None

        if action == "acquired":
            return self.lease_repository.acknowledge_acquired(
                lease_id,
                artifact_id,
                chunk_ids,
                ledger_version,
                chunk_range,
                bytes_count,
                request_id,
                ack_epoch_ns,
            )
        if action == "released":
            return self.lease_repository.acknowledge_released(
                lease_id,
                artifact_id,
                chunk_ids,
                chunk_range,
                ledger_version,
                bytes_count,
                request_id,
                ack_epoch_ns,
            )
        logger.warning(
            "Unknown acknowledgement action=%s for lease_id=%s", action, lease_id
        )
        return None

    def revoke(self, lease_id: str) -> MemoryTierLease | None:
        """Mark lease as revoking."""
        return self.lease_repository.mark_revoking(lease_id)

    def list_outstanding(
        self, node_id: str, states: Iterable[str] | None = None
    ) -> list[MemoryTierLease]:
        """List active/pending/revoking leases for a node."""
        return self.lease_repository.list_outstanding(node_id, states)

    def list_latest(self, node_id: str | None = None) -> list[MemoryTierSnapshot]:
        """Return latest snapshot per node (optionally filtered by node_id)."""
        return self.snapshot_repository.list_latest(node_id)
