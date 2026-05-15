#  Copyright (c) 2025-2026, TensorCast Team.

"""Service for artifact replica operations."""

from typing import List, Optional
from uuid import UUID

from tensorcast.global_store.exceptions import ValidationError
from tensorcast.global_store.metrics import (
    inc_replica_register,
    inc_replica_unregister,
    set_replicas_per_artifact,
    set_replicas_per_memtype,
    set_total_replicas,
)
from tensorcast.global_store.models import ByteSpaceKind, MemoryType, Replica
from tensorcast.global_store.repositories import ReplicaRepository
from tensorcast.global_store.services.address_validation import ensure_routable_address
from tensorcast.logger import init_logger

logger = init_logger(__name__)


class ArtifactService:
    """Business logic for artifact replica operations."""

    def __init__(self, replica_repository: ReplicaRepository):
        """Initialize service with repository."""
        self.replica_repository = replica_repository

    def register_replica(
        self,
        replica: Replica,
        *,
        cursor=None,
    ) -> Replica:
        """
        Register or update a artifact replica.

        Args:
            replica: Replica to register

        Returns:
            Registered or updated replica
        """
        # Validate replica
        if not replica.artifact_id:
            raise ValidationError("artifact_id is required")
        if replica.byte_space.kind == ByteSpaceKind.VIEW and not replica.byte_space.id:
            raise ValidationError("view replicas require byte_space.id (view_id)")
        if replica.byte_space.kind == ByteSpaceKind.CANONICAL and replica.byte_space.id:
            raise ValidationError("canonical replicas must not set byte_space.id")
        if not replica.node_id:
            raise ValidationError("Node ID is required")
        if not replica.worker_id:
            raise ValidationError("Worker ID is required")
        if replica.memory_size <= 0:
            raise ValidationError("memory_size must be positive")
        if replica.max_concurrency <= 0:
            raise ValidationError("max_concurrency must be positive")
        if not (1 <= replica.node_port <= 65535):
            raise ValidationError("node_port must be between 1 and 65535")

        ensure_routable_address(replica.node_address, "node_address")

        # Additional validation for transport metadata consistency
        if replica.remote_memory_keys:
            if not replica.buffer_sizes:
                raise ValidationError(
                    "buffer_sizes must be provided when remote_memory_keys are set"
                )
            if len(replica.remote_memory_keys) != len(replica.buffer_sizes):
                raise ValidationError(
                    "buffer_sizes length must match remote_memory_keys length"
                )
            if sum(replica.buffer_sizes) != replica.memory_size:
                raise ValidationError("sum(buffer_sizes) must equal memory_size")

        if cursor is not None:
            result = self.replica_repository.create_or_update_atomic(replica, cursor)
            inc_replica_register(result.artifact_id, result.memory_type.value)
        else:
            # Use atomic transaction to prevent race conditions
            with self.replica_repository.transaction() as tx_cursor:
                # Use database-level UPSERT to handle concurrent registrations
                result = self.replica_repository.create_or_update_atomic(
                    replica, tx_cursor
                )

                # Update metrics after successful transaction
                inc_replica_register(result.artifact_id, result.memory_type.value)

        # Update gauges outside transaction to avoid lock contention
        self._update_replica_gauges()

        logger.info(f"Registered replica {result.replica_id} for {replica.artifact_id}")
        return result

    def update_heartbeat(self, replica_id: UUID, artifact_id: str) -> bool:
        """Update replica heartbeat timestamp."""
        success = self.replica_repository.update_heartbeat(replica_id, artifact_id)
        if not success:
            logger.warning(f"Failed to update heartbeat for replica {replica_id}")
        return success

    def unregister_replica(self, replica_id: UUID, artifact_id: str) -> bool:
        """Unregister a artifact replica."""
        success = self.replica_repository.delete(replica_id, artifact_id)
        if success:
            logger.info(f"Unregistered replica {replica_id} for {artifact_id}")
            # Metrics
            # We don't have memory_type here. Fetching is expensive; skip label.
            inc_replica_unregister(artifact_id, "unknown")
            self._update_replica_gauges()
        else:
            logger.warning(f"Failed to unregister replica {replica_id}")
        return success

    def unregister_by_worker(
        self,
        *,
        worker_id: str,
        artifact_id: str,
        memory_type: MemoryType | None = None,
        device_id: int | None = None,
    ) -> bool:
        """Unregister a replica by worker identity and artifact id.

        Optionally disambiguate by memory_type and device_id.
        """
        try:
            replicas = self.replica_repository.get_replicas_by_worker(worker_id)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to query replicas for worker_id=%s", worker_id)
            return False

        candidates = [r for r in replicas if r.artifact_id == artifact_id]
        if memory_type is not None:
            candidates = [r for r in candidates if r.memory_type == memory_type]
        if device_id is not None:
            candidates = [r for r in candidates if r.device_id == device_id]

        if not candidates:
            logger.warning(
                "No replica found for artifact_id=%s worker_id=%s device_id=%s memory_type=%s",
                artifact_id,
                worker_id,
                device_id,
                memory_type.value if memory_type else "",
            )
            return False

        # Choose the most specific candidate (prefer GPU over others by priority, then latest created)
        candidates.sort(key=lambda r: (r.memory_type.priority, r.created_at or 0))
        chosen = candidates[0]
        ok = self.replica_repository.delete(chosen.replica_id, artifact_id)
        if ok:
            logger.info(
                "Unregistered replica by worker: artifact_id=%s replica_id=%s worker_id=%s device_id=%s memory_type=%s",
                artifact_id,
                chosen.replica_id,
                worker_id,
                chosen.device_id,
                chosen.memory_type.value,
            )
            inc_replica_unregister(artifact_id, chosen.memory_type.value)
            self._update_replica_gauges()
        else:
            logger.warning(
                "Failed to unregister replica by worker: artifact_id=%s replica_id=%s worker_id=%s",
                artifact_id,
                chosen.replica_id,
                worker_id,
            )
        return ok

    def get_artifact_replicas(
        self, artifact_id: str, view_id: str | None = None
    ) -> List[Replica]:
        """Get all available replicas for an artifact."""
        replicas = self.replica_repository.find_by_filters(
            artifact_id=artifact_id, view_id=view_id
        )

        # Filter only available replicas
        available = [r for r in replicas if r.is_available]

        # Sort by priority
        available.sort(
            key=lambda r: (
                # GPU > RAM > DISK
                0
                if r.memory_type == MemoryType.GPU
                else 1
                if r.memory_type == MemoryType.RAM
                else 2,
                # Lower load ratio
                r.load_ratio,
            )
        )

        return available

    def list_replicas(
        self,
        artifact_id: Optional[str] = None,
        view_id: Optional[str] = None,
        node_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
    ) -> List[Replica]:
        """List replicas with optional filters."""
        return self.replica_repository.find_by_filters(
            artifact_id=artifact_id,
            view_id=view_id,
            node_id=node_id,
            memory_type=memory_type,
        )

    def batch_get_replica_counts(
        self, artifact_ids: list[str]
    ) -> dict[str, tuple[int, int]]:
        """Return replica and available counts keyed by artifact_id."""
        return self.replica_repository.count_replicas_by_artifact_ids(artifact_ids)

    def mark_unavailable_by_worker(self, worker_id: str) -> None:
        """Mark all replicas for a worker as unavailable."""
        self.replica_repository.mark_unavailable_by_worker(worker_id)
        logger.info(f"Marked all replicas for worker {worker_id} as unavailable")

    # ------------------------------------------------------------------
    # Internal helpers – metrics
    # ------------------------------------------------------------------

    def _update_replica_gauges(self) -> None:
        """Refresh replica-related gauges from database state."""

        replicas = self.replica_repository.list_all_replicas()

        # Global count
        set_total_replicas(len(replicas))

        # Per-artifact and per-memtype counts
        per_model: dict[str, int] = {}
        per_memtype: dict[str, int] = {}

        for rep in replicas:
            per_model[rep.artifact_id] = per_model.get(rep.artifact_id, 0) + 1
            per_memtype[rep.memory_type.value] = (
                per_memtype.get(rep.memory_type.value, 0) + 1
            )
        for artifact_id, count in per_model.items():
            set_replicas_per_artifact(artifact_id, count)
        for memtype, count in per_memtype.items():
            set_replicas_per_memtype(memtype, count)
