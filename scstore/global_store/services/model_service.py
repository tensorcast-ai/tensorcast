#  Copyright (c) 2025, StepCast Team.

"""Service for model replica operations."""

from typing import List, Optional
from uuid import UUID

from scstore.global_store.exceptions import ValidationError
from scstore.global_store.metrics import (
    inc_replica_register,
    inc_replica_unregister,
    set_model_replicas,
    set_replicas_per_memtype,
    set_replicas_per_model,
)
from scstore.global_store.models import MemoryType, ModelReplica
from scstore.global_store.repositories import ModelReplicaRepository
from scstore.logger import init_logger

logger = init_logger(__name__)


class ModelService:
    """Business logic for model replica operations."""

    def __init__(self, replica_repository: ModelReplicaRepository):
        """Initialize service with repository."""
        self.replica_repository = replica_repository

    def register_replica(self, replica: ModelReplica) -> ModelReplica:
        """
        Register or update a model replica.

        Args:
            replica: ModelReplica to register

        Returns:
            Registered or updated replica
        """
        # Validate replica
        if not replica.model_name:
            raise ValidationError("Model name is required")
        if not replica.node_id:
            raise ValidationError("Node ID is required")
        if not replica.worker_id:
            raise ValidationError("Worker ID is required")
        if replica.memory_size <= 0:
            raise ValidationError("memory_size must be positive")
        if replica.max_concurrency <= 0:
            raise ValidationError("max_concurrency must be positive")

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

        # Use atomic transaction to prevent race conditions
        with self.replica_repository.transaction() as cursor:
            # Use database-level UPSERT to handle concurrent registrations
            result = self.replica_repository.create_or_update_atomic(replica, cursor)

            # Update metrics after successful transaction
            inc_replica_register(result.model_name, result.memory_type.value)

        # Update gauges outside transaction to avoid lock contention
        self._update_replica_gauges()

        logger.info(f"Registered replica {result.replica_id} for {replica.model_name}")
        return result

    def update_heartbeat(self, replica_id: UUID, model_name: str) -> bool:
        """Update replica heartbeat timestamp."""
        success = self.replica_repository.update_heartbeat(replica_id, model_name)
        if not success:
            logger.warning(f"Failed to update heartbeat for replica {replica_id}")
        return success

    def unregister_replica(self, replica_id: UUID, model_name: str) -> bool:
        """Unregister a model replica."""
        success = self.replica_repository.delete(replica_id, model_name)
        if success:
            logger.info(f"Unregistered replica {replica_id} for {model_name}")
            # Metrics
            # We don't have memory_type here. Fetching is expensive; skip label.
            inc_replica_unregister(model_name, "unknown")
            self._update_replica_gauges()
        else:
            logger.warning(f"Failed to unregister replica {replica_id}")
        return success

    def get_model_replicas(self, model_name: str) -> List[ModelReplica]:
        """Get all available replicas for a model."""
        replicas = self.replica_repository.find_by_filters(model_name=model_name)

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
        model_name: Optional[str] = None,
        node_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
    ) -> List[ModelReplica]:
        """List replicas with optional filters."""
        return self.replica_repository.find_by_filters(
            model_name=model_name,
            node_id=node_id,
            memory_type=memory_type,
        )

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
        set_model_replicas(len(replicas))

        # Per-model and per-memtype counts
        per_model: dict[str, int] = {}
        per_memtype: dict[str, int] = {}

        for rep in replicas:
            per_model[rep.model_name] = per_model.get(rep.model_name, 0) + 1
            per_memtype[rep.memory_type.value] = (
                per_memtype.get(rep.memory_type.value, 0) + 1
            )

        for model_name, count in per_model.items():
            set_replicas_per_model(model_name, count)

        for memtype, count in per_memtype.items():
            set_replicas_per_memtype(memtype, count)
