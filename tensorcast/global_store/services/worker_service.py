#  Copyright (c) 2025-2026, TensorCast Team.

"""Service for worker operations."""

import uuid

from tensorcast.global_store.config import get_config
from tensorcast.global_store.exceptions import ValidationError
from tensorcast.global_store.models import Worker
from tensorcast.global_store.repositories import ReplicaRepository, WorkerRepository
from tensorcast.global_store.services.address_validation import (
    ensure_routable_address,
)
from tensorcast.global_store.services.worker_control_reducer import WorkerControlReducer
from tensorcast.logger import init_logger

logger = init_logger(__name__)


class WorkerService:
    """Business logic for worker operations."""

    def __init__(
        self,
        worker_repository: WorkerRepository,
        replica_repository: ReplicaRepository,
        *,
        control_reducer: WorkerControlReducer | None = None,
    ):
        """Initialize service with repositories."""
        self.worker_repository = worker_repository
        self.replica_repository = replica_repository
        self.config = get_config()
        self._control_reducer = control_reducer

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _generate_worker_id(node_id: str) -> str:
        # UUID-based to avoid time-collision and cross-host clashes
        return f"worker_{node_id}_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _daemon_worker_key(daemon_id: str) -> str:
        return f"daemon:{daemon_id.strip()}"

    def resolve_control_worker_key(
        self,
        *,
        worker_id: str,
        daemon_id: str | None = None,
    ) -> str:
        """Resolve a reducer key to stable daemon identity when available."""
        normalized_daemon_id = (daemon_id or "").strip()
        if normalized_daemon_id:
            return self._daemon_worker_key(normalized_daemon_id)

        existing = self.worker_repository.find_by_id(worker_id, include_inactive=True)
        if existing and existing.daemon_id.strip():
            return self._daemon_worker_key(existing.daemon_id)

        logger.warning(
            "resolve_control_worker_key fallback worker_id=%s daemon_id=%s reason=no_daemon_identity",
            worker_id,
            normalized_daemon_id,
        )
        return f"worker:{worker_id}"

    def _submit_control_intent(self, *, worker_key: str, kind: str, operation):
        if self._control_reducer is None:
            return operation()
        return self._control_reducer.submit(
            worker_key=worker_key,
            kind=kind,
            operation=operation,
        )

    def register_worker(self, worker: Worker) -> Worker:
        """
        Register a new worker or update existing one.

        Args:
            worker: Worker to register

        Returns:
            Registered worker with assigned ID
        """
        # Validate worker
        if not worker.node_id:
            raise ValidationError("Node ID is required")
        if not worker.node_address:
            raise ValidationError("Node address is required")
        if not (1 <= worker.grpc_port <= 65535):
            raise ValidationError("gRPC port must be between 1 and 65535")
        if not (1 <= worker.p2p_port <= 65535):
            raise ValidationError("Comm port must be between 1 and 65535")

        ensure_routable_address(worker.node_address, "node_address")

        daemon_id = (worker.daemon_id or "").strip()
        if not daemon_id:
            raise ValidationError("daemon_id is required")

        def _register() -> Worker:
            # Prefer stable daemon_id identity; address/port are routing attributes.
            existing = self.worker_repository.find_by_daemon_id(
                daemon_id, include_inactive=True
            )
            by_addr = self.worker_repository.find_by_address_port(
                worker.node_address, worker.grpc_port, include_inactive=True
            )

            if existing:
                # Reject address/port conflicts, regardless of node_id match.
                if by_addr and by_addr.worker_id != existing.worker_id:
                    raise ValidationError(
                        "Address/port already registered by another worker"
                    )

                reset_state_tracking = bool(existing.inactive_at is not None)
                existing.daemon_id = daemon_id
                existing.node_id = worker.node_id
                existing.node_address = worker.node_address
                existing.grpc_port = worker.grpc_port
                existing.p2p_port = worker.p2p_port
                existing.mem_pool_total_size = worker.mem_pool_total_size
                existing.mem_pool_available_size = worker.mem_pool_available_size
                existing.accepting_new_requests = True
                existing.capability_flags = worker.capability_flags
                return self.worker_repository.update(
                    existing, reset_state_tracking=reset_state_tracking
                )

            if by_addr:
                # Address/port cannot be shared by different daemon_id values unless
                # the existing row is inactive.
                if by_addr.inactive_at is None:
                    raise ValidationError(
                        "Address/port already registered by another worker"
                    )

                reset_state_tracking = True
                by_addr.daemon_id = daemon_id
                by_addr.node_id = worker.node_id
                by_addr.node_address = worker.node_address
                by_addr.grpc_port = worker.grpc_port
                by_addr.p2p_port = worker.p2p_port
                by_addr.mem_pool_total_size = worker.mem_pool_total_size
                by_addr.mem_pool_available_size = worker.mem_pool_available_size
                by_addr.accepting_new_requests = True
                by_addr.capability_flags = worker.capability_flags
                return self.worker_repository.update(
                    by_addr, reset_state_tracking=reset_state_tracking
                )

            # Create new worker
            worker.worker_id = self._generate_worker_id(worker.node_id)
            worker.daemon_id = daemon_id
            logger.debug(f"Registering new worker {worker.worker_id}")
            return self.worker_repository.create(worker)

        return self._submit_control_intent(
            worker_key=self._daemon_worker_key(daemon_id),
            kind="register",
            operation=_register,
        )

    def find_worker_by_address(
        self, node_address: str, grpc_port: int, *, include_inactive: bool = True
    ) -> Worker | None:
        """
        Find a worker by address and port.

        Args:
            node_address: Node address
            grpc_port: gRPC port
            include_inactive: Include inactive worker rows

        Returns:
            Worker if found, None otherwise
        """
        return self.worker_repository.find_by_address_port(
            node_address, grpc_port, include_inactive=include_inactive
        )

    def heartbeat(
        self,
        worker_id: str,
        mem_pool_available_size: int,
        accepting_new_requests: bool,
        capability_flags: int | None = None,
        daemon_id: str | None = None,
    ) -> bool:
        """
        Process worker heartbeat.

        Args:
            worker_id: ID of the worker
            mem_pool_available_size: Available memory pool size
            accepting_new_requests: Whether accepting new requests
            daemon_id: Optional daemon identity for stable reducer lane selection

        Returns:
            True if successful, False if worker not found
        """

        def _heartbeat() -> bool:
            success = self.worker_repository.update_heartbeat(
                worker_id=worker_id,
                mem_pool_available_size=mem_pool_available_size,
                accepting_new_requests=accepting_new_requests,
                capability_flags=(
                    int(capability_flags) if capability_flags is not None else None
                ),
            )
            if not success:
                logger.warning(f"Worker {worker_id} not found for heartbeat")
            return success

        return self._submit_control_intent(
            worker_key=self.resolve_control_worker_key(
                worker_id=worker_id,
                daemon_id=daemon_id,
            ),
            kind="heartbeat",
            operation=_heartbeat,
        )

    def unregister_worker(self, worker_id: str) -> bool:
        """
        Unregister a worker and mark its replicas as unavailable.

        Args:
            worker_id: ID of the worker to unregister

        Returns:
            True if successful
        """

        def _unregister() -> bool:
            # Mark all replicas as unavailable
            self.replica_repository.mark_unavailable_by_worker(worker_id)

            # Mark worker inactive
            success = self.worker_repository.mark_inactive(worker_id)
            if success:
                logger.debug(f"Unregistered worker {worker_id}")
            else:
                logger.debug(f"Failed to unregister worker {worker_id}")
            return success

        return self._submit_control_intent(
            worker_key=self.resolve_control_worker_key(worker_id=worker_id),
            kind="unregister",
            operation=_unregister,
        )

    def list_active_workers(self, include_unavailable: bool = False) -> list[Worker]:
        """
        List all active workers.

        Args:
            include_unavailable: Include workers not accepting requests

        Returns:
            List of active workers
        """
        return self.worker_repository.find_active(include_unavailable)

    def cleanup_inactive_workers(self) -> list[str]:
        """
        Mark inactive workers and clean up their replicas.

        Returns:
            List of cleaned up worker IDs
        """

        timeout_seconds = self.config.heartbeat_timeout_ms / 1000
        stale_workers = self.worker_repository.list_timeout_candidates(timeout_seconds)
        cleaned_worker_ids: list[str] = []

        for worker_id, node_id, daemon_id in stale_workers:
            worker_key = self.resolve_control_worker_key(
                worker_id=worker_id,
                daemon_id=daemon_id,
            )

            def _cleanup_single(
                target_worker_id: str = worker_id,
                target_node_id: str = node_id,
            ) -> str | None:
                inactive_row = self.worker_repository.mark_inactive_if_timed_out(
                    target_worker_id, timeout_seconds
                )
                if inactive_row is None:
                    return None
                self.replica_repository.mark_unavailable_by_worker(target_worker_id)
                logger.info(
                    "Marked inactive worker %s on %s",
                    target_worker_id,
                    target_node_id,
                )
                return target_worker_id

            cleaned = self._submit_control_intent(
                worker_key=worker_key,
                kind="maintenance",
                operation=_cleanup_single,
            )
            if cleaned is not None:
                cleaned_worker_ids.append(str(cleaned))

        return cleaned_worker_ids

    def flush_heartbeats(self) -> None:
        """Legacy compatibility no-op: heartbeats are reducer-owned and immediate."""
