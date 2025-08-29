#  Copyright (c) 2025, TensorCast Team.

"""Service for worker operations."""

import threading
import time

from scstore.global_store.config import get_config
from scstore.global_store.exceptions import ValidationError
from scstore.global_store.models import Worker
from scstore.global_store.repositories import ReplicaRepository, WorkerRepository
from scstore.logger import init_logger

logger = init_logger(__name__)


class WorkerService:
    """Business logic for worker operations."""

    def __init__(
        self,
        worker_repository: WorkerRepository,
        replica_repository: ReplicaRepository,
    ):
        """Initialize service with repositories."""
        self.worker_repository = worker_repository
        self.replica_repository = replica_repository
        self.config = get_config()
        # Initialize heartbeat buffer and start flush thread
        self._heartbeat_buffer: list[tuple[str, int, bool]] = []
        self._heartbeat_lock = threading.Lock()
        self._start_heartbeat_batch_thread()

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

        # Check for existing worker
        existing = self.worker_repository.find_by_address_port(
            worker.node_address, worker.grpc_port
        )

        if existing:
            # Update existing worker
            existing.node_id = worker.node_id
            existing.p2p_port = worker.p2p_port
            existing.mem_pool_total_size = worker.mem_pool_total_size
            existing.mem_pool_available_size = worker.mem_pool_available_size
            existing.accepting_new_requests = True

            logger.info(f"Updating existing worker {existing.worker_id}")
            return self.worker_repository.update(existing)
        else:
            # Create new worker
            worker.worker_id = f"worker_{worker.node_id}_{int(time.time())}"
            logger.info(f"Registering new worker {worker.worker_id}")
            return self.worker_repository.create(worker)

    def find_worker_by_address(
        self, node_address: str, grpc_port: int
    ) -> Worker | None:
        """
        Find a worker by address and port.

        Args:
            node_address: Node address
            grpc_port: gRPC port

        Returns:
            Worker if found, None otherwise
        """
        return self.worker_repository.find_by_address_port(node_address, grpc_port)

    def heartbeat(
        self,
        worker_id: str,
        mem_pool_available_size: int,
        accepting_new_requests: bool,
    ) -> bool:
        """
        Process worker heartbeat.

        Args:
            worker_id: ID of the worker
            mem_pool_available_size: Available memory pool size
            accepting_new_requests: Whether accepting new requests

        Returns:
            True if successful, False if worker not found
        """
        # Buffer heartbeat update for batch processing
        existing = self.worker_repository.find_by_id(worker_id)
        if not existing:
            logger.warning(f"Worker {worker_id} not found for heartbeat")
            return False
        with self._heartbeat_lock:
            self._heartbeat_buffer.append(
                (worker_id, mem_pool_available_size, accepting_new_requests)
            )
            # For tests, flush immediately if buffer gets too large
            if len(self._heartbeat_buffer) >= 10:
                self._flush_heartbeats_immediate()
        return True

    def _flush_heartbeats_immediate(self) -> None:
        """Immediately flush heartbeats (for internal use)."""
        if not self._heartbeat_buffer:
            return
        updates_to_flush = self._heartbeat_buffer
        self._heartbeat_buffer = []

        try:
            count = self.worker_repository.batch_update_heartbeats(updates_to_flush)
            if count > 0:
                logger.debug(f"Immediately flushed {count} heartbeat updates")
        except Exception:
            logger.exception("Error in immediate heartbeat flush")
            # Re-queue failed updates
            self._heartbeat_buffer = updates_to_flush + self._heartbeat_buffer

    def unregister_worker(self, worker_id: str) -> bool:
        """
        Unregister a worker and mark its replicas as unavailable.

        Args:
            worker_id: ID of the worker to unregister

        Returns:
            True if successful
        """
        # Mark all replicas as unavailable
        self.replica_repository.mark_unavailable_by_worker(worker_id)

        # Delete worker
        success = self.worker_repository.delete(worker_id)

        if success:
            logger.info(f"Unregistered worker {worker_id}")
        else:
            logger.warning(f"Failed to unregister worker {worker_id}")

        return success

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
        Clean up inactive workers and their replicas.

        Returns:
            List of cleaned up worker IDs
        """
        # Delete inactive workers
        deleted = self.worker_repository.delete_inactive(
            self.config.heartbeat_timeout_ms / 1000
        )

        # Mark replicas as unavailable
        for worker_id, node_id in deleted:
            self.replica_repository.mark_unavailable_by_worker(worker_id)
            logger.info(f"Cleaned up inactive worker {worker_id} on {node_id}")

        return [worker_id for worker_id, _ in deleted]

    def flush_heartbeats(self) -> None:
        """Manually flush heartbeat buffer (for testing)."""
        self._flush_heartbeats()

    # ---------------------------------------------------------------------
    # Background batch flush for heartbeats
    # ---------------------------------------------------------------------
    def _start_heartbeat_batch_thread(self) -> None:
        """Start background thread to flush buffered heartbeats."""

        def batch_loop():
            while True:
                time.sleep(0.1)  # More frequent flushing for responsiveness
                self._flush_heartbeats()

        thread = threading.Thread(target=batch_loop, daemon=True)
        thread.start()

    def _flush_heartbeats(self) -> None:
        """Flush buffered heartbeats to the database."""
        # Atomically swap buffers to prevent losing heartbeats
        updates_to_flush = []
        with self._heartbeat_lock:
            if not self._heartbeat_buffer:
                return
            # Swap buffers atomically to avoid losing heartbeats
            updates_to_flush = self._heartbeat_buffer
            self._heartbeat_buffer = []

        # Process updates outside the lock to reduce contention
        try:
            count = self.worker_repository.batch_update_heartbeats(updates_to_flush)
            if count > 0:
                logger.debug(f"Flushed {count} heartbeat updates")
        except Exception:
            logger.exception("Error flushing heartbeat batch")
            # Re-queue failed updates for retry
            with self._heartbeat_lock:
                # Prepend failed updates to retry them
                self._heartbeat_buffer = updates_to_flush + self._heartbeat_buffer
