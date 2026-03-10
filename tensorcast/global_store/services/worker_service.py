#  Copyright (c) 2025-2026, TensorCast Team.

"""Service for worker operations."""

import threading
import time
import uuid
from dataclasses import dataclass

from tensorcast.global_store.config import get_config
from tensorcast.global_store.exceptions import ValidationError
from tensorcast.global_store.metrics import (
    inc_worker_heartbeat_flush_updates,
    observe_worker_heartbeat_flush_batch_size,
    set_worker_heartbeat_buffer_pending,
)
from tensorcast.global_store.models import Worker
from tensorcast.global_store.repositories import ReplicaRepository, WorkerRepository
from tensorcast.global_store.services.address_validation import (
    ensure_routable_address,
)
from tensorcast.global_store.services.worker_control_reducer import WorkerControlReducer
from tensorcast.logger import init_logger

logger = init_logger(__name__)


@dataclass
class _PendingHeartbeat:
    mem_pool_available_size: int
    accepting_new_requests: bool
    capability_flags: int | None
    observed_monotonic_s: float


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

        configured_timeout_s = max(
            1.0,
            float(self.config.heartbeat_timeout_ms) / 1000.0,
        )
        # Keep flush cadence tight so persisted liveness stays close to real time.
        self._heartbeat_flush_interval_s = min(
            0.2, max(0.05, configured_timeout_s / 40.0)
        )
        # Add a bounded grace window for timeout checks to tolerate buffered flush.
        self._heartbeat_timeout_grace_s = max(
            0.5, self._heartbeat_flush_interval_s * 4.0
        )
        self._heartbeat_verify_interval_s = min(
            5.0, max(0.5, configured_timeout_s / 6.0)
        )
        self._heartbeat_flush_max_batch_size = 512

        self._heartbeat_pending_mu = threading.Lock()
        self._heartbeat_pending: dict[str, _PendingHeartbeat] = {}

        self._known_workers_mu = threading.Lock()
        self._known_workers_last_verified: dict[str, float] = {}

        self._heartbeat_flush_event = threading.Event()
        self._heartbeat_stop_event = threading.Event()
        self._close_mu = threading.Lock()
        self._closed = False

        set_worker_heartbeat_buffer_pending(pending_workers=0)
        self._heartbeat_flush_thread = threading.Thread(
            target=self._heartbeat_flush_loop,
            name="gs-worker-heartbeat-flush",
            daemon=True,
        )
        self._heartbeat_flush_thread.start()
        logger.info(
            "WorkerService heartbeat flush enabled interval=%.3fs timeout_grace=%.3fs verify_interval=%.3fs",
            self._heartbeat_flush_interval_s,
            self._heartbeat_timeout_grace_s,
            self._heartbeat_verify_interval_s,
        )

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

    def _mark_worker_known_active(self, worker_id: str) -> None:
        with self._known_workers_mu:
            self._known_workers_last_verified[worker_id] = time.monotonic()

    def _is_worker_active(self, worker_id: str) -> bool:
        now = time.monotonic()
        with self._known_workers_mu:
            last_verified = self._known_workers_last_verified.get(worker_id)
            if (
                last_verified is not None
                and now - last_verified <= self._heartbeat_verify_interval_s
            ):
                return True

        worker = self.worker_repository.find_by_id(worker_id, include_inactive=False)
        if worker is None:
            with self._known_workers_mu:
                self._known_workers_last_verified.pop(worker_id, None)
            return False

        with self._known_workers_mu:
            self._known_workers_last_verified[worker_id] = now
        return True

    def _clear_worker_heartbeat_state(self, worker_id: str) -> None:
        with self._heartbeat_pending_mu:
            self._heartbeat_pending.pop(worker_id, None)
            pending_workers = len(self._heartbeat_pending)
        set_worker_heartbeat_buffer_pending(pending_workers=pending_workers)
        with self._known_workers_mu:
            self._known_workers_last_verified.pop(worker_id, None)

    def _enqueue_heartbeat(
        self,
        *,
        worker_id: str,
        mem_pool_available_size: int,
        accepting_new_requests: bool,
        capability_flags: int | None,
    ) -> None:
        now = time.monotonic()
        with self._heartbeat_pending_mu:
            self._heartbeat_pending[worker_id] = _PendingHeartbeat(
                mem_pool_available_size=mem_pool_available_size,
                accepting_new_requests=accepting_new_requests,
                capability_flags=capability_flags,
                observed_monotonic_s=now,
            )
            pending_workers = len(self._heartbeat_pending)

        set_worker_heartbeat_buffer_pending(pending_workers=pending_workers)
        if pending_workers >= self._heartbeat_flush_max_batch_size:
            self._heartbeat_flush_event.set()

    def _drain_pending_heartbeats(
        self, *, max_items: int
    ) -> list[tuple[str, _PendingHeartbeat]]:
        with self._heartbeat_pending_mu:
            if not self._heartbeat_pending:
                return []
            worker_ids = list(self._heartbeat_pending.keys())[: max(1, int(max_items))]
            drained = [
                (worker_id, self._heartbeat_pending.pop(worker_id))
                for worker_id in worker_ids
            ]
            pending_workers = len(self._heartbeat_pending)
        set_worker_heartbeat_buffer_pending(pending_workers=pending_workers)
        return drained

    def _flush_pending_batch(
        self, updates: list[tuple[str, _PendingHeartbeat]]
    ) -> None:
        if not updates:
            return

        observe_worker_heartbeat_flush_batch_size(batch_size=len(updates))
        batch_updates = [
            (
                worker_id,
                hb.mem_pool_available_size,
                hb.accepting_new_requests,
                hb.capability_flags,
            )
            for worker_id, hb in updates
        ]

        updated_count = self.worker_repository.batch_update_heartbeats(batch_updates)
        if updated_count > 0:
            inc_worker_heartbeat_flush_updates(result="batched", count=updated_count)
            for worker_id, _ in updates:
                self._mark_worker_known_active(worker_id)

        if updated_count == len(batch_updates):
            return

        # On full batch failure, fallback to per-worker writes to avoid dropping
        # all liveness signals due to transient write conflicts.
        if updated_count == 0:
            fallback_success = 0
            for worker_id, hb in updates:
                success = self.worker_repository.update_heartbeat(
                    worker_id=worker_id,
                    mem_pool_available_size=hb.mem_pool_available_size,
                    accepting_new_requests=hb.accepting_new_requests,
                    capability_flags=hb.capability_flags,
                )
                if success:
                    fallback_success += 1
                    self._mark_worker_known_active(worker_id)
                else:
                    self._clear_worker_heartbeat_state(worker_id)
            if fallback_success > 0:
                inc_worker_heartbeat_flush_updates(
                    result="fallback",
                    count=fallback_success,
                )
            dropped = len(batch_updates) - fallback_success
            if dropped > 0:
                inc_worker_heartbeat_flush_updates(result="dropped", count=dropped)
            return

        # Partial writes mean some workers became inactive between enqueue/flush.
        # Re-verify on next heartbeat by expiring known-worker cache entries.
        with self._known_workers_mu:
            for worker_id, _ in updates:
                self._known_workers_last_verified.pop(worker_id, None)
        dropped = len(batch_updates) - updated_count
        if dropped > 0:
            inc_worker_heartbeat_flush_updates(result="dropped", count=dropped)

    def _heartbeat_flush_loop(self) -> None:
        while not self._heartbeat_stop_event.is_set():
            self._heartbeat_flush_event.wait(timeout=self._heartbeat_flush_interval_s)
            self._heartbeat_flush_event.clear()
            while True:
                batch = self._drain_pending_heartbeats(
                    max_items=self._heartbeat_flush_max_batch_size
                )
                if not batch:
                    break
                self._flush_pending_batch(batch)
                if len(batch) < self._heartbeat_flush_max_batch_size:
                    break

    def close(self) -> None:
        """Stop background heartbeat flush loop and flush pending updates."""
        with self._close_mu:
            if self._closed:
                return
            self._closed = True
        self.flush_heartbeats()
        self._heartbeat_stop_event.set()
        self._heartbeat_flush_event.set()
        self._heartbeat_flush_thread.join(timeout=1.0)

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
            def _reclaim_endpoint_row(conflict: Worker, *, reason: str) -> None:
                logger.warning(
                    "register_worker endpoint takeover reason=%s "
                    "incoming_daemon_id=%s incoming_endpoint=%s:%s "
                    "replacing_worker_id=%s replacing_daemon_id=%s active=%s",
                    reason,
                    daemon_id,
                    worker.node_address,
                    int(worker.grpc_port),
                    conflict.worker_id,
                    (conflict.daemon_id or ""),
                    bool(conflict.inactive_at is None),
                )
                self.replica_repository.mark_unavailable_by_worker(conflict.worker_id)
                self.worker_repository.delete(conflict.worker_id)
                self._clear_worker_heartbeat_state(conflict.worker_id)

            # Prefer stable daemon_id identity; address/port are routing attributes.
            existing = self.worker_repository.find_by_daemon_id(
                daemon_id, include_inactive=True
            )
            by_addr = self.worker_repository.find_by_address_port(
                worker.node_address, worker.grpc_port, include_inactive=True
            )

            if existing:
                # Reclaim endpoint rows so daemon_id rebinding can proceed even
                # when the previous owner has not timed out yet.
                if by_addr and by_addr.worker_id != existing.worker_id:
                    _reclaim_endpoint_row(
                        by_addr,
                        reason="daemon_rebind",
                    )
                    by_addr = None

                reset_state_tracking = bool(existing.inactive_at is not None)
                if reset_state_tracking:
                    self.replica_repository.mark_unavailable_by_worker(
                        existing.worker_id
                    )
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
                _reclaim_endpoint_row(by_addr, reason="endpoint_takeover")
                by_addr = None

            # Create new worker
            worker.worker_id = self._generate_worker_id(worker.node_id)
            worker.daemon_id = daemon_id
            logger.debug(f"Registering new worker {worker.worker_id}")
            return self.worker_repository.create(worker)

        registered = self._submit_control_intent(
            worker_key=self._daemon_worker_key(daemon_id),
            kind="register",
            operation=_register,
        )
        self._mark_worker_known_active(registered.worker_id)
        return registered

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
        del daemon_id
        normalized_capability_flags = (
            int(capability_flags) if capability_flags is not None else None
        )
        if not self._is_worker_active(worker_id):
            logger.warning(f"Worker {worker_id} not found for heartbeat")
            self._clear_worker_heartbeat_state(worker_id)
            return False
        self._enqueue_heartbeat(
            worker_id=worker_id,
            mem_pool_available_size=mem_pool_available_size,
            accepting_new_requests=accepting_new_requests,
            capability_flags=normalized_capability_flags,
        )
        return True

    def unregister_worker(self, worker_id: str) -> bool:
        """
        Unregister a worker and mark its replicas as unavailable.

        Args:
            worker_id: ID of the worker to unregister

        Returns:
            True if successful
        """
        self.flush_heartbeats()

        def _unregister() -> bool:
            # Mark all replicas as unavailable
            self.replica_repository.mark_unavailable_by_worker(worker_id)

            # Mark worker inactive
            success = self.worker_repository.mark_inactive(worker_id)
            if success:
                logger.debug(f"Unregistered worker {worker_id}")
                self._clear_worker_heartbeat_state(worker_id)
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
        self.flush_heartbeats()
        return self.worker_repository.find_active(include_unavailable)

    def cleanup_inactive_workers(self) -> list[str]:
        """
        Mark inactive workers and clean up their replicas.

        Returns:
            List of cleaned up worker IDs
        """
        self.flush_heartbeats()
        timeout_seconds = (
            float(self.config.heartbeat_timeout_ms) / 1000.0
        ) + self._heartbeat_timeout_grace_s
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
                self._clear_worker_heartbeat_state(target_worker_id)
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
        """Flush buffered heartbeat updates to durable storage."""
        while True:
            batch = self._drain_pending_heartbeats(
                max_items=self._heartbeat_flush_max_batch_size
            )
            if not batch:
                return
            self._flush_pending_batch(batch)
