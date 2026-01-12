#  Copyright (c) 2025-2026, TensorCast Team.

"""
Recovery service for Global Store high availability.

Handles state recovery after failures, worker rediscovery, and state synchronization.
"""

import time
from uuid import UUID, uuid4

from tensorcast.global_store.exceptions import NotFoundError
from tensorcast.global_store.metrics import observe_state_sync
from tensorcast.global_store.models import Replica, Worker
from tensorcast.global_store.repositories import (
    ReplicaRepository,
    WorkerRepository,
)
from tensorcast.logger import init_logger
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.global_store.v1 import global_store_pb2

logger = init_logger(__name__)

ReplicaKey = tuple[str, str, str, int]


class RecoveryService:
    """Service for handling Global Store recovery and state synchronization."""

    def __init__(
        self,
        worker_repository: WorkerRepository,
        replica_repository: ReplicaRepository,
    ):
        self.worker_repository = worker_repository
        self.replica_repository = replica_repository

        # Recovery state tracking
        self.recovery_in_progress = False
        self.last_recovery_time = 0

    def initiate_recovery(self) -> bool:
        """
        Initiate Global Store recovery after restart.

        Returns:
            True if recovery was successful, False otherwise
        """
        if self.recovery_in_progress:
            logger.warning("Recovery already in progress")
            return False

        try:
            self.recovery_in_progress = True
            logger.info("Starting Global Store recovery")

            # Step 1: Validate database state
            self._validate_database_state()

            # Step 2: Mark all workers as potentially stale
            self._mark_workers_as_stale()

            # Step 3: Mark all replicas as potentially unavailable
            self._mark_replicas_as_stale()

            self.last_recovery_time = int(time.time())
            logger.info("Global Store recovery completed successfully")
            return True

        except Exception as e:
            logger.exception(f"Recovery failed: {e}")
            return False
        finally:
            self.recovery_in_progress = False

    def _validate_database_state(self) -> None:
        """Validate database integrity and clean up inconsistent state."""
        # Check for orphaned replicas (replicas without valid workers)
        orphaned_replicas = self.replica_repository.find_orphaned_replicas()
        if orphaned_replicas:
            logger.warning(
                f"Found {len(orphaned_replicas)} orphaned replicas, marking as unavailable"
            )
            for replica in orphaned_replicas:
                self.replica_repository.mark_unavailable(replica.replica_id)

    def _mark_workers_as_stale(self) -> None:
        """Mark all workers as potentially stale until they re-register or heartbeat."""
        workers = self.worker_repository.list_all_workers()
        logger.info(f"Marking {len(workers)} workers as stale")

        for worker in workers:
            self.worker_repository.mark_as_stale(worker.worker_id)

    def _mark_replicas_as_stale(self) -> None:
        """Mark all replicas as potentially stale until confirmed by workers."""
        replicas = self.replica_repository.list_all_replicas()
        logger.info(f"Marking {len(replicas)} replicas as stale")

        for replica in replicas:
            self.replica_repository.mark_as_stale(replica.replica_id)

    def handle_worker_recovery_registration(
        self, worker: Worker, previous_worker_id: str | None = None
    ) -> tuple[bool, bool]:
        """
        Handle worker registration during recovery.

        Args:
            worker: New worker registration
            previous_worker_id: Previous worker ID if this is a recovery

        Returns:
            Tuple of (registration_success, state_sync_required)
        """
        try:
            # If previous worker ID provided, set worker_id and clean up old state
            if previous_worker_id:
                worker.worker_id = previous_worker_id
                self._cleanup_previous_worker_state(
                    previous_worker_id, worker.worker_id
                )

            # Register new worker
            registered_worker = self.worker_repository.create_or_update(worker)

            # Ensure persisted state version is initialized
            self.worker_repository.ensure_state_version(registered_worker.worker_id)

            # Always require state sync during recovery
            state_sync_required = True

            logger.info(
                f"Worker recovery registration successful: {registered_worker.worker_id}"
                f"{' (replaced ' + previous_worker_id + ')' if previous_worker_id else ''}"
            )

            return True, state_sync_required

        except Exception as e:
            logger.exception(f"Worker recovery registration failed: {e}")
            return False, False

    def _cleanup_previous_worker_state(
        self, previous_worker_id: str, new_worker_id: str
    ) -> None:
        """Clean up state from previous worker instance."""
        logger.info(f"Cleaning up state for previous worker {previous_worker_id}")

        # Mark old worker as inactive
        try:
            self.worker_repository.mark_inactive(previous_worker_id)
        except NotFoundError:
            logger.warning(
                f"Previous worker {previous_worker_id} not found in database"
            )

        # Transfer or mark replicas as unavailable
        replicas = self.replica_repository.get_replicas_by_worker(previous_worker_id)
        for replica in replicas:
            # Update worker_id to new one if it's the same physical node
            try:
                self.replica_repository.update_worker_id(
                    replica.replica_id, new_worker_id
                )
                logger.debug(
                    f"Transferred replica {replica.replica_id} to new worker {new_worker_id}"
                )
            except Exception as e:
                logger.warning(f"Failed to transfer replica {replica.replica_id}: {e}")
                self.replica_repository.mark_unavailable(replica.replica_id)

    def synchronize_worker_state(
        self,
        worker_id: str,
        local_state: global_store_pb2.WorkerLocalState,
        sync_epoch: int,
        sync_request_id: int,
        force_full_sync: bool = False,
    ) -> tuple[bool, list[global_store_pb2.StateChange], int, str, bool]:
        """
        Synchronize worker state with global state.

        Args:
            worker_id: Worker ID
            local_state: Worker's local state
            sync_epoch: Monotonic sync epoch
            sync_request_id: Monotonic sync request id
            force_full_sync: When True, treat the provided inventory as
                authoritative even if empty (allows drains/retirements).

        Returns:
            Tuple of (success, state_changes, new_version, new_checksum, ignored)
        """
        _start = time.time()
        try:
            with self.worker_repository.transaction() as cursor:
                accepted = self.worker_repository.try_advance_state_sync_token(
                    worker_id, sync_epoch, sync_request_id, cursor
                )
                if not accepted:
                    current_version = self.worker_repository.ensure_state_version(
                        worker_id, cursor
                    )
                    current_checksum = self.worker_repository.get_state_checksum(
                        worker_id, cursor
                    )
                    duration = time.time() - _start
                    observe_state_sync(duration, success=True)
                    logger.info(
                        "Ignored stale state sync for worker %s: epoch=%d request_id=%d",
                        worker_id,
                        sync_epoch,
                        sync_request_id,
                    )
                    return True, [], current_version, current_checksum, True

                # Get current global state for this worker
                global_replicas = self.replica_repository.get_replicas_by_worker_atomic(
                    worker_id, cursor
                )

                # Compare states and generate changes
                state_changes = self._compute_state_changes(
                    local_state, global_replicas, force_full_sync
                )

                current_version = self.worker_repository.ensure_state_version(
                    worker_id, cursor
                )

                if state_changes:
                    # Apply state changes and bump version
                    self._apply_state_changes(worker_id, state_changes, cursor)

                    updated_replicas = (
                        self.replica_repository.get_replicas_by_worker_atomic(
                            worker_id, cursor
                        )
                    )
                    new_checksum = self._compute_state_checksum(updated_replicas)
                    new_version = current_version + 1
                    self.worker_repository.update_state_version_and_checksum(
                        worker_id, new_version, new_checksum, cursor
                    )
                else:
                    # No-op sync: do not change version; reconcile checksum in-place.
                    new_version = current_version
                    new_checksum = self._compute_state_checksum(global_replicas)
                    stored_checksum = self.worker_repository.get_state_checksum(
                        worker_id, cursor
                    )
                    if stored_checksum != new_checksum:
                        self.worker_repository.update_state_checksum(
                            worker_id, new_checksum, cursor
                        )

            duration = time.time() - _start
            observe_state_sync(duration, success=True)

            if state_changes:
                logger.info(
                    f"State synchronization completed for worker {worker_id}: "
                    f"{len(state_changes)} changes, version {new_version}"
                )
            else:
                logger.info(
                    f"State synchronization no-op for worker {worker_id}: "
                    f"0 changes, version unchanged ({new_version})"
                )

            return True, state_changes, new_version, new_checksum, False

        except Exception as e:
            duration = time.time() - _start if "_start" in locals() else 0.0
            observe_state_sync(duration, success=False)

            logger.exception(
                f"State synchronization failed for worker {worker_id}: {e}"
            )
            return False, [], 0, "", False

    def _compute_state_changes(
        self,
        local_state: global_store_pb2.WorkerLocalState,
        global_replicas: list[Replica],
        force_full_sync: bool,
    ) -> list[global_store_pb2.StateChange]:
        """Compute differences between local and global state."""
        state_changes = []

        def _memory_type_label(mem_type: common_pb2.MemoryType) -> str:
            if mem_type == common_pb2.MemoryType.MEMORY_TYPE_GPU:
                return "GPU"
            if mem_type == common_pb2.MemoryType.MEMORY_TYPE_RAM:
                return "RAM"
            return "DISK"

        # Convert to maps for comparison and fast lookup
        local_replicas_by_key: dict[ReplicaKey, common_pb2.ReplicaInfo] = {
            (
                r.ref.artifact_id,
                r.memory_info.node_id,
                _memory_type_label(r.memory_info.memory_type),
                r.memory_info.device_id,
            ): r
            for r in local_state.local_replicas
        }

        global_replicas_by_key: dict[ReplicaKey, Replica] = {
            (r.artifact_id, r.node_id, r.memory_type.value, r.device_id): r
            for r in global_replicas
        }

        local_replica_keys: set[ReplicaKey] = set(local_replicas_by_key.keys())
        global_replica_keys: set[ReplicaKey] = set(global_replicas_by_key.keys())

        # Find replicas to add (in local but not in global)
        to_add = local_replica_keys - global_replica_keys
        for key in to_add:
            local_replica = local_replicas_by_key[key]

            change = global_store_pb2.StateChange(
                type=global_store_pb2.StateChange.CHANGE_TYPE_ADD_REPLICA,
                replica_info=local_replica,
                reason="Local replica not found in global state",
            )
            state_changes.append(change)

        # Safe-removal semantics --------------------------------------------------
        # Default: empty inventory suppresses removals (treat as unknown).
        # When force_full_sync is True, empty inventory is authoritative and will
        # drive removals (used for drains / explicit full reconciliation).
        to_remove: set[ReplicaKey]
        if local_state.local_replicas:
            to_remove = global_replica_keys - local_replica_keys
        elif force_full_sync:
            to_remove = global_replica_keys
        else:
            to_remove = set()

        for key in to_remove:
            global_replica = global_replicas_by_key[key]

            # Convert to proto format
            replica_info = self._convert_replica_to_proto(global_replica)

            change = global_store_pb2.StateChange(
                type=global_store_pb2.StateChange.CHANGE_TYPE_REMOVE_REPLICA,
                replica_info=replica_info,
                reason="Global replica not found in local state",
            )
            state_changes.append(change)

        # Identify replicas that exist in both sets but have drifted metadata.
        shared_keys = local_replica_keys & global_replica_keys
        for key in shared_keys:
            local_replica = local_replicas_by_key[key]
            global_replica = global_replicas_by_key[key]

            desired_available = local_replica.stats.is_available
            current_available = global_replica.is_available

            availability_changed = desired_available != current_available
            node_address_changed = False
            node_port_changed = False
            memory_size_changed = False
            remote_keys_changed = False
            buffer_sizes_changed = False

            local_node_address = local_replica.memory_info.node_address
            local_node_port = local_replica.memory_info.node_port
            local_memory_size = local_replica.memory_info.memory_size
            local_remote_keys = list(local_replica.memory_info.remote_memory_keys)
            local_buffer_sizes = list(local_replica.memory_info.buffer_sizes)

            if local_node_address and local_node_address != global_replica.node_address:
                node_address_changed = True
            if local_node_port > 0 and local_node_port != global_replica.node_port:
                node_port_changed = True
            if (
                local_memory_size > 0
                and local_memory_size != global_replica.memory_size
            ):
                memory_size_changed = True
            if local_remote_keys:
                remote_keys_changed = local_remote_keys != list(
                    global_replica.remote_memory_keys
                )
            if local_buffer_sizes:
                buffer_sizes_changed = local_buffer_sizes != list(
                    global_replica.buffer_sizes
                )

            if not any(
                [
                    availability_changed,
                    node_address_changed,
                    node_port_changed,
                    memory_size_changed,
                    remote_keys_changed,
                    buffer_sizes_changed,
                ]
            ):
                continue

            updated_proto = self._convert_replica_to_proto(global_replica)
            reasons: list[str] = []
            if availability_changed:
                updated_proto.stats.is_available = desired_available
                reasons.append("availability")
            if node_address_changed:
                updated_proto.memory_info.node_address = local_node_address
                reasons.append("node_address")
            if node_port_changed:
                updated_proto.memory_info.node_port = local_node_port
                reasons.append("node_port")
            if memory_size_changed:
                updated_proto.memory_info.memory_size = local_memory_size
                reasons.append("memory_size")
            if remote_keys_changed:
                updated_proto.memory_info.remote_memory_keys[:] = local_remote_keys
                reasons.append("remote_memory_keys")
            if buffer_sizes_changed:
                updated_proto.memory_info.buffer_sizes[:] = local_buffer_sizes
                reasons.append("buffer_sizes")

            change = global_store_pb2.StateChange(
                type=global_store_pb2.StateChange.CHANGE_TYPE_UPDATE_REPLICA,
                replica_info=updated_proto,
                reason="Replica metadata changed (" + ", ".join(reasons) + ")",
            )
            state_changes.append(change)

        return state_changes

    def _apply_state_changes(
        self,
        worker_id: str,
        state_changes: list[global_store_pb2.StateChange],
        cursor,
    ) -> None:
        """Apply state changes to global state atomically."""
        for change in state_changes:
            if change.type == global_store_pb2.StateChange.CHANGE_TYPE_ADD_REPLICA:
                # Register new replica
                replica = self._convert_proto_to_replica(change.replica_info, worker_id)
                self.replica_repository.create_or_update_atomic(replica, cursor)
                logger.debug(f"Added replica: {replica.artifact_id}")

            elif change.type == global_store_pb2.StateChange.CHANGE_TYPE_REMOVE_REPLICA:
                # Remove replica
                replica_id_value = change.replica_info.ref.replica_id
                if not replica_id_value:
                    raise ValueError(
                        f"Missing replica_id for removal (artifact_id={change.replica_info.ref.artifact_id})"
                    )
                replica_id = UUID(replica_id_value)
                deleted = self.replica_repository.delete_atomic(replica_id, cursor)
                if not deleted:
                    raise NotFoundError(f"Replica {replica_id} not found")
                logger.debug(f"Removed replica: {change.replica_info.ref.artifact_id}")

            elif change.type == global_store_pb2.StateChange.CHANGE_TYPE_UPDATE_REPLICA:
                # Update replica
                replica = self._convert_proto_to_replica(change.replica_info, worker_id)
                self.replica_repository.update_atomic(replica, cursor)
                logger.debug(f"Updated replica: {replica.artifact_id}")

    def _convert_replica_to_proto(self, replica: Replica) -> common_pb2.ReplicaInfo:
        """Convert Replica to proto format."""
        # Map domain MemoryType to proto MemoryType
        if replica.memory_type.value == "GPU":
            proto_mem_type = common_pb2.MemoryType.MEMORY_TYPE_GPU
        elif replica.memory_type.value == "RAM":
            proto_mem_type = common_pb2.MemoryType.MEMORY_TYPE_RAM
        else:
            proto_mem_type = common_pb2.MemoryType.MEMORY_TYPE_DISK

        memory_info = common_pb2.MemoryInfo(
            node_id=replica.node_id,
            node_address=replica.node_address,
            node_port=replica.node_port,
            memory_size=replica.memory_size,
            memory_type=proto_mem_type,
            device_id=replica.device_id,
            remote_memory_keys=replica.remote_memory_keys,
            buffer_sizes=replica.buffer_sizes,
        )
        stats = common_pb2.ReplicaStats(
            max_concurrency=replica.max_concurrency,
            current_requests=replica.current_requests,
            is_available=replica.is_available,
        )
        if replica.created_at:
            from google.protobuf import timestamp_pb2

            ts = timestamp_pb2.Timestamp()
            ts.FromSeconds(int(replica.created_at.timestamp()))
            stats.registered_ts.CopyFrom(ts)

        return common_pb2.ReplicaInfo(
            ref=common_pb2.ReplicaRef(
                artifact_id=replica.artifact_id,
                replica_id=str(replica.replica_id),
            ),
            memory_info=memory_info,
            stats=stats,
        )

    def _convert_proto_to_replica(
        self, proto_replica: common_pb2.ReplicaInfo, worker_id: str
    ) -> Replica:
        """Convert proto format to Replica."""
        from tensorcast.global_store.models import MemoryType

        # Map proto MemoryType to domain MemoryType string enum
        if (
            proto_replica.memory_info.memory_type
            == common_pb2.MemoryType.MEMORY_TYPE_GPU
        ):
            dom_mem_type = "GPU"
        elif (
            proto_replica.memory_info.memory_type
            == common_pb2.MemoryType.MEMORY_TYPE_RAM
        ):
            dom_mem_type = "RAM"
        else:
            dom_mem_type = "DISK"

        return Replica(
            replica_id=UUID(proto_replica.ref.replica_id)
            if proto_replica.ref.replica_id
            else uuid4(),
            artifact_id=proto_replica.ref.artifact_id,
            node_id=proto_replica.memory_info.node_id,
            node_address=proto_replica.memory_info.node_address,
            node_port=proto_replica.memory_info.node_port,
            memory_size=proto_replica.memory_info.memory_size,
            memory_type=MemoryType(dom_mem_type),
            device_id=proto_replica.memory_info.device_id,
            max_concurrency=proto_replica.stats.max_concurrency,
            current_requests=proto_replica.stats.current_requests,
            is_available=proto_replica.stats.is_available,
            remote_memory_keys=list(proto_replica.memory_info.remote_memory_keys),
            buffer_sizes=list(proto_replica.memory_info.buffer_sizes),
            worker_id=worker_id,
        )

    def _compute_state_checksum(self, replicas: list[Replica]) -> str:
        """Compute checksum of replica state for consistency checking."""
        entries: list[tuple[str, str, str, int, int, str, bool]] = []
        for replica in replicas:
            mem_type = "DISK"
            device_id = 0
            if replica.memory_type.value == "GPU":
                mem_type = "GPU"
                device_id = replica.device_id
            elif replica.memory_type.value == "RAM":
                mem_type = "RAM"
                device_id = 0
            entries.append(
                (
                    replica.artifact_id,
                    replica.node_id or "",
                    replica.node_address or "",
                    replica.node_port or 0,
                    device_id,
                    mem_type,
                    replica.is_available,
                )
            )

        # Sort replicas by a stable key for consistent checksum
        entries.sort(key=lambda e: (e[0], e[5], e[4]))

        # Create string representation of state
        state_parts = [
            f"{artifact_id}:{node_id}:{node_address}:{node_port}:{device_id}:{memory_type}:{1 if available else 0};"
            for artifact_id, node_id, node_address, node_port, device_id, memory_type, available in entries
        ]
        state_str = "".join(state_parts)

        # Compute FNV-1a 64-bit hash for portability with C++ daemon
        hash_val = 0xCBF29CE484222325
        fnv_prime = 0x100000001B3
        for b in state_str.encode():
            hash_val ^= b
            hash_val = (hash_val * fnv_prime) & 0xFFFFFFFFFFFFFFFF

        return f"{hash_val:016x}"

    def ensure_worker_state_version(self, worker_id: str) -> int:
        """Ensure the worker has a non-zero state version."""
        return self.worker_repository.ensure_state_version(worker_id)

    def get_worker_state_version(self, worker_id: str) -> int:
        """Get current state version for a worker."""
        return self.worker_repository.get_state_version(worker_id)

    def request_full_state_sync(
        self, worker_id: str, sync_epoch: int, sync_request_id: int
    ) -> tuple[bool, list[common_pb2.ReplicaInfo], int, str, bool]:
        """
        Request full state synchronization for a worker.

        Returns:
            Tuple of (success, expected_replicas, new_version, new_checksum, ignored)
        """
        _start = time.time()
        try:
            with self.worker_repository.transaction() as cursor:
                accepted = self.worker_repository.try_advance_state_sync_token(
                    worker_id, sync_epoch, sync_request_id, cursor
                )
                if not accepted:
                    current_version = self.worker_repository.ensure_state_version(
                        worker_id, cursor
                    )
                    current_checksum = self.worker_repository.get_state_checksum(
                        worker_id, cursor
                    )
                    duration = time.time() - _start
                    observe_state_sync(duration, success=True)
                    logger.info(
                        "Ignored stale full-state sync for worker %s: epoch=%d request_id=%d",
                        worker_id,
                        sync_epoch,
                        sync_request_id,
                    )
                    return True, [], current_version, current_checksum, True

                # Get all replicas for this worker
                replicas = self.replica_repository.get_replicas_by_worker_atomic(
                    worker_id, cursor
                )

                # Convert to proto format
                proto_replicas = [
                    self._convert_replica_to_proto(replica) for replica in replicas
                ]

                # Full-state sync is informational; do NOT bump version.
                current_version = self.worker_repository.ensure_state_version(
                    worker_id, cursor
                )

                # Recompute checksum for current global state and persist if stale.
                new_checksum = self._compute_state_checksum(replicas)
                stored_checksum = self.worker_repository.get_state_checksum(
                    worker_id, cursor
                )
                if stored_checksum != new_checksum:
                    self.worker_repository.update_state_checksum(
                        worker_id, new_checksum, cursor
                    )

            duration = time.time() - _start
            observe_state_sync(duration, success=True)

            logger.info(
                f"Full state sync requested for worker {worker_id}: {len(replicas)} replicas"
            )

            return True, proto_replicas, current_version, new_checksum, False

        except Exception as e:
            duration = time.time() - _start if "_start" in locals() else 0.0
            observe_state_sync(duration, success=False)

            logger.exception(f"Full state sync failed for worker {worker_id}: {e}")
            return False, [], 0, "", False

    def is_recovery_complete(self) -> bool:
        """Check if recovery process is complete."""
        return not self.recovery_in_progress and self.last_recovery_time > 0

    def get_worker_state_checksum(self, worker_id: str) -> str:
        """Get checksum for the given worker's replica state."""
        checksum = self.worker_repository.get_state_checksum(worker_id)
        if checksum:
            return checksum

        replicas = self.replica_repository.get_replicas_by_worker(worker_id)
        checksum = self._compute_state_checksum(replicas)
        self.worker_repository.update_state_checksum(worker_id, checksum)
        return checksum

    def get_obsolete_artifacts(
        self, worker_id: str, registered_artifact_ids: list[str] | tuple[str, ...]
    ) -> list[str]:
        """Determine artifacts that exist on the worker but not in the global state.

        Args:
            worker_id: Worker identifier.
            registered_artifact_ids: Artifact IDs currently reported by the worker.

        Returns:
            List of artifact IDs that should be removed from the worker.
        """
        try:
            replicas = self.replica_repository.get_replicas_by_worker(worker_id)
            global_artifacts = {replica.artifact_id for replica in replicas}
            return [a for a in registered_artifact_ids if a not in global_artifacts]
        except Exception as e:
            logger.exception(
                f"Failed to compute obsolete artifacts for {worker_id}: {e}"
            )
            return []
