#  Copyright (c) 2025-2026, TensorCast Team.

"""
Recovery service for Global Store high availability.

Handles state recovery after failures, worker rediscovery, and state synchronization.
"""

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

from tensorcast.global_store.exceptions import (
    DatabaseError,
    NotFoundError,
    ValidationError,
)
from tensorcast.global_store.grpc_helpers import (
    coerce_db_datetime,
    timestamp_to_datetime,
)
from tensorcast.global_store.metrics import (
    inc_control_plane_conflict,
    inc_reconcile_result,
    inc_reconcile_retry_later,
    observe_state_sync,
)
from tensorcast.global_store.models import (
    Replica,
    Worker,
)
from tensorcast.global_store.replica_memory_codec import (
    export_state_to_proto,
    memory_info_to_replica,
    parse_transport_metadata,
    replica_to_memory_info,
)
from tensorcast.global_store.repositories import ReplicaRepository, WorkerRepository
from tensorcast.global_store.services.worker_control_reducer import WorkerControlReducer
from tensorcast.global_store.services.worker_service import WorkerService
from tensorcast.logger import init_logger
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.global_store.v1 import global_store_pb2

logger = init_logger(__name__)

ReplicaKey = tuple[str, str, str, str, int]


@dataclass(frozen=True)
class _ReplicaEpoch:
    replica_count: int
    max_updated_epoch_us: int


@dataclass(frozen=True)
class _ReconcileNoopCacheEntry:
    generation: int
    request_seq: int
    state_version: int
    state_checksum: str
    inventory_fingerprint: str
    replica_epoch: _ReplicaEpoch


class RecoveryService:
    """Service for handling Global Store recovery and state synchronization."""

    def __init__(
        self,
        worker_repository: WorkerRepository,
        replica_repository: ReplicaRepository,
        worker_service: WorkerService,
        control_reducer: WorkerControlReducer | None = None,
    ):
        self.worker_repository = worker_repository
        self.replica_repository = replica_repository
        self.worker_service = worker_service
        self._control_reducer = control_reducer

        # Recovery state tracking
        self.recovery_in_progress = False
        self.last_recovery_time = 0
        self._reconcile_noop_cache_mu = threading.Lock()
        self._reconcile_noop_cache: dict[str, _ReconcileNoopCacheEntry] = {}

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
            daemon_id = (worker.daemon_id or "").strip()
            if not daemon_id:
                raise ValidationError("daemon_id is required")

            if previous_worker_id:
                previous = self.worker_repository.find_by_id(
                    previous_worker_id, include_inactive=True
                )
                if previous and (previous.daemon_id or "").strip() != daemon_id:
                    raise ValidationError("previous_worker_id does not match daemon_id")

            registered_worker = self.worker_service.register_worker(worker)
            if previous_worker_id and previous_worker_id != registered_worker.worker_id:
                self._cleanup_previous_worker_state(
                    previous_worker_id, registered_worker.worker_id
                )

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

    @staticmethod
    def _inventory_fingerprint(inventory: list[common_pb2.ReplicaInfo]) -> str:
        entries: list[tuple] = []
        for replica in inventory:
            mem = replica.memory_info
            view_id = (
                mem.byte_space.id
                if mem.HasField("byte_space")
                and mem.byte_space.kind == common_pb2.BYTE_SPACE_KIND_VIEW
                else ""
            )
            transport_export_state = 0
            transport_export_generation = 0
            transport_remote_keys: tuple[str, ...] = ()
            transport_buffer_sizes: tuple[int, ...] = ()
            transport_verification_json = ""
            if mem.HasField("transport"):
                transport_export_state = int(mem.transport.export_state)
                transport_export_generation = int(mem.transport.export_generation)
                transport_remote_keys = tuple(mem.transport.remote_memory_keys)
                transport_buffer_sizes = tuple(mem.transport.buffer_sizes)
                transport_verification_json = mem.transport.verification_json
            entries.append(
                (
                    replica.ref.artifact_id,
                    view_id,
                    mem.node_id,
                    mem.node_address,
                    int(mem.node_port),
                    int(mem.memory_type),
                    int(mem.device_id),
                    int(mem.memory_size),
                    bool(replica.stats.is_available),
                    tuple(mem.remote_memory_keys),
                    tuple(mem.buffer_sizes),
                    mem.verification_json,
                    transport_export_state,
                    transport_export_generation,
                    transport_remote_keys,
                    transport_buffer_sizes,
                    transport_verification_json,
                )
            )
        entries.sort()
        state_str = "\n".join(str(entry) for entry in entries)
        hash_val = 0xCBF29CE484222325
        fnv_prime = 0x100000001B3
        for b in state_str.encode():
            hash_val ^= b
            hash_val = (hash_val * fnv_prime) & 0xFFFFFFFFFFFFFFFF
        return f"{hash_val:016x}"

    @staticmethod
    def _snapshot_observed_at(
        inventory: list[common_pb2.ReplicaInfo],
    ) -> datetime | None:
        observed_at = None
        for replica in inventory:
            if not replica.stats.HasField("registered_ts"):
                continue
            replica_observed_at = timestamp_to_datetime(replica.stats.registered_ts)
            if replica_observed_at is None:
                continue
            if observed_at is None or replica_observed_at < observed_at:
                observed_at = replica_observed_at
        return observed_at

    def _read_reconcile_noop_cache(
        self, worker_id: str
    ) -> _ReconcileNoopCacheEntry | None:
        with self._reconcile_noop_cache_mu:
            return self._reconcile_noop_cache.get(worker_id)

    def _write_reconcile_noop_cache(
        self, *, worker_id: str, entry: _ReconcileNoopCacheEntry
    ) -> None:
        with self._reconcile_noop_cache_mu:
            self._reconcile_noop_cache[worker_id] = entry

    def _try_cached_replay_or_stale(
        self,
        *,
        worker_id: str,
        generation: int,
        request_seq: int,
    ) -> (
        tuple[
            global_store_pb2.ReconcileResultKind,
            int,
            str,
            list[global_store_pb2.StateChange],
            list[common_pb2.ReplicaInfo],
            int,
        ]
        | None
    ):
        cached = self._read_reconcile_noop_cache(worker_id)
        if cached is None:
            return None
        if int(generation) != int(cached.generation):
            return None
        if int(request_seq) == int(cached.request_seq):
            if (
                self.worker_repository.find_by_id(worker_id, include_inactive=False)
                is None
            ):
                with self._reconcile_noop_cache_mu:
                    self._reconcile_noop_cache.pop(worker_id, None)
                return None
            return (
                global_store_pb2.RECONCILE_RESULT_KIND_NOOP,
                int(cached.state_version),
                cached.state_checksum,
                [],
                [],
                0,
            )
        if int(request_seq) < int(cached.request_seq):
            if (
                self.worker_repository.find_by_id(worker_id, include_inactive=False)
                is None
            ):
                with self._reconcile_noop_cache_mu:
                    self._reconcile_noop_cache.pop(worker_id, None)
                return None
            inc_control_plane_conflict(scope="reconcile_request_stale")
            return (
                global_store_pb2.RECONCILE_RESULT_KIND_IGNORED_STALE,
                int(cached.state_version),
                cached.state_checksum,
                [],
                [],
                0,
            )
        return None

    def reconcile_worker_state(
        self,
        *,
        worker_id: str,
        daemon_id: str,
        generation: int,
        request_seq: int,
        inventory: list[common_pb2.ReplicaInfo],
        request_kind: global_store_pb2.ReconcileRequestKind,
    ) -> tuple[
        global_store_pb2.ReconcileResultKind,
        int,
        str,
        list[global_store_pb2.StateChange],
        list[common_pb2.ReplicaInfo],
        int,
    ]:
        """Typed worker reconcile entrypoint for V2 control plane."""
        control_worker_key = self.worker_service.resolve_control_worker_key(
            worker_id=worker_id,
            daemon_id=daemon_id,
        )
        fast_start = time.time()
        cached_terminal = self._try_cached_replay_or_stale(
            worker_id=worker_id,
            generation=generation,
            request_seq=request_seq,
        )
        if cached_terminal is not None:
            inc_reconcile_result(
                result_kind=global_store_pb2.ReconcileResultKind.Name(
                    cached_terminal[0]
                )
            )
            observe_state_sync(time.time() - fast_start, success=True)
            return cached_terminal

        def _is_transient_tx_conflict(exc: Exception) -> bool:
            message = str(exc).lower()
            conflict_markers = (
                "write-write conflict",
                "conflict on tuple deletion",
                "conflict on update",
                "serialization",
                "transactioncontext error: conflict",
            )
            if isinstance(exc, DatabaseError):
                return any(marker in message for marker in conflict_markers)
            return any(marker in message for marker in conflict_markers)

        def _run_reconcile_with_retry() -> tuple[
            global_store_pb2.ReconcileResultKind,
            int,
            str,
            list[global_store_pb2.StateChange],
            list[common_pb2.ReplicaInfo],
            int,
        ]:
            max_attempts = 5
            for attempt in range(max_attempts):
                try:
                    return self._reconcile_worker_state_internal(
                        worker_id=worker_id,
                        daemon_id=daemon_id,
                        worker_key=control_worker_key,
                        generation=generation,
                        request_seq=request_seq,
                        inventory=inventory,
                        request_kind=request_kind,
                    )
                except Exception as exc:  # noqa: BLE001
                    if not _is_transient_tx_conflict(exc):
                        raise
                    if attempt == max_attempts - 1:
                        current_version = 0
                        current_checksum = ""
                        try:
                            with self.worker_repository.transaction() as cursor:
                                current_version = (
                                    self.worker_repository.ensure_state_version(
                                        worker_id, cursor
                                    )
                                )
                                current_checksum = (
                                    self.worker_repository.get_state_checksum(
                                        worker_id, cursor
                                    )
                                )
                        except Exception:  # noqa: BLE001
                            logger.warning(
                                "Failed to snapshot worker reconcile state after "
                                "reconcile conflict exhaustion worker_id=%s",
                                worker_id,
                            )

                        logger.warning(
                            "Reconcile transaction conflict exhausted retries; "
                            "returning RETRY_LATER worker_id=%s daemon_id=%s "
                            "worker_key=%s "
                            "generation=%s request_seq=%s request_kind=%s "
                            "inventory_size=%s",
                            worker_id,
                            daemon_id,
                            control_worker_key,
                            int(generation),
                            int(request_seq),
                            int(request_kind),
                            len(inventory),
                        )
                        inc_control_plane_conflict(
                            scope="reconcile_tx_conflict_exhausted"
                        )
                        inc_reconcile_retry_later(reason="tx_conflict_exhausted")
                        inc_reconcile_result(
                            result_kind=global_store_pb2.ReconcileResultKind.Name(
                                global_store_pb2.RECONCILE_RESULT_KIND_RETRY_LATER
                            )
                        )
                        return (
                            global_store_pb2.RECONCILE_RESULT_KIND_RETRY_LATER,
                            int(current_version),
                            current_checksum,
                            [],
                            [],
                            500,
                        )
                    backoff_s = min(0.2, 0.01 * (2**attempt))
                    logger.warning(
                        "Transient reconcile transaction conflict for worker %s "
                        "(attempt %s/%s) daemon_id=%s worker_key=%s generation=%s request_seq=%s "
                        "request_kind=%s inventory_size=%s: %s",
                        worker_id,
                        attempt + 1,
                        max_attempts,
                        daemon_id,
                        control_worker_key,
                        int(generation),
                        int(request_seq),
                        int(request_kind),
                        len(inventory),
                        exc,
                    )
                    time.sleep(backoff_s)
            raise RuntimeError("unreachable")

        if self._control_reducer is not None:
            return self._control_reducer.submit(
                worker_key=control_worker_key,
                kind="reconcile",
                operation=_run_reconcile_with_retry,
            )
        return _run_reconcile_with_retry()

    def _reconcile_worker_state_internal(
        self,
        *,
        worker_id: str,
        daemon_id: str,
        worker_key: str,
        generation: int,
        request_seq: int,
        inventory: list[common_pb2.ReplicaInfo],
        request_kind: global_store_pb2.ReconcileRequestKind,
    ) -> tuple[
        global_store_pb2.ReconcileResultKind,
        int,
        str,
        list[global_store_pb2.StateChange],
        list[common_pb2.ReplicaInfo],
        int,
    ]:
        def _record_result(
            result_kind: global_store_pb2.ReconcileResultKind,
        ) -> None:
            inc_reconcile_result(
                result_kind=global_store_pb2.ReconcileResultKind.Name(result_kind)
            )

        normalized_request_kind = request_kind
        if (
            normalized_request_kind
            == global_store_pb2.RECONCILE_REQUEST_KIND_UNSPECIFIED
        ):
            normalized_request_kind = global_store_pb2.RECONCILE_REQUEST_KIND_SNAPSHOT
        inventory_fingerprint = ""
        if normalized_request_kind == global_store_pb2.RECONCILE_REQUEST_KIND_SNAPSHOT:
            inventory_fingerprint = self._inventory_fingerprint(inventory)

        _start = time.time()
        try:
            with self.worker_repository.transaction() as cursor:
                daemon_row = cursor.execute(
                    """
                    SELECT daemon_id
                    FROM workers
                    WHERE worker_id = ?
                      AND inactive_at IS NULL
                    """,
                    [worker_id],
                ).fetchone()
                if daemon_row is None:
                    raise ValueError(f"Worker {worker_id} not found")
                persisted_daemon_id = str(daemon_row[0] or "").strip()

                current_version = self.worker_repository.ensure_state_version(
                    worker_id, cursor
                )
                current_checksum = self.worker_repository.get_state_checksum(
                    worker_id, cursor
                )
                persisted_generation, persisted_request_seq = (
                    self.worker_repository.get_reconcile_cursor(worker_id, cursor)
                )
                replica_epoch = _ReplicaEpoch(
                    *self.replica_repository.get_worker_replica_epoch_atomic(
                        worker_id, cursor
                    )
                )
                global_replicas: list[Replica] | None = None

                def _ensure_global_replicas() -> list[Replica]:
                    nonlocal global_replicas
                    if global_replicas is None:
                        global_replicas = (
                            self.replica_repository.get_replicas_by_worker_atomic(
                                worker_id, cursor
                            )
                        )
                    return global_replicas

                if (
                    daemon_id
                    and persisted_daemon_id
                    and daemon_id.strip() != persisted_daemon_id
                ):
                    expected_replicas = [
                        self._convert_replica_to_proto(replica)
                        for replica in _ensure_global_replicas()
                    ]
                    inc_control_plane_conflict(scope="reconcile_daemon_mismatch")
                    observe_state_sync(time.time() - _start, success=True)
                    _record_result(
                        global_store_pb2.RECONCILE_RESULT_KIND_REBASE_REQUIRED
                    )
                    return (
                        global_store_pb2.RECONCILE_RESULT_KIND_REBASE_REQUIRED,
                        current_version,
                        current_checksum,
                        [],
                        expected_replicas,
                        0,
                    )

                if generation < persisted_generation:
                    inc_control_plane_conflict(scope="reconcile_generation_stale")
                    observe_state_sync(time.time() - _start, success=True)
                    _record_result(global_store_pb2.RECONCILE_RESULT_KIND_IGNORED_STALE)
                    return (
                        global_store_pb2.RECONCILE_RESULT_KIND_IGNORED_STALE,
                        current_version,
                        current_checksum,
                        [],
                        [],
                        0,
                    )

                if (
                    generation == persisted_generation
                    and request_seq == persisted_request_seq
                ):
                    observe_state_sync(time.time() - _start, success=True)
                    _record_result(global_store_pb2.RECONCILE_RESULT_KIND_NOOP)
                    return (
                        global_store_pb2.RECONCILE_RESULT_KIND_NOOP,
                        current_version,
                        current_checksum,
                        [],
                        [],
                        0,
                    )

                if (
                    generation == persisted_generation
                    and request_seq < persisted_request_seq
                ):
                    inc_control_plane_conflict(scope="reconcile_request_stale")
                    observe_state_sync(time.time() - _start, success=True)
                    _record_result(global_store_pb2.RECONCILE_RESULT_KIND_IGNORED_STALE)
                    return (
                        global_store_pb2.RECONCILE_RESULT_KIND_IGNORED_STALE,
                        current_version,
                        current_checksum,
                        [],
                        [],
                        0,
                    )

                if generation > persisted_generation:
                    expected_replicas = [
                        self._convert_replica_to_proto(replica)
                        for replica in _ensure_global_replicas()
                    ]
                    inc_control_plane_conflict(scope="reconcile_generation_future")
                    observe_state_sync(time.time() - _start, success=True)
                    _record_result(
                        global_store_pb2.RECONCILE_RESULT_KIND_REBASE_REQUIRED
                    )
                    return (
                        global_store_pb2.RECONCILE_RESULT_KIND_REBASE_REQUIRED,
                        current_version,
                        current_checksum,
                        [],
                        expected_replicas,
                        0,
                    )

                if request_seq > persisted_request_seq + 1:
                    inc_control_plane_conflict(scope="reconcile_request_gap")
                    inc_reconcile_retry_later(reason="request_seq_gap")
                    observe_state_sync(time.time() - _start, success=True)
                    _record_result(global_store_pb2.RECONCILE_RESULT_KIND_RETRY_LATER)
                    return (
                        global_store_pb2.RECONCILE_RESULT_KIND_RETRY_LATER,
                        current_version,
                        current_checksum,
                        [],
                        [],
                        100,
                    )

                if (
                    normalized_request_kind
                    == global_store_pb2.RECONCILE_REQUEST_KIND_SNAPSHOT
                    and request_seq == persisted_request_seq + 1
                ):
                    cached = self._read_reconcile_noop_cache(worker_id)
                    if (
                        cached is not None
                        and int(cached.generation) == int(generation)
                        and int(cached.request_seq) == int(persisted_request_seq)
                        and int(cached.state_version) == int(current_version)
                        and cached.state_checksum == current_checksum
                        and cached.inventory_fingerprint == inventory_fingerprint
                        and cached.replica_epoch == replica_epoch
                    ):
                        result_kind = global_store_pb2.RECONCILE_RESULT_KIND_NOOP
                        new_version = int(current_version)
                        new_checksum = current_checksum
                        state_changes: list[global_store_pb2.StateChange] = []
                        self.worker_repository.update_reconcile_cursor(
                            worker_id,
                            generation=generation,
                            request_seq=request_seq,
                            state_version=new_version,
                            state_checksum=new_checksum,
                            last_reconcile_result=global_store_pb2.ReconcileResultKind.Name(
                                result_kind
                            ),
                            cursor=cursor,
                        )
                        self._write_reconcile_noop_cache(
                            worker_id=worker_id,
                            entry=_ReconcileNoopCacheEntry(
                                generation=int(generation),
                                request_seq=int(request_seq),
                                state_version=new_version,
                                state_checksum=new_checksum,
                                inventory_fingerprint=inventory_fingerprint,
                                replica_epoch=replica_epoch,
                            ),
                        )
                        observe_state_sync(time.time() - _start, success=True)
                        _record_result(result_kind)
                        return (
                            result_kind,
                            new_version,
                            new_checksum,
                            state_changes,
                            [],
                            0,
                        )

                force_full_sync = (
                    normalized_request_kind
                    == global_store_pb2.RECONCILE_REQUEST_KIND_SNAPSHOT
                )
                resolved_global_replicas = _ensure_global_replicas()
                state_changes = self._compute_state_changes(
                    inventory,
                    resolved_global_replicas,
                    force_full_sync,
                    self._snapshot_observed_at(inventory),
                )
                if state_changes:
                    adds = 0
                    removes = 0
                    updates = 0
                    for change in state_changes:
                        if (
                            change.type
                            == global_store_pb2.StateChange.CHANGE_TYPE_ADD_REPLICA
                        ):
                            adds += 1
                        elif (
                            change.type
                            == global_store_pb2.StateChange.CHANGE_TYPE_REMOVE_REPLICA
                        ):
                            removes += 1
                        elif (
                            change.type
                            == global_store_pb2.StateChange.CHANGE_TYPE_UPDATE_REPLICA
                        ):
                            updates += 1
                    logger.info(
                        "Reconcile applying changes worker_id=%s daemon_id=%s worker_key=%s generation=%s request_seq=%s adds=%s removes=%s updates=%s",
                        worker_id,
                        daemon_id,
                        worker_key,
                        int(generation),
                        int(request_seq),
                        adds,
                        removes,
                        updates,
                    )

                if state_changes:
                    self._apply_state_changes(worker_id, state_changes, cursor)
                    updated_replicas = (
                        self.replica_repository.get_replicas_by_worker_atomic(
                            worker_id, cursor
                        )
                    )
                    new_checksum = self._compute_state_checksum(updated_replicas)
                    new_version = current_version + 1
                    result_kind = global_store_pb2.RECONCILE_RESULT_KIND_APPLIED
                    self.worker_repository.update_state_version_and_checksum(
                        worker_id, new_version, new_checksum, cursor
                    )
                else:
                    new_version = current_version
                    new_checksum = self._compute_state_checksum(
                        resolved_global_replicas
                    )
                    result_kind = global_store_pb2.RECONCILE_RESULT_KIND_NOOP
                    if current_checksum != new_checksum:
                        self.worker_repository.update_state_checksum(
                            worker_id, new_checksum, cursor
                        )

                self.worker_repository.update_reconcile_cursor(
                    worker_id,
                    generation=generation,
                    request_seq=request_seq,
                    state_version=new_version,
                    state_checksum=new_checksum,
                    last_reconcile_result=global_store_pb2.ReconcileResultKind.Name(
                        result_kind
                    ),
                    cursor=cursor,
                )

                if state_changes:
                    final_replica_epoch = _ReplicaEpoch(
                        *self.replica_repository.get_worker_replica_epoch_atomic(
                            worker_id, cursor
                        )
                    )
                else:
                    final_replica_epoch = replica_epoch
                self._write_reconcile_noop_cache(
                    worker_id=worker_id,
                    entry=_ReconcileNoopCacheEntry(
                        generation=int(generation),
                        request_seq=int(request_seq),
                        state_version=int(new_version),
                        state_checksum=new_checksum,
                        inventory_fingerprint=inventory_fingerprint,
                        replica_epoch=final_replica_epoch,
                    ),
                )

            observe_state_sync(time.time() - _start, success=True)
            _record_result(result_kind)
            return result_kind, new_version, new_checksum, state_changes, [], 0

        except Exception:
            observe_state_sync(time.time() - _start, success=False)
            raise

    def _compute_state_changes(
        self,
        local_replicas: list[common_pb2.ReplicaInfo],
        global_replicas: list[Replica],
        force_full_sync: bool,
        snapshot_observed_at: datetime | None = None,
    ) -> list[global_store_pb2.StateChange]:
        """Compute differences between local and global state."""
        state_changes = []

        def _memory_type_label(mem_type: common_pb2.MemoryType) -> str:
            if mem_type == common_pb2.MemoryType.MEMORY_TYPE_GPU:
                return "GPU"
            if mem_type == common_pb2.MemoryType.MEMORY_TYPE_RAM:
                return "RAM"
            return "DISK"

        def _view_id_from_memory_info(mem_info: common_pb2.MemoryInfo) -> str:
            if (
                mem_info.HasField("byte_space")
                and mem_info.byte_space.kind == common_pb2.BYTE_SPACE_KIND_VIEW
            ):
                return mem_info.byte_space.id.strip()
            return ""

        # Convert to maps for comparison and fast lookup
        local_replicas_by_key: dict[ReplicaKey, common_pb2.ReplicaInfo] = {
            (
                r.ref.artifact_id,
                _view_id_from_memory_info(r.memory_info),
                r.memory_info.node_id,
                _memory_type_label(r.memory_info.memory_type),
                r.memory_info.device_id,
            ): r
            for r in local_replicas
        }

        global_replicas_by_key: dict[ReplicaKey, Replica] = {
            (
                r.artifact_id,
                r.byte_space.id or "",
                r.node_id,
                r.memory_type.value,
                r.device_id,
            ): r
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
        if local_replicas:
            to_remove = global_replica_keys - local_replica_keys
        elif force_full_sync:
            to_remove = global_replica_keys
        else:
            to_remove = set()

        for key in to_remove:
            global_replica = global_replicas_by_key[key]
            if snapshot_observed_at is not None:
                replica_created_at = coerce_db_datetime(global_replica.created_at)
                if (
                    replica_created_at is not None
                    and replica_created_at > snapshot_observed_at
                ):
                    logger.info(
                        "Reconcile preserving replica created after snapshot worker_id=%s replica_id=%s artifact_id=%s created_at=%s snapshot_observed_at=%s",
                        global_replica.worker_id,
                        global_replica.replica_id,
                        global_replica.artifact_id,
                        replica_created_at.isoformat(),
                        snapshot_observed_at.isoformat(),
                    )
                    continue

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
            (
                transport_authoritative,
                local_export_state,
                local_export_generation,
                local_remote_keys,
                local_buffer_sizes,
                local_verification_json,
            ) = parse_transport_metadata(local_replica.memory_info)

            if local_node_address and local_node_address != global_replica.node_address:
                node_address_changed = True
            if local_node_port > 0 and local_node_port != global_replica.node_port:
                node_port_changed = True
            if (
                local_memory_size > 0
                and local_memory_size != global_replica.memory_size
            ):
                memory_size_changed = True
            export_state_changed = False
            export_generation_changed = False
            verification_changed = False

            if transport_authoritative:
                remote_keys_changed = local_remote_keys != list(
                    global_replica.remote_memory_keys
                )
                buffer_sizes_changed = local_buffer_sizes != list(
                    global_replica.buffer_sizes
                )
                export_state_changed = local_export_state != global_replica.export_state
                export_generation_changed = (
                    local_export_generation != global_replica.export_generation
                )
                verification_changed = (local_verification_json or "") != (
                    global_replica.verification_json or ""
                )
            else:
                if local_remote_keys:
                    remote_keys_changed = local_remote_keys != list(
                        global_replica.remote_memory_keys
                    )
                if local_buffer_sizes:
                    buffer_sizes_changed = local_buffer_sizes != list(
                        global_replica.buffer_sizes
                    )
                if local_verification_json:
                    verification_changed = local_verification_json != (
                        global_replica.verification_json or ""
                    )

            if not any(
                [
                    availability_changed,
                    node_address_changed,
                    node_port_changed,
                    memory_size_changed,
                    remote_keys_changed,
                    buffer_sizes_changed,
                    export_state_changed,
                    export_generation_changed,
                    verification_changed,
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
            if transport_authoritative and any(
                [
                    remote_keys_changed,
                    buffer_sizes_changed,
                    export_state_changed,
                    export_generation_changed,
                    verification_changed,
                ]
            ):
                transport = updated_proto.memory_info.transport
                transport.export_state = export_state_to_proto(local_export_state)
                transport.export_generation = int(local_export_generation or 0)
                transport.remote_memory_keys[:] = local_remote_keys
                transport.buffer_sizes[:] = local_buffer_sizes
                transport.verification_json = local_verification_json or ""
                updated_proto.memory_info.remote_memory_keys[:] = local_remote_keys
                updated_proto.memory_info.buffer_sizes[:] = local_buffer_sizes
                updated_proto.memory_info.verification_json = (
                    local_verification_json or ""
                )
                if export_state_changed:
                    reasons.append("export_state")
                if export_generation_changed:
                    reasons.append("export_generation")
                if remote_keys_changed:
                    reasons.append("remote_memory_keys")
                if buffer_sizes_changed:
                    reasons.append("buffer_sizes")
                if verification_changed:
                    reasons.append("verification_json")
            else:
                if remote_keys_changed:
                    updated_proto.memory_info.remote_memory_keys[:] = local_remote_keys
                    reasons.append("remote_memory_keys")
                if buffer_sizes_changed:
                    updated_proto.memory_info.buffer_sizes[:] = local_buffer_sizes
                    reasons.append("buffer_sizes")
                if verification_changed:
                    updated_proto.memory_info.verification_json = (
                        local_verification_json or ""
                    )
                    reasons.append("verification_json")

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
                mem_info = change.replica_info.memory_info
                view_id = (
                    mem_info.byte_space.id
                    if mem_info.HasField("byte_space")
                    and mem_info.byte_space.kind == common_pb2.BYTE_SPACE_KIND_VIEW
                    else ""
                )
                logger.info(
                    "Reconcile adding replica worker_id=%s artifact_id=%s view_id=%s memory_type=%s device_id=%s node_id=%s",
                    worker_id,
                    change.replica_info.ref.artifact_id,
                    view_id,
                    common_pb2.MemoryType.Name(mem_info.memory_type),
                    int(mem_info.device_id),
                    mem_info.node_id,
                )
                replica = self._convert_proto_to_replica(change.replica_info, worker_id)
                self.replica_repository.create_or_update_atomic(replica, cursor)
                logger.debug(f"Added replica: {replica.artifact_id}")

            elif change.type == global_store_pb2.StateChange.CHANGE_TYPE_REMOVE_REPLICA:
                # Remove replica
                mem_info = change.replica_info.memory_info
                view_id = (
                    mem_info.byte_space.id
                    if mem_info.HasField("byte_space")
                    and mem_info.byte_space.kind == common_pb2.BYTE_SPACE_KIND_VIEW
                    else ""
                )
                replica_id_value = change.replica_info.ref.replica_id
                if not replica_id_value:
                    raise ValueError(
                        f"Missing replica_id for removal (artifact_id={change.replica_info.ref.artifact_id})"
                    )
                replica_id = UUID(replica_id_value)
                logger.info(
                    "Reconcile removing replica worker_id=%s replica_id=%s artifact_id=%s view_id=%s memory_type=%s device_id=%s node_id=%s",
                    worker_id,
                    replica_id,
                    change.replica_info.ref.artifact_id,
                    view_id,
                    common_pb2.MemoryType.Name(mem_info.memory_type),
                    int(mem_info.device_id),
                    mem_info.node_id,
                )
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
        memory_info = replica_to_memory_info(replica=replica)
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
        replica_id = (
            UUID(proto_replica.ref.replica_id)
            if proto_replica.ref.replica_id
            else uuid4()
        )
        return memory_info_to_replica(
            mem_info=proto_replica.memory_info,
            artifact_id=proto_replica.ref.artifact_id,
            max_concurrency=proto_replica.stats.max_concurrency,
            worker_id=worker_id,
            require_view_id=False,
            replica_id=replica_id,
            current_requests=proto_replica.stats.current_requests,
            is_available=proto_replica.stats.is_available,
        )

    def _compute_state_checksum(self, replicas: list[Replica]) -> str:
        """Compute checksum of replica state for consistency checking."""
        entries: list[tuple[str, str, str, str, int, int, str, bool]] = []
        for replica in replicas:
            mem_type = "DISK"
            device_id = 0
            if replica.memory_type.value == "GPU":
                mem_type = "GPU"
                device_id = replica.device_id
            elif replica.memory_type.value == "RAM":
                mem_type = "RAM"
                device_id = 0
            view_id = replica.byte_space.id or ""
            entries.append(
                (
                    replica.artifact_id,
                    view_id,
                    replica.node_id or "",
                    replica.node_address or "",
                    replica.node_port or 0,
                    device_id,
                    mem_type,
                    replica.is_available,
                )
            )

        # Sort replicas by a stable key for consistent checksum
        entries.sort(key=lambda e: (e[0], e[1], e[6], e[5]))

        # Create string representation of state
        state_parts = [
            f"{artifact_id}:{view_id}:{node_id}:{node_address}:{node_port}:{device_id}:{memory_type}:{1 if available else 0};"
            for artifact_id, view_id, node_id, node_address, node_port, device_id, memory_type, available in entries
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

    def is_recovery_complete(self) -> bool:
        """Check if recovery process is complete."""
        return not self.recovery_in_progress and self.last_recovery_time > 0

    def get_worker_state_checksum(self, worker_id: str) -> str:
        """Get checksum for the given worker's replica state."""
        checksum = self.worker_repository.get_state_checksum(worker_id)
        if checksum:
            return checksum

        replicas = self.replica_repository.get_replicas_by_worker(worker_id)
        return self._compute_state_checksum(replicas)

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
