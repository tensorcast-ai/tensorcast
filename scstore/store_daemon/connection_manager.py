#  Copyright (c) 2025, StepCast Team.

#
# ----------------------------------------------------------------------------
"""
Synchronous (thread-based) connection manager for Store Daemon high-availability.

This implementation removes all asyncio dependencies so the daemon can run in
pure multi-threaded environments.  It still provides the same public interface
(start/stop/register_model_replica/…) but performs all background work in
`threading.Thread`s and uses blocking gRPC stubs (`grpc.insecure_channel`).
"""

from __future__ import annotations

import contextlib
import hashlib
import threading
import time
from enum import Enum
from typing import Any, Dict

import grpc

from scstore.logger import init_logger
from scstore.proto import global_store_pb2, global_store_pb2_grpc, store_daemon_pb2
from scstore.store_daemon import metrics

# Import HighAvailabilityConfig for type safety
from .config import HighAvailabilityConfig

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Helpers & small utility classes
# ---------------------------------------------------------------------------


class ConnectionState(Enum):
    """High-level connection state."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


class StateChangeQueue:
    """Thread-safe queue for pending registrations/unregistrations."""

    def __init__(self) -> None:
        self._model_registrations: Dict[
            str, global_store_pb2.RegisterModelReplicaRequest
        ] = {}
        self._model_unregistrations: set[str] = set()
        self._lock = threading.Lock()

    # Public helpers ------------------------------------------------------------------

    def queue_model_registration(
        self, request: global_store_pb2.RegisterModelReplicaRequest
    ) -> None:
        key = f"{request.model_id}:{request.mem_info.node_id}:{request.mem_info.device_id}"
        with self._lock:
            self._model_registrations[key] = request
            self._model_unregistrations.discard(key)
            logger.debug("Queued model registration: %s", key)
            metrics.HA_PENDING_CHANGES.labels(type="registrations").set(
                len(self._model_registrations)
            )
            metrics.HA_PENDING_CHANGES.labels(type="unregistrations").set(
                len(self._model_unregistrations)
            )

    def queue_model_unregistration(
        self, model_id: str, node_id: str, device_id: int
    ) -> None:
        key = f"{model_id}:{node_id}:{device_id}"
        with self._lock:
            self._model_unregistrations.add(key)
            self._model_registrations.pop(key, None)
            logger.debug("Queued model unregistration: %s", key)
            metrics.HA_PENDING_CHANGES.labels(type="registrations").set(
                len(self._model_registrations)
            )
            metrics.HA_PENDING_CHANGES.labels(type="unregistrations").set(
                len(self._model_unregistrations)
            )

    def get_pending_changes(self) -> Dict[str, Any]:
        """Atomically obtain & clear pending changes."""
        with self._lock:
            changes = {
                "registrations": dict(self._model_registrations),
                "unregistrations": set(self._model_unregistrations),
            }
            self._model_registrations.clear()
            self._model_unregistrations.clear()
            # Reset pending changes metrics
            metrics.HA_PENDING_CHANGES.labels(type="registrations").set(0)
            metrics.HA_PENDING_CHANGES.labels(type="unregistrations").set(0)
            return changes

    def is_empty(self) -> bool:
        with self._lock:
            return not self._model_registrations and not self._model_unregistrations


# ---------------------------------------------------------------------------
# Main manager implementation
# ---------------------------------------------------------------------------


class GlobalStoreConnectionManager:
    """Connection manager variant that relies solely on threads."""

    def __init__(
        self,
        servicer,  # StoreDaemonServicer instance
        global_store_address: str,
        ha_config: HighAvailabilityConfig,
    ) -> None:
        self._servicer = servicer
        self._address = global_store_address

        # gRPC channel & stub (blocking version)
        self.grpc_channel: grpc.Channel | None = None
        self.global_store_stub: global_store_pb2_grpc.GlobalModelStoreStub | None = None

        # State & bookkeeping ----------------------------------------------------
        self._state_lock = threading.RLock()  # Reentrant lock for nested calls
        self._connection_state = ConnectionState.DISCONNECTED
        self.state_change_queue = StateChangeQueue()
        self._local_state_version = 0
        self._last_successful_sync = 0
        self._registered_models: set[str] = set()

        # Periodic/heartbeat intervals
        self._heartbeat_interval_sec: float = 10.0  # updated after RegisterWorker
        self._periodic_sync_interval_sec: int = int(
            ha_config.periodic_sync_interval_ms / 1000
        )
        self._state_sync_enabled = ha_config.enabled

        # Retry configuration
        self._max_retries = ha_config.max_retries
        self._connection_attempts = 0
        self._retry_delay_sec: float = ha_config.max_delay_ms / 1000.0

        # Thread handles
        self._conn_thread: threading.Thread | None = None
        self._hb_thread: threading.Thread | None = None
        self._sync_thread: threading.Thread | None = None
        self._monitor_thread: threading.Thread | None = None
        self._stop_evt = threading.Event()

        # Thread monitoring configuration
        self._thread_monitor_interval = ha_config.thread_monitor_interval_ms / 1000.0
        self._auto_restart_threads = ha_config.auto_restart_threads

    # Thread-safe properties
    @property
    def connection_state(self) -> ConnectionState:
        with self._state_lock:
            return self._connection_state

    @connection_state.setter
    def connection_state(self, value: ConnectionState) -> None:
        with self._state_lock:
            self._connection_state = value
            # Update metric
            state_map = {
                ConnectionState.DISCONNECTED: 0,
                ConnectionState.CONNECTING: 1,
                ConnectionState.CONNECTED: 2,
                ConnectionState.RECONNECTING: 3,
                ConnectionState.FAILED: 4,
            }
            metrics.HA_CONNECTION_STATE.set(state_map.get(value, -1))

    @property
    def local_state_version(self) -> int:
        with self._state_lock:
            return self._local_state_version

    @local_state_version.setter
    def local_state_version(self, value: int) -> None:
        with self._state_lock:
            self._local_state_version = value
            metrics.HA_STATE_VERSION.set(value)

    @property
    def last_successful_sync(self) -> int:
        with self._state_lock:
            return self._last_successful_sync

    @last_successful_sync.setter
    def last_successful_sync(self, value: int) -> None:
        with self._state_lock:
            self._last_successful_sync = value

    @property
    def registered_models(self) -> set[str]:
        with self._state_lock:
            return self._registered_models.copy()

    def _add_registered_model(self, model_id: str) -> None:
        with self._state_lock:
            self._registered_models.add(model_id)
            metrics.HA_REGISTERED_MODELS.set(len(self._registered_models))

    def _remove_registered_model(self, model_id: str) -> None:
        with self._state_lock:
            self._registered_models.discard(model_id)
            metrics.HA_REGISTERED_MODELS.set(len(self._registered_models))

    # ------------------------------------------------------------------
    # Public lifecycle helpers
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._conn_thread and self._conn_thread.is_alive():
            # already running
            return

        logger.info("Starting Global Store connection manager")
        self._stop_evt.clear()

        # Spawn background threads ------------------------------------------------
        self._conn_thread = threading.Thread(
            target=self._maintain_connection_loop,
            name="gs-conn-maintain",
            daemon=True,
        )
        self._conn_thread.start()

        self._hb_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="gs-heartbeat",
            daemon=True,
        )
        self._hb_thread.start()

        self._sync_thread = threading.Thread(
            target=self._periodic_sync_loop,
            name="gs-full-sync",
            daemon=True,
        )
        self._sync_thread.start()

        if self._auto_restart_threads:
            self._monitor_thread = threading.Thread(
                target=self._monitor_threads_loop,
                name="gs-thread-monitor",
                daemon=True,
            )
            self._monitor_thread.start()

    def stop(self, join_timeout: float = 5.0) -> None:
        logger.info("Stopping Global Store connection manager")
        self._stop_evt.set()

        for th in [
            self._hb_thread,
            self._sync_thread,
            self._conn_thread,
            self._monitor_thread,
        ]:
            if th and th.is_alive():
                th.join(timeout=join_timeout)

        # Close channel (best-effort)
        self._disconnect()

    # ------------------------------------------------------------------
    # Thread entry-points
    # ------------------------------------------------------------------

    def _maintain_connection_loop(self) -> None:
        while not self._stop_evt.is_set():
            try:
                if self.connection_state in (
                    ConnectionState.DISCONNECTED,
                    ConnectionState.FAILED,
                ):
                    # Check if we should attempt connection based on retry limits
                    if (
                        self._max_retries == -1
                        or self._connection_attempts < self._max_retries
                    ):
                        self._attempt_connection()
                    elif (
                        self._connection_attempts >= self._max_retries
                        and self.connection_state != ConnectionState.FAILED
                    ):
                        logger.error(
                            "Maximum connection retries (%d) exceeded. Stopping connection attempts.",
                            self._max_retries,
                        )
                        self.connection_state = ConnectionState.FAILED
                elif (
                    self.connection_state is ConnectionState.CONNECTED
                    and not self._check_connection_health()
                ):
                    logger.warning("Connection health failed – initiating reconnect")
                    self.connection_state = ConnectionState.RECONNECTING
                    self._disconnect()
                    # Reset retry count for reconnection attempts
                    self._connection_attempts = 0
                # Sleep outside of exception handler to avoid rapid loops –
                # use a dynamic interval when disconnected/failed.
                wait_time = (
                    self._heartbeat_interval_sec * 0.6
                    if self.connection_state
                    in (ConnectionState.FAILED, ConnectionState.DISCONNECTED)
                    and (
                        self._max_retries == -1
                        or self._connection_attempts < self._max_retries
                    )
                    else self._heartbeat_interval_sec * 2
                )
                self._stop_evt.wait(wait_time)
            except Exception as exc:
                logger.exception("Error in connection maintenance loop: %s", exc)
                self._stop_evt.wait(1.0)

    def _heartbeat_loop(self) -> None:
        while not self._stop_evt.is_set():
            try:
                if self.connection_state is ConnectionState.CONNECTED:
                    self._send_enhanced_heartbeat()
            except Exception as exc:
                logger.exception("Heartbeat loop error: %s", exc)
            finally:
                self._stop_evt.wait(self._heartbeat_interval_sec)

    def _periodic_sync_loop(self) -> None:
        while not self._stop_evt.is_set():
            if self.connection_state is ConnectionState.CONNECTED:
                try:
                    logger.debug("Periodic full state sync triggered")
                    self._perform_state_sync(force_full=True)
                except Exception as exc:
                    logger.error("Periodic state sync failed: %s", exc)
            self._stop_evt.wait(self._periodic_sync_interval_sec)

    def _monitor_threads_loop(self) -> None:
        """Monitor other threads and restart them if they die unexpectedly."""
        while not self._stop_evt.is_set():
            try:
                # Check connection thread
                if self._conn_thread and not self._conn_thread.is_alive():
                    logger.warning("Connection thread died, restarting...")
                    self._conn_thread = threading.Thread(
                        target=self._maintain_connection_loop,
                        name="gs-conn-maintain",
                        daemon=True,
                    )
                    self._conn_thread.start()
                    metrics.HA_THREAD_RESTARTS_TOTAL.labels(
                        thread_name="connection"
                    ).inc()

                # Check heartbeat thread
                if self._hb_thread and not self._hb_thread.is_alive():
                    logger.warning("Heartbeat thread died, restarting...")
                    self._hb_thread = threading.Thread(
                        target=self._heartbeat_loop,
                        name="gs-heartbeat",
                        daemon=True,
                    )
                    self._hb_thread.start()
                    metrics.HA_THREAD_RESTARTS_TOTAL.labels(
                        thread_name="heartbeat"
                    ).inc()

                # Check sync thread
                if self._sync_thread and not self._sync_thread.is_alive():
                    logger.warning("Sync thread died, restarting...")
                    self._sync_thread = threading.Thread(
                        target=self._periodic_sync_loop,
                        name="gs-full-sync",
                        daemon=True,
                    )
                    self._sync_thread.start()
                    metrics.HA_THREAD_RESTARTS_TOTAL.labels(thread_name="sync").inc()

            except Exception as exc:
                logger.exception("Error in thread monitor loop: %s", exc)

            self._stop_evt.wait(self._thread_monitor_interval)

    # ------------------------------------------------------------------
    # Core helpers (blocking)
    # ------------------------------------------------------------------

    def _attempt_connection(self) -> None:
        self._connection_attempts += 1
        retry_info = f"(attempt {self._connection_attempts}"
        if self._max_retries > 0:
            retry_info += f"/{self._max_retries})"
        else:
            retry_info += ", infinite retries)" if self._max_retries == -1 else ")"

        try:
            self.connection_state = ConnectionState.CONNECTING
            logger.info(
                "Connecting to Global Store at %s %s", self._address, retry_info
            )

            # Build blocking gRPC channel
            self.grpc_channel = grpc.insecure_channel(self._address)
            self.global_store_stub = global_store_pb2_grpc.GlobalModelStoreStub(
                self.grpc_channel
            )
            # Expose stub on servicer for backwards compatibility
            self._servicer.global_store_stub = self.global_store_stub

            # Smoke-test connection
            self._test_connection()
            self._register_worker()
            self._sync_pending_changes()

            self.connection_state = ConnectionState.CONNECTED
            self._connection_attempts = 0  # Reset on successful connection
            logger.info("Successfully connected to Global Store")
        except Exception as exc:
            logger.error("Failed to connect to Global Store: %s", exc)
            self.connection_state = ConnectionState.FAILED
            self._disconnect()

            # Log retry info
            if self._max_retries == -1 or self._connection_attempts < self._max_retries:
                logger.info(f"Will retry connection in {self._retry_delay_sec:.2f} s")
            else:
                logger.warning(
                    "Maximum connection retries reached. No more attempts will be made."
                )

            metrics.HA_CONNECTION_RETRIES_TOTAL.inc()

    # ---------- small helpers ---------------------------------------------------

    def _test_connection(self) -> None:
        if not self.global_store_stub:
            raise RuntimeError("Global Store stub not initialised")
        request = global_store_pb2.HealthCheckRequest()
        self.global_store_stub.HealthCheck(request, timeout=5.0)

    def _register_worker(self) -> None:
        if not (self.global_store_stub and self._servicer.checkpoint_store):
            logger.warning(
                "Checkpoint store unavailable – skipping worker registration"
            )
            return

        req = global_store_pb2.RegisterWorkerRequest(
            node_id=self._servicer.node_id,
            node_address=self._servicer.node_address,
            grpc_port=self._servicer.grpc_port,
            p2p_port=self._servicer.node_port,
            mem_pool_total_size=self._servicer.checkpoint_store.get_mem_pool_size(),
            mem_pool_available_size=self._servicer.checkpoint_store.get_available_memory(),
            is_recovery_registration=bool(self._servicer.worker_id),
            previous_worker_id=self._servicer.worker_id or "",
        )
        resp = self.global_store_stub.RegisterWorker(req)
        if resp.status != global_store_pb2.Status.OK:
            raise RuntimeError(f"RegisterWorker failed with status {resp.status}")

        self._servicer.worker_id = resp.worker_id
        self._heartbeat_interval_sec = max(resp.heartbeat_interval_ms / 1000.0, 1.0)
        logger.info(
            "Worker %s registered (heartbeat %.1fs)",
            resp.worker_id,
            self._heartbeat_interval_sec,
        )

    def _sync_pending_changes(self) -> None:
        pending = self.state_change_queue.get_pending_changes()
        # Ensure that the Global Store gRPC stub is available. This assertion
        # serves purely as a hint for static type checkers (e.g. mypy, pylint)
        # and has no runtime impact outside of a failed connection attempt.
        assert self.global_store_stub is not None, (
            "Global Store stub must be initialised"
        )
        if pending["registrations"]:
            logger.info(
                "Syncing %d queued model registrations", len(pending["registrations"])
            )
            for key, req in pending["registrations"].items():
                try:
                    self.global_store_stub.RegisterModelReplica(req)
                    logger.debug("Synced model registration: %s", key)
                except Exception as exc:
                    logger.error("Failed to sync registration %s: %s", key, exc)
                    self.state_change_queue.queue_model_registration(req)

        if pending["unregistrations"]:
            logger.info(
                "Syncing %d queued model unregistrations",
                len(pending["unregistrations"]),
            )
            for key in pending["unregistrations"]:
                model_id, *_ = key.split(":", 1)
                self._remove_registered_model(model_id)
                logger.debug("Marked %s as unregistered", model_id)

        self.last_successful_sync = int(time.time())

    def _check_connection_health(self) -> bool:
        if not self.global_store_stub:
            return False
        try:
            req = global_store_pb2.HealthCheckRequest()
            self.global_store_stub.HealthCheck(req, timeout=3.0)
            return True
        except Exception as exc:
            logger.debug("Health-check RPC failed: %s", exc)
            return False

    def _disconnect(self) -> None:
        if self.grpc_channel:
            with contextlib.suppress(Exception):
                self.grpc_channel.close()
        self.grpc_channel = None
        self.global_store_stub = None
        self.connection_state = ConnectionState.DISCONNECTED
        self._servicer.global_store_stub = None

    # ------------------------------------------------------------------
    # Heartbeat & State-sync helpers
    # ------------------------------------------------------------------

    def _send_enhanced_heartbeat(self) -> None:
        if not (self.global_store_stub and self._servicer.worker_id):
            return

        try:
            state_checksum = self._compute_current_state_checksum()
            request = global_store_pb2.WorkerHeartbeatRequest(
                worker_id=self._servicer.worker_id,
                mem_pool_available_size=self._servicer.checkpoint_store.get_available_memory()
                if self._servicer.checkpoint_store
                else 0,
                accepting_new_requests=not self._servicer.shutting_down,
                state_version=self.local_state_version,
                state_checksum=state_checksum,
                registered_model_ids=list(self.registered_models),
                last_successful_sync=self.last_successful_sync,
                global_store_status=global_store_pb2.CONNECTED,
            )
            response = self.global_store_stub.WorkerHeartbeat(request)
            metrics.HA_HEARTBEAT_TOTAL.labels(status="success").inc()

            if response.state_sync_required:
                logger.info("State sync requested by Global Store")
                self._perform_state_sync(
                    expected_version=response.expected_state_version
                )

            if response.obsolete_models:
                logger.info(
                    "Received %d obsolete models to remove",
                    len(response.obsolete_models),
                )
                for model_id in response.obsolete_models:
                    self._remove_registered_model(model_id)

        except Exception as exc:
            logger.error("Failed to send heartbeat: %s", exc)
            metrics.HA_HEARTBEAT_TOTAL.labels(status="failure").inc()

    def _compute_current_state_checksum(self) -> str:
        sorted_models = sorted(self.registered_models)
        raw = ":".join(sorted_models)
        return hashlib.md5(raw.encode()).hexdigest()

    def _perform_state_sync(
        self, expected_version: int = 0, *, force_full: bool = False
    ) -> None:
        # Note: expected_version can be used for optimized sync in future implementations
        sync_type = "full" if force_full else "incremental"
        try:
            local_state = global_store_pb2.WorkerLocalState(
                worker_id=self._servicer.worker_id,
                state_version=self.local_state_version,
                state_checksum=self._compute_current_state_checksum(),
                local_replicas=[],
                last_update_timestamp=int(time.time()),
            )
            req = global_store_pb2.SynchronizeWorkerStateRequest(
                worker_id=self._servicer.worker_id,
                local_state=local_state,
                force_full_sync=force_full,
            )
            # Safe-guard for type checker – stub is always set for connected state
            assert self.global_store_stub is not None
            resp = self.global_store_stub.SynchronizeWorkerState(req)
            if resp.status != global_store_pb2.Status.OK:
                logger.warning("State sync failed with status %s", resp.status)
                metrics.HA_STATE_SYNC_TOTAL.labels(
                    type=sync_type, status="failure"
                ).inc()
                return

            self.local_state_version = resp.new_state_version
            self.last_successful_sync = int(time.time())
            logger.info("State sync complete, new version %d", resp.new_state_version)
            metrics.HA_STATE_SYNC_TOTAL.labels(type=sync_type, status="success").inc()

            # Apply remote changes ----------------------------------------------------
            for change in resp.state_changes:
                change_type = change.type
                replica_info = change.replica_info
                model_id = replica_info.model_id

                if change_type == global_store_pb2.StateChange.REMOVE_REPLICA:
                    try:
                        self._servicer.replica_manager.unload_model(
                            model_id,
                            store_daemon_pb2.DEVICE_TYPE_GPU,
                        )
                    except Exception:
                        logger.debug("Failed to unload model %s during sync", model_id)
                    self._remove_registered_model(model_id)
                    metrics.HA_STATE_CHANGES_TOTAL.labels(
                        change_type="REMOVE_REPLICA"
                    ).inc()

                elif change_type == global_store_pb2.StateChange.ADD_REPLICA:
                    # Add a new replica that was registered elsewhere
                    logger.info("Adding replica for model %s from state sync", model_id)
                    self._add_registered_model(model_id)
                    metrics.HA_STATE_CHANGES_TOTAL.labels(
                        change_type="ADD_REPLICA"
                    ).inc()
                    # Note: We don't actually load the model here, just track it
                    # The model will be loaded when needed via LoadModel RPC

                elif change_type == global_store_pb2.StateChange.UPDATE_REPLICA:
                    # Update replica metadata (e.g., status changes)
                    logger.info(
                        "Updating replica for model %s from state sync", model_id
                    )
                    # For now, just ensure it's in our registered set
                    self._add_registered_model(model_id)
                    metrics.HA_STATE_CHANGES_TOTAL.labels(
                        change_type="UPDATE_REPLICA"
                    ).inc()
                    # Future: Could update local metadata if we maintain more info

        except Exception as exc:
            logger.error("State sync failed: %s", exc)
            metrics.HA_STATE_SYNC_TOTAL.labels(type=sync_type, status="failure").inc()

    # ------------------------------------------------------------------
    # Public proxies used by other components
    # ------------------------------------------------------------------

    def register_model_replica(
        self, request: global_store_pb2.RegisterModelReplicaRequest
    ) -> bool:
        if (
            self.connection_state is ConnectionState.CONNECTED
            and self.global_store_stub
        ):
            # Type-narrowing hint for static checkers
            assert self.global_store_stub is not None
            try:
                resp = self.global_store_stub.RegisterModelReplica(request)
                if resp.status == global_store_pb2.Status.OK:
                    self._add_registered_model(request.model_id)
                    self.local_state_version += 1
                    return True
            except Exception as exc:
                logger.error("RegisterModelReplica RPC failed: %s", exc)
        # Offline/failure – queue for later
        self.state_change_queue.queue_model_registration(request)
        return False

    def unregister_model_replica(
        self, model_id: str, replica_id: str, device_id: int = 0
    ) -> bool:
        if (
            self.connection_state is ConnectionState.CONNECTED
            and self.global_store_stub
        ):
            try:
                req = global_store_pb2.UnregisterModelReplicaRequest(
                    model_id=model_id,
                    replica_id=replica_id,
                )
                resp = self.global_store_stub.UnregisterModelReplica(req)
                if resp.status == global_store_pb2.Status.OK:
                    self._remove_registered_model(model_id)
                    self.local_state_version += 1
                    return True
            except Exception as exc:
                logger.error("UnregisterModelReplica RPC failed: %s", exc)
        # Queue for later
        self.state_change_queue.queue_model_unregistration(
            model_id, self._servicer.node_id, device_id
        )
        return False

    def get_connection_status(self) -> Dict[str, Any]:
        return {
            "state": self.connection_state.value,
            "connected": self.connection_state is ConnectionState.CONNECTED,
            "local_state_version": self.local_state_version,
            "last_successful_sync": self.last_successful_sync,
            "registered_models_count": len(self.registered_models),
            "pending_changes": not self.state_change_queue.is_empty(),
        }
