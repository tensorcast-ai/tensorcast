#  Copyright (c) 2025, StepCast Team.

"""Chunk state synchronization worker for DVMP integration."""

import logging
import threading
import time
from typing import TYPE_CHECKING

import grpc

from scstore.proto import global_store_pb2, global_store_pb2_grpc

if TYPE_CHECKING:
    from .servicer import StoreDaemonServicer

logger = logging.getLogger(__name__)


class ChunkSyncWorker:
    """Synchronizes chunk states with Global Store for distributed visibility.

    This worker periodically reports the state of chunks in the local DVMP
    to the Global Store, enabling distributed chunk location tracking.
    """

    def __init__(
        self,
        servicer: "StoreDaemonServicer",
        global_store_address: str,
        sync_interval_seconds: float = 0.05,  # 50ms as per spec
        batch_size: int = 1000,
    ):
        """Initialize the chunk sync worker.

        Args:
            servicer: StoreDaemon servicer instance to access Store Engine
            global_store_address: Address of the Global Store service
            sync_interval_seconds: How often to sync chunk states (default 50ms)
            batch_size: Maximum number of chunk updates per RPC
        """
        self.servicer = servicer
        self.global_store_address = global_store_address
        self.sync_interval = sync_interval_seconds
        self.batch_size = batch_size

        self._stop_event = threading.Event()
        self._sync_thread: threading.Thread | None = None
        self._channel: grpc.Channel | None = None
        # NOTE: The generated gRPC module exposes `GlobalStoreStub`,
        # *not* `GlobalStoreStub`.  Using the correct name avoids runtime and
        # type-checking errors.
        self._stub: global_store_pb2_grpc.GlobalStoreStub | None = None

        # Track last sync state to compute deltas
        self._last_chunk_states: dict[
            str, dict[int, int]
        ] = {}  # artifact_id -> chunk_idx -> state

    def start(self) -> None:
        """Start the chunk synchronization worker."""
        if self._sync_thread is not None and self._sync_thread.is_alive():
            logger.warning("Chunk sync worker already running")
            return

        # Skip starting if no global store address (no-op mode)
        if not self.global_store_address:
            logger.debug("Chunk sync worker in no-op mode (no global store address)")
            return

        # Create gRPC channel and stub
        self._channel = grpc.insecure_channel(self.global_store_address)
        self._stub = global_store_pb2_grpc.GlobalStoreStub(self._channel)

        self._stop_event.clear()
        self._sync_thread = threading.Thread(
            target=self._sync_loop, name="chunk-sync-worker", daemon=True
        )
        self._sync_thread.start()
        logger.info(f"Started chunk sync worker (interval={self.sync_interval}s)")

    def stop(self) -> None:
        """Stop the chunk synchronization worker."""
        if self._sync_thread is None:
            return

        logger.info("Stopping chunk sync worker...")
        self._stop_event.set()

        if self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
            if self._sync_thread.is_alive():
                logger.warning("Chunk sync worker thread did not stop gracefully")

        if self._channel:
            self._channel.close()
            self._channel = None
            self._stub = None

        self._sync_thread = None
        logger.info("Chunk sync worker stopped")

    def _sync_loop(self) -> None:
        """Main synchronization loop."""
        while not self._stop_event.is_set():
            try:
                self._perform_sync()
            except Exception as e:
                logger.exception(f"Error in chunk sync: {e}")

            # Sleep with periodic wake-ups for stop checking
            for _ in range(int(self.sync_interval * 10)):
                if self._stop_event.is_set():
                    break
                time.sleep(0.1)

    def _perform_sync(self) -> None:
        """Perform one synchronization cycle."""
        if not self.servicer.store_engine:
            return

        # Get current chunk states from all replicas
        current_states = self._get_all_chunk_states()

        # Compute deltas
        updates = []
        for artifact_id, chunk_states in current_states.items():
            last_states = self._last_chunk_states.get(artifact_id, {})

            for chunk_idx, state in chunk_states.items():
                if chunk_idx not in last_states or last_states[chunk_idx] != state:
                    updates.append(
                        global_store_pb2.ChunkStateUpdate(
                            artifact_id=artifact_id,
                            chunk_idx=chunk_idx,
                            state=self._map_chunk_state(state),
                            device_uuid=self._get_device_uuid(artifact_id),
                            replica=0,  # TODO: Get actual replica number
                        )
                    )

        # Send updates in batches
        if updates and self._stub is not None:
            for i in range(0, len(updates), self.batch_size):
                batch = updates[i : i + self.batch_size]
                try:
                    request = global_store_pb2.BatchUpdateChunkStatesRequest(
                        worker_id=self.servicer.worker_id,
                        node_id=self.servicer.worker_id,  # Using worker_id as node_id
                        updates=batch,
                    )
                    response = self._stub.BatchUpdateChunkStates(request, timeout=5.0)
                    if response.status.code != 0:
                        logger.warning(
                            f"Failed to update chunk states: {response.status.message}"
                        )
                except grpc.RpcError as e:
                    logger.error(f"RPC error updating chunk states: {e}")
        elif updates:
            logger.warning("Chunk updates generated but gRPC stub is not initialised")

        # Update last known states
        self._last_chunk_states = current_states

    def _get_all_chunk_states(self) -> dict[str, dict[int, int]]:
        """Get chunk states for all artifacts from Store Engine.

        Returns:
            Dictionary mapping artifact_id -> chunk_idx -> state
        """
        # Build up state map: artifact_id -> chunk_idx -> state
        result: dict[str, dict[int, int]] = {}

        # Get list of loaded replicas from replica manager
        for replica_info in self.servicer.replica_manager.get_loaded_replicas():
            # Use artifact identifier from replica info dict
            disk_path = str(replica_info.get("artifact_id", ""))
            artifact_id = self._extract_artifact_id(disk_path)

            # TODO: Replace placeholder with actual store_engine snapshot
            result[artifact_id] = {}

        return result

    def _map_chunk_state(self, state: int) -> global_store_pb2.ChunkState:
        """Map internal chunk state to protobuf enum."""
        # Map from ChunkState enum values
        # HOT = 0, LOCKED_TX = 1, COPIED_GPU = 2, COLD = 3, EVICTED = 4
        mapping = {
            0: global_store_pb2.ChunkState.CHUNK_HOT,
            1: global_store_pb2.ChunkState.CHUNK_LOCKED_TX,
            2: global_store_pb2.ChunkState.CHUNK_COPIED_GPU,
            3: global_store_pb2.ChunkState.CHUNK_COLD,
            4: global_store_pb2.ChunkState.CHUNK_EVICTED,
        }

        # Ensure a concrete (non-optional) enum is always returned for type-checking
        return (
            mapping[state]
            if state in mapping
            else global_store_pb2.ChunkState.CHUNK_HOT  # Default to HOT for unknown states
        )

    def _get_device_uuid(self, artifact_id: str) -> str:
        """Get device UUID for a replica."""
        # TODO: Get actual device UUID from replica metadata
        return ""

    def _extract_artifact_id(self, disk_path: str) -> str:
        """Extract replica ID from replica path."""
        # Simple extraction - use last component of path
        return disk_path.split("/")[-1] if "/" in disk_path else disk_path
