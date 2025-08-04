#  Copyright (c) 2025, StepCast Team.

"""Background worker for lifecycle management tasks."""

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .chunk_sync import ChunkSyncWorker
    from .process_watcher import ProcessWatcher
    from .replica_manager import ReplicaManager

logger = logging.getLogger(__name__)


class LifecycleWorker:
    """Manages background lifecycle tasks for the Store Daemon.

    Coordinates process watching and periodic eviction checks.
    """

    def __init__(
        self,
        process_watcher: "ProcessWatcher",
        replica_manager: "ReplicaManager",
        eviction_check_interval_seconds: float = 30.0,
        gpu_memory_limit_fraction: float = 0.9,
        chunk_sync_worker: "ChunkSyncWorker | None" = None,
    ):
        """Initialize the lifecycle worker.

        Args:
            process_watcher: Process watcher instance
            replica_manager: Replica manager instance
            eviction_check_interval_seconds: How often to check for eviction
            gpu_memory_limit_fraction: GPU memory usage threshold for eviction
            chunk_sync_worker: Optional chunk sync worker for DVMP integration
        """
        self.process_watcher = process_watcher
        self.replica_manager = replica_manager
        self.eviction_interval = eviction_check_interval_seconds
        self.gpu_memory_threshold = gpu_memory_limit_fraction
        self.chunk_sync_worker = chunk_sync_worker

        self._stop_event = threading.Event()
        self._eviction_thread: threading.Thread | None = None

    def start(self) -> None:
        """Start all lifecycle management tasks."""
        # Start process watcher
        self.process_watcher.start()

        # Start chunk sync worker if available
        if self.chunk_sync_worker:
            self.chunk_sync_worker.start()

        # Start eviction thread
        if self._eviction_thread is not None and self._eviction_thread.is_alive():
            logger.warning("Lifecycle worker eviction thread already running")
            return

        self._stop_event.clear()
        self._eviction_thread = threading.Thread(
            target=self._eviction_loop, name="LifecycleEviction", daemon=True
        )
        self._eviction_thread.start()
        logger.info(
            f"Started lifecycle worker with eviction interval {self.eviction_interval}s"
        )

    def stop(self) -> None:
        """Stop all lifecycle management tasks."""
        # Stop process watcher
        self.process_watcher.stop()

        # Stop chunk sync worker if available
        if self.chunk_sync_worker:
            self.chunk_sync_worker.stop()

        # Stop eviction thread
        self._stop_event.set()
        if self._eviction_thread:
            self._eviction_thread.join(timeout=5.0)
            self._eviction_thread = None
        logger.info("Stopped lifecycle worker")

    def _eviction_loop(self) -> None:
        """Main eviction check loop."""
        while not self._stop_event.is_set():
            try:
                # Check GPU memory usage and trigger eviction if needed
                self._check_and_evict()
            except Exception:
                logger.exception("Error in eviction check")

            # Sleep with interruptible wait
            self._stop_event.wait(self.eviction_interval)

    def _check_and_evict(self) -> None:
        """Check GPU memory usage and trigger eviction if necessary.

        The original implementation compared *overall* GPU utilisation against
        the configured threshold which led to excessive log-spam when other
        processes were occupying the majority of the device memory.  We now
        base the decision **solely on the memory consumed by the Store
        Daemon itself**.  Eviction (and the accompanying INFO-level log)
        therefore only happens when the *daemon* exceeds the configured
        fraction of the device capacity.
        """
        try:
            gpu_stats = self.replica_manager.get_gpu_memory_stats()

            for device_id, stats in gpu_stats.items():
                total_mem = stats.get("total", 0)
                if total_mem <= 0:
                    continue  # Skip devices without valid stats

                # ------------------------------------------------------------------
                # Memory attributed to the Store Daemon on this device
                # ------------------------------------------------------------------
                store_used = self.replica_manager.get_device_cache_bytes(device_id)
                usage_fraction = store_used / total_mem if total_mem else 0.0

                if usage_fraction > self.gpu_memory_threshold:
                    # Bytes that need to be freed to get back *below* the threshold
                    target_bytes = int(
                        max(0, store_used - self.gpu_memory_threshold * total_mem)
                    )
                    if target_bytes <= 0:
                        continue

                    logger.info(
                        "GPU %s daemon cache usage %.2f%% exceeds threshold %.2f%%, "
                        "triggering periodic eviction",
                        device_id,
                        usage_fraction * 100.0,
                        self.gpu_memory_threshold * 100.0,
                    )

                    self.replica_manager.periodic_evict(
                        device_id=device_id, bytes_needed=target_bytes
                    )
                else:
                    logger.debug(
                        "GPU %s daemon cache usage %.2f%% within threshold %.2f%%, no eviction",
                        device_id,
                        usage_fraction * 100.0,
                        self.gpu_memory_threshold * 100.0,
                    )

        except Exception:
            logger.exception("Error checking GPU memory for eviction")

    def trigger_eviction_check(self) -> None:
        """Manually trigger an eviction check (for testing/debugging)."""
        logger.info("Manually triggering eviction check")
        self._check_and_evict()

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
