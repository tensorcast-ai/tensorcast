#  Copyright (c) 2025, StepCast Team.

"""Replica management with reference counting and eviction logic.

This module handles the lifecycle of loaded model replicas including:
- Reference counting based on process PIDs
- Eviction when memory pressure is high
- Registration with global store
- Confirmation and unloading operations
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Set

# gRPC interactions are now handled exclusively in the C++ core – python layer no longer needs grpc.
from scstore import _checkpoint_store as _cs
from scstore.logger import init_logger
from scstore.proto import store_daemon_pb2

from .metrics import (
    EVICTIONS_TOTAL,
    GPU_CACHE_BYTES,
    MODEL_REF_COUNT,
    MODELS_UNLOADED_TOTAL,
    get_device_type_label,
)
from .replica_ref import ReplicaKey, ReplicaRefInfo

if TYPE_CHECKING:  # pragma: no cover
    from .servicer import StoreDaemonServicer

logger = init_logger(__name__)


def _normalize_gpu_stats(raw_stats: Any) -> Dict[int, Dict[str, int]]:
    """Normalize GPU stats from various formats to consistent structure.

    Args:
        raw_stats: Raw GPU stats from checkpoint store, can be:
            - Dict[str, int] with 'total' and 'free' keys (single GPU)
            - List of tuples [(total, free), ...] (multi-GPU)
            - Dict[int, Dict[str, int]] (already normalized)

    Returns:
        Dict mapping device_id to stats dict with 'total', 'free', 'used' keys
    """
    result: Dict[int, Dict[str, int]] = {}

    # Already normalized format
    if hasattr(raw_stats, "items") and all(
        hasattr(v, "get") and "total" in v and "free" in v
        for v in raw_stats.values()
        if hasattr(v, "get")
    ):
        return raw_stats

    # Single GPU dict format
    if hasattr(raw_stats, "get") and "total" in raw_stats and "free" in raw_stats:
        total_val = int(raw_stats["total"])
        free_val = int(raw_stats["free"])
        used_val = int(raw_stats.get("used", total_val - free_val))
        result[0] = {
            "total": total_val,
            "free": free_val,
            "used": used_val,
        }
        return result

    # List of tuples format
    if hasattr(raw_stats, "__getitem__") and hasattr(raw_stats, "__len__"):
        for device_id, stats in enumerate(raw_stats):
            if hasattr(stats, "__getitem__") and len(stats) >= 2:
                total_val = int(stats[0])
                free_val = int(stats[1])
                result[device_id] = {
                    "total": total_val,
                    "free": free_val,
                    "used": total_val - free_val,
                }

    return result


class ReplicaManager:
    """Manages model replica lifecycle with reference counting and eviction.

    This class handles:
    - Reference counting based on PIDs
    - Memory pressure-based eviction
    - Model confirmation and registration
    - Graceful unloading
    """

    # Retry constants
    MAX_RETRIES: int = 5
    CONFIRM_RETRY_DELAY_SEC: float = 0.05
    UNLOAD_RETRY_DELAY_SEC: float = 0.01

    def __init__(self, servicer: "StoreDaemonServicer") -> None:
        self._servicer = servicer
        self._checkpoint_store = servicer.checkpoint_store
        self._global_store_stub = servicer.global_store_stub

        # Reference counting data structures
        self._replicas: Dict[ReplicaKey, ReplicaRefInfo] = {}
        self._lock = threading.RLock()  # Reentrant lock for nested calls

        # Reverse lookup: model_path -> Set[ReplicaKey]
        self._model_to_keys: Dict[str, Set[ReplicaKey]] = {}

        # ---------------------------------------------------------------------
        # Reference Counting API
        # ---------------------------------------------------------------------

    def add_ref(
        self,
        model_path: str,
        device_id: int,
        pid: int,
        size_bytes: int = 0,
        keep_for_global: bool = False,
    ) -> bool:
        """Add a reference to a model replica.

        Args:
            model_path: Model identifier
            device_id: GPU device ID
            pid: Process ID of the user
            size_bytes: Size of the model in bytes
            keep_for_global: Whether to keep for global cache

        Returns:
            True if reference was added successfully
        """
        key = ReplicaKey(model_id=model_path, device_id=device_id)

        with self._lock:
            if key not in self._replicas:
                # Create new replica info
                self._replicas[key] = ReplicaRefInfo(
                    key=key,
                    size_bytes=size_bytes,
                    keep_for_global=keep_for_global,
                )

                # Update reverse lookup
                if model_path not in self._model_to_keys:
                    self._model_to_keys[model_path] = set()
                self._model_to_keys[model_path].add(key)

            replica_info = self._replicas[key]
            added = replica_info.add_pid(pid)

            if added:
                logger.info(
                    f"Added reference for PID {pid} to {key}, "
                    f"total refs: {replica_info.ref_count}"
                )
                # Update metrics
                MODEL_REF_COUNT.labels(
                    model=key.model_id, device_id=str(key.device_id)
                ).set(replica_info.ref_count)
            else:
                logger.debug(f"PID {pid} already has reference to {key}")

            return True

    def remove_ref(self, model_path: str, device_id: int, pid: int) -> bool:
        """Remove a reference from a model replica.

        Args:
            model_path: Model identifier
            device_id: GPU device ID
            pid: Process ID to remove

        Returns:
            True if reference was removed
        """
        key = ReplicaKey(model_id=model_path, device_id=device_id)

        with self._lock:
            if key not in self._replicas:
                logger.warning(f"Attempted to remove ref from unknown replica: {key}")
                return False

            replica_info = self._replicas[key]
            removed = replica_info.remove_pid(pid)

            if removed:
                logger.info(
                    f"Removed reference for PID {pid} from {key}, "
                    f"remaining refs: {replica_info.ref_count}"
                )

                # Update metrics
                MODEL_REF_COUNT.labels(
                    model=key.model_id, device_id=str(key.device_id)
                ).set(replica_info.ref_count)

                # If no more references and not kept for global, mark as evictable
                if replica_info.ref_count == 0 and not replica_info.keep_for_global:
                    logger.info(f"Replica {key} now evictable (no refs)")

            return removed

    def remove_pid_refs(self, pid: int) -> list[ReplicaKey]:
        """Remove all references for a given PID.

        Returns a list of replica keys whose reference count reached zero *after* the
        removal.  This enables the caller to safely decide on eviction without an
        additional unlocked check that could introduce race conditions.
        """
        zero_ref_keys: list[ReplicaKey] = []

        with self._lock:
            for key, replica_info in list(self._replicas.items()):
                if pid in replica_info.pids and replica_info.remove_pid(pid):
                    logger.info(
                        "Removed dead PID %s from %s, remaining refs: %s",
                        pid,
                        key,
                        replica_info.ref_count,
                    )

                    # Update metrics
                    MODEL_REF_COUNT.labels(
                        model=key.model_id, device_id=str(key.device_id)
                    ).set(replica_info.ref_count)

                    if replica_info.ref_count == 0:
                        zero_ref_keys.append(key)

        return zero_ref_keys

    def get_replica_info(
        self, model_path: str, device_id: int
    ) -> ReplicaRefInfo | None:
        """Get replica reference information."""
        key = ReplicaKey(model_id=model_path, device_id=device_id)
        with self._lock:
            return self._replicas.get(key)

    def get_loaded_models(self) -> List[Dict[str, Any]]:
        """Get information about all loaded models."""
        with self._lock:
            return [
                {
                    "model_id": replica_info.key.model_id,
                    "device_id": replica_info.key.device_id,
                    "ref_count": replica_info.ref_count,
                    "pids": list(replica_info.pids),
                    "size_bytes": replica_info.size_bytes,
                    "keep_for_global": replica_info.keep_for_global,
                    "last_access_ts": replica_info.last_access_ts,
                }
                for replica_info in self._replicas.values()
            ]

    def has_pid_refs(self, pid: int) -> bool:
        """Return True if the given PID still owns references to any replica."""
        with self._lock:
            return any(pid in r.pids for r in self._replicas.values())

    # ---------------------------------------------------------------------
    # Eviction API
    # ---------------------------------------------------------------------

    def maybe_evict(self, bytes_needed: int, device_id: int) -> List[ReplicaKey]:
        """Try to evict models to free up memory.

        Args:
            bytes_needed: Bytes needed to be freed
            device_id: GPU device to evict from

        Returns:
            List of replica keys that were evicted
        """
        candidates = self._select_eviction_candidates(bytes_needed, device_id)
        return [
            replica_info.key
            for replica_info in candidates
            if self._evict_replica(replica_info)
        ]

    def periodic_evict(self, device_id: int, bytes_needed: int) -> List[ReplicaKey]:
        """Periodic eviction triggered by lifecycle worker.

        Args:
            device_id: GPU device to evict from
            bytes_needed: Bytes to free up

        Returns:
            List of replica keys that were evicted
        """
        logger.info(
            f"Periodic eviction on device {device_id}, "
            f"need to free {bytes_needed / 1024**3:.2f} GB"
        )
        return self.maybe_evict(bytes_needed, device_id)

    def _select_eviction_candidates(
        self, bytes_needed: int, device_id: int
    ) -> List[ReplicaRefInfo]:
        """Select replicas to evict based on LRU and global cache policy.

        Prioritizes eviction in the following order:
        1. Replicas with ref_count == 0 and keep_for_global == False (local cache)
        2. If global cache usage exceeds `global_cache_fraction`, also include
           keep_for_global == True replicas (oldest first) until the usage is
           brought below the threshold.
        3. Within each group, evict by least-recently-used (oldest `last_access_ts`).
        """
        with self._lock:
            # ------------- Gather runtime information -------------
            # GPU memory stats to figure out device total capacity.
            total_device_mem = None
            try:
                gpu_stats = self.get_gpu_memory_stats()
                if device_id in gpu_stats:
                    total_device_mem = gpu_stats[device_id]["total"]
            except Exception:  # pragma: no cover
                total_device_mem = None

            # Compute current global cache bytes for the device
            global_bytes = sum(
                r.size_bytes
                for r in self._replicas.values()
                if r.key.device_id == device_id and r.keep_for_global
            )

            # Threshold from config (default 0.2 if missing)
            try:
                cfg_fraction = (
                    self._servicer.config.lifecycle.global_cache_fraction
                    if self._servicer.config is not None
                    else 0.2
                )
            except Exception:  # pragma: no cover
                cfg_fraction = 0.2

            # ------------- Build candidate lists -------------
            # Local cache (non-global) replicas without active refs
            local_candidates = [
                r
                for r in self._replicas.values()
                if r.key.device_id == device_id
                and r.ref_count == 0
                and not r.keep_for_global
            ]

            # Global cache replicas without active refs
            global_candidates = [
                r
                for r in self._replicas.values()
                if r.key.device_id == device_id
                and r.ref_count == 0
                and r.keep_for_global
            ]

            # Sort both lists by LRU (oldest first)
            local_candidates.sort(key=lambda r: r.last_access_ts)
            global_candidates.sort(key=lambda r: r.last_access_ts)

            selected: List[ReplicaRefInfo] = []
            freed = 0

            # First, evict from local cache until we meet `bytes_needed`
            for replica in local_candidates:
                selected.append(replica)
                freed += replica.size_bytes
                if freed >= bytes_needed:
                    break

            # If still need bytes OR global cache exceeds limit, include global replicas
            need_more = freed < bytes_needed
            global_over_limit = (
                total_device_mem is not None
                and total_device_mem > 0
                and global_bytes > cfg_fraction * total_device_mem
            )

            if need_more or global_over_limit:
                # Determine additional bytes to free from global cache
                extra_global_bytes_needed = 0
                if global_over_limit and total_device_mem is not None:
                    allowed_global = cfg_fraction * total_device_mem
                    extra_global_bytes_needed = int(
                        max(0, global_bytes - allowed_global)
                    )

                # combined_needed = max(bytes_needed - freed, extra_global_bytes_needed)

                for replica in global_candidates:
                    if replica in selected:
                        continue
                    selected.append(replica)
                    freed += replica.size_bytes
                    if freed >= bytes_needed + extra_global_bytes_needed:
                        break

            logger.info(
                f"Selected {len(selected)} replicas for eviction (freeing ~{freed / 1024**3:.2f} GB) on device {device_id}; "
                f"global cache usage was {global_bytes / 1024**3:.2f} GB / total {(total_device_mem or 0) / 1024**3:.2f} GB"
            )

            return selected

    def _evict_replica(self, replica_info: ReplicaRefInfo) -> bool:
        """Evict a single replica from memory.

        Returns:
            True if eviction was successful
        """
        key = replica_info.key
        model_path = key.model_id

        logger.info(f"Evicting replica {key}")

        try:
            # Determine device type for this replica
            device_type = (
                store_daemon_pb2.DEVICE_TYPE_GPU
                if key.device_id >= 0
                else store_daemon_pb2.DEVICE_TYPE_CPU
            )

            # Unload from checkpoint store
            if self.unload_model(
                model_path,
                device_type=device_type,
                device_id=key.device_id,
            ):
                # unload_model() already removes from tracking and updates metrics
                # Update eviction metric
                EVICTIONS_TOTAL.labels(reason="memory").inc()

                logger.info(f"Successfully evicted replica {key}")
                return True
            else:
                logger.error(f"Failed to evict replica {key}")
                return False

        except Exception:
            logger.exception(f"Error evicting replica {key}")
            return False

    # ---------------------------------------------------------------------
    # GPU Memory Stats
    # ---------------------------------------------------------------------

    def get_gpu_memory_stats(self) -> Dict[int, Dict[str, int]]:
        """Get GPU memory statistics for all devices.

        Returns:
            Dict mapping device_id to memory stats (total, used, free)
        """
        try:
            assert self._checkpoint_store is not None
            raw_stats = self._checkpoint_store.get_gpu_memory_stats()

            # ------------------------------------------------------------------
            # The fake Python CheckpointStore used by the interaction test-suite
            # returns a *mapping* with keys ``total`` / ``used`` / ``free``,
            # whereas the C++ implementation returns a *sequence* of
            # ``(total, free)`` tuples.  Support both shapes here so that the
            # higher-level code does not need to care about the underlying
            # store flavour.
            # ------------------------------------------------------------------
            result = _normalize_gpu_stats(raw_stats)

            # Update cache metrics
            self._update_cache_metrics()

            return result

        except Exception:
            logger.exception("Error getting GPU memory stats")
            return {}

    def _update_cache_metrics(self) -> None:
        """Update GPU cache usage metrics."""
        try:
            local_bytes = 0
            global_bytes = 0

            with self._lock:
                for replica_info in self._replicas.values():
                    if replica_info.key.device_id >= 0:  # GPU replica
                        if replica_info.keep_for_global:
                            global_bytes += replica_info.size_bytes
                        else:
                            local_bytes += replica_info.size_bytes

            GPU_CACHE_BYTES.labels(type="local").set(local_bytes)
            GPU_CACHE_BYTES.labels(type="global").set(global_bytes)

        except Exception:
            logger.warning("Failed to update cache metrics")

    # ---------------------------------------------------------------------
    # Original Confirmation/Registration API (enhanced)
    # ---------------------------------------------------------------------

    def unload_model(
        self,
        model_path: str,
        device_type: store_daemon_pb2.DeviceType = store_daemon_pb2.DEVICE_TYPE_GPU,
        *,
        device_id: int | None = None,
        pid: int | None = None,
    ) -> bool:
        """Unload a model from memory.

        Args:
            model_path: Identifier of the model to unload.
            device_type: DEVICE_TYPE_CPU / DEVICE_TYPE_GPU where the replica resides.
                The special DEVICE_TYPE_DISK is treated as a no-op because no memory
                needs to be released in that case.
            device_id: Numerical device ordinal. If *None*, falls back to a
                best-effort inference based on currently tracked replicas.
                **Callers are encouraged to always pass this argument**.
            pid: Optional process ID whose reference should be released prior
                to the unload attempt.

        Returns:
            True when the model replica was unloaded (or skipped because it
            still has active references); False on hard failures.
        """

        # ------------------------------------------------------------------
        # DISK replicas are persisted on storage and never occupy CPU/GPU
        # memory managed by the daemon.  Unloading therefore becomes a no-op
        # that should always succeed.  We treat this early to avoid the
        # generic path which targets the C++ CheckpointStore and expects a
        # valid in-memory instance.
        # ------------------------------------------------------------------
        if device_type == store_daemon_pb2.DeviceType.DEVICE_TYPE_DISK:
            logger.debug(
                "UnloadModel: received DISK replica for %s – no action required",
                model_path,
            )
            return True

        # Resolve the device ordinal – favour the explicit argument when
        # provided to support multi-GPU setups.
        assert device_id is not None, "device_id must be provided"
        key = ReplicaKey(model_id=model_path, device_id=device_id)

        # Remove reference if PID provided
        if pid is not None:
            self.remove_ref(model_path, device_id, pid)

        # Check if model should be unloaded
        with self._lock:
            replica_info = self._replicas.get(key)
            if replica_info and replica_info.ref_count > 0:
                logger.info(
                    f"Not unloading {key} - still has {replica_info.ref_count} references"
                )
                return True  # Not an error, just skip unloading

        # ------------------------------------------------------------------
        # Deregister replica from Global Store (if applicable) BEFORE unload
        # ------------------------------------------------------------------
        replica_id: str | None = None
        with self._lock:
            info = self._replicas.get(key)
            replica_id = info.replica_id if info else None

        if replica_id and self._servicer.global_store_enabled:
            # Connection manager and stub are guaranteed to be initialised when
            # global_store_enabled is True – remove redundant fallbacks.
            assert self._servicer.connection_manager is not None, (
                "Connection manager must be initialised when global_store_enabled"
            )

            try:
                success = self._servicer.connection_manager.unregister_model_replica(
                    model_id=model_path,
                    replica_id=replica_id,
                    device_id=device_id,
                )
                if success:
                    logger.info(
                        "Unregistered replica %s for %s from Global Store before unload",
                        replica_id,
                        model_path,
                    )
                else:
                    logger.warning(
                        "Failed to unregister replica %s for %s (queued for retry)",
                        replica_id,
                        model_path,
                    )
            except Exception:
                logger.exception(
                    "Failed to unregister replica %s for %s before unload",
                    replica_id,
                    model_path,
                )

        assert self._checkpoint_store is not None
        # Disable remote model access via communication engine (if previously enabled)
        if self._servicer.enable_p2p_engine:
            gs_stub = self._servicer.global_store_stub
            assert gs_stub is not None

            try:
                cpp_location = self._get_cpp_model_location(device_type)
                # Gracefully handle absence of the method in checkpoint_store.
                inst_key = self._make_instance_key(model_path, device_id)
                self._checkpoint_store.disable_remote_instance_access(
                    inst_key, cpp_location
                )

                logger.info("Disabled remote model access for %s", model_path)
            except Exception:
                logger.exception(
                    "Failed to disable remote model access for %s before unload",
                    model_path,
                )

        # Proceed with unloading
        success = False
        for _ in range(self.MAX_RETRIES):
            inst_key = self._make_instance_key(model_path, device_id)
            if self._checkpoint_store.unload_instance(inst_key) == 0:
                logger.info("UnloadModel: success %s", model_path)
                success = True
                break

            time.sleep(self.UNLOAD_RETRY_DELAY_SEC)

        if not success:
            logger.error(
                "UnloadModel failed for model %s after %d retries",
                model_path,
                self.MAX_RETRIES,
            )
            return False

        # Remove from tracking
        with self._lock:
            self._replicas.pop(key, None)
            if model_path in self._model_to_keys:
                self._model_to_keys[model_path].discard(key)
                if not self._model_to_keys[model_path]:
                    del self._model_to_keys[model_path]

        # Record successful unload metric
        MODELS_UNLOADED_TOTAL.labels(
            device_type=get_device_type_label(device_type)
        ).inc()

        return True

    # ---------------------------------------------------------------------
    # Shutdown Support
    # ---------------------------------------------------------------------

    def shutdown_evict_local_replicas(self) -> int:
        """Evict all local (non-global) replicas during shutdown.

        Returns:
            Number of replicas evicted
        """
        evicted_count = 0

        with self._lock:
            # Get all local replicas (keep_for_global=False)
            local_replicas = [
                r for r in self._replicas.values() if not r.keep_for_global
            ]

        logger.info(f"Shutdown: evicting {len(local_replicas)} local replicas")

        for replica in local_replicas:
            if self._evict_replica(replica):
                evicted_count += 1
                # Update shutdown eviction metric
                EVICTIONS_TOTAL.labels(reason="shutdown").inc()

        return evicted_count

    # ---------------------------------------------------------------------
    # Static mapping helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _get_cpp_model_location(
        device_type: store_daemon_pb2.DeviceType,
    ) -> _cs.ModelLocation:
        if device_type == store_daemon_pb2.DEVICE_TYPE_CPU:
            return _cs.ModelLocation(_cs.ModelLocation.CPU)
        if device_type == store_daemon_pb2.DEVICE_TYPE_GPU:
            return _cs.ModelLocation(_cs.ModelLocation.GPU)
        raise ValueError(
            f"Unsupported device type for C++ location mapping: {device_type}"
        )

    # ------------------------------------------------------------------
    # Test-helpers (not used by production code)
    # ------------------------------------------------------------------

    def get_replica_ref_count(self, model_path: str) -> int:
        """Return the cumulative reference count for *model_path* across devices."""
        with self._lock:
            keys = self._model_to_keys.get(model_path, set())
            return sum(self._replicas[k].ref_count for k in keys if k in self._replicas)

    def get_metrics(self) -> Dict[str, Any]:
        """Return a lightweight metrics snapshot compatible with interaction tests."""
        with self._lock:
            loaded_models: Set[str] = {
                k.model_id for k in self._replicas if self._replicas[k].ref_count >= 0
            }
        return {
            "loaded_models": len(loaded_models),
            "models": list(loaded_models),
        }

    def get_device_cache_bytes(self, device_id: int) -> int:
        """Return total GPU memory (in bytes) currently used by the Store Daemon
        on the specified `device_id`.

        This includes both *local* and *global* cache replicas. The value is
        derived from the sizes recorded in the in-memory `ReplicaRefInfo`
        objects and therefore only reflects memory that is actually
        attributable to the Store Daemon, **not** memory that might be used by
        other processes on the same GPU.
        """
        with self._lock:
            return sum(
                r.size_bytes
                for r in self._replicas.values()
                if r.key.device_id == device_id
            )

    @staticmethod
    def _make_instance_key(model_path: str, device_id: int) -> _cs.InstanceKey:
        """Construct an InstanceKey for *model_path* on the given *device_id*."""

        dev_key = _cs.DeviceKey()
        dev_key.type = _cs.DeviceType.GPU if device_id >= 0 else _cs.DeviceType.CPU
        dev_key.ordinal = device_id
        dev_key.uuid = ""

        inst_key = _cs.InstanceKey()
        inst_key.model_id = model_path
        inst_key.device = dev_key
        inst_key.replica = 0
        return inst_key
