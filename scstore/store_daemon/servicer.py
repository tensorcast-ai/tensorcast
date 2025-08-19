#  Copyright (c) 2025, StepCast Team.

"""StoreDaemon gRPC servicer implementation."""

import contextlib
import ctypes
import os
import socket
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError

import grpc

from scstore import _checkpoint_store as _cs
from scstore.logger import init_logger
from scstore.proto import (
    global_store_pb2_grpc,
    store_daemon_pb2,
    store_daemon_pb2_grpc,
)

from .chunk_sync import ChunkSyncWorker
from .config import StoreDaemonConfig
from .connection_manager import GlobalStoreConnectionManager
from .health_check import HealthCheckServer
from .lifecycle_worker import LifecycleWorker
from .metrics import (
    ACTIVE_OPERATIONS,
    ASYNC_LOAD_WAIT_DURATION,
    MEMORY_POOL_AVAILABLE,
    MEMORY_POOL_TOTAL,
    MODEL_VERIFICATION_LATENCY,
    MODEL_VERIFICATION_TOTAL,
    MODELS_ALLOCATED_TOTAL,
    PENDING_LOADS,
    WORKER_HEALTHY,
    WORKER_REGISTERED,
    WORKER_UPTIME_SECONDS,
    get_device_type_label,
)
from .model_loader import LoadResult, ModelLoader
from .process_watcher import ProcessWatcher
from .replica_manager import ReplicaManager
from .utils import AtomicCounter, read_verification_json, resolve_device_id

logger = init_logger(__name__)


class StoreDaemonServicer(store_daemon_pb2_grpc.StoreDaemonServicer):
    """
    StoreDaemonServicer implements the model storage and loading functionality.

    It handles model registration, loading from different sources (remote, disk),
    and manages model lifecycle in both CPU and GPU memory.
    """

    def __init__(
        self,
        config: StoreDaemonConfig,
    ):
        """
        Initialize the StoreDaemonServicer.

        Args:
            config: StoreDaemonConfig instance containing all configuration parameters
        """

        # Store configuration
        self.config = config

        # Extract commonly used values for convenience
        storage_path = config.server.storage_path
        mem_pool_size = config.server.mem_pool_size
        num_thread = config.server.num_threads
        chunk_size = config.server.chunk_size
        enable_p2p_access = config.server.enable_p2p_access
        enable_p2p_engine = config.server.enable_p2p_engine
        global_store_address = config.global_store_address
        local_p2p_port = config.network.p2p_port
        local_grpc_port = config.server.port
        enable_rdma = config.server.enable_rdma
        pinned_memory_timeout_ms = config.server.pinned_memory_timeout_ms

        self._validate_init_params(
            str(storage_path), mem_pool_size, num_thread, chunk_size
        )

        # Configuration
        self.enable_p2p_access = enable_p2p_access
        self.enable_p2p_engine = enable_p2p_engine
        self.global_store_enabled = bool(global_store_address)
        self.global_store_address = global_store_address
        self.enable_rdma = enable_rdma

        # Make storage path available for helper routines (e.g. verification)
        self.storage_path = storage_path

        # State
        self.start_time = time.time()
        self._active_chunk_locks = {}  # Track active chunk locks for P2P transfers
        self.shutting_down = False
        self.worker_id: str | None = None
        self.active_operations = AtomicCounter()

        # Attributes initialised later in __init__; avoid Optional types when not necessary
        # Attributes initialised later in __init__; avoid Optional types when not necessary
        self._load_executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=num_thread if num_thread and num_thread > 0 else 4,
            thread_name_prefix="model-loader",
        )

        # Track pending async loads: key = (model_path, replica_uuid), value = Future
        self._pending_loads: dict[tuple[str, str], Future] = {}
        self._pending_loads_lock = threading.Lock()

        # ------------------------------------------------------------------
        # Verification task handling.  Each verification runs in the background
        # using a dedicated thread-pool so that expensive hashing does not
        # block either the gRPC worker threads or the model-loading executor.
        # ------------------------------------------------------------------

        self._verification_executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=max(2, num_thread // 2),
            thread_name_prefix="model-verifier",
        )

        # Map (model_path, replica_uuid) -> (VerificationStatus, err_msg)
        self._verification_results: dict[
            tuple[str, str], tuple[store_daemon_pb2.VerificationStatus, str]
        ] = {}

        # Lock protecting _verification_results to ensure thread-safety
        self._verification_lock = threading.Lock()

        # Initialize network info
        self._initialize_network_info(local_p2p_port, local_grpc_port)

        # Load the checkpoint store C++ extension
        try:
            # torch is required for the checkpoint store C++ extension
            import torch  # noqa: F401

            ctypes.CDLL(os.path.join(os.path.dirname(__file__), "../lib/libscstore.so"))
            import scstore._checkpoint_store as _cs
        except OSError as e:
            logger.error(f"Failed to load C++ extension: {e}")
            raise

        self.pinned_memory_timeout_ms = pinned_memory_timeout_ms

        # Initialize checkpoint store
        logger.info(
            f"StorageServicer: storage_path={storage_path}, "
            f"mem_pool_size={mem_pool_size}, num_thread={num_thread}, "
            f"chunk_size={chunk_size}, "
            f"enable_p2p_access={enable_p2p_access}, "
            f"enable_p2p_engine={self.enable_p2p_engine}, "
            f"global_store_enabled={self.global_store_enabled}, "
            f"pinned_memory_timeout_ms={pinned_memory_timeout_ms}"
        )

        assert _cs is not None
        # (Phase-3) Create a single CommunicationManager instance when
        # communication is enabled, then inject it into CheckpointStore so
        # that multiple stores (if any) share the same engine.

        comm_manager_obj = None
        if self.enable_p2p_engine:
            try:
                comm_manager_obj = _cs.CommunicationManager(
                    "0.0.0.0", local_p2p_port, self.enable_rdma
                )
                logger.info(
                    f"CommunicationManager initialized with enable_rdma={self.enable_rdma}, p2p_port={local_p2p_port}"
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Failed to initialise CommunicationManager; falling back to internal engine"
                )

        cs_cfg: dict[str, object] = {
            "storage_path": str(storage_path),
            "memory_pool_size": mem_pool_size,
            "num_thread": 2,  # TODO: align with shared thread pool
            "chunk_size": chunk_size,
            "pinned_memory_timeout_ms": pinned_memory_timeout_ms,
            "p2p_port": local_p2p_port,
        }

        if self.global_store_enabled and self.global_store_address:
            cs_cfg["global_store_address"] = self.global_store_address

        if comm_manager_obj is not None:
            cs_cfg["comm_manager"] = comm_manager_obj

        self.checkpoint_store = _cs.create_checkpoint_store(cs_cfg)

        # Initialize global store connection
        self.global_store_stub: global_store_pb2_grpc.GlobalModelStoreStub | None = None
        self.grpc_channel: grpc.Channel | None = None  # Declared once here
        self.connection_manager: GlobalStoreConnectionManager | None = None

        if self.global_store_enabled:
            self._initialize_high_availability_connection()

        # Initialize components
        self.replica_manager: ReplicaManager = ReplicaManager(self)
        self.model_loader = ModelLoader(self)
        self.health_check_server: HealthCheckServer | None = None
        self.worker_manager: None = None  # Deprecated

        # Initialize process watcher for PID lifecycle tracking
        self.process_watcher: ProcessWatcher = ProcessWatcher(
            check_interval_seconds=config.lifecycle.proc_check_interval_s
            if config
            else 5.0,
            on_pid_dead=self._on_pid_dead,
        )

        # Initialize chunk_sync_worker - this is a required component
        # If global store is not enabled, create a no-op worker
        sync_interval = 10
        if self.global_store_enabled and self.global_store_address:
            # Will be properly initialized in _initialize_high_availability_connection
            # Create a placeholder for now that will be replaced
            self.chunk_sync_worker = ChunkSyncWorker(
                servicer=self,
                global_store_address=self.global_store_address,
                sync_interval_seconds=sync_interval,
            )
        else:
            # Create a no-op worker when global store is disabled
            self.chunk_sync_worker = ChunkSyncWorker(
                servicer=self,
                global_store_address="",  # Empty address for no-op mode
                sync_interval_seconds=sync_interval,
            )

        self.lifecycle_worker: LifecycleWorker = LifecycleWorker(
            process_watcher=self.process_watcher,
            replica_manager=self.replica_manager,
            eviction_check_interval_seconds=config.lifecycle.eviction_check_interval_s,
            gpu_memory_limit_fraction=config.lifecycle.gpu_memory_limit_fraction,
            chunk_sync_worker=self.chunk_sync_worker,
        )

        # Initialize metrics
        self._initialize_metrics()

        # Start health check server
        if config and config.network.health_check_port:
            self._start_health_check_server(config.network.health_check_port)

        # Start lifecycle worker if initialized
        if self.lifecycle_worker:
            self.lifecycle_worker.start()
            logger.info("Started lifecycle worker for eviction management")

    def _initialize_high_availability_connection(self):
        """Initialize high availability connection to Global Store (thread-based)."""
        try:
            # Pass the HighAvailabilityConfig object directly for type safety
            ha_config = self.config.high_availability

            # Initialise connection manager with strongly-typed config
            self.connection_manager = GlobalStoreConnectionManager(
                servicer=self,
                global_store_address=self.global_store_address or "",
                ha_config=ha_config,
            )

            # Start connection manager (spawns its own background threads)
            self.connection_manager.start()

            logger.info("High availability connection manager initialised and started")

        except Exception as e:
            logger.error(f"Failed to initialise HA connection manager: {e}")

    def _connect_to_global_store(self) -> None:
        """Connect to global store if address is provided (legacy method)."""
        if not self.global_store_address:
            logger.warning("Attempted to connect to global store without an address.")
            return

        try:
            self.grpc_channel = grpc.insecure_channel(self.global_store_address)
            self.global_store_stub = global_store_pb2_grpc.GlobalModelStoreStub(
                self.grpc_channel
            )
            logger.info(f"Connected to GlobalStore at {self.global_store_address}")

            # Re-initialize chunk_sync_worker with proper connection
            # Stop the placeholder worker first
            if self.chunk_sync_worker:
                self.chunk_sync_worker.stop()

            sync_interval = 10
            self.chunk_sync_worker = ChunkSyncWorker(
                servicer=self,
                global_store_address=self.global_store_address,
                sync_interval_seconds=sync_interval,
            )

            # Update lifecycle worker with the new chunk_sync_worker
            self.lifecycle_worker.chunk_sync_worker = self.chunk_sync_worker

        except Exception as e:
            logger.exception(
                f"Failed to connect to GlobalStore at {self.global_store_address}: {e}",
            )
            self.global_store_stub = None

    def _validate_init_params(
        self, storage_path: str, mem_pool_size: int, num_thread: int, chunk_size: int
    ) -> None:
        """Validate initialization parameters."""

        if mem_pool_size <= 0:
            logger.error("mem_pool_size must be greater than 0")
            raise ValueError("Invalid mem_pool_size")

        if num_thread <= 0:
            logger.error("num_thread must be greater than 0")
            raise ValueError("Invalid num_thread")

        if chunk_size <= 0:
            logger.error("chunk_size must be greater than 0")
            raise ValueError("Invalid chunk_size")

    def _initialize_network_info(
        self, local_p2p_port: int, local_grpc_port: int
    ) -> None:
        """Initialize network information for remote(RDMA/TCP) communications."""
        # Set default values
        self.node_id = "unknown"
        self.node_address = "unknown"
        self.node_port = local_p2p_port
        self.grpc_port = local_grpc_port

        try:
            hostname = socket.gethostname()
            node_address = socket.gethostbyname(hostname)

            self.node_id = hostname
            self.node_address = node_address

            logger.info(
                f"Using network interface: {self.node_address}:{self.node_port} (remote), "
                f"{self.node_address}:{self.grpc_port} (gRPC)"
            )
        except Exception as e:
            logger.error(f"Failed to initialize network info: {e}")
            logger.info(
                f"Using fallback values: {self.node_address}:{self.node_port} (remote), "
                f"{self.node_address}:{self.grpc_port} (gRPC)"
            )
            self.node_id = "localhost"
            self.node_address = "127.0.0.1"

    def _initialize_metrics(self):
        """Initialize Prometheus metrics."""
        try:
            # Set static metrics
            if self.checkpoint_store:
                MEMORY_POOL_TOTAL.set(self.checkpoint_store.get_mem_pool_size())
            WORKER_HEALTHY.set(1)  # Initial state is healthy
            WORKER_REGISTERED.set(0)  # Initial unregistered

            metrics_port = self.config.network.metrics_port if self.config else 9091

            logger.info(
                f"Prometheus metrics initialized at {self.node_address}:{metrics_port}"
            )

            # ------------------------------------------------------------------
            # Register C++ global metrics collector so that its metrics are
            # exported alongside the existing Python metrics.
            # ------------------------------------------------------------------
            try:
                from prometheus_client import REGISTRY

                from .ckpt_collector import GlobalMetricsCollector

                REGISTRY.register(GlobalMetricsCollector())  # pyright: ignore[reportArgumentType]
                logger.info("Global C++ metrics collector registered")
            except ValueError:
                # Collector may already be registered when running unit
                # tests which instantiate multiple servicers in the same
                # process.  Ignore duplicate registration errors.
                logger.debug("Global metrics collector already registered")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Failed to register global metrics collector: {exc}")
        except Exception as e:
            logger.error(f"Failed to initialize metrics: {e}")

    def _start_health_check_server(self, port: int):
        """Start health check server."""
        try:
            self.health_check_server = HealthCheckServer(self, port)
            self.health_check_server.start()
        except Exception as e:
            logger.error(f"Failed to start health check server: {e}")

    def _update_runtime_metrics(self):
        """Update runtime metrics."""
        try:
            # Update memory usage
            if self.checkpoint_store:
                MEMORY_POOL_AVAILABLE.set(self.checkpoint_store.get_available_memory())

            # Update active operations
            ACTIVE_OPERATIONS.set(self.active_operations.get())

            # Update uptime
            WORKER_UPTIME_SECONDS.set(int(time.time() - self.start_time))

            # Update health status
            WORKER_HEALTHY.set(0 if self.shutting_down else 1)

            # Update registration status
            WORKER_REGISTERED.set(1 if self.worker_id else 0)

        except Exception as e:
            logger.warning(f"Failed to update runtime metrics: {e}")

    def _on_pid_dead(self, pid: int) -> None:
        """Callback when a process is detected as dead."""
        logger.info(f"Process {pid} detected as dead, removing references")
        zero_ref_keys = self.replica_manager.remove_pid_refs(pid)

        # Remove PID from process watcher (safe - PID already dead)
        self.process_watcher.remove_pid(pid)

        if zero_ref_keys:
            logger.info(
                "Marked %s replicas as evictable due to dead PID %s",
                len(zero_ref_keys),
                pid,
            )
            # Note: Don't immediately evict - let the lifecycle worker handle eviction
            # based on memory pressure and eviction policies

    def __del__(self) -> None:
        """Clean up resources when the object is destroyed."""
        # Check if grpc_channel exists and close it
        if self.grpc_channel is not None:
            try:
                self.grpc_channel.close()
            except Exception as e:
                logger.error(f"Error closing gRPC channel: {e}")

        # Check if _load_executor exists and shutdown
        if self._load_executor is not None:
            try:
                self._load_executor.shutdown(wait=False)
            except Exception as e:
                logger.error(f"Error shutting down load executor: {e}")

        # Check if _verification_executor exists and shutdown
        if self._verification_executor is not None:
            try:
                self._verification_executor.shutdown(wait=False)
            except Exception as e:
                logger.error(f"Error shutting down verification executor: {e}")

    # ========== High Availability Support Methods ==========

    def get_connection_status(self) -> dict:
        """Return current connection status from the connection manager."""
        if self.connection_manager:
            return self.connection_manager.get_connection_status()
        return {
            "state": "legacy",
            "connected": bool(self.global_store_stub),
            "local_state_version": 0,
            "last_successful_sync": 0,
            "registered_models_count": 0,
            "pending_changes": False,
        }

    # gRPC service methods

    def LoadModel(
        self, request: store_daemon_pb2.LoadModelRequest, context: grpc.ServicerContext
    ) -> store_daemon_pb2.LoadModelResponse:
        """Load a model into memory (async version)."""
        # Increment active operations
        self.active_operations.increment()
        self._update_runtime_metrics()

        try:
            # Extract PID and track it if provided
            self.process_watcher.add_pid(request.pid)

            # Use async loading - returns immediately after allocation
            allocation_success, returned_mem_handle, loading_future = (
                self.model_loader.start_async_load(request, context)
            )

            if not allocation_success:
                # Memory allocation failed
                return store_daemon_pb2.LoadModelResponse(
                    status=store_daemon_pb2.LoadModelStatus.LOAD_MODEL_STATUS_FAILED
                )

            assert returned_mem_handle is not None

            # Store the loading future for later confirmation
            load_key = (request.model_path, request.replica_uuid)
            with self._pending_loads_lock:
                self._pending_loads[load_key] = loading_future

            # ---- New unified reference tracking ---------------------------------
            # Immediately record a reference for the requesting PID after the
            # memory allocation succeeds.  This guarantees that lifecycle and
            # eviction logic can correctly account for the (still loading)
            # replica.  Any subsequent load-/verification-failure will
            # automatically drop this reference in *cleanup_pending_load*.
            try:
                device_id = resolve_device_id(request.device_uuid, default=0)
                self.replica_manager.add_ref(
                    model_path=request.model_path,
                    device_id=device_id,
                    pid=request.pid,
                    size_bytes=request.size_bytes,
                    keep_for_global=request.keep_for_global,
                )
            except Exception:
                logger.exception(
                    "Failed to add initial reference for model %s (pid=%s)",
                    request.model_path,
                    request.pid,
                )
            # --------------------------------------------------------------------

            # Update metrics
            MODELS_ALLOCATED_TOTAL.labels(
                device_type=get_device_type_label(request.target_device_type)
            ).inc()
            PENDING_LOADS.inc()

            # Register cleanup callback
            def cleanup_pending_load(fut):
                """Remove from pending loads when complete."""
                with self._pending_loads_lock:
                    self._pending_loads.pop(load_key, None)
                # Update pending loads metric
                PENDING_LOADS.dec()
                # Check if load failed and update verification results
                try:
                    result = fut.result()
                    load_result = LoadResult.from_value(result)
                    if not load_result.success:
                        # On failure drop the previously added reference to keep
                        # reference counts consistent.
                        try:
                            device_id_local = resolve_device_id(
                                request.device_uuid, default=0
                            )
                            self.replica_manager.remove_ref(
                                model_path=request.model_path,
                                device_id=device_id_local,
                                pid=request.pid,
                            )
                        except Exception:
                            logger.exception(
                                "Failed to remove reference for %s after load failure",
                                request.model_path,
                            )
                        with self._verification_lock:
                            self._verification_results[load_key] = (
                                store_daemon_pb2.VerificationStatus.VERIFICATION_STATUS_FAILED,
                                "Model loading failed",
                            )
                except Exception as e:
                    logger.exception(f"Exception in loading future for {load_key}: {e}")
                    with self._verification_lock:
                        self._verification_results[load_key] = (
                            store_daemon_pb2.VerificationStatus.VERIFICATION_STATUS_FAILED,
                            str(e),
                        )

            loading_future.add_done_callback(cleanup_pending_load)

            # Return success with ALLOCATED status
            response = store_daemon_pb2.LoadModelResponse(
                model_path=request.model_path,
                status=store_daemon_pb2.LoadModelStatus.LOAD_MODEL_STATUS_ALLOCATED,
            )

            response.mem_handle.CopyFrom(
                store_daemon_pb2.MemCopyHandle(cuda_ipc_handle=returned_mem_handle)
            )

            return response

        finally:
            # Decrement active operations
            self.active_operations.decrement()
            self._update_runtime_metrics()

    def ConfirmModel(
        self,
        request: store_daemon_pb2.ConfirmModelRequest,
        context: grpc.ServicerContext,
    ) -> store_daemon_pb2.ConfirmModelResponse:
        """Confirm that a model has been successfully loaded into memory."""
        model_path = request.model_path
        replica_uuid = request.replica_uuid
        device_type = request.target_device_type

        # Validate request parameters
        if not model_path:
            logger.error("model_path is empty in ConfirmModel request")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return store_daemon_pb2.ConfirmModelResponse(model_path="", code=1)

        # Only GPU device type is supported
        if device_type != store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU:
            logger.error(
                f"Unsupported device type {device_type} in ConfirmModel request. Only GPU is supported."
            )
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            return store_daemon_pb2.ConfirmModelResponse(model_path=model_path, code=1)

        # Check if there's a pending load future
        load_key = (model_path, replica_uuid)
        loading_future = None

        with self._pending_loads_lock:
            loading_future = self._pending_loads.get(load_key)

        if loading_future:
            # Wait for the loading to complete
            wait_start = time.time()
            try:
                # Wait with a reasonable timeout (e.g., 30 seconds)
                result = loading_future.result(timeout=30.0)

                # Record wait duration
                wait_duration = time.time() - wait_start
                ASYNC_LOAD_WAIT_DURATION.labels(
                    device_type=get_device_type_label(device_type)
                ).observe(wait_duration)

                # Check the result
                load_result = LoadResult.from_value(result)

                if not load_result.success:
                    logger.error(f"Model loading failed for {model_path}")
                    context.set_code(grpc.StatusCode.INTERNAL)
                    return store_daemon_pb2.ConfirmModelResponse(
                        model_path=model_path, code=1
                    )

            except TimeoutError:
                logger.error(f"ConfirmModel timed out waiting for {model_path} to load")
                context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
                return store_daemon_pb2.ConfirmModelResponse(
                    model_path=model_path, code=1
                )
            except Exception as e:
                logger.exception(
                    f"Exception while waiting for {model_path} to load: {e}"
                )
                context.set_code(grpc.StatusCode.INTERNAL)
                return store_daemon_pb2.ConfirmModelResponse(
                    model_path=model_path, code=1
                )

        # For GPU loads, still need to confirm and register
        # Registration is now fully automatic and tied to successful model
        # verification.  *ConfirmModel* therefore only serves as a convenient
        # blocking call for clients that need to wait until the asynchronous
        # load step has completed.

        return store_daemon_pb2.ConfirmModelResponse(model_path=model_path, code=0)

    def UnloadModel(
        self,
        request: store_daemon_pb2.UnloadModelRequest,
        context: grpc.ServicerContext,
    ) -> store_daemon_pb2.UnloadModelResponse:
        """Unload a model from memory."""
        model_path = request.model_path
        device_type = request.target_device_type

        if not model_path:
            logger.error("model_path is empty in UnloadModel request")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return store_daemon_pb2.UnloadModelResponse(model_path="", code=1)

        # Extract optional fields
        pid = request.pid

        # Delegate to replica manager with PID tracking
        success = self.replica_manager.unload_model(
            model_path=model_path, device_type=device_type, pid=pid
        )

        # NOTE: We intentionally keep the PID under watch even if it currently holds
        # zero model references.  This avoids racy situations where another thread
        # may concurrently LoadModel for the same PID between the ref-count check
        # and the removal call.  The watcher will automatically stop monitoring
        # the PID once the process actually exits (detected in _on_pid_dead).

        if not success:
            context.set_code(grpc.StatusCode.INTERNAL)
            return store_daemon_pb2.UnloadModelResponse(model_path=model_path, code=1)

        # Update runtime metrics
        self._update_runtime_metrics()

        # Reset any previous error code on the context – treated as success.
        with contextlib.suppress(Exception):
            context.set_code(grpc.StatusCode.OK)

        return store_daemon_pb2.UnloadModelResponse(model_path=model_path, code=0)

    def ClearMem(
        self, request: store_daemon_pb2.ClearMemRequest, context: grpc.ServicerContext
    ) -> store_daemon_pb2.ClearMemResponse:
        """Clear all memory used by the storage daemon."""
        if not self.checkpoint_store:
            logger.error("Checkpoint store not initialized")
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            return store_daemon_pb2.ClearMemResponse()

        ret = self.checkpoint_store.clear_mem()
        if ret != 0:
            logger.error(f"ClearMem failed with error code {ret}")
            context.set_code(grpc.StatusCode.INTERNAL)
        else:
            logger.info("ClearMem: success")

        # Clear any stale error codes from the gRPC context.
        with contextlib.suppress(Exception):
            context.set_code(grpc.StatusCode.OK)

        return store_daemon_pb2.ClearMemResponse()

    def GetServerConfig(
        self,
        request: store_daemon_pb2.GetServerConfigRequest,
        context: grpc.ServicerContext,
    ) -> store_daemon_pb2.GetServerConfigResponse:
        """Get server configuration."""
        if not self.checkpoint_store:
            return store_daemon_pb2.GetServerConfigResponse(
                mem_pool_size=0,
                chunk_size=0,
            )

        return store_daemon_pb2.GetServerConfigResponse(
            mem_pool_size=self.checkpoint_store.get_mem_pool_size(),
            chunk_size=self.checkpoint_store.get_chunk_size(),
        )

    def GetWorkerStatus(
        self,
        request: store_daemon_pb2.GetWorkerStatusRequest,
        context: grpc.ServicerContext,
    ) -> store_daemon_pb2.GetWorkerStatusResponse:
        """Get worker status."""
        try:
            uptime_seconds = int(time.time() - self.start_time)

            mem_pool_total_size = 0
            mem_pool_available_size = 0
            if self.checkpoint_store:
                mem_pool_total_size = self.checkpoint_store.get_mem_pool_size()
                mem_pool_available_size = self.checkpoint_store.get_available_memory()

            return store_daemon_pb2.GetWorkerStatusResponse(
                is_registered=bool(self.worker_id),
                is_healthy=not self.shutting_down,
                is_shutting_down=self.shutting_down,
                mem_pool_total_size=mem_pool_total_size,
                mem_pool_available_size=mem_pool_available_size,
                uptime_seconds=uptime_seconds,
                worker_id=self.worker_id or "",
            )
        except Exception as e:
            logger.exception(f"Error getting worker status: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to get worker status: {str(e)}")
            return store_daemon_pb2.GetWorkerStatusResponse()

    def GetLoadedModels(
        self,
        request: store_daemon_pb2.GetLoadedModelsRequest,
        context: grpc.ServicerContext,
    ) -> store_daemon_pb2.GetLoadedModelsResponse:
        """Get information about all loaded models with reference tracking."""
        try:
            # Get loaded models from replica manager
            loaded_models = self.replica_manager.get_loaded_models()

            # Apply filters if provided
            if request.HasField("model_id_filter") and request.model_id_filter:
                loaded_models = [
                    m for m in loaded_models if request.model_id_filter in m["model_id"]
                ]

            if request.HasField("device_id_filter"):
                loaded_models = [
                    m
                    for m in loaded_models
                    if m["device_id"] == request.device_id_filter
                ]

            # Convert to proto format
            proto_models = []
            total_size = 0

            for model in loaded_models:
                proto_model = store_daemon_pb2.LoadedModelInfo(
                    model_id=model["model_id"],
                    device_id=model["device_id"],
                    ref_count=model["ref_count"],
                    pids=model["pids"],
                    size_bytes=model["size_bytes"],
                    keep_for_global=model["keep_for_global"],
                    last_access_timestamp=int(model["last_access_ts"]),
                )
                proto_models.append(proto_model)
                total_size += model["size_bytes"]

            return store_daemon_pb2.GetLoadedModelsResponse(
                models=proto_models,
                total_models=len(proto_models),
                total_size_bytes=total_size,
            )

        except Exception as e:
            logger.exception(f"Error getting loaded models: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to get loaded models: {str(e)}")
            return store_daemon_pb2.GetLoadedModelsResponse()

    # ========== Memory TensorDict Registration RPCs ==========

    def BeginRegisterTensorDict(
        self,
        request: store_daemon_pb2.BeginRegisterTensorDictRequest,
        context: grpc.ServicerContext,
    ) -> store_daemon_pb2.BeginRegisterTensorDictResponse:
        try:
            reg: _cs.CheckpointStore.TensorDictRegistration = {
                "model_id": request.model_id,
                "tensor_index_key": request.tensor_index_key
                if request.WhichOneof("index") == "tensor_index_key"
                else "",
                "tensor_index_data": request.tensor_index_data.data.decode("utf-8")
                if request.WhichOneof("index") == "tensor_index_data"
                else None,
                "schema_version": request.tensor_index_data.schema_version
                if request.WhichOneof("index") == "tensor_index_data"
                else "v2",
                "encoding": request.tensor_index_data.encoding
                if request.WhichOneof("index") == "tensor_index_data"
                else "json",
                "device_id": request.device_id,
                "total_size_bytes": int(request.total_size),
                "enable_p2p": request.enable_p2p,
                "ttl_ms": int(request.ttl_ms) if request.HasField("ttl_ms") else 0,
            }

            result = self.checkpoint_store.begin_register_tensor_dict(reg)

            resp = store_daemon_pb2.BeginRegisterTensorDictResponse(
                registration_id=result["registration_id"],
                device_id=result["device_id"],
                size=result["size_bytes"],
            )
            resp.daemon_ipc_handle = result["daemon_ipc_handle"]
            return resp
        except ValueError as e:
            logger.exception(
                "BeginRegisterTensorDict failed with invalid argument: %s", e
            )
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return store_daemon_pb2.BeginRegisterTensorDictResponse()
        except MemoryError as e:
            logger.exception("BeginRegisterTensorDict failed with memory error: %s", e)
            context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
            context.set_details(str(e))
            return store_daemon_pb2.BeginRegisterTensorDictResponse()
        except Exception as e:  # noqa: BLE001
            logger.exception("BeginRegisterTensorDict failed: %s", e)
            # Check for specific error messages in the exception string
            error_msg = str(e).lower()
            if "not found" in error_msg:
                context.set_code(grpc.StatusCode.NOT_FOUND)
            elif "invalid" in error_msg or "argument" in error_msg:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            elif "memory" in error_msg or "resource" in error_msg:
                context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
            elif (
                "deadline" in error_msg or "timeout" in error_msg or "ttl" in error_msg
            ):
                context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
            else:
                context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return store_daemon_pb2.BeginRegisterTensorDictResponse()

    def CommitRegisteredTensorDict(
        self,
        request: store_daemon_pb2.CommitRegisteredTensorDictRequest,
        context: grpc.ServicerContext,
    ) -> store_daemon_pb2.CommitRegisteredTensorDictResponse:
        try:
            result = self.checkpoint_store.commit_registered_tensor_dict(
                request.registration_id
            )
            # Build descriptor from returned dict
            desc = store_daemon_pb2.ModelDescriptor(
                model_id=str(result["model_id"]),
                index_multihash=str(result.get("index_multihash", "")),
                data_multihash=str(result.get("data_multihash", "")),
                schema_version=str(result.get("schema_version", "")),
                encoding=str(result.get("encoding", "")),
                total_size=int(result.get("size_bytes", 0)),
            )
            return store_daemon_pb2.CommitRegisteredTensorDictResponse(
                registration_id=str(result["registration_id"]),
                model_id=str(result["model_id"]),
                device_id=int(result["device_id"]),
                size=int(result["size_bytes"]),
                descriptor=desc,
            )
        except ValueError as e:
            logger.exception(
                "CommitRegisteredTensorDict failed with invalid argument: %s", e
            )
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return store_daemon_pb2.CommitRegisteredTensorDictResponse()
        except Exception as e:  # noqa: BLE001
            logger.exception("CommitRegisteredTensorDict failed: %s", e)
            # Check for specific error messages in the exception string
            error_msg = str(e).lower()
            if "not found" in error_msg:
                context.set_code(grpc.StatusCode.NOT_FOUND)
            elif (
                "deadline" in error_msg or "expired" in error_msg or "ttl" in error_msg
            ):
                context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
            elif "invalid" in error_msg:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            else:
                context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return store_daemon_pb2.CommitRegisteredTensorDictResponse()

    def AbortRegisteredTensorDict(
        self,
        request: store_daemon_pb2.AbortRegisteredTensorDictRequest,
        context: grpc.ServicerContext,
    ) -> store_daemon_pb2.AbortRegisteredTensorDictResponse:
        try:
            self.checkpoint_store.abort_registered_tensor_dict(request.registration_id)
            return store_daemon_pb2.AbortRegisteredTensorDictResponse(ok=True)
        except ValueError as e:
            logger.exception(
                "AbortRegisteredTensorDict failed with invalid argument: %s", e
            )
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return store_daemon_pb2.AbortRegisteredTensorDictResponse(ok=False)
        except Exception as e:  # noqa: BLE001
            logger.exception("AbortRegisteredTensorDict failed: %s", e)
            # Check for specific error messages in the exception string
            error_msg = str(e).lower()
            if "not found" in error_msg:
                context.set_code(grpc.StatusCode.NOT_FOUND)
            elif "invalid" in error_msg:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            else:
                context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return store_daemon_pb2.AbortRegisteredTensorDictResponse(ok=False)

    def LockTransportChunks(
        self,
        request: store_daemon_pb2.LockChunksRequest,
        context: grpc.ServicerContext,
    ) -> store_daemon_pb2.LockChunksResponse:
        """Lock chunks for P2P transport to prevent concurrent eviction."""
        import uuid

        try:
            # Convert chunk indices to list
            indices = list(request.chunk_indices)

            # Call checkpoint store to lock chunks

            dev = _cs.DeviceKey()
            dev.type = _cs.DeviceType.NONE
            dev.ordinal = -1
            dev.uuid = ""
            inst_key = _cs.InstanceKey()
            inst_key.model_id = request.model_id
            inst_key.device = dev
            inst_key.replica = 0

            status = self.checkpoint_store.lock_chunks(inst_key, indices)

            if status != 0:
                context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
                context.set_details(
                    f"Failed to lock chunks for model {request.model_id}"
                )
                return store_daemon_pb2.LockChunksResponse()

            # Generate a unique lock token
            lock_token = str(uuid.uuid4())

            # Store the lock information for later unlock
            if not hasattr(self, "_active_chunk_locks"):
                self._active_chunk_locks = {}
            self._active_chunk_locks[lock_token] = (inst_key, indices)

            logger.info(
                f"Locked {len(indices)} chunks for model {request.model_id}, token: {lock_token}"
            )

            return store_daemon_pb2.LockChunksResponse(lock_token=lock_token)

        except Exception as e:
            logger.exception(f"Error locking chunks: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to lock chunks: {str(e)}")
            return store_daemon_pb2.LockChunksResponse()

    def UnlockTransportChunks(
        self,
        request: store_daemon_pb2.UnlockChunksRequest,
        context: grpc.ServicerContext,
    ) -> store_daemon_pb2.UnlockChunksResponse:
        """Unlock chunks after P2P transport completion."""
        try:
            if not hasattr(self, "_active_chunk_locks"):
                self._active_chunk_locks = {}

            # Retrieve lock information
            lock_info = self._active_chunk_locks.pop(request.lock_token, None)

            if lock_info is None:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Lock token not found: {request.lock_token}")
                return store_daemon_pb2.UnlockChunksResponse()

            inst_key, indices = lock_info

            # Call checkpoint store to unlock chunks
            # Note: We set copied_gpu=False here as this is called by Global Store
            # The actual GPU copy status will be updated by the target StoreDaemon
            status = self.checkpoint_store.unlock_chunks(inst_key, indices, False)

            if status != 0:
                logger.warning(
                    f"Failed to unlock chunks for model {inst_key.model_id}, token: {request.lock_token}"
                )
            else:
                logger.info(
                    f"Unlocked {len(indices)} chunks for model {inst_key.model_id}, token: {request.lock_token}"
                )

            return store_daemon_pb2.UnlockChunksResponse()

        except Exception as e:
            logger.exception(f"Error unlocking chunks: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to unlock chunks: {str(e)}")
            return store_daemon_pb2.UnlockChunksResponse()

    def graceful_shutdown(self):
        """Initiate graceful shutdown."""
        self.shutting_down = True

        # Stop lifecycle worker first (which stops process watcher)
        self.lifecycle_worker.stop()

        # Stop chunk sync worker
        self.chunk_sync_worker.stop()

        # Evict local replicas during shutdown
        if self.replica_manager is not None:
            evicted_count = self.replica_manager.shutdown_evict_local_replicas()
            logger.info(f"Evicted {evicted_count} local replicas during shutdown")

        # Stop connection manager gracefully
        if self.connection_manager:
            try:
                self.connection_manager.stop()
            except Exception:
                logger.debug("Failed to stop connection manager during shutdown")

        # Shutdown auxiliary services
        if self.health_check_server:
            self.health_check_server.stop()

        self.model_loader.shutdown()
        logger.info("Graceful shutdown completed")

    # ------------------------------------------------------------------
    # Verification helpers
    # ------------------------------------------------------------------

    def schedule_verification(
        self,
        *,
        model_path: str,
        replica_uuid: str,
        device_uuid: str,
        cuda_ptr: int,
        pid: int,
        keep_for_global: bool = False,
        size_bytes: int = 0,
    ) -> None:
        """Handle reference counting and (optionally) schedule data verification.

        This helper now solely schedules data integrity verification.  Reference
        accounting is performed directly in *LoadModel* to guarantee a single
        authoritative code-path.
        """

        # Resolve device ID early (best-effort)
        device_id = resolve_device_id(device_uuid, default=0)

        assert pid > 0, "pid must be provided"

        # ------- Verification scheduling (mandatory) -------
        verification_path = os.path.join(
            self.storage_path, model_path, "verification.json"
        )
        expected_info = read_verification_json(verification_path)
        memory_size = int(expected_info.get("model_size", 0))

        # Guard – require valid pointer and expected_info to proceed.
        if not (cuda_ptr and expected_info):
            return

        if not replica_uuid:
            return  # cannot track without unique identifier

        key = (model_path, replica_uuid)
        with self._verification_lock:
            self._verification_results[key] = (
                store_daemon_pb2.VerificationStatus.VERIFICATION_STATUS_IN_PROGRESS,
                "",
            )

        future: Future = self._verification_executor.submit(
            self._perform_verification,
            device_id,
            cuda_ptr,
            memory_size,
            expected_info,
        )
        future.add_done_callback(
            lambda f, k=key: self._verification_done_callback(f, k)
        )

    def _perform_verification(
        self,
        device_id: int,
        cuda_ptr: int,
        memory_size: int,
        expected_info: dict,
    ) -> tuple[bool, str]:
        """Run integrity verification using the provided CUDA IPC handle."""

        from scstore import _C as _ckpt_helpers  # pylint: disable=import-error

        start_ts = time.time()

        try:
            logger.info(
                "Verifying model data from GPU (device_id=%s, cuda_ptr=%s, memory_size=%s)",
                device_id,
                cuda_ptr,
                memory_size,
            )
            passed: bool = _ckpt_helpers.verify_model_data_from_gpu(
                device_id,
                cuda_ptr,
                memory_size,
                expected_info,
                1,
            )
            logger.info("Verification passed: %s", passed)

            latency = time.time() - start_ts
            MODEL_VERIFICATION_LATENCY.observe(latency)
            MODEL_VERIFICATION_TOTAL.labels(
                status="passed" if passed else "failed"
            ).inc()

            return (passed, "" if passed else "verification mismatch")

        except Exception as exc:  # noqa: BLE001
            MODEL_VERIFICATION_TOTAL.labels(status="failed").inc()
            return False, str(exc)

    def _verification_done_callback(self, future: Future, key: tuple[str, str]):
        """Store the verification result when the background job finishes."""

        passed, err_msg = False, "internal error"
        try:
            passed, err_msg = future.result()
        except Exception as exc:  # noqa: BLE001
            err_msg = str(exc)

        status = (
            store_daemon_pb2.VerificationStatus.VERIFICATION_STATUS_PASSED
            if passed
            else store_daemon_pb2.VerificationStatus.VERIFICATION_STATUS_FAILED
        )

        with self._verification_lock:
            self._verification_results[key] = (status, err_msg)

        # ------------------------------------------------------------------
        # Post-verification actions:
        #   • On success   → automatically register the replica with the
        #     Global-Store (via ReplicaManager.confirm_model).
        #   • On failure   → unload the faulty replica to free resources.
        # ------------------------------------------------------------------
        model_path, replica_uuid = key

        if passed:
            try:
                self.replica_manager.confirm_model(
                    model_path=model_path,
                    replica_uuid=replica_uuid,
                    device_type=store_daemon_pb2.DEVICE_TYPE_GPU,
                )
            except Exception:
                logger.exception(
                    "Automatic registration failed for verified replica %s", key
                )
        else:
            try:
                self.replica_manager.unload_model(
                    model_path,
                    device_type=store_daemon_pb2.DEVICE_TYPE_GPU,
                )
            except Exception:
                logger.exception(
                    "Failed to unload replica %s after verification failure", key
                )

    # ------------------------------------------------------------------
    # gRPC APIs for verification
    # ------------------------------------------------------------------

    def WaitModelVerification(
        self,
        request: store_daemon_pb2.VerificationRequest,
        context: grpc.ServicerContext,
    ) -> store_daemon_pb2.VerificationResponse:
        """Allow clients to query the verification status for a model replica."""

        key = (request.model_identifier, request.replica_uuid)

        deadline = time.time() + (
            request.timeout_ms / 1000 if request.timeout_ms else 30
        )

        while True:
            with self._verification_lock:
                if key in self._verification_results:
                    status, err = self._verification_results[key]
                else:
                    status, err = (
                        store_daemon_pb2.VerificationStatus.VERIFICATION_STATUS_UNKNOWN,
                        "No record",
                    )

            if status in (
                store_daemon_pb2.VerificationStatus.VERIFICATION_STATUS_PASSED,
                store_daemon_pb2.VerificationStatus.VERIFICATION_STATUS_FAILED,
            ):
                return store_daemon_pb2.VerificationResponse(status=status, err_msg=err)

            if time.time() >= deadline:
                return store_daemon_pb2.VerificationResponse(
                    status=status, err_msg="timeout"
                )

            # Brief sleep to avoid busy-looping.
            time.sleep(0.1)

    def GetDetailedStatus(
        self,
        request: store_daemon_pb2.GetDetailedStatusRequest,
        context: grpc.ServicerContext,
    ) -> store_daemon_pb2.GetDetailedStatusResponse:
        """Get detailed status information about the StoreDaemon."""
        try:
            # Basic status info
            uptime_seconds = int(time.time() - self.start_time)

            # Memory pool info
            mem_pool_info = store_daemon_pb2.MemoryPoolInfo()
            if self.checkpoint_store:
                mem_pool_info.total_size_bytes = (
                    self.checkpoint_store.get_mem_pool_size()
                )
                mem_pool_info.available_bytes = (
                    self.checkpoint_store.get_available_memory()
                )
                mem_pool_info.allocated_bytes = (
                    mem_pool_info.total_size_bytes - mem_pool_info.available_bytes
                )
                mem_pool_info.chunk_size_bytes = self.checkpoint_store.get_chunk_size()
                # TODO: Get allocated_chunks_count from C++ layer when available

            # Parse C++ metrics to get detailed model and GPU information
            gpu_devices = []
            cpu_models = []
            total_models = 0
            total_model_size = 0
            comm_info = store_daemon_pb2.CommunicationInfo(
                enabled=self.enable_p2p_engine
            )

            # Parse metrics text from C++ layer
            if self.checkpoint_store:
                try:
                    import scstore._checkpoint_store as _cs

                    metrics_text = _cs.get_global_metrics_text().decode()

                    # Parse metrics to extract model and GPU information
                    # Map device_id (int) -> {"total": value, "free": value}
                    gpu_memory_metrics: dict[int, dict[str, int]] = {}
                    model_counts = {"cpu": 0, "gpu": 0}

                    for line in metrics_text.split("\n"):
                        if not line or line.startswith("#"):
                            continue

                        # Parse GPU memory metrics
                        if "store_daemon_gpu_memory_bytes" in line:
                            # Expected format:
                            # store_daemon_gpu_memory_bytes{device_id="0",memory_type="total"} 1.234e+10
                            if "{" in line and "}" in line:
                                labels_part = line[line.find("{") + 1 : line.find("}")]
                                value = float(line.split()[-1])

                                device_id = None
                                memory_type = None
                                for label in labels_part.split(","):
                                    if "device_id=" in label:
                                        try:
                                            device_id = int(
                                                label.split("=")[1].strip('"')
                                            )
                                        except ValueError:
                                            device_id = None
                                    elif "memory_type=" in label:
                                        memory_type = label.split("=")[1].strip('"')

                                if device_id is not None and memory_type:
                                    if device_id not in gpu_memory_metrics:
                                        gpu_memory_metrics[device_id] = {}
                                    gpu_memory_metrics[device_id][memory_type] = int(
                                        value
                                    )

                        # Parse model counts
                        elif "store_daemon_models_in_memory" in line:
                            # Format: store_daemon_models_in_memory{location="cpu|gpu"} value
                            if '{location="cpu"}' in line:
                                model_counts["cpu"] = int(float(line.split()[-1]))
                            elif '{location="gpu"}' in line:
                                model_counts["gpu"] = int(float(line.split()[-1]))

                        # Parse total model size
                        elif (
                            "store_daemon_total_model_size_bytes" in line
                            and "{" not in line
                        ):
                            total_model_size = int(float(line.split()[-1]))

                        # Parse RDMA metrics
                        elif (
                            "store_daemon_p2p_transfers_total" in line
                            and "{" not in line
                        ):
                            comm_info.total_transfers = int(float(line.split()[-1]))
                        elif (
                            "store_daemon_p2p_bytes_transferred_total" in line
                            and "{" not in line
                        ):
                            comm_info.total_bytes_transferred = int(
                                float(line.split()[-1])
                            )
                        elif (
                            "store_daemon_p2p_transfer_errors_total" in line
                            and "{" not in line
                        ):
                            comm_info.total_transfer_errors = int(
                                float(line.split()[-1])
                            )

                    # Build GPU device info
                    # ------------------------------------------------------------------
                    # Resolve the CUDA device ordinal from the device UUID.  The helper
                    # `get_device_uuid_map()` provided by the `_C` extension returns a
                    # mapping of `device_id -> device_uuid`.  Converting it to the
                    # inverse dictionary lets us fetch the correct `device_id` for each
                    # UUID reported by the metrics collector.  Falling back to `-1`
                    # indicates that the mapping was unavailable (e.g. CPU-only build).
                    # ------------------------------------------------------------------
                    try:
                        from scstore import (
                            _C as _ckpt_helpers,  # pylint: disable=import-error
                        )

                        dev_uuid_map = _ckpt_helpers.get_device_uuid_map()
                    except Exception as _exc:  # noqa: BLE001
                        logger.debug("Failed to resolve device UUID map: %s", _exc)
                        dev_uuid_map = {}

                    for device_id, memory_info in gpu_memory_metrics.items():
                        gpu_info = store_daemon_pb2.GpuDeviceInfo()
                        gpu_info.device_id = int(device_id)
                        gpu_info.device_uuid = dev_uuid_map.get(device_id, "")
                        gpu_info.total_memory_bytes = memory_info.get("total", 0)
                        gpu_info.free_memory_bytes = memory_info.get("free", 0)
                        gpu_info.used_memory_bytes = (
                            gpu_info.total_memory_bytes - gpu_info.free_memory_bytes
                        )
                        # TODO: Get actual loaded models per GPU when available from C++ layer
                        gpu_devices.append(gpu_info)

                    total_models = model_counts["cpu"] + model_counts["gpu"]

                    # Note: Individual model details are not available from current C++ API
                    # We would need to extend the C++ layer to expose per-model information

                except Exception as e:
                    logger.warning(f"Failed to parse C++ metrics: {e}")

            # Build response
            response = store_daemon_pb2.GetDetailedStatusResponse(
                is_registered=bool(self.worker_id),
                is_healthy=not self.shutting_down,
                is_shutting_down=self.shutting_down,
                uptime_seconds=uptime_seconds,
                worker_id=self.worker_id or "",
                memory_pool_info=mem_pool_info,
                gpu_devices=gpu_devices,
                cpu_models=cpu_models,
                communication_info=comm_info,
                total_models_loaded=total_models,
                total_model_size_bytes=total_model_size,
                storage_path=str(self.storage_path),
                num_worker_threads=self.config.server.num_threads if self.config else 0,
            )

            return response

        except Exception as e:
            logger.exception(f"Error getting detailed status: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to get detailed status: {str(e)}")
            return store_daemon_pb2.GetDetailedStatusResponse()
