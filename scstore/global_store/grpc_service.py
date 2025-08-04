#  Copyright (c) 2025, StepCast Team.

"""
gRPC service implementation for Global Store.

This provides the gRPC interface layer, delegating business logic to services.
"""

import threading
import time
from uuid import UUID

import duckdb  # DuckDB is a runtime dependency; ignore missing stubs in type checker
import grpc

from scstore.global_store.config import get_config
from scstore.global_store.db_utils import init_db, optimize_db
from scstore.global_store.exceptions import (
    NotFoundError,
    TimeoutError,
    ValidationError,
)
from scstore.global_store.models import MemoryType, ModelReplica, Worker
from scstore.global_store.repositories import (
    ChunkDirectoryRepository,
    ModelReplicaRepository,
    TransportRepository,
    WorkerRepository,
)
from scstore.global_store.services import (
    ChunkService,
    ModelService,
    RecoveryService,
    TransportService,
    WorkerService,
)
from scstore.logger import init_logger
from scstore.proto import global_store_pb2, global_store_pb2_grpc

logger = init_logger(__name__)


class GlobalModelStoreServicer(global_store_pb2_grpc.GlobalModelStoreServicer):
    """
    gRPC service implementation for the Global Store.

    This class handles gRPC requests and delegates to service layer for
    business logic. Thread safety is handled by DuckDB's cursor model.
    """

    def __init__(self, db_file: str | None = None):
        """
        Initialize the Global Store with DuckDB backend.

        Args:
            db_file: Path to DuckDB file. None for in-memory database.
        """
        # Initialize configuration
        self.config = get_config()

        # Initialize database connection
        if db_file is None:
            logger.info("Using in-memory DuckDB database")
            self.connection = duckdb.connect()
        else:
            logger.info(f"Using persistent DuckDB database at: {db_file}")
            self.connection = duckdb.connect(str(db_file))

        # Initialize database schema
        cursor = self.connection.cursor()
        init_db(cursor)

        # Initialize repositories
        self.model_replica_repository = ModelReplicaRepository(self.connection)
        self.transport_repository = TransportRepository(self.connection)
        self.worker_repository = WorkerRepository(self.connection)
        self.chunk_directory_repository = ChunkDirectoryRepository(self.connection)

        # Initialize services
        self.model_service = ModelService(self.model_replica_repository)
        self.transport_service = TransportService(
            self.model_replica_repository, self.transport_repository
        )
        self.worker_service = WorkerService(
            self.worker_repository, self.model_replica_repository
        )
        self.chunk_service = ChunkService(self.chunk_directory_repository)

        # Initialize recovery service for high availability
        self.recovery_service = RecoveryService(
            self.worker_repository, self.model_replica_repository
        )

        self._initiate_startup_recovery()

        # Start background cleanup thread
        self._start_cleanup_thread()
        # Cleanup and optimization are now handled by a single maintenance thread

    def _initiate_startup_recovery(self):
        """Initiate recovery process on startup."""
        try:
            logger.info("Initiating startup recovery process")
            recovery_success = self.recovery_service.initiate_recovery()
            if recovery_success:
                logger.info("Startup recovery completed successfully")
            else:
                logger.warning(
                    "Startup recovery failed, continuing with normal operation"
                )
        except Exception as e:
            logger.exception(f"Error during startup recovery: {e}")

    def _start_cleanup_thread(self):
        """Start background maintenance thread that performs both cleanup and periodic database optimizations."""

        def maintenance_loop():
            # Allow workers some time to register before first maintenance pass
            time.sleep(self.config.heartbeat_timeout_ms / 1000 * 2)

            optimize_interval_sec = self.config.optimize_interval_ms / 1000
            cleanup_interval_sec = self.config.cleanup_interval_ms / 1000
            last_optimize_ts = time.time()

            while True:
                try:
                    # -------- Cleanup inactive workers --------
                    self.worker_service.cleanup_inactive_workers()

                    # -------- Cleanup expired transports --------
                    if self.transport_service:
                        try:
                            self.transport_service.cleanup_expired_transports()
                        except Exception:
                            logger.exception("Error cleaning up expired transports")

                    # -------- Periodic database optimization --------
                    if time.time() - last_optimize_ts >= optimize_interval_sec:
                        try:
                            optimize_db(self.connection)
                        except Exception:
                            # optimize_db already logs; safeguard thread
                            logger.exception("Error optimizing database")
                        last_optimize_ts = time.time()

                except Exception:
                    logger.exception("Error in maintenance thread")

                # Sleep until the next cleanup interval
                time.sleep(cleanup_interval_sec)

        self.cleanup_thread = threading.Thread(target=maintenance_loop, daemon=True)
        self.cleanup_thread.start()

    # ========== Model Replica Methods ==========

    def GetModelInfo(
        self,
        request: global_store_pb2.GetModelInfoRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.GetModelInfoResponse:
        """Get model information and available replicas."""
        model_name = request.model_name
        try:
            if not model_name:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Model name is required")
                return global_store_pb2.GetModelInfoResponse(
                    status=global_store_pb2.Status.ERROR
                )

            # Get available replicas
            replicas = self.model_service.get_model_replicas(model_name)

            # Convert to proto format
            available_replicas = []
            for replica in replicas:
                mem_info = self._replica_to_memory_info(replica)
                available_replicas.append(mem_info)

            # Create response
            model_info = global_store_pb2.ModelInfo(
                model_name=model_name, available_replicas=available_replicas
            )

            status = (
                global_store_pb2.Status.OK
                if available_replicas
                else global_store_pb2.Status.NOT_FOUND
            )

            logger.info(
                f"GetModelInfo for {model_name}: "
                f"found {len(available_replicas)} replicas"
            )

            return global_store_pb2.GetModelInfoResponse(
                status=status, model_info=model_info
            )

        except Exception as e:
            logger.exception(f"Error getting model info for {model_name}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.GetModelInfoResponse(
                status=global_store_pb2.Status.ERROR
            )

    def RegisterModelReplica(
        self,
        request: global_store_pb2.RegisterModelReplicaRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.RegisterModelReplicaResponse:
        """Register or update a model replica."""
        try:
            # Convert proto to domain model
            replica = self._memory_info_to_replica(
                request.mem_info,
                request.model_name,
                request.max_concurrency,
                request.worker_id,
            )

            # Register replica
            registered = self.model_service.register_replica(replica)

            return global_store_pb2.RegisterModelReplicaResponse(
                status=global_store_pb2.Status.OK,
                model_name=registered.model_name,
                replica_id=str(registered.replica_id),
            )

        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return global_store_pb2.RegisterModelReplicaResponse(
                status=global_store_pb2.Status.ERROR
            )
        except Exception as e:
            logger.exception("Error registering model replica")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.RegisterModelReplicaResponse(
                status=global_store_pb2.Status.ERROR
            )

    def UpdateModelReplica(
        self,
        request: global_store_pb2.UpdateModelReplicaRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UpdateModelReplicaResponse:
        """Update model replica heartbeat."""
        try:
            replica_id = UUID(request.replica_id)
            model_name = request.model_name

            success = self.model_service.update_heartbeat(replica_id, model_name)

            status = (
                global_store_pb2.Status.OK
                if success
                else global_store_pb2.Status.NOT_FOUND
            )

            return global_store_pb2.UpdateModelReplicaResponse(
                status=status,
                model_name=model_name,
                replica_id=request.replica_id,
            )

        except Exception as e:
            logger.exception("Error updating model replica")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.UpdateModelReplicaResponse(
                status=global_store_pb2.Status.ERROR,
                model_name=request.model_name,
                replica_id=request.replica_id,
            )

    def UnregisterModelReplica(
        self,
        request: global_store_pb2.UnregisterModelReplicaRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UnregisterModelReplicaResponse:
        """Unregister a model replica."""
        try:
            replica_id = UUID(request.replica_id)
            model_name = request.model_name

            success = self.model_service.unregister_replica(replica_id, model_name)

            status = (
                global_store_pb2.Status.OK
                if success
                else global_store_pb2.Status.NOT_FOUND
            )

            return global_store_pb2.UnregisterModelReplicaResponse(status=status)

        except Exception as e:
            logger.exception("Error unregistering model replica")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.UnregisterModelReplicaResponse(
                status=global_store_pb2.Status.ERROR
            )

    def ListModelReplicas(
        self,
        request: global_store_pb2.ListModelReplicasRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.ListModelReplicasResponse:
        """List model replicas with optional filters."""
        try:
            # Apply filters
            model_name_filter: str | None = (
                request.model_name if request.HasField("model_name") else None
            )
            node_id_filter: str | None = (
                request.node_id if request.HasField("node_id") else None
            )
            memory_type_filter: MemoryType | None = None
            if request.HasField("memory_type"):
                memory_type_filter = MemoryType(
                    global_store_pb2.MemoryType.Name(request.memory_type)
                )

            # Get replicas
            replicas = self.model_service.list_replicas(
                model_name=model_name_filter,
                node_id=node_id_filter,
                memory_type=memory_type_filter,
            )

            # Group by model name
            model_replicas = {}
            for replica in replicas:
                if replica.model_name not in model_replicas:
                    model_replicas[replica.model_name] = (
                        global_store_pb2.MemoryInfoList(list=[])
                    )

                mem_info = self._replica_to_memory_info(replica)
                model_replicas[replica.model_name].list.append(mem_info)

            return global_store_pb2.ListModelReplicasResponse(
                model_replicas=model_replicas
            )

        except Exception as e:
            logger.exception("Error listing model replicas")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.ListModelReplicasResponse()

    # ========== Transport Methods ==========

    def RequestModelReplicaTransport(
        self,
        request: global_store_pb2.RequestModelReplicaTransportRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.RequestModelReplicaTransportResponse:
        """Request model transport with load balancing."""
        try:
            wait_timeout_ms = request.wait_timeout_ms

            # Request transport
            replica, transport_id = self.transport_service.request_transport(
                model_name=request.model_name,
                source_node_id=request.source_node_id,
                source_address=request.source_address,
                source_port=request.source_port,
                wait_timeout_ms=wait_timeout_ms,
            )

            # Convert to proto format
            remote_info = self._replica_to_memory_info(replica)

            return global_store_pb2.RequestModelReplicaTransportResponse(
                status=global_store_pb2.Status.OK,
                remote_memory_info=remote_info,
                transport_id=str(transport_id),
            )

        except TimeoutError:
            logger.warning(f"Timeout waiting for model {request.model_name}")
            return global_store_pb2.RequestModelReplicaTransportResponse(
                status=global_store_pb2.Status.TIMED_OUT
            )
        except Exception as e:
            logger.exception("Error requesting model transport")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.RequestModelReplicaTransportResponse(
                status=global_store_pb2.Status.ERROR
            )

    def CompleteModelReplicaTransport(
        self,
        request: global_store_pb2.CompleteModelReplicaTransportRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.CompleteModelReplicaTransportResponse:
        """Complete model transport and release resources."""
        try:
            transport_id = UUID(request.transport_id)

            # Complete transport
            self.transport_service.complete_transport(transport_id)

            return global_store_pb2.CompleteModelReplicaTransportResponse(
                status=global_store_pb2.Status.OK
            )

        except NotFoundError:
            logger.warning(f"Transport not found: {request.transport_id}")
            return global_store_pb2.CompleteModelReplicaTransportResponse(
                status=global_store_pb2.Status.NOT_FOUND
            )
        except Exception as e:
            logger.exception("Error completing model transport")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.CompleteModelReplicaTransportResponse(
                status=global_store_pb2.Status.ERROR
            )

    # ========== Worker Methods ==========

    def RegisterWorker(
        self,
        request: global_store_pb2.RegisterWorkerRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.RegisterWorkerResponse:
        """Register a new worker."""
        try:
            # Convert proto to domain model
            worker = Worker(
                node_id=request.node_id,
                node_address=request.node_address,
                grpc_port=request.grpc_port,
                p2p_port=request.p2p_port,
                mem_pool_total_size=request.mem_pool_total_size,
                mem_pool_available_size=request.mem_pool_available_size,
            )

            # Check if this is a recovery registration
            is_recovery = request.is_recovery_registration
            previous_worker_id = (
                request.previous_worker_id if request.previous_worker_id else None
            )

            if is_recovery:
                # Handle recovery registration through recovery service
                success, state_sync_required = (
                    self.recovery_service.handle_worker_recovery_registration(
                        worker, previous_worker_id
                    )
                )

                if not success:
                    return global_store_pb2.RegisterWorkerResponse(
                        status=global_store_pb2.Status.ERROR
                    )

                # Get the registered worker to get the worker_id
                registered = self.worker_service.find_worker_by_address(
                    worker.node_address, worker.grpc_port
                )

                if not registered:
                    return global_store_pb2.RegisterWorkerResponse(
                        status=global_store_pb2.Status.ERROR
                    )

                return global_store_pb2.RegisterWorkerResponse(
                    status=global_store_pb2.Status.OK,
                    worker_id=registered.worker_id,
                    heartbeat_interval_ms=self.config.default_heartbeat_interval_ms,
                    state_sync_required=state_sync_required,
                    expected_state_version=self.recovery_service.get_worker_state_version(
                        registered.worker_id
                    ),
                )
            else:
                # Normal registration
                registered = self.worker_service.register_worker(worker)

                return global_store_pb2.RegisterWorkerResponse(
                    status=global_store_pb2.Status.OK,
                    worker_id=registered.worker_id,
                    heartbeat_interval_ms=self.config.default_heartbeat_interval_ms,
                    state_sync_required=False,
                    expected_state_version=0,
                )

        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return global_store_pb2.RegisterWorkerResponse(
                status=global_store_pb2.Status.ERROR
            )
        except Exception as e:
            logger.exception("Error registering worker")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.RegisterWorkerResponse(
                status=global_store_pb2.Status.ERROR
            )

    def WorkerHeartbeat(
        self,
        request: global_store_pb2.WorkerHeartbeatRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.WorkerHeartbeatResponse:
        """Process enhanced worker heartbeat."""
        try:
            # Check if this is an enhanced heartbeat
            if request.state_version > 0:
                # Enhanced heartbeat with state information
                return self._handle_enhanced_heartbeat(request, context)
            else:
                # Legacy heartbeat
                success = self.worker_service.heartbeat(
                    worker_id=request.worker_id,
                    mem_pool_available_size=request.mem_pool_available_size,
                    accepting_new_requests=request.accepting_new_requests,
                )

                status = (
                    global_store_pb2.Status.OK
                    if success
                    else global_store_pb2.Status.NOT_FOUND
                )

                return global_store_pb2.WorkerHeartbeatResponse(status=status)

        except Exception as e:
            logger.exception("Error processing worker heartbeat")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.WorkerHeartbeatResponse(
                status=global_store_pb2.Status.ERROR
            )

    def _handle_enhanced_heartbeat(
        self,
        request: global_store_pb2.WorkerHeartbeatRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.WorkerHeartbeatResponse:
        """Handle enhanced heartbeat with state synchronization."""
        try:
            # Process basic heartbeat first
            success = self.worker_service.heartbeat(
                worker_id=request.worker_id,
                mem_pool_available_size=request.mem_pool_available_size,
                accepting_new_requests=request.accepting_new_requests,
            )

            if not success:
                return global_store_pb2.WorkerHeartbeatResponse(
                    status=global_store_pb2.Status.NOT_FOUND
                )

            # Check if state synchronization is needed
            current_version = self.recovery_service.get_worker_state_version(
                request.worker_id
            )
            state_sync_required = request.state_version < current_version

            # Checksum validation (if provided by worker)
            if request.state_checksum:
                server_checksum = self.recovery_service.get_worker_state_checksum(
                    request.worker_id
                )

                if request.state_checksum != server_checksum:
                    logger.debug(
                        "State checksum mismatch for worker %s: local=%s, global=%s",
                        request.worker_id,
                        request.state_checksum,
                        server_checksum,
                    )
                    state_sync_required = True

            # Detect obsolete models reported by the worker but not present in global state
            obsolete_models: list[str] = []
            if request.registered_models:
                obsolete_models = self.recovery_service.get_obsolete_models(
                    request.worker_id, list(request.registered_models)
                )

            # If obsolete models exist instruct state sync
            if obsolete_models:
                state_sync_required = True

            return global_store_pb2.WorkerHeartbeatResponse(
                status=global_store_pb2.Status.OK,
                state_sync_required=state_sync_required,
                expected_state_version=current_version,
                obsolete_models=obsolete_models,
                server_timestamp=int(time.time()),
            )

        except Exception as e:
            logger.exception(
                f"Error handling enhanced heartbeat for worker {request.worker_id}"
            )
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.WorkerHeartbeatResponse(
                status=global_store_pb2.Status.ERROR
            )

    def UnregisterWorker(
        self,
        request: global_store_pb2.UnregisterWorkerRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UnregisterWorkerResponse:
        """Unregister a worker."""
        try:
            success = self.worker_service.unregister_worker(request.worker_id)

            status = (
                global_store_pb2.Status.OK
                if success
                else global_store_pb2.Status.NOT_FOUND
            )

            return global_store_pb2.UnregisterWorkerResponse(status=status)

        except Exception as e:
            logger.exception("Error unregistering worker")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.UnregisterWorkerResponse(
                status=global_store_pb2.Status.ERROR
            )

    def ListActiveWorkers(
        self,
        request: global_store_pb2.ListActiveWorkersRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.ListActiveWorkersResponse:
        """List active workers."""
        try:
            workers = self.worker_service.list_active_workers(
                include_unavailable=request.include_unavailable
            )

            # Convert to proto format
            worker_infos = []
            for worker in workers:
                worker_info = global_store_pb2.ListActiveWorkersResponse.WorkerInfo(
                    worker_id=worker.worker_id,
                    node_id=worker.node_id,
                    node_address=worker.node_address,
                    grpc_port=worker.grpc_port,
                    p2p_port=worker.p2p_port,
                    mem_pool_total_size=worker.mem_pool_total_size,
                    mem_pool_available_size=worker.mem_pool_available_size,
                    accepting_new_requests=worker.accepting_new_requests,
                    last_heartbeat_timestamp=int(
                        worker.last_heartbeat.timestamp()
                        if worker.last_heartbeat
                        else 0
                    ),
                    state_version=self.recovery_service.get_worker_state_version(
                        worker.worker_id
                    ),
                    status=self._determine_worker_status(worker),
                )
                worker_infos.append(worker_info)

            logger.info(f"Listed {len(worker_infos)} active workers")

            return global_store_pb2.ListActiveWorkersResponse(workers=worker_infos)

        except Exception as e:
            logger.exception("Error listing active workers")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.ListActiveWorkersResponse()

    # ========== High Availability Methods ==========

    def SynchronizeWorkerState(
        self,
        request: global_store_pb2.SynchronizeWorkerStateRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.SynchronizeWorkerStateResponse:
        """Synchronize worker state for high availability."""
        try:
            success, state_changes, new_version, new_checksum = (
                self.recovery_service.synchronize_worker_state(
                    request.worker_id, request.local_state
                )
            )

            if success:
                return global_store_pb2.SynchronizeWorkerStateResponse(
                    status=global_store_pb2.Status.OK,
                    new_state_version=new_version,
                    state_changes=state_changes,
                    new_state_checksum=new_checksum,
                )
            else:
                return global_store_pb2.SynchronizeWorkerStateResponse(
                    status=global_store_pb2.Status.ERROR
                )

        except Exception as e:
            logger.exception(
                f"Error synchronizing worker state for {request.worker_id}"
            )
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.SynchronizeWorkerStateResponse(
                status=global_store_pb2.Status.ERROR
            )

    def RequestFullStateSync(
        self,
        request: global_store_pb2.RequestFullStateSyncRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.RequestFullStateSyncResponse:
        """Request full state synchronization for a worker."""
        try:
            success, expected_replicas, new_version, new_checksum = (
                self.recovery_service.request_full_state_sync(request.worker_id)
            )

            if success:
                return global_store_pb2.RequestFullStateSyncResponse(
                    status=global_store_pb2.Status.OK,
                    new_state_version=new_version,
                    expected_replicas=expected_replicas,
                    new_state_checksum=new_checksum,
                )
            else:
                return global_store_pb2.RequestFullStateSyncResponse(
                    status=global_store_pb2.Status.ERROR
                )

        except Exception as e:
            logger.exception(
                f"Error requesting full state sync for {request.worker_id}"
            )
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.RequestFullStateSyncResponse(
                status=global_store_pb2.Status.ERROR
            )

    # ========== Utility Methods ==========

    def HealthCheck(
        self,
        request: global_store_pb2.HealthCheckRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.HealthCheckResponse:
        """Simple health check endpoint."""
        return global_store_pb2.HealthCheckResponse(status=global_store_pb2.Status.OK)

    # ========== Chunk Directory Methods ==========

    def QueryChunkLocations(
        self,
        request: global_store_pb2.QueryChunkLocationsRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.QueryChunkLocationsResponse:
        """Query chunk locations for distributed memory pool."""
        try:
            # Convert chunk indices to list
            chunk_indices = (
                list(request.chunk_indices) if request.chunk_indices else None
            )

            # Query chunk locations
            locations = self.chunk_service.query_chunk_locations(
                request.model_id, chunk_indices
            )

            return global_store_pb2.QueryChunkLocationsResponse(
                status=global_store_pb2.Status.OK,
                locations=locations,
            )

        except Exception as e:
            logger.exception(f"Error querying chunk locations: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.QueryChunkLocationsResponse(
                status=global_store_pb2.Status.ERROR
            )

    def BatchUpdateChunkStates(
        self,
        request: global_store_pb2.BatchUpdateChunkStatesRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.BatchUpdateChunkStatesResponse:
        """Batch update chunk states from StoreDaemon."""
        try:
            # Update chunk states
            updates_list = list(
                request.updates
            )  # Convert RepeatedCompositeFieldContainer to list for typing
            updates_applied = self.chunk_service.batch_update_chunk_states(
                request.worker_id,
                request.node_id,
                updates_list,
            )

            return global_store_pb2.BatchUpdateChunkStatesResponse(
                status=global_store_pb2.Status.OK,
                updates_applied=updates_applied,
            )

        except Exception as e:
            logger.exception(f"Error updating chunk states: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.BatchUpdateChunkStatesResponse(
                status=global_store_pb2.Status.ERROR,
                updates_applied=0,
            )

    # ========== Helper Methods ==========

    def _replica_to_memory_info(
        self, replica: ModelReplica
    ) -> global_store_pb2.MemoryInfo:
        """Convert ModelReplica to MemoryInfo proto."""
        return global_store_pb2.MemoryInfo(
            node_id=replica.node_id,
            node_address=replica.node_address,
            node_port=replica.node_port,
            memory_size=replica.memory_size,
            memory_type=global_store_pb2.MemoryType.Value(replica.memory_type.value),  # pyright: ignore[reportArgumentType]
            device_id=replica.device_id,
            remote_memory_keys=replica.remote_memory_keys,
            buffer_sizes=replica.buffer_sizes,
        )

    def _memory_info_to_replica(
        self,
        mem_info: global_store_pb2.MemoryInfo,
        model_name: str,
        max_concurrency: int,
        worker_id: str,
    ) -> ModelReplica:
        """Convert MemoryInfo proto to ModelReplica."""
        return ModelReplica(
            model_name=model_name,
            node_id=mem_info.node_id,
            node_address=mem_info.node_address,
            node_port=mem_info.node_port,
            memory_size=mem_info.memory_size,
            memory_type=MemoryType(
                global_store_pb2.MemoryType.Name(mem_info.memory_type)
            ),
            device_id=mem_info.device_id,
            max_concurrency=max_concurrency,
            remote_memory_keys=list(mem_info.remote_memory_keys),
            buffer_sizes=list(mem_info.buffer_sizes),
            worker_id=worker_id,
        )

    def _determine_worker_status(
        self, worker: Worker
    ) -> global_store_pb2.ConnectionStatus:
        """Determine worker connection status based on heartbeat and flags.

        Heuristics:
        1. If the worker heartbeats within the timeout window and is accepting new
           requests – consider it CONNECTED.
        2. If the heartbeat is recent (<= 2× timeout) but the worker is not
           accepting new requests – treat it as RECONNECTING (perhaps draining
           or recovering).
        3. Otherwise mark it DISCONNECTED.
        """
        # Protect against missing heartbeat information.
        if not worker.last_heartbeat:
            return global_store_pb2.ConnectionStatus.DISCONNECTED

        now_ts = time.time()
        last_hb_ts = worker.last_heartbeat.timestamp()

        timeout_sec = self.config.heartbeat_timeout_ms / 1000.0

        age = now_ts - last_hb_ts

        # Case 1 – healthy
        if age <= timeout_sec and worker.accepting_new_requests:
            return global_store_pb2.ConnectionStatus.CONNECTED

        # Case 2 – within extended window or not accepting
        if age <= timeout_sec * 2:
            return global_store_pb2.ConnectionStatus.RECONNECTING

        # Otherwise
        return global_store_pb2.ConnectionStatus.DISCONNECTED

    # ========== Testing Utilities ===========

    def reset_state(self) -> None:
        """Reset all in-memory database state (for test isolation).

        This helper is *only* intended for the interaction-tests suite.  It
        clears every user-mutable table so that each test starts from a clean
        slate while keeping the same DuckDB connection and background worker
        threads alive.
        """
        cursor = self.connection.cursor()

        # Order matters due to foreign-key constraints (replica_counters ➜ model_replicas)
        tables = [
            "model_transports",  # Depends on model_replicas via replica_id FK
            "replica_counters",
            "model_replicas",
            "workers",
        ]
        for table in tables:
            try:
                cursor.execute(f"DELETE FROM {table}")
            except Exception:  # noqa: BLE001 – best-effort cleanup for tests
                logger.exception(f"Failed to truncate table {table} during reset_state")

        # Reset Prometheus gauges that might retain state across tests
        try:
            from scstore.global_store import metrics as gs_metrics  # local import

            gs_metrics.ACTIVE_TRANSPORTS_GAUGE.set(0)
        except Exception:
            # Metric subsystem not critical for state reset – ignore
            logger.debug("Failed to reset Prometheus gauges during reset_state")

        # Ensure all outstanding metrics / in-memory counters in services are in sync.
        # The simplest way is to recreate the repositories & services bound to the
        # existing connection so that any internal caches are discarded.
        self.model_replica_repository = ModelReplicaRepository(self.connection)
        self.transport_repository = TransportRepository(self.connection)
        self.worker_repository = WorkerRepository(self.connection)

        self.model_service = ModelService(self.model_replica_repository)
        self.transport_service = TransportService(
            self.model_replica_repository, self.transport_repository
        )
        self.worker_service = WorkerService(
            self.worker_repository, self.model_replica_repository
        )
        self.recovery_service = RecoveryService(
            self.worker_repository, self.model_replica_repository
        )

        logger.debug(
            "GlobalModelStoreServicer state has been reset for the next test run"
        )
