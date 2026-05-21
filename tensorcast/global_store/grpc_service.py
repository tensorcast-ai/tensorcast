#  Copyright (c) 2025-2026, TensorCast Team.

"""
gRPC service implementation for Global Store.

This provides the gRPC interface layer, delegating business logic to services.
"""

import json
import time
from pathlib import Path
from typing import Any

import duckdb  # DuckDB is a runtime dependency; ignore missing stubs in type checker
import grpc

from tensorcast.global_store.config import get_config
from tensorcast.global_store.db_utils import init_db
from tensorcast.global_store.exceptions import (
    ValidationError,
)
from tensorcast.global_store.grpc_helpers import (
    coerce_db_datetime,
    datetime_to_timestamp,
    hex_sha256_to_multibase,
    index_bytes_to_multibase_sha256,
    is_safe_relative_path,
    multibase_sha256_to_hex,
    sha256_digest_to_multibase,
    timestamp_to_datetime,
)
from tensorcast.global_store.maintenance_coordinator import (
    GlobalStoreMaintenanceCoordinator,
)
from tensorcast.global_store.models import (
    ExportState,
    Instance,
    PlacementPlan,
    Replica,
    Worker,
)
from tensorcast.global_store.replica_memory_codec import (
    export_state_from_proto,
    export_state_to_proto,
    memory_info_to_replica,
    parse_transport_metadata,
    replica_to_memory_info,
)
from tensorcast.global_store.repositories import (
    ArtifactBindingRepository,
    ArtifactDiskLocationRepository,
    ArtifactLayoutAttachmentRepository,
    ArtifactPersistenceStatusRepository,
    ArtifactPlacementRepository,
    AssemblyAttemptRepository,
    AssemblyLayoutBindingRepository,
    AssemblyReadinessCutRepository,
    AssemblySlotOccupancyRepository,
    ChunkDirectoryRepository,
    ClusterInfoRepository,
    GroupRealizationRepository,
    GroupVersionSetRepository,
    InstanceRepository,
    LayoutSpecRepository,
    LeafRepository,
    OperationRepository,
    PendingTransportRequestRepository,
    ProgressiveCoverageRepository,
    ProofRepository,
    ReplicaRepository,
    ShardHomeLeaseRepository,
    TransportRepository,
    ViewCoverageRepository,
    ViewRepository,
    WorkerRepository,
)
from tensorcast.global_store.repositories.artifact_index_repository import (
    ArtifactIndexRepository,
)
from tensorcast.global_store.repositories.artifact_repository import ArtifactRepository
from tensorcast.global_store.repositories.base import db_execution_lock
from tensorcast.global_store.repositories.idempotency_repository import (
    IdempotencyRepository,
)
from tensorcast.global_store.repositories.key_mapping_repository import (
    KeyMappingRepository,
)
from tensorcast.global_store.rpc.artifact_binding_rpc_handler import (
    ArtifactBindingRpcHandler,
)
from tensorcast.global_store.rpc.artifact_index_rpc_handler import (
    ArtifactIndexRpcHandler,
)
from tensorcast.global_store.rpc.artifact_query_rpc_handler import (
    ArtifactQueryRpcHandler,
)
from tensorcast.global_store.rpc.assembly_attempt_rpc_handler import (
    AssemblyAttemptRpcHandler,
)
from tensorcast.global_store.rpc.assembly_readiness_cut_rpc_handler import (
    AssemblyReadinessCutRpcHandler,
)
from tensorcast.global_store.rpc.assembly_slot_occupancy_rpc_handler import (
    AssemblySlotOccupancyRpcHandler,
)
from tensorcast.global_store.rpc.chunk_rpc_handler import ChunkRpcHandler
from tensorcast.global_store.rpc.disk_location_rpc_handler import DiskLocationRpcHandler
from tensorcast.global_store.rpc.group_realization_rpc_handler import (
    GroupRealizationRpcHandler,
)
from tensorcast.global_store.rpc.instance_rpc_handler import InstanceRpcHandler
from tensorcast.global_store.rpc.key_mapping_rpc_handler import KeyMappingRpcHandler
from tensorcast.global_store.rpc.layout_binding_rpc_handler import (
    LayoutBindingRpcHandler,
)
from tensorcast.global_store.rpc.layout_spec_rpc_handler import LayoutSpecRpcHandler
from tensorcast.global_store.rpc.operation_rpc_handler import OperationRpcHandler
from tensorcast.global_store.rpc.placement_persistence_rpc_handler import (
    PlacementPersistenceRpcHandler,
)
from tensorcast.global_store.rpc.progressive_rpc_handler import ProgressiveRpcHandler
from tensorcast.global_store.rpc.replica_lifecycle_rpc_handler import (
    ReplicaLifecycleRpcHandler,
)
from tensorcast.global_store.rpc.replica_registration_rpc_handler import (
    ReplicaRegistrationRpcHandler,
)
from tensorcast.global_store.rpc.shard_home_lease_rpc_handler import (
    ShardHomeLeaseRpcHandler,
)
from tensorcast.global_store.rpc.transport_rpc_handler import TransportRpcHandler
from tensorcast.global_store.rpc.view_proof_rpc_handler import ViewProofRpcHandler
from tensorcast.global_store.rpc.worker_rpc_handler import WorkerRpcHandler
from tensorcast.global_store.rpc.worker_state_sync_rpc_handler import (
    WorkerStateSyncRpcHandler,
)
from tensorcast.global_store.rpc_servicer_mixins import (
    ArtifactCatalogRpcServicerMixin,
    AssemblyViewRpcServicerMixin,
    ClusterRuntimeRpcServicerMixin,
    WorkflowOrchestrationRpcServicerMixin,
)
from tensorcast.global_store.services import (
    ArtifactService,
    ChunkService,
    GroupRealizationService,
    InstanceService,
    PlacementService,
    ProgressiveReplicationService,
    RecoveryService,
    ShardHomeLeaseService,
    TransportService,
    ViewStateService,
    WorkerControlReducer,
    WorkerService,
)
from tensorcast.logger import init_logger
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.global_store.v1 import (
    global_store_pb2,
    global_store_pb2_grpc,
)

logger = init_logger(__name__)


class GlobalStoreServicer(
    ClusterRuntimeRpcServicerMixin,
    ArtifactCatalogRpcServicerMixin,
    AssemblyViewRpcServicerMixin,
    WorkflowOrchestrationRpcServicerMixin,
    global_store_pb2_grpc.ClusterRuntimeServiceServicer,
    global_store_pb2_grpc.ArtifactCatalogServiceServicer,
    global_store_pb2_grpc.AssemblyViewServiceServicer,
    global_store_pb2_grpc.WorkflowOrchestrationServiceServicer,
    global_store_pb2_grpc.ClusterAdminServiceServicer,
):
    """
    gRPC service implementation for the Global Store.

    This class handles gRPC requests and delegates to service layer for
    business logic. Thread safety is handled by DuckDB's cursor artifact.
    """

    progressive_replication_service: ProgressiveReplicationService

    _POLICY_FROM_PROTO = {
        global_store_pb2.PLACEMENT_POLICY_LOCAL_ONLY: "local_only",
        global_store_pb2.PLACEMENT_POLICY_REPLICATED: "replicated",
        global_store_pb2.PLACEMENT_POLICY_SHARDED: "sharded",
    }
    _POLICY_TO_PROTO = {v: k for k, v in _POLICY_FROM_PROTO.items()}

    _TARGET_STATE_FROM_PROTO = {
        global_store_pb2.PLACEMENT_TARGET_STATE_UNSPECIFIED: "pending",
        global_store_pb2.PLACEMENT_TARGET_STATE_PENDING: "pending",
        global_store_pb2.PLACEMENT_TARGET_STATE_COPYING: "copying",
        global_store_pb2.PLACEMENT_TARGET_STATE_COMPLETE: "complete",
        global_store_pb2.PLACEMENT_TARGET_STATE_FAILED: "failed",
        global_store_pb2.PLACEMENT_TARGET_STATE_SKIPPED: "skipped",
    }
    _TARGET_STATE_TO_PROTO = {
        "pending": global_store_pb2.PLACEMENT_TARGET_STATE_PENDING,
        "copying": global_store_pb2.PLACEMENT_TARGET_STATE_COPYING,
        "complete": global_store_pb2.PLACEMENT_TARGET_STATE_COMPLETE,
        "failed": global_store_pb2.PLACEMENT_TARGET_STATE_FAILED,
        "skipped": global_store_pb2.PLACEMENT_TARGET_STATE_SKIPPED,
    }

    _PERSISTENCE_STATE_FROM_PROTO = {
        global_store_pb2.PERSISTENCE_STATE_PENDING: "pending",
        global_store_pb2.PERSISTENCE_STATE_RUNNING: "running",
        global_store_pb2.PERSISTENCE_STATE_DEGRADED: "degraded",
        global_store_pb2.PERSISTENCE_STATE_SUCCESS: "success",
        global_store_pb2.PERSISTENCE_STATE_FAILED: "failed",
    }
    _PERSISTENCE_STATE_TO_PROTO = {
        v: k for k, v in _PERSISTENCE_STATE_FROM_PROTO.items()
    }

    _DISK_LOCATION_KIND_FROM_PROTO = {
        global_store_pb2.DISK_LOCATION_KIND_UNSPECIFIED: "MANAGED",
        global_store_pb2.DISK_LOCATION_KIND_MANAGED: "MANAGED",
        global_store_pb2.DISK_LOCATION_KIND_IMPORTED: "IMPORTED",
    }
    _DISK_LOCATION_KIND_TO_PROTO = {
        "MANAGED": global_store_pb2.DISK_LOCATION_KIND_MANAGED,
        "IMPORTED": global_store_pb2.DISK_LOCATION_KIND_IMPORTED,
    }
    worker_service: WorkerService
    instance_service: InstanceService
    transport_service: TransportService

    def _init_repositories(self) -> None:
        """Initialize repository objects used by all domains."""
        self.replica_repository = ReplicaRepository(self.connection)

        self.artifacts_repo = ArtifactRepository(self.connection)
        self.artifact_indices = ArtifactIndexRepository(self.connection)
        self.transport_repository = TransportRepository(self.connection)
        self.pending_transport_request_repository = PendingTransportRequestRepository(
            self.connection
        )
        self.worker_repository = WorkerRepository(self.connection)
        self.idempotency_repository = IdempotencyRepository(self.connection)
        self.instance_repository = InstanceRepository(self.connection)
        self.chunk_directory_repository = ChunkDirectoryRepository(self.connection)
        self.view_repository = ViewRepository(self.connection)
        self.view_coverage_repository = ViewCoverageRepository(self.connection)
        self.leaf_repository = LeafRepository(self.connection)
        self.binding_repository = ArtifactBindingRepository(self.connection)
        self.key_mapping_repository = KeyMappingRepository(self.connection)
        self.group_version_set_repository = GroupVersionSetRepository(self.connection)
        self.group_realization_repository = GroupRealizationRepository(self.connection)
        self.cluster_info_repository = ClusterInfoRepository(self.connection)
        self.disk_location_repository = ArtifactDiskLocationRepository(self.connection)
        self.placement_repository = ArtifactPlacementRepository(self.connection)
        self.persistence_status_repository = ArtifactPersistenceStatusRepository(
            self.connection
        )
        self.layout_spec_repository = LayoutSpecRepository(self.connection)
        self.assembly_attempt_repository = AssemblyAttemptRepository(self.connection)
        self.assembly_layout_binding_repository = AssemblyLayoutBindingRepository(
            self.connection
        )
        self.assembly_readiness_cut_repository = AssemblyReadinessCutRepository(
            self.connection
        )
        self.artifact_layout_attachment_repository = ArtifactLayoutAttachmentRepository(
            self.connection
        )
        self.assembly_slot_occupancy_repository = AssemblySlotOccupancyRepository(
            self.connection
        )
        self.proof_repository = ProofRepository(self.connection)
        self.operation_repository = OperationRepository(self.connection)
        self.shard_home_lease_repository = ShardHomeLeaseRepository(self.connection)
        self.progressive_coverage_repository = ProgressiveCoverageRepository(
            self.connection
        )

    def _init_catalog_handlers(self) -> None:
        """Initialize handlers for artifact catalog and workflow metadata."""
        self.operation_rpc_handler = OperationRpcHandler(
            operation_repository=self.operation_repository,
            default_ttl_ms=int(self.config.limits.operation_leases.default_ttl_ms),
            max_ttl_ms=int(self.config.limits.operation_leases.max_ttl_ms),
            min_status_update_interval_ms=int(
                self.config.limits.operation_writes.min_status_update_interval_ms
            ),
            datetime_to_timestamp=datetime_to_timestamp,
            coerce_db_datetime=coerce_db_datetime,
            logger=logger,
        )
        self.artifact_binding_rpc_handler = ArtifactBindingRpcHandler(
            binding_repository=self.binding_repository,
            datetime_to_timestamp=datetime_to_timestamp,
            logger=logger,
        )
        alias_cache_ttl_ms = max(
            0, int(self.config.key_mapping_policy.alias_cache_ttl_ms)
        )
        alias_cache_ttl_seconds = (
            0 if alias_cache_ttl_ms == 0 else max(1, (alias_cache_ttl_ms + 999) // 1000)
        )
        self.key_mapping_rpc_handler = KeyMappingRpcHandler(
            key_mapping_repository=self.key_mapping_repository,
            artifact_repository=self.artifacts_repo,
            artifact_index_repository=self.artifact_indices,
            multibase_sha256_to_hex=multibase_sha256_to_hex,
            alias_cache_ttl_seconds=alias_cache_ttl_seconds,
            logger=logger,
        )
        self.group_realization_service = GroupRealizationService(
            version_set_repository=self.group_version_set_repository,
            realization_repository=self.group_realization_repository,
            key_mapping_repository=self.key_mapping_repository,
            config=self.config.group_realization,
        )
        self.group_realization_rpc_handler = GroupRealizationRpcHandler(
            group_realization_service=self.group_realization_service,
            operation_repository=self.operation_repository,
            logger=logger,
        )
        self.artifact_index_rpc_handler = ArtifactIndexRpcHandler(
            artifact_index_repository=self.artifact_indices,
            artifact_repository=self.artifacts_repo,
            multibase_sha256_to_hex=multibase_sha256_to_hex,
            logger=logger,
        )

    def _init_view_and_layout_handlers(self) -> None:
        """Initialize assembly/view/layout services and handlers."""
        self.chunk_service = ChunkService(self.chunk_directory_repository)
        self.view_state_service = ViewStateService(
            self.view_repository,
            self.leaf_repository,
            self.view_coverage_repository,
            self.layout_spec_repository,
            self.assembly_layout_binding_repository,
            self.proof_repository,
        )
        self.view_proof_rpc_handler = ViewProofRpcHandler(
            config=self.config,
            artifact_repository=self.artifacts_repo,
            view_repository=self.view_repository,
            view_coverage_repository=self.view_coverage_repository,
            proof_repository=self.proof_repository,
            view_state_service=self.view_state_service,
            timestamp_to_datetime=timestamp_to_datetime,
            datetime_to_timestamp=datetime_to_timestamp,
            get_tensor_intervals_for_artifact_id=self._get_tensor_intervals_for_artifact_id,
            logger=logger,
        )
        self.layout_spec_rpc_handler = LayoutSpecRpcHandler(
            artifact_indices=self.artifact_indices,
            layout_spec_repository=self.layout_spec_repository,
            multibase_sha256_to_hex=multibase_sha256_to_hex,
            sha256_digest_to_multibase=sha256_digest_to_multibase,
            logger=logger,
        )
        self.assembly_attempt_rpc_handler = AssemblyAttemptRpcHandler(
            assembly_attempt_repository=self.assembly_attempt_repository,
            datetime_to_timestamp=datetime_to_timestamp,
            logger=logger,
        )
        self.layout_binding_rpc_handler = LayoutBindingRpcHandler(
            connection=self.connection,
            artifact_repository=self.artifacts_repo,
            layout_spec_repository=self.layout_spec_repository,
            assembly_layout_binding_repository=self.assembly_layout_binding_repository,
            artifact_layout_attachment_repository=self.artifact_layout_attachment_repository,
            datetime_to_timestamp=datetime_to_timestamp,
            coerce_db_datetime=coerce_db_datetime,
            logger=logger,
        )
        self.assembly_readiness_cut_rpc_handler = AssemblyReadinessCutRpcHandler(
            assembly_readiness_cut_repository=self.assembly_readiness_cut_repository,
            datetime_to_timestamp=datetime_to_timestamp,
            logger=logger,
        )
        self.assembly_slot_occupancy_rpc_handler = AssemblySlotOccupancyRpcHandler(
            assembly_slot_occupancy_repository=self.assembly_slot_occupancy_repository,
            datetime_to_timestamp=datetime_to_timestamp,
            coerce_db_datetime=coerce_db_datetime,
            logger=logger,
        )

    def _init_disk_location_handler(self) -> None:
        """Initialize disk-location metadata handler for managed persistence."""
        self.cluster_id = self.cluster_info_repository.get_or_create_cluster_id()
        self.disk_location_rpc_handler = DiskLocationRpcHandler(
            disk_location_repository=self.disk_location_repository,
            cluster_id=self.cluster_id,
            is_safe_relative_path=is_safe_relative_path,
            disk_location_kind_from_proto=self._DISK_LOCATION_KIND_FROM_PROTO,
            disk_location_kind_to_proto=self._DISK_LOCATION_KIND_TO_PROTO,
            datetime_to_timestamp=datetime_to_timestamp,
            coerce_db_datetime=coerce_db_datetime,
            logger=logger,
        )

    def _start_maintenance(self) -> None:
        """Start periodic background maintenance."""
        self._maintenance_coordinator = GlobalStoreMaintenanceCoordinator(
            config=self.config,
            connection=self.connection,
            get_worker_service=lambda: self.worker_service,
            get_instance_service=lambda: self.instance_service,
            get_transport_service=lambda: self.transport_service,
            get_group_realization_service=lambda: self.group_realization_service,
            get_progressive_replication_service=lambda: (
                self.progressive_replication_service
            ),
            logger=logger,
        )
        self.cleanup_thread = self._maintenance_coordinator.start()

    def __init__(self, db_file: str | None = None):
        """
        Initialize the Global Store with DuckDB backend.

        Args:
            db_file: Path to DuckDB file. None for in-memory database.
        """
        # Initialize configuration
        self.config = get_config()
        self._runtime_info: dict[str, Any] = {}

        # Initialize database connection
        db_path: Path | None = None
        if db_file:
            db_path = Path(db_file).expanduser()
            db_path.parent.mkdir(parents=True, exist_ok=True)

        if db_path is None:
            logger.info("Using in-memory DuckDB database")
            self.connection = duckdb.connect()
        else:
            logger.info("Using persistent DuckDB database at: %s", db_path)
            self.connection = duckdb.connect(str(db_path))

        # Initialize database schema
        with db_execution_lock():
            cursor = self.connection.cursor()
            try:
                init_db(cursor)
            finally:
                cursor.close()

        self._init_repositories()
        self._init_catalog_handlers()
        self._init_view_and_layout_handlers()
        self._rebuild_runtime_services_and_handlers()

        self._initiate_startup_recovery()

        self._init_disk_location_handler()
        self._start_maintenance()

    def _rebuild_runtime_services_and_handlers(self) -> None:
        """Rebuild services/handlers that depend on mutable worker/replica repositories."""
        previous_worker_service = getattr(self, "worker_service", None)
        if previous_worker_service is not None:
            try:
                previous_worker_service.close()
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Failed to close previous WorkerService during runtime rebuild"
                )

        reducer = getattr(self, "worker_control_reducer", None)
        if reducer is None:
            reducer_config = self.config.worker_control_reducer
            reducer = WorkerControlReducer(
                shard_count=reducer_config.shard_count,
                queue_capacity=reducer_config.queue_capacity,
                coalesce_window_ms=reducer_config.coalesce_window_ms,
                logger=logger,
            )
            reducer.start()
            self.worker_control_reducer = reducer

        self.artifact_service = ArtifactService(self.replica_repository)
        self.replica_registration_rpc_handler = ReplicaRegistrationRpcHandler(
            artifact_service=self.artifact_service,
            artifact_repository=self.artifacts_repo,
            artifact_index_repository=self.artifact_indices,
            idempotency_repository=self.idempotency_repository,
            memory_info_to_replica_artifact_id=self._memory_info_to_replica_artifact_id,
            index_bytes_to_multibase_sha256=index_bytes_to_multibase_sha256,
            hex_sha256_to_multibase=hex_sha256_to_multibase,
            control_reducer=self.worker_control_reducer,
            logger=logger,
        )
        self.transport_service = TransportService(
            self.replica_repository,
            self.transport_repository,
            self.pending_transport_request_repository,
        )
        self.transport_rpc_handler = TransportRpcHandler(
            transport_service=self.transport_service,
            replica_to_memory_info=self._replica_to_memory_info,
            logger=logger,
        )
        self.progressive_replication_service = ProgressiveReplicationService(
            self.progressive_coverage_repository
        )
        self.progressive_rpc_handler = ProgressiveRpcHandler(
            service=self.progressive_replication_service,
            logger=logger,
        )
        self.replica_lifecycle_rpc_handler = ReplicaLifecycleRpcHandler(
            artifact_service=self.artifact_service,
            replica_repository=self.replica_repository,
            transport_repository=self.transport_repository,
            transport_service=self.transport_service,
            replica_to_memory_info=self._replica_to_memory_info,
            logger=logger,
        )
        self.worker_service = WorkerService(
            self.worker_repository,
            self.replica_repository,
            control_reducer=self.worker_control_reducer,
        )
        self.instance_service = InstanceService(
            self.instance_repository, self.worker_repository
        )
        self.recovery_service = RecoveryService(
            self.worker_repository,
            self.replica_repository,
            self.worker_service,
            control_reducer=self.worker_control_reducer,
        )
        self.placement_service = PlacementService(
            self.worker_repository,
            self.placement_repository,
            self.persistence_status_repository,
        )
        self.placement_persistence_rpc_handler = PlacementPersistenceRpcHandler(
            placement_service=self.placement_service,
            policy_from_proto=self._policy_from_proto,
            persistence_state_from_proto=self._persistence_state_from_proto,
            target_state_from_proto=self._target_state_from_proto,
            plan_to_proto=self._plan_to_proto,
            logger=logger,
        )
        self.artifact_query_rpc_handler = ArtifactQueryRpcHandler(
            artifact_service=self.artifact_service,
            artifact_repository=self.artifacts_repo,
            view_state_service=self.view_state_service,
            replica_to_memory_info=self._replica_to_memory_info,
            datetime_to_timestamp=datetime_to_timestamp,
            coerce_db_datetime=coerce_db_datetime,
            logger=logger,
        )
        self.chunk_rpc_handler = ChunkRpcHandler(
            chunk_service=self.chunk_service,
            logger=logger,
        )
        self.worker_rpc_handler = WorkerRpcHandler(
            worker_service=self.worker_service,
            worker_repository=self.worker_repository,
            idempotency_repository=self.idempotency_repository,
            recovery_service=self.recovery_service,
            default_heartbeat_interval_ms=self.config.default_heartbeat_interval_ms,
            determine_worker_status=self._determine_worker_status,
            logger=logger,
        )
        self.instance_rpc_handler = InstanceRpcHandler(
            instance_service=self.instance_service,
            default_heartbeat_interval_ms=self.config.default_heartbeat_interval_ms,
            determine_instance_status=self._determine_instance_status,
            logger=logger,
        )
        self.worker_state_sync_rpc_handler = WorkerStateSyncRpcHandler(
            recovery_service=self.recovery_service,
            logger=logger,
        )
        self.shard_home_lease_service = ShardHomeLeaseService(
            self.shard_home_lease_repository
        )
        self.shard_home_lease_rpc_handler = ShardHomeLeaseRpcHandler(
            shard_home_lease_service=self.shard_home_lease_service,
            datetime_to_timestamp=datetime_to_timestamp,
            logger=logger,
        )

    def set_runtime_info(
        self,
        *,
        listen_host: str | None,
        listen_port: int | None,
        advertise_host: str | None,
        advertise_port: int | None,
        metrics_port: int | None,
        cluster_token: str | None,
        db_file: str | None,
        version: str | None,
    ) -> None:
        self._runtime_info = {
            "listen_host": listen_host,
            "listen_port": listen_port,
            "advertise_host": advertise_host,
            "advertise_port": advertise_port,
            "metrics_port": metrics_port,
            "cluster_token": cluster_token,
            "cluster_id": self.cluster_id,
            "db_file": db_file,
            "version": version,
        }

    def HealthCheck(
        self,
        request: global_store_pb2.HealthCheckRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.HealthCheckResponse:
        """Minimal health check endpoint (status + cluster token)."""
        info = getattr(self, "_runtime_info", {}) or {}
        cluster_token = info.get("cluster_token") or self.config.cluster_token
        return global_store_pb2.HealthCheckResponse(
            status=global_store_pb2.Status.STATUS_OK,
            cluster_token=cluster_token or "",
        )

    def GetServerInfo(
        self,
        request: global_store_pb2.GetServerInfoRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.GetServerInfoResponse:
        """Server metadata endpoint for advertised/bind addresses and diagnostics."""
        info = getattr(self, "_runtime_info", {}) or {}
        listen_host = info.get("listen_host") or self.config.listen_host
        listen_port = info.get("listen_port") or self.config.listen_port
        advertise_host = info.get("advertise_host") or self.config.advertise_host
        advertise_port = info.get("advertise_port") or self.config.advertise_port
        metrics_port = info.get("metrics_port") or self.config.metrics_port
        db_file = info.get("db_file") or (
            str(self.config.db_file) if self.config.db_file else ""
        )
        cluster_id = info.get("cluster_id") or self.cluster_id or ""
        version = info.get("version") or ""
        listen_address = (
            f"{listen_host}:{listen_port}" if listen_host and listen_port else ""
        )
        advertise_address = (
            f"{advertise_host}:{advertise_port}"
            if advertise_host and advertise_port
            else ""
        )
        return global_store_pb2.GetServerInfoResponse(
            status=global_store_pb2.Status.STATUS_OK,
            advertise_address=advertise_address,
            advertise_host=advertise_host or "",
            advertise_port=int(advertise_port or 0),
            listen_address=listen_address,
            listen_host=listen_host or "",
            listen_port=int(listen_port or 0),
            metrics_port=int(metrics_port or 0),
            version=version,
            db_file=db_file or "",
            cluster_id=cluster_id,
        )

    def _get_tensor_intervals_for_artifact_id(
        self, *, artifact_id: str
    ) -> dict[str, tuple[int, int]]:
        row = self.artifacts_repo.get(artifact_id)
        if not row:
            raise ValidationError("artifact_id not found for canonical index lookup")
        index_multihash = row.get("index_multihash")
        if not index_multihash:
            raise ValidationError("canonical index not recorded for artifact_id")
        index_key = multibase_sha256_to_hex(str(index_multihash))
        if not index_key:
            raise ValidationError("invalid index_multihash stored for artifact_id")
        data = self.artifact_indices.get(index_key)
        if data is None:
            raise ValidationError("canonical index bytes missing for artifact_id")
        try:
            decoded = json.loads(bytes(data).decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise ValidationError("failed to decode canonical index bytes") from exc
        if not isinstance(decoded, dict):
            raise ValidationError("canonical index must be a JSON object")

        intervals: dict[str, tuple[int, int]] = {}
        for name, entry in decoded.items():
            if not isinstance(name, str):
                continue
            if not isinstance(entry, list) or len(entry) < 2:
                continue
            try:
                tensor_off = int(entry[0])
                tensor_bytes = int(entry[1])
            except Exception:
                continue
            if tensor_off < 0 or tensor_bytes < 0:
                continue
            intervals[name] = (tensor_off, tensor_bytes)
        if not intervals:
            raise ValidationError("canonical index missing tensor entries")
        return intervals

    @classmethod
    def _policy_from_proto(cls, value: global_store_pb2.PlacementPolicy) -> str:
        try:
            return cls._POLICY_FROM_PROTO[value]
        except KeyError as exc:  # noqa: BLE001
            raise ValidationError("placement_policy is required") from exc

    @classmethod
    def _policy_to_proto(cls, value: str) -> global_store_pb2.PlacementPolicy:
        normalized = value.strip().lower()
        if normalized not in cls._POLICY_TO_PROTO:
            raise ValidationError(f"Unknown placement policy '{value}'")
        return cls._POLICY_TO_PROTO[normalized]

    @classmethod
    def _target_state_from_proto(
        cls, value: global_store_pb2.PlacementTargetState
    ) -> str:
        try:
            return cls._TARGET_STATE_FROM_PROTO[value]
        except KeyError as exc:  # noqa: BLE001
            raise ValidationError("target_state is required") from exc

    @classmethod
    def _target_state_to_proto(
        cls, value: str
    ) -> global_store_pb2.PlacementTargetState:
        normalized = value.strip().lower()
        if normalized not in cls._TARGET_STATE_TO_PROTO:
            raise ValidationError(f"Unknown target state '{value}'")
        return cls._TARGET_STATE_TO_PROTO[normalized]

    @classmethod
    def _persistence_state_from_proto(
        cls, value: global_store_pb2.PersistenceState
    ) -> str:
        try:
            return cls._PERSISTENCE_STATE_FROM_PROTO[value]
        except KeyError as exc:  # noqa: BLE001
            raise ValidationError("persistence state is required") from exc

    @classmethod
    def _persistence_state_to_proto(
        cls, value: str
    ) -> global_store_pb2.PersistenceState:
        normalized = value.strip().lower()
        if normalized not in cls._PERSISTENCE_STATE_TO_PROTO:
            raise ValidationError(f"Unknown persistence state '{value}'")
        return cls._PERSISTENCE_STATE_TO_PROTO[normalized]

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

    # RPC methods are implemented by domain servicer mixins.

    # ========== Helper Methods ==========

    def _plan_to_proto(
        self, plan: PlacementPlan
    ) -> global_store_pb2.PlanPlacementResponse:
        response = global_store_pb2.PlanPlacementResponse(
            plan_id=plan.plan_id,
            effective_policy=self._policy_to_proto(plan.policy),
            degraded=bool(plan.degraded_reason),
            degraded_reason=plan.degraded_reason or "",
        )
        for shard in plan.shards:
            placement = response.placements.add()
            placement.shard.shard_id = shard.shard_id
            placement.shard.shard_idx = shard.shard_idx
            placement.shard.size_bytes = shard.size_bytes
            placement.shard.content_digest = shard.content_digest
            placement.shard.byte_range_start = shard.byte_range_start
            placement.shard.byte_range_length = shard.byte_range_length
            placement.shard.chunk_ids.extend(shard.chunk_ids)
            placement.degraded_reason = shard.degraded_reason or ""
            for target in plan.targets:
                if target.shard_idx != shard.shard_idx:
                    continue
                target_proto = placement.targets.add()
                target_proto.node_id = target.node_id
                if target.lease_id:
                    target_proto.lease_id = target.lease_id
                target_proto.target_state = self._target_state_to_proto(
                    target.target_state
                )
                if target.degraded_reason:
                    target_proto.degraded_reason = target.degraded_reason
        return response

    @staticmethod
    def _export_state_from_proto(
        state: common_pb2.ReplicaTransportMetadata.ExportState,
    ) -> ExportState:
        return export_state_from_proto(state)

    @staticmethod
    def _export_state_to_proto(
        state: ExportState,
    ) -> common_pb2.ReplicaTransportMetadata.ExportState:
        return export_state_to_proto(state)

    def _parse_transport_metadata(
        self, mem_info: common_pb2.MemoryInfo
    ) -> tuple[bool, ExportState, int, list[str], list[int], str | None]:
        return parse_transport_metadata(mem_info)

    def _replica_to_memory_info(self, replica: Replica) -> common_pb2.MemoryInfo:
        """Convert Replica to MemoryInfo proto."""
        return replica_to_memory_info(
            replica=replica,
            datetime_to_timestamp=datetime_to_timestamp,
        )

    def _memory_info_to_replica_artifact_id(
        self,
        mem_info: common_pb2.MemoryInfo,
        artifact_id: str,
        max_concurrency: int,
        worker_id: str,
    ) -> Replica:
        """Convert MemoryInfo proto to Replica using content-addressed artifact_id."""
        return memory_info_to_replica(
            mem_info=mem_info,
            artifact_id=artifact_id,
            max_concurrency=max_concurrency,
            worker_id=worker_id,
            require_view_id=True,
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
            return global_store_pb2.ConnectionStatus.CONNECTION_STATUS_DISCONNECTED

        now_ts = time.time()
        last_hb_ts = worker.last_heartbeat.timestamp()

        timeout_sec = self.config.heartbeat_timeout_ms / 1000.0

        age = now_ts - last_hb_ts

        # Case 1 – healthy
        if age <= timeout_sec and worker.accepting_new_requests:
            return global_store_pb2.ConnectionStatus.CONNECTION_STATUS_CONNECTED

        # Case 2 – within extended window or not accepting
        if age <= timeout_sec * 2:
            return global_store_pb2.ConnectionStatus.CONNECTION_STATUS_RECONNECTING

        # Otherwise
        return global_store_pb2.ConnectionStatus.CONNECTION_STATUS_DISCONNECTED

    def _determine_instance_status(
        self, inst: Instance
    ) -> global_store_pb2.ConnectionStatus:
        if not inst.last_heartbeat:
            return global_store_pb2.ConnectionStatus.CONNECTION_STATUS_DISCONNECTED
        now_ts = time.time()
        last_hb_ts = inst.last_heartbeat.timestamp()
        timeout_sec = self.config.heartbeat_timeout_ms / 1000.0
        age = now_ts - last_hb_ts
        if age <= timeout_sec:
            return global_store_pb2.ConnectionStatus.CONNECTION_STATUS_CONNECTED
        if age <= timeout_sec * 2:
            return global_store_pb2.ConnectionStatus.CONNECTION_STATUS_RECONNECTING
        return global_store_pb2.ConnectionStatus.CONNECTION_STATUS_DISCONNECTED

    # ========== Testing Utilities ===========

    def reset_state(self) -> None:
        """Reset all in-memory database state (for test isolation).

        This helper is *only* intended for the interaction-tests suite.  It
        clears every user-mutable table so that each test starts from a clean
        slate while keeping the same DuckDB connection and background worker
        threads alive.
        """
        # Order matters due to foreign-key constraints (replica_counters ➜ artifact_replicas)
        tables = [
            "artifact_transports",  # Depends on artifact_replicas via replica_id FK
            "pending_transport_requests",
            "replica_counters",
            "artifact_replicas",
            "artifact_disk_locations",
            "artifact_indices",
            "artifacts",
            "worker_reconcile_state",
            "worker_liveness",
            "workers",
            "instances",
        ]
        with db_execution_lock():
            cursor = self.connection.cursor()
            try:
                for table in tables:
                    try:
                        cursor.execute(f"DELETE FROM {table}")
                    except Exception:  # noqa: BLE001 – best-effort cleanup for tests
                        logger.exception(
                            f"Failed to truncate table {table} during reset_state"
                        )
            finally:
                cursor.close()

        # Reset Prometheus gauges that might retain state across tests
        try:
            from tensorcast.global_store import metrics as gs_metrics  # local import

            gs_metrics.ACTIVE_TRANSPORTS_GAUGE.set(0)
        except Exception:
            # Metric subsystem not critical for state reset – ignore
            logger.debug("Failed to reset Prometheus gauges during reset_state")

        # Ensure all outstanding metrics / in-memory counters in services are in sync.
        # Recreate repositories and rebuild all runtime components that depend on them.
        self.replica_repository = ReplicaRepository(self.connection)
        self.transport_repository = TransportRepository(self.connection)
        self.pending_transport_request_repository = PendingTransportRequestRepository(
            self.connection
        )
        self.key_mapping_repository = KeyMappingRepository(self.connection)
        self.group_version_set_repository = GroupVersionSetRepository(self.connection)
        self.group_realization_repository = GroupRealizationRepository(self.connection)
        self.worker_repository = WorkerRepository(self.connection)
        self.instance_repository = InstanceRepository(self.connection)
        self._init_catalog_handlers()
        self._rebuild_runtime_services_and_handlers()

        logger.debug("GlobalStoreServicer state has been reset for the next test run")


def register_global_store_servicers(
    server: grpc.Server, servicer: GlobalStoreServicer
) -> None:
    """Register all Global Store domain services on a gRPC server."""
    global_store_pb2_grpc.add_ClusterRuntimeServiceServicer_to_server(servicer, server)
    global_store_pb2_grpc.add_ArtifactCatalogServiceServicer_to_server(servicer, server)
    global_store_pb2_grpc.add_AssemblyViewServiceServicer_to_server(servicer, server)
    global_store_pb2_grpc.add_WorkflowOrchestrationServiceServicer_to_server(
        servicer, server
    )
    global_store_pb2_grpc.add_ClusterAdminServiceServicer_to_server(servicer, server)
