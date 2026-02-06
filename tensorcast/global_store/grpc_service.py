#  Copyright (c) 2025-2026, TensorCast Team.

"""
gRPC service implementation for Global Store.

This provides the gRPC interface layer, delegating business logic to services.
"""

import base64
import binascii
import hashlib
import ipaddress
import json
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Optional, cast
from uuid import UUID

import duckdb  # DuckDB is a runtime dependency; ignore missing stubs in type checker
import grpc
from google.protobuf import timestamp_pb2

from tensorcast.common.identity import ArtifactIdKind, infer_artifact_id_kind
from tensorcast.global_store import metrics as gs_metrics
from tensorcast.global_store.config import get_config
from tensorcast.global_store.db_utils import init_db, optimize_db
from tensorcast.global_store.exceptions import (
    DatabaseError,
    NotFoundError,
    TimeoutError,
    ValidationError,
)
from tensorcast.global_store.models import (
    ByteSpaceKind,
    ByteSpaceRef,
    ExportState,
    Instance,
    MemoryType,
    PersistenceShardStatus,
    PersistenceStatus,
    PlacementPlan,
    PlacementShard,
    PlacementTarget,
    Replica,
    Worker,
)
from tensorcast.global_store.repositories import (
    ArtifactBindingRepository,
    ArtifactDiskLocationRepository,
    ArtifactLayoutAttachmentRepository,
    ArtifactPersistenceStatusRepository,
    ArtifactPlacementRepository,
    AssemblyLayoutBindingRepository,
    AssemblyRuntimePolicyRepository,
    ChunkDirectoryRepository,
    ClusterInfoRepository,
    InstanceRepository,
    LayoutSpecRepository,
    LeafRepository,
    OperationRepository,
    ProofRepository,
    ReplicaRepository,
    TransportRepository,
    ViewCoverageRepository,
    ViewRepository,
    WorkerRepository,
)
from tensorcast.global_store.repositories.artifact_index_repository import (
    ArtifactIndexRepository,
)
from tensorcast.global_store.repositories.artifact_repository import ArtifactRepository
from tensorcast.global_store.repositories.key_mapping_repository import (
    KeyMappingRepository,
)
from tensorcast.global_store.rpc.artifact_binding_rpc_handler import (
    ArtifactBindingRpcHandler,
)
from tensorcast.global_store.rpc.key_mapping_rpc_handler import KeyMappingRpcHandler
from tensorcast.global_store.rpc.operation_rpc_handler import OperationRpcHandler
from tensorcast.global_store.services import (
    ArtifactService,
    ChunkService,
    InstanceService,
    PlacementService,
    RecoveryService,
    TransportService,
    ViewStateService,
    WorkerService,
)
from tensorcast.global_store.services.view_state_service import (
    LeafWritePayload,
    PieceProofDigestPayload,
    ViewUpsertPayload,
)
from tensorcast.logger import init_logger
from tensorcast.observability.otel import set_span_attributes
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.global_store.v1 import (
    global_store_pb2,
    global_store_pb2_grpc,
)
from tensorcast.proto.layout.v1 import layout_pb2
from tensorcast.proto.operation.v1 import operation_pb2

logger = init_logger(__name__)


class GlobalStoreServicer(global_store_pb2_grpc.GlobalStoreServiceServicer):
    """
    gRPC service implementation for the Global Store.

    This class handles gRPC requests and delegates to service layer for
    business logic. Thread safety is handled by DuckDB's cursor artifact.
    """

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
        cursor = self.connection.cursor()
        init_db(cursor)

        # Initialize repositories
        self.replica_repository = ReplicaRepository(self.connection)

        self.artifacts_repo = ArtifactRepository(self.connection)
        self.artifact_indices = ArtifactIndexRepository(self.connection)
        self.transport_repository = TransportRepository(self.connection)
        self.worker_repository = WorkerRepository(self.connection)
        self.instance_repository = InstanceRepository(self.connection)
        self.chunk_directory_repository = ChunkDirectoryRepository(self.connection)
        self.view_repository = ViewRepository(self.connection)
        self.view_coverage_repository = ViewCoverageRepository(self.connection)
        self.leaf_repository = LeafRepository(self.connection)
        self.binding_repository = ArtifactBindingRepository(self.connection)
        self.key_mapping_repository = KeyMappingRepository(self.connection)
        self.cluster_info_repository = ClusterInfoRepository(self.connection)
        self.disk_location_repository = ArtifactDiskLocationRepository(self.connection)
        self.placement_repository = ArtifactPlacementRepository(self.connection)
        self.persistence_status_repository = ArtifactPersistenceStatusRepository(
            self.connection
        )
        self.layout_spec_repository = LayoutSpecRepository(self.connection)
        self.assembly_layout_binding_repository = AssemblyLayoutBindingRepository(
            self.connection
        )
        self.artifact_layout_attachment_repository = ArtifactLayoutAttachmentRepository(
            self.connection
        )
        self.assembly_runtime_policy_repository = AssemblyRuntimePolicyRepository(
            self.connection
        )
        self.proof_repository = ProofRepository(self.connection)
        self.operation_repository = OperationRepository(self.connection)
        self.operation_rpc_handler = OperationRpcHandler(
            operation_repository=self.operation_repository,
            default_ttl_ms=int(self.config.limits.operation_leases.default_ttl_ms),
            max_ttl_ms=int(self.config.limits.operation_leases.max_ttl_ms),
            min_status_update_interval_ms=int(
                self.config.limits.operation_writes.min_status_update_interval_ms
            ),
            datetime_to_timestamp=self._datetime_to_timestamp,
            coerce_db_datetime=self._coerce_db_datetime,
            logger=logger,
        )
        self.artifact_binding_rpc_handler = ArtifactBindingRpcHandler(
            binding_repository=self.binding_repository,
            datetime_to_timestamp=self._datetime_to_timestamp,
            logger=logger,
        )
        self.key_mapping_rpc_handler = KeyMappingRpcHandler(
            key_mapping_repository=self.key_mapping_repository,
            logger=logger,
        )

        # Initialize services
        self.artifact_service = ArtifactService(self.replica_repository)
        self.transport_service = TransportService(
            self.replica_repository, self.transport_repository
        )
        self.worker_service = WorkerService(
            self.worker_repository, self.replica_repository
        )
        self.instance_service = InstanceService(
            self.instance_repository, self.worker_repository
        )
        self.chunk_service = ChunkService(self.chunk_directory_repository)
        self.view_state_service = ViewStateService(
            self.view_repository,
            self.leaf_repository,
            self.view_coverage_repository,
            self.layout_spec_repository,
            self.assembly_layout_binding_repository,
            self.proof_repository,
        )
        self.placement_service = PlacementService(
            self.worker_repository,
            self.placement_repository,
            self.persistence_status_repository,
        )

        # Initialize recovery service for high availability
        self.recovery_service = RecoveryService(
            self.worker_repository, self.replica_repository, self.worker_service
        )

        self._initiate_startup_recovery()

        self.cluster_id = self.cluster_info_repository.get_or_create_cluster_id()

        # Start background cleanup thread
        self._start_cleanup_thread()
        # Cleanup and optimization are now handled by a single maintenance thread

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

    @staticmethod
    def _timestamp_to_datetime(ts: timestamp_pb2.Timestamp | None) -> datetime | None:
        """Convert protobuf Timestamp to timezone-aware datetime (UTC)."""
        if ts is None:
            return None
        return ts.ToDatetime(tzinfo=timezone.utc)

    @staticmethod
    def _datetime_to_timestamp(dt: datetime | None) -> timestamp_pb2.Timestamp | None:
        """Convert datetime to protobuf Timestamp (UTC)."""
        if dt is None:
            return None
        normalized = (
            dt.astimezone(timezone.utc)
            if dt.tzinfo
            else dt.replace(tzinfo=timezone.utc)
        )
        proto = timestamp_pb2.Timestamp()
        proto.FromDatetime(normalized)
        return proto

    @staticmethod
    def _coerce_db_datetime(value: object) -> datetime | None:
        """Best-effort conversion for DuckDB timestamp outputs."""
        if value is None:
            return None
        if isinstance(value, datetime):
            candidate = value
        elif isinstance(value, str):
            candidate = datetime.fromisoformat(value)
        else:
            raise ValueError(f"Unsupported datetime value: {value!r}")
        return (
            candidate.astimezone(timezone.utc)
            if candidate.tzinfo
            else candidate.replace(tzinfo=timezone.utc)
        )

    @staticmethod
    def _is_safe_relative_path(path: str) -> bool:
        if not path:
            return False
        if "\\" in path:
            return False
        pure = PurePosixPath(path)
        if pure.is_absolute():
            return False
        return ".." not in pure.parts

    @staticmethod
    def _multibase_sha256_to_hex(value: str) -> str | None:
        """Convert multibase base32 multihash (sha2-256) to lowercase hex digest."""
        if not value or value[0] != "b":
            return None
        payload = value[1:]
        if not payload:
            return None
        padding_needed = (-len(payload)) % 8
        padded = payload + ("=" * padding_needed)
        try:
            decoded = base64.b32decode(padded.upper(), casefold=True)
        except binascii.Error:
            return None
        if len(decoded) != 34 or decoded[0] != 0x12 or decoded[1] != 0x20:
            return None
        digest = decoded[2:]
        if len(digest) != 32:
            return None
        return digest.hex()

    @staticmethod
    def _sha256_digest_to_multibase(digest: bytes) -> str | None:
        """Convert raw SHA-256 digest bytes to multibase base32 multihash."""
        if len(digest) != 32:
            return None
        multihash = b"\x12\x20" + digest
        b32 = base64.b32encode(multihash).decode("ascii").lower().rstrip("=")
        return f"b{b32}"

    @classmethod
    def _index_bytes_to_multibase_sha256(cls, data: bytes) -> str | None:
        if not data:
            return None
        digest = hashlib.sha256(data).digest()
        return cls._sha256_digest_to_multibase(digest)

    @classmethod
    def _hex_sha256_to_multibase(cls, value: str) -> str | None:
        if not value:
            return None
        try:
            digest = bytes.fromhex(value)
        except ValueError:
            return None
        return cls._sha256_digest_to_multibase(digest)

    def _get_tensor_intervals_for_artifact_id(
        self, *, artifact_id: str
    ) -> dict[str, tuple[int, int]]:
        row = self.artifacts_repo.get(artifact_id)
        if not row:
            raise ValidationError("artifact_id not found for canonical index lookup")
        index_multihash = row.get("index_multihash")
        if not index_multihash:
            raise ValidationError("canonical index not recorded for artifact_id")
        index_key = self._multibase_sha256_to_hex(str(index_multihash))
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
                    # -------- Cleanup inactive instances --------
                    if self.instance_service:
                        try:
                            self.instance_service.cleanup_inactive_instances()
                        except Exception:
                            logger.exception("Error cleaning up inactive instances")

                    # -------- Cleanup expired transports --------
                    if self.transport_service:
                        try:
                            self.transport_service.cleanup_expired_transports()
                        except Exception:
                            logger.exception("Error cleaning up expired transports")

                    # -------- Retention / GC (v2) --------
                    try:
                        self._run_retention_gc()
                    except Exception:
                        logger.exception("Error applying retention / GC policies")

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

    def _run_retention_gc(self) -> None:
        retention = self.config.limits.retention
        now = datetime.now(timezone.utc)

        # Terminal operations and snapshots
        if retention.operations_ttl_ms > 0:
            cutoff = now - timedelta(milliseconds=int(retention.operations_ttl_ms))
            row = self.connection.execute(
                """
                SELECT COUNT(*)
                FROM operations
                WHERE state IN ('success','failed','cancelled','degraded')
                  AND updated_at < ?
                """,
                [cutoff],
            ).fetchone()
            count = int(row[0]) if row else 0
            if count > 0:
                self.connection.execute(
                    """
                    DELETE FROM operations
                    WHERE state IN ('success','failed','cancelled','degraded')
                      AND updated_at < ?
                    """,
                    [cutoff],
                )
                gs_metrics.inc_gc_rows_deleted(table="operations", count=count)

        # Assembly-scoped proof commitments (post-seal cleanup / bounded retention)
        if retention.assembly_proof_commitments_ttl_ms > 0:
            cutoff = now - timedelta(
                milliseconds=int(retention.assembly_proof_commitments_ttl_ms)
            )
            row = self.connection.execute(
                """
                SELECT COUNT(*)
                FROM assembly_proof_commitments
                WHERE created_at < ?
                """,
                [cutoff],
            ).fetchone()
            count = int(row[0]) if row else 0
            if count > 0:
                self.connection.execute(
                    """
                    DELETE FROM assembly_proof_commitments
                    WHERE created_at < ?
                    """,
                    [cutoff],
                )
                gs_metrics.inc_gc_rows_deleted(
                    table="assembly_proof_commitments", count=count
                )

        # Per-piece proof digests (audit/debug)
        if retention.piece_proof_digests_ttl_ms > 0:
            cutoff = now - timedelta(
                milliseconds=int(retention.piece_proof_digests_ttl_ms)
            )
            row = self.connection.execute(
                """
                SELECT COUNT(*)
                FROM piece_proof_digests
                WHERE created_at < ?
                """,
                [cutoff],
            ).fetchone()
            count = int(row[0]) if row else 0
            if count > 0:
                self.connection.execute(
                    """
                    DELETE FROM piece_proof_digests
                    WHERE created_at < ?
                    """,
                    [cutoff],
                )
                gs_metrics.inc_gc_rows_deleted(table="piece_proof_digests", count=count)

    # ========== Artifact Replica Methods ==========

    def GetArtifactInfoById(
        self,
        request: global_store_pb2.GetArtifactInfoByIdRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.GetArtifactInfoByIdResponse:
        """Content-addressed query by artifact_id (mi2:...)."""
        artifact_id = request.artifact_id
        # Attach business attribute to current span (no-op if tracing disabled)
        set_span_attributes({"tc.artifact.id": artifact_id})
        try:
            if not artifact_id:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("artifact_id is required")
                return global_store_pb2.GetArtifactInfoByIdResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )

            artifact_row: Optional[dict[str, object]] = self.artifacts_repo.get(
                artifact_id
            )

            include_replicas = True
            if request.HasField("include_replicas"):
                include_replicas = request.include_replicas.value

            include_leaves = request.include_leaves
            include_view_meta = request.include_view_meta

            space_kind: Optional[str] = None  # 'C' or 'V'
            space_id: Optional[str] = None

            requested_byte_space: common_pb2.ByteSpaceRef | None = (
                request.requested_byte_space
                if request.HasField("requested_byte_space")
                else None
            )

            def build_hash_space_ref(
                *, space_kind: str, space_id: str
            ) -> common_pb2.HashSpaceRef:
                if space_kind == "C":
                    return common_pb2.HashSpaceRef(
                        byte_space=common_pb2.ByteSpaceRef(
                            kind=common_pb2.BYTE_SPACE_KIND_CANONICAL
                        ),
                        canonical_index_multihash=space_id,
                    )
                if space_kind == "V":
                    return common_pb2.HashSpaceRef(
                        byte_space=common_pb2.ByteSpaceRef(
                            kind=common_pb2.BYTE_SPACE_KIND_VIEW, id=space_id
                        )
                    )
                raise ValidationError(f"Unsupported space_kind: {space_kind}")

            if include_leaves or include_view_meta:  # noqa: SIM102
                if requested_byte_space is None:
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details(
                        "requested_byte_space required when requesting metadata or leaves"
                    )
                    return global_store_pb2.GetArtifactInfoByIdResponse(
                        status=global_store_pb2.Status.STATUS_ERROR
                    )

            if requested_byte_space is not None:
                if requested_byte_space.kind == common_pb2.BYTE_SPACE_KIND_CANONICAL:
                    space_kind = "C"
                    if not artifact_row or not artifact_row.get("index_multihash"):
                        context.set_code(grpc.StatusCode.NOT_FOUND)
                        context.set_details("canonical index not recorded")
                        return global_store_pb2.GetArtifactInfoByIdResponse(
                            status=global_store_pb2.Status.STATUS_NOT_FOUND
                        )
                    space_id = cast(str, artifact_row["index_multihash"])
                elif requested_byte_space.kind == common_pb2.BYTE_SPACE_KIND_VIEW:
                    view_id = requested_byte_space.id
                    if not view_id:
                        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                        context.set_details("requested_byte_space VIEW requires id")
                        return global_store_pb2.GetArtifactInfoByIdResponse(
                            status=global_store_pb2.Status.STATUS_ERROR
                        )
                    space_kind = "V"
                    space_id = view_id
                else:
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details("unsupported requested_byte_space kind")
                    return global_store_pb2.GetArtifactInfoByIdResponse(
                        status=global_store_pb2.Status.STATUS_ERROR
                    )

            available_replicas: list[common_pb2.MemoryInfo] = []
            if include_replicas:
                replica_view_id = space_id if space_kind == "V" else None
                replicas = self.artifact_service.get_artifact_replicas(
                    artifact_id, view_id=replica_view_id
                )
                available_replicas = [self._replica_to_memory_info(r) for r in replicas]

            view_meta_msg: Optional[global_store_pb2.ViewMeta] = None
            leaves_proto: list[global_store_pb2.Leaf] = []
            # partial_coverage: missing byte ranges (units: bytes) in the requested HashSpaceRef byte stream.
            # partial_leaf_coverage: missing Merkle leaf digests (units: leaf indices) for the requested HashSpaceRef.
            partial_byte_details: list[global_store_pb2.PartialCoverageDetail] = []
            partial_leaf_details: list[global_store_pb2.PartialLeafCoverageDetail] = []
            leaf_filter: Optional[list[int]] = None
            view_missing = False
            partial_leaf_miss = False

            view_row: Optional[dict[str, object]] = None
            if space_kind == "V" and (include_leaves or include_view_meta):
                view_row = self.view_state_service.get_view(
                    artifact_id=artifact_id, view_id=space_id or ""
                )
                if view_row is None:
                    view_missing = True
                    context.set_code(grpc.StatusCode.NOT_FOUND)
                    context.set_details("view metadata not found")

            if include_view_meta:
                if space_kind != "V":
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details(
                        "view metadata is only available for view byte space"
                    )
                    return global_store_pb2.GetArtifactInfoByIdResponse(
                        status=global_store_pb2.Status.STATUS_ERROR
                    )
                if view_row is not None:
                    view_size_value = cast(int, view_row["view_size"])
                    view_meta_msg = global_store_pb2.ViewMeta(
                        view_spec_json=str(view_row["view_spec_json"]),
                        view_size=int(view_size_value),
                    )
                    view_data_hash = view_row.get("view_data_hash")
                    if view_data_hash:
                        view_meta_msg.view_data_hash = str(view_data_hash)
                    verified_at = self._coerce_db_datetime(view_row.get("verified_at"))
                    proto_ts = self._datetime_to_timestamp(verified_at)
                    if proto_ts is not None:
                        view_meta_msg.verified_at.CopyFrom(proto_ts)

            if include_leaves:
                if space_kind is None or space_id is None:
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details(
                        "space selection required when requesting leaves"
                    )
                    return global_store_pb2.GetArtifactInfoByIdResponse(
                        status=global_store_pb2.Status.STATUS_ERROR
                    )

                def add_missing_leaf_ranges(
                    detail: global_store_pb2.PartialLeafCoverageDetail,
                    leaf_indices: list[int],
                ) -> None:
                    """Append missing leaf indices as compact ranges."""
                    if not leaf_indices:
                        return
                    sorted_unique = sorted(set(leaf_indices))
                    start = sorted_unique[0]
                    prev = start
                    count = 1
                    for idx in sorted_unique[1:]:
                        if idx == prev + 1:
                            count += 1
                        else:
                            detail.missing_leaf_ranges.append(
                                global_store_pb2.LeafIndexRange(
                                    start=int(start), count=int(count)
                                )
                            )
                            start = idx
                            count = 1
                        prev = idx
                    detail.missing_leaf_ranges.append(
                        global_store_pb2.LeafIndexRange(
                            start=int(start), count=int(count)
                        )
                    )

                leaf_filter = list(request.leaf_idxs) if request.leaf_idxs else None
                if space_kind == "V" and view_row is None:
                    partial_leaf_miss = True
                    detail = global_store_pb2.PartialLeafCoverageDetail(
                        hash_space=build_hash_space_ref(
                            space_kind="V", space_id=space_id or ""
                        ),
                    )
                    if leaf_filter:
                        add_missing_leaf_ranges(detail, leaf_filter)
                    partial_leaf_details.append(detail)
                else:
                    leaf_rows = self.view_state_service.get_leaves(
                        artifact_id=artifact_id,
                        space_kind=space_kind,
                        space_id=space_id,
                        leaf_idxs=leaf_filter,
                    )
                    leaves_proto = [
                        global_store_pb2.Leaf(leaf_idx=idx, digest=digest)
                        for idx, digest in leaf_rows
                    ]
                    if leaf_filter and len(leaves_proto) != len(set(leaf_filter)):
                        partial_leaf_miss = True
                        context.set_code(grpc.StatusCode.NOT_FOUND)
                        context.set_details(
                            "requested leaf digests not fully available"
                        )
                        detail = global_store_pb2.PartialLeafCoverageDetail(
                            hash_space=build_hash_space_ref(
                                space_kind=space_kind, space_id=space_id or ""
                            ),
                        )
                        existing = {leaf.leaf_idx for leaf in leaves_proto}
                        missing = sorted(
                            idx for idx in set(leaf_filter) if idx not in existing
                        )
                        add_missing_leaf_ranges(detail, missing)
                        partial_leaf_details.append(detail)

            has_payload = False
            if include_replicas and available_replicas:
                has_payload = True
            if include_leaves and leaves_proto:
                has_payload = True
            if include_view_meta and view_meta_msg is not None:
                has_payload = True

            need_not_found = (
                view_missing
                or partial_leaf_miss
                or (
                    not has_payload
                    and (include_replicas or include_leaves or include_view_meta)
                )
            )
            status = (
                global_store_pb2.Status.STATUS_NOT_FOUND
                if need_not_found
                else global_store_pb2.Status.STATUS_OK
            )

            descriptor_pb: Optional[common_pb2.ArtifactDescriptor] = None
            if artifact_row is not None:
                id_kind_value = str(artifact_row.get("id_kind") or "").upper()
                descriptor_pb = common_pb2.ArtifactDescriptor(
                    artifact_id=artifact_id,
                    index_multihash=str(artifact_row.get("index_multihash") or ""),
                    data_multihash=str(artifact_row.get("data_multihash") or ""),
                    schema_version=str(artifact_row.get("schema_version") or ""),
                    encoding=str(artifact_row.get("encoding") or ""),
                    total_size=0,
                )
                if id_kind_value == ArtifactIdKind.CGID.value:
                    descriptor_pb.id_kind = (
                        common_pb2.ArtifactIdKind.ARTIFACT_ID_KIND_CGID
                    )
                elif id_kind_value == ArtifactIdKind.MI2.value:
                    descriptor_pb.id_kind = (
                        common_pb2.ArtifactIdKind.ARTIFACT_ID_KIND_MI2
                    )
                else:
                    descriptor_pb.id_kind = (
                        common_pb2.ArtifactIdKind.ARTIFACT_ID_KIND_UNSPECIFIED
                    )

            response = global_store_pb2.GetArtifactInfoByIdResponse(status=status)
            if include_replicas:
                response.replicas.extend(available_replicas)
            if include_leaves:
                response.leaves.extend(leaves_proto)
            if view_meta_msg is not None:
                response.view_meta.CopyFrom(view_meta_msg)
            if partial_byte_details:
                response.partial_coverage.extend(partial_byte_details)
            if partial_leaf_details:
                response.partial_leaf_coverage.extend(partial_leaf_details)
            if descriptor_pb is not None:
                response.descriptor.CopyFrom(descriptor_pb)
            return response

        except Exception as e:
            logger.exception(f"Error getting artifact info by id for {artifact_id}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.GetArtifactInfoByIdResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def UpdateArtifactViewState(
        self,
        request: global_store_pb2.UpdateArtifactViewStateRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UpdateArtifactViewStateResponse:
        """Upsert view metadata, leaf digests, and proof digests."""
        artifact_id = request.artifact_id
        if not artifact_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("artifact_id is required")
            return global_store_pb2.UpdateArtifactViewStateResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

        leaf_count = len(request.leaf_writes)
        proof_count = len(request.proof_digests)
        has_digest_write = (leaf_count + proof_count) > 0

        if has_digest_write:
            limits = self.config.limits.digest_writes
            digest_bytes = sum(len(leaf.digest) for leaf in request.leaf_writes) + sum(
                len(d.digest) for d in request.proof_digests
            )
            total = leaf_count + proof_count
            too_large = (
                leaf_count > limits.max_leaf_writes_per_request
                or proof_count > limits.max_proof_digests_per_request
                or total > limits.max_total_digests_per_request
                or digest_bytes > limits.max_digest_bytes_per_request
            )
            if too_large:
                gs_metrics.inc_digest_request_rejected(reason="too_large")
                context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
                context.set_details(
                    "digest write request exceeds configured limits "
                    f"(leaf_writes={leaf_count}, proof_digests={proof_count}, total_digests={total}, "
                    f"digest_bytes={digest_bytes})"
                )
                return global_store_pb2.UpdateArtifactViewStateResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )

        try:
            view_payload: Optional[ViewUpsertPayload] = None
            if request.HasField("view"):
                view = request.view
                if not view.view_id:
                    raise ValidationError("view.view_id is required")
                if not view.view_spec_json:
                    raise ValidationError("view.view_spec_json is required")
                if view.view_size <= 0:
                    raise ValidationError("view.view_size must be positive")
                verified_at = (
                    self._timestamp_to_datetime(view.verified_at)
                    if view.HasField("verified_at")
                    else None
                )
                view_data_hash = view.view_data_hash or None
                canonical_size: Optional[int] = None
                canonical_covered: Optional[int] = None
                canonical_ranges: list[tuple[int, int]] = []
                if view.HasField("canonical_coverage"):
                    canonical_size = int(view.canonical_coverage.total_bytes)
                    canonical_covered = int(view.canonical_coverage.covered_bytes)
                if view.canonical_ranges:
                    canonical_ranges = [
                        (int(r.off), int(r.len)) for r in view.canonical_ranges
                    ]
                view_payload = ViewUpsertPayload(
                    artifact_id=artifact_id,
                    view_id=view.view_id,
                    view_spec_json=view.view_spec_json,
                    view_size=view.view_size,
                    view_data_hash=view_data_hash,
                    verified_at=verified_at,
                    canonical_size_bytes=canonical_size,
                    canonical_bytes_covered=canonical_covered,
                    canonical_ranges=canonical_ranges or None,
                )

            leaf_payloads: list[LeafWritePayload] = []
            for leaf in request.leaf_writes:
                if not leaf.digest:
                    raise ValidationError("leaf.digest is required")
                if len(leaf.digest) != 32:
                    raise ValidationError("leaf.digest must be 32 bytes (raw sha256)")

                if not leaf.HasField("hash_space"):
                    raise ValidationError("leaf.hash_space must be set")
                hash_space = leaf.hash_space
                if not hash_space.HasField("byte_space"):
                    raise ValidationError("leaf.hash_space.byte_space must be set")
                byte_space = hash_space.byte_space

                if byte_space.kind == common_pb2.BYTE_SPACE_KIND_CANONICAL:
                    if not hash_space.canonical_index_multihash:
                        raise ValidationError(
                            "leaf.hash_space.canonical_index_multihash is required for CANONICAL"
                        )
                    space_kind = "C"
                    space_id = hash_space.canonical_index_multihash
                elif byte_space.kind == common_pb2.BYTE_SPACE_KIND_VIEW:
                    if not byte_space.id:
                        raise ValidationError(
                            "leaf.hash_space.byte_space VIEW requires id"
                        )
                    space_kind = "V"
                    space_id = byte_space.id
                else:
                    raise ValidationError(
                        f"Unsupported leaf.hash_space.byte_space kind: {byte_space.kind}"
                    )

                leaf_payloads.append(
                    LeafWritePayload(
                        artifact_id=artifact_id,
                        space_kind=space_kind,
                        space_id=space_id,
                        leaf_idx=leaf.leaf_idx,
                        digest=bytes(leaf.digest),
                    )
                )

            proof_payloads: list[PieceProofDigestPayload] = []
            for digest in request.proof_digests:
                if not digest.view_id:
                    raise ValidationError("proof_digests.view_id is required")
                if not digest.tensor_name:
                    raise ValidationError("proof_digests.tensor_name is required")
                if not digest.proof_schema_version:
                    raise ValidationError(
                        "proof_digests.proof_schema_version is required"
                    )
                if not digest.digest:
                    raise ValidationError("proof_digests.digest is required")
                proof_payloads.append(
                    PieceProofDigestPayload(
                        artifact_id=artifact_id,
                        view_id=digest.view_id,
                        tensor_name=digest.tensor_name,
                        proof_schema_version=digest.proof_schema_version,
                        proof_chunk_idx=int(digest.proof_chunk_idx),
                        digest=bytes(digest.digest),
                    )
                )

            tensor_intervals: dict[str, tuple[int, int]] | None = None
            if (
                view_payload is not None
                and artifact_id.startswith("cgid:")
                and view_payload.canonical_ranges
            ):
                tensor_intervals = self._get_tensor_intervals_for_artifact_id(
                    artifact_id=artifact_id
                )

            self.view_state_service.update_view_state(
                view=view_payload,
                leaf_writes=leaf_payloads,
                proof_digests=proof_payloads,
                tensor_intervals=tensor_intervals,
            )
            return global_store_pb2.UpdateArtifactViewStateResponse(
                status=global_store_pb2.Status.STATUS_OK
            )
        except ValidationError as exc:
            if has_digest_write:
                gs_metrics.inc_digest_request_rejected(reason="invalid")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.UpdateArtifactViewStateResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except ValueError as exc:
            message = str(exc)
            if has_digest_write:
                if "conflict" in message:
                    gs_metrics.inc_digest_request_rejected(reason="conflict")
                    grid = "unknown"
                    if "leaves conflict" in message:
                        grid = "leaves"
                    elif "piece_proof_digests conflict" in message:
                        grid = "piece_proof_digests"
                    elif "assembly_proof_commitments conflict" in message:
                        grid = "assembly_proof_commitments"
                    elif "tensor_proof_commitments conflict" in message:
                        grid = "tensor_proof_commitments"
                    gs_metrics.inc_digest_conflict(grid=grid)
                else:
                    gs_metrics.inc_digest_request_rejected(reason="invalid")
            if any(
                key in message
                for key in (
                    "coverage metadata missing",
                    "overlapping canonical coverage",
                    "canonical range mismatch",
                    "view_data_hash conflict",
                    "conflict",
                    "proof",
                )
            ):
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            else:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(message)
            return global_store_pb2.UpdateArtifactViewStateResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except DatabaseError as exc:
            message = str(exc)
            if has_digest_write:
                if "conflict" in message:
                    gs_metrics.inc_digest_request_rejected(reason="conflict")
                    grid = "unknown"
                    if "leaves conflict" in message:
                        grid = "leaves"
                    elif "piece_proof_digests conflict" in message:
                        grid = "piece_proof_digests"
                    elif "assembly_proof_commitments conflict" in message:
                        grid = "assembly_proof_commitments"
                    elif "tensor_proof_commitments conflict" in message:
                        grid = "tensor_proof_commitments"
                    gs_metrics.inc_digest_conflict(grid=grid)
                else:
                    gs_metrics.inc_digest_request_rejected(reason="invalid")
            if any(
                key in message
                for key in (
                    "coverage metadata missing",
                    "overlapping canonical coverage",
                    "canonical range mismatch",
                    "view_data_hash conflict",
                    "conflict",
                    "proof",
                )
            ):
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            else:
                context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(message)
            return global_store_pb2.UpdateArtifactViewStateResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Failed to update artifact view state for artifact_id=%s", artifact_id
            )
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.UpdateArtifactViewStateResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def WriteTensorProofCommitments(
        self,
        request: global_store_pb2.WriteTensorProofCommitmentsRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.WriteTensorProofCommitmentsResponse:
        mi2_id = request.mi2_id
        set_span_attributes({"tc.artifact.id": mi2_id})

        if not mi2_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("mi2_id is required")
            return global_store_pb2.WriteTensorProofCommitmentsResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        if not mi2_id.startswith("mi2:"):
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("mi2_id must be a content-addressed mi2 id")
            return global_store_pb2.WriteTensorProofCommitmentsResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        if not request.proof_schema_version:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("proof_schema_version is required")
            return global_store_pb2.WriteTensorProofCommitmentsResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

        artifact_row = self.artifacts_repo.get(mi2_id)
        if artifact_row is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("artifact not found")
            return global_store_pb2.WriteTensorProofCommitmentsResponse(
                status=global_store_pb2.Status.STATUS_NOT_FOUND
            )

        proof_count = len(request.commitments)
        has_digest_write = proof_count > 0
        if has_digest_write:
            limits = self.config.limits.digest_writes
            digest_bytes = sum(len(c.digest) for c in request.commitments)
            too_large = (
                proof_count > limits.max_proof_digests_per_request
                or proof_count > limits.max_total_digests_per_request
                or digest_bytes > limits.max_digest_bytes_per_request
            )
            if too_large:
                gs_metrics.inc_digest_request_rejected(reason="too_large")
                context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
                context.set_details(
                    "digest write request exceeds configured limits "
                    f"(commitments={proof_count}, digest_bytes={digest_bytes})"
                )
                return global_store_pb2.WriteTensorProofCommitmentsResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )

        try:
            inserted = 0
            with self.proof_repository.transaction() as cursor:
                for entry in request.commitments:
                    if not entry.tensor_name:
                        raise ValidationError("commitments.tensor_name is required")
                    if not entry.digest:
                        raise ValidationError("commitments.digest is required")
                    if len(entry.digest) != 32:
                        raise ValidationError(
                            "commitments.digest must be 32 bytes (raw sha256)"
                        )
                    if self.proof_repository.upsert_tensor_proof_commitment(
                        mi2_id=mi2_id,
                        tensor_name=entry.tensor_name,
                        proof_schema_version=request.proof_schema_version,
                        proof_chunk_idx=int(entry.proof_chunk_idx),
                        digest=bytes(entry.digest),
                        cursor=cursor,
                    ):
                        inserted += 1

            gs_metrics.inc_digest_entries_written(
                grid="tensor_proof_commitments", count=inserted
            )
            return global_store_pb2.WriteTensorProofCommitmentsResponse(
                status=global_store_pb2.Status.STATUS_OK, inserted=int(inserted)
            )
        except ValidationError as exc:
            if has_digest_write:
                gs_metrics.inc_digest_request_rejected(reason="invalid")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.WriteTensorProofCommitmentsResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except ValueError as exc:
            message = str(exc)
            if has_digest_write:
                if "conflict" in message:
                    gs_metrics.inc_digest_request_rejected(reason="conflict")
                    grid = (
                        "tensor_proof_commitments"
                        if "tensor_proof_commitments conflict" in message
                        else "unknown"
                    )
                    gs_metrics.inc_digest_conflict(grid=grid)
                else:
                    gs_metrics.inc_digest_request_rejected(reason="invalid")
            if "conflict" in message:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            else:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(message)
            return global_store_pb2.WriteTensorProofCommitmentsResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except DatabaseError as exc:
            message = str(exc)
            if has_digest_write:
                if "conflict" in message:
                    gs_metrics.inc_digest_request_rejected(reason="conflict")
                    grid = (
                        "tensor_proof_commitments"
                        if "tensor_proof_commitments conflict" in message
                        else "unknown"
                    )
                    gs_metrics.inc_digest_conflict(grid=grid)
                else:
                    gs_metrics.inc_digest_request_rejected(reason="invalid")
            if "conflict" in message:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            else:
                context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(message)
            return global_store_pb2.WriteTensorProofCommitmentsResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("WriteTensorProofCommitments failed for mi2_id=%s", mi2_id)
            if has_digest_write:
                gs_metrics.inc_digest_request_rejected(reason="internal")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.WriteTensorProofCommitmentsResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def CheckProofCommitmentsMatch(
        self,
        request: global_store_pb2.CheckProofCommitmentsMatchRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.CheckProofCommitmentsMatchResponse:
        assembly_id = request.assembly_id
        mi2_id = request.mi2_id
        set_span_attributes(
            {"tc.artifact.id": mi2_id, "tc.artifact.assembly_id": assembly_id}
        )

        if not assembly_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("assembly_id is required")
            return global_store_pb2.CheckProofCommitmentsMatchResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        if not mi2_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("mi2_id is required")
            return global_store_pb2.CheckProofCommitmentsMatchResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        if not assembly_id.startswith("cgid:"):
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("assembly_id must be a cgid id")
            return global_store_pb2.CheckProofCommitmentsMatchResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        if not mi2_id.startswith("mi2:"):
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("mi2_id must be a content-addressed mi2 id")
            return global_store_pb2.CheckProofCommitmentsMatchResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        if not request.proof_schema_version:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("proof_schema_version is required")
            return global_store_pb2.CheckProofCommitmentsMatchResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        if not request.tensor_names:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("tensor_names is required")
            return global_store_pb2.CheckProofCommitmentsMatchResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

        if self.artifacts_repo.get(assembly_id) is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("assembly not found")
            return global_store_pb2.CheckProofCommitmentsMatchResponse(
                status=global_store_pb2.Status.STATUS_NOT_FOUND
            )
        if self.artifacts_repo.get(mi2_id) is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("artifact not found")
            return global_store_pb2.CheckProofCommitmentsMatchResponse(
                status=global_store_pb2.Status.STATUS_NOT_FOUND
            )

        try:
            match = self.proof_repository.commitments_match(
                assembly_id=assembly_id,
                mi2_id=mi2_id,
                proof_schema_version=request.proof_schema_version,
                tensor_names=list(request.tensor_names),
            )
            return global_store_pb2.CheckProofCommitmentsMatchResponse(
                status=global_store_pb2.Status.STATUS_OK, match=match
            )
        except ValidationError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.CheckProofCommitmentsMatchResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except ValueError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.CheckProofCommitmentsMatchResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except DatabaseError as exc:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.CheckProofCommitmentsMatchResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "CheckProofCommitmentsMatch failed for assembly_id=%s mi2_id=%s",
                assembly_id,
                mi2_id,
            )
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.CheckProofCommitmentsMatchResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def ListViews(
        self,
        request: global_store_pb2.ListViewsRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.ListViewsResponse:
        """List view metadata for an artifact."""
        artifact_id = request.artifact_id
        if not artifact_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("artifact_id is required")
            return global_store_pb2.ListViewsResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        try:
            page_size = 100
            page_token = ""
            if request.HasField("pagination"):
                if request.pagination.page_size:
                    page_size = int(request.pagination.page_size)
                if request.pagination.page_token:
                    page_token = request.pagination.page_token
            offset = int(page_token) if page_token else 0

            rows, total = self.view_repository.list_by_artifact(
                artifact_id=artifact_id, limit=page_size, offset=offset
            )

            views: list[global_store_pb2.ViewInfo] = []
            for row in rows:
                item = global_store_pb2.ViewInfo(
                    view_id=str(row["view_id"]),
                    view_spec_json=str(row["view_spec_json"]),
                    view_size=int(row["view_size"]),
                )
                if row.get("view_data_hash"):
                    item.view_data_hash = str(row["view_data_hash"])
                if row.get("verified_at"):
                    ts = self._datetime_to_timestamp(row["verified_at"])
                    if ts is not None:
                        item.verified_at.CopyFrom(ts)
                if (
                    row.get("canonical_size_bytes") is not None
                    or row.get("canonical_bytes_covered") is not None
                ):
                    coverage = global_store_pb2.CanonicalCoverage()
                    if row.get("canonical_size_bytes") is not None:
                        coverage.total_bytes = int(row["canonical_size_bytes"])
                    if row.get("canonical_bytes_covered") is not None:
                        coverage.covered_bytes = int(row["canonical_bytes_covered"])
                    item.canonical_coverage.CopyFrom(coverage)

                ranges = self.view_coverage_repository.get_ranges(
                    artifact_id=artifact_id, view_id=str(row["view_id"])
                )
                for off, length in ranges:
                    item.canonical_ranges.add(off=off, len=length)

                views.append(item)

            next_token = ""
            if offset + page_size < total:
                next_token = str(offset + page_size)
            page_info = common_pb2.PageInfo(
                next_page_token=next_token, total_size=total
            )

            return global_store_pb2.ListViewsResponse(
                status=global_store_pb2.Status.STATUS_OK,
                views=views,
                page_info=page_info,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to list views for %s", artifact_id)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.ListViewsResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    # ========== Layout v2 ==========

    def _tensor_names_for_index_multihash(self, *, index_multihash: str) -> set[str]:
        index_key = self._multibase_sha256_to_hex(index_multihash)
        if not index_key:
            raise ValidationError("invalid index_multihash")
        data = self.artifact_indices.get(index_key)
        if data is None:
            raise ValidationError("canonical index bytes missing for index_multihash")
        try:
            decoded = json.loads(bytes(data).decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise ValidationError("failed to decode canonical index bytes") from exc
        if not isinstance(decoded, dict):
            raise ValidationError("canonical index must be a JSON object")
        return {name for name in decoded if isinstance(name, str)}

    def _canonicalize_layout_spec(
        self, *, layout: layout_pb2.LayoutSpec, tensor_names: set[str]
    ) -> layout_pb2.LayoutSpec:
        if int(layout.layout_schema_version) != 1:
            raise ValidationError("layout_schema_version must be 1")
        if not layout.index_multihash:
            raise ValidationError("layout.index_multihash is required")

        out = layout_pb2.LayoutSpec()
        out.CopyFrom(layout)

        # Canonicalize expected_view_ids.
        deduped = sorted(set(out.expected_view_ids))
        del out.expected_view_ids[:]
        out.expected_view_ids.extend(deduped)

        # Validate tensor keys and normalize policy defaults.
        has_replicated = False
        for tensor_name, policy in out.tensors.items():
            if tensor_name not in tensor_names:
                raise ValidationError("layout references unknown tensor_name")
            if policy.overlap_mode == layout_pb2.OVERLAP_MODE_UNSPECIFIED:
                policy.overlap_mode = layout_pb2.OVERLAP_MODE_DISJOINT
            if policy.overlap_mode == layout_pb2.OVERLAP_MODE_REPLICATE_EQUAL:
                has_replicated = True

        if has_replicated:
            if not out.proof_schema_version:
                raise ValidationError(
                    "layout.proof_schema_version is required when using REPLICATE_EQUAL"
                )
        else:
            if out.proof_schema_version:
                raise ValidationError(
                    "layout.proof_schema_version must be empty unless REPLICATE_EQUAL is used"
                )
        return out

    def PutLayoutSpec(
        self,
        request: global_store_pb2.PutLayoutSpecRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.PutLayoutSpecResponse:
        try:
            if not request.HasField("layout"):
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("layout is required")
                return global_store_pb2.PutLayoutSpecResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )
            tensor_names = self._tensor_names_for_index_multihash(
                index_multihash=request.layout.index_multihash
            )
            canonical = self._canonicalize_layout_spec(
                layout=request.layout, tensor_names=tensor_names
            )
            payload = canonical.SerializeToString(deterministic=True)
            digest = hashlib.sha256(payload).digest()
            layout_id = self._sha256_digest_to_multibase(digest)
            if not layout_id:
                raise ValidationError("failed to compute layout_id")

            with self.layout_spec_repository.transaction() as cursor:
                self.layout_spec_repository.put(
                    layout_id=layout_id,
                    index_multihash=canonical.index_multihash,
                    layout_proto=payload,
                    layout_json=(request.layout_json or None),
                    cursor=cursor,
                )

            return global_store_pb2.PutLayoutSpecResponse(
                status=global_store_pb2.Status.STATUS_OK,
                layout_id=layout_id,
            )
        except ValidationError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.PutLayoutSpecResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except (ValueError, DatabaseError) as exc:
            message = str(exc)
            if "layout_id collision" in message:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            else:
                context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(message)
            return global_store_pb2.PutLayoutSpecResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("PutLayoutSpec failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.PutLayoutSpecResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def GetLayoutSpec(
        self,
        request: global_store_pb2.GetLayoutSpecRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.GetLayoutSpecResponse:
        layout_id = request.layout_id
        if not layout_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("layout_id is required")
            return global_store_pb2.GetLayoutSpecResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        try:
            row = self.layout_spec_repository.get(layout_id=layout_id)
            if row is None:
                return global_store_pb2.GetLayoutSpecResponse(
                    status=global_store_pb2.Status.STATUS_NOT_FOUND
                )
            layout = layout_pb2.LayoutSpec()
            layout.ParseFromString(row["layout_proto"])
            record = layout_pb2.LayoutSpecRecord(layout_id=layout_id, layout=layout)
            if row.get("layout_json"):
                record.layout_json = str(row["layout_json"])
            return global_store_pb2.GetLayoutSpecResponse(
                status=global_store_pb2.Status.STATUS_OK,
                record=record,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("GetLayoutSpec failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.GetLayoutSpecResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def GetAssemblyLayoutBinding(
        self,
        request: global_store_pb2.GetAssemblyLayoutBindingRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.GetAssemblyLayoutBindingResponse:
        assembly_id = request.assembly_id
        if not assembly_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("assembly_id is required")
            return global_store_pb2.GetAssemblyLayoutBindingResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        try:
            row = self.assembly_layout_binding_repository.get(assembly_id=assembly_id)
            if row is None:
                return global_store_pb2.GetAssemblyLayoutBindingResponse(
                    status=global_store_pb2.Status.STATUS_NOT_FOUND
                )
            binding = global_store_pb2.AssemblyLayoutBinding(
                assembly_id=str(row["assembly_id"]),
                layout_id=str(row["layout_id"]),
                binding_version=int(row["binding_version"]),
            )
            ts = self._datetime_to_timestamp(
                self._coerce_db_datetime(row.get("updated_at"))
            )
            if ts is not None:
                binding.updated_at.CopyFrom(ts)
            return global_store_pb2.GetAssemblyLayoutBindingResponse(
                status=global_store_pb2.Status.STATUS_OK,
                binding=binding,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("GetAssemblyLayoutBinding failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.GetAssemblyLayoutBindingResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def _assembly_has_any_cross_view_overlap(self, *, assembly_id: str) -> bool:
        rows = self.connection.execute(
            """
            SELECT view_id, range_offset, range_length
            FROM view_coverage_ranges
            WHERE artifact_id = ?
            ORDER BY range_offset ASC
            """,
            [assembly_id],
        ).fetchall()
        max_end = -1
        max_view: str | None = None
        for row in rows:
            view_id = str(row[0])
            start = int(row[1])
            end = start + int(row[2])
            if start < max_end and max_view is not None and view_id != max_view:
                return True
            if end > max_end:
                max_end = end
                max_view = view_id
        return False

    def UpdateAssemblyLayoutBinding(
        self,
        request: global_store_pb2.UpdateAssemblyLayoutBindingRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UpdateAssemblyLayoutBindingResponse:
        assembly_id = request.assembly_id
        layout_id = request.layout_id
        if not assembly_id or not layout_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("assembly_id and layout_id are required")
            return global_store_pb2.UpdateAssemblyLayoutBindingResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

        try:
            expected_version = int(request.expected_binding_version)
            layout_row = self.layout_spec_repository.get(layout_id=layout_id)
            if layout_row is None:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("layout_id not found")
                return global_store_pb2.UpdateAssemblyLayoutBindingResponse(
                    status=global_store_pb2.Status.STATUS_NOT_FOUND
                )

            artifact_row = self.artifacts_repo.get(assembly_id)
            if expected_version == 0 and (
                not artifact_row or not artifact_row.get("index_multihash")
            ):
                # Allow first-time binding creation to establish the assembly's immutable
                # index_multihash (without requiring prior replica registration).
                self.connection.execute(
                    """
                    INSERT INTO artifacts (
                        artifact_id,
                        index_multihash,
                        data_multihash,
                        schema_version,
                        encoding,
                        hash_params_json,
                        id_kind
                    ) VALUES (?, ?, NULL, 'v3', 'json', NULL, 'CGID')
                    ON CONFLICT (artifact_id) DO NOTHING
                    """,
                    [assembly_id, str(layout_row.get("index_multihash"))],
                )
                artifact_row = self.artifacts_repo.get(assembly_id)
            if not artifact_row or not artifact_row.get("index_multihash"):
                raise ValidationError("canonical index not recorded for assembly_id")
            if str(artifact_row.get("index_multihash")) != str(
                layout_row.get("index_multihash")
            ):
                raise ValidationError(
                    "layout.index_multihash does not match assembly index_multihash"
                )

            # Safety rule: REPLICATE_EQUAL -> DISJOINT tightening is allowed only when no overlaps exist.
            existing = self.assembly_layout_binding_repository.get(
                assembly_id=assembly_id
            )
            if existing is not None:
                old_layout_row = self.layout_spec_repository.get(
                    layout_id=str(existing["layout_id"])
                )
                old_has_rep = False
                new_has_rep = False
                if old_layout_row is not None:
                    old_spec = layout_pb2.LayoutSpec()
                    old_spec.ParseFromString(old_layout_row["layout_proto"])
                    old_has_rep = any(
                        p.overlap_mode == layout_pb2.OVERLAP_MODE_REPLICATE_EQUAL
                        for p in old_spec.tensors.values()
                    )
                new_spec = layout_pb2.LayoutSpec()
                new_spec.ParseFromString(layout_row["layout_proto"])
                new_has_rep = any(
                    p.overlap_mode == layout_pb2.OVERLAP_MODE_REPLICATE_EQUAL
                    for p in new_spec.tensors.values()
                )
                if (
                    old_has_rep
                    and not new_has_rep
                    and self._assembly_has_any_cross_view_overlap(
                        assembly_id=assembly_id
                    )
                ):
                    raise ValueError("cannot tighten to DISJOINT while overlaps exist")

            with self.assembly_layout_binding_repository.transaction() as cursor:
                updated = self.assembly_layout_binding_repository.update(
                    assembly_id=assembly_id,
                    layout_id=layout_id,
                    expected_binding_version=expected_version,
                    cursor=cursor,
                )

            binding = global_store_pb2.AssemblyLayoutBinding(
                assembly_id=str(updated["assembly_id"]),
                layout_id=str(updated["layout_id"]),
                binding_version=int(updated["binding_version"]),
            )
            ts = self._datetime_to_timestamp(
                self._coerce_db_datetime(updated.get("updated_at"))
            )
            if ts is not None:
                binding.updated_at.CopyFrom(ts)
            return global_store_pb2.UpdateAssemblyLayoutBindingResponse(
                status=global_store_pb2.Status.STATUS_OK,
                binding=binding,
            )
        except ValidationError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.UpdateAssemblyLayoutBindingResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except (ValueError, DatabaseError) as exc:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details(str(exc))
            return global_store_pb2.UpdateAssemblyLayoutBindingResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("UpdateAssemblyLayoutBinding failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.UpdateAssemblyLayoutBindingResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def AttachLayoutToArtifact(
        self,
        request: global_store_pb2.AttachLayoutToArtifactRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.AttachLayoutToArtifactResponse:
        mi2_id = request.mi2_id
        layout_id = request.layout_id
        if not mi2_id or not layout_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("mi2_id and layout_id are required")
            return global_store_pb2.AttachLayoutToArtifactResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        try:
            artifact_row = self.artifacts_repo.get(mi2_id)
            if not artifact_row or not artifact_row.get("index_multihash"):
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("artifact not found")
                return global_store_pb2.AttachLayoutToArtifactResponse(
                    status=global_store_pb2.Status.STATUS_NOT_FOUND
                )
            layout_row = self.layout_spec_repository.get(layout_id=layout_id)
            if layout_row is None:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("layout not found")
                return global_store_pb2.AttachLayoutToArtifactResponse(
                    status=global_store_pb2.Status.STATUS_NOT_FOUND
                )
            if str(layout_row.get("index_multihash")) != str(
                artifact_row.get("index_multihash")
            ):
                raise ValidationError(
                    "layout.index_multihash does not match artifact index_multihash"
                )
            with self.artifact_layout_attachment_repository.transaction() as cursor:
                self.artifact_layout_attachment_repository.attach(
                    mi2_id=mi2_id, layout_id=layout_id, cursor=cursor
                )
            return global_store_pb2.AttachLayoutToArtifactResponse(
                status=global_store_pb2.Status.STATUS_OK
            )
        except ValidationError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.AttachLayoutToArtifactResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("AttachLayoutToArtifact failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.AttachLayoutToArtifactResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def ListArtifactLayouts(
        self,
        request: global_store_pb2.ListArtifactLayoutsRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.ListArtifactLayoutsResponse:
        mi2_id = request.mi2_id
        if not mi2_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("mi2_id is required")
            return global_store_pb2.ListArtifactLayoutsResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        try:
            layout_ids = self.artifact_layout_attachment_repository.list_by_artifact(
                mi2_id=mi2_id
            )
            return global_store_pb2.ListArtifactLayoutsResponse(
                status=global_store_pb2.Status.STATUS_OK,
                layout_ids=layout_ids,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("ListArtifactLayouts failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.ListArtifactLayoutsResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def GetAssemblyRuntimePolicy(
        self,
        request: global_store_pb2.GetAssemblyRuntimePolicyRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.GetAssemblyRuntimePolicyResponse:
        assembly_id = request.assembly_id
        if not assembly_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("assembly_id is required")
            return global_store_pb2.GetAssemblyRuntimePolicyResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        try:
            row = self.assembly_runtime_policy_repository.get(assembly_id=assembly_id)
            if row is None:
                return global_store_pb2.GetAssemblyRuntimePolicyResponse(
                    status=global_store_pb2.Status.STATUS_NOT_FOUND
                )
            policy = global_store_pb2.AssemblyRuntimePolicy(
                assembly_id=str(row["assembly_id"]),
                policy_version=int(row["policy_version"]),
                policy_json=str(row["policy_json"]),
            )
            ts = self._datetime_to_timestamp(
                self._coerce_db_datetime(row.get("updated_at"))
            )
            if ts is not None:
                policy.updated_at.CopyFrom(ts)
            return global_store_pb2.GetAssemblyRuntimePolicyResponse(
                status=global_store_pb2.Status.STATUS_OK,
                policy=policy,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("GetAssemblyRuntimePolicy failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.GetAssemblyRuntimePolicyResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def UpdateAssemblyRuntimePolicy(
        self,
        request: global_store_pb2.UpdateAssemblyRuntimePolicyRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UpdateAssemblyRuntimePolicyResponse:
        assembly_id = request.assembly_id
        if not assembly_id or not request.policy_json:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("assembly_id and policy_json are required")
            return global_store_pb2.UpdateAssemblyRuntimePolicyResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        try:
            expected = int(request.expected_policy_version)
            with self.assembly_runtime_policy_repository.transaction() as cursor:
                row = self.assembly_runtime_policy_repository.update(
                    assembly_id=assembly_id,
                    policy_json=str(request.policy_json),
                    expected_policy_version=expected,
                    cursor=cursor,
                )
            policy = global_store_pb2.AssemblyRuntimePolicy(
                assembly_id=str(row["assembly_id"]),
                policy_version=int(row["policy_version"]),
                policy_json=str(row["policy_json"]),
            )
            ts = self._datetime_to_timestamp(
                self._coerce_db_datetime(row.get("updated_at"))
            )
            if ts is not None:
                policy.updated_at.CopyFrom(ts)
            return global_store_pb2.UpdateAssemblyRuntimePolicyResponse(
                status=global_store_pb2.Status.STATUS_OK,
                policy=policy,
            )
        except (ValueError, DatabaseError) as exc:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details(str(exc))
            return global_store_pb2.UpdateAssemblyRuntimePolicyResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("UpdateAssemblyRuntimePolicy failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.UpdateAssemblyRuntimePolicyResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    # ========== Unified Operations ==========

    def AcquireOperationLease(
        self,
        request: operation_pb2.AcquireOperationLeaseRequest,
        context: grpc.ServicerContext,
    ) -> operation_pb2.AcquireOperationLeaseResponse:
        return self.operation_rpc_handler.acquire_operation_lease(request, context)

    def KeepaliveOperationLease(
        self,
        request: operation_pb2.KeepaliveOperationLeaseRequest,
        context: grpc.ServicerContext,
    ) -> operation_pb2.KeepaliveOperationLeaseResponse:
        return self.operation_rpc_handler.keepalive_operation_lease(request, context)

    def ReleaseOperationLease(
        self,
        request: operation_pb2.ReleaseOperationLeaseRequest,
        context: grpc.ServicerContext,
    ) -> operation_pb2.ReleaseOperationLeaseResponse:
        return self.operation_rpc_handler.release_operation_lease(request, context)

    def GetOperation(
        self,
        request: operation_pb2.GetOperationRequest,
        context: grpc.ServicerContext,
    ) -> operation_pb2.GetOperationResponse:
        return self.operation_rpc_handler.get_operation(request, context)

    def UpdateOperation(
        self,
        request: operation_pb2.UpdateOperationRequest,
        context: grpc.ServicerContext,
    ) -> operation_pb2.UpdateOperationResponse:
        return self.operation_rpc_handler.update_operation(request, context)

    def GetArtifactBinding(
        self,
        request: global_store_pb2.GetArtifactBindingRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.GetArtifactBindingResponse:
        return self.artifact_binding_rpc_handler.get_artifact_binding(request, context)

    def UpsertArtifactBinding(
        self,
        request: global_store_pb2.UpsertArtifactBindingRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UpsertArtifactBindingResponse:
        return self.artifact_binding_rpc_handler.upsert_artifact_binding(
            request, context
        )

    def RegisterReplica(
        self,
        request: global_store_pb2.RegisterReplicaRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.RegisterReplicaResponse:
        """Register or update a artifact replica."""
        try:
            schema_version_value = "v3"
            if request.HasField("schema_version"):
                candidate_schema_version = request.schema_version.strip()
                if candidate_schema_version and candidate_schema_version != "v3":
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details("schema_version must be 'v3'")
                    return global_store_pb2.RegisterReplicaResponse(
                        status=global_store_pb2.Status.STATUS_ERROR
                    )
                if candidate_schema_version:
                    schema_version_value = candidate_schema_version

            # Convert proto to domain artifact
            replica = self._memory_info_to_replica_artifact_id(
                request.mem_info,
                request.artifact_id,
                request.max_concurrency,
                request.worker_id,
            )
            preserve_transport = not request.mem_info.HasField("transport")

            # Register replica
            registered = self.artifact_service.register_replica(
                replica, preserve_transport=preserve_transport
            )

            # Enrich span with key business attributes (best-effort only)
            from contextlib import suppress

            with suppress(Exception):
                span_attrs: dict[str, bool | int | float | str] = {
                    "tc.artifact.id": registered.artifact_id,
                    "tc.replica.id": str(registered.replica_id),
                    "tc.memory.type": str(replica.memory_type.value),
                    "tc.memory.size": int(replica.memory_size),
                    "tc.device.id": int(replica.device_id),
                }
                worker_id = replica.worker_id
                if worker_id:
                    span_attrs["tc.worker.id"] = worker_id
                set_span_attributes(span_attrs)

            # RFC-0007: Persist artifact descriptor into `artifacts` table.
            artifact_id = registered.artifact_id
            descriptor = request.descriptor if request.HasField("descriptor") else None
            if descriptor is not None and descriptor.artifact_id:
                artifact_id = descriptor.artifact_id

            kind = infer_artifact_id_kind(artifact_id) if artifact_id else None
            if descriptor is not None and descriptor.id_kind:
                kind = (
                    ArtifactIdKind.MI2
                    if descriptor.id_kind
                    == common_pb2.ArtifactIdKind.ARTIFACT_ID_KIND_MI2
                    else ArtifactIdKind.CGID
                    if descriptor.id_kind
                    == common_pb2.ArtifactIdKind.ARTIFACT_ID_KIND_CGID
                    else kind
                )

            if artifact_id:
                index_mh = None
                data_mh = None
                encoding = "json"
                schema_version = schema_version_value
                id_kind = "MI2" if kind is ArtifactIdKind.MI2 else "CGID"
                if descriptor is not None:
                    if descriptor.index_multihash:
                        index_mh = descriptor.index_multihash
                    if descriptor.data_multihash:
                        data_mh = descriptor.data_multihash
                    if descriptor.encoding:
                        encoding = descriptor.encoding
                    if descriptor.schema_version:
                        schema_version = descriptor.schema_version
                if kind is ArtifactIdKind.MI2 and (index_mh is None or data_mh is None):
                    parts = artifact_id.split(":", 2)
                    if len(parts) == 3:
                        index_mh = index_mh or parts[1]
                        data_mh = data_mh or parts[2]
                if not index_mh:
                    if (
                        request.HasField("tensor_index_data")
                        and request.tensor_index_data
                    ):
                        derived = self._index_bytes_to_multibase_sha256(
                            request.tensor_index_data
                        )
                        if derived is not None:
                            index_mh = derived
                    if not index_mh and request.mem_info.tensor_index_key:
                        derived = self._hex_sha256_to_multibase(
                            request.mem_info.tensor_index_key
                        )
                        if derived is not None:
                            index_mh = derived
                try:
                    self.artifacts_repo.upsert_artifact(
                        artifact_id=artifact_id,
                        index_multihash=index_mh,
                        data_multihash=data_mh,
                        schema_version=schema_version,
                        encoding=encoding,
                        hash_params_json=None,
                        id_kind=id_kind,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        f"Failed to upsert artifacts entry for {artifact_id}: {e}"
                    )

            # If canonical index data is provided, store it for de-duplication
            if request.HasField("tensor_index_data") and request.tensor_index_data:
                try:
                    _ = self.artifact_indices.upsert_index(
                        index_data=request.tensor_index_data,
                        encoding=(
                            request.encoding if request.HasField("encoding") else "json"
                        ),
                        schema_version=schema_version_value,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        f"Failed to upsert artifact index for artifact_id={artifact_id}: {e}"
                    )

            return global_store_pb2.RegisterReplicaResponse(
                status=global_store_pb2.Status.STATUS_OK,
                artifact_id=registered.artifact_id,
                replica_id=str(registered.replica_id),
            )

        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return global_store_pb2.RegisterReplicaResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except Exception as e:
            logger.exception("Error registering artifact replica")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.RegisterReplicaResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def UpdateReplica(
        self,
        request: global_store_pb2.UpdateReplicaRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UpdateReplicaResponse:
        """Update artifact replica heartbeat."""
        try:
            replica_id = UUID(request.replica_id)
            artifact_id = request.artifact_id

            set_span_attributes(
                {
                    "tc.artifact.id": artifact_id,
                    "tc.replica.id": str(replica_id),
                }
            )

            success = self.artifact_service.update_heartbeat(replica_id, artifact_id)

            status = (
                global_store_pb2.Status.STATUS_OK
                if success
                else global_store_pb2.Status.STATUS_NOT_FOUND
            )

            return global_store_pb2.UpdateReplicaResponse(
                status=status,
                artifact_id=artifact_id,
                replica_id=request.replica_id,
            )

        except Exception as e:
            logger.exception("Error updating artifact replica")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.UpdateReplicaResponse(
                status=global_store_pb2.Status.STATUS_ERROR,
                artifact_id=request.artifact_id,
                replica_id=request.replica_id,
            )

    def UnregisterReplica(
        self,
        request: global_store_pb2.UnregisterReplicaRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UnregisterReplicaResponse:
        """Unregister a artifact replica."""
        try:
            replica_id = UUID(request.replica_id)
            artifact_id = request.artifact_id

            success = self.artifact_service.unregister_replica(replica_id, artifact_id)

            status = (
                global_store_pb2.Status.STATUS_OK
                if success
                else global_store_pb2.Status.STATUS_NOT_FOUND
            )

            return global_store_pb2.UnregisterReplicaResponse(status=status)

        except Exception as e:
            logger.exception("Error unregistering artifact replica")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.UnregisterReplicaResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def UnregisterReplicaByWorker(
        self,
        request: global_store_pb2.UnregisterReplicaByWorkerRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UnregisterReplicaByWorkerResponse:
        """Unregister a replica by (artifact_id, worker_id[, device_id, memory_type])."""
        try:
            artifact_id = request.artifact_id
            worker_id = request.worker_id
            if not artifact_id or not worker_id:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("artifact_id and worker_id are required")
                return global_store_pb2.UnregisterReplicaByWorkerResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )

            mem_type = None
            if request.HasField("memory_type"):
                # Map proto enum to domain enum
                from tensorcast.global_store.models import MemoryType

                mt = request.memory_type
                mem_type = (
                    MemoryType.GPU
                    if mt == common_pb2.MEMORY_TYPE_GPU
                    else MemoryType.RAM
                    if mt == common_pb2.MEMORY_TYPE_RAM
                    else MemoryType.DISK
                )

            device_id = request.device_id if request.HasField("device_id") else None

            success = self.artifact_service.unregister_by_worker(
                worker_id=worker_id,
                artifact_id=artifact_id,
                memory_type=mem_type,
                device_id=device_id,
            )

            status = (
                global_store_pb2.Status.STATUS_OK
                if success
                else global_store_pb2.Status.STATUS_NOT_FOUND
            )
            return global_store_pb2.UnregisterReplicaByWorkerResponse(status=status)

        except Exception as e:  # noqa: BLE001
            logger.exception("Error in UnregisterReplicaByWorker")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.UnregisterReplicaByWorkerResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def MarkReplicaUnavailable(
        self,
        request: global_store_pb2.MarkReplicaUnavailableRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.MarkReplicaUnavailableResponse:
        try:
            if not request.replica_id:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("replica_id is required")
                return global_store_pb2.MarkReplicaUnavailableResponse(
                    status=global_store_pb2.Status.STATUS_ERROR, updated=False
                )
            replica_id = UUID(request.replica_id)
            replica = self.replica_repository.find_by_replica_id(replica_id)
            if replica is None or (
                request.artifact_id and replica.artifact_id != request.artifact_id
            ):
                return global_store_pb2.MarkReplicaUnavailableResponse(
                    status=global_store_pb2.Status.STATUS_NOT_FOUND, updated=False
                )

            set_span_attributes(
                {
                    "tc.artifact.id": replica.artifact_id,
                    "tc.replica.id": str(replica_id),
                }
            )

            updated = self.replica_repository.mark_unavailable(replica_id)
            return global_store_pb2.MarkReplicaUnavailableResponse(
                status=global_store_pb2.Status.STATUS_OK, updated=updated
            )
        except ValueError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.MarkReplicaUnavailableResponse(
                status=global_store_pb2.Status.STATUS_ERROR, updated=False
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error marking replica unavailable")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.MarkReplicaUnavailableResponse(
                status=global_store_pb2.Status.STATUS_ERROR, updated=False
            )

    def WaitReplicaDrain(
        self,
        request: global_store_pb2.WaitReplicaDrainRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.WaitReplicaDrainResponse:
        if not request.replica_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("replica_id is required")
            return global_store_pb2.WaitReplicaDrainResponse(
                status=global_store_pb2.Status.STATUS_ERROR,
                drained=False,
                current_requests=0,
            )
        try:
            replica_id = UUID(request.replica_id)
        except ValueError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.WaitReplicaDrainResponse(
                status=global_store_pb2.Status.STATUS_ERROR,
                drained=False,
                current_requests=0,
            )

        timeout_ms = int(request.timeout_ms or 0)
        interval = (
            self.transport_service.config.transport_wait_retry_interval_ms / 1000.0
        )

        current = self.replica_repository.get_current_requests(replica_id)
        if current is None:
            return global_store_pb2.WaitReplicaDrainResponse(
                status=global_store_pb2.Status.STATUS_NOT_FOUND,
                drained=False,
                current_requests=0,
            )

        drained = current == 0
        if not drained and timeout_ms > 0:
            now = time.monotonic()
            deadline = now + (timeout_ms / 1000.0)
            context_remaining = context.time_remaining()
            if context_remaining is not None:
                deadline = min(deadline, now + max(context_remaining, 0.0))

            while not drained:
                now = time.monotonic()
                remaining = deadline - now
                if remaining <= 0:
                    break
                time.sleep(min(interval, remaining))
                current = self.replica_repository.get_current_requests(replica_id)
                if current is None:
                    return global_store_pb2.WaitReplicaDrainResponse(
                        status=global_store_pb2.Status.STATUS_NOT_FOUND,
                        drained=False,
                        current_requests=0,
                    )
                drained = current == 0

        oldest_age_ms = self.transport_repository.get_oldest_in_progress_age_ms(
            replica_id
        )
        response = global_store_pb2.WaitReplicaDrainResponse(
            status=global_store_pb2.Status.STATUS_OK
            if drained
            else global_store_pb2.Status.STATUS_TIMED_OUT,
            drained=drained,
            current_requests=int(current or 0),
        )
        if oldest_age_ms is not None:
            response.oldest_transport_age_ms = int(oldest_age_ms)
        return response

    # Legacy ListReplicas removed in favor of ListReplicasV2

    def ListReplicasV2(
        self,
        request: global_store_pb2.ListReplicasV2Request,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.ListReplicasV2Response:
        """List replicas with filtering + pagination (flat records).

        Token format: opaque string encoding of integer offset.
        """
        try:
            # Filters
            artifact_id_filter: str | None = (
                request.artifact_id if request.HasField("artifact_id") else None
            )
            node_id_filter: str | None = (
                request.node_id if request.HasField("node_id") else None
            )
            memory_type_filter = None
            if request.HasField("memory_type"):
                mt = request.memory_type
                if mt == common_pb2.MemoryType.MEMORY_TYPE_GPU:
                    memory_type_filter = MemoryType.GPU
                elif mt == common_pb2.MemoryType.MEMORY_TYPE_RAM:
                    memory_type_filter = MemoryType.RAM
                elif mt == common_pb2.MemoryType.MEMORY_TYPE_DISK:
                    memory_type_filter = MemoryType.DISK

            # Fetch all matching replicas
            replicas = self.artifact_service.list_replicas(
                artifact_id=artifact_id_filter,
                node_id=node_id_filter,
                memory_type=memory_type_filter,
            )

            # Pagination
            page_size = (
                int(request.pagination.page_size)
                if request.pagination and request.pagination.page_size
                else 100
            )
            start = 0
            if request.pagination and request.pagination.page_token:
                try:
                    start = int(request.pagination.page_token)
                except ValueError:
                    start = 0

            end = min(start + page_size, len(replicas))
            sliced = replicas[start:end]
            next_token = str(end) if end < len(replicas) else ""

            # Map to flat records
            records: list[global_store_pb2.ArtifactReplicaRecord] = [
                global_store_pb2.ArtifactReplicaRecord(
                    artifact_id=r.artifact_id,
                    memory_info=self._replica_to_memory_info(r),
                )
                for r in sliced
            ]

            return global_store_pb2.ListReplicasV2Response(
                replicas=records,
                page_info=common_pb2.PageInfo(
                    next_page_token=next_token, total_size=len(replicas)
                ),
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("Error in ListReplicasV2")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.ListReplicasV2Response(
                page_info=common_pb2.PageInfo(next_page_token="", total_size=0)
            )

    def GetArtifactIndex(
        self,
        request: global_store_pb2.GetArtifactIndexRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.GetArtifactIndexResponse:
        """Fetch canonical tensor index bytes by key for de-duplication/UPSERT."""
        try:
            if not request.tensor_index_key:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("tensor_index_key is required")
                return global_store_pb2.GetArtifactIndexResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )

            data = self.artifact_indices.get(request.tensor_index_key)
            if data is None:
                return global_store_pb2.GetArtifactIndexResponse(
                    status=global_store_pb2.Status.STATUS_NOT_FOUND
                )

            # Defaults until we persist per-key metadata
            return global_store_pb2.GetArtifactIndexResponse(
                status=global_store_pb2.Status.STATUS_OK,
                tensor_index_data=data,
                encoding="json",
                schema_version="v3",
            )
        except Exception as e:
            logger.exception("Error getting artifact index")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.GetArtifactIndexResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def GetArtifactIndexById(
        self,
        request: global_store_pb2.GetArtifactIndexByIdRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.GetArtifactIndexByIdResponse:
        """Fetch canonical tensor index bytes by artifact_id.

        Looks up the artifacts table to get index_multihash, then returns the
        canonical index bytes from artifact_indices. Returns NOT_FOUND if
        either artifact or index is missing.
        """
        try:
            artifact_id = request.artifact_id
            if not artifact_id:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("artifact_id is required")
                return global_store_pb2.GetArtifactIndexByIdResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )

            row = self.artifacts_repo.get(artifact_id)
            if not row:
                return global_store_pb2.GetArtifactIndexByIdResponse(
                    status=global_store_pb2.Status.STATUS_NOT_FOUND
                )
            index_multihash = row.get("index_multihash")
            index_key = self._multibase_sha256_to_hex(str(index_multihash))
            if not index_key:
                logger.warning(
                    "Invalid index_multihash stored for %s; cannot derive SHA key",
                    artifact_id,
                )
                return global_store_pb2.GetArtifactIndexByIdResponse(
                    status=global_store_pb2.Status.STATUS_NOT_FOUND
                )
            data = self.artifact_indices.get(index_key)
            if data is None:
                return global_store_pb2.GetArtifactIndexByIdResponse(
                    status=global_store_pb2.Status.STATUS_NOT_FOUND
                )
            return global_store_pb2.GetArtifactIndexByIdResponse(
                status=global_store_pb2.Status.STATUS_OK,
                tensor_index_data=data,
                encoding=str(row.get("encoding") or "json"),
                schema_version=str(row.get("schema_version") or "v3"),
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("Error getting artifact index by id")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.GetArtifactIndexByIdResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    # ========== Transport Methods ==========

    def RequestReplicaTransport(
        self,
        request: global_store_pb2.RequestReplicaTransportRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.RequestReplicaTransportResponse:
        """Request artifact transport with load balancing."""
        try:
            # Normalize wait timeout
            if request.HasField("wait_timeout_dur"):
                d = request.wait_timeout_dur
                wait_timeout_ms = int(d.seconds * 1000 + d.nanos / 1_000_000)
            else:
                wait_timeout_ms = 0

            # Pre-attributes for routing decision visibility
            set_span_attributes(
                {
                    "tc.artifact.id": request.artifact_id,
                    "tc.source.address": request.source_address,
                    "tc.source.port": int(request.source_port),
                    "tc.request.wait_timeout_ms": int(wait_timeout_ms),
                }
            )

            requested_view_id: str | None = None
            if request.HasField("requested_byte_space"):
                space = request.requested_byte_space
                if space.kind == common_pb2.BYTE_SPACE_KIND_VIEW:
                    if not space.id:
                        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                        context.set_details("requested_byte_space VIEW requires id")
                        return global_store_pb2.RequestReplicaTransportResponse(
                            status=global_store_pb2.Status.STATUS_ERROR
                        )
                    requested_view_id = space.id
                elif space.kind in (
                    common_pb2.BYTE_SPACE_KIND_CANONICAL,
                    common_pb2.BYTE_SPACE_KIND_UNSPECIFIED,
                ):
                    requested_view_id = None
                else:
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details("unsupported requested_byte_space kind")
                    return global_store_pb2.RequestReplicaTransportResponse(
                        status=global_store_pb2.Status.STATUS_ERROR
                    )

            # Request transport
            replica, transport_id = self.transport_service.request_transport(
                artifact_id=request.artifact_id,
                view_id=requested_view_id,
                source_node_id=request.source_node_id,
                source_address=request.source_address,
                source_port=request.source_port,
                wait_timeout_ms=wait_timeout_ms,
            )

            # Convert to proto format
            remote_info = self._replica_to_memory_info(replica)

            # Transport ID is known at this point (best-effort only)
            from contextlib import suppress

            with suppress(Exception):
                set_span_attributes({"tc.transport.id": str(transport_id)})

            return global_store_pb2.RequestReplicaTransportResponse(
                status=global_store_pb2.Status.STATUS_OK,
                remote_memory_info=remote_info,
                transport_id=str(transport_id),
            )

        except NotFoundError:
            logger.info(f"No replicas registered for artifact {request.artifact_id}")
            return global_store_pb2.RequestReplicaTransportResponse(
                status=global_store_pb2.Status.STATUS_NOT_FOUND
            )
        except TimeoutError:
            logger.warning(f"Timeout waiting for artifact {request.artifact_id}")
            return global_store_pb2.RequestReplicaTransportResponse(
                status=global_store_pb2.Status.STATUS_TIMED_OUT
            )
        except Exception as e:
            logger.exception("Error requesting artifact transport")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.RequestReplicaTransportResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def CompleteReplicaTransport(
        self,
        request: global_store_pb2.CompleteReplicaTransportRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.CompleteReplicaTransportResponse:
        """Complete artifact transport and release resources."""
        try:
            transport_id = UUID(request.transport_id)

            # Attach known attributes; CompleteReplicaTransportRequest only carries transport_id.
            set_span_attributes({"tc.transport.id": str(transport_id)})

            # Complete transport
            self.transport_service.complete_transport(transport_id)

            return global_store_pb2.CompleteReplicaTransportResponse(
                status=global_store_pb2.Status.STATUS_OK
            )

        except NotFoundError:
            logger.warning(f"Transport not found: {request.transport_id}")
            return global_store_pb2.CompleteReplicaTransportResponse(
                status=global_store_pb2.Status.STATUS_NOT_FOUND
            )
        except Exception as e:
            logger.exception("Error completing artifact transport")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.CompleteReplicaTransportResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    # ========== Key Mapping ==========

    def UpsertKeyMapping(
        self,
        request: global_store_pb2.UpsertKeyMappingRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UpsertKeyMappingResponse:
        return self.key_mapping_rpc_handler.upsert_key_mapping(request, context)

    def ResolveKeyMapping(
        self,
        request: global_store_pb2.ResolveKeyMappingRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.ResolveKeyMappingResponse:
        return self.key_mapping_rpc_handler.resolve_key_mapping(request, context)

    def SwapKeyMapping(
        self,
        request: global_store_pb2.SwapKeyMappingRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.SwapKeyMappingResponse:
        return self.key_mapping_rpc_handler.swap_key_mapping(request, context)

    def RevokeKeyMapping(
        self,
        request: global_store_pb2.RevokeKeyMappingRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.RevokeKeyMappingResponse:
        return self.key_mapping_rpc_handler.revoke_key_mapping(request, context)

    # ========== Worker Methods ==========

    def RegisterWorker(
        self,
        request: global_store_pb2.RegisterWorkerRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.RegisterWorkerResponse:
        """Register a new worker."""
        try:
            daemon_id = (request.daemon_id or "").strip()
            if not daemon_id:
                context.set_details("daemon_id is required")
                return global_store_pb2.RegisterWorkerResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )
            # Convert proto to domain artifact
            worker = Worker(
                daemon_id=daemon_id,
                node_id=request.node_id,
                node_address=request.node_address,
                grpc_port=request.grpc_port,
                p2p_port=request.p2p_port,
                mem_pool_total_size=request.mem_pool_total_size,
                mem_pool_available_size=request.mem_pool_available_size,
                capability_flags=int(request.capability_flags),
            )

            # Span attributes with worker metadata
            set_span_attributes(
                {
                    "tc.worker.daemon_id": str(worker.daemon_id),
                    "tc.worker.node_id": worker.node_id,
                    "tc.worker.node_address": worker.node_address,
                    "tc.worker.grpc_port": int(worker.grpc_port),
                    "tc.worker.p2p_port": int(worker.p2p_port),
                    "tc.mem_pool.total_bytes": int(worker.mem_pool_total_size),
                    "tc.mem_pool.available_bytes": int(worker.mem_pool_available_size),
                    "tc.worker.is_recovery": bool(request.is_recovery_registration),
                    "tc.worker.capability_flags": int(worker.capability_flags),
                }
            )

            # Check if this is a recovery registration
            is_recovery = request.is_recovery_registration
            previous_worker_id = (
                request.previous_worker_id if request.previous_worker_id else None
            )

            # Reject loopback/unspecified IPs up front for both normal and recovery registrations
            try:
                addr = ipaddress.ip_address(worker.node_address)
                if addr.is_loopback or addr.is_unspecified:
                    raise ValidationError(
                        f"Invalid node_address '{worker.node_address}'. Use a routable (non-loopback, non-unspecified) IP of the external interface; 127.0.0.1 and 0.0.0.0 are not allowed."
                    )
            except ValueError:
                # Not an IP literal; allow hostnames (may resolve to routable IPs)
                pass

            if is_recovery:
                # Handle recovery registration through recovery service
                success, state_sync_required = (
                    self.recovery_service.handle_worker_recovery_registration(
                        worker, previous_worker_id
                    )
                )

                if not success:
                    return global_store_pb2.RegisterWorkerResponse(
                        status=global_store_pb2.Status.STATUS_ERROR
                    )

                # Get the registered worker to get the worker_id
                registered = self.worker_service.find_worker_by_address(
                    worker.node_address, worker.grpc_port
                )

                if not registered:
                    return global_store_pb2.RegisterWorkerResponse(
                        status=global_store_pb2.Status.STATUS_ERROR
                    )

                # Single, enriched registration log (recovery)
                logger.info(
                    "Worker registered: worker_id=%s daemon_id=%s node_id=%s addr=%s:%d p2p=%d mem_total=%d mem_avail=%d is_recovery=%s prev_worker_id=%s state_sync_required=%s expected_state_version=%d",
                    registered.worker_id,
                    (registered.daemon_id or ""),
                    worker.node_id,
                    worker.node_address,
                    int(worker.grpc_port),
                    int(worker.p2p_port),
                    int(worker.mem_pool_total_size),
                    int(worker.mem_pool_available_size),
                    True,
                    (previous_worker_id or ""),
                    bool(state_sync_required),
                    int(
                        self.recovery_service.ensure_worker_state_version(
                            registered.worker_id
                        )
                    ),
                )

                return global_store_pb2.RegisterWorkerResponse(
                    status=global_store_pb2.Status.STATUS_OK,
                    worker_id=registered.worker_id,
                    heartbeat_interval_ms=self.config.default_heartbeat_interval_ms,
                    state_sync_required=state_sync_required,
                    expected_state_version=self.recovery_service.ensure_worker_state_version(
                        registered.worker_id
                    ),
                )
            else:
                # Normal registration
                # Reject cross-host duplicates on the same address:port; allow same-host restart/update
                existing = self.worker_service.find_worker_by_address(
                    worker.node_address, worker.grpc_port
                )
                if (
                    existing
                    and existing.node_id != worker.node_id
                    and existing.inactive_at is None
                ):
                    logger.error(
                        "Registration conflict: %s:%d already owned by worker_id=%s (node=%s); attempted by node=%s.",
                        worker.node_address,
                        worker.grpc_port,
                        existing.worker_id,
                        existing.node_id,
                        worker.node_id,
                    )
                    # Map to generic error status (client logs will include details)
                    return global_store_pb2.RegisterWorkerResponse(
                        status=global_store_pb2.Status.STATUS_ERROR
                    )

                registered = self.worker_service.register_worker(worker)
                expected_state_version = (
                    self.recovery_service.ensure_worker_state_version(
                        registered.worker_id
                    )
                )

                # Single, enriched registration log (normal)
                logger.info(
                    "Worker registered: worker_id=%s daemon_id=%s node_id=%s addr=%s:%d p2p=%d mem_total=%d mem_avail=%d is_recovery=%s",
                    registered.worker_id,
                    (registered.daemon_id or ""),
                    worker.node_id,
                    worker.node_address,
                    int(worker.grpc_port),
                    int(worker.p2p_port),
                    int(worker.mem_pool_total_size),
                    int(worker.mem_pool_available_size),
                    False,
                )

                return global_store_pb2.RegisterWorkerResponse(
                    status=global_store_pb2.Status.STATUS_OK,
                    worker_id=registered.worker_id,
                    heartbeat_interval_ms=self.config.default_heartbeat_interval_ms,
                    state_sync_required=False,
                    expected_state_version=expected_state_version,
                )

        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return global_store_pb2.RegisterWorkerResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except Exception as e:
            logger.exception("Error registering worker")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.RegisterWorkerResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def WorkerHeartbeat(
        self,
        request: global_store_pb2.WorkerHeartbeatRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.WorkerHeartbeatResponse:
        """Process worker heartbeat (enhanced-only)."""
        try:
            # Detect possible duplicate worker_id usage across different addresses
            try:
                w = self.worker_repository.find_by_id(request.worker_id)
                if w and request.HasField("state_version"):
                    # Best-effort detection: if the heartbeat's implied source differs from DB registration
                    # we log a warning to aid diagnosis of shared storage / misconfigured listen.host.
                    # Note: request doesn't carry node_address; this check is limited.
                    pass
            except Exception:
                # Non-fatal diagnostics
                logger.debug(
                    "worker lookup during heartbeat diagnostics failed", exc_info=True
                )
            set_span_attributes(
                {
                    "tc.worker.id": request.worker_id,
                    "tc.mem_pool.available_bytes": int(request.mem_pool_available_size),
                    "tc.worker.accepting_new_requests": bool(
                        request.accepting_new_requests
                    ),
                    "tc.worker.state_version": int(request.state_version),
                    "tc.worker.capability_flags": int(request.capability_flags),
                }
            )
            if request.state_version <= 0:
                logger.warning(
                    "Rejected legacy heartbeat for worker %s: state_version must be >= 1",
                    request.worker_id,
                )
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(
                    "state_version must be >= 1; legacy heartbeats are not supported"
                )
                return global_store_pb2.WorkerHeartbeatResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )

            return self._handle_enhanced_heartbeat(request, context)

        except Exception as e:
            logger.exception("Error processing worker heartbeat")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.WorkerHeartbeatResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def _handle_enhanced_heartbeat(
        self,
        request: global_store_pb2.WorkerHeartbeatRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.WorkerHeartbeatResponse:
        """Handle enhanced heartbeat with state synchronization."""
        try:
            capability_flags: int | None = None
            if request.HasField("capability_flags"):
                capability_flags = int(request.capability_flags)
            # Process basic heartbeat first
            success = self.worker_service.heartbeat(
                worker_id=request.worker_id,
                mem_pool_available_size=request.mem_pool_available_size,
                accepting_new_requests=request.accepting_new_requests,
                capability_flags=capability_flags,
            )

            if not success:
                return global_store_pb2.WorkerHeartbeatResponse(
                    status=global_store_pb2.Status.STATUS_NOT_FOUND
                )

            # Check if state synchronization is needed
            current_version = self.recovery_service.ensure_worker_state_version(
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

            # Detect obsolete artifacts reported by the worker but not present in global state
            obsolete_replicas: list[str] = []
            if request.registered_artifact_ids:
                obsolete_replicas = self.recovery_service.get_obsolete_artifacts(
                    request.worker_id, list(request.registered_artifact_ids)
                )

            # If obsolete artifacts exist instruct state sync
            if obsolete_replicas:
                state_sync_required = True

            ts = timestamp_pb2.Timestamp()
            ts.FromSeconds(int(time.time()))
            return global_store_pb2.WorkerHeartbeatResponse(
                status=global_store_pb2.Status.STATUS_OK,
                state_sync_required=state_sync_required,
                expected_state_version=current_version,
                obsolete_replicas=obsolete_replicas,
                server_timestamp_ts=ts,
            )

        except Exception as e:
            logger.exception(
                f"Error handling enhanced heartbeat for worker {request.worker_id}"
            )
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.WorkerHeartbeatResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def UnregisterWorker(
        self,
        request: global_store_pb2.UnregisterWorkerRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UnregisterWorkerResponse:
        """Unregister a worker."""
        try:
            # Pre-fetch worker details for enriched logging
            worker_before = None
            try:
                worker_before = self.worker_repository.find_by_id(
                    request.worker_id, include_inactive=True
                )
            except Exception:
                # Best-effort; proceed even if lookup fails
                worker_before = None

            success = self.worker_service.unregister_worker(request.worker_id)

            status = (
                global_store_pb2.Status.STATUS_OK
                if success
                else global_store_pb2.Status.STATUS_NOT_FOUND
            )
            if success:
                # Single, enriched deregistration log
                if worker_before:
                    logger.info(
                        "Worker unregistered: worker_id=%s graceful=%s node_id=%s addr=%s:%d p2p=%d",
                        request.worker_id,
                        getattr(request, "is_graceful_shutdown", False),
                        worker_before.node_id,
                        worker_before.node_address,
                        int(worker_before.grpc_port),
                        int(worker_before.p2p_port),
                    )
                else:
                    logger.info(
                        "Worker unregistered: worker_id=%s graceful=%s",
                        request.worker_id,
                        getattr(request, "is_graceful_shutdown", False),
                    )
            else:
                logger.warning(
                    "UnregisterWorker failed: worker %s not found",
                    request.worker_id,
                )

            return global_store_pb2.UnregisterWorkerResponse(status=status)

        except Exception as e:
            logger.exception("Error unregistering worker")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.UnregisterWorkerResponse(
                status=global_store_pb2.Status.STATUS_ERROR
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
            gs_metrics.set_capability_counts(scope="worker", entries=workers)
            required_flags = int(getattr(request, "required_capability_flags", 0))
            if required_flags:
                workers = [
                    worker
                    for worker in workers
                    if (int(worker.capability_flags) & required_flags) == required_flags
                ]

            # Convert to proto format
            worker_infos = []
            for worker in workers:
                # Build Timestamp for last heartbeat
                last_ts = timestamp_pb2.Timestamp()
                if worker.last_heartbeat:
                    last_ts.FromSeconds(int(worker.last_heartbeat.timestamp()))
                else:
                    last_ts.FromSeconds(0)

                worker_info = global_store_pb2.ListActiveWorkersResponse.WorkerInfo(
                    worker_id=worker.worker_id,
                    daemon_id=str(worker.daemon_id or ""),
                    node_id=worker.node_id,
                    node_address=worker.node_address,
                    grpc_port=worker.grpc_port,
                    p2p_port=worker.p2p_port,
                    mem_pool_total_size=worker.mem_pool_total_size,
                    mem_pool_available_size=worker.mem_pool_available_size,
                    accepting_new_requests=worker.accepting_new_requests,
                    capability_flags=int(worker.capability_flags),
                    last_heartbeat_ts=last_ts,
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

    # ========== Instance Methods ==========

    def RegisterInstance(
        self,
        request: global_store_pb2.RegisterInstanceRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.RegisterInstanceResponse:
        """Register a new engine instance."""
        try:
            instance = Instance(
                instance_id=request.instance_id,
                daemon_id=request.daemon_id,
                worker_id=request.worker_id if request.HasField("worker_id") else None,
                engine=request.engine,
                signals_endpoint=request.signals_endpoint or None,
                labels=dict(request.labels) if request.labels else {},
                capability_flags=int(request.capability_flags),
            )
            registered = self.instance_service.register_instance(instance)
            return global_store_pb2.RegisterInstanceResponse(
                status=global_store_pb2.Status.STATUS_OK,
                instance_id=registered.instance_id,
                heartbeat_interval_ms=self.config.default_heartbeat_interval_ms,
            )
        except ValidationError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return global_store_pb2.RegisterInstanceResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("Error registering instance")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.RegisterInstanceResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def InstanceHeartbeat(
        self,
        request: global_store_pb2.InstanceHeartbeatRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.InstanceHeartbeatResponse:
        """Process instance heartbeat."""
        try:
            worker_id = request.worker_id if request.HasField("worker_id") else None
            capability_flags: int | None = None
            if request.HasField("capability_flags"):
                capability_flags = int(request.capability_flags)
            success = self.instance_service.heartbeat(
                request.instance_id,
                worker_id=worker_id,
                capability_flags=capability_flags,
            )
            return global_store_pb2.InstanceHeartbeatResponse(
                status=global_store_pb2.Status.STATUS_OK
                if success
                else global_store_pb2.Status.STATUS_NOT_FOUND
            )
        except ValidationError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return global_store_pb2.InstanceHeartbeatResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("Error processing instance heartbeat")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.InstanceHeartbeatResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def UnregisterInstance(
        self,
        request: global_store_pb2.UnregisterInstanceRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UnregisterInstanceResponse:
        """Unregister an instance."""
        try:
            success = self.instance_service.unregister_instance(request.instance_id)
            return global_store_pb2.UnregisterInstanceResponse(
                status=global_store_pb2.Status.STATUS_OK
                if success
                else global_store_pb2.Status.STATUS_NOT_FOUND
            )
        except ValidationError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return global_store_pb2.UnregisterInstanceResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("Error unregistering instance")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.UnregisterInstanceResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def ListActiveInstances(
        self,
        request: global_store_pb2.ListActiveInstancesRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.ListActiveInstancesResponse:
        """List active instances."""
        try:
            instances = self.instance_service.list_active_instances(
                include_unavailable=request.include_unavailable
            )
            gs_metrics.set_capability_counts(scope="instance", entries=instances)
            required_flags = int(getattr(request, "required_capability_flags", 0))
            if required_flags:
                instances = [
                    inst
                    for inst in instances
                    if (int(inst.capability_flags) & required_flags) == required_flags
                ]
            infos: list[global_store_pb2.ListActiveInstancesResponse.InstanceInfo] = []
            for inst in instances:
                last_ts = timestamp_pb2.Timestamp()
                if inst.last_heartbeat:
                    last_ts.FromSeconds(int(inst.last_heartbeat.timestamp()))
                else:
                    last_ts.FromSeconds(0)
                infos.append(
                    global_store_pb2.ListActiveInstancesResponse.InstanceInfo(
                        instance_id=inst.instance_id,
                        daemon_id=inst.daemon_id,
                        worker_id=inst.worker_id or "",
                        engine=inst.engine,
                        signals_endpoint=inst.signals_endpoint or "",
                        last_heartbeat_ts=last_ts,
                        labels=dict(inst.labels) if inst.labels else {},
                        capability_flags=int(inst.capability_flags),
                        status=self._determine_instance_status(inst),
                    )
                )
            return global_store_pb2.ListActiveInstancesResponse(instances=infos)
        except Exception as e:  # noqa: BLE001
            logger.exception("Error listing active instances")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.ListActiveInstancesResponse()

    # ========== High Availability Methods ==========

    def SynchronizeWorkerState(
        self,
        request: global_store_pb2.SynchronizeWorkerStateRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.SynchronizeWorkerStateResponse:
        """Synchronize worker state for high availability."""
        try:
            success, state_changes, new_version, new_checksum, ignored = (
                self.recovery_service.synchronize_worker_state(
                    request.worker_id,
                    request.local_state,
                    request.sync_epoch,
                    request.sync_request_id,
                    request.force_full_sync,
                )
            )

            if success:
                return global_store_pb2.SynchronizeWorkerStateResponse(
                    status=global_store_pb2.Status.STATUS_OK,
                    new_state_version=new_version,
                    state_changes=state_changes,
                    new_state_checksum=new_checksum,
                    ignored=ignored,
                )
            else:
                return global_store_pb2.SynchronizeWorkerStateResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )

        except Exception as e:
            logger.exception(
                f"Error synchronizing worker state for {request.worker_id}"
            )
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.SynchronizeWorkerStateResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def RequestFullStateSync(
        self,
        request: global_store_pb2.RequestFullStateSyncRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.RequestFullStateSyncResponse:
        """Request full state synchronization for a worker."""
        try:
            success, expected_replicas, new_version, new_checksum, ignored = (
                self.recovery_service.request_full_state_sync(
                    request.worker_id,
                    request.sync_epoch,
                    request.sync_request_id,
                )
            )

            if success:
                return global_store_pb2.RequestFullStateSyncResponse(
                    status=global_store_pb2.Status.STATUS_OK,
                    new_state_version=new_version,
                    expected_replicas=expected_replicas,
                    new_state_checksum=new_checksum,
                    ignored=ignored,
                )
            else:
                return global_store_pb2.RequestFullStateSyncResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )

        except Exception as e:
            logger.exception(
                f"Error requesting full state sync for {request.worker_id}"
            )
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.RequestFullStateSyncResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    # ========== Utility Methods ==========

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
                request.artifact_id, chunk_indices
            )

            return global_store_pb2.QueryChunkLocationsResponse(
                status=global_store_pb2.Status.STATUS_OK,
                locations=locations,
            )

        except Exception as e:
            logger.exception(f"Error querying chunk locations: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.QueryChunkLocationsResponse(
                status=global_store_pb2.Status.STATUS_ERROR
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
                status=global_store_pb2.Status.STATUS_OK,
                updates_applied=updates_applied,
            )

        except Exception as e:
            logger.exception(f"Error updating chunk states: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.BatchUpdateChunkStatesResponse(
                status=global_store_pb2.Status.STATUS_ERROR,
                updates_applied=0,
            )

    # ========== Placement & Persistence Methods ==========

    def PlanPlacement(
        self,
        request: global_store_pb2.PlanPlacementRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.PlanPlacementResponse:
        if not request.artifact_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("artifact_id is required")
            return global_store_pb2.PlanPlacementResponse()
        if not request.source_node_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("source_node_id is required")
            return global_store_pb2.PlanPlacementResponse()
        try:
            policy = self._policy_from_proto(request.placement_policy)
        except ValidationError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.PlanPlacementResponse()

        shard_models: list[PlacementShard] = []
        for shard in request.shards:
            shard_id = shard.shard_id or f"{request.artifact_id}:{shard.shard_idx}"
            shard_models.append(
                PlacementShard(
                    plan_id="",
                    shard_idx=shard.shard_idx,
                    shard_id=shard_id,
                    size_bytes=shard.size_bytes,
                    content_digest=shard.content_digest,
                    byte_range_start=shard.byte_range_start,
                    byte_range_length=shard.byte_range_length,
                    chunk_ids=list(shard.chunk_ids),
                )
            )

        try:
            plan = self.placement_service.plan_placement(
                request.artifact_id,
                policy,
                shard_models,
                source_node_id=request.source_node_id,
            )
        except ValidationError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.PlanPlacementResponse()
        except Exception:
            logger.exception("Failed to plan placement")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Failed to plan placement")
            return global_store_pb2.PlanPlacementResponse()

        return self._plan_to_proto(plan)

    def ReportPersistenceStatus(
        self,
        request: global_store_pb2.ReportPersistenceStatusRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.ReportPersistenceStatusResponse:
        if not request.task_id or not request.plan_id or not request.artifact_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("task_id, plan_id, and artifact_id are required")
            return global_store_pb2.ReportPersistenceStatusResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        try:
            state = self._persistence_state_from_proto(request.state)
        except ValidationError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.ReportPersistenceStatusResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

        shard_statuses: list[PersistenceShardStatus] = []
        try:
            for shard in request.shard_statuses:
                shard_state = self._persistence_state_from_proto(shard.state)
                targets = [
                    PlacementTarget(
                        plan_id=request.plan_id,
                        shard_idx=shard.shard_idx,
                        node_id=target.node_id,
                        lease_id=target.lease_id or None,
                        target_state=self._target_state_from_proto(target.target_state),
                        degraded_reason=target.degraded_reason or None,
                    )
                    for target in shard.targets
                ]
                shard_statuses.append(
                    PersistenceShardStatus(
                        shard_id=shard.shard_id,
                        shard_idx=shard.shard_idx,
                        state=shard_state,
                        progress=shard.progress,
                        degraded_reason=shard.degraded_reason or None,
                        last_error=shard.last_error or None,
                        targets=targets,
                    )
                )
        except ValidationError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.ReportPersistenceStatusResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except Exception:
            logger.exception("Failed to decode shard status payload")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Failed to decode shard status payload")
            return global_store_pb2.ReportPersistenceStatusResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

        try:
            status_model = PersistenceStatus(
                task_id=request.task_id,
                plan_id=request.plan_id,
                artifact_id=request.artifact_id,
                state=state,
                progress=request.progress,
                last_error=request.last_error or None,
                degraded_reason=request.degraded_reason or None,
            )
            self.placement_service.record_status(status_model, shard_statuses)
        except Exception:
            logger.exception("Failed to persist ReportPersistenceStatus")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Failed to persist ReportPersistenceStatus")
            return global_store_pb2.ReportPersistenceStatusResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        return global_store_pb2.ReportPersistenceStatusResponse(
            status=global_store_pb2.Status.STATUS_OK
        )

    # ========== Disk Locations ==========

    def UpsertArtifactDiskLocation(
        self,
        request: global_store_pb2.UpsertArtifactDiskLocationRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UpsertArtifactDiskLocationResponse:
        try:
            artifact_id = request.artifact_id.strip()
            cluster_id = request.cluster_id.strip()
            relative_path = request.relative_path.strip()
            if not artifact_id or not cluster_id or not relative_path:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(
                    "artifact_id, cluster_id, and relative_path are required"
                )
                return global_store_pb2.UpsertArtifactDiskLocationResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )
            if cluster_id != self.cluster_id:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("cluster_id does not match server cluster_id")
                return global_store_pb2.UpsertArtifactDiskLocationResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )
            if not self._is_safe_relative_path(relative_path):
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("relative_path must be a safe, relative path")
                return global_store_pb2.UpsertArtifactDiskLocationResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )
            kind = self._DISK_LOCATION_KIND_FROM_PROTO.get(request.kind, "MANAGED")
            self.disk_location_repository.upsert(
                artifact_id=artifact_id,
                cluster_id=cluster_id,
                relative_path=relative_path,
                kind=kind,
                is_deleted=bool(request.is_deleted),
            )
            return global_store_pb2.UpsertArtifactDiskLocationResponse(
                status=global_store_pb2.Status.STATUS_OK
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("Error in UpsertArtifactDiskLocation")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.UpsertArtifactDiskLocationResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def ListArtifactDiskLocations(
        self,
        request: global_store_pb2.ListArtifactDiskLocationsRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.ListArtifactDiskLocationsResponse:
        try:
            artifact_id = request.artifact_id.strip()
            if not artifact_id:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("artifact_id is required")
                return global_store_pb2.ListArtifactDiskLocationsResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )
            include_deleted = bool(request.include_deleted)
            rows = self.disk_location_repository.list_by_artifact(
                artifact_id, include_deleted=include_deleted
            )
            locations: list[global_store_pb2.ArtifactDiskLocation] = []
            for row in rows:
                created = self._datetime_to_timestamp(
                    self._coerce_db_datetime(row.get("created_at"))
                )
                updated = self._datetime_to_timestamp(
                    self._coerce_db_datetime(row.get("updated_at"))
                )
                deleted_at = None
                if row.get("deleted_at") is not None:
                    deleted_at = self._datetime_to_timestamp(
                        self._coerce_db_datetime(row.get("deleted_at"))
                    )
                kind = self._DISK_LOCATION_KIND_TO_PROTO.get(
                    (row.get("kind") or "MANAGED").upper(),
                    global_store_pb2.DISK_LOCATION_KIND_MANAGED,
                )
                msg = global_store_pb2.ArtifactDiskLocation(
                    artifact_id=row.get("artifact_id", ""),
                    cluster_id=row.get("cluster_id", ""),
                    relative_path=row.get("relative_path", ""),
                    kind=kind,
                    created_at=created,
                    updated_at=updated,
                    is_deleted=bool(row.get("is_deleted", False)),
                )
                if deleted_at is not None:
                    msg.deleted_at.CopyFrom(deleted_at)
                locations.append(msg)
            return global_store_pb2.ListArtifactDiskLocationsResponse(
                status=global_store_pb2.Status.STATUS_OK, locations=locations
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("Error in ListArtifactDiskLocations")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.ListArtifactDiskLocationsResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

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
        if state == common_pb2.ReplicaTransportMetadata.EXPORT_STATE_EXPORTABLE:
            return ExportState.EXPORTABLE
        if state == common_pb2.ReplicaTransportMetadata.EXPORT_STATE_DRAINING:
            return ExportState.DRAINING
        return ExportState.PRESENCE_ONLY

    @staticmethod
    def _export_state_to_proto(
        state: ExportState,
    ) -> common_pb2.ReplicaTransportMetadata.ExportState:
        if state is ExportState.EXPORTABLE:
            return common_pb2.ReplicaTransportMetadata.EXPORT_STATE_EXPORTABLE
        if state is ExportState.DRAINING:
            return common_pb2.ReplicaTransportMetadata.EXPORT_STATE_DRAINING
        return common_pb2.ReplicaTransportMetadata.EXPORT_STATE_PRESENCE_ONLY

    def _parse_transport_metadata(
        self, mem_info: common_pb2.MemoryInfo
    ) -> tuple[bool, ExportState, int, list[str], list[int], str | None]:
        transport_authoritative = mem_info.HasField("transport")
        export_state = ExportState.PRESENCE_ONLY
        export_generation = 0
        remote_keys: list[str] = []
        buffer_sizes: list[int] = []
        verification_json: str | None = None

        if not transport_authoritative:
            # New protocol: without `transport` presence, this is a presence-only
            # update and must not update transport metadata via legacy fields.
            return (
                transport_authoritative,
                export_state,
                export_generation,
                remote_keys,
                buffer_sizes,
                verification_json,
            )

        transport = mem_info.transport
        export_state = self._export_state_from_proto(transport.export_state)
        export_generation = int(transport.export_generation or 0)
        remote_keys = list(transport.remote_memory_keys)
        buffer_sizes = [int(size) for size in transport.buffer_sizes]
        verification_json = (
            transport.verification_json if transport.verification_json else None
        )

        # Presence-only is a routing-withdraw state and must not retain keys or
        # export-bound verification metadata.
        if export_state is ExportState.PRESENCE_ONLY:
            return (
                transport_authoritative,
                export_state,
                export_generation,
                [],
                [],
                None,
            )

        keys_required = export_state is ExportState.EXPORTABLE
        if keys_required and not remote_keys:
            export_state = ExportState.PRESENCE_ONLY
            remote_keys = []
            buffer_sizes = []
            verification_json = None

        # Validate keys when provided. Draining may omit keys entirely.
        if remote_keys or buffer_sizes:
            keys_valid = True
            if len(remote_keys) != len(buffer_sizes):
                keys_valid = False
            else:
                total = 0
                for size in buffer_sizes:
                    if size <= 0:
                        keys_valid = False
                        break
                    total += int(size)
                if keys_valid and mem_info.memory_size > 0:
                    keys_valid = total == mem_info.memory_size
            if not keys_valid:
                remote_keys = []
                buffer_sizes = []
                export_state = ExportState.PRESENCE_ONLY
                verification_json = None

        return (
            transport_authoritative,
            export_state,
            export_generation,
            remote_keys,
            buffer_sizes,
            verification_json,
        )

    def _replica_to_memory_info(self, replica: Replica) -> common_pb2.MemoryInfo:
        """Convert Replica to MemoryInfo proto."""
        # Map domain enum to proto enum
        if replica.memory_type == MemoryType.GPU:
            proto_mem_type = common_pb2.MemoryType.MEMORY_TYPE_GPU
        elif replica.memory_type == MemoryType.RAM:
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
            verification_json=replica.verification_json or "",
        )
        transport = memory_info.transport
        transport.export_state = self._export_state_to_proto(replica.export_state)
        transport.export_generation = int(replica.export_generation or 0)
        transport.remote_memory_keys.extend(replica.remote_memory_keys)
        transport.buffer_sizes.extend([int(size) for size in replica.buffer_sizes])
        if replica.verification_json:
            transport.verification_json = replica.verification_json
        if replica.byte_space.kind == ByteSpaceKind.VIEW:
            memory_info.byte_space.CopyFrom(
                common_pb2.ByteSpaceRef(
                    kind=common_pb2.BYTE_SPACE_KIND_VIEW,
                    id=replica.byte_space.id or "",
                )
            )
        else:
            memory_info.byte_space.CopyFrom(
                common_pb2.ByteSpaceRef(kind=common_pb2.BYTE_SPACE_KIND_CANONICAL)
            )
        creation_proto = self._datetime_to_timestamp(replica.created_at)
        if creation_proto is not None:
            memory_info.creation_ts.CopyFrom(creation_proto)
        expires_proto = self._datetime_to_timestamp(replica.expires_at)
        if expires_proto is not None:
            memory_info.expires_at.CopyFrom(expires_proto)
        return memory_info

    def _memory_info_to_replica_artifact_id(
        self,
        mem_info: common_pb2.MemoryInfo,
        artifact_id: str,
        max_concurrency: int,
        worker_id: str,
    ) -> Replica:
        """Convert MemoryInfo proto to Replica using content-addressed artifact_id."""
        (
            transport_authoritative,
            export_state,
            export_generation,
            remote_keys,
            buffer_sizes,
            verification_json,
        ) = self._parse_transport_metadata(mem_info)

        # Map proto enum to domain enum
        if mem_info.memory_type == common_pb2.MemoryType.MEMORY_TYPE_GPU:
            domain_mem_type = MemoryType.GPU
        elif mem_info.memory_type == common_pb2.MemoryType.MEMORY_TYPE_RAM:
            domain_mem_type = MemoryType.RAM
        else:
            domain_mem_type = MemoryType.DISK

        byte_space = ByteSpaceRef.canonical()
        if mem_info.HasField("byte_space"):
            if mem_info.byte_space.kind == common_pb2.BYTE_SPACE_KIND_VIEW:
                view_id = mem_info.byte_space.id.strip()
                if not view_id:
                    raise ValidationError("byte_space VIEW requires id")
                byte_space = ByteSpaceRef.view(view_id)
            elif mem_info.byte_space.kind in (
                common_pb2.BYTE_SPACE_KIND_CANONICAL,
                common_pb2.BYTE_SPACE_KIND_UNSPECIFIED,
            ):
                byte_space = ByteSpaceRef.canonical()

        return Replica(
            artifact_id=artifact_id,
            byte_space=byte_space,
            node_id=mem_info.node_id,
            node_address=mem_info.node_address,
            node_port=mem_info.node_port,
            memory_size=mem_info.memory_size,
            memory_type=domain_mem_type,
            device_id=mem_info.device_id,
            max_concurrency=max_concurrency,
            remote_memory_keys=remote_keys,
            buffer_sizes=buffer_sizes,
            export_state=export_state,
            export_generation=export_generation,
            worker_id=worker_id,
            verification_json=verification_json,
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
        cursor = self.connection.cursor()

        # Order matters due to foreign-key constraints (replica_counters ➜ artifact_replicas)
        tables = [
            "artifact_transports",  # Depends on artifact_replicas via replica_id FK
            "replica_counters",
            "artifact_replicas",
            "artifact_disk_locations",
            "artifact_indices",
            "artifacts",
            "workers",
            "instances",
        ]
        for table in tables:
            try:
                cursor.execute(f"DELETE FROM {table}")
            except Exception:  # noqa: BLE001 – best-effort cleanup for tests
                logger.exception(f"Failed to truncate table {table} during reset_state")

        # Reset Prometheus gauges that might retain state across tests
        try:
            from tensorcast.global_store import metrics as gs_metrics  # local import

            gs_metrics.ACTIVE_TRANSPORTS_GAUGE.set(0)
        except Exception:
            # Metric subsystem not critical for state reset – ignore
            logger.debug("Failed to reset Prometheus gauges during reset_state")

        # Ensure all outstanding metrics / in-memory counters in services are in sync.
        # The simplest way is to recreate the repositories & services bound to the
        # existing connection so that any internal caches are discarded.
        self.replica_repository = ReplicaRepository(self.connection)
        self.replica_repository = ReplicaRepository(self.connection)
        self.transport_repository = TransportRepository(self.connection)
        self.worker_repository = WorkerRepository(self.connection)
        self.instance_repository = InstanceRepository(self.connection)

        self.artifact_service = ArtifactService(self.replica_repository)
        self.transport_service = TransportService(
            self.replica_repository, self.transport_repository
        )
        self.worker_service = WorkerService(
            self.worker_repository, self.replica_repository
        )
        self.recovery_service = RecoveryService(
            self.worker_repository, self.replica_repository, self.worker_service
        )

        logger.debug("GlobalStoreServicer state has been reset for the next test run")
