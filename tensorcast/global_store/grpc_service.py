#  Copyright (c) 2025-2026, TensorCast Team.

"""
gRPC service implementation for Global Store.

This provides the gRPC interface layer, delegating business logic to services.
"""

import base64
import binascii
import ipaddress
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional, cast
from uuid import UUID

import duckdb  # DuckDB is a runtime dependency; ignore missing stubs in type checker
import grpc
from google.protobuf import timestamp_pb2

from tensorcast.common.identity import ArtifactIdKind, infer_artifact_id_kind
from tensorcast.global_store.config import get_config
from tensorcast.global_store.db_utils import init_db, optimize_db
from tensorcast.global_store.exceptions import (
    NotFoundError,
    TimeoutError,
    ValidationError,
)
from tensorcast.global_store.models import (
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
    ArtifactPersistenceStatusRepository,
    ArtifactPlacementRepository,
    ChunkDirectoryRepository,
    LeafRepository,
    ReplicaRepository,
    TransportRepository,
    VariantRepository,
    WorkerRepository,
)
from tensorcast.global_store.repositories.artifact_index_repository import (
    ArtifactIndexRepository,
)
from tensorcast.global_store.repositories.artifact_repository import ArtifactRepository
from tensorcast.global_store.repositories.key_mapping_repository import (
    KeyMappingRepository,
)
from tensorcast.global_store.services import (
    ArtifactService,
    ChunkService,
    PlacementService,
    RecoveryService,
    TransportService,
    ViewStateService,
    WorkerService,
)
from tensorcast.global_store.services.view_state_service import (
    LeafWritePayload,
    VariantUpsertPayload,
)
from tensorcast.logger import init_logger
from tensorcast.observability.otel import set_span_attributes
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.global_store.v1 import (
    global_store_pb2,
    global_store_pb2_grpc,
)

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
        self.replica_repository = ReplicaRepository(self.connection)

        self.artifacts_repo = ArtifactRepository(self.connection)
        self.artifact_indices = ArtifactIndexRepository(self.connection)
        self.transport_repository = TransportRepository(self.connection)
        self.worker_repository = WorkerRepository(self.connection)
        self.chunk_directory_repository = ChunkDirectoryRepository(self.connection)
        self.variant_repository = VariantRepository(self.connection)
        self.leaf_repository = LeafRepository(self.connection)
        self.key_mapping_repository = KeyMappingRepository(self.connection)
        self.placement_repository = ArtifactPlacementRepository(self.connection)
        self.persistence_status_repository = ArtifactPersistenceStatusRepository(
            self.connection
        )

        # Initialize services
        self.artifact_service = ArtifactService(self.replica_repository)
        self.transport_service = TransportService(
            self.replica_repository, self.transport_repository
        )
        self.worker_service = WorkerService(
            self.worker_repository, self.replica_repository
        )
        self.chunk_service = ChunkService(self.chunk_directory_repository)
        self.view_state_service = ViewStateService(
            self.variant_repository, self.leaf_repository
        )
        self.placement_service = PlacementService(
            self.worker_repository,
            self.placement_repository,
            self.persistence_status_repository,
        )

        # Initialize recovery service for high availability
        self.recovery_service = RecoveryService(
            self.worker_repository, self.replica_repository
        )

        self._initiate_startup_recovery()

        # Start background cleanup thread
        self._start_cleanup_thread()
        # Cleanup and optimization are now handled by a single maintenance thread

    def set_runtime_info(
        self,
        *,
        listen_host: str | None,
        listen_port: int | None,
        metrics_port: int | None,
        cluster_token: str | None,
        db_file: str | None,
        version: str | None,
    ) -> None:
        self._runtime_info = {
            "listen_host": listen_host,
            "listen_port": listen_port,
            "metrics_port": metrics_port,
            "cluster_token": cluster_token,
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

            space_field = request.WhichOneof("space")
            space_kind: Optional[str] = None  # 'C' or 'V'
            space_id: Optional[str] = None

            if include_leaves or include_view_meta or space_field is not None:
                if space_field is None:
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details(
                        "space selector required when requesting metadata or leaves"
                    )
                    return global_store_pb2.GetArtifactInfoByIdResponse(
                        status=global_store_pb2.Status.STATUS_ERROR
                    )
                if space_field == "canonical":
                    if not request.canonical:
                        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                        context.set_details("canonical must be true when selected")
                        return global_store_pb2.GetArtifactInfoByIdResponse(
                            status=global_store_pb2.Status.STATUS_ERROR
                        )
                    space_kind = "C"
                    if not artifact_row or not artifact_row.get("index_multihash"):
                        context.set_code(grpc.StatusCode.NOT_FOUND)
                        context.set_details("canonical index not recorded")
                        return global_store_pb2.GetArtifactInfoByIdResponse(
                            status=global_store_pb2.Status.STATUS_NOT_FOUND
                        )
                    space_id = cast(str, artifact_row["index_multihash"])
                elif space_field == "view_id":
                    view_id = request.view_id
                    if not view_id:
                        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                        context.set_details(
                            "view_id is required when selecting variant space"
                        )
                        return global_store_pb2.GetArtifactInfoByIdResponse(
                            status=global_store_pb2.Status.STATUS_ERROR
                        )
                    space_kind = "V"
                    space_id = view_id
                else:
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details("unsupported space selector")
                    return global_store_pb2.GetArtifactInfoByIdResponse(
                        status=global_store_pb2.Status.STATUS_ERROR
                    )

            available_replicas: list[common_pb2.MemoryInfo] = []
            if include_replicas:
                replicas = self.artifact_service.get_artifact_replicas(artifact_id)
                available_replicas = [self._replica_to_memory_info(r) for r in replicas]

            view_meta_msg: Optional[global_store_pb2.ViewMeta] = None
            leaves_proto: list[global_store_pb2.Leaf] = []
            partial_details: list[global_store_pb2.PartialCoverageDetail] = []
            leaf_filter: Optional[list[int]] = None
            variant_missing = False
            partial_leaf_miss = False

            variant_row: Optional[dict[str, object]] = None
            if space_kind == "V" and (include_leaves or include_view_meta):
                variant_row = self.view_state_service.get_variant(
                    artifact_id=artifact_id, view_id=space_id or ""
                )
                if variant_row is None:
                    variant_missing = True
                    context.set_code(grpc.StatusCode.NOT_FOUND)
                    context.set_details("variant metadata not found")

            if include_view_meta:
                if space_kind != "V":
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details(
                        "view metadata is only available for variant space"
                    )
                    return global_store_pb2.GetArtifactInfoByIdResponse(
                        status=global_store_pb2.Status.STATUS_ERROR
                    )
                if variant_row is not None:
                    view_size_value = cast(int, variant_row["view_size"])
                    view_meta_msg = global_store_pb2.ViewMeta(
                        view_spec_json=str(variant_row["view_spec_json"]),
                        view_size=int(view_size_value),
                    )
                    view_data_hash = variant_row.get("view_data_hash")
                    if view_data_hash:
                        view_meta_msg.view_data_hash = str(view_data_hash)
                    verified_at = self._coerce_db_datetime(
                        variant_row.get("verified_at")
                    )
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
                leaf_filter = list(request.leaf_idxs) if request.leaf_idxs else None
                if space_kind == "V" and variant_row is None:
                    partial_leaf_miss = True
                    detail = global_store_pb2.PartialCoverageDetail(
                        space_kind=global_store_pb2.BYTE_SPACE_KIND_VARIANT,
                        space_id=space_id or "",
                    )
                    if leaf_filter:
                        for idx in sorted(set(leaf_filter)):
                            detail.missing_ranges.append(
                                global_store_pb2.Range(off=int(idx), len=1)
                            )
                    partial_details.append(detail)
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
                        detail = global_store_pb2.PartialCoverageDetail(
                            space_kind=global_store_pb2.BYTE_SPACE_KIND_VARIANT
                            if space_kind == "V"
                            else global_store_pb2.BYTE_SPACE_KIND_CANONICAL,
                            space_id=space_id or "",
                        )
                        existing = {leaf.leaf_idx for leaf in leaves_proto}
                        missing = sorted(
                            idx for idx in set(leaf_filter) if idx not in existing
                        )
                        for idx in missing:
                            detail.missing_ranges.append(
                                global_store_pb2.Range(off=int(idx), len=1)
                            )
                        partial_details.append(detail)

            has_payload = False
            if include_replicas and available_replicas:
                has_payload = True
            if include_leaves and leaves_proto:
                has_payload = True
            if include_view_meta and view_meta_msg is not None:
                has_payload = True

            need_not_found = (
                variant_missing
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
            if partial_details:
                response.partial_coverage.extend(partial_details)
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
        """Upsert variant metadata and leaf digests."""
        artifact_id = request.artifact_id
        if not artifact_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("artifact_id is required")
            return global_store_pb2.UpdateArtifactViewStateResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

        try:
            variant_payload: Optional[VariantUpsertPayload] = None
            if request.HasField("variant"):
                variant = request.variant
                if not variant.view_id:
                    raise ValidationError("variant.view_id is required")
                if not variant.view_spec_json:
                    raise ValidationError("variant.view_spec_json is required")
                if variant.view_size <= 0:
                    raise ValidationError("variant.view_size must be positive")
                verified_at = (
                    self._timestamp_to_datetime(variant.verified_at)
                    if variant.HasField("verified_at")
                    else None
                )
                view_data_hash = variant.view_data_hash or None
                canonical_size: Optional[int] = None
                canonical_covered: Optional[int] = None
                if variant.HasField("canonical_coverage"):
                    canonical_size = int(variant.canonical_coverage.total_bytes)
                    canonical_covered = int(variant.canonical_coverage.covered_bytes)
                variant_payload = VariantUpsertPayload(
                    artifact_id=artifact_id,
                    view_id=variant.view_id,
                    view_spec_json=variant.view_spec_json,
                    view_size=variant.view_size,
                    view_data_hash=view_data_hash,
                    verified_at=verified_at,
                    canonical_size_bytes=canonical_size,
                    canonical_bytes_covered=canonical_covered,
                )

            leaf_payloads: list[LeafWritePayload] = []
            for leaf in request.leaf_writes:
                if leaf.space_kind == global_store_pb2.BYTE_SPACE_KIND_UNSPECIFIED:
                    raise ValidationError("leaf.space_kind must be set")
                if not leaf.space_id:
                    raise ValidationError("leaf.space_id is required")
                if not leaf.digest:
                    raise ValidationError("leaf.digest is required")
                if leaf.space_kind == global_store_pb2.BYTE_SPACE_KIND_CANONICAL:
                    space_kind = "C"
                elif leaf.space_kind == global_store_pb2.BYTE_SPACE_KIND_VARIANT:
                    space_kind = "V"
                else:
                    raise ValidationError(f"Unsupported space_kind: {leaf.space_kind}")

                leaf_payloads.append(
                    LeafWritePayload(
                        artifact_id=artifact_id,
                        space_kind=space_kind,
                        space_id=leaf.space_id,
                        leaf_idx=leaf.leaf_idx,
                        digest=bytes(leaf.digest),
                    )
                )

            self.view_state_service.update_view_state(
                variant=variant_payload,
                leaf_writes=leaf_payloads,
            )
            return global_store_pb2.UpdateArtifactViewStateResponse(
                status=global_store_pb2.Status.STATUS_OK
            )
        except ValidationError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.UpdateArtifactViewStateResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except ValueError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
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

            # Register replica
            registered = self.artifact_service.register_replica(replica)

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

            # RFC-0007: Persist content-addressed descriptor into `artifacts` table if possible.
            # Parse `mi2:` artifact_id and extract index/data multihash when present. If the
            # upstream StoreDaemon later extends the proto to include descriptor, prefer that.
            artifact_id = registered.artifact_id
            kind = infer_artifact_id_kind(artifact_id) if artifact_id else None
            if artifact_id and kind is ArtifactIdKind.MI2:
                parts = artifact_id.split(":", 2)
                index_mh = parts[1] if len(parts) == 3 else None
                data_mh = parts[2] if len(parts) == 3 else None
                encoding = "json"
                try:
                    self.artifacts_repo.upsert_artifact(
                        artifact_id=artifact_id,
                        index_multihash=index_mh,
                        data_multihash=data_mh,
                        schema_version=schema_version_value,
                        encoding=encoding,
                        hash_params_json=None,
                        id_kind="MI2",
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        f"Failed to upsert artifacts entry for {artifact_id}: {e}"
                    )
            elif artifact_id and kind is ArtifactIdKind.CGID:
                try:
                    self.artifacts_repo.upsert_artifact(
                        artifact_id=artifact_id,
                        index_multihash=None,
                        data_multihash=None,
                        schema_version=schema_version_value,
                        encoding="json",
                        hash_params_json=None,
                        id_kind="CGID",
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        f"Failed to upsert CGID artifact entry for {artifact_id}: {e}"
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

            # Request transport
            replica, transport_id = self.transport_service.request_transport(
                artifact_id=request.artifact_id,
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
        """Create or update a key → artifact mapping with uniqueness check.

        Conflict when the key already exists but points to a different artifact_id.
        """
        try:
            key = request.key.strip()
            artifact_id = request.artifact_id.strip()
            if not key or not artifact_id:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("key and artifact_id are required")
                return global_store_pb2.UpsertKeyMappingResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )

            existing = self.key_mapping_repository.get(key)
            if existing and existing.get("artifact_id") != artifact_id:
                return global_store_pb2.UpsertKeyMappingResponse(
                    status=global_store_pb2.Status.STATUS_ERROR,
                    conflict_reason=(
                        f"key already mapped to {existing.get('artifact_id')}"
                    ),
                )

            ttl_seconds = None
            if request.HasField("ttl"):
                d = request.ttl
                ttl_seconds = int(d.seconds + (d.nanos // 1_000_000_000))

            self.key_mapping_repository.upsert(
                key=key,
                artifact_id=artifact_id,
                replica_uuid=(request.replica_uuid or None),
                daemon_address=(request.daemon_address or None),
                disk_path=(request.disk_path or None),
                ttl_seconds=ttl_seconds,
            )
            return global_store_pb2.UpsertKeyMappingResponse(
                status=global_store_pb2.Status.STATUS_OK
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("Error in UpsertKeyMapping")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.UpsertKeyMappingResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def ResolveKeyMapping(
        self,
        request: global_store_pb2.ResolveKeyMappingRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.ResolveKeyMappingResponse:
        try:
            key = request.key.strip()
            if not key:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("key is required")
                return global_store_pb2.ResolveKeyMappingResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )
            row = self.key_mapping_repository.get(key)
            if not row:
                return global_store_pb2.ResolveKeyMappingResponse(
                    status=global_store_pb2.Status.STATUS_NOT_FOUND
                )
            return global_store_pb2.ResolveKeyMappingResponse(
                status=global_store_pb2.Status.STATUS_OK,
                artifact_id=row.get("artifact_id", ""),
                replica_uuid=row.get("replica_uuid", "") or "",
                daemon_address=row.get("daemon_address", "") or "",
                disk_path=row.get("disk_path", "") or "",
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("Error in ResolveKeyMapping")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.ResolveKeyMappingResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def RevokeKeyMapping(
        self,
        request: global_store_pb2.RevokeKeyMappingRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.RevokeKeyMappingResponse:
        try:
            key = request.key.strip()
            if not key:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("key is required")
                return global_store_pb2.RevokeKeyMappingResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )
            ok = self.key_mapping_repository.delete(key)
            return global_store_pb2.RevokeKeyMappingResponse(
                status=(
                    global_store_pb2.Status.STATUS_OK
                    if ok
                    else global_store_pb2.Status.STATUS_NOT_FOUND
                )
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("Error in RevokeKeyMapping")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return global_store_pb2.RevokeKeyMappingResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    # ========== Worker Methods ==========

    def RegisterWorker(
        self,
        request: global_store_pb2.RegisterWorkerRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.RegisterWorkerResponse:
        """Register a new worker."""
        try:
            # Convert proto to domain artifact
            worker = Worker(
                node_id=request.node_id,
                node_address=request.node_address,
                grpc_port=request.grpc_port,
                p2p_port=request.p2p_port,
                mem_pool_total_size=request.mem_pool_total_size,
                mem_pool_available_size=request.mem_pool_available_size,
            )

            # Span attributes with worker metadata
            set_span_attributes(
                {
                    "tc.worker.node_id": worker.node_id,
                    "tc.worker.node_address": worker.node_address,
                    "tc.worker.grpc_port": int(worker.grpc_port),
                    "tc.worker.p2p_port": int(worker.p2p_port),
                    "tc.mem_pool.total_bytes": int(worker.mem_pool_total_size),
                    "tc.mem_pool.available_bytes": int(worker.mem_pool_available_size),
                    "tc.worker.is_recovery": bool(request.is_recovery_registration),
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
                    "Worker registered: worker_id=%s node_id=%s addr=%s:%d p2p=%d mem_total=%d mem_avail=%d is_recovery=%s prev_worker_id=%s state_sync_required=%s expected_state_version=%d",
                    registered.worker_id,
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
                if existing and existing.node_id != worker.node_id:
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
                    "Worker registered: worker_id=%s node_id=%s addr=%s:%d p2p=%d mem_total=%d mem_avail=%d is_recovery=%s",
                    registered.worker_id,
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
            # Process basic heartbeat first
            success = self.worker_service.heartbeat(
                worker_id=request.worker_id,
                mem_pool_available_size=request.mem_pool_available_size,
                accepting_new_requests=request.accepting_new_requests,
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
                worker_before = self.worker_repository.find_by_id(request.worker_id)
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
                    node_id=worker.node_id,
                    node_address=worker.node_address,
                    grpc_port=worker.grpc_port,
                    p2p_port=worker.p2p_port,
                    mem_pool_total_size=worker.mem_pool_total_size,
                    mem_pool_available_size=worker.mem_pool_available_size,
                    accepting_new_requests=worker.accepting_new_requests,
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
                    request.worker_id, request.local_state, request.force_full_sync
                )
            )

            if success:
                return global_store_pb2.SynchronizeWorkerStateResponse(
                    status=global_store_pb2.Status.STATUS_OK,
                    new_state_version=new_version,
                    state_changes=state_changes,
                    new_state_checksum=new_checksum,
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
            success, expected_replicas, new_version, new_checksum = (
                self.recovery_service.request_full_state_sync(request.worker_id)
            )

            if success:
                return global_store_pb2.RequestFullStateSyncResponse(
                    status=global_store_pb2.Status.STATUS_OK,
                    new_state_version=new_version,
                    expected_replicas=expected_replicas,
                    new_state_checksum=new_checksum,
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
        """Simple health check endpoint."""
        info = getattr(self, "_runtime_info", {}) or {}
        listen_host = info.get("listen_host") or getattr(
            self.config, "listen_host", None
        )
        listen_port = info.get("listen_port") or getattr(
            self.config, "listen_port", None
        )
        metrics_port = info.get("metrics_port") or getattr(
            self.config, "metrics_port", None
        )
        cluster_token = info.get("cluster_token") or getattr(
            self.config, "cluster_token", None
        )
        db_file = info.get("db_file") or (
            str(self.config.db_file) if getattr(self.config, "db_file", None) else ""
        )
        version = info.get("version") or ""
        listen_address = (
            f"{listen_host}:{listen_port}" if listen_host and listen_port else ""
        )
        return global_store_pb2.HealthCheckResponse(
            status=global_store_pb2.Status.STATUS_OK,
            cluster_token=cluster_token or "",
            listen_address=listen_address,
            listen_host=listen_host or "",
            listen_port=int(listen_port or 0),
            metrics_port=int(metrics_port or 0),
            version=version,
            db_file=db_file or "",
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
        # Derive transport chunk sizes if client omitted them.
        remote_keys = list(mem_info.remote_memory_keys)
        buffer_sizes = list(mem_info.buffer_sizes)
        if remote_keys and not buffer_sizes and len(remote_keys) == 1:
            # Fallback: treat the replica as a single contiguous region.
            # This keeps validation invariants while allowing simpler clients/tests.
            buffer_sizes = [mem_info.memory_size]

        # Map proto enum to domain enum
        if mem_info.memory_type == common_pb2.MemoryType.MEMORY_TYPE_GPU:
            domain_mem_type = MemoryType.GPU
        elif mem_info.memory_type == common_pb2.MemoryType.MEMORY_TYPE_RAM:
            domain_mem_type = MemoryType.RAM
        else:
            domain_mem_type = MemoryType.DISK

        return Replica(
            artifact_id=artifact_id,
            node_id=mem_info.node_id,
            node_address=mem_info.node_address,
            node_port=mem_info.node_port,
            memory_size=mem_info.memory_size,
            memory_type=domain_mem_type,
            device_id=mem_info.device_id,
            max_concurrency=max_concurrency,
            remote_memory_keys=remote_keys,
            buffer_sizes=buffer_sizes,
            worker_id=worker_id,
            verification_json=(
                mem_info.verification_json
                if hasattr(mem_info, "verification_json") and mem_info.verification_json
                else None
            ),
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
            "artifact_indices",
            "artifacts",
            "workers",
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

        self.artifact_service = ArtifactService(self.replica_repository)
        self.transport_service = TransportService(
            self.replica_repository, self.transport_repository
        )
        self.worker_service = WorkerService(
            self.worker_repository, self.replica_repository
        )
        self.recovery_service = RecoveryService(
            self.worker_repository, self.replica_repository
        )

        logger.debug("GlobalStoreServicer state has been reset for the next test run")
