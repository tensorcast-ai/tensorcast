#  Copyright (c) 2025-2026, TensorCast Team.

"""Disk location RPC handler extracted from Global Store gRPC servicer."""

from __future__ import annotations

from datetime import datetime
from typing import Callable

import grpc
from google.protobuf import timestamp_pb2

from tensorcast.global_store.repositories.artifact_disk_location_repository import (
    ArtifactDiskLocationRepository,
)
from tensorcast.proto.global_store.v1 import global_store_pb2


class DiskLocationRpcHandler:
    """Owns artifact disk location gRPC behavior and error mapping."""

    def __init__(
        self,
        *,
        disk_location_repository: ArtifactDiskLocationRepository,
        cluster_id: str,
        is_safe_relative_path: Callable[[str], bool],
        disk_location_kind_from_proto: dict[int, str],
        disk_location_kind_to_proto: dict[str, int],
        datetime_to_timestamp: Callable[
            [datetime | None], timestamp_pb2.Timestamp | None
        ],
        coerce_db_datetime: Callable[[object], datetime | None],
        logger,
    ) -> None:
        self._disk_location_repository = disk_location_repository
        self._cluster_id = cluster_id
        self._is_safe_relative_path = is_safe_relative_path
        self._disk_location_kind_from_proto = disk_location_kind_from_proto
        self._disk_location_kind_to_proto = disk_location_kind_to_proto
        self._datetime_to_timestamp = datetime_to_timestamp
        self._coerce_db_datetime = coerce_db_datetime
        self._logger = logger

    def upsert_artifact_disk_location(
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
            if cluster_id != self._cluster_id:
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
            kind = self._disk_location_kind_from_proto.get(request.kind, "MANAGED")
            self._disk_location_repository.upsert(
                artifact_id=artifact_id,
                cluster_id=cluster_id,
                relative_path=relative_path,
                kind=kind,
                is_deleted=bool(request.is_deleted),
            )
            return global_store_pb2.UpsertArtifactDiskLocationResponse(
                status=global_store_pb2.Status.STATUS_OK
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error in UpsertArtifactDiskLocation")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.UpsertArtifactDiskLocationResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def list_artifact_disk_locations(
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
            rows = self._disk_location_repository.list_by_artifact(
                artifact_id,
                include_deleted=include_deleted,
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
                kind = self._disk_location_kind_to_proto.get(
                    (row.get("kind") or "MANAGED").upper(),
                    global_store_pb2.DISK_LOCATION_KIND_MANAGED,
                )
                message = global_store_pb2.ArtifactDiskLocation(
                    artifact_id=row.get("artifact_id", ""),
                    cluster_id=row.get("cluster_id", ""),
                    relative_path=row.get("relative_path", ""),
                    kind=kind,
                    created_at=created,
                    updated_at=updated,
                    is_deleted=bool(row.get("is_deleted", False)),
                )
                if deleted_at is not None:
                    message.deleted_at.CopyFrom(deleted_at)
                locations.append(message)
            return global_store_pb2.ListArtifactDiskLocationsResponse(
                status=global_store_pb2.Status.STATUS_OK,
                locations=locations,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error in ListArtifactDiskLocations")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.ListArtifactDiskLocationsResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
