#  Copyright (c) 2025-2026, TensorCast Team.

"""Artifact binding RPC handler extracted from Global Store gRPC servicer."""

from __future__ import annotations

from datetime import datetime
from typing import Callable

import grpc
from google.protobuf import timestamp_pb2

from tensorcast.global_store.repositories.artifact_binding_repository import (
    ArtifactBindingRepository,
)
from tensorcast.proto.global_store.v1 import global_store_pb2


class ArtifactBindingRpcHandler:
    """Owns ArtifactBinding gRPC behavior and error mapping."""

    def __init__(
        self,
        *,
        binding_repository: ArtifactBindingRepository,
        datetime_to_timestamp: Callable[
            [datetime | None], timestamp_pb2.Timestamp | None
        ],
        logger,
    ) -> None:
        self._binding_repository = binding_repository
        self._datetime_to_timestamp = datetime_to_timestamp
        self._logger = logger

    def get_artifact_binding(
        self,
        request: global_store_pb2.GetArtifactBindingRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.GetArtifactBindingResponse:
        artifact_id = request.artifact_id
        if not artifact_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("artifact_id is required")
            return global_store_pb2.GetArtifactBindingResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        try:
            row = self._binding_repository.get(artifact_id)
            if row is None:
                return global_store_pb2.GetArtifactBindingResponse(
                    status=global_store_pb2.Status.STATUS_NOT_FOUND
                )
            kind = global_store_pb2.ARTIFACT_BINDING_KIND_UNSPECIFIED
            if str(row["kind"]).lower() == "seal":
                kind = global_store_pb2.ARTIFACT_BINDING_KIND_SEAL
            binding = global_store_pb2.ArtifactBinding(
                from_artifact_id=str(row["from_artifact_id"]),
                to_artifact_id=str(row["to_artifact_id"]),
                kind=kind,
            )
            ts = self._datetime_to_timestamp(row.get("created_at"))
            if ts is not None:
                binding.created_at.CopyFrom(ts)
            return global_store_pb2.GetArtifactBindingResponse(
                status=global_store_pb2.Status.STATUS_OK,
                binding=binding,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception(
                "Failed to get artifact binding for %s",
                artifact_id,
            )
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.GetArtifactBindingResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def upsert_artifact_binding(
        self,
        request: global_store_pb2.UpsertArtifactBindingRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UpsertArtifactBindingResponse:
        if not request.HasField("binding"):
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("binding is required")
            return global_store_pb2.UpsertArtifactBindingResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        binding = request.binding
        if not binding.from_artifact_id or not binding.to_artifact_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("binding requires from_artifact_id and to_artifact_id")
            return global_store_pb2.UpsertArtifactBindingResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        kind = "seal"
        if (
            binding.kind == global_store_pb2.ARTIFACT_BINDING_KIND_UNSPECIFIED
            or binding.kind == global_store_pb2.ARTIFACT_BINDING_KIND_SEAL
        ):
            kind = "seal"
        else:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("unsupported binding kind")
            return global_store_pb2.UpsertArtifactBindingResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        try:
            row, created = self._binding_repository.upsert(
                from_artifact_id=binding.from_artifact_id,
                to_artifact_id=binding.to_artifact_id,
                kind=kind,
            )
            kind_proto = global_store_pb2.ARTIFACT_BINDING_KIND_SEAL
            resp_binding = global_store_pb2.ArtifactBinding(
                from_artifact_id=str(row["from_artifact_id"]),
                to_artifact_id=str(row["to_artifact_id"]),
                kind=kind_proto,
            )
            ts = self._datetime_to_timestamp(row.get("created_at"))
            if ts is not None:
                resp_binding.created_at.CopyFrom(ts)
            return global_store_pb2.UpsertArtifactBindingResponse(
                status=global_store_pb2.Status.STATUS_OK,
                binding=resp_binding,
                created=created,
            )
        except ValueError as exc:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details(str(exc))
            return global_store_pb2.UpsertArtifactBindingResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Failed to upsert artifact binding")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.UpsertArtifactBindingResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
