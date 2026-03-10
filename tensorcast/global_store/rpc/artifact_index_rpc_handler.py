#  Copyright (c) 2025-2026, TensorCast Team.

"""Artifact index RPC handler extracted from Global Store gRPC servicer."""

from __future__ import annotations

from typing import Callable

import grpc

from tensorcast.global_store.repositories.artifact_index_repository import (
    ArtifactIndexRepository,
)
from tensorcast.global_store.repositories.artifact_repository import ArtifactRepository
from tensorcast.proto.global_store.v1 import global_store_pb2


class ArtifactIndexRpcHandler:
    """Owns artifact index lookup RPC behavior and error mapping."""

    def __init__(
        self,
        *,
        artifact_index_repository: ArtifactIndexRepository,
        artifact_repository: ArtifactRepository,
        multibase_sha256_to_hex: Callable[[str], str | None],
        logger,
    ) -> None:
        self._artifact_index_repository = artifact_index_repository
        self._artifact_repository = artifact_repository
        self._multibase_sha256_to_hex = multibase_sha256_to_hex
        self._logger = logger

    def get_artifact_index(
        self,
        request: global_store_pb2.GetArtifactIndexRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.GetArtifactIndexResponse:
        """Fetch canonical tensor index bytes by key."""
        try:
            if not request.tensor_index_key:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("tensor_index_key is required")
                return global_store_pb2.GetArtifactIndexResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )

            data = self._artifact_index_repository.get(request.tensor_index_key)
            if data is None:
                return global_store_pb2.GetArtifactIndexResponse(
                    status=global_store_pb2.Status.STATUS_NOT_FOUND
                )

            return global_store_pb2.GetArtifactIndexResponse(
                status=global_store_pb2.Status.STATUS_OK,
                tensor_index_data=data,
                encoding="json",
                schema_version="v3",
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error getting artifact index")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.GetArtifactIndexResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def get_artifact_index_by_id(
        self,
        request: global_store_pb2.GetArtifactIndexByIdRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.GetArtifactIndexByIdResponse:
        """Fetch canonical tensor index bytes by artifact id."""
        try:
            artifact_id = request.artifact_id
            if not artifact_id:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("artifact_id is required")
                return global_store_pb2.GetArtifactIndexByIdResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )

            row = self._artifact_repository.get(artifact_id)
            if not row:
                return global_store_pb2.GetArtifactIndexByIdResponse(
                    status=global_store_pb2.Status.STATUS_NOT_FOUND
                )
            index_multihash = row.get("index_multihash")
            index_key = self._multibase_sha256_to_hex(str(index_multihash))
            if not index_key:
                self._logger.warning(
                    "Invalid index_multihash stored for %s; cannot derive SHA key",
                    artifact_id,
                )
                return global_store_pb2.GetArtifactIndexByIdResponse(
                    status=global_store_pb2.Status.STATUS_NOT_FOUND
                )
            data = self._artifact_index_repository.get(index_key)
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
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error getting artifact index by id")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.GetArtifactIndexByIdResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
