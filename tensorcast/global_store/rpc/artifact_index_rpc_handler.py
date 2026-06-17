#  Copyright (c) 2025-2026, TensorCast Team.

"""Artifact index RPC handler extracted from Global Store gRPC servicer."""

from __future__ import annotations

import hashlib
from typing import Callable

import grpc

from tensorcast.common.identity import is_msa1_artifact_id
from tensorcast.global_store.repositories.artifact_index_repository import (
    ArtifactIndexRepository,
)
from tensorcast.global_store.repositories.artifact_repository import ArtifactRepository
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.global_store.v1 import global_store_pb2


class ArtifactIndexRpcHandler:
    """Owns artifact index metadata RPC behavior and error mapping."""

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

    def upsert_artifact_metadata(
        self,
        request: global_store_pb2.UpsertArtifactMetadataRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UpsertArtifactMetadataResponse:
        """Upsert artifact + canonical index rows without creating a replica."""
        try:
            if not request.HasField("descriptor"):
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("descriptor is required")
                return global_store_pb2.UpsertArtifactMetadataResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )

            descriptor = request.descriptor
            artifact_id = descriptor.artifact_id.strip()
            index_multihash = descriptor.index_multihash.strip()
            if not artifact_id:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("descriptor.artifact_id is required")
                return global_store_pb2.UpsertArtifactMetadataResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )
            if not index_multihash:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("descriptor.index_multihash is required")
                return global_store_pb2.UpsertArtifactMetadataResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )
            if not request.canonical_index_data:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("canonical_index_data is required")
                return global_store_pb2.UpsertArtifactMetadataResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )

            expected_index_key = self._multibase_sha256_to_hex(index_multihash)
            if not expected_index_key:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("descriptor.index_multihash is invalid")
                return global_store_pb2.UpsertArtifactMetadataResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )
            actual_index_key = hashlib.sha256(request.canonical_index_data).hexdigest()
            if actual_index_key != expected_index_key:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details(
                    "canonical_index_data digest does not match descriptor.index_multihash"
                )
                return global_store_pb2.UpsertArtifactMetadataResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )

            schema_version = descriptor.schema_version.strip() or "v3"
            encoding = descriptor.encoding.strip() or "json"
            data_multihash = descriptor.data_multihash.strip() or None
            id_kind = (
                "CGID"
                if descriptor.id_kind == common_pb2.ArtifactIdKind.ARTIFACT_ID_KIND_CGID
                else "MI2"
            )

            if artifact_id.startswith("mi2:"):
                parts = artifact_id.split(":", 2)
                if len(parts) == 3:
                    if parts[1] != index_multihash:
                        context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                        context.set_details(
                            "descriptor.index_multihash does not match artifact_id"
                        )
                        return global_store_pb2.UpsertArtifactMetadataResponse(
                            status=global_store_pb2.Status.STATUS_ERROR
                        )
                    if data_multihash is not None and parts[2] != data_multihash:
                        context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                        context.set_details(
                            "descriptor.data_multihash does not match artifact_id"
                        )
                        return global_store_pb2.UpsertArtifactMetadataResponse(
                            status=global_store_pb2.Status.STATUS_ERROR
                        )
                    data_multihash = parts[2]

            with self._artifact_repository.transaction() as cursor:
                _ = self._artifact_index_repository.upsert_index(
                    index_data=request.canonical_index_data,
                    encoding=encoding,
                    schema_version=schema_version,
                    cursor=cursor,
                )
                self._artifact_repository.upsert_artifact(
                    artifact_id=artifact_id,
                    index_multihash=index_multihash,
                    data_multihash=data_multihash,
                    schema_version=schema_version,
                    encoding=encoding,
                    hash_params_json=None,
                    id_kind=id_kind,
                    cursor=cursor,
                )

            return global_store_pb2.UpsertArtifactMetadataResponse(
                status=global_store_pb2.Status.STATUS_OK
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error upserting artifact metadata")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.UpsertArtifactMetadataResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

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
            if is_msa1_artifact_id(artifact_id):
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details(
                    "msa1 artifact_id is daemon-session-local and is not valid on Global Store surfaces"
                )
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
