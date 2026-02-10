#  Copyright (c) 2025-2026, TensorCast Team.

"""Replica registration RPC handler extracted from Global Store gRPC servicer."""

from __future__ import annotations

from contextlib import suppress
from typing import Callable

import grpc

from tensorcast.common.identity import ArtifactIdKind, infer_artifact_id_kind
from tensorcast.global_store.exceptions import ValidationError
from tensorcast.global_store.models import Replica
from tensorcast.global_store.repositories.artifact_index_repository import (
    ArtifactIndexRepository,
)
from tensorcast.global_store.repositories.artifact_repository import ArtifactRepository
from tensorcast.global_store.services.artifact_service import ArtifactService
from tensorcast.observability.otel import set_span_attributes
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.global_store.v1 import global_store_pb2


class ReplicaRegistrationRpcHandler:
    """Owns replica registration gRPC behavior and error mapping."""

    def __init__(
        self,
        *,
        artifact_service: ArtifactService,
        artifact_repository: ArtifactRepository,
        artifact_index_repository: ArtifactIndexRepository,
        memory_info_to_replica_artifact_id: Callable[
            [common_pb2.MemoryInfo, str, int, str], Replica
        ],
        index_bytes_to_multibase_sha256: Callable[[bytes], str | None],
        hex_sha256_to_multibase: Callable[[str], str | None],
        logger,
    ) -> None:
        self._artifact_service = artifact_service
        self._artifact_repository = artifact_repository
        self._artifact_index_repository = artifact_index_repository
        self._memory_info_to_replica_artifact_id = memory_info_to_replica_artifact_id
        self._index_bytes_to_multibase_sha256 = index_bytes_to_multibase_sha256
        self._hex_sha256_to_multibase = hex_sha256_to_multibase
        self._logger = logger

    def register_replica(
        self,
        request: global_store_pb2.RegisterReplicaRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.RegisterReplicaResponse:
        """Register or update an artifact replica."""
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

            replica = self._memory_info_to_replica_artifact_id(
                request.mem_info,
                request.artifact_id,
                request.max_concurrency,
                request.worker_id,
            )
            preserve_transport = not request.mem_info.HasField("transport")
            artifact_id = request.artifact_id
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
                artifact_index_encoding = (
                    request.encoding if request.HasField("encoding") else "json"
                )
                with self._artifact_service.replica_repository.transaction() as cursor:
                    self._artifact_repository.upsert_artifact(
                        artifact_id=artifact_id,
                        index_multihash=index_mh,
                        data_multihash=data_mh,
                        schema_version=schema_version,
                        encoding=encoding,
                        hash_params_json=None,
                        id_kind=id_kind,
                        cursor=cursor,
                    )
                    if (
                        request.HasField("tensor_index_data")
                        and request.tensor_index_data
                    ):
                        _ = self._artifact_index_repository.upsert_index(
                            index_data=request.tensor_index_data,
                            encoding=artifact_index_encoding,
                            schema_version=schema_version_value,
                            cursor=cursor,
                        )
                    registered = self._artifact_service.register_replica(
                        replica,
                        preserve_transport=preserve_transport,
                        cursor=cursor,
                    )
            else:
                with self._artifact_service.replica_repository.transaction() as cursor:
                    if (
                        request.HasField("tensor_index_data")
                        and request.tensor_index_data
                    ):
                        _ = self._artifact_index_repository.upsert_index(
                            index_data=request.tensor_index_data,
                            encoding=(
                                request.encoding
                                if request.HasField("encoding")
                                else "json"
                            ),
                            schema_version=schema_version_value,
                            cursor=cursor,
                        )
                    registered = self._artifact_service.register_replica(
                        replica,
                        preserve_transport=preserve_transport,
                        cursor=cursor,
                    )

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

            return global_store_pb2.RegisterReplicaResponse(
                status=global_store_pb2.Status.STATUS_OK,
                artifact_id=registered.artifact_id,
                replica_id=str(registered.replica_id),
            )

        except ValidationError as exc:
            self._logger.error("Validation error: %s", exc)
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.RegisterReplicaResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error registering artifact replica")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.RegisterReplicaResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
