#  Copyright (c) 2025-2026, TensorCast Team.

"""Replica registration RPC handler extracted from Global Store gRPC servicer."""

from __future__ import annotations

from contextlib import suppress
from typing import Callable

import grpc

from tensorcast.common.identity import ArtifactIdKind, infer_artifact_id_kind
from tensorcast.global_store import metrics as gs_metrics
from tensorcast.global_store.exceptions import ValidationError
from tensorcast.global_store.models import Replica
from tensorcast.global_store.repositories.artifact_index_repository import (
    ArtifactIndexRepository,
)
from tensorcast.global_store.repositories.artifact_repository import ArtifactRepository
from tensorcast.global_store.repositories.base import is_transient_tx_conflict
from tensorcast.global_store.repositories.idempotency_repository import (
    IdempotencyRepository,
)
from tensorcast.global_store.rpc.idempotency import (
    begin_idempotent_operation,
    decode_stored_grpc_status,
    encode_stored_grpc_status,
)
from tensorcast.global_store.services.artifact_service import ArtifactService
from tensorcast.global_store.services.worker_control_reducer import WorkerControlReducer
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
        idempotency_repository: IdempotencyRepository,
        memory_info_to_replica_artifact_id: Callable[
            [common_pb2.MemoryInfo, str, int, str], Replica
        ],
        index_bytes_to_multibase_sha256: Callable[[bytes], str | None],
        hex_sha256_to_multibase: Callable[[str], str | None],
        control_reducer: WorkerControlReducer | None,
        logger,
    ) -> None:
        self._artifact_service = artifact_service
        self._artifact_repository = artifact_repository
        self._artifact_index_repository = artifact_index_repository
        self._idempotency_repository = idempotency_repository
        self._memory_info_to_replica_artifact_id = memory_info_to_replica_artifact_id
        self._index_bytes_to_multibase_sha256 = index_bytes_to_multibase_sha256
        self._hex_sha256_to_multibase = hex_sha256_to_multibase
        self._control_reducer = control_reducer
        self._logger = logger

    @staticmethod
    def _client_request_id(
        request: global_store_pb2.RegisterReplicaRequest,
    ) -> str:
        if not request.HasField("client_request_id"):
            return ""
        return request.client_request_id.strip()

    @staticmethod
    def _resolved_artifact_id(
        request: global_store_pb2.RegisterReplicaRequest,
    ) -> str:
        if request.HasField("descriptor") and request.descriptor.artifact_id:
            return request.descriptor.artifact_id
        return request.artifact_id

    @staticmethod
    def _apply_grpc_status(
        *,
        context: grpc.ServicerContext,
        code: grpc.StatusCode | None,
        details: str,
    ) -> None:
        if code is None or code == grpc.StatusCode.OK:
            return
        context.set_code(code)
        context.set_details(details)

    @staticmethod
    def _replay_response(
        *,
        context: grpc.ServicerContext,
        record_response_status: str,
        record_response_proto: bytes,
    ) -> global_store_pb2.RegisterReplicaResponse:
        response = global_store_pb2.RegisterReplicaResponse()
        response.ParseFromString(record_response_proto)
        code, details = decode_stored_grpc_status(record_response_status)
        if code != grpc.StatusCode.OK:
            context.set_code(code)
            context.set_details(details)
        return response

    def register_replica(
        self,
        request: global_store_pb2.RegisterReplicaRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.RegisterReplicaResponse:
        """Register or update an artifact replica."""
        response = global_store_pb2.RegisterReplicaResponse(
            status=global_store_pb2.Status.STATUS_ERROR
        )
        grpc_code: grpc.StatusCode | None = None
        grpc_details = ""
        should_finalize_idempotency = False
        client_request_id = self._client_request_id(request)
        operation_kind = "register_replica"
        resolved_artifact_id = self._resolved_artifact_id(request)
        try:
            schema_version_value = "v3"
            if request.HasField("schema_version"):
                candidate_schema_version = request.schema_version.strip()
                if candidate_schema_version and candidate_schema_version != "v3":
                    grpc_code = grpc.StatusCode.INVALID_ARGUMENT
                    grpc_details = "schema_version must be 'v3'"
                    self._apply_grpc_status(
                        context=context,
                        code=grpc_code,
                        details=grpc_details,
                    )
                    return response
                if candidate_schema_version:
                    schema_version_value = candidate_schema_version

            if client_request_id:
                idempotency_result = begin_idempotent_operation(
                    idempotency_repository=self._idempotency_repository,
                    operation_kind=operation_kind,
                    client_request_id=client_request_id,
                    request_payload=request.SerializeToString(deterministic=True),
                )
                if idempotency_result.payload_mismatch:
                    grpc_code = grpc.StatusCode.FAILED_PRECONDITION
                    grpc_details = (
                        "client_request_id already used with a different payload"
                    )
                    self._apply_grpc_status(
                        context=context,
                        code=grpc_code,
                        details=grpc_details,
                    )
                    return response
                if idempotency_result.timed_out_waiting_for_replay:
                    grpc_code = grpc.StatusCode.ABORTED
                    grpc_details = (
                        "duplicate request is still in progress for client_request_id"
                    )
                    self._apply_grpc_status(
                        context=context,
                        code=grpc_code,
                        details=grpc_details,
                    )
                    return response
                if not idempotency_result.should_execute:
                    replay_record = idempotency_result.replay_record
                    if replay_record is None:
                        grpc_code = grpc.StatusCode.INTERNAL
                        grpc_details = "idempotency replay record is missing"
                        self._apply_grpc_status(
                            context=context,
                            code=grpc_code,
                            details=grpc_details,
                        )
                        return response
                    return self._replay_response(
                        context=context,
                        record_response_status=replay_record.response_status,
                        record_response_proto=replay_record.response_proto,
                    )
                should_finalize_idempotency = True

            def _register_impl() -> Replica:
                replica = self._memory_info_to_replica_artifact_id(
                    request.mem_info,
                    resolved_artifact_id,
                    request.max_concurrency,
                    request.worker_id,
                )
                preserve_transport = not request.mem_info.HasField("transport")
                artifact_id = resolved_artifact_id
                descriptor = (
                    request.descriptor if request.HasField("descriptor") else None
                )

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
                    if kind is ArtifactIdKind.MI2 and (
                        index_mh is None or data_mh is None
                    ):
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
                    with (
                        self._artifact_service.replica_repository.transaction() as cursor
                    ):
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
                        return self._artifact_service.register_replica(
                            replica,
                            preserve_transport=preserve_transport,
                            cursor=cursor,
                        )

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
                    return self._artifact_service.register_replica(
                        replica,
                        preserve_transport=preserve_transport,
                        cursor=cursor,
                    )

            reducer_key = ""
            if resolved_artifact_id:
                reducer_key = f"artifact:{resolved_artifact_id}"
            elif request.worker_id:
                reducer_key = f"worker:{request.worker_id}"

            if self._control_reducer is not None and reducer_key:
                registered = self._control_reducer.submit(
                    worker_key=reducer_key,
                    kind="register_replica",
                    operation=_register_impl,
                )
            else:
                registered = _register_impl()

            with suppress(Exception):
                span_attrs: dict[str, bool | int | float | str] = {
                    "tc.artifact.id": registered.artifact_id,
                    "tc.replica.id": str(registered.replica_id),
                    "tc.memory.type": str(registered.memory_type.value),
                    "tc.memory.size": int(registered.memory_size),
                    "tc.device.id": int(registered.device_id),
                }
                worker_id = registered.worker_id
                if worker_id:
                    span_attrs["tc.worker.id"] = worker_id
                set_span_attributes(span_attrs)

            response = global_store_pb2.RegisterReplicaResponse(
                status=global_store_pb2.Status.STATUS_OK,
                artifact_id=registered.artifact_id,
                replica_id=str(registered.replica_id),
            )
            return response

        except ValidationError as exc:
            self._logger.error("Validation error: %s", exc)
            grpc_code = grpc.StatusCode.INVALID_ARGUMENT
            grpc_details = str(exc)
            self._apply_grpc_status(
                context=context,
                code=grpc_code,
                details=grpc_details,
            )
            return response
        except Exception as exc:  # noqa: BLE001
            if is_transient_tx_conflict(exc):
                gs_metrics.inc_control_plane_conflict(
                    scope="register_replica_tx_conflict"
                )
                grpc_code = grpc.StatusCode.ABORTED
                grpc_details = f"RegisterReplica transaction conflict: {exc}"
                self._logger.error(
                    "Control-plane conflict operation_kind=%s worker_id=%s artifact_id=%s client_request_id=%s error=%s",
                    operation_kind,
                    request.worker_id,
                    resolved_artifact_id,
                    client_request_id,
                    exc,
                )
                self._apply_grpc_status(
                    context=context,
                    code=grpc_code,
                    details=grpc_details,
                )
                return response
            self._logger.exception("Error registering artifact replica")
            grpc_code = grpc.StatusCode.INTERNAL
            grpc_details = str(exc)
            self._apply_grpc_status(
                context=context,
                code=grpc_code,
                details=grpc_details,
            )
            return response
        finally:
            if should_finalize_idempotency and client_request_id:
                try:
                    self._idempotency_repository.finalize_operation(
                        client_request_id=client_request_id,
                        response_status=encode_stored_grpc_status(
                            grpc_code,
                            grpc_details,
                        ),
                        response_proto=response.SerializeToString(),
                    )
                except Exception:  # noqa: BLE001
                    self._logger.exception(
                        "Failed to finalize idempotency record for RegisterReplica client_request_id=%s",
                        client_request_id,
                    )
