#  Copyright (c) 2026, TensorCast Team.

"""Assembly attempt durable-row RPC handler."""

from __future__ import annotations

from datetime import datetime
from typing import Callable, cast

import grpc
from google.protobuf import timestamp_pb2

from tensorcast.global_store.exceptions import ValidationError
from tensorcast.global_store.repositories.assembly_attempt_repository import (
    AssemblyAttemptRepository,
)
from tensorcast.proto.global_store.v1 import global_store_pb2


class AssemblyAttemptRpcHandler:
    """Owns durable assembly-attempt RPC behavior."""

    def __init__(
        self,
        *,
        assembly_attempt_repository: AssemblyAttemptRepository,
        datetime_to_timestamp: Callable[
            [datetime | None], timestamp_pb2.Timestamp | None
        ],
        logger,
    ) -> None:
        self._assembly_attempt_repository = assembly_attempt_repository
        self._datetime_to_timestamp = datetime_to_timestamp
        self._logger = logger

    def get_assembly_attempt(
        self,
        request: global_store_pb2.GetAssemblyAttemptRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.GetAssemblyAttemptResponse:
        try:
            key_name, key_value = request.WhichOneof("key"), None
            if key_name == "attempt_id":
                key_value = str(request.attempt_id)
                row = self._assembly_attempt_repository.get(attempt_id=key_value)
            elif key_name == "workspace_assembly_id":
                key_value = str(request.workspace_assembly_id)
                row = self._assembly_attempt_repository.get(
                    workspace_assembly_id=key_value
                )
            else:
                raise ValidationError(
                    "exactly one of attempt_id or workspace_assembly_id is required"
                )
            if row is None:
                return global_store_pb2.GetAssemblyAttemptResponse(
                    status=global_store_pb2.Status.STATUS_NOT_FOUND
                )
            return global_store_pb2.GetAssemblyAttemptResponse(
                status=global_store_pb2.Status.STATUS_OK,
                attempt=self._to_proto(row),
            )
        except ValidationError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.GetAssemblyAttemptResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("GetAssemblyAttempt failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.GetAssemblyAttemptResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def upsert_assembly_attempt(
        self,
        request: global_store_pb2.UpsertAssemblyAttemptRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UpsertAssemblyAttemptResponse:
        if not request.HasField("attempt"):
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("attempt is required")
            return global_store_pb2.UpsertAssemblyAttemptResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        try:
            attempt = request.attempt
            self._validate_attempt(attempt)
            row = self._assembly_attempt_repository.upsert(
                attempt_id=str(attempt.attempt_id),
                workspace_assembly_id=str(attempt.workspace_assembly_id),
                layout_id=str(attempt.layout_id),
                attempt_intent_digest=str(attempt.attempt_intent_digest),
                coordinator_operation_id=str(attempt.coordinator_operation_id),
                attempt_record_proto=bytes(attempt.attempt_record_proto),
            )
            return global_store_pb2.UpsertAssemblyAttemptResponse(
                status=global_store_pb2.Status.STATUS_OK,
                attempt=self._to_proto(row),
            )
        except ValidationError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.UpsertAssemblyAttemptResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("UpsertAssemblyAttempt failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.UpsertAssemblyAttemptResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    @staticmethod
    def _validate_attempt(attempt: global_store_pb2.AssemblyAttempt) -> None:
        required = {
            "attempt_id": attempt.attempt_id,
            "workspace_assembly_id": attempt.workspace_assembly_id,
            "layout_id": attempt.layout_id,
            "attempt_intent_digest": attempt.attempt_intent_digest,
            "coordinator_operation_id": attempt.coordinator_operation_id,
        }
        missing = [name for name, value in required.items() if not str(value).strip()]
        if missing:
            raise ValidationError(
                f"missing required attempt fields: {', '.join(missing)}"
            )
        if not attempt.attempt_record_proto:
            raise ValidationError("attempt_record_proto is required")

    def _to_proto(
        self,
        row: dict[str, object],
    ) -> global_store_pb2.AssemblyAttempt:
        attempt = global_store_pb2.AssemblyAttempt(
            attempt_id=str(row["attempt_id"]),
            workspace_assembly_id=str(row["workspace_assembly_id"]),
            layout_id=str(row["layout_id"]),
            attempt_intent_digest=str(row["attempt_intent_digest"]),
            coordinator_operation_id=str(row["coordinator_operation_id"]),
            attempt_record_proto=bytes(cast(bytes, row["attempt_record_proto"])),
        )
        created_at = self._datetime_to_timestamp(
            cast(datetime | None, row.get("created_at"))
        )
        if created_at is not None:
            attempt.created_at.CopyFrom(created_at)
        updated_at = self._datetime_to_timestamp(
            cast(datetime | None, row.get("updated_at"))
        )
        if updated_at is not None:
            attempt.updated_at.CopyFrom(updated_at)
        return attempt


__all__ = ["AssemblyAttemptRpcHandler"]
