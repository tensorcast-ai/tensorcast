#  Copyright (c) 2026, TensorCast Team.

"""Assembly readiness-cut RPC handler."""

from __future__ import annotations

from datetime import datetime
from typing import Callable, cast

import grpc
from google.protobuf import timestamp_pb2

from tensorcast.global_store.exceptions import ValidationError
from tensorcast.global_store.repositories.assembly_readiness_cut_repository import (
    AssemblyReadinessCutRepository,
)
from tensorcast.proto.global_store.v1 import global_store_pb2


class AssemblyReadinessCutRpcHandler:
    """Owns durable assembly-readiness-cut RPC behavior."""

    def __init__(
        self,
        *,
        assembly_readiness_cut_repository: AssemblyReadinessCutRepository,
        datetime_to_timestamp: Callable[
            [datetime | None], timestamp_pb2.Timestamp | None
        ],
        logger,
    ) -> None:
        self._assembly_readiness_cut_repository = assembly_readiness_cut_repository
        self._datetime_to_timestamp = datetime_to_timestamp
        self._logger = logger

    def get_assembly_readiness_cut(
        self,
        request: global_store_pb2.GetAssemblyReadinessCutRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.GetAssemblyReadinessCutResponse:
        if not request.attempt_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("attempt_id is required")
            return global_store_pb2.GetAssemblyReadinessCutResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        try:
            row = self._assembly_readiness_cut_repository.get(
                attempt_id=str(request.attempt_id)
            )
            if row is None:
                return global_store_pb2.GetAssemblyReadinessCutResponse(
                    status=global_store_pb2.Status.STATUS_NOT_FOUND
                )
            return global_store_pb2.GetAssemblyReadinessCutResponse(
                status=global_store_pb2.Status.STATUS_OK,
                readiness_cut=self._to_proto(row),
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("GetAssemblyReadinessCut failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.GetAssemblyReadinessCutResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def upsert_assembly_readiness_cut(
        self,
        request: global_store_pb2.UpsertAssemblyReadinessCutRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UpsertAssemblyReadinessCutResponse:
        if not request.HasField("readiness_cut"):
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("readiness_cut is required")
            return global_store_pb2.UpsertAssemblyReadinessCutResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        cut = request.readiness_cut
        try:
            if not cut.attempt_id or not cut.readiness_cut_proto:
                raise ValidationError("attempt_id and readiness_cut_proto are required")
            row = self._assembly_readiness_cut_repository.upsert(
                attempt_id=str(cut.attempt_id),
                readiness_cut_proto=bytes(cut.readiness_cut_proto),
            )
            return global_store_pb2.UpsertAssemblyReadinessCutResponse(
                status=global_store_pb2.Status.STATUS_OK,
                readiness_cut=self._to_proto(row),
            )
        except ValidationError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.UpsertAssemblyReadinessCutResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("UpsertAssemblyReadinessCut failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.UpsertAssemblyReadinessCutResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def _to_proto(
        self,
        row: dict[str, object],
    ) -> global_store_pb2.AssemblyReadinessCut:
        cut = global_store_pb2.AssemblyReadinessCut(
            attempt_id=str(row["attempt_id"]),
            readiness_cut_proto=bytes(cast(bytes, row["readiness_cut_proto"])),
        )
        created_at = self._datetime_to_timestamp(
            cast(datetime | None, row.get("created_at"))
        )
        if created_at is not None:
            cut.created_at.CopyFrom(created_at)
        updated_at = self._datetime_to_timestamp(
            cast(datetime | None, row.get("updated_at"))
        )
        if updated_at is not None:
            cut.updated_at.CopyFrom(updated_at)
        return cut


__all__ = ["AssemblyReadinessCutRpcHandler"]
