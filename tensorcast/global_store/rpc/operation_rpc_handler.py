#  Copyright (c) 2025-2026, TensorCast Team.

"""Operation RPC handler extracted from Global Store gRPC servicer."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, cast

import grpc
from google.protobuf import any_pb2, timestamp_pb2

from tensorcast.global_store.exceptions import DatabaseError, ValidationError
from tensorcast.global_store.repositories.operation_repository import (
    OperationRepository,
)
from tensorcast.proto.operation.v1 import operation_pb2


class OperationRpcHandler:
    """Owns OperationLease/Get/Update gRPC behavior and error mapping."""

    _OP_STATE_TO_DB: dict[int, str] = {
        operation_pb2.OPERATION_STATE_PENDING: "pending",
        operation_pb2.OPERATION_STATE_RUNNING: "running",
        operation_pb2.OPERATION_STATE_SUCCESS: "success",
        operation_pb2.OPERATION_STATE_FAILED: "failed",
        operation_pb2.OPERATION_STATE_CANCELLED: "cancelled",
        operation_pb2.OPERATION_STATE_DEGRADED: "degraded",
    }

    def __init__(
        self,
        *,
        operation_repository: OperationRepository,
        default_ttl_ms: int,
        max_ttl_ms: int,
        min_status_update_interval_ms: int,
        datetime_to_timestamp: Callable[
            [datetime | None], timestamp_pb2.Timestamp | None
        ],
        coerce_db_datetime: Callable[[object], datetime | None],
        logger,
    ) -> None:
        self._operation_repository = operation_repository
        self._default_ttl_ms = int(default_ttl_ms)
        self._max_ttl_ms = int(max_ttl_ms)
        self._min_status_update_interval_ms = int(min_status_update_interval_ms)
        self._datetime_to_timestamp = datetime_to_timestamp
        self._coerce_db_datetime = coerce_db_datetime
        self._logger = logger

    def _op_state_to_db(self, state: int) -> str:
        if state not in self._OP_STATE_TO_DB:
            raise ValidationError("operation state must be set")
        return self._OP_STATE_TO_DB[state]

    def _resolve_ttl_ms(self, requested_ttl_ms: int) -> int:
        ttl_ms = int(requested_ttl_ms)
        if ttl_ms <= 0:
            ttl_ms = self._default_ttl_ms
        return min(ttl_ms, self._max_ttl_ms)

    @staticmethod
    def _merge_ref(
        target: operation_pb2.OperationRef, source: operation_pb2.OperationRef
    ) -> None:
        if not target.operation_id and source.operation_id:
            target.operation_id = source.operation_id
        if not target.kind and source.kind:
            target.kind = source.kind
        if not target.target_artifact_id and source.target_artifact_id:
            target.target_artifact_id = source.target_artifact_id
        if not target.authority_scope_kind and source.authority_scope_kind:
            target.authority_scope_kind = source.authority_scope_kind
        if not target.authority_scope_id and source.authority_scope_id:
            target.authority_scope_id = source.authority_scope_id
        if not target.attachment_kind and source.attachment_kind:
            target.attachment_kind = source.attachment_kind
        if not target.recovery_class and source.recovery_class:
            target.recovery_class = source.recovery_class
        if not target.fencing_digest and source.fencing_digest:
            target.fencing_digest = source.fencing_digest

    def _apply_continuation_metadata_from_snapshot(
        self,
        *,
        ref: operation_pb2.OperationRef,
        snapshot: any_pb2.Any,
    ) -> None:
        metadata = operation_pb2.OperationContinuationMetadata()
        if snapshot.Unpack(metadata):
            self._merge_ref(ref, metadata.ref)

    def acquire_operation_lease(
        self,
        request: operation_pb2.AcquireOperationLeaseRequest,
        context: grpc.ServicerContext,
    ) -> operation_pb2.AcquireOperationLeaseResponse:
        if (
            not request.operation_id
            or not request.kind
            or not request.target_artifact_id
        ):
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(
                "operation_id, kind, and target_artifact_id are required"
            )
            return operation_pb2.AcquireOperationLeaseResponse()
        if not request.owner_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("owner_id is required")
            return operation_pb2.AcquireOperationLeaseResponse()
        try:
            ttl_ms = self._resolve_ttl_ms(int(request.ttl_ms))
            now = datetime.now(timezone.utc)
            ts = timestamp_pb2.Timestamp()
            ts.FromDatetime(now)
            initial_status = operation_pb2.OperationStatus(
                state=operation_pb2.OPERATION_STATE_PENDING,
                message="",
                progress=0.0,
                as_of=ts,
            )
            initial_proto = initial_status.SerializeToString(deterministic=True)
            initial_state = self._op_state_to_db(initial_status.state)

            with self._operation_repository.transaction() as cursor:
                acquired, lease = self._operation_repository.acquire_lease(
                    operation_id=str(request.operation_id),
                    kind=str(request.kind),
                    target_artifact_id=str(request.target_artifact_id),
                    owner_id=str(request.owner_id),
                    ttl_ms=ttl_ms,
                    initial_state=initial_state,
                    initial_status_proto=initial_proto,
                    cursor=cursor,
                )

            lease_msg = operation_pb2.OperationLease(
                operation_id=str(lease["operation_id"]),
                lease_token=str(lease["lease_token"]),
                owner_id=str(lease["owner_id"]),
                lease_generation=int(lease["lease_generation"]),
            )
            expires_ts = self._datetime_to_timestamp(
                self._coerce_db_datetime(lease.get("expires_at"))
            )
            if expires_ts is not None:
                lease_msg.expires_at.CopyFrom(expires_ts)

            if not acquired:
                context.set_code(grpc.StatusCode.ALREADY_EXISTS)
                context.set_details("operation lease held by another owner")
            return operation_pb2.AcquireOperationLeaseResponse(
                acquired=bool(acquired),
                lease=lease_msg,
            )
        except ValidationError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return operation_pb2.AcquireOperationLeaseResponse()
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("AcquireOperationLease failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return operation_pb2.AcquireOperationLeaseResponse()

    def keepalive_operation_lease(
        self,
        request: operation_pb2.KeepaliveOperationLeaseRequest,
        context: grpc.ServicerContext,
    ) -> operation_pb2.KeepaliveOperationLeaseResponse:
        if not request.lease_token:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("lease_token is required")
            return operation_pb2.KeepaliveOperationLeaseResponse()
        try:
            ttl_ms = self._resolve_ttl_ms(int(request.ttl_ms))
            with self._operation_repository.transaction() as cursor:
                lease = self._operation_repository.keepalive_lease(
                    lease_token=str(request.lease_token),
                    ttl_ms=ttl_ms,
                    cursor=cursor,
                )
            lease_msg = operation_pb2.OperationLease(
                operation_id=str(lease["operation_id"]),
                lease_token=str(lease["lease_token"]),
                owner_id=str(lease["owner_id"]),
                lease_generation=int(lease["lease_generation"]),
            )
            expires_ts = self._datetime_to_timestamp(
                self._coerce_db_datetime(lease.get("expires_at"))
            )
            if expires_ts is not None:
                lease_msg.expires_at.CopyFrom(expires_ts)
            return operation_pb2.KeepaliveOperationLeaseResponse(lease=lease_msg)
        except (ValueError, DatabaseError) as exc:
            message = str(exc)
            if "lease_token not found" in message:
                context.set_code(grpc.StatusCode.NOT_FOUND)
            else:
                context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(message)
            return operation_pb2.KeepaliveOperationLeaseResponse()
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("KeepaliveOperationLease failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return operation_pb2.KeepaliveOperationLeaseResponse()

    def release_operation_lease(
        self,
        request: operation_pb2.ReleaseOperationLeaseRequest,
        context: grpc.ServicerContext,
    ) -> operation_pb2.ReleaseOperationLeaseResponse:
        if not request.lease_token:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("lease_token is required")
            return operation_pb2.ReleaseOperationLeaseResponse(released=False)
        try:
            with self._operation_repository.transaction() as cursor:
                released = self._operation_repository.release_lease(
                    lease_token=str(request.lease_token),
                    cursor=cursor,
                )
            return operation_pb2.ReleaseOperationLeaseResponse(released=bool(released))
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("ReleaseOperationLease failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return operation_pb2.ReleaseOperationLeaseResponse(released=False)

    def get_operation(
        self,
        request: operation_pb2.GetOperationRequest,
        context: grpc.ServicerContext,
    ) -> operation_pb2.GetOperationResponse:
        if not request.operation_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("operation_id is required")
            return operation_pb2.GetOperationResponse()
        try:
            row = self._operation_repository.get(operation_id=str(request.operation_id))
            if row is None:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("operation not found")
                return operation_pb2.GetOperationResponse()
            status = operation_pb2.OperationStatus()
            status.ParseFromString(row["status_proto"])
            snapshot = any_pb2.Any()
            if row.get("snapshot_proto"):
                snapshot.ParseFromString(cast(bytes, row["snapshot_proto"]))
            resp = operation_pb2.GetOperationResponse(
                ref=operation_pb2.OperationRef(
                    operation_id=str(row["operation_id"]),
                    kind=str(row["kind"]),
                    target_artifact_id=str(row["target_artifact_id"]),
                ),
                status=status,
                lease_generation=int(row["lease_generation"]),
                lease_owner=str(row["lease_owner"] or ""),
            )
            if row.get("snapshot_proto"):
                self._apply_continuation_metadata_from_snapshot(
                    ref=resp.ref,
                    snapshot=snapshot,
                )
            expires_ts = self._datetime_to_timestamp(
                self._coerce_db_datetime(row.get("lease_expires_at"))
            )
            if expires_ts is not None:
                resp.lease_expires_at.CopyFrom(expires_ts)
            if row.get("snapshot_proto"):
                resp.snapshot.CopyFrom(snapshot)
            return resp
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("GetOperation failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return operation_pb2.GetOperationResponse()

    def update_operation(
        self,
        request: operation_pb2.UpdateOperationRequest,
        context: grpc.ServicerContext,
    ) -> operation_pb2.UpdateOperationResponse:
        if not request.operation_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("operation_id is required")
            return operation_pb2.UpdateOperationResponse()
        try:
            state = self._op_state_to_db(int(request.status.state))
            status_proto = request.status.SerializeToString(deterministic=True)
            snapshot_proto: bytes | None = None
            if request.HasField("snapshot") and (
                request.snapshot.type_url or request.snapshot.value
            ):
                snapshot_proto = request.snapshot.SerializeToString(deterministic=True)

            with self._operation_repository.transaction() as cursor:
                self._operation_repository.update_operation(
                    operation_id=str(request.operation_id),
                    lease_generation=int(request.lease_generation),
                    state=state,
                    status_proto=status_proto,
                    snapshot_proto=snapshot_proto,
                    min_status_update_interval_ms=self._min_status_update_interval_ms,
                    cursor=cursor,
                )
            return operation_pb2.UpdateOperationResponse()
        except ValidationError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return operation_pb2.UpdateOperationResponse()
        except (ValueError, DatabaseError) as exc:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details(str(exc))
            return operation_pb2.UpdateOperationResponse()
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("UpdateOperation failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return operation_pb2.UpdateOperationResponse()
