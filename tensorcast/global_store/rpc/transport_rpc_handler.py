#  Copyright (c) 2025-2026, TensorCast Team.

"""Transport RPC handler extracted from Global Store gRPC servicer."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable
from uuid import UUID

import grpc
from google.protobuf import timestamp_pb2

from tensorcast.global_store.exceptions import (
    NotFoundError,
    TimeoutError,
    ValidationError,
)
from tensorcast.global_store.models import (
    Replica,
    TransportCompletionOutcome,
    TransportSchedulingGroup,
)
from tensorcast.global_store.services.transport_service import TransportService
from tensorcast.observability.otel import set_span_attributes
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.global_store.v1 import global_store_pb2


class TransportRpcHandler:
    """Owns transport routing RPC behavior and error mapping."""

    def __init__(
        self,
        *,
        transport_service: TransportService,
        replica_to_memory_info: Callable[[Replica], common_pb2.MemoryInfo],
        logger,
    ) -> None:
        self._transport_service = transport_service
        self._replica_to_memory_info = replica_to_memory_info
        self._logger = logger

    def request_replica_transport(
        self,
        request: global_store_pb2.RequestReplicaTransportRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.RequestReplicaTransportResponse:
        """Request artifact transport with load balancing."""
        try:
            wait_timeout_ms = 0
            if request.HasField("wait_timeout_dur"):
                duration = request.wait_timeout_dur
                wait_timeout_ms = int(
                    duration.seconds * 1000 + duration.nanos / 1_000_000
                )

            set_span_attributes(
                {
                    "tc.artifact.id": request.artifact_id,
                    "tc.source.address": request.source_address,
                    "tc.source.port": int(request.source_port),
                    "tc.request.wait_timeout_ms": int(wait_timeout_ms),
                }
            )

            requested_view_id: str | None = None
            if request.HasField("requested_byte_space"):
                space = request.requested_byte_space
                if space.kind == common_pb2.BYTE_SPACE_KIND_VIEW:
                    if not space.id:
                        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                        context.set_details("requested_byte_space VIEW requires id")
                        return global_store_pb2.RequestReplicaTransportResponse(
                            status=global_store_pb2.Status.STATUS_ERROR
                        )
                    requested_view_id = space.id
                elif space.kind in (
                    common_pb2.BYTE_SPACE_KIND_CANONICAL,
                    common_pb2.BYTE_SPACE_KIND_UNSPECIFIED,
                ):
                    requested_view_id = None
                else:
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details("unsupported requested_byte_space kind")
                    return global_store_pb2.RequestReplicaTransportResponse(
                        status=global_store_pb2.Status.STATUS_ERROR
                    )

            scheduling_group: TransportSchedulingGroup | None = None
            if request.HasField("scheduling_group"):
                group = request.scheduling_group
                group_id = group.group_id.strip()
                group_kind = group.group_kind.strip()
                part_id = group.part_id.strip()
                if not group_id or not group_kind or not part_id:
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details(
                        "scheduling_group requires non-empty group_id/group_kind/part_id"
                    )
                    return global_store_pb2.RequestReplicaTransportResponse(
                        status=global_store_pb2.Status.STATUS_ERROR
                    )
                if int(group.total_parts) <= 0:
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details("scheduling_group.total_parts must be > 0")
                    return global_store_pb2.RequestReplicaTransportResponse(
                        status=global_store_pb2.Status.STATUS_ERROR
                    )
                scheduling_group = TransportSchedulingGroup(
                    group_id=group_id,
                    group_kind=group_kind,
                    total_parts=int(group.total_parts),
                    part_id=part_id,
                    priority=int(group.priority),
                    epoch=int(group.epoch),
                )

            requester_worker_id = request.requester_worker_id.strip() or None
            request_id = request.request_id.strip()
            if not request_id:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("request_id is required")
                return global_store_pb2.RequestReplicaTransportResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )

            replica, transport_id = self._transport_service.request_transport(
                artifact_id=request.artifact_id,
                view_id=requested_view_id,
                source_node_id=request.source_node_id,
                source_address=request.source_address,
                source_port=request.source_port,
                wait_timeout_ms=wait_timeout_ms,
                scheduling_group=scheduling_group,
                requester_worker_id=requester_worker_id,
                request_id=request_id,
            )

            remote_info = self._replica_to_memory_info(replica)

            set_span_attributes({"tc.transport.id": str(transport_id)})

            return global_store_pb2.RequestReplicaTransportResponse(
                status=global_store_pb2.Status.STATUS_OK,
                remote_memory_info=remote_info,
                transport_id=str(transport_id),
            )

        except NotFoundError:
            self._logger.info(
                "No replicas registered for artifact %s",
                request.artifact_id,
            )
            return global_store_pb2.RequestReplicaTransportResponse(
                status=global_store_pb2.Status.STATUS_NOT_FOUND
            )
        except TimeoutError:
            self._logger.warning(
                "Timeout waiting for artifact %s",
                request.artifact_id,
            )
            return global_store_pb2.RequestReplicaTransportResponse(
                status=global_store_pb2.Status.STATUS_TIMED_OUT
            )
        except ValidationError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.RequestReplicaTransportResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error requesting artifact transport")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.RequestReplicaTransportResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def complete_replica_transport(
        self,
        request: global_store_pb2.CompleteReplicaTransportRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.CompleteReplicaTransportResponse:
        """Complete artifact transport and release resources."""
        try:
            transport_id = UUID(request.transport_id)
            set_span_attributes({"tc.transport.id": str(transport_id)})
            outcome = self._completion_outcome_from_proto(request.outcome)
            if outcome == TransportCompletionOutcome.UNSPECIFIED:
                raise ValidationError(
                    "complete transport requires explicit outcome (SUCCESS/FAILED/EXPIRED/CANCELLED)"
                )
            outcome_detail = request.outcome_detail.strip() or None
            self._transport_service.complete_transport(
                transport_id=transport_id,
                outcome=outcome,
                outcome_detail=outcome_detail,
            )
            return global_store_pb2.CompleteReplicaTransportResponse(
                status=global_store_pb2.Status.STATUS_OK
            )

        except NotFoundError:
            self._logger.warning("Transport not found: %s", request.transport_id)
            return global_store_pb2.CompleteReplicaTransportResponse(
                status=global_store_pb2.Status.STATUS_NOT_FOUND
            )
        except ValidationError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.CompleteReplicaTransportResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error completing artifact transport")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.CompleteReplicaTransportResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def query_transport_window(
        self,
        request: global_store_pb2.QueryTransportWindowRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.QueryTransportWindowResponse:
        try:
            if not request.HasField("created_at_start"):
                raise ValidationError("created_at_start is required")
            if not request.HasField("created_at_end"):
                raise ValidationError("created_at_end is required")
            started_at = request.created_at_start.ToDatetime(tzinfo=timezone.utc)
            finished_at = request.created_at_end.ToDatetime(tzinfo=timezone.utc)
            limit = int(request.limit) if int(request.limit) > 0 else 200_000
            rows = self._transport_service.query_transport_window(
                started_at=started_at,
                finished_at=finished_at,
                limit=limit,
            )
            payload_rows: list[global_store_pb2.TransportWindowRow] = []
            for row in rows:
                created_at = timestamp_pb2.Timestamp()
                created_at.FromDatetime(self._as_utc_datetime(row.created_at))
                payload = global_store_pb2.TransportWindowRow(
                    transport_id=str(row.transport_id),
                    replica_id=str(row.replica_id),
                    artifact_id=str(row.artifact_id),
                    status=str(row.status),
                    completion_outcome=str(row.completion_outcome),
                    request_id=str(row.request_id),
                    requester_worker_id=str(row.requester_worker_id),
                    group_id=str(row.group_id),
                    group_kind=str(row.group_kind),
                    group_part_id=str(row.group_part_id),
                    group_total_parts=max(0, int(row.group_total_parts)),
                    replica_memory_size_bytes=max(
                        0, int(row.replica_memory_size_bytes)
                    ),
                )
                payload.created_at.CopyFrom(created_at)
                if row.completed_at is not None:
                    completed_at = timestamp_pb2.Timestamp()
                    completed_at.FromDatetime(self._as_utc_datetime(row.completed_at))
                    payload.completed_at.CopyFrom(completed_at)
                payload_rows.append(payload)
            return global_store_pb2.QueryTransportWindowResponse(
                status=global_store_pb2.Status.STATUS_OK,
                rows=payload_rows,
            )
        except ValidationError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.QueryTransportWindowResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error querying transport window")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.QueryTransportWindowResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    @staticmethod
    def _completion_outcome_from_proto(
        proto_outcome: global_store_pb2.TransportCompletionOutcome,
    ) -> TransportCompletionOutcome:
        if proto_outcome == global_store_pb2.TRANSPORT_COMPLETION_OUTCOME_SUCCESS:
            return TransportCompletionOutcome.SUCCESS
        if proto_outcome == global_store_pb2.TRANSPORT_COMPLETION_OUTCOME_FAILED:
            return TransportCompletionOutcome.FAILED
        if proto_outcome == global_store_pb2.TRANSPORT_COMPLETION_OUTCOME_EXPIRED:
            return TransportCompletionOutcome.EXPIRED
        if proto_outcome == global_store_pb2.TRANSPORT_COMPLETION_OUTCOME_CANCELLED:
            return TransportCompletionOutcome.CANCELLED
        return TransportCompletionOutcome.UNSPECIFIED

    @staticmethod
    def _as_utc_datetime(raw: datetime) -> datetime:
        if raw.tzinfo is None:
            return raw.replace(tzinfo=timezone.utc)
        return raw.astimezone(timezone.utc)
