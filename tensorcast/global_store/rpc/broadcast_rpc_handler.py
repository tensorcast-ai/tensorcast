#  Copyright (c) 2026, TensorCast Team.

"""Broadcast session RPC handler."""

from __future__ import annotations

from datetime import datetime, timezone

import grpc
from google.protobuf.timestamp_pb2 import Timestamp

from tensorcast.global_store.models import (
    BroadcastEdge,
    BroadcastEdgeState,
    BroadcastSession,
    BroadcastSessionState,
)
from tensorcast.global_store.services import BroadcastService
from tensorcast.proto.global_store.v1 import global_store_pb2


_SESSION_STATE_TO_PROTO = {
    BroadcastSessionState.PLANNING: global_store_pb2.BROADCAST_SESSION_STATE_PLANNING,
    BroadcastSessionState.ACTIVE: global_store_pb2.BROADCAST_SESSION_STATE_ACTIVE,
    BroadcastSessionState.COMPLETED: global_store_pb2.BROADCAST_SESSION_STATE_COMPLETED,
    BroadcastSessionState.FAILED: global_store_pb2.BROADCAST_SESSION_STATE_FAILED,
    BroadcastSessionState.CANCELLED: global_store_pb2.BROADCAST_SESSION_STATE_CANCELLED,
}

_EDGE_STATE_TO_PROTO = {
    BroadcastEdgeState.PLANNED: global_store_pb2.BROADCAST_EDGE_STATE_PLANNED,
    BroadcastEdgeState.ASSIGNED: global_store_pb2.BROADCAST_EDGE_STATE_ASSIGNED,
    BroadcastEdgeState.MATERIALIZING: global_store_pb2.BROADCAST_EDGE_STATE_MATERIALIZING,
    BroadcastEdgeState.COMPLETED: global_store_pb2.BROADCAST_EDGE_STATE_COMPLETED,
    BroadcastEdgeState.FAILED: global_store_pb2.BROADCAST_EDGE_STATE_FAILED,
    BroadcastEdgeState.CANCELLED: global_store_pb2.BROADCAST_EDGE_STATE_CANCELLED,
}


class BroadcastRpcHandler:
    """Owns broadcast planning RPC behavior and error mapping."""

    def __init__(self, *, broadcast_service: BroadcastService, logger) -> None:
        self._broadcast_service = broadcast_service
        self._logger = logger

    def create_broadcast_session(
        self,
        request: global_store_pb2.CreateBroadcastSessionRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.CreateBroadcastSessionResponse:
        try:
            requested_view_id = (
                request.requested_view_id
                if request.HasField("requested_view_id")
                else None
            )
            session = self._broadcast_service.create_session(
                session_id=request.session_id,
                artifact_id=request.artifact_id,
                requested_view_id=requested_view_id,
                epoch=int(request.epoch),
                fanout=int(request.fanout),
                target_worker_ids=[target.worker_id for target in request.targets],
                target_daemon_ids=[target.daemon_id for target in request.targets],
                root_replica_id=request.root_replica_id,
                strict_parent=bool(request.strict_parent),
                max_attempts=int(request.max_attempts),
            )
            edges = self._broadcast_service.list_edges(session.session_id)
            return global_store_pb2.CreateBroadcastSessionResponse(
                status=global_store_pb2.STATUS_OK,
                session=self._session_to_proto(session),
                edges=[self._edge_to_proto(edge) for edge in edges],
            )
        except ValueError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.CreateBroadcastSessionResponse(
                status=global_store_pb2.STATUS_ERROR
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("CreateBroadcastSession failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.CreateBroadcastSessionResponse(
                status=global_store_pb2.STATUS_ERROR
            )

    def get_broadcast_session(
        self,
        request: global_store_pb2.GetBroadcastSessionRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.GetBroadcastSessionResponse:
        try:
            session = self._broadcast_service.get_session(request.session_id)
            if session is None:
                return global_store_pb2.GetBroadcastSessionResponse(
                    status=global_store_pb2.STATUS_NOT_FOUND
                )
            return global_store_pb2.GetBroadcastSessionResponse(
                status=global_store_pb2.STATUS_OK,
                session=self._session_to_proto(session),
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("GetBroadcastSession failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.GetBroadcastSessionResponse(
                status=global_store_pb2.STATUS_ERROR
            )

    def list_broadcast_edges(
        self,
        request: global_store_pb2.ListBroadcastEdgesRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.ListBroadcastEdgesResponse:
        try:
            return global_store_pb2.ListBroadcastEdgesResponse(
                status=global_store_pb2.STATUS_OK,
                edges=[
                    self._edge_to_proto(edge)
                    for edge in self._broadcast_service.list_edges(request.session_id)
                ],
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("ListBroadcastEdges failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.ListBroadcastEdgesResponse(
                status=global_store_pb2.STATUS_ERROR
            )

    def cancel_broadcast_session(
        self,
        request: global_store_pb2.CancelBroadcastSessionRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.CancelBroadcastSessionResponse:
        try:
            cancelled = self._broadcast_service.cancel_session(request.session_id)
            return global_store_pb2.CancelBroadcastSessionResponse(
                status=(
                    global_store_pb2.STATUS_OK
                    if cancelled
                    else global_store_pb2.STATUS_NOT_FOUND
                ),
                cancelled=cancelled,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("CancelBroadcastSession failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.CancelBroadcastSessionResponse(
                status=global_store_pb2.STATUS_ERROR
            )

    def _session_to_proto(
        self,
        session: BroadcastSession,
    ) -> global_store_pb2.BroadcastSessionInfo:
        message = global_store_pb2.BroadcastSessionInfo(
            session_id=session.session_id,
            artifact_id=session.artifact_id,
            epoch=int(session.epoch),
            fanout=int(session.fanout),
            max_attempts=int(session.max_attempts),
            strict_parent=bool(session.strict_parent),
            state=_SESSION_STATE_TO_PROTO.get(
                session.state,
                global_store_pb2.BROADCAST_SESSION_STATE_UNSPECIFIED,
            ),
            root_replica_id=str(session.root_replica_id or ""),
        )
        if session.requested_view_id is not None:
            message.requested_view_id = session.requested_view_id
        self._copy_timestamp(message.created_at, session.created_at)
        self._copy_timestamp(message.updated_at, session.updated_at)
        self._copy_timestamp(message.completed_at, session.completed_at)
        return message

    def _edge_to_proto(
        self,
        edge: BroadcastEdge,
    ) -> global_store_pb2.BroadcastEdgeInfo:
        message = global_store_pb2.BroadcastEdgeInfo(
            edge_id=edge.edge_id,
            session_id=edge.session_id,
            parent_worker_id=edge.parent_worker_id,
            parent_replica_id=str(edge.parent_replica_id),
            child_worker_id=edge.child_worker_id,
            level=int(edge.level),
            attempt=int(edge.attempt),
            state=_EDGE_STATE_TO_PROTO.get(
                edge.state,
                global_store_pb2.BROADCAST_EDGE_STATE_UNSPECIFIED,
            ),
            transport_request_id=edge.transport_request_id or "",
            failure_reason=edge.failure_reason or "",
        )
        self._copy_timestamp(message.created_at, edge.created_at)
        self._copy_timestamp(message.updated_at, edge.updated_at)
        self._copy_timestamp(message.completed_at, edge.completed_at)
        return message

    @staticmethod
    def _copy_timestamp(target: Timestamp, value: datetime | None) -> None:
        if value is None:
            return
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        target.FromDatetime(value)
