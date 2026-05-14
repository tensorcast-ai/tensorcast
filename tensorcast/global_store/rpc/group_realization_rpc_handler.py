#  Copyright (c) 2026, TensorCast Team.

"""Group realization RPC handler."""

from __future__ import annotations

import grpc

from tensorcast.global_store.exceptions import (
    ConflictError,
    NotFoundError,
    ValidationError,
)
from tensorcast.global_store.repositories.group_version_set_repository import (
    parse_requested_byte_space_json,
)
from tensorcast.global_store.repositories.operation_repository import (
    OperationRepository,
)
from tensorcast.global_store.services.group_realization_service import (
    GroupRealizationFeatureDisabledError,
    GroupRealizationPreconditionError,
    GroupRealizationService,
)
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.global_store.v1 import global_store_pb2

_KIND_TO_PROTO = {
    "same_selection": global_store_pb2.GROUP_REALIZATION_KIND_SAME_SELECTION,
    "per_part_selection": global_store_pb2.GROUP_REALIZATION_KIND_PER_PART_SELECTION,
}

_STATE_TO_PROTO = {
    "open": global_store_pb2.GROUP_REALIZATION_STATE_OPEN,
    "resolved": global_store_pb2.GROUP_REALIZATION_STATE_RESOLVED,
    "preparing": global_store_pb2.GROUP_REALIZATION_STATE_PREPARING,
    "ready_to_publish": global_store_pb2.GROUP_REALIZATION_STATE_READY_TO_PUBLISH,
    "published": global_store_pb2.GROUP_REALIZATION_STATE_PUBLISHED,
    "aborted": global_store_pb2.GROUP_REALIZATION_STATE_ABORTED,
    "expired": global_store_pb2.GROUP_REALIZATION_STATE_EXPIRED,
}

_MEMBER_STATE_TO_PROTO = {
    "joined": global_store_pb2.GROUP_REALIZATION_MEMBER_STATE_JOINED,
    "preparing": global_store_pb2.GROUP_REALIZATION_MEMBER_STATE_PREPARING,
    "prepared": global_store_pb2.GROUP_REALIZATION_MEMBER_STATE_PREPARED,
    "published": global_store_pb2.GROUP_REALIZATION_MEMBER_STATE_PUBLISHED,
    "failed": global_store_pb2.GROUP_REALIZATION_MEMBER_STATE_FAILED,
    "cancelled": global_store_pb2.GROUP_REALIZATION_MEMBER_STATE_CANCELLED,
    "expired": global_store_pb2.GROUP_REALIZATION_MEMBER_STATE_EXPIRED,
}


def _version_set_ref(version_set: dict) -> global_store_pb2.GroupVersionSetRef:
    return global_store_pb2.GroupVersionSetRef(
        version_set_id=str(version_set["version_set_id"]),
        manifest_hash=bytes(version_set["manifest_hash"]),
        manifest_generation=int(version_set["manifest_generation"]),
    )


def _part_to_proto(part: dict) -> global_store_pb2.GroupVersionSetPart:
    return global_store_pb2.GroupVersionSetPart(
        part_id=str(part["part_id"]),
        selection=part["selection"],
        requested_byte_space=GroupRealizationRpcHandler.parse_byte_space(part),
        selection_hash=bytes(part["selection_hash"]),
        logical_layout_hash=bytes(part.get("logical_layout_hash") or b""),
        part_metadata_json=str(part.get("part_metadata_json") or ""),
    )


class GroupRealizationRpcHandler:
    """Owns group realization gRPC behavior and error mapping."""

    def __init__(
        self,
        *,
        group_realization_service: GroupRealizationService,
        operation_repository: OperationRepository,
        logger,
    ) -> None:
        self._service = group_realization_service
        self._operation_repository = operation_repository
        self._logger = logger

    @staticmethod
    def parse_byte_space(part: dict) -> common_pb2.ByteSpaceRef:
        return parse_requested_byte_space_json(str(part["requested_byte_space"]))

    def _map_error(self, exc: Exception, context: grpc.ServicerContext) -> None:
        if isinstance(exc, GroupRealizationFeatureDisabledError):
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details(str(exc))
            return
        if isinstance(exc, (NotFoundError, KeyError)):
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return
        if isinstance(exc, ConflictError):
            context.set_code(grpc.StatusCode.ALREADY_EXISTS)
            context.set_details(str(exc))
            return
        if isinstance(exc, GroupRealizationPreconditionError):
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details(str(exc))
            return
        if isinstance(exc, (ValidationError, ValueError)):
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return
        context.set_code(grpc.StatusCode.INTERNAL)
        context.set_details(str(exc))

    def register_group_version_set(
        self,
        request: global_store_pb2.RegisterGroupVersionSetRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.RegisterGroupVersionSetResponse:
        try:
            key_generation = (
                int(request.key_generation)
                if request.HasField("key_generation")
                else None
            )
            version_set, parts = self._service.register_version_set(
                realization_kind=request.realization_kind,
                parts=list(request.parts),
                namespace=request.namespace or None,
                key=request.key or None,
                key_generation=key_generation,
            )
            return global_store_pb2.RegisterGroupVersionSetResponse(
                status=global_store_pb2.STATUS_OK,
                version_set=_version_set_ref(version_set),
                realization_kind=_KIND_TO_PROTO.get(
                    version_set["realization_kind"],
                    global_store_pb2.GROUP_REALIZATION_KIND_UNSPECIFIED,
                ),
                parts=[_part_to_proto(part) for part in parts],
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error in RegisterGroupVersionSet")
            self._map_error(exc, context)
            return global_store_pb2.RegisterGroupVersionSetResponse(
                status=global_store_pb2.STATUS_ERROR
            )

    def begin_or_join_group_realization(
        self,
        request: global_store_pb2.BeginOrJoinGroupRealizationRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.BeginOrJoinGroupRealizationResponse:
        try:
            result = self._service.begin_or_join(request=request)
            transaction = result["transaction"]
            return global_store_pb2.BeginOrJoinGroupRealizationResponse(
                status=global_store_pb2.STATUS_OK,
                transaction_id=transaction["transaction_id"],
                version_set=_version_set_ref(result["version_set"]),
                realization_kind=_KIND_TO_PROTO.get(
                    transaction["realization_kind"],
                    global_store_pb2.GROUP_REALIZATION_KIND_UNSPECIFIED,
                ),
                part=_part_to_proto(result["part"]),
                state=_STATE_TO_PROTO.get(
                    transaction["state"],
                    global_store_pb2.GROUP_REALIZATION_STATE_UNSPECIFIED,
                ),
                transaction_fingerprint=result["transaction_fingerprint"],
                key_generation=int(result["key_generation"] or 0),
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error in BeginOrJoinGroupRealization")
            self._map_error(exc, context)
            return global_store_pb2.BeginOrJoinGroupRealizationResponse(
                status=global_store_pb2.STATUS_ERROR
            )

    def report_group_realization_prepared(
        self,
        request: global_store_pb2.ReportGroupRealizationPreparedRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.ReportGroupRealizationPreparedResponse:
        try:
            transaction, member, fingerprint = self._service.report_prepared(
                request=request
            )
            return global_store_pb2.ReportGroupRealizationPreparedResponse(
                status=global_store_pb2.STATUS_OK,
                state=_STATE_TO_PROTO.get(
                    transaction["state"],
                    global_store_pb2.GROUP_REALIZATION_STATE_UNSPECIFIED,
                ),
                member_state=_MEMBER_STATE_TO_PROTO.get(
                    member["state"],
                    global_store_pb2.GROUP_REALIZATION_MEMBER_STATE_UNSPECIFIED,
                ),
                member_fingerprint=fingerprint,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error in ReportGroupRealizationPrepared")
            self._map_error(exc, context)
            return global_store_pb2.ReportGroupRealizationPreparedResponse(
                status=global_store_pb2.STATUS_ERROR
            )

    def _authority_ok(
        self,
        authority: global_store_pb2.GroupPublishAuthority,
        *,
        transaction_id: str,
    ) -> bool:
        value = authority.WhichOneof("value")
        if value == "capability_token":
            return False
        if value == "operation_lease_id":
            token = authority.operation_lease_id.strip()
            lease = (
                self._operation_repository.get_active_lease_token(lease_token=token)
                if token
                else None
            )
            return (
                lease is not None
                and lease["kind"] == "group_realization_publish"
                and lease["target_artifact_id"] == transaction_id
            )
        return False

    def publish_group_realization(
        self,
        request: global_store_pb2.PublishGroupRealizationRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.PublishGroupRealizationResponse:
        try:
            if (
                self._service._config.publish_authority_mode  # noqa: SLF001
                == "COORDINATOR_EXPLICIT"
                and not self._authority_ok(
                    request.authority,
                    transaction_id=request.transaction_id,
                )
            ):
                context.set_code(grpc.StatusCode.PERMISSION_DENIED)
                context.set_details("publish authority is required")
                return global_store_pb2.PublishGroupRealizationResponse(
                    status=global_store_pb2.STATUS_ERROR
                )
            transaction = self._service.publish(
                transaction_id=request.transaction_id,
                require_ready_to_publish=bool(request.require_ready_to_publish),
            )
            return global_store_pb2.PublishGroupRealizationResponse(
                status=global_store_pb2.STATUS_OK,
                state=_STATE_TO_PROTO.get(
                    transaction["state"],
                    global_store_pb2.GROUP_REALIZATION_STATE_UNSPECIFIED,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error in PublishGroupRealization")
            self._map_error(exc, context)
            return global_store_pb2.PublishGroupRealizationResponse(
                status=global_store_pb2.STATUS_ERROR
            )

    def wait_group_realization_published(
        self,
        request: global_store_pb2.WaitGroupRealizationPublishedRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.WaitGroupRealizationPublishedResponse:
        try:
            transaction = self._service.wait_published(
                transaction_id=request.transaction_id,
                deadline_unix_nanos=int(request.deadline_unix_nanos),
            )
            status = (
                global_store_pb2.STATUS_OK
                if transaction["state"] == "published"
                else global_store_pb2.STATUS_TIMED_OUT
            )
            return global_store_pb2.WaitGroupRealizationPublishedResponse(
                status=status,
                state=_STATE_TO_PROTO.get(
                    transaction["state"],
                    global_store_pb2.GROUP_REALIZATION_STATE_UNSPECIFIED,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error in WaitGroupRealizationPublished")
            self._map_error(exc, context)
            return global_store_pb2.WaitGroupRealizationPublishedResponse(
                status=global_store_pb2.STATUS_ERROR
            )

    def abort_group_realization(
        self,
        request: global_store_pb2.AbortGroupRealizationRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.AbortGroupRealizationResponse:
        try:
            transaction = self._service.abort(
                transaction_id=request.transaction_id,
                failure_code=request.failure_code or "aborted",
                failure_detail=request.failure_detail,
            )
            return global_store_pb2.AbortGroupRealizationResponse(
                status=global_store_pb2.STATUS_OK,
                state=_STATE_TO_PROTO.get(
                    transaction["state"],
                    global_store_pb2.GROUP_REALIZATION_STATE_UNSPECIFIED,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error in AbortGroupRealization")
            self._map_error(exc, context)
            return global_store_pb2.AbortGroupRealizationResponse(
                status=global_store_pb2.STATUS_ERROR
            )

    def get_group_realization(
        self,
        request: global_store_pb2.GetGroupRealizationRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.GetGroupRealizationResponse:
        try:
            diagnostic = self._service.get_diagnostic(
                transaction_id=request.transaction_id
            )
            if diagnostic is None:
                return global_store_pb2.GetGroupRealizationResponse(
                    status=global_store_pb2.STATUS_NOT_FOUND
                )
            transaction = diagnostic["transaction"]
            members = [
                global_store_pb2.GroupRealizationMemberDiagnostic(
                    part_id=member["part_id"],
                    state=_MEMBER_STATE_TO_PROTO.get(
                        member["state"],
                        global_store_pb2.GROUP_REALIZATION_MEMBER_STATE_UNSPECIFIED,
                    ),
                    daemon_id=member["daemon_id"],
                    daemon_session_id=member["daemon_session_id"] or "",
                    worker_id=member["worker_id"] or "",
                    materialization_attempt_id=member["materialization_attempt_id"]
                    or "",
                    staged_value=global_store_pb2.StagedBindingValueRef(
                        daemon_id=member["daemon_id"],
                        daemon_session_id=member["daemon_session_id"] or "",
                        binding_id=member["staged_binding_id"] or "",
                        binding_value_id=member["staged_binding_value_id"] or "",
                        staging_token=member["staging_token"] or "",
                        staging_epoch=int(member["staging_epoch"] or 0),
                    ),
                    source_replica_id=member["source_replica_id"] or "",
                    source_export_generation=int(
                        member["source_export_generation"] or 0
                    ),
                    child_transport_request_id=member["child_transport_request_id"]
                    or "",
                    failure_code=member["failure_code"] or "",
                    failure_detail=member["failure_detail"] or "",
                    member_fingerprint=member["member_fingerprint"] or b"",
                )
                for member in diagnostic["members"]
            ]
            return global_store_pb2.GetGroupRealizationResponse(
                status=global_store_pb2.STATUS_OK,
                transaction_id=transaction["transaction_id"],
                version_set=global_store_pb2.GroupVersionSetRef(
                    version_set_id=transaction["version_set_id"],
                    manifest_hash=transaction["manifest_hash"] or b"",
                    manifest_generation=1,
                ),
                realization_kind=_KIND_TO_PROTO.get(
                    transaction["realization_kind"],
                    global_store_pb2.GROUP_REALIZATION_KIND_UNSPECIFIED,
                ),
                state=_STATE_TO_PROTO.get(
                    transaction["state"],
                    global_store_pb2.GROUP_REALIZATION_STATE_UNSPECIFIED,
                ),
                required_part_ids=transaction["required_part_ids"],
                missing_part_ids=diagnostic["missing_part_ids"],
                members=members,
                group_kind=transaction["group_kind"],
                group_id=transaction["group_id"],
                epoch=int(transaction["epoch"]),
                total_parts=int(transaction["total_parts"]),
                prepared_count=int(transaction["prepared_count"]),
                failed_count=int(transaction["failed_count"]),
                published_count=int(transaction["published_count"]),
                failure_code=transaction["failure_code"] or "",
                failure_detail=transaction["failure_detail"] or "",
                deadline_unix_nanos=int(transaction["deadline_unix_nanos"] or 0),
                publish_authority_mode=self._service._config.publish_authority_mode,  # noqa: SLF001
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error in GetGroupRealization")
            self._map_error(exc, context)
            return global_store_pb2.GetGroupRealizationResponse(
                status=global_store_pb2.STATUS_ERROR
            )
