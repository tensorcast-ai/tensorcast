#  Copyright (c) 2025-2026, TensorCast Team.

"""RPC handler for progressive replica dissemination APIs."""

from __future__ import annotations

from datetime import datetime, timezone

import grpc

from tensorcast.global_store.exceptions import NotFoundError, ValidationError
from tensorcast.global_store.models import (
    ProgressiveAssignment,
    ProgressiveAssignmentState,
    ProgressiveCoverageIdentity,
    ProgressiveCoverageKind,
    ProgressiveCoverageReport,
    ProgressiveCoverageState,
    ProgressiveExportState,
)
from tensorcast.global_store.services.progressive_service import (
    ProgressiveReplicationService,
)
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.global_store.v1 import global_store_pb2


def _bytes_hex(value: bytes) -> str:
    return bytes(value or b"").hex()


def _byte_space_kind_to_text(value: int) -> str:
    if value == common_pb2.BYTE_SPACE_KIND_CANONICAL:
        return "canonical"
    if value == common_pb2.BYTE_SPACE_KIND_VIEW:
        return "view"
    return ""


def _deadline_from_unix_nanos(value: int) -> datetime | None:
    if int(value or 0) <= 0:
        return None
    return datetime.fromtimestamp(float(value) / 1_000_000_000.0, tz=timezone.utc)


def _deadline_to_unix_nanos(value: datetime) -> int:
    return int(value.timestamp() * 1_000_000_000)


def _memory_type_to_proto(value: str) -> common_pb2.MemoryType:
    normalized = str(value or "").strip().upper()
    if normalized == "GPU":
        return common_pb2.MEMORY_TYPE_GPU
    if normalized == "RAM":
        return common_pb2.MEMORY_TYPE_RAM
    if normalized == "DISK":
        return common_pb2.MEMORY_TYPE_DISK
    return common_pb2.MEMORY_TYPE_UNSPECIFIED


_COVERAGE_STATE_FROM_PROTO = {
    global_store_pb2.PROGRESSIVE_COVERAGE_STATE_PENDING: ProgressiveCoverageState.PENDING,
    global_store_pb2.PROGRESSIVE_COVERAGE_STATE_VERIFIED: ProgressiveCoverageState.VERIFIED,
    global_store_pb2.PROGRESSIVE_COVERAGE_STATE_FAILED: ProgressiveCoverageState.FAILED,
    global_store_pb2.PROGRESSIVE_COVERAGE_STATE_RETIRED: ProgressiveCoverageState.RETIRED,
}

_COVERAGE_STATE_TO_PROTO = {
    ProgressiveCoverageState.PENDING: global_store_pb2.PROGRESSIVE_COVERAGE_STATE_PENDING,
    ProgressiveCoverageState.VERIFIED: global_store_pb2.PROGRESSIVE_COVERAGE_STATE_VERIFIED,
    ProgressiveCoverageState.FAILED: global_store_pb2.PROGRESSIVE_COVERAGE_STATE_FAILED,
    ProgressiveCoverageState.RETIRED: global_store_pb2.PROGRESSIVE_COVERAGE_STATE_RETIRED,
}

_EXPORT_STATE_FROM_PROTO = {
    global_store_pb2.PROGRESSIVE_EXPORT_STATE_NOT_EXPORTABLE: ProgressiveExportState.NOT_EXPORTABLE,
    global_store_pb2.PROGRESSIVE_EXPORT_STATE_IN_PROGRESS_EXPORTABLE: ProgressiveExportState.IN_PROGRESS_EXPORTABLE,
    global_store_pb2.PROGRESSIVE_EXPORT_STATE_COMPLETE_EXPORTABLE: ProgressiveExportState.COMPLETE_EXPORTABLE,
}

_ASSIGNMENT_STATE_FROM_PROTO = {
    global_store_pb2.PROGRESSIVE_ASSIGNMENT_STATE_CLAIMED: ProgressiveAssignmentState.CLAIMED,
    global_store_pb2.PROGRESSIVE_ASSIGNMENT_STATE_READING: ProgressiveAssignmentState.READING,
    global_store_pb2.PROGRESSIVE_ASSIGNMENT_STATE_SUCCEEDED: ProgressiveAssignmentState.SUCCEEDED,
    global_store_pb2.PROGRESSIVE_ASSIGNMENT_STATE_FAILED: ProgressiveAssignmentState.FAILED,
    global_store_pb2.PROGRESSIVE_ASSIGNMENT_STATE_EXPIRED: ProgressiveAssignmentState.EXPIRED,
    global_store_pb2.PROGRESSIVE_ASSIGNMENT_STATE_CANCELLED: ProgressiveAssignmentState.CANCELLED,
}

_ASSIGNMENT_STATE_TO_PROTO = {
    ProgressiveAssignmentState.CLAIMED: global_store_pb2.PROGRESSIVE_ASSIGNMENT_STATE_CLAIMED,
    ProgressiveAssignmentState.READING: global_store_pb2.PROGRESSIVE_ASSIGNMENT_STATE_READING,
    ProgressiveAssignmentState.SUCCEEDED: global_store_pb2.PROGRESSIVE_ASSIGNMENT_STATE_SUCCEEDED,
    ProgressiveAssignmentState.FAILED: global_store_pb2.PROGRESSIVE_ASSIGNMENT_STATE_FAILED,
    ProgressiveAssignmentState.EXPIRED: global_store_pb2.PROGRESSIVE_ASSIGNMENT_STATE_EXPIRED,
    ProgressiveAssignmentState.CANCELLED: global_store_pb2.PROGRESSIVE_ASSIGNMENT_STATE_CANCELLED,
}


class ProgressiveRpcHandler:
    """Owns progressive dissemination RPC behavior and error mapping."""

    def __init__(self, *, service: ProgressiveReplicationService, logger) -> None:
        self._service = service
        self._logger = logger

    def report_progressive_coverage(
        self,
        request: global_store_pb2.ReportProgressiveCoverageRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.ReportProgressiveCoverageResponse:
        try:
            report = self._coverage_report_from_proto(request)
            result = self._service.report_progressive_coverage(report)
            return global_store_pb2.ReportProgressiveCoverageResponse(
                status=global_store_pb2.Status.STATUS_OK,
                coverage_id=result.coverage_id,
                state=_COVERAGE_STATE_TO_PROTO[result.state],
                updated=result.updated,
                throttled=result.throttled,
                reason=result.reason,
            )
        except ValidationError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.ReportProgressiveCoverageResponse(
                status=global_store_pb2.Status.STATUS_ERROR,
                reason=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error reporting progressive coverage")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.ReportProgressiveCoverageResponse(
                status=global_store_pb2.Status.STATUS_ERROR,
                reason=str(exc),
            )

    def find_progressive_source(
        self,
        request: global_store_pb2.FindProgressiveSourceRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.FindProgressiveSourceResponse:
        try:
            result = self._service.find_progressive_source(
                identity=self._identity_from_proto(request.identity),
                next_unit=int(request.next_unit),
                max_units=int(request.max_units),
                requester_daemon_id=request.requester_daemon_id.strip(),
                requester_worker_id=request.requester_worker_id.strip(),
                requester_source_domain=request.requester_source_domain.strip(),
                requester_materialization_attempt_id=request.requester_materialization_attempt_id.strip(),
                request_fingerprint=bytes(request.request_fingerprint),
                deadline_at=_deadline_from_unix_nanos(request.deadline_unix_nanos),
            )
            if result.assignment is None:
                return global_store_pb2.FindProgressiveSourceResponse(
                    status=global_store_pb2.Status.STATUS_NOT_FOUND,
                    no_eligible_reason=result.no_eligible_reason,
                )
            response = global_store_pb2.FindProgressiveSourceResponse(
                status=global_store_pb2.Status.STATUS_OK,
                replayed=result.replayed,
            )
            response.assignment.CopyFrom(self._assignment_to_proto(result.assignment))
            return response
        except ValidationError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.FindProgressiveSourceResponse(
                status=global_store_pb2.Status.STATUS_ERROR,
                no_eligible_reason=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error finding progressive source")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.FindProgressiveSourceResponse(
                status=global_store_pb2.Status.STATUS_ERROR,
                no_eligible_reason=str(exc),
            )

    def complete_progressive_assignment(
        self,
        request: global_store_pb2.CompleteProgressiveAssignmentRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.CompleteProgressiveAssignmentResponse:
        try:
            outcome = _ASSIGNMENT_STATE_FROM_PROTO.get(request.outcome)
            if outcome is None:
                raise ValidationError(
                    "complete progressive assignment requires explicit outcome"
                )
            released = self._service.complete_progressive_assignment(
                assignment_id=request.assignment_id,
                outcome=outcome,
                outcome_detail=request.outcome_detail.strip() or None,
            )
            return global_store_pb2.CompleteProgressiveAssignmentResponse(
                status=global_store_pb2.Status.STATUS_OK,
                released_counter=released,
            )
        except NotFoundError:
            return global_store_pb2.CompleteProgressiveAssignmentResponse(
                status=global_store_pb2.Status.STATUS_NOT_FOUND
            )
        except ValidationError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.CompleteProgressiveAssignmentResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error completing progressive assignment")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.CompleteProgressiveAssignmentResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def retire_progressive_coverage(
        self,
        request: global_store_pb2.RetireProgressiveCoverageRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.RetireProgressiveCoverageResponse:
        try:
            state = _COVERAGE_STATE_FROM_PROTO.get(request.state)
            if state is None:
                raise ValidationError("retire progressive coverage requires state")
            retired_rows, retired_assignments = (
                self._service.retire_progressive_coverage(
                    coverage_id=request.coverage_id.strip() or None,
                    replica_id=request.replica_id.strip() or None,
                    daemon_id=request.daemon_id.strip() or None,
                    source_export_generation=(
                        int(request.source_export_generation)
                        if int(request.source_export_generation) > 0
                        else None
                    ),
                    state=state,
                    reason=request.reason.strip() or None,
                )
            )
            return global_store_pb2.RetireProgressiveCoverageResponse(
                status=global_store_pb2.Status.STATUS_OK,
                retired_coverage_rows=retired_rows,
                retired_assignments=retired_assignments,
            )
        except ValidationError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.RetireProgressiveCoverageResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error retiring progressive coverage")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.RetireProgressiveCoverageResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def expire_progressive_state(
        self,
        request: global_store_pb2.ExpireProgressiveStateRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.ExpireProgressiveStateResponse:
        try:
            expired_coverage, expired_assignments = (
                self._service.expire_progressive_state(
                    coverage_batch_limit=(
                        int(request.coverage_batch_limit)
                        if int(request.coverage_batch_limit) > 0
                        else None
                    ),
                    assignment_batch_limit=(
                        int(request.assignment_batch_limit)
                        if int(request.assignment_batch_limit) > 0
                        else None
                    ),
                )
            )
            return global_store_pb2.ExpireProgressiveStateResponse(
                status=global_store_pb2.Status.STATUS_OK,
                expired_coverage_rows=expired_coverage,
                expired_assignments=expired_assignments,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error expiring progressive state")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.ExpireProgressiveStateResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def _coverage_report_from_proto(
        self, request: global_store_pb2.ReportProgressiveCoverageRequest
    ) -> ProgressiveCoverageReport:
        state = _COVERAGE_STATE_FROM_PROTO.get(request.state)
        if state is None:
            raise ValidationError("progressive coverage state is required")
        export_state = _EXPORT_STATE_FROM_PROTO.get(request.export_state)
        if export_state is None:
            raise ValidationError("progressive coverage export_state is required")
        if (
            request.coverage_kind
            != global_store_pb2.PROGRESSIVE_COVERAGE_KIND_BYTE_PREFIX
        ):
            raise ValidationError("progressive v1 supports byte-prefix coverage only")
        return ProgressiveCoverageReport(
            coverage_id=request.coverage_id.strip(),
            identity=self._identity_from_proto(request.identity),
            replica_id=request.replica_id.strip(),
            daemon_id=request.daemon_id.strip(),
            daemon_session_id=request.daemon_session_id.strip() or None,
            worker_id=request.worker_id.strip(),
            source_export_generation=int(request.source_export_generation),
            coverage_epoch=int(request.coverage_epoch),
            coverage_kind=ProgressiveCoverageKind.BYTE_PREFIX,
            state=state,
            export_state=export_state,
            verified_units=int(request.verified_units),
            verified_bytes=int(request.verified_bytes),
            completed_units=int(request.completed_units),
            completed_bytes=int(request.completed_bytes),
            total_units=int(request.total_units),
            total_bytes=int(request.total_bytes),
            materialization_attempt_id=request.materialization_attempt_id.strip(),
            source_transport_id=request.source_transport_id.strip() or None,
            source_domain=request.source_domain.strip(),
            seed_transport_kind=request.seed_transport_kind.strip() or None,
            deadline_at=_deadline_from_unix_nanos(request.deadline_unix_nanos),
        )

    @staticmethod
    def _identity_from_proto(
        proto: global_store_pb2.ProgressiveCoverageIdentity,
    ) -> ProgressiveCoverageIdentity:
        byte_space = proto.byte_space
        hash_byte_space = proto.hash_space.byte_space
        return ProgressiveCoverageIdentity(
            artifact_id=proto.artifact_id.strip(),
            byte_space_kind=_byte_space_kind_to_text(byte_space.kind),
            byte_space_id=byte_space.id.strip(),
            selection_hash=_bytes_hex(proto.selection_hash),
            logical_layout_hash=_bytes_hex(proto.logical_layout_hash),
            hash_space_kind=_byte_space_kind_to_text(hash_byte_space.kind),
            hash_space_id=hash_byte_space.id.strip(),
            canonical_index_multihash=proto.hash_space.canonical_index_multihash.strip(),
            coverage_order_hash=_bytes_hex(proto.coverage_order_hash),
            group_version_set_id=proto.group_version_set_id.strip(),
            group_part_id=proto.group_part_id.strip(),
        )

    @staticmethod
    def _assignment_to_proto(
        assignment: ProgressiveAssignment,
    ) -> global_store_pb2.ProgressiveSourceAssignment:
        proto = global_store_pb2.ProgressiveSourceAssignment(
            assignment_id=assignment.assignment_id,
            coverage_id=assignment.coverage_id,
            replica_id=assignment.replica_id,
            source_daemon_id=assignment.source_daemon_id,
            source_worker_id=assignment.source_worker_id,
            source_domain=assignment.source_domain,
            seed_transport_kind=assignment.seed_transport_kind or "",
            requester_daemon_id=assignment.requester_daemon_id,
            requester_worker_id=assignment.requester_worker_id,
            start_unit=assignment.start_unit,
            end_unit_exclusive=assignment.end_unit_exclusive,
            start_byte=assignment.start_byte,
            end_byte_exclusive=assignment.end_byte_exclusive,
            source_export_generation=assignment.source_export_generation,
            deadline_unix_nanos=_deadline_to_unix_nanos(assignment.deadline_at),
            state=_ASSIGNMENT_STATE_TO_PROTO[assignment.state],
        )
        if assignment.source_memory is not None:
            source_memory = assignment.source_memory
            memory_info = proto.source_memory_info
            memory_info.node_id = source_memory.node_id
            memory_info.node_address = source_memory.node_address
            memory_info.node_port = int(source_memory.node_port)
            memory_info.memory_size = int(source_memory.memory_size)
            memory_info.memory_type = _memory_type_to_proto(source_memory.memory_type)
            memory_info.device_id = max(0, int(source_memory.device_id))
            memory_info.byte_space.kind = common_pb2.BYTE_SPACE_KIND_CANONICAL
            transport = memory_info.transport
            transport.export_state = (
                common_pb2.ReplicaTransportMetadata.EXPORT_STATE_EXPORTABLE
            )
            transport.export_generation = int(assignment.source_export_generation)
            transport.remote_memory_keys.extend(
                source_memory.transport.remote_memory_keys
            )
            transport.buffer_sizes.extend(
                int(size) for size in source_memory.transport.buffer_sizes
            )
            if source_memory.transport.verification_json:
                transport.verification_json = source_memory.transport.verification_json
        return proto
