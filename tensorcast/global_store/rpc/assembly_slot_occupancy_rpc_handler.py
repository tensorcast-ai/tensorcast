#  Copyright (c) 2026, TensorCast Team.

"""Assembly slot-occupancy RPC handler."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable

import grpc
from google.protobuf import timestamp_pb2

from tensorcast.global_store.exceptions import ValidationError
from tensorcast.global_store.repositories.assembly_slot_occupancy_repository import (
    AssemblySlotOccupancyRepository,
)
from tensorcast.proto.global_store.v1 import global_store_pb2


class AssemblySlotOccupancyRpcHandler:
    """Owns durable `(attempt_id, slot_id)` slot occupancy RPC behavior."""

    _DEFAULT_ACCEPTED_LEASE_TTL = timedelta(seconds=20)

    def __init__(
        self,
        *,
        assembly_slot_occupancy_repository: AssemblySlotOccupancyRepository,
        datetime_to_timestamp: Callable[
            [datetime | None], timestamp_pb2.Timestamp | None
        ],
        coerce_db_datetime: Callable[[object], datetime | None],
        logger,
    ) -> None:
        self._assembly_slot_occupancy_repository = assembly_slot_occupancy_repository
        self._datetime_to_timestamp = datetime_to_timestamp
        self._coerce_db_datetime = coerce_db_datetime
        self._logger = logger

    def get_assembly_slot_occupancy(
        self,
        request: global_store_pb2.GetAssemblySlotOccupancyRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.GetAssemblySlotOccupancyResponse:
        if not request.attempt_id or not request.slot_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("attempt_id and slot_id are required")
            return global_store_pb2.GetAssemblySlotOccupancyResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        try:
            row = self._assembly_slot_occupancy_repository.get(
                attempt_id=str(request.attempt_id),
                slot_id=str(request.slot_id),
            )
            if row is None:
                return global_store_pb2.GetAssemblySlotOccupancyResponse(
                    status=global_store_pb2.Status.STATUS_NOT_FOUND
                )
            return global_store_pb2.GetAssemblySlotOccupancyResponse(
                status=global_store_pb2.Status.STATUS_OK,
                occupancy=self._to_proto(row),
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("GetAssemblySlotOccupancy failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.GetAssemblySlotOccupancyResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def upsert_assembly_slot_occupancy(
        self,
        request: global_store_pb2.UpsertAssemblySlotOccupancyRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UpsertAssemblySlotOccupancyResponse:
        if not request.HasField("occupancy"):
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("occupancy is required")
            return global_store_pb2.UpsertAssemblySlotOccupancyResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        occupancy = request.occupancy
        try:
            self._validate_occupancy(occupancy)
            lease_expires_at = None
            if occupancy.HasField("lease_expires_at"):
                lease_expires_at = self._coerce_db_datetime(
                    occupancy.lease_expires_at.ToDatetime()
                )
            elif occupancy.state == "accepted":
                lease_expires_at = (
                    datetime.now(timezone.utc) + self._DEFAULT_ACCEPTED_LEASE_TTL
                )
            row = self._assembly_slot_occupancy_repository.claim_slot(
                attempt_id=str(occupancy.attempt_id),
                slot_id=str(occupancy.slot_id),
                structural_view_id=str(occupancy.structural_view_id or "") or None,
                binding_id=str(occupancy.binding_id),
                binding_value_id=str(occupancy.binding_value_id),
                coverage_plan_hash=str(occupancy.coverage_plan_hash),
                contributor_daemon_id=str(occupancy.contributor_daemon_id),
                coordinator_operation_id=str(occupancy.coordinator_operation_id),
                coordinator_generation=int(occupancy.coordinator_generation),
                lease_id=str(occupancy.lease_id),
                lease_generation=int(occupancy.lease_generation),
                lease_expires_at=lease_expires_at,
                state=str(occupancy.state),
            )
            if row is None:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details(
                    "assembly slot occupancy is already held by a live contributor"
                )
                return global_store_pb2.UpsertAssemblySlotOccupancyResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )
            return global_store_pb2.UpsertAssemblySlotOccupancyResponse(
                status=global_store_pb2.Status.STATUS_OK,
                occupancy=self._to_proto(row),
            )
        except ValidationError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.UpsertAssemblySlotOccupancyResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("UpsertAssemblySlotOccupancy failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.UpsertAssemblySlotOccupancyResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def list_assembly_slot_occupancies(
        self,
        request: global_store_pb2.ListAssemblySlotOccupanciesRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.ListAssemblySlotOccupanciesResponse:
        try:
            rows = self._assembly_slot_occupancy_repository.list(
                attempt_id=str(request.attempt_id or "") or None,
                slot_id=str(request.slot_id or "") or None,
                binding_id=str(request.binding_id or "") or None,
                binding_value_id=str(request.binding_value_id or "") or None,
                states=tuple(str(state) for state in request.states)
                if request.states
                else None,
            )
            response = global_store_pb2.ListAssemblySlotOccupanciesResponse(
                status=global_store_pb2.Status.STATUS_OK
            )
            response.occupancies.extend(self._to_proto(row) for row in rows)
            return response
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("ListAssemblySlotOccupancies failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.ListAssemblySlotOccupanciesResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def update_assembly_slot_occupancy_state(
        self,
        request: global_store_pb2.UpdateAssemblySlotOccupancyStateRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UpdateAssemblySlotOccupancyStateResponse:
        if not request.attempt_id or not request.slot_id or not request.state:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("attempt_id, slot_id, and state are required")
            return global_store_pb2.UpdateAssemblySlotOccupancyStateResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        if request.state == "accepted" and not request.HasField("lease_expires_at"):
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("accepted occupancy updates require lease_expires_at")
            return global_store_pb2.UpdateAssemblySlotOccupancyStateResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        try:
            row = self._assembly_slot_occupancy_repository.update_state_if_current(
                attempt_id=str(request.attempt_id),
                slot_id=str(request.slot_id),
                state=str(request.state),
                expected_lease_id=str(request.expected_lease_id or "") or None,
                expected_lease_generation=(
                    int(request.expected_lease_generation)
                    if request.expected_lease_generation > 0
                    else None
                ),
                current_states=tuple(str(state) for state in request.current_states)
                if request.current_states
                else None,
                lease_expires_at=(
                    self._coerce_db_datetime(request.lease_expires_at.ToDatetime())
                    if request.HasField("lease_expires_at")
                    else None
                ),
            )
            if row is None:
                return global_store_pb2.UpdateAssemblySlotOccupancyStateResponse(
                    status=global_store_pb2.Status.STATUS_NOT_FOUND
                )
            return global_store_pb2.UpdateAssemblySlotOccupancyStateResponse(
                status=global_store_pb2.Status.STATUS_OK,
                occupancy=self._to_proto(row),
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("UpdateAssemblySlotOccupancyState failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.UpdateAssemblySlotOccupancyStateResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    @staticmethod
    def _validate_occupancy(
        occupancy: global_store_pb2.AssemblySlotOccupancy,
    ) -> None:
        required = {
            "attempt_id": occupancy.attempt_id,
            "slot_id": occupancy.slot_id,
            "binding_id": occupancy.binding_id,
            "binding_value_id": occupancy.binding_value_id,
            "coverage_plan_hash": occupancy.coverage_plan_hash,
            "contributor_daemon_id": occupancy.contributor_daemon_id,
            "coordinator_operation_id": occupancy.coordinator_operation_id,
            "lease_id": occupancy.lease_id,
            "state": occupancy.state,
        }
        missing = [name for name, value in required.items() if not str(value).strip()]
        if missing:
            raise ValidationError(
                f"missing required occupancy fields: {', '.join(missing)}"
            )
        if occupancy.coordinator_generation <= 0:
            raise ValidationError("coordinator_generation must be > 0")
        if occupancy.lease_generation <= 0:
            raise ValidationError("lease_generation must be > 0")

    def _to_proto(
        self,
        row: dict[str, object],
    ) -> global_store_pb2.AssemblySlotOccupancy:
        occupancy = global_store_pb2.AssemblySlotOccupancy(
            attempt_id=str(row["attempt_id"]),
            slot_id=str(row["slot_id"]),
            structural_view_id=str(row["structural_view_id"] or ""),
            binding_id=str(row["binding_id"]),
            binding_value_id=str(row["binding_value_id"]),
            coverage_plan_hash=str(row["coverage_plan_hash"]),
            contributor_daemon_id=str(row["contributor_daemon_id"]),
            coordinator_operation_id=str(row["coordinator_operation_id"]),
            coordinator_generation=int(row["coordinator_generation"]),
            lease_id=str(row["lease_id"]),
            lease_generation=int(row["lease_generation"]),
            state=str(row["state"]),
        )
        lease_expires_at = self._datetime_to_timestamp(
            self._coerce_db_datetime(row.get("lease_expires_at"))
        )
        if lease_expires_at is not None:
            occupancy.lease_expires_at.CopyFrom(lease_expires_at)
        created_at = self._datetime_to_timestamp(
            self._coerce_db_datetime(row.get("created_at"))
        )
        if created_at is not None:
            occupancy.created_at.CopyFrom(created_at)
        updated_at = self._datetime_to_timestamp(
            self._coerce_db_datetime(row.get("updated_at"))
        )
        if updated_at is not None:
            occupancy.updated_at.CopyFrom(updated_at)
        return occupancy


__all__ = ["AssemblySlotOccupancyRpcHandler"]
