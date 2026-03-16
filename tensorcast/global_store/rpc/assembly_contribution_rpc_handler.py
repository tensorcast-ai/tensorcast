#  Copyright (c) 2026, TensorCast Team.

"""Assembly contribution occupancy RPC handler."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable

import grpc
from google.protobuf import timestamp_pb2

from tensorcast.global_store.exceptions import ValidationError
from tensorcast.global_store.repositories.assembly_contribution_repository import (
    AssemblyContributionRepository,
)
from tensorcast.proto.global_store.v1 import global_store_pb2


class AssemblyContributionRpcHandler:
    """Owns durable `(assembly_id, view_id)` contributor row operations."""

    _DEFAULT_ACCEPTED_LEASE_TTL = timedelta(seconds=20)

    def __init__(
        self,
        *,
        assembly_contribution_repository: AssemblyContributionRepository,
        datetime_to_timestamp: Callable[
            [datetime | None], timestamp_pb2.Timestamp | None
        ],
        coerce_db_datetime: Callable[[object], datetime | None],
        logger,
    ) -> None:
        self._assembly_contribution_repository = assembly_contribution_repository
        self._datetime_to_timestamp = datetime_to_timestamp
        self._coerce_db_datetime = coerce_db_datetime
        self._logger = logger

    def get_assembly_contribution(
        self,
        request: global_store_pb2.GetAssemblyContributionRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.GetAssemblyContributionResponse:
        if not request.assembly_id or not request.view_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("assembly_id and view_id are required")
            return global_store_pb2.GetAssemblyContributionResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        try:
            row = self._assembly_contribution_repository.get(
                assembly_id=request.assembly_id,
                view_id=request.view_id,
            )
            if row is None:
                return global_store_pb2.GetAssemblyContributionResponse(
                    status=global_store_pb2.Status.STATUS_NOT_FOUND
                )
            return global_store_pb2.GetAssemblyContributionResponse(
                status=global_store_pb2.Status.STATUS_OK,
                contribution=self._to_proto(row),
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("GetAssemblyContribution failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.GetAssemblyContributionResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def upsert_assembly_contribution(
        self,
        request: global_store_pb2.UpsertAssemblyContributionRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UpsertAssemblyContributionResponse:
        if not request.HasField("contribution"):
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("contribution is required")
            return global_store_pb2.UpsertAssemblyContributionResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        contribution = request.contribution
        try:
            self._validate_contribution(contribution)
            lease_expires_at = None
            if contribution.HasField("lease_expires_at"):
                lease_expires_at = self._coerce_db_datetime(
                    contribution.lease_expires_at.ToDatetime()
                )
            elif contribution.state == "accepted":
                lease_expires_at = (
                    datetime.now(timezone.utc) + self._DEFAULT_ACCEPTED_LEASE_TTL
                )
            row = self._assembly_contribution_repository.claim_slot(
                assembly_id=str(contribution.assembly_id),
                view_id=str(contribution.view_id),
                binding_id=str(contribution.binding_id),
                binding_value_id=str(contribution.binding_value_id),
                coverage_plan_hash=str(contribution.coverage_plan_hash),
                contributor_daemon_id=str(contribution.contributor_daemon_id),
                coordinator_operation_id=str(contribution.coordinator_operation_id),
                coordinator_generation=int(contribution.coordinator_generation),
                lease_id=str(contribution.lease_id),
                lease_generation=int(contribution.lease_generation),
                lease_expires_at=lease_expires_at,
                state=str(contribution.state),
            )
            if row is None:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details(
                    "assembly contribution slot is already occupied by a live contributor"
                )
                return global_store_pb2.UpsertAssemblyContributionResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )
            return global_store_pb2.UpsertAssemblyContributionResponse(
                status=global_store_pb2.Status.STATUS_OK,
                contribution=self._to_proto(row),
            )
        except ValidationError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.UpsertAssemblyContributionResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("UpsertAssemblyContribution failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.UpsertAssemblyContributionResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def list_assembly_contributions(
        self,
        request: global_store_pb2.ListAssemblyContributionsRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.ListAssemblyContributionsResponse:
        try:
            rows = self._assembly_contribution_repository.list(
                assembly_id=str(request.assembly_id or "") or None,
                view_id=str(request.view_id or "") or None,
                binding_id=str(request.binding_id or "") or None,
                binding_value_id=str(request.binding_value_id or "") or None,
                states=tuple(str(state) for state in request.states)
                if request.states
                else None,
            )
            response = global_store_pb2.ListAssemblyContributionsResponse(
                status=global_store_pb2.Status.STATUS_OK
            )
            response.contributions.extend(self._to_proto(row) for row in rows)
            return response
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("ListAssemblyContributions failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.ListAssemblyContributionsResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def update_assembly_contribution_state(
        self,
        request: global_store_pb2.UpdateAssemblyContributionStateRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UpdateAssemblyContributionStateResponse:
        if not request.assembly_id or not request.view_id or not request.state:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("assembly_id, view_id, and state are required")
            return global_store_pb2.UpdateAssemblyContributionStateResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        if request.state == "accepted" and not request.HasField("lease_expires_at"):
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(
                "accepted contribution updates require lease_expires_at"
            )
            return global_store_pb2.UpdateAssemblyContributionStateResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        try:
            row = self._assembly_contribution_repository.update_state_if_current(
                assembly_id=str(request.assembly_id),
                view_id=str(request.view_id),
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
                return global_store_pb2.UpdateAssemblyContributionStateResponse(
                    status=global_store_pb2.Status.STATUS_NOT_FOUND
                )
            return global_store_pb2.UpdateAssemblyContributionStateResponse(
                status=global_store_pb2.Status.STATUS_OK,
                contribution=self._to_proto(row),
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("UpdateAssemblyContributionState failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.UpdateAssemblyContributionStateResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    @staticmethod
    def _validate_contribution(
        contribution: global_store_pb2.AssemblyContribution,
    ) -> None:
        required = {
            "assembly_id": contribution.assembly_id,
            "view_id": contribution.view_id,
            "binding_id": contribution.binding_id,
            "binding_value_id": contribution.binding_value_id,
            "coverage_plan_hash": contribution.coverage_plan_hash,
            "contributor_daemon_id": contribution.contributor_daemon_id,
            "coordinator_operation_id": contribution.coordinator_operation_id,
            "lease_id": contribution.lease_id,
            "state": contribution.state,
        }
        missing = [name for name, value in required.items() if not str(value).strip()]
        if missing:
            raise ValidationError(
                f"missing required contribution fields: {', '.join(missing)}"
            )
        if contribution.coordinator_generation <= 0:
            raise ValidationError("coordinator_generation must be > 0")
        if contribution.lease_generation <= 0:
            raise ValidationError("lease_generation must be > 0")

    def _to_proto(
        self, row: dict[str, object]
    ) -> global_store_pb2.AssemblyContribution:
        contribution = global_store_pb2.AssemblyContribution(
            assembly_id=str(row["assembly_id"]),
            view_id=str(row["view_id"]),
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
            contribution.lease_expires_at.CopyFrom(lease_expires_at)
        created_at = self._datetime_to_timestamp(
            self._coerce_db_datetime(row.get("created_at"))
        )
        if created_at is not None:
            contribution.created_at.CopyFrom(created_at)
        updated_at = self._datetime_to_timestamp(
            self._coerce_db_datetime(row.get("updated_at"))
        )
        if updated_at is not None:
            contribution.updated_at.CopyFrom(updated_at)
        return contribution


__all__ = ["AssemblyContributionRpcHandler"]
