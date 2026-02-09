#  Copyright (c) 2025-2026, TensorCast Team.

"""Assembly runtime policy RPC handler."""

from __future__ import annotations

from datetime import datetime
from typing import Callable

import grpc
from google.protobuf import timestamp_pb2

from tensorcast.global_store.exceptions import DatabaseError
from tensorcast.global_store.repositories.assembly_runtime_policy_repository import (
    AssemblyRuntimePolicyRepository,
)
from tensorcast.proto.global_store.v1 import global_store_pb2


class LayoutRuntimePolicyRpcHandler:
    """Owns assembly runtime policy get/update behavior."""

    def __init__(
        self,
        *,
        assembly_runtime_policy_repository: AssemblyRuntimePolicyRepository,
        datetime_to_timestamp: Callable[
            [datetime | None], timestamp_pb2.Timestamp | None
        ],
        coerce_db_datetime: Callable[[object], datetime | None],
        logger,
    ) -> None:
        self._assembly_runtime_policy_repository = assembly_runtime_policy_repository
        self._datetime_to_timestamp = datetime_to_timestamp
        self._coerce_db_datetime = coerce_db_datetime
        self._logger = logger

    def get_assembly_runtime_policy(
        self,
        request: global_store_pb2.GetAssemblyRuntimePolicyRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.GetAssemblyRuntimePolicyResponse:
        assembly_id = request.assembly_id
        if not assembly_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("assembly_id is required")
            return global_store_pb2.GetAssemblyRuntimePolicyResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        try:
            row = self._assembly_runtime_policy_repository.get(assembly_id=assembly_id)
            if row is None:
                return global_store_pb2.GetAssemblyRuntimePolicyResponse(
                    status=global_store_pb2.Status.STATUS_NOT_FOUND
                )
            policy = global_store_pb2.AssemblyRuntimePolicy(
                assembly_id=str(row["assembly_id"]),
                policy_version=int(row["policy_version"]),
                policy_json=str(row["policy_json"]),
            )
            ts = self._datetime_to_timestamp(
                self._coerce_db_datetime(row.get("updated_at"))
            )
            if ts is not None:
                policy.updated_at.CopyFrom(ts)
            return global_store_pb2.GetAssemblyRuntimePolicyResponse(
                status=global_store_pb2.Status.STATUS_OK,
                policy=policy,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("GetAssemblyRuntimePolicy failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.GetAssemblyRuntimePolicyResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def update_assembly_runtime_policy(
        self,
        request: global_store_pb2.UpdateAssemblyRuntimePolicyRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UpdateAssemblyRuntimePolicyResponse:
        assembly_id = request.assembly_id
        if not assembly_id or not request.policy_json:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("assembly_id and policy_json are required")
            return global_store_pb2.UpdateAssemblyRuntimePolicyResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        try:
            expected = int(request.expected_policy_version)
            with self._assembly_runtime_policy_repository.transaction() as cursor:
                row = self._assembly_runtime_policy_repository.update(
                    assembly_id=assembly_id,
                    policy_json=str(request.policy_json),
                    expected_policy_version=expected,
                    cursor=cursor,
                )
            policy = global_store_pb2.AssemblyRuntimePolicy(
                assembly_id=str(row["assembly_id"]),
                policy_version=int(row["policy_version"]),
                policy_json=str(row["policy_json"]),
            )
            ts = self._datetime_to_timestamp(
                self._coerce_db_datetime(row.get("updated_at"))
            )
            if ts is not None:
                policy.updated_at.CopyFrom(ts)
            return global_store_pb2.UpdateAssemblyRuntimePolicyResponse(
                status=global_store_pb2.Status.STATUS_OK,
                policy=policy,
            )
        except (ValueError, DatabaseError) as exc:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details(str(exc))
            return global_store_pb2.UpdateAssemblyRuntimePolicyResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("UpdateAssemblyRuntimePolicy failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.UpdateAssemblyRuntimePolicyResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
