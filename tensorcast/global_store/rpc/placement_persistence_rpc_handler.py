#  Copyright (c) 2025-2026, TensorCast Team.

"""Placement and persistence RPC handler extracted from Global Store servicer."""

from __future__ import annotations

from typing import Callable

import grpc

from tensorcast.global_store.exceptions import ValidationError
from tensorcast.global_store.models import (
    PersistenceShardStatus,
    PersistenceStatus,
    PlacementPlan,
    PlacementShard,
    PlacementTarget,
)
from tensorcast.global_store.services.placement_service import PlacementService
from tensorcast.proto.global_store.v1 import global_store_pb2


class PlacementPersistenceRpcHandler:
    """Owns placement planning and persistence status gRPC behavior."""

    def __init__(
        self,
        *,
        placement_service: PlacementService,
        policy_from_proto: Callable[[int], str],
        persistence_state_from_proto: Callable[[int], str],
        target_state_from_proto: Callable[[int], str],
        plan_to_proto: Callable[
            [PlacementPlan], global_store_pb2.PlanPlacementResponse
        ],
        logger,
    ) -> None:
        self._placement_service = placement_service
        self._policy_from_proto = policy_from_proto
        self._persistence_state_from_proto = persistence_state_from_proto
        self._target_state_from_proto = target_state_from_proto
        self._plan_to_proto = plan_to_proto
        self._logger = logger

    def plan_placement(
        self,
        request: global_store_pb2.PlanPlacementRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.PlanPlacementResponse:
        if not request.artifact_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("artifact_id is required")
            return global_store_pb2.PlanPlacementResponse()
        if not request.source_node_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("source_node_id is required")
            return global_store_pb2.PlanPlacementResponse()
        try:
            policy = self._policy_from_proto(request.placement_policy)
        except ValidationError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.PlanPlacementResponse()

        shard_models: list[PlacementShard] = []
        for shard in request.shards:
            shard_id = shard.shard_id or f"{request.artifact_id}:{shard.shard_idx}"
            shard_models.append(
                PlacementShard(
                    plan_id="",
                    shard_idx=shard.shard_idx,
                    shard_id=shard_id,
                    size_bytes=shard.size_bytes,
                    content_digest=shard.content_digest,
                    byte_range_start=shard.byte_range_start,
                    byte_range_length=shard.byte_range_length,
                    chunk_ids=list(shard.chunk_ids),
                )
            )

        try:
            plan = self._placement_service.plan_placement(
                request.artifact_id,
                policy,
                shard_models,
                source_node_id=request.source_node_id,
            )
        except ValidationError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.PlanPlacementResponse()
        except Exception:  # noqa: BLE001
            self._logger.exception("Failed to plan placement")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Failed to plan placement")
            return global_store_pb2.PlanPlacementResponse()

        return self._plan_to_proto(plan)

    def report_persistence_status(
        self,
        request: global_store_pb2.ReportPersistenceStatusRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.ReportPersistenceStatusResponse:
        if not request.task_id or not request.plan_id or not request.artifact_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("task_id, plan_id, and artifact_id are required")
            return global_store_pb2.ReportPersistenceStatusResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        try:
            state = self._persistence_state_from_proto(request.state)
        except ValidationError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.ReportPersistenceStatusResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

        shard_statuses: list[PersistenceShardStatus] = []
        try:
            for shard in request.shard_statuses:
                shard_state = self._persistence_state_from_proto(shard.state)
                targets = [
                    PlacementTarget(
                        plan_id=request.plan_id,
                        shard_idx=shard.shard_idx,
                        node_id=target.node_id,
                        lease_id=target.lease_id or None,
                        target_state=self._target_state_from_proto(target.target_state),
                        degraded_reason=target.degraded_reason or None,
                    )
                    for target in shard.targets
                ]
                shard_statuses.append(
                    PersistenceShardStatus(
                        shard_id=shard.shard_id,
                        shard_idx=shard.shard_idx,
                        state=shard_state,
                        progress=shard.progress,
                        degraded_reason=shard.degraded_reason or None,
                        last_error=shard.last_error or None,
                        targets=targets,
                    )
                )
        except ValidationError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.ReportPersistenceStatusResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except Exception:  # noqa: BLE001
            self._logger.exception("Failed to decode shard status payload")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Failed to decode shard status payload")
            return global_store_pb2.ReportPersistenceStatusResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

        try:
            status_model = PersistenceStatus(
                task_id=request.task_id,
                plan_id=request.plan_id,
                artifact_id=request.artifact_id,
                state=state,
                progress=request.progress,
                last_error=request.last_error or None,
                degraded_reason=request.degraded_reason or None,
            )
            self._placement_service.record_status(status_model, shard_statuses)
        except Exception:  # noqa: BLE001
            self._logger.exception("Failed to persist ReportPersistenceStatus")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Failed to persist ReportPersistenceStatus")
            return global_store_pb2.ReportPersistenceStatusResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        return global_store_pb2.ReportPersistenceStatusResponse(
            status=global_store_pb2.Status.STATUS_OK
        )
