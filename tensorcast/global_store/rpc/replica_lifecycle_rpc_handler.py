#  Copyright (c) 2025-2026, TensorCast Team.

"""Replica lifecycle RPC handler extracted from Global Store gRPC servicer."""

from __future__ import annotations

import time
from typing import Callable
from uuid import UUID

import grpc

from tensorcast.global_store.models import MemoryType, Replica
from tensorcast.global_store.repositories.replica_repository import ReplicaRepository
from tensorcast.global_store.repositories.transport_repository import (
    TransportRepository,
)
from tensorcast.global_store.services.artifact_service import ArtifactService
from tensorcast.global_store.services.transport_service import TransportService
from tensorcast.observability.otel import set_span_attributes
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.global_store.v1 import global_store_pb2


class ReplicaLifecycleRpcHandler:
    """Owns replica lifecycle gRPC behavior and error mapping."""

    def __init__(
        self,
        *,
        artifact_service: ArtifactService,
        replica_repository: ReplicaRepository,
        transport_repository: TransportRepository,
        transport_service: TransportService,
        replica_to_memory_info: Callable[[Replica], common_pb2.MemoryInfo],
        logger,
    ) -> None:
        self._artifact_service = artifact_service
        self._replica_repository = replica_repository
        self._transport_repository = transport_repository
        self._transport_service = transport_service
        self._replica_to_memory_info = replica_to_memory_info
        self._logger = logger

    def update_replica(
        self,
        request: global_store_pb2.UpdateReplicaRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UpdateReplicaResponse:
        """Update artifact replica heartbeat."""
        try:
            replica_id = UUID(request.replica_id)
            artifact_id = request.artifact_id

            set_span_attributes(
                {
                    "tc.artifact.id": artifact_id,
                    "tc.replica.id": str(replica_id),
                }
            )

            success = self._artifact_service.update_heartbeat(replica_id, artifact_id)

            status = (
                global_store_pb2.Status.STATUS_OK
                if success
                else global_store_pb2.Status.STATUS_NOT_FOUND
            )

            return global_store_pb2.UpdateReplicaResponse(
                status=status,
                artifact_id=artifact_id,
                replica_id=request.replica_id,
            )

        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error updating artifact replica")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.UpdateReplicaResponse(
                status=global_store_pb2.Status.STATUS_ERROR,
                artifact_id=request.artifact_id,
                replica_id=request.replica_id,
            )

    def unregister_replica(
        self,
        request: global_store_pb2.UnregisterReplicaRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UnregisterReplicaResponse:
        """Unregister an artifact replica."""
        try:
            replica_id = UUID(request.replica_id)
            artifact_id = request.artifact_id

            success = self._artifact_service.unregister_replica(replica_id, artifact_id)
            status = (
                global_store_pb2.Status.STATUS_OK
                if success
                else global_store_pb2.Status.STATUS_NOT_FOUND
            )

            return global_store_pb2.UnregisterReplicaResponse(status=status)

        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error unregistering artifact replica")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.UnregisterReplicaResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def unregister_replica_by_worker(
        self,
        request: global_store_pb2.UnregisterReplicaByWorkerRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UnregisterReplicaByWorkerResponse:
        """Unregister a replica by (artifact_id, worker_id[, device_id, memory_type])."""
        try:
            artifact_id = request.artifact_id
            worker_id = request.worker_id
            if not artifact_id or not worker_id:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("artifact_id and worker_id are required")
                return global_store_pb2.UnregisterReplicaByWorkerResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )

            memory_type = None
            if request.HasField("memory_type"):
                mt = request.memory_type
                memory_type = (
                    MemoryType.GPU
                    if mt == common_pb2.MEMORY_TYPE_GPU
                    else MemoryType.RAM
                    if mt == common_pb2.MEMORY_TYPE_RAM
                    else MemoryType.DISK
                )

            device_id = request.device_id if request.HasField("device_id") else None

            success = self._artifact_service.unregister_by_worker(
                worker_id=worker_id,
                artifact_id=artifact_id,
                memory_type=memory_type,
                device_id=device_id,
            )

            status = (
                global_store_pb2.Status.STATUS_OK
                if success
                else global_store_pb2.Status.STATUS_NOT_FOUND
            )
            return global_store_pb2.UnregisterReplicaByWorkerResponse(status=status)

        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error in UnregisterReplicaByWorker")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.UnregisterReplicaByWorkerResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def mark_replica_unavailable(
        self,
        request: global_store_pb2.MarkReplicaUnavailableRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.MarkReplicaUnavailableResponse:
        try:
            if not request.replica_id:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("replica_id is required")
                return global_store_pb2.MarkReplicaUnavailableResponse(
                    status=global_store_pb2.Status.STATUS_ERROR,
                    updated=False,
                )
            replica_id = UUID(request.replica_id)
            replica = self._replica_repository.find_by_replica_id(replica_id)
            if replica is None or (
                request.artifact_id and replica.artifact_id != request.artifact_id
            ):
                return global_store_pb2.MarkReplicaUnavailableResponse(
                    status=global_store_pb2.Status.STATUS_NOT_FOUND,
                    updated=False,
                )

            set_span_attributes(
                {
                    "tc.artifact.id": replica.artifact_id,
                    "tc.replica.id": str(replica_id),
                }
            )

            updated = self._replica_repository.mark_unavailable(replica_id)
            return global_store_pb2.MarkReplicaUnavailableResponse(
                status=global_store_pb2.Status.STATUS_OK,
                updated=updated,
            )
        except ValueError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.MarkReplicaUnavailableResponse(
                status=global_store_pb2.Status.STATUS_ERROR,
                updated=False,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error marking replica unavailable")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.MarkReplicaUnavailableResponse(
                status=global_store_pb2.Status.STATUS_ERROR,
                updated=False,
            )

    def wait_replica_drain(
        self,
        request: global_store_pb2.WaitReplicaDrainRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.WaitReplicaDrainResponse:
        if not request.replica_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("replica_id is required")
            return global_store_pb2.WaitReplicaDrainResponse(
                status=global_store_pb2.Status.STATUS_ERROR,
                drained=False,
                current_requests=0,
            )
        try:
            replica_id = UUID(request.replica_id)
        except ValueError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.WaitReplicaDrainResponse(
                status=global_store_pb2.Status.STATUS_ERROR,
                drained=False,
                current_requests=0,
            )

        timeout_ms = int(request.timeout_ms or 0)
        interval = (
            self._transport_service.config.transport_wait_retry_interval_ms / 1000.0
        )

        current = self._replica_repository.get_current_requests(replica_id)
        if current is None:
            return global_store_pb2.WaitReplicaDrainResponse(
                status=global_store_pb2.Status.STATUS_NOT_FOUND,
                drained=False,
                current_requests=0,
            )

        drained = current == 0
        if not drained and timeout_ms > 0:
            now = time.monotonic()
            deadline = now + (timeout_ms / 1000.0)
            context_remaining = context.time_remaining()
            if context_remaining is not None:
                deadline = min(deadline, now + max(context_remaining, 0.0))

            while not drained:
                now = time.monotonic()
                remaining = deadline - now
                if remaining <= 0:
                    break
                time.sleep(min(interval, remaining))
                current = self._replica_repository.get_current_requests(replica_id)
                if current is None:
                    return global_store_pb2.WaitReplicaDrainResponse(
                        status=global_store_pb2.Status.STATUS_NOT_FOUND,
                        drained=False,
                        current_requests=0,
                    )
                drained = current == 0

        oldest_age_ms = self._transport_repository.get_oldest_in_progress_age_ms(
            replica_id
        )
        response = global_store_pb2.WaitReplicaDrainResponse(
            status=global_store_pb2.Status.STATUS_OK
            if drained
            else global_store_pb2.Status.STATUS_TIMED_OUT,
            drained=drained,
            current_requests=int(current or 0),
        )
        if oldest_age_ms is not None:
            response.oldest_transport_age_ms = int(oldest_age_ms)
        return response

    def list_replicas_v2(
        self,
        request: global_store_pb2.ListReplicasV2Request,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.ListReplicasV2Response:
        """List replicas with filtering + pagination (flat records)."""
        try:
            artifact_id_filter: str | None = (
                request.artifact_id if request.HasField("artifact_id") else None
            )
            node_id_filter: str | None = (
                request.node_id if request.HasField("node_id") else None
            )
            memory_type_filter = None
            if request.HasField("memory_type"):
                mt = request.memory_type
                if mt == common_pb2.MemoryType.MEMORY_TYPE_GPU:
                    memory_type_filter = MemoryType.GPU
                elif mt == common_pb2.MemoryType.MEMORY_TYPE_RAM:
                    memory_type_filter = MemoryType.RAM
                elif mt == common_pb2.MemoryType.MEMORY_TYPE_DISK:
                    memory_type_filter = MemoryType.DISK

            replicas = self._artifact_service.list_replicas(
                artifact_id=artifact_id_filter,
                node_id=node_id_filter,
                memory_type=memory_type_filter,
            )

            page_size = (
                int(request.pagination.page_size)
                if request.pagination and request.pagination.page_size
                else 100
            )
            start = 0
            if request.pagination and request.pagination.page_token:
                try:
                    start = int(request.pagination.page_token)
                except ValueError:
                    start = 0

            end = min(start + page_size, len(replicas))
            sliced = replicas[start:end]
            next_token = str(end) if end < len(replicas) else ""

            records: list[global_store_pb2.ArtifactReplicaRecord] = [
                global_store_pb2.ArtifactReplicaRecord(
                    artifact_id=replica.artifact_id,
                    memory_info=self._replica_to_memory_info(replica),
                )
                for replica in sliced
            ]

            return global_store_pb2.ListReplicasV2Response(
                replicas=records,
                page_info=common_pb2.PageInfo(
                    next_page_token=next_token,
                    total_size=len(replicas),
                ),
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error in ListReplicasV2")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.ListReplicasV2Response(
                page_info=common_pb2.PageInfo(next_page_token="", total_size=0)
            )
