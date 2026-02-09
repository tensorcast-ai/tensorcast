#  Copyright (c) 2025-2026, TensorCast Team.

"""Chunk directory RPC handler extracted from Global Store gRPC servicer."""

from __future__ import annotations

import grpc

from tensorcast.global_store.services.chunk_service import ChunkService
from tensorcast.proto.global_store.v1 import global_store_pb2


class ChunkRpcHandler:
    """Owns chunk directory gRPC behavior and error mapping."""

    def __init__(self, *, chunk_service: ChunkService, logger) -> None:
        self._chunk_service = chunk_service
        self._logger = logger

    def query_chunk_locations(
        self,
        request: global_store_pb2.QueryChunkLocationsRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.QueryChunkLocationsResponse:
        """Query chunk locations for distributed memory pool."""
        try:
            chunk_indices = (
                list(request.chunk_indices) if request.chunk_indices else None
            )
            locations = self._chunk_service.query_chunk_locations(
                request.artifact_id,
                chunk_indices,
            )

            return global_store_pb2.QueryChunkLocationsResponse(
                status=global_store_pb2.Status.STATUS_OK,
                locations=locations,
            )

        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error querying chunk locations: %s", exc)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.QueryChunkLocationsResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def batch_update_chunk_states(
        self,
        request: global_store_pb2.BatchUpdateChunkStatesRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.BatchUpdateChunkStatesResponse:
        """Batch update chunk states from StoreDaemon."""
        try:
            updates_list = list(request.updates)
            updates_applied = self._chunk_service.batch_update_chunk_states(
                request.worker_id,
                request.node_id,
                updates_list,
            )

            return global_store_pb2.BatchUpdateChunkStatesResponse(
                status=global_store_pb2.Status.STATUS_OK,
                updates_applied=updates_applied,
            )

        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error updating chunk states: %s", exc)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.BatchUpdateChunkStatesResponse(
                status=global_store_pb2.Status.STATUS_ERROR,
                updates_applied=0,
            )
