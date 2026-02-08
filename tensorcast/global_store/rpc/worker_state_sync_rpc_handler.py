#  Copyright (c) 2025-2026, TensorCast Team.

"""High-availability worker state synchronization RPC handler."""

from __future__ import annotations

import grpc

from tensorcast.global_store.services.recovery_service import RecoveryService
from tensorcast.proto.global_store.v1 import global_store_pb2


class WorkerStateSyncRpcHandler:
    """Owns worker HA state sync/request-full-sync RPC behavior."""

    def __init__(self, *, recovery_service: RecoveryService, logger) -> None:
        self._recovery_service = recovery_service
        self._logger = logger

    def synchronize_worker_state(
        self,
        request: global_store_pb2.SynchronizeWorkerStateRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.SynchronizeWorkerStateResponse:
        """Synchronize worker state for high availability."""
        try:
            success, state_changes, new_version, new_checksum, ignored = (
                self._recovery_service.synchronize_worker_state(
                    request.worker_id,
                    request.local_state,
                    request.sync_epoch,
                    request.sync_request_id,
                    request.force_full_sync,
                )
            )

            if success:
                return global_store_pb2.SynchronizeWorkerStateResponse(
                    status=global_store_pb2.Status.STATUS_OK,
                    new_state_version=new_version,
                    state_changes=state_changes,
                    new_state_checksum=new_checksum,
                    ignored=ignored,
                )
            return global_store_pb2.SynchronizeWorkerStateResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

        except Exception as exc:  # noqa: BLE001
            self._logger.exception(
                "Error synchronizing worker state for %s",
                request.worker_id,
            )
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.SynchronizeWorkerStateResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def request_full_state_sync(
        self,
        request: global_store_pb2.RequestFullStateSyncRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.RequestFullStateSyncResponse:
        """Request full state synchronization for a worker."""
        try:
            success, expected_replicas, new_version, new_checksum, ignored = (
                self._recovery_service.request_full_state_sync(
                    request.worker_id,
                    request.sync_epoch,
                    request.sync_request_id,
                )
            )

            if success:
                return global_store_pb2.RequestFullStateSyncResponse(
                    status=global_store_pb2.Status.STATUS_OK,
                    new_state_version=new_version,
                    expected_replicas=expected_replicas,
                    new_state_checksum=new_checksum,
                    ignored=ignored,
                )
            return global_store_pb2.RequestFullStateSyncResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

        except Exception as exc:  # noqa: BLE001
            self._logger.exception(
                "Error requesting full state sync for %s",
                request.worker_id,
            )
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.RequestFullStateSyncResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
