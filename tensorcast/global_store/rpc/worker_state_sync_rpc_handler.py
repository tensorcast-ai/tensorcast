#  Copyright (c) 2025-2026, TensorCast Team.

"""High-availability worker state synchronization RPC handler."""

from __future__ import annotations

import grpc

from tensorcast.global_store.metrics import inc_reconcile_result
from tensorcast.global_store.services.recovery_service import RecoveryService
from tensorcast.proto.global_store.v1 import global_store_pb2


class WorkerStateSyncRpcHandler:
    """Owns worker HA reconcile RPC behavior."""

    def __init__(self, *, recovery_service: RecoveryService, logger) -> None:
        self._recovery_service = recovery_service
        self._logger = logger

    def reconcile_worker_state(
        self,
        request: global_store_pb2.ReconcileWorkerStateRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.ReconcileWorkerStateResponse:
        """Reconcile worker state for high availability."""
        try:
            (
                result_kind,
                new_version,
                new_checksum,
                state_changes,
                expected_replicas,
                retry_after_ms,
            ) = self._recovery_service.reconcile_worker_state(
                worker_id=request.worker_id,
                daemon_id=request.daemon_id,
                generation=request.generation,
                request_seq=request.request_seq,
                inventory=list(request.inventory),
                request_kind=request.request_kind,
            )
            return global_store_pb2.ReconcileWorkerStateResponse(
                result_kind=result_kind,
                retry_after_ms=retry_after_ms,
                new_state_version=new_version,
                new_state_checksum=new_checksum,
                state_changes=state_changes,
                expected_replicas=expected_replicas,
            )
        except ValueError as exc:
            self._logger.warning(
                "Reconcile request rejected for worker %s: %s",
                request.worker_id,
                exc,
            )
            inc_reconcile_result(
                result_kind=global_store_pb2.ReconcileResultKind.Name(
                    global_store_pb2.RECONCILE_RESULT_KIND_FATAL
                )
            )
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return global_store_pb2.ReconcileWorkerStateResponse(
                result_kind=global_store_pb2.RECONCILE_RESULT_KIND_FATAL
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception(
                "Error reconciling worker state for %s",
                request.worker_id,
            )
            inc_reconcile_result(
                result_kind=global_store_pb2.ReconcileResultKind.Name(
                    global_store_pb2.RECONCILE_RESULT_KIND_FATAL
                )
            )
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.ReconcileWorkerStateResponse(
                result_kind=global_store_pb2.RECONCILE_RESULT_KIND_FATAL
            )
