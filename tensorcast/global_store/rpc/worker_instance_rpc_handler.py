#  Copyright (c) 2025-2026, TensorCast Team.

"""Compatibility wrapper for worker/instance/HA RPC handlers."""

from __future__ import annotations

from typing import Callable

import grpc

from tensorcast.global_store.models import Instance, Worker
from tensorcast.global_store.repositories.worker_repository import WorkerRepository
from tensorcast.global_store.rpc.instance_rpc_handler import InstanceRpcHandler
from tensorcast.global_store.rpc.worker_rpc_handler import WorkerRpcHandler
from tensorcast.global_store.rpc.worker_state_sync_rpc_handler import (
    WorkerStateSyncRpcHandler,
)
from tensorcast.global_store.services.instance_service import InstanceService
from tensorcast.global_store.services.recovery_service import RecoveryService
from tensorcast.global_store.services.worker_service import WorkerService
from tensorcast.proto.global_store.v1 import global_store_pb2


class WorkerInstanceRpcHandler:
    """Backward-compatible facade around split worker/instance/sync handlers."""

    def __init__(
        self,
        *,
        worker_service: WorkerService,
        worker_repository: WorkerRepository,
        recovery_service: RecoveryService,
        instance_service: InstanceService,
        default_heartbeat_interval_ms: int,
        determine_worker_status: Callable[[Worker], int],
        determine_instance_status: Callable[[Instance], int],
        logger,
    ) -> None:
        # Keep legacy attributes for tests and callers that introspect internals.
        self._worker_service = worker_service
        self._recovery_service = recovery_service

        self._worker_rpc_handler = WorkerRpcHandler(
            worker_service=worker_service,
            worker_repository=worker_repository,
            recovery_service=recovery_service,
            default_heartbeat_interval_ms=default_heartbeat_interval_ms,
            determine_worker_status=determine_worker_status,
            logger=logger,
        )
        self._instance_rpc_handler = InstanceRpcHandler(
            instance_service=instance_service,
            default_heartbeat_interval_ms=default_heartbeat_interval_ms,
            determine_instance_status=determine_instance_status,
            logger=logger,
        )
        self._worker_state_sync_rpc_handler = WorkerStateSyncRpcHandler(
            recovery_service=recovery_service,
            logger=logger,
        )

    def register_worker(
        self,
        request: global_store_pb2.RegisterWorkerRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.RegisterWorkerResponse:
        return self._worker_rpc_handler.register_worker(request, context)

    def worker_heartbeat(
        self,
        request: global_store_pb2.WorkerHeartbeatRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.WorkerHeartbeatResponse:
        return self._worker_rpc_handler.worker_heartbeat(request, context)

    def unregister_worker(
        self,
        request: global_store_pb2.UnregisterWorkerRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UnregisterWorkerResponse:
        return self._worker_rpc_handler.unregister_worker(request, context)

    def list_active_workers(
        self,
        request: global_store_pb2.ListActiveWorkersRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.ListActiveWorkersResponse:
        return self._worker_rpc_handler.list_active_workers(request, context)

    def register_instance(
        self,
        request: global_store_pb2.RegisterInstanceRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.RegisterInstanceResponse:
        return self._instance_rpc_handler.register_instance(request, context)

    def instance_heartbeat(
        self,
        request: global_store_pb2.InstanceHeartbeatRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.InstanceHeartbeatResponse:
        return self._instance_rpc_handler.instance_heartbeat(request, context)

    def unregister_instance(
        self,
        request: global_store_pb2.UnregisterInstanceRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UnregisterInstanceResponse:
        return self._instance_rpc_handler.unregister_instance(request, context)

    def list_active_instances(
        self,
        request: global_store_pb2.ListActiveInstancesRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.ListActiveInstancesResponse:
        return self._instance_rpc_handler.list_active_instances(request, context)

    def synchronize_worker_state(
        self,
        request: global_store_pb2.SynchronizeWorkerStateRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.SynchronizeWorkerStateResponse:
        return self._worker_state_sync_rpc_handler.synchronize_worker_state(
            request, context
        )

    def request_full_state_sync(
        self,
        request: global_store_pb2.RequestFullStateSyncRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.RequestFullStateSyncResponse:
        return self._worker_state_sync_rpc_handler.request_full_state_sync(
            request, context
        )
