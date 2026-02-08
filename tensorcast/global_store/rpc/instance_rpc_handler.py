#  Copyright (c) 2025-2026, TensorCast Team.

"""Instance lifecycle RPC handler."""

from __future__ import annotations

from typing import Callable

import grpc
from google.protobuf import timestamp_pb2

from tensorcast.global_store import metrics as gs_metrics
from tensorcast.global_store.exceptions import ValidationError
from tensorcast.global_store.models import Instance
from tensorcast.global_store.services.instance_service import InstanceService
from tensorcast.proto.global_store.v1 import global_store_pb2


class InstanceRpcHandler:
    """Owns instance registration/heartbeat/listing RPC behavior."""

    def __init__(
        self,
        *,
        instance_service: InstanceService,
        default_heartbeat_interval_ms: int,
        determine_instance_status: Callable[[Instance], int],
        logger,
    ) -> None:
        self._instance_service = instance_service
        self._default_heartbeat_interval_ms = int(default_heartbeat_interval_ms)
        self._determine_instance_status = determine_instance_status
        self._logger = logger

    def register_instance(
        self,
        request: global_store_pb2.RegisterInstanceRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.RegisterInstanceResponse:
        """Register a new engine instance."""
        try:
            instance = Instance(
                instance_id=request.instance_id,
                daemon_id=request.daemon_id,
                worker_id=request.worker_id if request.HasField("worker_id") else None,
                engine=request.engine,
                signals_endpoint=request.signals_endpoint or None,
                labels=dict(request.labels) if request.labels else {},
                capability_flags=int(request.capability_flags),
            )
            registered = self._instance_service.register_instance(instance)
            return global_store_pb2.RegisterInstanceResponse(
                status=global_store_pb2.Status.STATUS_OK,
                instance_id=registered.instance_id,
                heartbeat_interval_ms=self._default_heartbeat_interval_ms,
            )
        except ValidationError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.RegisterInstanceResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error registering instance")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.RegisterInstanceResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def instance_heartbeat(
        self,
        request: global_store_pb2.InstanceHeartbeatRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.InstanceHeartbeatResponse:
        """Process instance heartbeat."""
        try:
            worker_id = request.worker_id if request.HasField("worker_id") else None
            capability_flags: int | None = None
            if request.HasField("capability_flags"):
                capability_flags = int(request.capability_flags)
            success = self._instance_service.heartbeat(
                request.instance_id,
                worker_id=worker_id,
                capability_flags=capability_flags,
            )
            return global_store_pb2.InstanceHeartbeatResponse(
                status=global_store_pb2.Status.STATUS_OK
                if success
                else global_store_pb2.Status.STATUS_NOT_FOUND
            )
        except ValidationError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.InstanceHeartbeatResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error processing instance heartbeat")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.InstanceHeartbeatResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def unregister_instance(
        self,
        request: global_store_pb2.UnregisterInstanceRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UnregisterInstanceResponse:
        """Unregister an instance."""
        try:
            success = self._instance_service.unregister_instance(request.instance_id)
            return global_store_pb2.UnregisterInstanceResponse(
                status=global_store_pb2.Status.STATUS_OK
                if success
                else global_store_pb2.Status.STATUS_NOT_FOUND
            )
        except ValidationError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.UnregisterInstanceResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error unregistering instance")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.UnregisterInstanceResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def list_active_instances(
        self,
        request: global_store_pb2.ListActiveInstancesRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.ListActiveInstancesResponse:
        """List active instances."""
        try:
            instances = self._instance_service.list_active_instances(
                include_unavailable=request.include_unavailable
            )
            gs_metrics.set_capability_counts(scope="instance", entries=instances)
            required_flags = int(getattr(request, "required_capability_flags", 0))
            if required_flags:
                instances = [
                    inst
                    for inst in instances
                    if (int(inst.capability_flags) & required_flags) == required_flags
                ]
            infos: list[global_store_pb2.ListActiveInstancesResponse.InstanceInfo] = []
            for instance in instances:
                last_timestamp = timestamp_pb2.Timestamp()
                if instance.last_heartbeat:
                    last_timestamp.FromSeconds(int(instance.last_heartbeat.timestamp()))
                else:
                    last_timestamp.FromSeconds(0)
                infos.append(
                    global_store_pb2.ListActiveInstancesResponse.InstanceInfo(
                        instance_id=instance.instance_id,
                        daemon_id=instance.daemon_id,
                        worker_id=instance.worker_id or "",
                        engine=instance.engine,
                        signals_endpoint=instance.signals_endpoint or "",
                        last_heartbeat_ts=last_timestamp,
                        labels=dict(instance.labels) if instance.labels else {},
                        capability_flags=int(instance.capability_flags),
                        status=self._determine_instance_status(instance),
                    )
                )
            return global_store_pb2.ListActiveInstancesResponse(instances=infos)
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error listing active instances")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.ListActiveInstancesResponse()
