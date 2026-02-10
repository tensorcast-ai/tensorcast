#  Copyright (c) 2025-2026, TensorCast Team.

"""Worker lifecycle RPC handler."""

from __future__ import annotations

import ipaddress
import time
from typing import Callable

import grpc
from google.protobuf import timestamp_pb2

from tensorcast.global_store import metrics as gs_metrics
from tensorcast.global_store.exceptions import ValidationError
from tensorcast.global_store.models import Worker
from tensorcast.global_store.repositories.worker_repository import WorkerRepository
from tensorcast.global_store.services.recovery_service import RecoveryService
from tensorcast.global_store.services.worker_service import WorkerService
from tensorcast.observability.otel import set_span_attributes
from tensorcast.proto.global_store.v1 import global_store_pb2


class WorkerRpcHandler:
    """Owns worker registration/heartbeat/listing RPC behavior."""

    def __init__(
        self,
        *,
        worker_service: WorkerService,
        worker_repository: WorkerRepository,
        recovery_service: RecoveryService,
        default_heartbeat_interval_ms: int,
        determine_worker_status: Callable[[Worker], global_store_pb2.ConnectionStatus],
        logger,
    ) -> None:
        self._worker_service = worker_service
        self._worker_repository = worker_repository
        self._recovery_service = recovery_service
        self._default_heartbeat_interval_ms = int(default_heartbeat_interval_ms)
        self._determine_worker_status = determine_worker_status
        self._logger = logger

    def register_worker(
        self,
        request: global_store_pb2.RegisterWorkerRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.RegisterWorkerResponse:
        """Register a new worker."""
        try:
            daemon_id = (request.daemon_id or "").strip()
            if not daemon_id:
                context.set_details("daemon_id is required")
                return global_store_pb2.RegisterWorkerResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )
            worker = Worker(
                daemon_id=daemon_id,
                node_id=request.node_id,
                node_address=request.node_address,
                grpc_port=request.grpc_port,
                p2p_port=request.p2p_port,
                mem_pool_total_size=request.mem_pool_total_size,
                mem_pool_available_size=request.mem_pool_available_size,
                capability_flags=int(request.capability_flags),
            )

            set_span_attributes(
                {
                    "tc.worker.daemon_id": str(worker.daemon_id),
                    "tc.worker.node_id": worker.node_id,
                    "tc.worker.node_address": worker.node_address,
                    "tc.worker.grpc_port": int(worker.grpc_port),
                    "tc.worker.p2p_port": int(worker.p2p_port),
                    "tc.mem_pool.total_bytes": int(worker.mem_pool_total_size),
                    "tc.mem_pool.available_bytes": int(worker.mem_pool_available_size),
                    "tc.worker.is_recovery": bool(request.is_recovery_registration),
                    "tc.worker.capability_flags": int(worker.capability_flags),
                }
            )

            is_recovery = request.is_recovery_registration
            previous_worker_id = (
                request.previous_worker_id if request.previous_worker_id else None
            )

            try:
                addr = ipaddress.ip_address(worker.node_address)
                if addr.is_loopback or addr.is_unspecified:
                    raise ValidationError(
                        f"Invalid node_address '{worker.node_address}'. Use a routable (non-loopback, non-unspecified) IP of the external interface; 127.0.0.1 and 0.0.0.0 are not allowed."
                    )
            except ValueError:
                pass

            if is_recovery:
                success, state_sync_required = (
                    self._recovery_service.handle_worker_recovery_registration(
                        worker, previous_worker_id
                    )
                )

                if not success:
                    return global_store_pb2.RegisterWorkerResponse(
                        status=global_store_pb2.Status.STATUS_ERROR
                    )

                registered = self._worker_service.find_worker_by_address(
                    worker.node_address, worker.grpc_port
                )

                if not registered:
                    return global_store_pb2.RegisterWorkerResponse(
                        status=global_store_pb2.Status.STATUS_ERROR
                    )

                self._logger.info(
                    "Worker registered: worker_id=%s daemon_id=%s node_id=%s addr=%s:%d p2p=%d mem_total=%d mem_avail=%d is_recovery=%s prev_worker_id=%s state_sync_required=%s expected_state_version=%d",
                    registered.worker_id,
                    (registered.daemon_id or ""),
                    worker.node_id,
                    worker.node_address,
                    int(worker.grpc_port),
                    int(worker.p2p_port),
                    int(worker.mem_pool_total_size),
                    int(worker.mem_pool_available_size),
                    True,
                    (previous_worker_id or ""),
                    bool(state_sync_required),
                    int(
                        self._recovery_service.ensure_worker_state_version(
                            registered.worker_id
                        )
                    ),
                )
                reconcile_generation, _ = self._worker_repository.get_reconcile_cursor(
                    registered.worker_id
                )

                return global_store_pb2.RegisterWorkerResponse(
                    status=global_store_pb2.Status.STATUS_OK,
                    worker_id=registered.worker_id,
                    heartbeat_interval_ms=self._default_heartbeat_interval_ms,
                    state_sync_required=state_sync_required,
                    expected_state_version=self._recovery_service.ensure_worker_state_version(
                        registered.worker_id
                    ),
                    reconcile_generation=int(reconcile_generation),
                )

            existing = self._worker_service.find_worker_by_address(
                worker.node_address, worker.grpc_port
            )
            if (
                existing
                and existing.node_id != worker.node_id
                and existing.inactive_at is None
            ):
                self._logger.error(
                    "Registration conflict: %s:%d already owned by worker_id=%s (node=%s); attempted by node=%s.",
                    worker.node_address,
                    worker.grpc_port,
                    existing.worker_id,
                    existing.node_id,
                    worker.node_id,
                )
                return global_store_pb2.RegisterWorkerResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )

            registered = self._worker_service.register_worker(worker)
            expected_state_version = self._recovery_service.ensure_worker_state_version(
                registered.worker_id
            )
            reconcile_generation, _ = self._worker_repository.get_reconcile_cursor(
                registered.worker_id
            )

            self._logger.info(
                "Worker registered: worker_id=%s daemon_id=%s node_id=%s addr=%s:%d p2p=%d mem_total=%d mem_avail=%d is_recovery=%s",
                registered.worker_id,
                (registered.daemon_id or ""),
                worker.node_id,
                worker.node_address,
                int(worker.grpc_port),
                int(worker.p2p_port),
                int(worker.mem_pool_total_size),
                int(worker.mem_pool_available_size),
                False,
            )

            return global_store_pb2.RegisterWorkerResponse(
                status=global_store_pb2.Status.STATUS_OK,
                worker_id=registered.worker_id,
                heartbeat_interval_ms=self._default_heartbeat_interval_ms,
                state_sync_required=False,
                expected_state_version=expected_state_version,
                reconcile_generation=int(reconcile_generation),
            )

        except ValidationError as exc:
            self._logger.error("Validation error: %s", exc)
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.RegisterWorkerResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error registering worker")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.RegisterWorkerResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def worker_heartbeat(
        self,
        request: global_store_pb2.WorkerHeartbeatRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.WorkerHeartbeatResponse:
        """Process worker heartbeat (enhanced-only)."""
        try:
            try:
                worker = self._worker_repository.find_by_id(request.worker_id)
                if worker and request.HasField("state_version"):
                    pass
            except Exception:
                self._logger.debug(
                    "worker lookup during heartbeat diagnostics failed",
                    exc_info=True,
                )
            set_span_attributes(
                {
                    "tc.worker.id": request.worker_id,
                    "tc.mem_pool.available_bytes": int(request.mem_pool_available_size),
                    "tc.worker.accepting_new_requests": bool(
                        request.accepting_new_requests
                    ),
                    "tc.worker.state_version": int(request.state_version),
                    "tc.worker.capability_flags": int(request.capability_flags),
                }
            )
            if request.state_version <= 0:
                self._logger.warning(
                    "Rejected legacy heartbeat for worker %s: state_version must be >= 1",
                    request.worker_id,
                )
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(
                    "state_version must be >= 1; legacy heartbeats are not supported"
                )
                return global_store_pb2.WorkerHeartbeatResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )
            return self._handle_enhanced_heartbeat(request, context)

        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error processing worker heartbeat")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.WorkerHeartbeatResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def _handle_enhanced_heartbeat(
        self,
        request: global_store_pb2.WorkerHeartbeatRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.WorkerHeartbeatResponse:
        """Handle enhanced heartbeat with state synchronization."""
        try:
            capability_flags: int | None = None
            if request.HasField("capability_flags"):
                capability_flags = int(request.capability_flags)
            success = self._worker_service.heartbeat(
                worker_id=request.worker_id,
                mem_pool_available_size=request.mem_pool_available_size,
                accepting_new_requests=request.accepting_new_requests,
                capability_flags=capability_flags,
            )

            if not success:
                return global_store_pb2.WorkerHeartbeatResponse(
                    status=global_store_pb2.Status.STATUS_NOT_FOUND
                )

            current_version = self._recovery_service.ensure_worker_state_version(
                request.worker_id
            )
            state_sync_required = request.state_version < current_version

            if request.state_checksum:
                server_checksum = self._recovery_service.get_worker_state_checksum(
                    request.worker_id
                )

                if request.state_checksum != server_checksum:
                    self._logger.debug(
                        "State checksum mismatch for worker %s: local=%s, global=%s",
                        request.worker_id,
                        request.state_checksum,
                        server_checksum,
                    )
                    state_sync_required = True

            obsolete_replicas: list[str] = []
            if request.registered_artifact_ids:
                obsolete_replicas = self._recovery_service.get_obsolete_artifacts(
                    request.worker_id,
                    list(request.registered_artifact_ids),
                )

            if obsolete_replicas:
                state_sync_required = True

            timestamp = timestamp_pb2.Timestamp()
            timestamp.FromSeconds(int(time.time()))
            return global_store_pb2.WorkerHeartbeatResponse(
                status=global_store_pb2.Status.STATUS_OK,
                state_sync_required=state_sync_required,
                expected_state_version=current_version,
                obsolete_replicas=obsolete_replicas,
                server_timestamp_ts=timestamp,
            )

        except Exception as exc:  # noqa: BLE001
            self._logger.exception(
                "Error handling enhanced heartbeat for worker %s",
                request.worker_id,
            )
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.WorkerHeartbeatResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def unregister_worker(
        self,
        request: global_store_pb2.UnregisterWorkerRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UnregisterWorkerResponse:
        """Unregister a worker."""
        try:
            worker_before = None
            try:
                worker_before = self._worker_repository.find_by_id(
                    request.worker_id, include_inactive=True
                )
            except Exception:
                worker_before = None

            success = self._worker_service.unregister_worker(request.worker_id)

            status = (
                global_store_pb2.Status.STATUS_OK
                if success
                else global_store_pb2.Status.STATUS_NOT_FOUND
            )
            if success:
                if worker_before:
                    self._logger.info(
                        "Worker unregistered: worker_id=%s graceful=%s node_id=%s addr=%s:%d p2p=%d",
                        request.worker_id,
                        getattr(request, "is_graceful_shutdown", False),
                        worker_before.node_id,
                        worker_before.node_address,
                        int(worker_before.grpc_port),
                        int(worker_before.p2p_port),
                    )
                else:
                    self._logger.info(
                        "Worker unregistered: worker_id=%s graceful=%s",
                        request.worker_id,
                        getattr(request, "is_graceful_shutdown", False),
                    )
            else:
                self._logger.warning(
                    "UnregisterWorker failed: worker %s not found",
                    request.worker_id,
                )

            return global_store_pb2.UnregisterWorkerResponse(status=status)

        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error unregistering worker")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.UnregisterWorkerResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def list_active_workers(
        self,
        request: global_store_pb2.ListActiveWorkersRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.ListActiveWorkersResponse:
        """List active workers."""
        try:
            workers = self._worker_service.list_active_workers(
                include_unavailable=request.include_unavailable
            )
            gs_metrics.set_capability_counts(scope="worker", entries=workers)
            required_flags = int(getattr(request, "required_capability_flags", 0))
            if required_flags:
                workers = [
                    worker
                    for worker in workers
                    if (int(worker.capability_flags) & required_flags) == required_flags
                ]

            worker_infos: list[
                global_store_pb2.ListActiveWorkersResponse.WorkerInfo
            ] = []
            for worker in workers:
                last_timestamp = timestamp_pb2.Timestamp()
                if worker.last_heartbeat:
                    last_timestamp.FromSeconds(int(worker.last_heartbeat.timestamp()))
                else:
                    last_timestamp.FromSeconds(0)

                worker_info = global_store_pb2.ListActiveWorkersResponse.WorkerInfo(
                    worker_id=worker.worker_id,
                    daemon_id=str(worker.daemon_id or ""),
                    node_id=worker.node_id,
                    node_address=worker.node_address,
                    grpc_port=worker.grpc_port,
                    p2p_port=worker.p2p_port,
                    mem_pool_total_size=worker.mem_pool_total_size,
                    mem_pool_available_size=worker.mem_pool_available_size,
                    accepting_new_requests=worker.accepting_new_requests,
                    capability_flags=int(worker.capability_flags),
                    last_heartbeat_ts=last_timestamp,
                    state_version=self._recovery_service.get_worker_state_version(
                        worker.worker_id
                    ),
                    status=self._determine_worker_status(worker),
                )
                worker_infos.append(worker_info)

            self._logger.info("Listed %d active workers", len(worker_infos))
            return global_store_pb2.ListActiveWorkersResponse(workers=worker_infos)

        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error listing active workers")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.ListActiveWorkersResponse()
