#  Copyright (c) 2025-2026, TensorCast Team.
"""Client utilities for interacting with the TensorCast Store Daemon."""

from __future__ import annotations

import atexit
import os
import random
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import timezone
from threading import RLock
from typing import (
    TYPE_CHECKING,
    Any,
    Iterator,
    Literal,
    NoReturn,
    Sequence,
    cast,
    overload,
)

import grpc
from opentelemetry import trace
from opentelemetry.trace import SpanKind

if TYPE_CHECKING:
    from tensorcast.api._config import StorePolicy
    from tensorcast.proto.operation.v1 import operation_pb2
from tensorcast.error_reporting import debug_errors_enabled
from tensorcast.logger import init_logger
from tensorcast.observability.otel import ensure_client_otel, set_span_attributes

# Use v2 daemon proto path
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.proto.daemon.v2 import (
    store_daemon_pb2_grpc as store_daemon_pb2_grpc,
)
from tensorcast.types import (
    ArtifactDescriptor,
    ArtifactIdKind,
    BeginRegisterArtifactResult,
    CanonicalRange,
    CoalescedHandshake,
    CommitResult,
    DeregisterArtifactOutcome,
    LeaseHandshake,
    LeaseSegment,
    LocalStableTierResult,
    Plan,
    RegisterStorage,
    RegisterTensorAlias,
    SealAssemblyResult,
    ServerConfig,
    StableDramHandshake,
    VramRegionHandle,
)

logger = init_logger(__name__)

# -----------------------------------------------------------------------------
# Client-side diagnostics (process-scoped)
# -----------------------------------------------------------------------------
_METRICS_LOCK: RLock = RLock()
_METRIC_CHANNEL_REFRESHES: int = 0
_METRIC_RPC_RETRIES: int = 0


@dataclass(frozen=True, slots=True)
class KeyMappingResolution:
    artifact_id: str
    used_disk_path: str
    generation: int
    cache_ttl_seconds: int


@dataclass(frozen=True, slots=True)
class SwapKeyMappingResult:
    ok: bool
    artifact_id: str
    generation: int


def _raise_grpc_error(err: Exception, *, cause: BaseException | None) -> NoReturn:
    if debug_errors_enabled() and cause is not None:
        raise err from cause
    raise err from None


def _grpc_details(err: grpc.RpcError) -> str:
    with suppress(Exception):
        details = err.details() or ""
        if details:
            return str(details)
    return ""


def _grpc_message(err: grpc.RpcError, *, fallback: str) -> str:
    details = _grpc_details(err)
    return details if details else fallback


def _inc_channel_refresh(server_address: str) -> int:
    global _METRIC_CHANNEL_REFRESHES
    with _METRICS_LOCK:
        _METRIC_CHANNEL_REFRESHES += 1
        cur = _METRIC_CHANNEL_REFRESHES
    logger.info(
        "client_channel_refresh addr=%s pid=%s total=%d",
        server_address,
        os.getpid(),
        cur,
    )
    return cur


def _inc_rpc_retry(
    server_address: str, method_name: str, attempt: int, code: "grpc.StatusCode"
) -> int:
    global _METRIC_RPC_RETRIES
    with _METRICS_LOCK:
        _METRIC_RPC_RETRIES += 1
        cur = _METRIC_RPC_RETRIES
    try:
        code_name = code.name
    except Exception:
        code_name = str(code)
    logger.info(
        "client_rpc_retry method=%s attempt=%d code=%s addr=%s pid=%s total_retries=%d",
        method_name,
        int(attempt),
        code_name,
        server_address,
        os.getpid(),
        cur,
    )
    return cur


def get_host_pid() -> int:
    """Get the host PID of the current process.

    In containerized environments, the PID inside the container may differ
    from the PID on the host. This function attempts to get the host PID
    by reading the NSpid field from /proc/self/status.

    Returns:
        The host PID if available, otherwise falls back to os.getpid()
    """
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("NSpid:"):
                    # NSpid shows PIDs in different namespaces
                    # The last value is typically the host PID
                    pids = line.strip().split()[1:]
                    if pids:
                        return int(pids[-1])
    except (IOError, OSError, ValueError) as e:
        logger.warning(f"Failed to get host PID: {e}, falling back to os.getpid()")

    # Fallback to regular PID if host PID cannot be determined
    return os.getpid()


# This is a singleton class that manages the checkpoint
class DaemonCtl:
    _PERSISTENCE_STATE_FROM_PROTO = {
        store_daemon_pb2.PERSISTENCE_STATE_PENDING: "pending",
        store_daemon_pb2.PERSISTENCE_STATE_RUNNING: "running",
        store_daemon_pb2.PERSISTENCE_STATE_DEGRADED: "degraded",
        store_daemon_pb2.PERSISTENCE_STATE_SUCCESS: "success",
        store_daemon_pb2.PERSISTENCE_STATE_FAILED: "failed",
    }

    def __init__(self, server_address="127.0.0.1:8073"):
        # SDK-library safe: ensure OTel is active or ask app to init. No downgrade.
        ensure_client_otel("tensorcast-client", role="client")

        self.server_address = server_address
        self._ch_lock: RLock = RLock()
        self.channel = self._create_channel(server_address)
        self.stub = store_daemon_pb2_grpc.StoreDaemonServiceStub(self.channel)
        self.stub_v2 = store_daemon_pb2_grpc.StoreDaemonServiceStub(self.channel)
        self.checkpoints_in_gpu = {}

        # Check environment variable
        env_use_host_pid = os.environ.get("TENSORCAST_USE_HOST_PID", "").lower()
        self.use_host_pid = env_use_host_pid in ("true", "1", "yes")

        if self.use_host_pid:
            logger.info("DaemonCtl configured to use host PID")

    # Channel helpers with sane keepalive defaults
    @staticmethod
    def _channel_options() -> list[tuple[str, int]]:
        return [
            ("grpc.keepalive_time_ms", 10_000),
            ("grpc.keepalive_timeout_ms", 3_000),
            ("grpc.keepalive_permit_without_calls", 1),
            ("grpc.http2.min_time_between_pings_ms", 10_000),
            ("grpc.http2.max_pings_without_data", 0),
            ("grpc.http2.min_ping_interval_without_data_ms", 10_000),
        ]

    def _create_channel(self, addr: str) -> grpc.Channel:
        return grpc.insecure_channel(addr, options=self._channel_options())

    def _refresh_channel(self) -> None:
        with self._ch_lock:
            with suppress(Exception):
                self.channel.close()
            self.channel = self._create_channel(self.server_address)
            self.stub = store_daemon_pb2_grpc.StoreDaemonServiceStub(self.channel)
            self.stub_v2 = store_daemon_pb2_grpc.StoreDaemonServiceStub(self.channel)
            _inc_channel_refresh(self.server_address)

    def _unary_call(
        self,
        method,
        request,
        *,
        timeout: float | int | None = None,
        retries: int = 1,
        span: trace.Span | None = None,
    ):
        # Determine the RPC method name so we can rebind against a fresh stub
        # after channel refresh. Prefer the grpc MultiCallable "_method" path,
        # fall back to Python __name__.
        method_path = getattr(method, "_method", None)
        # gRPC MultiCallable._method may be bytes (e.g., b"/pkg.Service/Method").
        # Decode to str and extract the final segment for getattr(self.stub, name).
        resolved_name = None
        if isinstance(method_path, (bytes, bytearray)):
            try:
                method_path_str = method_path.decode("ascii", errors="ignore")
            except Exception:
                method_path_str = ""
            if method_path_str:
                resolved_name = method_path_str.rsplit("/", 1)[-1]
        elif isinstance(method_path, str):
            resolved_name = method_path.rsplit("/", 1)[-1]
        if not resolved_name:
            resolved_name = getattr(method, "__name__", None)

        cur_method = method
        last_err: Exception | None = None
        for attempt in range(retries + 1):
            if attempt > 0 and resolved_name:
                # Rebind the method on the (potentially) refreshed stub
                reb = getattr(self.stub, resolved_name, None)
                if reb is not None:
                    cur_method = reb
            try:
                return cur_method(request, timeout=timeout)
            except grpc.RpcError as e:  # noqa: BLE001
                last_err = e
                code = e.code()
                if span is not None:
                    with suppress(Exception):
                        span.record_exception(e)
                        span.set_attribute("rpc.grpc.status_code", str(code.name))
                        span.set_attribute("retry.attempt", int(attempt))
                # Retry on transient errors
                if (
                    code
                    in (
                        grpc.StatusCode.UNAVAILABLE,
                        grpc.StatusCode.INTERNAL,
                        grpc.StatusCode.UNKNOWN,
                        grpc.StatusCode.DEADLINE_EXCEEDED,
                    )
                    and attempt < retries
                ):
                    # best-effort method name for logging
                    mname = method_path or resolved_name or "<callable>"
                    _inc_rpc_retry(self.server_address, str(mname), attempt + 1, code)
                    self._refresh_channel()
                    time.sleep(0.05 + random.random() * 0.1)
                    continue
                break
        assert last_err is not None
        raise last_err

    def __del__(self):
        # Best-effort channel cleanup (avoid module globals during shutdown)
        try:
            channel = getattr(self, "channel", None)
            if channel is not None:
                channel.close()
        except Exception:
            pass

    def close(self) -> None:
        """Close underlying gRPC channel."""
        channel = getattr(self, "channel", None)
        if channel is None:
            return
        with suppress(Exception):
            channel.close()

    def __enter__(self) -> "DaemonCtl":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _host_port(self) -> tuple[str, int]:
        """Parse and return (host, port) from server_address.

        Returns (server_address, 0) if parsing fails.
        """
        try:
            host, port_s = self.server_address.split(":", 1)
            return host, int(port_s)
        except Exception:
            return self.server_address, 0

    @contextmanager
    def _client_span(self, name: str) -> Iterator[trace.Span]:
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(name, kind=SpanKind.CLIENT) as span:
            host, port = self._host_port()
            set_span_attributes({"server.address": host, "server.port": port})
            yield span

    def _get_effective_pid(self) -> int:
        return get_host_pid() if self.use_host_pid else os.getpid()

    def unload_from_cpu(self, disk_path):
        with self._client_span("Client/UnloadReplica") as span:
            request = store_daemon_pb2.UnloadReplicaRequest(
                disk_path=disk_path,
                target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_CPU,
            )
            try:
                response = self._unary_call(
                    self.stub.UnloadReplica, request, span=span, timeout=10.0, retries=1
                )
            except grpc.RpcError as e:
                span.record_exception(e)
                span.set_attribute("rpc.grpc.status_code", str(e.code().value[0]))
                logger.error(f"Error: {e}")
                return False
            else:
                return response

    def unload_replica(
        self,
        replica_uuid: str,
        *,
        disk_path: str = "",
        target_device_type: int = store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
        version: str = "",
    ) -> bool:
        if not replica_uuid:
            return False
        with self._client_span("Client/UnloadReplica") as span:
            request = store_daemon_pb2.UnloadReplicaRequest(
                disk_path=disk_path,
                replica_uuid=replica_uuid,
                target_device_type=cast(Any, target_device_type),
                pid=self._get_effective_pid(),
            )
            if version:
                request.version = version
            try:
                self._unary_call(
                    self.stub.UnloadReplica,
                    request,
                    span=span,
                    timeout=10.0,
                    retries=1,
                )
                return True
            except grpc.RpcError as e:
                span.record_exception(e)
                logger.error(f"UnloadReplica failed: {e}")
                return False

    def load_into_gpu(
        self,
        disk_path: str,
        replica_uuid: str,
        device_uuid: str,
        pinned_allocation_timeout_ms: int = int(30e3),
        wait_for_completion: bool = True,
    ):
        logger.debug(
            f"load_into_gpu: {disk_path}, {replica_uuid}, wait_for_completion={wait_for_completion}"
        )

        # Choose between host PID and regular PID based on configuration
        pid = self._get_effective_pid()
        if self.use_host_pid:
            logger.debug(f"Using host PID: {pid}")
        else:
            logger.debug(f"Using container PID: {pid}")

        with self._client_span("Client/MaterializeReplica") as span:
            request = store_daemon_pb2.MaterializeReplicaRequest(
                pid=pid,
                disk_path=disk_path,
                replica_uuid=replica_uuid,
                device_uuid=device_uuid,
                target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
                pinned_allocation_timeout_ms=pinned_allocation_timeout_ms,
            )
            try:
                response = self._unary_call(
                    self.stub.MaterializeReplica,
                    request,
                    timeout=60,
                    span=span,
                    retries=1,
                )
            except grpc.RpcError as e:
                span.record_exception(e)
                if e.code() == grpc.StatusCode.CANCELLED:
                    raise RuntimeError(
                        _grpc_message(e, fallback="Artifact not loaded")
                    ) from e
                elif e.code() == grpc.StatusCode.UNAVAILABLE:
                    raise RuntimeError(
                        f"Local StoreDaemon ({self.server_address}) is not available."
                    ) from e
                else:
                    raise RuntimeError(
                        _grpc_message(e, fallback="MaterializeReplica RPC failed")
                    ) from e

        load_status = response.status
        if (
            load_status
            == store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_FAILED
        ):
            raise RuntimeError(f"Artifact allocation failed for {disk_path}")

        if not wait_for_completion:
            # In async mode, return both handle and status
            logger.info(
                f"Artifact allocation initiated (async): {disk_path}, {replica_uuid}"
            )

            assert response.mem_handle is not None
            return response.mem_handle.cuda_ipc_handle, load_status

        # In sync mode, wait for confirmation
        logger.info(f"Artifact loaded: {disk_path}, {replica_uuid}")

        # For sync mode with new async backend, we need to confirm
        if (
            response.status
            == store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_ALLOCATED
        ):
            # Wait for asynchronous loading to complete
            success = self.confirm_replica_loaded(disk_path, replica_uuid)
            if not success:
                raise RuntimeError(
                    f"Failed to confirm artifact loading for {disk_path}"
                )

        assert response.mem_handle is not None
        return response.mem_handle.cuda_ipc_handle

    def materialize_by_artifact_id(
        self,
        artifact_id: str,
        replica_uuid: str,
        device_uuid: str,
        pinned_allocation_timeout_ms: int = int(30e3),
        wait_for_completion: bool = True,
        view: common_pb2.ViewSpec | None = None,
        view_id: str | None = None,
        placement: store_daemon_pb2.TransformPlacement | None = None,
        return_response: bool = False,
        disk_path: str | None = None,
        preference: store_daemon_pb2.SourcePreference | None = None,
    ) -> store_daemon_pb2.MaterializeReplicaResponse | bytes | tuple[bytes, int]:
        """Materialize a replica by content-addressed artifact_id via daemon.

        Mirrors load_into_gpu but sets artifact_id instead of disk_path.
        Returns CUDA IPC handle bytes (or (handle, status) when async) unless
        ``return_response`` is True, in which case the full gRPC response is returned.
        """
        if view is not None and view_id is not None:
            raise ValueError("Specify only one of view or view_id")

        logger.debug(
            "materialize_by_artifact_id: %s, %s, wait_for_completion=%s",
            artifact_id,
            replica_uuid,
            wait_for_completion,
        )

        pid = self._get_effective_pid()
        with self._client_span("Client/MaterializeReplica") as span:
            request = store_daemon_pb2.MaterializeReplicaRequest(
                pid=pid,
                artifact_id=artifact_id,
                replica_uuid=replica_uuid,
                device_uuid=device_uuid,
                target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
                pinned_allocation_timeout_ms=pinned_allocation_timeout_ms,
            )
            if disk_path:
                request.disk_path = disk_path
            if preference is not None:
                request.preference = preference
            if view is not None:
                request.view.CopyFrom(view)
            elif view_id:
                request.view_id = view_id
            if placement is not None:
                request.placement = placement
            try:
                response = self._unary_call(
                    self.stub.MaterializeReplica,
                    request,
                    timeout=60,
                    span=span,
                    retries=1,
                )
            except grpc.RpcError as e:
                span.record_exception(e)
                if e.code() == grpc.StatusCode.CANCELLED:
                    raise RuntimeError(
                        _grpc_message(e, fallback="Artifact not loaded")
                    ) from e
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    raise RuntimeError(
                        f"Local StoreDaemon ({self.server_address}) is not available."
                    ) from e
                raise RuntimeError(
                    _grpc_message(e, fallback="MaterializeReplica RPC failed")
                ) from e

        load_status = response.status
        if (
            load_status
            == store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_FAILED
        ):
            raise RuntimeError(f"Artifact allocation failed for {artifact_id}")

        if not wait_for_completion:
            logger.info(
                "Artifact allocation initiated (async): %s, %s",
                artifact_id,
                replica_uuid,
            )
            assert response.mem_handle is not None
            if return_response:
                return response
            return response.mem_handle.cuda_ipc_handle, load_status

        logger.info("Artifact loaded: %s, %s", artifact_id, replica_uuid)

        if (
            response.status
            == store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_ALLOCATED
        ):
            # Confirm using empty disk_path (daemon ignores it for P2P sessions)
            success = self.confirm_replica_loaded(disk_path or "", replica_uuid)
            if not success:
                raise RuntimeError(
                    f"Failed to confirm artifact loading for {artifact_id}"
                )

        if return_response:
            return response

        assert response.mem_handle is not None
        return response.mem_handle.cuda_ipc_handle

    def materialize_into_target_v2(
        self,
        *,
        artifact_id: str,
        target_layout: store_daemon_pb2.TargetLayout,
        device_uuid: str,
        preference: store_daemon_pb2.SourcePreference | None = None,
        source_policy: store_daemon_pb2.SourcePolicy | None = None,
        disk_path: str | None = None,
        verify_checksums: bool = True,
        tensor_names: Sequence[str] | None = None,
        view: common_pb2.ViewSpec | None = None,
        view_id: str | None = None,
        view_subset_hash: bytes | None = None,
        placement: store_daemon_pb2.TransformPlacement | None = None,
        pid: int | None = None,
        operation_id: str | None = None,
        return_response: bool = True,
    ) -> store_daemon_pb2.MaterializeIntoTargetResponse:
        if not artifact_id:
            raise ValueError("artifact_id is required")
        if not device_uuid:
            raise ValueError("device_uuid is required")
        if view is not None and view_id is not None:
            raise ValueError("Specify at most one of view or view_id")
        pid_value = self._get_effective_pid() if pid is None else int(pid)
        with self._client_span("Client/MaterializeIntoTarget") as span:
            if preference is not None:
                preference_value = preference
            elif (
                source_policy is not None
                and source_policy.preference
                != store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_UNSPECIFIED
            ):
                preference_value = source_policy.preference
            else:
                preference_value = (
                    store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_AUTO
                )
            request = store_daemon_pb2.MaterializeIntoTargetRequest(
                artifact_id=artifact_id,
                target_layout=target_layout,
                device_uuid=device_uuid,
                pid=pid_value,
                preference=preference_value,
            )
            if tensor_names is not None:
                request.tensor_names.extend(list(tensor_names))
            if source_policy is not None:
                request.source_policy.CopyFrom(source_policy)
            if disk_path:
                request.disk_fallback.disk_path = disk_path
                request.disk_fallback.verify_checksums = bool(verify_checksums)
            if view_subset_hash is not None:
                request.view_subset_hash = bytes(view_subset_hash)
            if view is not None:
                request.view.CopyFrom(view)
            elif view_id is not None:
                request.view_id = str(view_id)
            if placement is not None:
                request.placement = placement
            if operation_id:
                request.operation_id = str(operation_id)
            try:
                response: store_daemon_pb2.MaterializeIntoTargetResponse = (
                    self._unary_call(
                        self.stub_v2.MaterializeIntoTarget,
                        request,
                        timeout=60,
                        span=span,
                        retries=1,
                    )
                )
            except grpc.RpcError as e:  # noqa: BLE001
                span.record_exception(e)
                code = e.code()
                if code == grpc.StatusCode.UNAVAILABLE:
                    raise RuntimeError(
                        f"Local StoreDaemon ({self.server_address}) is not available."
                    ) from e
                if code == grpc.StatusCode.NOT_FOUND:
                    raise RuntimeError(
                        f"Artifact id '{artifact_id}' was not found by StoreDaemon at {self.server_address}."
                    ) from e
                raise RuntimeError(
                    _grpc_message(e, fallback="MaterializeIntoTargetV2 RPC failed")
                ) from e
        if not return_response:
            raise RuntimeError(
                "materialize_into_target_v2 requires return_response=True"
            )
        return response

    def query_replica_status(
        self, ticket: store_daemon_pb2.ReplicaTicket
    ) -> store_daemon_pb2.QueryReplicaStatusResponse:
        with self._client_span("Client/QueryReplicaStatus") as span:
            request = store_daemon_pb2.QueryReplicaStatusRequest(ticket=ticket)
            try:
                response: store_daemon_pb2.QueryReplicaStatusResponse = (
                    self._unary_call(
                        self.stub_v2.QueryReplicaStatus,
                        request,
                        timeout=5.0,
                        span=span,
                        retries=1,
                    )
                )
            except grpc.RpcError as e:  # noqa: BLE001
                span.record_exception(e)
                raise
        return response

    def wait_replica_status(
        self,
        ticket: store_daemon_pb2.ReplicaTicket,
        *,
        timeout_ms: int | None = None,
    ) -> store_daemon_pb2.WaitReplicaStatusResponse:
        with self._client_span("Client/WaitReplicaStatus") as span:
            request = store_daemon_pb2.WaitReplicaStatusRequest(ticket=ticket)
            if timeout_ms is not None:
                request.timeout_ms = int(timeout_ms)
            timeout_s: float = 610.0
            if timeout_ms is not None and timeout_ms > 0:
                timeout_s = max(1.0, float(timeout_ms) / 1000.0 + 1.0)
            try:
                response: store_daemon_pb2.WaitReplicaStatusResponse = self._unary_call(
                    self.stub_v2.WaitReplicaStatus,
                    request,
                    timeout=timeout_s,
                    span=span,
                    retries=1,
                )
            except grpc.RpcError as e:  # noqa: BLE001
                span.record_exception(e)
                raise
        return response

    def release_replica(
        self, ticket: store_daemon_pb2.ReplicaTicket
    ) -> store_daemon_pb2.ReleaseReplicaResponse:
        with self._client_span("Client/ReleaseReplica") as span:
            request = store_daemon_pb2.ReleaseReplicaRequest(ticket=ticket)
            try:
                response: store_daemon_pb2.ReleaseReplicaResponse = self._unary_call(
                    self.stub_v2.ReleaseReplica,
                    request,
                    timeout=5.0,
                    span=span,
                    retries=1,
                )
            except grpc.RpcError as e:  # noqa: BLE001
                span.record_exception(e)
                raise
        return response

    def create_placement_lease(
        self,
        *,
        artifact_id: str,
        view_id: str,
        device_id: int,
        ttl_ms: int | None,
        timeout_s: float | None = None,
    ) -> store_daemon_pb2.CreatePlacementLeaseResponse:
        with self._client_span("Client/CreatePlacementLease") as span:
            request = store_daemon_pb2.CreatePlacementLeaseRequest(
                artifact_id=str(artifact_id),
                view_id=str(view_id),
                device_id=int(device_id),
            )
            if ttl_ms is not None:
                request.ttl_ms = int(ttl_ms)
            call_timeout = 5.0 if timeout_s is None else float(timeout_s)
            try:
                response: store_daemon_pb2.CreatePlacementLeaseResponse = (
                    self._unary_call(
                        self.stub_v2.CreatePlacementLease,
                        request,
                        timeout=call_timeout,
                        span=span,
                        retries=1,
                    )
                )
            except grpc.RpcError as e:  # noqa: BLE001
                span.record_exception(e)
                raise
        return response

    def renew_placement_lease(
        self,
        *,
        lease_token: bytes,
        ttl_ms: int,
        timeout_s: float | None = None,
    ) -> store_daemon_pb2.RenewPlacementLeaseResponse:
        with self._client_span("Client/RenewPlacementLease") as span:
            request = store_daemon_pb2.RenewPlacementLeaseRequest(
                lease_token=bytes(lease_token),
                ttl_ms=int(ttl_ms),
            )
            call_timeout = 5.0 if timeout_s is None else float(timeout_s)
            try:
                response: store_daemon_pb2.RenewPlacementLeaseResponse = (
                    self._unary_call(
                        self.stub_v2.RenewPlacementLease,
                        request,
                        timeout=call_timeout,
                        span=span,
                        retries=1,
                    )
                )
            except grpc.RpcError as e:  # noqa: BLE001
                span.record_exception(e)
                raise
        return response

    def release_placement_lease(
        self,
        *,
        lease_token: bytes,
        timeout_s: float | None = None,
    ) -> store_daemon_pb2.ReleasePlacementLeaseResponse:
        with self._client_span("Client/ReleasePlacementLease") as span:
            request = store_daemon_pb2.ReleasePlacementLeaseRequest(
                lease_token=bytes(lease_token),
            )
            call_timeout = 5.0 if timeout_s is None else float(timeout_s)
            try:
                response: store_daemon_pb2.ReleasePlacementLeaseResponse = (
                    self._unary_call(
                        self.stub_v2.ReleasePlacementLease,
                        request,
                        timeout=call_timeout,
                        span=span,
                        retries=1,
                    )
                )
            except grpc.RpcError as e:  # noqa: BLE001
                span.record_exception(e)
                raise
        return response

    def acquire_retention_handle(
        self,
        *,
        selection: common_pb2.ArtifactSelection,
        policy: store_daemon_pb2.StorePolicy | None = None,
        ttl_ms: int | None = None,
        timeout_s: float | None = None,
    ) -> store_daemon_pb2.AcquireRetentionHandleResponse:
        with self._client_span("Client/AcquireRetentionHandle") as span:
            request = store_daemon_pb2.AcquireRetentionHandleRequest(
                selection=selection,
            )
            if policy is not None:
                request.policy.CopyFrom(policy)
            if ttl_ms is not None:
                request.ttl_ms = int(ttl_ms)
            call_timeout = 5.0 if timeout_s is None else float(timeout_s)
            try:
                response: store_daemon_pb2.AcquireRetentionHandleResponse = (
                    self._unary_call(
                        self.stub_v2.AcquireRetentionHandle,
                        request,
                        timeout=call_timeout,
                        span=span,
                        retries=1,
                    )
                )
            except grpc.RpcError as e:  # noqa: BLE001
                span.record_exception(e)
                raise
        return response

    def renew_retention_handle(
        self,
        *,
        handle_token: bytes,
        extend_ttl_ms: int,
        timeout_s: float | None = None,
    ) -> store_daemon_pb2.RenewRetentionHandleResponse:
        with self._client_span("Client/RenewRetentionHandle") as span:
            request = store_daemon_pb2.RenewRetentionHandleRequest(
                handle_token=bytes(handle_token),
                extend_ttl_ms=int(extend_ttl_ms),
            )
            call_timeout = 5.0 if timeout_s is None else float(timeout_s)
            try:
                response: store_daemon_pb2.RenewRetentionHandleResponse = (
                    self._unary_call(
                        self.stub_v2.RenewRetentionHandle,
                        request,
                        timeout=call_timeout,
                        span=span,
                        retries=1,
                    )
                )
            except grpc.RpcError as e:  # noqa: BLE001
                span.record_exception(e)
                raise
        return response

    def release_retention_handle(
        self,
        *,
        handle_token: bytes,
        timeout_s: float | None = None,
    ) -> store_daemon_pb2.ReleaseRetentionHandleResponse:
        with self._client_span("Client/ReleaseRetentionHandle") as span:
            request = store_daemon_pb2.ReleaseRetentionHandleRequest(
                handle_token=bytes(handle_token),
            )
            call_timeout = 5.0 if timeout_s is None else float(timeout_s)
            try:
                response: store_daemon_pb2.ReleaseRetentionHandleResponse = (
                    self._unary_call(
                        self.stub_v2.ReleaseRetentionHandle,
                        request,
                        timeout=call_timeout,
                        span=span,
                        retries=1,
                    )
                )
            except grpc.RpcError as e:  # noqa: BLE001
                span.record_exception(e)
                raise
        return response

    @overload
    def materialize_by_artifact_id_v2(
        self,
        artifact_id: str,
        replica_uuid: str,
        device_uuid: str,
        pinned_allocation_timeout_ms: int = int(30e3),
        *,
        wait_for_completion: bool = True,
        view: common_pb2.ViewSpec | None = None,
        view_id: str | None = None,
        placement: store_daemon_pb2.TransformPlacement | None = None,
        return_response: Literal[True],
        disk_path: str | None = None,
        preference: store_daemon_pb2.SourcePreference | None = None,
        source_policy: store_daemon_pb2.SourcePolicy | None = None,
        tensor_names: Sequence[str] | None = None,
        verify_checksums: bool = True,
        view_subset_hash: bytes | None = None,
        target_device_type: store_daemon_pb2.DeviceType = store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
        lease_mode: store_daemon_pb2.LeaseMode = store_daemon_pb2.LeaseMode.LEASE_MODE_UNSPECIFIED,
        timeout_s: float | int | None = None,
    ) -> store_daemon_pb2.MaterializeReplicaResponse: ...

    @overload
    def materialize_by_artifact_id_v2(
        self,
        artifact_id: str,
        replica_uuid: str,
        device_uuid: str,
        pinned_allocation_timeout_ms: int = int(30e3),
        *,
        wait_for_completion: Literal[False],
        view: common_pb2.ViewSpec | None = None,
        view_id: str | None = None,
        placement: store_daemon_pb2.TransformPlacement | None = None,
        return_response: Literal[False] = False,
        disk_path: str | None = None,
        preference: store_daemon_pb2.SourcePreference | None = None,
        source_policy: store_daemon_pb2.SourcePolicy | None = None,
        tensor_names: Sequence[str] | None = None,
        verify_checksums: bool = True,
        view_subset_hash: bytes | None = None,
        target_device_type: store_daemon_pb2.DeviceType = store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
        lease_mode: store_daemon_pb2.LeaseMode = store_daemon_pb2.LeaseMode.LEASE_MODE_UNSPECIFIED,
        timeout_s: float | int | None = None,
    ) -> tuple[bytes, store_daemon_pb2.MaterializeReplicaStatus]: ...

    @overload
    def materialize_by_artifact_id_v2(
        self,
        artifact_id: str,
        replica_uuid: str,
        device_uuid: str,
        pinned_allocation_timeout_ms: int = int(30e3),
        *,
        wait_for_completion: Literal[True] = True,
        view: common_pb2.ViewSpec | None = None,
        view_id: str | None = None,
        placement: store_daemon_pb2.TransformPlacement | None = None,
        return_response: Literal[False] = False,
        disk_path: str | None = None,
        preference: store_daemon_pb2.SourcePreference | None = None,
        source_policy: store_daemon_pb2.SourcePolicy | None = None,
        tensor_names: Sequence[str] | None = None,
        verify_checksums: bool = True,
        view_subset_hash: bytes | None = None,
        target_device_type: store_daemon_pb2.DeviceType = store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
        lease_mode: store_daemon_pb2.LeaseMode = store_daemon_pb2.LeaseMode.LEASE_MODE_UNSPECIFIED,
        timeout_s: float | int | None = None,
    ) -> bytes: ...

    def materialize_by_artifact_id_v2(
        self,
        artifact_id: str,
        replica_uuid: str,
        device_uuid: str,
        pinned_allocation_timeout_ms: int = int(30e3),
        wait_for_completion: bool = True,
        view: common_pb2.ViewSpec | None = None,
        view_id: str | None = None,
        placement: store_daemon_pb2.TransformPlacement | None = None,
        return_response: bool = False,
        disk_path: str | None = None,
        preference: store_daemon_pb2.SourcePreference | None = None,
        source_policy: store_daemon_pb2.SourcePolicy | None = None,
        tensor_names: Sequence[str] | None = None,
        verify_checksums: bool = True,
        view_subset_hash: bytes | None = None,
        target_device_type: store_daemon_pb2.DeviceType = store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
        lease_mode: store_daemon_pb2.LeaseMode = store_daemon_pb2.LeaseMode.LEASE_MODE_UNSPECIFIED,
        timeout_s: float | int | None = None,
    ) -> (
        store_daemon_pb2.MaterializeReplicaResponse
        | bytes
        | tuple[bytes, store_daemon_pb2.MaterializeReplicaStatus]
    ):
        if view is not None and view_id is not None:
            raise ValueError("Specify only one of view or view_id")
        logger.debug(
            "materialize_by_artifact_id_v2: %s, %s, wait_for_completion=%s",
            artifact_id,
            replica_uuid,
            wait_for_completion,
        )
        pid = self._get_effective_pid()
        with self._client_span("Client/MaterializeReplicaV2") as span:
            if preference is not None:
                preference_value = preference
            elif (
                source_policy is not None
                and source_policy.preference
                != store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_UNSPECIFIED
            ):
                preference_value = source_policy.preference
            else:
                preference_value = (
                    store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_AUTO
                )
            device_uuid_value = (
                ""
                if target_device_type == store_daemon_pb2.DeviceType.DEVICE_TYPE_CPU
                else device_uuid
            )
            request = store_daemon_pb2.MaterializeReplicaRequest(
                pid=pid,
                artifact_id=artifact_id,
                replica_uuid=replica_uuid,
                device_uuid=device_uuid_value,
                target_device_type=target_device_type,
                pinned_allocation_timeout_ms=pinned_allocation_timeout_ms,
                preference=preference_value,
                lease_mode=lease_mode,
            )
            if source_policy is not None:
                request.source_policy.CopyFrom(source_policy)
            if disk_path:
                request.disk_fallback.disk_path = disk_path
                request.disk_fallback.verify_checksums = bool(verify_checksums)
            if tensor_names:
                request.tensor_names.extend(tensor_names)
            if view_subset_hash:
                request.view_subset_hash = view_subset_hash
            if view is not None:
                request.view.CopyFrom(view)
            elif view_id:
                request.view_id = view_id
            if placement is not None:
                request.placement = placement
            try:
                response: store_daemon_pb2.MaterializeReplicaResponse = (
                    self._unary_call(
                        self.stub_v2.MaterializeReplica,
                        request,
                        timeout=60 if timeout_s is None else timeout_s,
                        span=span,
                        retries=1,
                    )
                )
            except grpc.RpcError as e:  # noqa: BLE001
                span.record_exception(e)
                code = e.code()
                if code == grpc.StatusCode.UNAVAILABLE:
                    raise RuntimeError(
                        f"Local StoreDaemon ({self.server_address}) is not available."
                    ) from e
                if code == grpc.StatusCode.NOT_FOUND:
                    raise RuntimeError(
                        f"Artifact id '{artifact_id}' was not found by StoreDaemon at {self.server_address}."
                    ) from e
                raise RuntimeError(
                    _grpc_message(e, fallback="MaterializeReplicaV2 RPC failed")
                ) from e

        load_status = response.status
        if (
            load_status
            == store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_FAILED
        ):
            raise RuntimeError(f"Artifact allocation failed for {artifact_id}")

        if not wait_for_completion:
            logger.info(
                "Artifact allocation initiated (async): %s, %s",
                artifact_id,
                replica_uuid,
            )
            assert response.mem_handle is not None
            if return_response:
                return response
            if target_device_type != store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU:
                raise ValueError(
                    "materialize_by_artifact_id_v2 must use return_response=True for non-GPU targets"
                )
            return response.mem_handle.cuda_ipc_handle, load_status

        logger.info("Artifact loaded: %s, %s", artifact_id, replica_uuid)
        if (
            response.status
            == store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_ALLOCATED
        ):
            success = self.confirm_replica_loaded(
                disk_path or "",
                replica_uuid,
                target_device_type=target_device_type,
            )
            if not success:
                raise RuntimeError(
                    f"Failed to confirm artifact loading for {artifact_id}"
                )
        if return_response:
            return response
        if target_device_type != store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU:
            raise ValueError(
                "materialize_by_artifact_id_v2 must use return_response=True for non-GPU targets"
            )
        assert response.mem_handle is not None
        return response.mem_handle.cuda_ipc_handle

    @overload
    def materialize_by_key_v2(
        self,
        key: str,
        replica_uuid: str,
        device_id: int,
        pinned_allocation_timeout_ms: int = int(30e3),
        *,
        wait_for_completion: bool = True,
        return_response: Literal[True],
        preference: store_daemon_pb2.SourcePreference | None = None,
        source_policy: store_daemon_pb2.SourcePolicy | None = None,
        tensor_names: Sequence[str] | None = None,
        view_subset_hash: bytes | None = None,
        target_device_type: store_daemon_pb2.DeviceType = store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
        lease_mode: store_daemon_pb2.LeaseMode = store_daemon_pb2.LeaseMode.LEASE_MODE_UNSPECIFIED,
        timeout_s: float | int | None = None,
    ) -> store_daemon_pb2.MaterializeByKeyResponse: ...

    @overload
    def materialize_by_key_v2(
        self,
        key: str,
        replica_uuid: str,
        device_id: int,
        pinned_allocation_timeout_ms: int = int(30e3),
        *,
        wait_for_completion: Literal[False],
        return_response: Literal[False] = False,
        preference: store_daemon_pb2.SourcePreference | None = None,
        source_policy: store_daemon_pb2.SourcePolicy | None = None,
        tensor_names: Sequence[str] | None = None,
        view_subset_hash: bytes | None = None,
        target_device_type: store_daemon_pb2.DeviceType = store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
        lease_mode: store_daemon_pb2.LeaseMode = store_daemon_pb2.LeaseMode.LEASE_MODE_UNSPECIFIED,
        timeout_s: float | int | None = None,
    ) -> tuple[bytes, store_daemon_pb2.MaterializeReplicaStatus, str, str]: ...

    @overload
    def materialize_by_key_v2(
        self,
        key: str,
        replica_uuid: str,
        device_id: int,
        pinned_allocation_timeout_ms: int = int(30e3),
        *,
        wait_for_completion: Literal[True] = True,
        return_response: Literal[False] = False,
        preference: store_daemon_pb2.SourcePreference | None = None,
        source_policy: store_daemon_pb2.SourcePolicy | None = None,
        tensor_names: Sequence[str] | None = None,
        view_subset_hash: bytes | None = None,
        target_device_type: store_daemon_pb2.DeviceType = store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
        lease_mode: store_daemon_pb2.LeaseMode = store_daemon_pb2.LeaseMode.LEASE_MODE_UNSPECIFIED,
        timeout_s: float | int | None = None,
    ) -> tuple[bytes, str, str]: ...

    def materialize_by_key_v2(
        self,
        key: str,
        replica_uuid: str,
        device_id: int,
        pinned_allocation_timeout_ms: int = int(30e3),
        wait_for_completion: bool = True,
        return_response: bool = False,
        preference: store_daemon_pb2.SourcePreference | None = None,
        source_policy: store_daemon_pb2.SourcePolicy | None = None,
        tensor_names: Sequence[str] | None = None,
        view_subset_hash: bytes | None = None,
        target_device_type: store_daemon_pb2.DeviceType = store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
        lease_mode: store_daemon_pb2.LeaseMode = store_daemon_pb2.LeaseMode.LEASE_MODE_UNSPECIFIED,
        timeout_s: float | int | None = None,
    ) -> (
        store_daemon_pb2.MaterializeByKeyResponse
        | tuple[bytes, store_daemon_pb2.MaterializeReplicaStatus, str, str]
        | tuple[bytes, str, str]
    ):
        logger.debug(
            "materialize_by_key_v2: %s, %s, wait_for_completion=%s",
            key,
            replica_uuid,
            wait_for_completion,
        )
        pid = self._get_effective_pid()
        with self._client_span("Client/MaterializeByKeyV2") as span:
            if preference is not None:
                preference_value = preference
            elif (
                source_policy is not None
                and source_policy.preference
                != store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_UNSPECIFIED
            ):
                preference_value = source_policy.preference
            else:
                preference_value = (
                    store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_AUTO
                )
            device_id_value = (
                0
                if target_device_type == store_daemon_pb2.DeviceType.DEVICE_TYPE_CPU
                else int(device_id)
            )
            request = store_daemon_pb2.MaterializeByKeyRequest(
                key=key,
                device_id=device_id_value,
                pinned_allocation_timeout_ms=int(pinned_allocation_timeout_ms),
                pid=pid,
                replica_uuid=replica_uuid,
                preference=preference_value,
                target_device_type=target_device_type,
                lease_mode=lease_mode,
            )
            if source_policy is not None:
                request.source_policy.CopyFrom(source_policy)
            if tensor_names:
                request.tensor_names.extend(tensor_names)
            if view_subset_hash:
                request.view_subset_hash = view_subset_hash
            try:
                response: store_daemon_pb2.MaterializeByKeyResponse = self._unary_call(
                    self.stub_v2.MaterializeByKey,
                    request,
                    timeout=60 if timeout_s is None else timeout_s,
                    span=span,
                    retries=1,
                )
            except grpc.RpcError as e:  # noqa: BLE001
                span.record_exception(e)
                code = e.code()
                if code == grpc.StatusCode.UNAVAILABLE:
                    raise RuntimeError(
                        f"Local StoreDaemon ({self.server_address}) is not available."
                    ) from e
                if code == grpc.StatusCode.NOT_FOUND:
                    detail = ""
                    with suppress(Exception):
                        detail = e.details() or ""
                    message = (
                        f"Artifact key '{key}' was not found by StoreDaemon at "
                        f"{self.server_address}."
                    )
                    if detail:
                        message += f" Daemon response: {detail}."
                    raise RuntimeError(message) from e
                status_name = "UNKNOWN"
                detail_msg = _grpc_message(e, fallback="RPC failed")
                with suppress(Exception):
                    if code is not None:
                        status_name = code.name
                raise RuntimeError(
                    f"MaterializeByKeyV2 RPC failed with status={status_name}: {detail_msg}"
                ) from e

        load_status = response.status
        if (
            load_status
            == store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_FAILED
        ):
            raise RuntimeError(f"Artifact allocation failed for key={key}")

        used_disk_path = (
            response.used_disk_path if hasattr(response, "used_disk_path") else ""
        )
        artifact_id = response.artifact_id if hasattr(response, "artifact_id") else ""

        handle_bytes = (
            response.mem_handle.cuda_ipc_handle
            if response.HasField("mem_handle")
            else b""
        )
        if not wait_for_completion:
            assert response.mem_handle is not None
            if return_response:
                return response
            if target_device_type != store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU:
                raise ValueError(
                    "materialize_by_key_v2 must use return_response=True for non-GPU targets"
                )
            return (handle_bytes, load_status, used_disk_path, artifact_id)

        if (
            response.status
            == store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_ALLOCATED
        ):
            success = self.confirm_replica_loaded(
                used_disk_path,
                replica_uuid,
                target_device_type=target_device_type,
            )
            if not success:
                raise RuntimeError(f"Failed to confirm artifact loading for key={key}")

        assert response.mem_handle is not None
        if return_response:
            return response
        if target_device_type != store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU:
            raise ValueError(
                "materialize_by_key_v2 must use return_response=True for non-GPU targets"
            )
        return (handle_bytes, used_disk_path, artifact_id)

    def resolve_artifact_from_disk_v2(
        self, *, disk_path: str, verify_checksums: bool = True
    ) -> store_daemon_pb2.ResolveArtifactFromDiskResponse:
        if not disk_path:
            raise ValueError("disk_path is required")
        with self._client_span("Client/ResolveArtifactFromDiskV2") as span:
            request = store_daemon_pb2.ResolveArtifactFromDiskRequest(
                disk_path=disk_path, verify_checksums=bool(verify_checksums)
            )
            try:
                return self._unary_call(
                    self.stub_v2.ResolveArtifactFromDisk,
                    request,
                    timeout=15.0,
                    span=span,
                    retries=1,
                )
            except grpc.RpcError as e:  # noqa: BLE001
                span.record_exception(e)
                code = e.code()
                if code == grpc.StatusCode.UNAVAILABLE:
                    _raise_grpc_error(
                        RuntimeError(
                            f"Local StoreDaemon ({self.server_address}) is not available."
                        ),
                        cause=e,
                    )
                if code == grpc.StatusCode.INVALID_ARGUMENT:
                    _raise_grpc_error(
                        ValueError(_grpc_message(e, fallback="invalid argument")),
                        cause=e,
                    )
                if code == grpc.StatusCode.NOT_FOUND:
                    _raise_grpc_error(
                        FileNotFoundError(
                            _grpc_message(e, fallback="artifact not found on disk")
                        ),
                        cause=e,
                    )
                if code == grpc.StatusCode.PERMISSION_DENIED:
                    _raise_grpc_error(
                        PermissionError(
                            _grpc_message(e, fallback="disk_path not permitted")
                        ),
                        cause=e,
                    )
                _raise_grpc_error(
                    RuntimeError(
                        "ResolveArtifactFromDiskV2 RPC failed: "
                        f"{_grpc_message(e, fallback='rpc failed')}"
                    ),
                    cause=e,
                )

    def materialize_by_key(
        self,
        key: str,
        replica_uuid: str,
        device_id: int,
        pinned_allocation_timeout_ms: int = int(30e3),
        wait_for_completion: bool = True,
        return_response: bool = False,
    ):
        """Materialize a replica by key via daemon.

        Returns
            If wait_for_completion=True: (cuda_ipc_handle_bytes, used_disk_path, artifact_id)
            If wait_for_completion=False: (cuda_ipc_handle_bytes, load_status, used_disk_path, artifact_id)
        """
        logger.debug(
            f"materialize_by_key: {key}, {replica_uuid}, wait_for_completion={wait_for_completion}"
        )

        pid = self._get_effective_pid()
        with self._client_span("Client/MaterializeByKey") as span:
            request = store_daemon_pb2.MaterializeByKeyRequest(
                key=key,
                device_id=int(device_id),
                pinned_allocation_timeout_ms=int(pinned_allocation_timeout_ms),
                pid=pid,
                replica_uuid=replica_uuid,
            )
            try:
                response = self._unary_call(
                    self.stub.MaterializeByKey,
                    request,
                    timeout=60,
                    span=span,
                    retries=1,
                )
            except grpc.RpcError as e:
                span.record_exception(e)
                code = e.code()
                if code == grpc.StatusCode.UNAVAILABLE:
                    raise RuntimeError(
                        f"Local StoreDaemon ({self.server_address}) is not available."
                    ) from e
                if code == grpc.StatusCode.NOT_FOUND:
                    detail = ""
                    with suppress(Exception):
                        detail = e.details() or ""
                    message = (
                        f"Artifact key '{key}' was not found by StoreDaemon at "
                        f"{self.server_address}."
                    )
                    if detail:
                        message += f" Daemon response: {detail}."
                    message += " Verify the key spelling or register the artifact before loading."
                    raise RuntimeError(message) from e
                status_name = "UNKNOWN"
                detail_msg = _grpc_message(e, fallback="RPC failed")
                with suppress(Exception):
                    if code is not None:
                        status_name = code.name
                raise RuntimeError(
                    f"MaterializeByKey RPC failed with status={status_name}: {detail_msg}"
                ) from e

        load_status = response.status
        if (
            load_status
            == store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_FAILED
        ):
            raise RuntimeError(f"Artifact allocation failed for key={key}")

        used_disk_path = (
            response.used_disk_path if hasattr(response, "used_disk_path") else ""
        )
        artifact_id = response.artifact_id if hasattr(response, "artifact_id") else ""

        handle_bytes = response.mem_handle.cuda_ipc_handle
        if not wait_for_completion:
            assert response.mem_handle is not None
            if return_response:
                return response
            return (handle_bytes, load_status, used_disk_path, artifact_id)

        # Synchronous path: confirm completion using the session created with replica_uuid
        if (
            load_status
            == store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_ALLOCATED
        ):
            success = self.confirm_replica_loaded(used_disk_path, replica_uuid)
            if not success:
                raise RuntimeError(f"Failed to confirm artifact loading for key={key}")

        assert response.mem_handle is not None
        if return_response:
            return response
        return (handle_bytes, used_disk_path, artifact_id)

    def confirm_replica_loaded(
        self,
        disk_path: str,
        replica_uuid: str,
        *,
        target_device_type: store_daemon_pb2.DeviceType = store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
    ) -> bool:
        with self._client_span("Client/ConfirmReplica") as span:
            request = store_daemon_pb2.ConfirmReplicaRequest(
                disk_path=disk_path,
                replica_uuid=replica_uuid,
                target_device_type=target_device_type,
            )
            try:
                _ = self._unary_call(
                    self.stub.ConfirmReplica,
                    request,
                    timeout=10.0,
                    span=span,
                    retries=1,
                )
                logger.info("Artifact loaded")
                return True
            except grpc.RpcError as e:
                span.record_exception(e)
                if e.code() == grpc.StatusCode.CANCELLED:
                    logger.error("Artifact not loaded")
                    return False
                else:
                    logger.error(f"Error: {e}")
                    return False

    def get_server_config(self) -> ServerConfig:
        with self._client_span("Client/GetServerConfig") as span:
            request = store_daemon_pb2.GetServerConfigRequest()
            try:
                response: store_daemon_pb2.GetServerConfigResponse = self._unary_call(
                    self.stub.GetServerConfig,
                    request,
                    timeout=5.0,
                    span=span,
                    retries=1,
                )
            except grpc.RpcError as e:
                span.record_exception(e)
                logger.error(f"Error: {e}")
                raise RuntimeError("GetServerConfig failed") from e
            else:
                # Map both legacy and new fields for smooth migration
                local_handle_socket_path = str(
                    getattr(response, "local_handle_socket_path", "") or ""
                )
                cpu_shared_memory_enabled = bool(
                    getattr(response, "cpu_shared_memory_enabled", False)
                )
                if not local_handle_socket_path and cpu_shared_memory_enabled:
                    try:
                        from tensorcast.cli_utils.paths import (
                            discover_local_handle_socket_path,
                            get_session_address,
                        )

                        session_address = get_session_address()
                        if session_address and session_address == self.server_address:
                            discovered = discover_local_handle_socket_path()
                            if discovered is not None:
                                local_handle_socket_path = str(discovered)
                    except Exception:
                        pass
                return ServerConfig(
                    tx_slice_bytes=int(getattr(response, "tx_slice_bytes", 0)),
                    mem_pool_size=int(getattr(response, "mem_pool_size", 0)),
                    artifact_chunk_bytes=int(
                        getattr(response, "artifact_chunk_bytes", 0)
                    ),
                    local_handle_socket_path=local_handle_socket_path,
                    cpu_shared_memory_enabled=cpu_shared_memory_enabled,
                )

    def get_worker_status(self) -> store_daemon_pb2.GetWorkerStatusResponse:
        with self._client_span("Client/GetWorkerStatus") as span:
            request = store_daemon_pb2.GetWorkerStatusRequest()
            try:
                response: store_daemon_pb2.GetWorkerStatusResponse = self._unary_call(
                    self.stub_v2.GetWorkerStatus,
                    request,
                    timeout=5.0,
                    span=span,
                    retries=1,
                )
            except grpc.RpcError as e:  # noqa: BLE001
                span.record_exception(e)
                raise
        return response

    # ------------------------------------------------------------------
    # Memory Artifact registration (outer-layer client API)
    # ------------------------------------------------------------------

    def begin_register_artifact(
        self,
        *,
        device_id: int,
        total_size_bytes: int,
        ttl_ms: int | None = None,
        tensor_index_key: str | None = None,
        tensor_index_data: bytes | None = None,
        encoding: str = "json",
        schema_version: str = "v3",
        client_artifact_id: str | None = None,
        plan: Plan | None = None,
        policy: StorePolicy | dict[str, object] | str | None = None,
        timeout_s: float = 30.0,
        view: store_daemon_pb2.ViewRegistrationOptions | None = None,
    ) -> BeginRegisterArtifactResult:
        """Begin unified artifact registration.

        Args:
            device_id: target device ordinal (single-GPU invariant).
            total_size_bytes: AVBS total size (8B aligned).
            ttl_ms: optional TTL used by Lease/UMA/VS plans.
            tensor_index_key: optional hex key of canonical index.
            tensor_index_data: optional canonical index bytes (preferred).
            encoding: "json" or "cbor" for index bytes.
            schema_version: index schema version (e.g., "v3").
            plan: oneof plan options dict: {"kind": "coalesced"|"uma_vs"|"lease", ...}.

        Returns:
            Dict with registration_id, device_id, total_size, and optional handshake:
            - coalesced: {"daemon_ipc_handle": bytes}
            - uma_vs: {"vs": {"stream_token" or ring info}}
        """
        if device_id < 0 or total_size_bytes <= 0:
            raise ValueError("Invalid arguments for begin_register_artifact")
        if not tensor_index_key and tensor_index_data is None:
            raise ValueError(
                "Either tensor_index_key or tensor_index_data must be provided"
            )

        req = store_daemon_pb2.BeginRegisterArtifactRequest(
            device_id=int(device_id),
            total_size=int(total_size_bytes),
            owner_pid=int(self._get_effective_pid()),
        )
        if ttl_ms is not None and ttl_ms > 0:
            req.ttl_ms = int(ttl_ms)
        if tensor_index_data is not None:
            req.tensor_index_data.CopyFrom(
                store_daemon_pb2.TensorIndexData(
                    data=tensor_index_data,
                    schema_version=schema_version,
                    encoding=encoding,
                )
            )
        else:
            req.tensor_index_key = tensor_index_key or ""

        if client_artifact_id:
            req.client_artifact_id = client_artifact_id

        # Plan oneof
        if plan is None:
            from tensorcast.types import CoalescedPlan as _DefaultPlan

            plan = _DefaultPlan()
        kind = plan.kind
        plan.apply_to_begin_request(req)
        policy_proto = self._policy_to_proto(policy)
        if policy_proto is not None:
            req.policy.CopyFrom(policy_proto)
        if view is not None:
            req.view.CopyFrom(view)

        with self._client_span("Client/BeginRegisterArtifact") as span:
            set_span_attributes(
                {
                    "tc.device.id": int(device_id),
                    "tc.size.bytes": int(total_size_bytes),
                    "tc.plan": str(kind),
                }
            )
            try:
                resp = self._unary_call(
                    self.stub.BeginRegisterArtifact,
                    req,
                    timeout=timeout_s,
                    span=span,
                    retries=1,
                )
            except grpc.RpcError as e:
                span.record_exception(e)
                code = e.code()
                if code == grpc.StatusCode.UNAVAILABLE:
                    raise RuntimeError(
                        f"Local StoreDaemon ({self.server_address}) is not available."
                    ) from e
                if code == grpc.StatusCode.INVALID_ARGUMENT:
                    raise ValueError(
                        _grpc_message(e, fallback="invalid argument")
                    ) from e
                if code == grpc.StatusCode.RESOURCE_EXHAUSTED:
                    raise MemoryError(
                        _grpc_message(e, fallback="resource exhausted")
                    ) from e
                if code == grpc.StatusCode.DEADLINE_EXCEEDED:
                    raise TimeoutError(
                        _grpc_message(e, fallback="deadline exceeded")
                    ) from e
                raise RuntimeError(
                    "BeginRegisterArtifact failed: "
                    f"{_grpc_message(e, fallback='rpc failed')}"
                ) from e

            # Build typed handshake result
            if resp.HasField("coalesced"):
                handshake = CoalescedHandshake(
                    daemon_ipc_handle=bytes(resp.coalesced.daemon_ipc_handle)
                )
            elif resp.HasField("lease"):
                handshake = LeaseHandshake()
            elif resp.HasField("stable_dram"):
                handshake = StableDramHandshake(
                    staging_cuda_ipc_handle=bytes(
                        resp.stable_dram.staging_cuda_ipc_handle
                    )
                )
            else:
                # Should not happen
                raise RuntimeError("BeginRegisterArtifact: missing handshake")

            return BeginRegisterArtifactResult(
                registration_id=resp.registration_id,
                device_id=int(resp.device_id),
                total_size=int(resp.total_size),
                handshake=handshake,
            )

    def commit_registered_artifact(
        self,
        registration_id: str,
        *,
        timeout_s: float = 30.0,
    ) -> "CommitResult":
        """Commit a previously begun tensor dict registration.

        Returns CommitResult(descriptor, existed).
        """

        if not registration_id:
            raise ValueError("registration_id is required")

        req = store_daemon_pb2.CommitRegisteredArtifactRequest(
            registration_id=registration_id
        )
        with self._client_span("Client/CommitRegisteredArtifact") as span:
            try:
                resp = self._unary_call(
                    self.stub.CommitRegisteredArtifact,
                    req,
                    timeout=timeout_s,
                    span=span,
                    retries=0,
                )
            except grpc.RpcError as e:
                span.record_exception(e)
                code = e.code()
                if code == grpc.StatusCode.UNAVAILABLE:
                    raise RuntimeError(
                        f"Local StoreDaemon ({self.server_address}) is not available."
                    ) from e
                if code == grpc.StatusCode.INVALID_ARGUMENT:
                    raise ValueError(
                        _grpc_message(e, fallback="invalid argument")
                    ) from e
                if code == grpc.StatusCode.NOT_FOUND:
                    raise KeyError(
                        _grpc_message(e, fallback="registration not found")
                    ) from e
                if code == grpc.StatusCode.DEADLINE_EXCEEDED:
                    raise TimeoutError(
                        _grpc_message(e, fallback="deadline exceeded")
                    ) from e
                raise RuntimeError(
                    "CommitRegisteredArtifact failed: "
                    f"{_grpc_message(e, fallback='rpc failed')}"
                ) from e

            existed = bool(resp.existed)
            if existed:
                set_span_attributes({"tc.register.existed": True})
            desc = resp.artifact_descriptor

            ad = ArtifactDescriptor(
                artifact_id=desc.artifact_id,
                index_multihash=desc.index_multihash,
                data_multihash=desc.data_multihash,
                schema_version=desc.schema_version,
                encoding=desc.encoding,
                total_size=int(desc.total_size),
                id_kind=desc.id_kind,
            )
            canonical_ranges: tuple[CanonicalRange, ...] = tuple(
                CanonicalRange(offset=int(r.offset), length=int(r.length))
                for r in resp.canonical_ranges
            )
            view_index_json = (
                bytes(resp.view_index_json) if resp.view_index_json else None
            )
            view_id = resp.view_id or None
            view_data_hash = resp.view_data_hash or None
            registration_kind = "piece" if resp.allow_partial else "canonical"
            local_stable_tier = None
            if resp.HasField("local_stable_tier"):
                status = resp.local_stable_tier.status
                if status == store_daemon_pb2.LOCAL_STABLE_TIER_STATUS_READY:
                    status_text = "ready"
                elif status == store_daemon_pb2.LOCAL_STABLE_TIER_STATUS_DEGRADED:
                    status_text = "degraded"
                else:
                    status_text = "skipped"
                local_stable_tier = LocalStableTierResult(
                    status=status_text,
                    message=resp.local_stable_tier.message or None,
                )
            return CommitResult(
                descriptor=ad,
                existed=existed,
                view_id=view_id,
                view_index_json=view_index_json,
                view_data_hash=view_data_hash,
                canonical_ranges=canonical_ranges,
                registration_kind=registration_kind,
                allow_partial=bool(resp.allow_partial),
                local_stable_tier=local_stable_tier,
            )

    def abort_registered_artifact(
        self, registration_id: str, *, timeout_s: float = 15.0
    ) -> bool:
        """Abort a pending tensor dict registration and free allocated memory."""
        if not registration_id:
            raise ValueError("registration_id is required")

        req = store_daemon_pb2.AbortRegisteredArtifactRequest(
            registration_id=registration_id
        )
        with self._client_span("Client/AbortRegisteredArtifact") as span:
            try:
                self._unary_call(
                    self.stub.AbortRegisteredArtifact,
                    req,
                    timeout=timeout_s,
                    span=span,
                    retries=1,
                )
            except grpc.RpcError as e:
                span.record_exception(e)
                code = e.code()
                if code == grpc.StatusCode.UNAVAILABLE:
                    raise RuntimeError(
                        f"Local StoreDaemon ({self.server_address}) is not available."
                    ) from e
                if code == grpc.StatusCode.INVALID_ARGUMENT:
                    raise ValueError(
                        _grpc_message(e, fallback="invalid argument")
                    ) from e
                if code == grpc.StatusCode.NOT_FOUND:
                    # Treat as already-aborted/missing
                    logger.warning(
                        "AbortRegisteredArtifact: registration not found: %s",
                        registration_id,
                    )
                    return False
                raise RuntimeError(
                    "AbortRegisteredArtifact failed: "
                    f"{_grpc_message(e, fallback='rpc failed')}"
                ) from e

            return True

    # ------------------------------------------------------------------
    # Registration feed/keepalive helpers (Lease only)
    # ------------------------------------------------------------------

    def feed_register_artifact_lease_segments(
        self,
        registration_id: str,
        segments: list[LeaseSegment],
        *,
        storages: list["RegisterStorage"] | None = None,
        tensor_aliases: list["RegisterTensorAlias"] | None = None,
    ) -> bool:
        # Stream lease segments via FeedRegisterArtifactStream
        def _iter():
            req = store_daemon_pb2.FeedRegisterArtifactStreamRequest(
                registration_id=registration_id
            )
            for s in segments:
                seg = req.lease_segments.segments.add()
                seg.storage_id = s.storage_id
                seg.storage_offset = int(s.storage_offset)
                seg.length = int(s.length)
                seg.artifact_offset = int(s.artifact_offset)
            if storages:
                for storage in storages:
                    entry = req.storage_entries.add()
                    entry.storage_id = storage.storage_id
                    entry.device_id = int(storage.device_id)
                    if storage.cuda_ipc_handle is not None:
                        entry.cuda_ipc_handle = storage.cuda_ipc_handle
                    if storage.vram_region_id:
                        entry.vram_region_id = storage.vram_region_id
                    entry.storage_length = int(storage.storage_length)
                    entry.mapping_base_offset = int(storage.mapping_base_offset)
            if tensor_aliases:
                for alias in tensor_aliases:
                    dst = req.tensor_aliases.add()
                    dst.name = alias.name
                    dst.storage_id = alias.storage_id
                    dst.storage_offset = int(alias.storage_offset)
                    dst.logical_length = int(alias.logical_length)
                    dst.shape.extend(int(v) for v in alias.shape)
                    dst.stride.extend(int(v) for v in alias.stride)
                    dst.dtype = alias.dtype
            yield req

        try:
            # Do not retry streaming feeds; iterator cannot be reused safely.
            self._unary_call(
                self.stub.FeedRegisterArtifactStream, _iter(), timeout=30.0, retries=0
            )
            return True
        except grpc.RpcError as e:  # noqa: BLE001
            logger.error(f"FeedRegisterArtifactStream(lease) failed: {e}")
            return False

    def register_vram_region(
        self,
        *,
        device_id: int,
        size_bytes: int,
        ttl_ms: int,
        cuda_ipc_handle: bytes,
        session_id: str | None = None,
        region_name: str | None = None,
        timeout_s: float = 10.0,
    ) -> VramRegionHandle:
        if device_id < 0:
            raise ValueError("device_id must be non-negative")
        if size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        if ttl_ms < 0:
            raise ValueError("ttl_ms must be non-negative (0 disables TTL)")
        if not cuda_ipc_handle:
            raise ValueError("cuda_ipc_handle must not be empty")

        req = store_daemon_pb2.RegisterVramRegionRequest(
            device_id=int(device_id),
            cuda_ipc_handle=bytes(cuda_ipc_handle),
            size_bytes=int(size_bytes),
            ttl_ms=int(ttl_ms),
            owner_pid=int(self._get_effective_pid()),
        )
        if session_id:
            req.session_id = session_id
        if region_name:
            req.region_name = region_name

        with self._client_span("Client/RegisterVramRegion") as span:
            set_span_attributes(
                {
                    "tc.device.id": int(device_id),
                    "tc.region.size_bytes": int(size_bytes),
                    "tc.region.ttl_ms": int(ttl_ms),
                }
            )
            try:
                resp = self._unary_call(
                    self.stub.RegisterVramRegion,
                    req,
                    timeout=timeout_s,
                    span=span,
                    retries=1,
                )
            except grpc.RpcError as e:
                span.record_exception(e)
                code = e.code()
                if code == grpc.StatusCode.UNAVAILABLE:
                    raise RuntimeError(
                        f"Local StoreDaemon ({self.server_address}) is not available."
                    ) from e
                if code == grpc.StatusCode.INVALID_ARGUMENT:
                    raise ValueError(
                        _grpc_message(e, fallback="invalid argument")
                    ) from e
                if code == grpc.StatusCode.FAILED_PRECONDITION:
                    raise RuntimeError(
                        _grpc_message(e, fallback="failed precondition")
                    ) from e
                if code == grpc.StatusCode.RESOURCE_EXHAUSTED:
                    raise MemoryError(
                        _grpc_message(e, fallback="resource exhausted")
                    ) from e
                raise RuntimeError(
                    "RegisterVramRegion failed: "
                    f"{_grpc_message(e, fallback='rpc failed')}"
                ) from e

        expires_at = None
        if resp.HasField("expires_at"):
            expires_at = resp.expires_at.ToDatetime(tzinfo=timezone.utc)
        return VramRegionHandle(
            region_id=str(resp.region_id),
            ttl_ms=int(resp.ttl_ms),
            expires_at=expires_at,
        )

    def unregister_vram_region(
        self,
        region_id: str,
        *,
        session_id: str | None = None,
        force: bool | None = None,
        timeout_s: float = 10.0,
    ) -> bool:
        if not region_id:
            raise ValueError("region_id is required")
        req = store_daemon_pb2.UnregisterVramRegionRequest(
            region_id=region_id,
            owner_pid=int(self._get_effective_pid()),
        )
        if session_id:
            req.session_id = session_id
        if force is not None:
            req.force = bool(force)

        with self._client_span("Client/UnregisterVramRegion") as span:
            set_span_attributes({"tc.region.id": region_id})
            try:
                resp = self._unary_call(
                    self.stub.UnregisterVramRegion,
                    req,
                    timeout=timeout_s,
                    span=span,
                    retries=1,
                )
            except grpc.RpcError as e:
                span.record_exception(e)
                code = e.code()
                if code == grpc.StatusCode.UNAVAILABLE:
                    raise RuntimeError(
                        f"Local StoreDaemon ({self.server_address}) is not available."
                    ) from e
                if code == grpc.StatusCode.NOT_FOUND:
                    return False
                if code == grpc.StatusCode.FAILED_PRECONDITION:
                    raise RuntimeError(
                        _grpc_message(e, fallback="failed precondition")
                    ) from e
                raise RuntimeError(
                    "UnregisterVramRegion failed: "
                    f"{_grpc_message(e, fallback='rpc failed')}"
                ) from e

        return bool(resp.released)

    def deregister_artifact(
        self,
        artifact_id: str,
        *,
        session_id: str | None = None,
        wait_for_drain: bool = True,
        drain_timeout_ms: int | None = None,
        extend_ttl_ms: int | None = None,
        owner_pid: int | None = None,
        device_id: int | None = None,
        byte_space: common_pb2.ByteSpaceRef | None = None,
        release_regions: bool | None = None,
        operation_id: str | None = None,
        timeout_s: float = 60.0,
    ) -> DeregisterArtifactOutcome:
        if not artifact_id:
            raise ValueError("artifact_id is required")
        req = store_daemon_pb2.DeregisterArtifactRequest(
            artifact_id=artifact_id,
            wait_for_drain=bool(wait_for_drain),
        )
        req.owner_pid = (
            int(owner_pid) if owner_pid is not None else int(self._get_effective_pid())
        )
        if session_id:
            req.session_id = session_id
        if drain_timeout_ms is not None:
            req.drain_timeout_ms = int(drain_timeout_ms)
        if extend_ttl_ms is not None:
            req.extend_ttl_ms = int(extend_ttl_ms)
        if device_id is not None:
            req.device_id = int(device_id)
        if byte_space is not None:
            req.byte_space.CopyFrom(byte_space)
        if release_regions is not None:
            req.release_regions = bool(release_regions)
        if operation_id:
            req.operation_id = str(operation_id)

        with self._client_span("Client/DeregisterArtifact") as span:
            set_span_attributes(
                {
                    "tc.artifact.id": artifact_id,
                    "tc.deregister.wait": bool(wait_for_drain),
                }
            )
            try:
                resp = self._unary_call(
                    self.stub.DeregisterArtifact,
                    req,
                    timeout=timeout_s,
                    span=span,
                    retries=0,
                )
            except grpc.RpcError as e:
                span.record_exception(e)
                code = e.code()
                if code == grpc.StatusCode.UNAVAILABLE:
                    raise RuntimeError(
                        f"Local StoreDaemon ({self.server_address}) is not available."
                    ) from e
                if code == grpc.StatusCode.NOT_FOUND:
                    return DeregisterArtifactOutcome(
                        drained=True,
                        removed=False,
                        released_region_ids=(),
                        message=_grpc_message(e, fallback="artifact not found"),
                    )
                if code == grpc.StatusCode.FAILED_PRECONDITION:
                    raise RuntimeError(
                        _grpc_message(e, fallback="failed precondition")
                    ) from e
                if code == grpc.StatusCode.INVALID_ARGUMENT:
                    raise ValueError(
                        _grpc_message(e, fallback="invalid argument")
                    ) from e
                if code == grpc.StatusCode.DEADLINE_EXCEEDED:
                    raise TimeoutError(
                        _grpc_message(e, fallback="deadline exceeded")
                    ) from e
                raise RuntimeError(
                    "DeregisterArtifact failed: "
                    f"{_grpc_message(e, fallback='rpc failed')}"
                ) from e

        return DeregisterArtifactOutcome(
            drained=bool(resp.drained),
            removed=bool(resp.removed),
            released_region_ids=tuple(resp.released_region_ids),
            message=resp.message or None,
        )

    def publish_target_replica(
        self,
        *,
        target_write_token: bytes,
        byte_space: common_pb2.ByteSpaceRef,
        ttl_ms: int | None = None,
        owner_pid: int | None = None,
        operation_id: str | None = None,
        timeout_s: float = 60.0,
    ) -> store_daemon_pb2.PublishTargetReplicaResponse:
        if not target_write_token:
            raise ValueError("target_write_token is required")
        if byte_space is None:
            raise ValueError("byte_space is required")
        req = store_daemon_pb2.PublishTargetReplicaRequest(
            target_write_token=bytes(target_write_token),
            byte_space=byte_space,
        )
        if ttl_ms is not None:
            req.ttl_ms = int(ttl_ms)
        if owner_pid is not None:
            req.owner_pid = int(owner_pid)
        if operation_id:
            req.operation_id = str(operation_id)

        with self._client_span("Client/PublishTargetReplica") as span:
            try:
                resp = self._unary_call(
                    self.stub_v2.PublishTargetReplica,
                    req,
                    timeout=timeout_s,
                    span=span,
                    retries=0,
                )
            except grpc.RpcError as e:  # noqa: BLE001
                span.record_exception(e)
                code = e.code()
                if code == grpc.StatusCode.UNAVAILABLE:
                    raise RuntimeError(
                        f"Local StoreDaemon ({self.server_address}) is not available."
                    ) from e
                raise RuntimeError(
                    "PublishTargetReplica failed: "
                    f"{_grpc_message(e, fallback='rpc failed')}"
                ) from e
        return resp

    def retire_published_replica(
        self,
        *,
        artifact_id: str,
        byte_space: common_pb2.ByteSpaceRef,
        lease_id: str | None = None,
        owner_pid: int | None = None,
        device_id: int | None = None,
        wait_for_drain: bool = True,
        drain_timeout_ms: int | None = None,
        operation_id: str | None = None,
        timeout_s: float = 60.0,
    ) -> store_daemon_pb2.RetirePublishedReplicaResponse:
        if not artifact_id:
            raise ValueError("artifact_id is required")
        if byte_space is None:
            raise ValueError("byte_space is required")
        req = store_daemon_pb2.RetirePublishedReplicaRequest(
            artifact_id=artifact_id,
            byte_space=byte_space,
            wait_for_drain=bool(wait_for_drain),
        )
        if lease_id:
            req.lease_id = str(lease_id)
        if owner_pid is not None:
            req.owner_pid = int(owner_pid)
        if device_id is not None:
            req.device_id = int(device_id)
        if drain_timeout_ms is not None:
            req.drain_timeout_ms = int(drain_timeout_ms)
        if operation_id:
            req.operation_id = str(operation_id)

        with self._client_span("Client/RetirePublishedReplica") as span:
            try:
                resp = self._unary_call(
                    self.stub_v2.RetirePublishedReplica,
                    req,
                    timeout=timeout_s,
                    span=span,
                    retries=0,
                )
            except grpc.RpcError as e:  # noqa: BLE001
                span.record_exception(e)
                code = e.code()
                if code == grpc.StatusCode.UNAVAILABLE:
                    raise RuntimeError(
                        f"Local StoreDaemon ({self.server_address}) is not available."
                    ) from e
                if code == grpc.StatusCode.NOT_FOUND:
                    return store_daemon_pb2.RetirePublishedReplicaResponse(
                        drained=True, removed=False
                    )
                if code == grpc.StatusCode.DEADLINE_EXCEEDED:
                    raise TimeoutError(
                        _grpc_message(e, fallback="deadline exceeded")
                    ) from e
                raise RuntimeError(
                    "RetirePublishedReplica failed: "
                    f"{_grpc_message(e, fallback='rpc failed')}"
                ) from e
        return resp

    def feed_register_artifact_view_chunks(
        self,
        registration_id: str,
        data: bytes | bytearray | memoryview,
        *,
        chunk_bytes: int = 4 * 1024 * 1024,
    ) -> bool:
        mv = memoryview(data)

        def _iter():
            offset = 0
            while offset < mv.nbytes:
                chunk = mv[offset : offset + chunk_bytes]
                req = store_daemon_pb2.FeedRegisterArtifactStreamRequest(
                    registration_id=registration_id
                )
                req.view_chunk.view_offset = int(offset)
                req.view_chunk.data = bytes(chunk)
                yield req
                offset += len(chunk)

        try:
            self._unary_call(
                self.stub.FeedRegisterArtifactStream, _iter(), timeout=30.0, retries=0
            )
            return True
        except grpc.RpcError as e:  # noqa: BLE001
            logger.error(f"FeedRegisterArtifactStream(view) failed: {e}")
            return False

    def keep_alive_registered_artifact(
        self, registration_id: str, ttl_ms: int, epoch: int
    ) -> bool:
        req = store_daemon_pb2.KeepAliveRegisterArtifactRequest(
            registration_id=registration_id,
            ttl_ms=int(ttl_ms),
            epoch=int(epoch),
            owner_pid=int(self._get_effective_pid()),
        )
        try:
            self._unary_call(
                self.stub.KeepAliveRegisterArtifact,
                req,
                timeout=10.0,
                retries=1,
            )
            return True
        except grpc.RpcError as e:
            logger.error(f"KeepAliveRegisterArtifact failed: {e}")
            return False

    def revoke_registered_artifact(
        self, registration_id: str, reason: str = ""
    ) -> bool:
        req = store_daemon_pb2.RevokeRegisteredArtifactRequest(
            registration_id=registration_id, reason=reason
        )
        try:
            self._unary_call(
                self.stub.RevokeRegisteredArtifact, req, timeout=10.0, retries=1
            )
            return True
        except grpc.RpcError as e:
            logger.error(f"RevokeRegisteredArtifact failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Verification helpers
    # ------------------------------------------------------------------

    def wait_artifact_verification(
        self,
        artifact_identifier: str,
        replica_uuid: str,
        timeout_ms: int = 30000,
    ) -> store_daemon_pb2.WaitReplicaVerificationResponse | None:
        """Block until the daemon returns a PASSED/FAILED status or timeout."""

        request = store_daemon_pb2.WaitReplicaVerificationRequest(
            artifact_id=artifact_identifier,
            replica_uuid=replica_uuid,
            timeout_ms=timeout_ms,
        )

        with self._client_span("Client/WaitReplicaVerification") as span:
            set_span_attributes(
                {
                    "tc.artifact.id": artifact_identifier,
                    "tc.timeout.ms": int(timeout_ms),
                }
            )
            try:
                response = self._unary_call(
                    self.stub.WaitReplicaVerification,
                    request,
                    timeout=timeout_ms / 1000 + 5,
                    span=span,
                    retries=1,
                )
                return response
            except grpc.RpcError as e:
                span.record_exception(e)
                logger.error(f"wait_artifact_verification RPC failed: {e}")
                return None

    # ------------------------------------------------------------------
    # Key mapping publish via daemon
    # ------------------------------------------------------------------

    def publish_replica_key(
        self,
        *,
        key: str,
        descriptor: "ArtifactDescriptor",
        disk_path: str = "",
        fail_if_exists: bool = True,
        timeout_s: float = 5.0,
    ) -> bool:
        from tensorcast.proto.common.v1 import common_pb2 as common_pb2

        req = store_daemon_pb2.PublishReplicaKeyRequest(
            key=key,
            disk_path=disk_path,
            fail_if_exists=bool(fail_if_exists),
        )
        # Map our typed descriptor into proto
        pb = common_pb2.ArtifactDescriptor(
            artifact_id=descriptor.artifact_id,
            index_multihash=descriptor.index_multihash or "",
            data_multihash=descriptor.data_multihash or "",
            schema_version=descriptor.schema_version or "",
            encoding=descriptor.encoding or "",
            total_size=int(descriptor.total_size),
        )
        pb.id_kind = (
            common_pb2.ArtifactIdKind.ARTIFACT_ID_KIND_CGID
            if descriptor.id_kind is ArtifactIdKind.CGID
            else common_pb2.ArtifactIdKind.ARTIFACT_ID_KIND_MI2
        )
        req.artifact_descriptor.CopyFrom(pb)

        with self._client_span("Client/PublishReplicaKey") as span:
            try:
                resp = self._unary_call(
                    self.stub.PublishReplicaKey,
                    req,
                    timeout=timeout_s,
                    span=span,
                    retries=1,
                )
                if not resp.ok:
                    reason = resp.conflict_reason or "key already mapped"
                    logger.warning("PublishReplicaKey refused key %s: %s", key, reason)
                    return False
                return True
            except grpc.RpcError as e:
                logger.error(f"PublishReplicaKey failed: {e}")
                raise

    # ------------------------------------------------------------------
    # Key mapping helpers to keep API layer decoupled from Global Store
    # ------------------------------------------------------------------

    def resolve_key_mapping(
        self, key: str, *, timeout_s: float = 10.0
    ) -> KeyMappingResolution:
        """Resolve a human-friendly key via daemon."""
        if not key:
            raise ValueError("key is required")
        with self._client_span("Client/ResolveKeyMapping") as span:
            request = store_daemon_pb2.ResolveKeyMappingRequest(key=key)
            resp = self._unary_call(
                self.stub.ResolveKeyMapping,
                request,
                timeout=timeout_s,
                span=span,
                retries=1,
            )
            return KeyMappingResolution(
                artifact_id=resp.artifact_id or "",
                used_disk_path=resp.used_disk_path or "",
                generation=int(getattr(resp, "generation", 0) or 0),
                cache_ttl_seconds=int(getattr(resp, "cache_ttl_seconds", 0) or 0),
            )

    def swap_key_mapping(
        self,
        *,
        key: str,
        new_artifact_id: str,
        expected_artifact_id: str | None = None,
        expected_generation: int | None = None,
        operation_id: str | None = None,
        timeout_s: float = 10.0,
    ) -> SwapKeyMappingResult:
        if not key:
            raise ValueError("key is required")
        if not new_artifact_id:
            raise ValueError("new_artifact_id is required")
        req = store_daemon_pb2.SwapKeyMappingRequest(
            key=key,
            new_artifact_id=new_artifact_id,
        )
        if expected_artifact_id:
            req.expected_artifact_id = str(expected_artifact_id)
        if expected_generation is not None:
            req.expected_generation = int(expected_generation)
        if operation_id:
            req.operation_id = str(operation_id)

        with self._client_span("Client/SwapKeyMapping") as span:
            resp = self._unary_call(
                self.stub.SwapKeyMapping,
                req,
                timeout=timeout_s,
                span=span,
                retries=1,
            )
            return SwapKeyMappingResult(
                ok=bool(getattr(resp, "ok", False)),
                artifact_id=getattr(resp, "artifact_id", "") or "",
                generation=int(getattr(resp, "generation", 0) or 0),
            )

    def get_artifact_index_by_id(
        self, artifact_id: str, *, timeout_s: float = 10.0
    ) -> bytes:
        """Fetch canonical tensor index bytes by artifact_id via daemon."""
        if not artifact_id:
            raise ValueError("artifact_id is required")
        with self._client_span("Client/GetArtifactIndexById") as span:
            request = store_daemon_pb2.GetArtifactIndexByIdRequest(
                artifact_id=artifact_id
            )
            resp = self._unary_call(
                self.stub.GetArtifactIndexById,
                request,
                timeout=timeout_s,
                span=span,
                retries=1,
            )
            return resp.tensor_index_data

    def seal_assembly(
        self,
        assembly_id: str,
        *,
        publish_canonical: bool = True,
        timeout_s: float = 120.0,
    ) -> SealAssemblyResult:
        if not assembly_id:
            raise ValueError("assembly_id is required")
        req = store_daemon_pb2.SealAssemblyRequest(
            assembly_id=assembly_id,
            publish_canonical=bool(publish_canonical),
        )
        with self._client_span("Client/SealAssembly") as span:
            try:
                resp = self._unary_call(
                    self.stub.SealAssembly,
                    req,
                    timeout=timeout_s,
                    span=span,
                    retries=0,
                )
            except grpc.RpcError as e:
                span.record_exception(e)
                code = e.code()
                if code == grpc.StatusCode.UNAVAILABLE:
                    raise RuntimeError(
                        f"Local StoreDaemon ({self.server_address}) is not available."
                    ) from e
                if code == grpc.StatusCode.INVALID_ARGUMENT:
                    raise ValueError(
                        _grpc_message(e, fallback="invalid argument")
                    ) from e
                if code == grpc.StatusCode.NOT_FOUND:
                    raise KeyError(_grpc_message(e, fallback="not found")) from e
                if code == grpc.StatusCode.DEADLINE_EXCEEDED:
                    raise TimeoutError(
                        _grpc_message(e, fallback="deadline exceeded")
                    ) from e
                raise RuntimeError(
                    f"SealAssembly failed: {_grpc_message(e, fallback='rpc failed')}"
                ) from e

        desc = resp.descriptor
        ad = ArtifactDescriptor(
            artifact_id=desc.artifact_id,
            index_multihash=desc.index_multihash,
            data_multihash=desc.data_multihash,
            schema_version=desc.schema_version,
            encoding=desc.encoding,
            total_size=int(desc.total_size),
            id_kind=desc.id_kind,
        )
        sealed_artifact_id = resp.sealed_artifact_id or ad.artifact_id
        return SealAssemblyResult(
            sealed_artifact_id=sealed_artifact_id,
            descriptor=ad,
            already_sealed=bool(resp.already_sealed),
        )

    def start_seal_assembly(
        self,
        *,
        assembly_id: str,
        layout_id: str | None = None,
        timeout_s: float = 10.0,
    ) -> store_daemon_pb2.StartSealAssemblyResponse:
        if not assembly_id:
            raise ValueError("assembly_id is required")
        req = store_daemon_pb2.StartSealAssemblyRequest(
            assembly_id=assembly_id,
            layout_id=str(layout_id) if layout_id else "",
        )
        with self._client_span("Client/StartSealAssembly") as span:
            resp = self._unary_call(
                self.stub.StartSealAssembly,
                req,
                timeout=timeout_s,
                span=span,
                retries=0,
            )
        return resp

    def get_operation(
        self, operation_id: str, *, timeout_s: float = 10.0
    ) -> "operation_pb2.GetOperationResponse":
        if not operation_id:
            raise ValueError("operation_id is required")
        from tensorcast.proto.operation.v1 import operation_pb2

        req = operation_pb2.GetOperationRequest(operation_id=operation_id)
        with self._client_span("Client/GetOperation") as span:
            return self._unary_call(
                self.stub.GetOperation,
                req,
                timeout=timeout_s,
                span=span,
                retries=1,
            )

    def wait_operation(
        self,
        operation_id: str,
        *,
        timeout_ms: int,
        timeout_s: float,
    ) -> "operation_pb2.GetOperationResponse":
        if not operation_id:
            raise ValueError("operation_id is required")
        if timeout_ms < 0:
            raise ValueError("timeout_ms must be >= 0")
        from tensorcast.proto.operation.v1 import operation_pb2

        req = store_daemon_pb2.WaitOperationRequest(
            operation_id=operation_id,
            timeout_ms=int(timeout_ms),
        )
        with self._client_span("Client/WaitOperation") as span:
            resp = self._unary_call(
                self.stub.WaitOperation,
                req,
                timeout=timeout_s,
                span=span,
                retries=0,
            )
        # WaitOperationResponse wraps the canonical GetOperationResponse proto.
        op = operation_pb2.GetOperationResponse()
        op.CopyFrom(resp.operation)
        return op

    @classmethod
    def _policy_to_proto(
        cls, policy: StorePolicy | dict[str, object] | str | None
    ) -> store_daemon_pb2.StorePolicy | None:
        if policy is None:
            return None
        from tensorcast.api._config import StorePolicy

        resolved = StorePolicy.parse(policy)
        if resolved is None:
            return None
        return resolved.to_proto()

    @classmethod
    def _persistence_state_from_proto(
        cls, state: store_daemon_pb2.PersistenceState
    ) -> str:
        return cls._PERSISTENCE_STATE_FROM_PROTO.get(state, "unknown")

    def start_persistence(
        self,
        *,
        artifact_id: str,
        policy: StorePolicy | dict[str, object] | str | None = None,
        timeout_s: float = 10.0,
    ) -> store_daemon_pb2.StartPersistenceResponse:
        if not artifact_id:
            raise ValueError("artifact_id is required")
        req = store_daemon_pb2.StartPersistenceRequest(artifact_id=artifact_id)
        policy_proto = self._policy_to_proto(policy)
        if policy_proto is not None:
            req.policy.CopyFrom(policy_proto)
        with self._client_span("Client/StartPersistence") as span:
            return self._unary_call(
                self.stub.StartPersistence, req, timeout=timeout_s, span=span, retries=0
            )

    def query_persistence_status(
        self,
        *,
        task_id: str | None = None,
        artifact_id: str | None = None,
        timeout_s: float = 10.0,
    ) -> store_daemon_pb2.QueryPersistenceStatusResponse:
        if not task_id and not artifact_id:
            raise ValueError("task_id or artifact_id is required")
        req = store_daemon_pb2.QueryPersistenceStatusRequest()
        if task_id:
            req.task_id = task_id
        if artifact_id:
            req.artifact_id = artifact_id
        with self._client_span("Client/QueryPersistenceStatus") as span:
            return self._unary_call(
                self.stub.QueryPersistenceStatus,
                req,
                timeout=timeout_s,
                span=span,
                retries=0,
            )

    def ping(self, timeout_s: float = 2.0) -> bool:
        try:
            _ = self.get_server_config()
            return True
        except Exception:
            return False


# -----------------------------------------------------------------------------
# Shared client cache
# -----------------------------------------------------------------------------
_CLIENT_LOCK: RLock = RLock()
_CLIENT_INSTANCE: DaemonCtl | None = None
_CLIENT_ADDRESS: str | None = None


def get_daemon_client(server_address: str = "127.0.0.1:8073") -> DaemonCtl:
    """Get or create the singleton DaemonCtl for the current process.

    All TensorCast API calls share a single client bound to the daemon address
    established during `tensorcast.startup.init()`. Requesting a different
    address after initialization is a programmer error and will raise.
    """
    global _CLIENT_INSTANCE, _CLIENT_ADDRESS
    with _CLIENT_LOCK:
        if _CLIENT_INSTANCE is None:
            _CLIENT_INSTANCE = DaemonCtl(server_address)
            _CLIENT_ADDRESS = server_address
            return _CLIENT_INSTANCE
        assert _CLIENT_ADDRESS is not None
        if server_address != _CLIENT_ADDRESS:
            raise RuntimeError(
                "TensorCast client already initialized for address "
                f"{_CLIENT_ADDRESS}; refusing to create a second client for {server_address}."
            )
        return _CLIENT_INSTANCE


def _shutdown_daemon_clients() -> None:
    global _CLIENT_INSTANCE, _CLIENT_ADDRESS
    with _CLIENT_LOCK:
        client = _CLIENT_INSTANCE
        _CLIENT_INSTANCE = None
        _CLIENT_ADDRESS = None
    if client is not None:
        with suppress(Exception):
            client.close()


def release_daemon_client(server_address: str) -> None:
    """Close and drop the singleton DaemonCtl if it matches `server_address`."""

    global _CLIENT_INSTANCE, _CLIENT_ADDRESS
    with _CLIENT_LOCK:
        if _CLIENT_INSTANCE is None:
            return
        if _CLIENT_ADDRESS != server_address:
            raise RuntimeError(
                "Attempted to release TensorCast client bound to "
                f"{_CLIENT_ADDRESS}; expected {server_address}."
            )
        client = _CLIENT_INSTANCE
        _CLIENT_INSTANCE = None
        _CLIENT_ADDRESS = None
    with suppress(Exception):
        client.close()


atexit.register(_shutdown_daemon_clients)
