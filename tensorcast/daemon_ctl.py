#  Copyright (c) 2025, TensorCast Team.


import atexit
import os
import random
import time
from contextlib import contextmanager, suppress
from threading import RLock
from typing import Any, Iterator, Tuple, cast

import grpc
from opentelemetry import trace
from opentelemetry.trace import SpanKind

from tensorcast.logger import init_logger
from tensorcast.observability.otel import ensure_client_otel, set_span_attributes

# Use v1 daemon proto path
from tensorcast.proto.daemon.v1 import (
    store_daemon_pb2 as store_daemon_pb2,
)
from tensorcast.proto.daemon.v1 import (
    store_daemon_pb2_grpc as store_daemon_pb2_grpc,
)
from tensorcast.types import (
    ArtifactDescriptor,
    BeginRegisterArtifactResult,
    CanonicalRange,
    CoalescedHandshake,
    CommitResult,
    LeaseHandshake,
    LeaseSegment,
    Plan,
    RegisterStorage,
    RegisterTensorAlias,
    ServerConfig,
)

logger = init_logger(__name__)

# -----------------------------------------------------------------------------
# Client-side diagnostics (process-scoped)
# -----------------------------------------------------------------------------
_METRICS_LOCK: RLock = RLock()
_METRIC_CHANNEL_REFRESHES: int = 0
_METRIC_RPC_RETRIES: int = 0


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
    def __init__(self, server_address="127.0.0.1:8073"):
        # SDK-library safe: ensure OTel is active or ask app to init. No downgrade.
        ensure_client_otel("tensorcast-client", role="client")

        self.server_address = server_address
        self._ch_lock: RLock = RLock()
        self.channel = self._create_channel(server_address)
        self.stub = store_daemon_pb2_grpc.StoreDaemonServiceStub(self.channel)
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
        # Best-effort channel cleanup
        with suppress(Exception):
            self.close()

    def close(self) -> None:
        """Close underlying gRPC channel."""
        with suppress(Exception):
            self.channel.close()

    def __enter__(self) -> "DaemonCtl":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _host_port(self) -> Tuple[str, int]:
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
                    raise RuntimeError(f"Artifact not loaded {e}") from e
                elif e.code() == grpc.StatusCode.UNAVAILABLE:
                    raise RuntimeError(
                        f"Local StoreDaemon ({self.server_address}) is not available."
                    ) from e
                else:
                    raise RuntimeError(f"Error: {e}") from e

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
        view: store_daemon_pb2.ViewSpec | None = None,
        view_id: str | None = None,
        placement: store_daemon_pb2.TransformPlacement | None = None,
        return_response: bool = False,
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
                    raise RuntimeError(f"Artifact not loaded {e}") from e
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    raise RuntimeError(
                        f"Local StoreDaemon ({self.server_address}) is not available."
                    ) from e
                raise RuntimeError(f"Error: {e}") from e

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
            success = self.confirm_replica_loaded("", replica_uuid)
            if not success:
                raise RuntimeError(
                    f"Failed to confirm artifact loading for {artifact_id}"
                )

        if return_response:
            return response

        assert response.mem_handle is not None
        return response.mem_handle.cuda_ipc_handle

    def materialize_by_key(
        self,
        key: str,
        replica_uuid: str,
        device_id: int,
        pinned_allocation_timeout_ms: int = int(30e3),
        wait_for_completion: bool = True,
    ):
        """Materialize a replica by RFC-0014 key via daemon.

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
                detail_msg = str(e)
                with suppress(Exception):
                    if code is not None:
                        status_name = code.name
                    detail_msg = e.details() or detail_msg
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

        if not wait_for_completion:
            assert response.mem_handle is not None
            return (
                response.mem_handle.cuda_ipc_handle,
                load_status,
                used_disk_path,
                artifact_id,
            )

        # Synchronous path: confirm completion using the session created with replica_uuid
        if (
            load_status
            == store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_ALLOCATED
        ):
            success = self.confirm_replica_loaded("", replica_uuid)
            if not success:
                raise RuntimeError(f"Failed to confirm artifact loading for key={key}")

        assert response.mem_handle is not None
        return (
            response.mem_handle.cuda_ipc_handle,
            used_disk_path,
            artifact_id,
        )

    def confirm_replica_loaded(self, disk_path: str, replica_uuid: str) -> bool:
        with self._client_span("Client/ConfirmReplica") as span:
            request = store_daemon_pb2.ConfirmReplicaRequest(
                disk_path=disk_path,
                replica_uuid=replica_uuid,
                target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
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
                return ServerConfig(
                    tx_slice_bytes=int(getattr(response, "tx_slice_bytes", 0)),
                    mem_pool_size=int(getattr(response, "mem_pool_size", 0)),
                    artifact_chunk_bytes=int(
                        getattr(response, "artifact_chunk_bytes", 0)
                    ),
                )

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
        schema_version: str = "v2",
        plan: Plan | None = None,
        timeout_s: float = 30.0,
        view: store_daemon_pb2.ViewRegistrationOptions | None = None,
    ) -> BeginRegisterArtifactResult:
        """Begin unified artifact registration (RFC-0014).

        Args:
            device_id: target device ordinal (single-GPU invariant per RFC-0014).
            total_size_bytes: AVBS total size (8B aligned).
            ttl_ms: optional TTL used by Lease/UMA/VS plans.
            tensor_index_key: optional hex key of canonical index.
            tensor_index_data: optional canonical index bytes (preferred).
            encoding: "json" or "cbor" for index bytes.
            schema_version: index schema version (e.g., "v2").
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

        # Plan oneof
        if plan is None:
            from tensorcast.types import CoalescedPlan as _DefaultPlan

            plan = _DefaultPlan()
        kind = plan.kind
        plan.apply_to_begin_request(req)
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
                    raise ValueError(str(e)) from e
                if code == grpc.StatusCode.RESOURCE_EXHAUSTED:
                    raise MemoryError(str(e)) from e
                if code == grpc.StatusCode.DEADLINE_EXCEEDED:
                    raise TimeoutError(str(e)) from e
                raise RuntimeError(f"BeginRegisterArtifact failed: {e}") from e

            # Build typed handshake result
            if resp.HasField("coalesced"):
                handshake = CoalescedHandshake(
                    daemon_ipc_handle=bytes(resp.coalesced.daemon_ipc_handle)
                )
            elif resp.HasField("lease"):
                handshake = LeaseHandshake()
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
                    raise ValueError(str(e)) from e
                if code == grpc.StatusCode.NOT_FOUND:
                    raise KeyError(str(e)) from e
                if code == grpc.StatusCode.DEADLINE_EXCEEDED:
                    raise TimeoutError(str(e)) from e
                raise RuntimeError(f"CommitRegisteredArtifact failed: {e}") from e

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
            return CommitResult(
                descriptor=ad,
                existed=existed,
                view_id=view_id,
                view_index_json=view_index_json,
                view_data_hash=view_data_hash,
                canonical_ranges=canonical_ranges,
                allow_partial=bool(resp.allow_partial),
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
                    raise ValueError(str(e)) from e
                if code == grpc.StatusCode.NOT_FOUND:
                    # Treat as already-aborted/missing
                    logger.warning(
                        "AbortRegisteredArtifact: registration not found: %s",
                        registration_id,
                    )
                    return False
                raise RuntimeError(f"AbortRegisteredArtifact failed: {e}") from e

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
                seg.device_id = int(s.device_id)
                seg.cuda_ipc_handle = s.cuda_ipc_handle
                seg.base_addr = int(s.base_addr)
                seg.length = int(s.length)
                seg.dst_offset = int(s.dst_offset)
            if storages:
                for storage in storages:
                    entry = req.storage_entries.add()
                    entry.storage_id = storage.storage_id
                    entry.device_id = int(storage.device_id)
                    entry.cuda_ipc_handle = storage.cuda_ipc_handle
                    entry.storage_length = int(storage.storage_length)
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
    # RFC-0014: Key mapping publish via daemon
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
            index_multihash=descriptor.index_multihash,
            data_multihash=descriptor.data_multihash,
            schema_version=descriptor.schema_version,
            encoding=descriptor.encoding,
            total_size=int(descriptor.total_size),
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
                return bool(resp.ok)
            except grpc.RpcError as e:
                logger.error(f"PublishReplicaKey failed: {e}")
                return False

    # ------------------------------------------------------------------
    # RFC-0014 helpers to keep API layer decoupled from Global Store
    # ------------------------------------------------------------------

    def resolve_key_mapping(
        self, key: str, *, timeout_s: float = 10.0
    ) -> tuple[str, str]:
        """Resolve a human-friendly key to (artifact_id, used_disk_path) via daemon.

        Returns a tuple (artifact_id, used_disk_path). Raises on RPC errors.
        """
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
            return resp.artifact_id, resp.used_disk_path

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
