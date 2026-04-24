#  Copyright (c) 2025-2026, TensorCast Team.
"""Client utilities for interacting with the TensorCast Store Daemon."""

from __future__ import annotations

import array
import atexit
import fcntl
import os
import random
import socket
import struct
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import timezone
from functools import lru_cache
from threading import RLock
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    NoReturn,
    cast,
    overload,
)

import grpc
from opentelemetry import trace
from opentelemetry.trace import SpanKind

if TYPE_CHECKING:
    import torch

    from tensorcast.api._config import StorePolicy
    from tensorcast.proto.node_agent.v1 import node_agent_pb2
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
from tensorcast.proto.node_agent.v1 import node_agent_pb2
from tensorcast.proto.operation.v1 import operation_pb2
from tensorcast.types import (
    ArtifactDescriptor,
    ArtifactIdKind,
    AssemblyAttemptRef,
    AssemblyCloseoutContract,
    AssemblyReadinessPolicy,
    AssemblyRequirementSetRef,
    BeginRegisterArtifactResult,
    CanonicalRange,
    CoalescedHandshake,
    CommitResult,
    DeregisterArtifactOutcome,
    HostSharedRegionAttachment,
    HostSharedRegionClass,
    LeaseHandshake,
    LeaseSegment,
    LocalRegionHandle,
    LocalStableTierResult,
    Plan,
    RegionMemoryKind,
    RegisterStorage,
    RegisterTensorAlias,
    RepresentationPublishSpec,
    SealAssemblyResult,
    ServerConfig,
    ServingRuntimePolicy,
    StableDramHandshake,
    VramRegionHandle,
)

logger = init_logger(__name__)

# Raise message limits for large stable_dram CPU-stream uploads.
_DEFAULT_GRPC_MAX_MESSAGE_BYTES = 64 * 1024 * 1024
_DEFAULT_FEED_VIEW_CHUNK_BYTES = 4 * 1024 * 1024
_FEED_VIEW_CHUNK_HEADROOM_BYTES = 64 * 1024
_DEFAULT_FEED_PROGRESS_LOG_INTERVAL_BYTES = 0

# -----------------------------------------------------------------------------
# Client-side diagnostics (process-scoped)
# -----------------------------------------------------------------------------
_METRICS_LOCK: RLock = RLock()
_METRIC_CHANNEL_REFRESHES: int = 0
_METRIC_RPC_RETRIES: int = 0
_LOCAL_HANDLE_RESP_LABELS: dict[int, str] = {
    0: "ok",
    1: "not_found",
    2: "failed_precondition",
    3: "permission_denied",
    4: "internal",
}


@dataclass(frozen=True, slots=True)
class KeyMappingResolution:
    artifact_id: str
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


def _normalize_source_policy(
    source_policy: store_daemon_pb2.SourcePolicy | None = None,
) -> store_daemon_pb2.SourcePolicy:
    resolved = store_daemon_pb2.SourcePolicy()
    if source_policy is not None:
        resolved.CopyFrom(source_policy)
    if (
        resolved.preference
        == store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_UNSPECIFIED
    ):
        resolved.preference = store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_AUTO
    return resolved


def _parse_env_float(name: str, default: float, *, min_value: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; using default=%s", name, raw, default)
        return default
    if value < min_value:
        logger.warning(
            "Invalid %s=%r; expected >= %s, using default=%s",
            name,
            raw,
            min_value,
            default,
        )
        return default
    return value


def _parse_env_int(name: str, default: int, *, min_value: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; using default=%s", name, raw, default)
        return default
    if value < min_value:
        logger.warning(
            "Invalid %s=%r; expected >= %s, using default=%s",
            name,
            raw,
            min_value,
            default,
        )
        return default
    return value


@lru_cache(maxsize=1)
def _grpc_max_send_message_bytes() -> int:
    return _parse_env_int(
        "TENSORCAST_GRPC_MAX_SEND_MESSAGE_BYTES",
        default=_DEFAULT_GRPC_MAX_MESSAGE_BYTES,
        min_value=1024 * 1024,
    )


@lru_cache(maxsize=1)
def _grpc_max_receive_message_bytes() -> int:
    return _parse_env_int(
        "TENSORCAST_GRPC_MAX_RECEIVE_MESSAGE_BYTES",
        default=_DEFAULT_GRPC_MAX_MESSAGE_BYTES,
        min_value=1024 * 1024,
    )


@lru_cache(maxsize=1)
def _feed_view_chunk_safe_max_bytes() -> int:
    configured_chunk = _parse_env_int(
        "TENSORCAST_FEED_VIEW_CHUNK_BYTES",
        default=_DEFAULT_FEED_VIEW_CHUNK_BYTES,
        min_value=1,
    )
    safe_cap = (
        min(
            _grpc_max_send_message_bytes(),
            _grpc_max_receive_message_bytes(),
        )
        - _FEED_VIEW_CHUNK_HEADROOM_BYTES
    )
    if safe_cap <= 0:
        return 1
    return min(configured_chunk, safe_cap)


@lru_cache(maxsize=1)
def _feed_view_stream_timeout_seconds() -> float | None:
    timeout_s = _parse_env_float(
        "TENSORCAST_FEED_VIEW_TIMEOUT_SECONDS",
        default=0.0,
        min_value=0.0,
    )
    if timeout_s == 0.0:
        return None
    return timeout_s


@lru_cache(maxsize=1)
def _feed_view_progress_log_interval_bytes() -> int:
    return _parse_env_int(
        "TENSORCAST_FEED_PROGRESS_LOG_INTERVAL_BYTES",
        default=_DEFAULT_FEED_PROGRESS_LOG_INTERVAL_BYTES,
        min_value=0,
    )


@lru_cache(maxsize=1)
def _import_artifact_from_path_timeout_seconds() -> float | None:
    # ImportArtifactFromPath can scan and hash very large directories.
    timeout_s = _parse_env_float(
        "TENSORCAST_IMPORT_ARTIFACT_TIMEOUT_SECONDS",
        default=600.0,
        min_value=0.0,
    )
    # 0 means "no RPC deadline" (timeout=None in grpc Python API).
    if timeout_s == 0.0:
        return None
    return timeout_s


@lru_cache(maxsize=1)
def _import_artifact_from_path_retries() -> int:
    # Avoid re-running expensive disk scans by default.
    return _parse_env_int(
        "TENSORCAST_IMPORT_ARTIFACT_RETRIES",
        default=0,
        min_value=0,
    )


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


def _raise_import_artifact_from_path_rpc_error(
    server_address: str, err: grpc.RpcError
) -> NoReturn:
    code = err.code()
    if code == grpc.StatusCode.UNAVAILABLE:
        _raise_grpc_error(
            RuntimeError(
                f"Local StoreDaemon ({server_address}) is not available. Msg: {err.details()}"
            ),
            cause=err,
        )
    if code == grpc.StatusCode.INVALID_ARGUMENT:
        _raise_grpc_error(
            ValueError(_grpc_message(err, fallback="invalid argument")),
            cause=err,
        )
    if code == grpc.StatusCode.NOT_FOUND:
        _raise_grpc_error(
            FileNotFoundError(
                _grpc_message(err, fallback="artifact not found on disk")
            ),
            cause=err,
        )
    if code == grpc.StatusCode.PERMISSION_DENIED:
        _raise_grpc_error(
            PermissionError(_grpc_message(err, fallback="import path not permitted")),
            cause=err,
        )
    _raise_grpc_error(
        RuntimeError(
            "ImportArtifactFromPathV2 RPC failed: "
            f"{_grpc_message(err, fallback='rpc failed')}"
        ),
        cause=err,
    )


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
        # Long-running unary RPCs (e.g., full materialization) can legitimately run
        # for several minutes. Use conservative ping defaults to avoid transport
        # resets caused by overly aggressive keepalive probes.
        keepalive_time_ms = _parse_env_int(
            "TENSORCAST_GRPC_KEEPALIVE_TIME_MS",
            default=600_000,
            min_value=1_000,
        )
        keepalive_timeout_ms = _parse_env_int(
            "TENSORCAST_GRPC_KEEPALIVE_TIMEOUT_MS",
            default=20_000,
            min_value=1_000,
        )
        min_time_between_pings_ms = _parse_env_int(
            "TENSORCAST_GRPC_MIN_TIME_BETWEEN_PINGS_MS",
            default=600_000,
            min_value=1_000,
        )
        min_ping_interval_without_data_ms = _parse_env_int(
            "TENSORCAST_GRPC_MIN_PING_INTERVAL_WITHOUT_DATA_MS",
            default=600_000,
            min_value=1_000,
        )
        max_pings_without_data = _parse_env_int(
            "TENSORCAST_GRPC_MAX_PINGS_WITHOUT_DATA",
            default=0,
            min_value=0,
        )
        keepalive_permit_without_calls = _parse_env_int(
            "TENSORCAST_GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS",
            default=1,
            min_value=0,
        )
        max_send_message_length = _grpc_max_send_message_bytes()
        max_receive_message_length = _grpc_max_receive_message_bytes()
        return [
            ("grpc.max_send_message_length", max_send_message_length),
            ("grpc.max_receive_message_length", max_receive_message_length),
            ("grpc.keepalive_time_ms", keepalive_time_ms),
            ("grpc.keepalive_timeout_ms", keepalive_timeout_ms),
            ("grpc.keepalive_permit_without_calls", keepalive_permit_without_calls),
            ("grpc.http2.min_time_between_pings_ms", min_time_between_pings_ms),
            ("grpc.http2.max_pings_without_data", max_pings_without_data),
            (
                "grpc.http2.min_ping_interval_without_data_ms",
                min_ping_interval_without_data_ms,
            ),
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
        total_start = time.monotonic()
        for attempt in range(retries + 1):
            call_timeout: float | int | None = timeout
            if timeout is not None:
                total_budget_s = max(0.0, float(timeout))
                elapsed_total_s = time.monotonic() - total_start
                remaining_s = total_budget_s - elapsed_total_s
                if remaining_s <= 0.0:
                    logger.warning(
                        "client_rpc_budget_exhausted method=%s addr=%s attempts=%d total_budget_s=%.3f elapsed_total_s=%.3f",
                        method_path or resolved_name or "<callable>",
                        self.server_address,
                        int(attempt),
                        total_budget_s,
                        elapsed_total_s,
                    )
                    break
                call_timeout = max(0.001, remaining_s)
            if attempt > 0 and resolved_name:
                # Rebind the method on the (potentially) refreshed stub
                reb = getattr(self.stub, resolved_name, None)
                if reb is not None:
                    cur_method = reb
            attempt_start = time.monotonic()
            try:
                response = cur_method(request, timeout=call_timeout)
                elapsed_attempt_s = time.monotonic() - attempt_start
                if elapsed_attempt_s >= 30.0:
                    logger.info(
                        "client_rpc_slow method=%s addr=%s attempt=%d timeout_s=%.3f elapsed_s=%.3f",
                        method_path or resolved_name or "<callable>",
                        self.server_address,
                        int(attempt),
                        float(call_timeout) if call_timeout is not None else -1.0,
                        elapsed_attempt_s,
                    )
                return response
            except grpc.RpcError as e:  # noqa: BLE001
                last_err = e
                code = e.code()
                elapsed_attempt_s = time.monotonic() - attempt_start
                elapsed_total_s = time.monotonic() - total_start
                if span is not None:
                    with suppress(Exception):
                        span.record_exception(e)
                        span.set_attribute("rpc.grpc.status_code", str(code.name))
                        span.set_attribute("retry.attempt", int(attempt))
                # Retry on transient errors
                should_retry = (
                    code
                    in (
                        grpc.StatusCode.UNAVAILABLE,
                        grpc.StatusCode.INTERNAL,
                        grpc.StatusCode.UNKNOWN,
                        grpc.StatusCode.DEADLINE_EXCEEDED,
                    )
                    and attempt < retries
                )
                if should_retry:
                    # Keep the whole unary call bounded by the original timeout
                    # instead of granting each retry a full timeout window.
                    if timeout is not None:
                        total_budget_s = max(0.0, float(timeout))
                        remaining_after_failure_s = total_budget_s - elapsed_total_s
                        if remaining_after_failure_s <= 0.0:
                            logger.warning(
                                "client_rpc_retry_suppressed method=%s code=%s addr=%s attempt=%d elapsed_attempt_s=%.3f elapsed_total_s=%.3f total_budget_s=%.3f",
                                method_path or resolved_name or "<callable>",
                                str(getattr(code, "name", code)),
                                self.server_address,
                                int(attempt),
                                elapsed_attempt_s,
                                elapsed_total_s,
                                total_budget_s,
                            )
                            break
                    # best-effort method name for logging
                    mname = method_path or resolved_name or "<callable>"
                    _inc_rpc_retry(self.server_address, str(mname), attempt + 1, code)
                    self._refresh_channel()
                    time.sleep(0.05 + random.random() * 0.1)
                    continue
                expected_business_codes = {
                    grpc.StatusCode.NOT_FOUND,
                    grpc.StatusCode.ALREADY_EXISTS,
                    grpc.StatusCode.INVALID_ARGUMENT,
                    grpc.StatusCode.FAILED_PRECONDITION,
                    grpc.StatusCode.PERMISSION_DENIED,
                    grpc.StatusCode.UNAUTHENTICATED,
                }
                log_fn = (
                    logger.debug if code in expected_business_codes else logger.warning
                )
                log_fn(
                    "client_rpc_failed method=%s code=%s addr=%s attempt=%d elapsed_attempt_s=%.3f elapsed_total_s=%.3f retries=%d",
                    method_path or resolved_name or "<callable>",
                    str(getattr(code, "name", code)),
                    self.server_address,
                    int(attempt),
                    elapsed_attempt_s,
                    elapsed_total_s,
                    int(retries),
                )
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

    @staticmethod
    def _ensure_fd_cloexec(fd: int) -> None:
        flags = fcntl.fcntl(fd, fcntl.F_GETFD)
        fcntl.fcntl(fd, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)

    def _local_handle_socket_path(self) -> str:
        config = self.get_server_config()
        if not config.local_handle_socket_path:
            raise RuntimeError("Daemon local_handle_socket_path is missing")
        return config.local_handle_socket_path

    def _request_local_handle_fd(
        self,
        *,
        opcode: int,
        token: bytes,
        local_handle_socket_path: str,
        timeout_s: float,
    ) -> int:
        if not local_handle_socket_path:
            raise RuntimeError("Local handle socket path is required")
        if not token:
            raise RuntimeError("Local handle token is empty")
        if len(token) > 1024:
            raise RuntimeError("Local handle token is too large")

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout_s)
            try:
                sock.connect(local_handle_socket_path)
            except OSError as exc:
                raise RuntimeError(
                    f"Failed to connect LocalHandle socket at {local_handle_socket_path}"
                ) from exc
            try:
                sock.sendall(bytes([opcode]) + struct.pack("=I", len(token)) + token)
            except OSError as exc:
                raise RuntimeError("LocalHandle send failed") from exc

            recv_flags = getattr(socket, "MSG_CMSG_CLOEXEC", 0)
            try:
                data, ancdata, _, _ = sock.recvmsg(
                    1, socket.CMSG_SPACE(struct.calcsize("i")), recv_flags
                )
            except (OSError, TimeoutError) as exc:
                raise RuntimeError("LocalHandle recv failed") from exc
            if not data:
                raise RuntimeError("Local handle server returned empty response")
            code = int(data[0])
            if code != 0:
                label = _LOCAL_HANDLE_RESP_LABELS.get(code, f"unknown({code})")
                raise RuntimeError(f"LocalHandle request failed: {label}")

            recv_fds: list[int] = []
            for level, ctype, cmsg_data in ancdata:
                if level == socket.SOL_SOCKET and ctype == socket.SCM_RIGHTS:
                    fds = array.array("i")
                    fds.frombytes(cmsg_data)
                    recv_fds.extend(int(fd) for fd in fds)
            if not recv_fds:
                raise RuntimeError("LocalHandle returned no file descriptor")

            fd = recv_fds[0]
            for extra_fd in recv_fds[1:]:
                with suppress(OSError):
                    os.close(extra_fd)
            self._ensure_fd_cloexec(fd)
            return fd

    def _release_local_handle_token(
        self,
        *,
        token: bytes,
        local_handle_socket_path: str,
        timeout_s: float,
    ) -> bool:
        if not local_handle_socket_path:
            raise RuntimeError("Local handle socket path is required")
        if not token:
            raise RuntimeError("Local handle token is empty")
        if len(token) > 1024:
            raise RuntimeError("Local handle token is too large")

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout_s)
            try:
                sock.connect(local_handle_socket_path)
            except OSError as exc:
                raise RuntimeError(
                    f"Failed to connect LocalHandle socket at {local_handle_socket_path}"
                ) from exc
            try:
                sock.sendall(bytes([2]) + struct.pack("=I", len(token)) + token)
                response = sock.recv(1)
            except (OSError, TimeoutError) as exc:
                raise RuntimeError("LocalHandle release failed") from exc
            if not response:
                raise RuntimeError("Local handle server returned empty response")
            code = int(response[0])
            if code == 0:
                return True
            label = _LOCAL_HANDLE_RESP_LABELS.get(code, f"unknown({code})")
            raise RuntimeError(f"LocalHandle release failed: {label}")

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
        raise RuntimeError(
            "load_into_gpu is no longer supported; disk-path materialization was removed. "
            "Use Store.from_disk(...) to import and then materialize by artifact_id or key."
        )

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
        source_policy: store_daemon_pb2.SourcePolicy | None = None,
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
            selection = common_pb2.ArtifactSelection(artifact_id=artifact_id)
            if view is not None:
                selection.view_spec.CopyFrom(view)
            elif view_id:
                selection.view_id = view_id
            resolved_source_policy = _normalize_source_policy(source_policy)
            request = store_daemon_pb2.MaterializeReplicaRequest(
                pid=pid,
                selection=selection,
                replica_uuid=replica_uuid,
                device_uuid=device_uuid,
                target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
                pinned_allocation_timeout_ms=pinned_allocation_timeout_ms,
                wait_for_completion=wait_for_completion,
            )
            request.source_policy.CopyFrom(resolved_source_policy)
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
            # Confirm using disk path from the daemon response.
            confirm_timeout_s = 500.0
            success = self.confirm_replica_loaded(
                response.disk_path or "",
                replica_uuid,
                timeout_s=confirm_timeout_s,
            )
            if not success:
                raise RuntimeError(
                    "Failed to confirm artifact loading: "
                    f"artifact_id={artifact_id}, "
                    f"replica_uuid={replica_uuid}, "
                    f"disk_path={response.disk_path or ''}"
                )

        if return_response:
            return response

        assert response.mem_handle is not None
        return response.mem_handle.cuda_ipc_handle

    def materialize_into_target_v2(
        self,
        *,
        selection: common_pb2.ArtifactSelection,
        target_layout: store_daemon_pb2.TargetLayout,
        device_uuid: str,
        source_policy: store_daemon_pb2.SourcePolicy | None = None,
        serving_runtime_policy: "ServingRuntimePolicy | None" = None,
        placement: store_daemon_pb2.TransformPlacement | None = None,
        pid: int | None = None,
        operation_id: str | None = None,
        timeout_s: float = 600.0,
        return_response: bool = True,
    ) -> store_daemon_pb2.MaterializeIntoTargetResponse:
        if not isinstance(selection, common_pb2.ArtifactSelection):
            raise ValueError("selection is required")
        if not selection.artifact_id:
            raise ValueError("selection.artifact_id is required")
        if not device_uuid:
            raise ValueError("device_uuid is required")
        pid_value = self._get_effective_pid() if pid is None else int(pid)
        with self._client_span("Client/MaterializeIntoTarget") as span:
            resolved_source_policy = _normalize_source_policy(source_policy)
            request = store_daemon_pb2.MaterializeIntoTargetRequest(
                selection=selection,
                target_layout=target_layout,
                device_uuid=device_uuid,
                pid=pid_value,
            )
            request.source_policy.CopyFrom(resolved_source_policy)
            if serving_runtime_policy is not None:
                request.serving_artifact_policy.CopyFrom(
                    serving_runtime_policy.to_proto()
                )
            if placement is not None:
                request.placement = placement
            if operation_id:
                request.operation_id = str(operation_id)
            try:
                response: store_daemon_pb2.MaterializeIntoTargetResponse = (
                    self._unary_call(
                        self.stub_v2.MaterializeIntoTarget,
                        request,
                        timeout=float(timeout_s),
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
                        f"Artifact id '{selection.artifact_id}' was not found by StoreDaemon at {self.server_address}."
                    ) from e
                raise RuntimeError(
                    _grpc_message(e, fallback="MaterializeIntoTargetV2 RPC failed")
                ) from e
        if not return_response:
            raise RuntimeError(
                "materialize_into_target_v2 requires return_response=True"
            )
        return response

    def materialize_into_mapped_target(
        self,
        *,
        selection: common_pb2.ArtifactSelection,
        target_layout: store_daemon_pb2.TargetLayout,
        device_uuid: str,
        copy_plan,
        dst_tensors: Mapping[str, torch.Tensor],
        source_policy: store_daemon_pb2.SourcePolicy | None = None,
        serving_runtime_policy: "ServingRuntimePolicy | None" = None,
        placement: store_daemon_pb2.TransformPlacement | None = None,
        pid: int | None = None,
        operation_id: str | None = None,
        timeout_s: float = 600.0,
        return_response: bool = True,
    ) -> store_daemon_pb2.MaterializeIntoTargetResponse:
        from tensorcast.api.store.mapped_binding import normalize_copy_plan

        if not isinstance(selection, common_pb2.ArtifactSelection):
            raise ValueError("selection is required")
        if not selection.artifact_id:
            raise ValueError("selection.artifact_id is required")
        if not device_uuid:
            raise ValueError("device_uuid is required")
        if not isinstance(dst_tensors, Mapping) or not dst_tensors:
            raise ValueError("dst_tensors must be a non-empty mapping")
        normalized_plan = normalize_copy_plan(copy_plan)
        pid_value = self._get_effective_pid() if pid is None else int(pid)
        with self._client_span("Client/MaterializeIntoMappedTarget") as span:
            resolved_source_policy = _normalize_source_policy(source_policy)
            request = store_daemon_pb2.MaterializeIntoMappedTargetRequest(
                selection=selection,
                target_layout=target_layout,
                device_uuid=device_uuid,
                pid=pid_value,
            )
            plan_proto = store_daemon_pb2.CopyPlan(version=1)
            for entry in normalized_plan:
                entry_proto = store_daemon_pb2.CopyPlanEntry(
                    ckpt_name=str(entry.ckpt_name),
                    dst_name=str(entry.dst_name),
                )
                if entry.ckpt_range is not None:
                    entry_proto.ckpt_range.dim = int(entry.ckpt_range.dim)
                    entry_proto.ckpt_range.start = int(entry.ckpt_range.start)
                    entry_proto.ckpt_range.end = int(entry.ckpt_range.end)
                if entry.dst_range is not None:
                    entry_proto.dst_range.dim = int(entry.dst_range.dim)
                    entry_proto.dst_range.start = int(entry.dst_range.start)
                    entry_proto.dst_range.end = int(entry.dst_range.end)
                plan_proto.entries.append(entry_proto)
            request.copy_plan.CopyFrom(plan_proto)
            for name, tensor in dst_tensors.items():
                spec = store_daemon_pb2.MappedTensorSpec(
                    name=str(name),
                    dtype=str(tensor.dtype),
                    storage_offset=int(tensor.storage_offset()),
                    logical_length=int(tensor.numel()) * int(tensor.element_size()),
                )
                spec.shape.extend(int(v) for v in tensor.shape)
                spec.stride.extend(int(v) for v in tensor.stride())
                request.dst_tensors.append(spec)
            request.source_policy.CopyFrom(resolved_source_policy)
            if serving_runtime_policy is not None:
                request.serving_artifact_policy.CopyFrom(
                    serving_runtime_policy.to_proto()
                )
            if placement is not None:
                request.placement = placement
            if operation_id:
                request.operation_id = str(operation_id)
            try:
                response: store_daemon_pb2.MaterializeIntoTargetResponse = (
                    self._unary_call(
                        self.stub_v2.MaterializeIntoMappedTarget,
                        request,
                        timeout=float(timeout_s),
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
                if code == grpc.StatusCode.UNIMPLEMENTED:
                    raise RuntimeError(
                        "MaterializeIntoMappedTarget is not supported by the connected StoreDaemon."
                    ) from e
                if code == grpc.StatusCode.NOT_FOUND:
                    raise RuntimeError(
                        f"Artifact id '{selection.artifact_id}' was not found by StoreDaemon at {self.server_address}."
                    ) from e
                raise RuntimeError(
                    _grpc_message(e, fallback="MaterializeIntoMappedTarget RPC failed")
                ) from e
        if not return_response:
            raise RuntimeError(
                "materialize_into_mapped_target requires return_response=True"
            )
        return response

    def create_owned_binding(
        self,
        *,
        source_selection: common_pb2.ArtifactSelection,
        target_layout: store_daemon_pb2.TargetLayout,
        target_index_bytes: bytes,
        device_uuid: str,
        binding_layout_id: str,
        source_policy: store_daemon_pb2.SourcePolicy | None = None,
        serving_runtime_policy: "ServingRuntimePolicy | None" = None,
        placement: store_daemon_pb2.TransformPlacement | None = None,
        copy_plan: store_daemon_pb2.CopyPlan | None = None,
        dst_specs: Iterable[store_daemon_pb2.MappedTensorSpec] | None = None,
        pid: int | None = None,
        operation_id: str | None = None,
        timeout_s: float = 600.0,
    ) -> store_daemon_pb2.CreateOwnedBindingResponse:
        if not isinstance(source_selection, common_pb2.ArtifactSelection):
            raise ValueError("source_selection is required")
        if not source_selection.artifact_id:
            raise ValueError("source_selection.artifact_id is required")
        if not device_uuid:
            raise ValueError("device_uuid is required")
        if not target_index_bytes:
            raise ValueError("target_index_bytes is required")
        if not binding_layout_id:
            raise ValueError("binding_layout_id is required")
        pid_value = self._get_effective_pid() if pid is None else int(pid)
        with self._client_span("Client/CreateOwnedBinding") as span:
            resolved_source_policy = _normalize_source_policy(source_policy)
            request = store_daemon_pb2.CreateOwnedBindingRequest(
                source_selection=source_selection,
                target_layout=target_layout,
                target_index_bytes=bytes(target_index_bytes),
                device_uuid=device_uuid,
                binding_layout_id=str(binding_layout_id),
                pid=pid_value,
            )
            request.source_policy.CopyFrom(resolved_source_policy)
            if serving_runtime_policy is not None:
                request.serving_artifact_policy.CopyFrom(
                    serving_runtime_policy.to_proto()
                )
            if placement is not None:
                request.placement = placement
            if copy_plan is not None:
                request.copy_plan.CopyFrom(copy_plan)
            if dst_specs is not None:
                for spec in dst_specs:
                    request.dst_tensors.add().CopyFrom(spec)
            if operation_id:
                request.operation_id = str(operation_id)
            try:
                response: store_daemon_pb2.CreateOwnedBindingResponse = (
                    self._unary_call(
                        self.stub_v2.CreateOwnedBinding,
                        request,
                        timeout=float(timeout_s),
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
                raise RuntimeError(
                    _grpc_message(e, fallback="CreateOwnedBinding RPC failed")
                ) from e
        return response

    def create_binding(
        self,
        *,
        ownership: store_daemon_pb2.BindingOwnership,
        target_layout: store_daemon_pb2.TargetLayout,
        target_index_bytes: bytes,
        device_uuid: str,
        binding_layout_id: str,
        initial_selection: common_pb2.ArtifactSelection | None = None,
        source_artifact_id: str | None = None,
        target_publication_token: bytes | None = None,
        copy_plan: store_daemon_pb2.CopyPlan | None = None,
        dst_specs: Iterable[store_daemon_pb2.MappedTensorSpec] | None = None,
        pid: int | None = None,
        timeout_s: float = 600.0,
    ) -> store_daemon_pb2.CreateBindingResponse:
        if ownership == store_daemon_pb2.BindingOwnership.BINDING_OWNERSHIP_UNSPECIFIED:
            raise ValueError("ownership is required")
        if not device_uuid:
            raise ValueError("device_uuid is required")
        if not target_index_bytes:
            raise ValueError("target_index_bytes is required")
        if not binding_layout_id:
            raise ValueError("binding_layout_id is required")
        pid_value = self._get_effective_pid() if pid is None else int(pid)
        with self._client_span("Client/CreateBinding") as span:
            request = store_daemon_pb2.CreateBindingRequest(
                ownership=ownership,
                target_layout=target_layout,
                target_index_bytes=bytes(target_index_bytes),
                device_uuid=device_uuid,
                binding_layout_id=str(binding_layout_id),
                pid=pid_value,
            )
            if initial_selection is not None:
                request.initial_selection.CopyFrom(initial_selection)
            if source_artifact_id:
                request.source_artifact_id = str(source_artifact_id)
            if target_publication_token:
                request.target_publication_token = bytes(target_publication_token)
            if copy_plan is not None:
                request.copy_plan.CopyFrom(copy_plan)
            if dst_specs is not None:
                for spec in dst_specs:
                    request.dst_tensors.add().CopyFrom(spec)
            try:
                response: store_daemon_pb2.CreateBindingResponse = self._unary_call(
                    self.stub_v2.CreateBinding,
                    request,
                    timeout=float(timeout_s),
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
                raise RuntimeError(
                    _grpc_message(e, fallback="CreateBinding RPC failed")
                ) from e
        return response

    def commit_binding_artifact(
        self,
        *,
        binding_id: str,
        selection: common_pb2.ArtifactSelection,
        source_artifact_id: str | None = None,
        target_publication_token: bytes | None = None,
        timeout_s: float = 30.0,
    ) -> store_daemon_pb2.CommitBindingArtifactResponse:
        if not binding_id:
            raise ValueError("binding_id is required")
        if not isinstance(selection, common_pb2.ArtifactSelection):
            raise ValueError("selection is required")
        with self._client_span("Client/CommitBindingArtifact") as span:
            request = store_daemon_pb2.CommitBindingArtifactRequest(
                binding_id=str(binding_id),
                selection=selection,
            )
            if source_artifact_id:
                request.source_artifact_id = str(source_artifact_id)
            if target_publication_token:
                request.target_publication_token = bytes(target_publication_token)
            try:
                response: store_daemon_pb2.CommitBindingArtifactResponse = (
                    self._unary_call(
                        self.stub_v2.CommitBindingArtifact,
                        request,
                        timeout=float(timeout_s),
                        span=span,
                        retries=1,
                    )
                )
            except grpc.RpcError as e:  # noqa: BLE001
                span.record_exception(e)
                raise RuntimeError(
                    _grpc_message(e, fallback="CommitBindingArtifact RPC failed")
                ) from e
        return response

    def begin_binding_update(
        self,
        *,
        binding_id: str,
        timeout_s: float = 30.0,
    ) -> store_daemon_pb2.BeginBindingUpdateResponse:
        if not binding_id:
            raise ValueError("binding_id is required")
        with self._client_span("Client/BeginBindingUpdate") as span:
            request = store_daemon_pb2.BeginBindingUpdateRequest(
                binding_id=str(binding_id)
            )
            try:
                response: store_daemon_pb2.BeginBindingUpdateResponse = (
                    self._unary_call(
                        self.stub_v2.BeginBindingUpdate,
                        request,
                        timeout=float(timeout_s),
                        span=span,
                        retries=1,
                    )
                )
            except grpc.RpcError as e:  # noqa: BLE001
                span.record_exception(e)
                raise RuntimeError(
                    _grpc_message(e, fallback="BeginBindingUpdate RPC failed")
                ) from e
        return response

    def seal_binding(
        self,
        *,
        binding_id: str,
        update_epoch: str,
        timeout_s: float = 30.0,
    ) -> store_daemon_pb2.SealBindingResponse:
        if not binding_id:
            raise ValueError("binding_id is required")
        if not update_epoch:
            raise ValueError("update_epoch is required")
        with self._client_span("Client/SealBinding") as span:
            request = store_daemon_pb2.SealBindingRequest(
                binding_id=str(binding_id),
                update_epoch=str(update_epoch),
            )
            try:
                response: store_daemon_pb2.SealBindingResponse = self._unary_call(
                    self.stub_v2.SealBinding,
                    request,
                    timeout=float(timeout_s),
                    span=span,
                    retries=1,
                )
            except grpc.RpcError as e:  # noqa: BLE001
                span.record_exception(e)
                raise RuntimeError(
                    _grpc_message(e, fallback="SealBinding RPC failed")
                ) from e
        return response

    def promote_binding_current_value(
        self,
        *,
        binding_id: str,
        binding_value_id: str,
        timeout_s: float = 30.0,
    ) -> store_daemon_pb2.PromoteBindingCurrentValueResponse:
        if not binding_id:
            raise ValueError("binding_id is required")
        if not binding_value_id:
            raise ValueError("binding_value_id is required")
        with self._client_span("Client/PromoteBindingCurrentValue") as span:
            request = store_daemon_pb2.PromoteBindingCurrentValueRequest(
                binding_id=str(binding_id),
                binding_value_id=str(binding_value_id),
            )
            try:
                response: store_daemon_pb2.PromoteBindingCurrentValueResponse = (
                    self._unary_call(
                        self.stub_v2.PromoteBindingCurrentValue,
                        request,
                        timeout=float(timeout_s),
                        span=span,
                        retries=1,
                    )
                )
            except grpc.RpcError as e:  # noqa: BLE001
                span.record_exception(e)
                raise RuntimeError(
                    _grpc_message(
                        e,
                        fallback="PromoteBindingCurrentValue RPC failed",
                    )
                ) from e
        return response

    def refill_owned_binding(
        self,
        *,
        binding_id: str,
        artifact_id: str,
        public_disk_source: store_daemon_pb2.PublicDiskSourceHandle | None = None,
        source_selection: common_pb2.ArtifactSelection | None = None,
        realization_plan: store_daemon_pb2.BindingRealizationPlan | None = None,
        source_policy: store_daemon_pb2.SourcePolicy | None = None,
        execution_topology: store_daemon_pb2.SourceExecutionTopology | None = None,
        collective_policy: store_daemon_pb2.CollectivePolicy | None = None,
        serving_runtime_policy: "ServingRuntimePolicy | None" = None,
        placement: store_daemon_pb2.TransformPlacement | None = None,
        operation_id: str | None = None,
        timeout_s: float = 600.0,
    ) -> store_daemon_pb2.RefillOwnedBindingResponse:
        if not binding_id:
            raise ValueError("binding_id is required")
        if not artifact_id and public_disk_source is None:
            raise ValueError("artifact_id is required unless public_disk_source is set")
        with self._client_span("Client/RefillOwnedBinding") as span:
            resolved_source_policy = _normalize_source_policy(source_policy)
            request = store_daemon_pb2.RefillOwnedBindingRequest(
                binding_id=str(binding_id),
                artifact_id=str(artifact_id),
            )
            if public_disk_source is not None:
                request.public_disk_source.CopyFrom(public_disk_source)
            if source_selection is not None:
                request.source_selection.CopyFrom(source_selection)
            if realization_plan is not None:
                request.realization_plan.CopyFrom(realization_plan)
            request.source_policy.CopyFrom(resolved_source_policy)
            if execution_topology is not None:
                request.execution_topology.CopyFrom(execution_topology)
            if collective_policy is not None:
                request.collective_policy = collective_policy
            if serving_runtime_policy is not None:
                request.serving_artifact_policy.CopyFrom(
                    serving_runtime_policy.to_proto()
                )
            if placement is not None:
                request.placement = placement
            if operation_id:
                request.operation_id = str(operation_id)
            try:
                response: store_daemon_pb2.RefillOwnedBindingResponse = (
                    self._unary_call(
                        self.stub_v2.RefillOwnedBinding,
                        request,
                        timeout=float(timeout_s),
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
                raise RuntimeError(
                    _grpc_message(e, fallback="RefillOwnedBinding RPC failed")
                ) from e
        return response

    def close_owned_binding(
        self,
        *,
        binding_id: str,
        timeout_s: float = 30.0,
    ) -> store_daemon_pb2.CloseOwnedBindingResponse:
        if not binding_id:
            raise ValueError("binding_id is required")
        with self._client_span("Client/CloseOwnedBinding") as span:
            request = store_daemon_pb2.CloseOwnedBindingRequest(
                binding_id=str(binding_id)
            )
            try:
                response: store_daemon_pb2.CloseOwnedBindingResponse = self._unary_call(
                    self.stub_v2.CloseOwnedBinding,
                    request,
                    timeout=float(timeout_s),
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
                    _grpc_message(e, fallback="CloseOwnedBinding RPC failed")
                ) from e
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

    def batch_exists(
        self,
        *,
        selections: Iterable[common_pb2.ArtifactSelection],
        timeout_s: float = 30.0,
    ) -> store_daemon_pb2.BatchExistsResponse:
        request = store_daemon_pb2.BatchExistsRequest()
        for selection in selections:
            request.selections.add().CopyFrom(selection)
        with self._client_span("Client/BatchExists") as span:
            try:
                response: store_daemon_pb2.BatchExistsResponse = self._unary_call(
                    self.stub_v2.BatchExists,
                    request,
                    timeout=float(timeout_s),
                    span=span,
                    retries=1,
                )
            except grpc.RpcError as e:  # noqa: BLE001
                span.record_exception(e)
                raise RuntimeError(
                    _grpc_message(e, fallback="BatchExists RPC failed")
                ) from e
        return response

    def batch_get_into_region(
        self,
        *,
        selections: Iterable[common_pb2.ArtifactSelection],
        target_layout: store_daemon_pb2.TargetLayout,
        pid: int,
        device_uuid: str,
        operation_id: str | None = None,
        timeout_s: float = 600.0,
    ) -> store_daemon_pb2.BatchGetIntoRegionResponse:
        request = store_daemon_pb2.BatchGetIntoRegionRequest(
            target_layout=target_layout,
            pid=int(pid),
            device_uuid=str(device_uuid),
        )
        for selection in selections:
            request.selections.add().CopyFrom(selection)
        if operation_id:
            request.operation_id = str(operation_id)
        with self._client_span("Client/BatchGetIntoRegion") as span:
            try:
                response: store_daemon_pb2.BatchGetIntoRegionResponse = (
                    self._unary_call(
                        self.stub_v2.BatchGetIntoRegion,
                        request,
                        timeout=float(timeout_s),
                        span=span,
                        retries=1,
                    )
                )
            except grpc.RpcError as e:  # noqa: BLE001
                span.record_exception(e)
                raise RuntimeError(
                    _grpc_message(e, fallback="BatchGetIntoRegion RPC failed")
                ) from e
        return response

    def batch_put_if_absent_from_region(
        self,
        *,
        items: Iterable[store_daemon_pb2.BatchPutIfAbsentFromRegionItem],
        source_layout: store_daemon_pb2.TargetLayout,
        pid: int,
        device_uuid: str,
        ttl_ms: int | None = None,
        operation_id: str | None = None,
        timeout_s: float = 600.0,
    ) -> store_daemon_pb2.BatchPutIfAbsentFromRegionResponse:
        request = store_daemon_pb2.BatchPutIfAbsentFromRegionRequest(
            source_layout=source_layout,
            pid=int(pid),
            device_uuid=str(device_uuid),
        )
        for item in items:
            request.items.add().CopyFrom(item)
        if ttl_ms is not None:
            request.ttl_ms = int(ttl_ms)
        if operation_id:
            request.operation_id = str(operation_id)
        with self._client_span("Client/BatchPutIfAbsentFromRegion") as span:
            try:
                response: store_daemon_pb2.BatchPutIfAbsentFromRegionResponse = (
                    self._unary_call(
                        self.stub_v2.BatchPutIfAbsentFromRegion,
                        request,
                        timeout=float(timeout_s),
                        span=span,
                        retries=1,
                    )
                )
            except grpc.RpcError as e:  # noqa: BLE001
                span.record_exception(e)
                raise RuntimeError(
                    _grpc_message(e, fallback="BatchPutIfAbsentFromRegion RPC failed")
                ) from e
        return response

    def batch_touch_ttl(
        self,
        *,
        artifact_ids: Iterable[str],
        ttl_ms: int,
        timeout_s: float = 30.0,
    ) -> store_daemon_pb2.BatchTouchTtlResponse:
        request = store_daemon_pb2.BatchTouchTtlRequest(
            artifact_ids=[
                str(artifact_id) for artifact_id in artifact_ids if str(artifact_id)
            ],
            ttl_ms=int(ttl_ms),
        )
        with self._client_span("Client/BatchTouchTtl") as span:
            try:
                response: store_daemon_pb2.BatchTouchTtlResponse = self._unary_call(
                    self.stub_v2.BatchTouchTtl,
                    request,
                    timeout=float(timeout_s),
                    span=span,
                    retries=1,
                )
            except grpc.RpcError as e:  # noqa: BLE001
                span.record_exception(e)
                raise RuntimeError(
                    _grpc_message(e, fallback="BatchTouchTtl RPC failed")
                ) from e
        return response

    def home_batch_exists(
        self,
        *,
        fence: store_daemon_pb2.RouteFence,
        artifact_ids: Iterable[str],
        timeout_s: float = 30.0,
    ) -> store_daemon_pb2.HomeBatchExistsResponse:
        request = store_daemon_pb2.HomeBatchExistsRequest(
            fence=fence,
            artifact_ids=[str(item) for item in artifact_ids if str(item)],
        )
        with self._client_span("Client/HomeBatchExists") as span:
            try:
                response: store_daemon_pb2.HomeBatchExistsResponse = self._unary_call(
                    self.stub_v2.HomeBatchExists,
                    request,
                    timeout=float(timeout_s),
                    span=span,
                    retries=1,
                )
            except grpc.RpcError as e:  # noqa: BLE001
                span.record_exception(e)
                raise RuntimeError(
                    _grpc_message(e, fallback="HomeBatchExists RPC failed")
                ) from e
        return response

    def home_batch_get(
        self,
        *,
        fence: store_daemon_pb2.RouteFence,
        artifact_ids: Iterable[str],
        operation_id: str | None = None,
        timeout_s: float = 30.0,
    ) -> store_daemon_pb2.HomeBatchGetResponse:
        request = store_daemon_pb2.HomeBatchGetRequest(
            fence=fence,
            artifact_ids=[str(item) for item in artifact_ids if str(item)],
        )
        if operation_id:
            request.operation_id = str(operation_id)
        with self._client_span("Client/HomeBatchGet") as span:
            try:
                response: store_daemon_pb2.HomeBatchGetResponse = self._unary_call(
                    self.stub_v2.HomeBatchGet,
                    request,
                    timeout=float(timeout_s),
                    span=span,
                    retries=1,
                )
            except grpc.RpcError as e:  # noqa: BLE001
                span.record_exception(e)
                raise RuntimeError(
                    _grpc_message(e, fallback="HomeBatchGet RPC failed")
                ) from e
        return response

    def home_batch_put_if_absent(
        self,
        *,
        fence: store_daemon_pb2.RouteFence,
        items: Iterable[store_daemon_pb2.HomeBatchPutIfAbsentItem],
        ttl_ms: int | None = None,
        operation_id: str | None = None,
        timeout_s: float = 30.0,
    ) -> store_daemon_pb2.HomeBatchPutIfAbsentResponse:
        request = store_daemon_pb2.HomeBatchPutIfAbsentRequest(
            fence=fence,
        )
        for item in items:
            request.items.add().CopyFrom(item)
        if ttl_ms is not None:
            request.ttl_ms = int(ttl_ms)
        if operation_id:
            request.operation_id = str(operation_id)
        with self._client_span("Client/HomeBatchPutIfAbsent") as span:
            try:
                response: store_daemon_pb2.HomeBatchPutIfAbsentResponse = (
                    self._unary_call(
                        self.stub_v2.HomeBatchPutIfAbsent,
                        request,
                        timeout=float(timeout_s),
                        span=span,
                        retries=1,
                    )
                )
            except grpc.RpcError as e:  # noqa: BLE001
                span.record_exception(e)
                raise RuntimeError(
                    _grpc_message(e, fallback="HomeBatchPutIfAbsent RPC failed")
                ) from e
        return response

    def home_batch_touch_ttl(
        self,
        *,
        fence: store_daemon_pb2.RouteFence,
        artifact_ids: Iterable[str],
        ttl_ms: int,
        timeout_s: float = 30.0,
    ) -> store_daemon_pb2.HomeBatchTouchTtlResponse:
        request = store_daemon_pb2.HomeBatchTouchTtlRequest(
            fence=fence,
            artifact_ids=[str(item) for item in artifact_ids if str(item)],
            ttl_ms=int(ttl_ms),
        )
        with self._client_span("Client/HomeBatchTouchTtl") as span:
            try:
                response: store_daemon_pb2.HomeBatchTouchTtlResponse = self._unary_call(
                    self.stub_v2.HomeBatchTouchTtl,
                    request,
                    timeout=float(timeout_s),
                    span=span,
                    retries=1,
                )
            except grpc.RpcError as e:  # noqa: BLE001
                span.record_exception(e)
                raise RuntimeError(
                    _grpc_message(e, fallback="HomeBatchTouchTtl RPC failed")
                ) from e
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
        selection: common_pb2.ArtifactSelection,
        replica_uuid: str,
        device_uuid: str,
        pinned_allocation_timeout_ms: int = int(30e3),
        *,
        wait_for_completion: bool = True,
        wait_for_shared_disk_ms: int = 0,
        placement: store_daemon_pb2.TransformPlacement | None = None,
        return_response: Literal[True],
        source_policy: store_daemon_pb2.SourcePolicy | None = None,
        serving_runtime_policy: "ServingRuntimePolicy | None" = None,
        export_policy: store_daemon_pb2.ExportPolicy | None = None,
        need_view_data_hash: bool = True,
        target_device_type: store_daemon_pb2.DeviceType = store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
        lease_mode: store_daemon_pb2.LeaseMode = store_daemon_pb2.LeaseMode.LEASE_MODE_UNSPECIFIED,
        collective_load_group: store_daemon_pb2.CollectiveLoadGroup | None = None,
        timeout_s: float | int | None = None,
        timing_out: dict[str, float] | None = None,
    ) -> store_daemon_pb2.MaterializeReplicaResponse: ...

    @overload
    def materialize_by_artifact_id_v2(
        self,
        selection: common_pb2.ArtifactSelection,
        replica_uuid: str,
        device_uuid: str,
        pinned_allocation_timeout_ms: int = int(30e3),
        *,
        wait_for_completion: Literal[False],
        wait_for_shared_disk_ms: int = 0,
        placement: store_daemon_pb2.TransformPlacement | None = None,
        return_response: Literal[False] = False,
        source_policy: store_daemon_pb2.SourcePolicy | None = None,
        serving_runtime_policy: "ServingRuntimePolicy | None" = None,
        export_policy: store_daemon_pb2.ExportPolicy | None = None,
        need_view_data_hash: bool = True,
        target_device_type: store_daemon_pb2.DeviceType = store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
        lease_mode: store_daemon_pb2.LeaseMode = store_daemon_pb2.LeaseMode.LEASE_MODE_UNSPECIFIED,
        collective_load_group: store_daemon_pb2.CollectiveLoadGroup | None = None,
        timeout_s: float | int | None = None,
        timing_out: dict[str, float] | None = None,
    ) -> tuple[bytes, store_daemon_pb2.MaterializeReplicaStatus]: ...

    @overload
    def materialize_by_artifact_id_v2(
        self,
        selection: common_pb2.ArtifactSelection,
        replica_uuid: str,
        device_uuid: str,
        pinned_allocation_timeout_ms: int = int(30e3),
        *,
        wait_for_completion: Literal[True] = True,
        wait_for_shared_disk_ms: int = 0,
        placement: store_daemon_pb2.TransformPlacement | None = None,
        return_response: Literal[False] = False,
        source_policy: store_daemon_pb2.SourcePolicy | None = None,
        export_policy: store_daemon_pb2.ExportPolicy | None = None,
        need_view_data_hash: bool = True,
        target_device_type: store_daemon_pb2.DeviceType = store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
        lease_mode: store_daemon_pb2.LeaseMode = store_daemon_pb2.LeaseMode.LEASE_MODE_UNSPECIFIED,
        collective_load_group: store_daemon_pb2.CollectiveLoadGroup | None = None,
        timeout_s: float | int | None = None,
        timing_out: dict[str, float] | None = None,
    ) -> bytes: ...

    def materialize_by_artifact_id_v2(
        self,
        selection: common_pb2.ArtifactSelection,
        replica_uuid: str,
        device_uuid: str,
        pinned_allocation_timeout_ms: int = int(30e3),
        wait_for_completion: bool = True,
        wait_for_shared_disk_ms: int = 0,
        placement: store_daemon_pb2.TransformPlacement | None = None,
        return_response: bool = False,
        source_policy: store_daemon_pb2.SourcePolicy | None = None,
        serving_runtime_policy: "ServingRuntimePolicy | None" = None,
        export_policy: store_daemon_pb2.ExportPolicy | None = None,
        need_view_data_hash: bool = True,
        target_device_type: store_daemon_pb2.DeviceType = store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
        lease_mode: store_daemon_pb2.LeaseMode = store_daemon_pb2.LeaseMode.LEASE_MODE_UNSPECIFIED,
        collective_load_group: store_daemon_pb2.CollectiveLoadGroup | None = None,
        timeout_s: float | int | None = None,
        timing_out: dict[str, float] | None = None,
    ) -> (
        store_daemon_pb2.MaterializeReplicaResponse
        | bytes
        | tuple[bytes, store_daemon_pb2.MaterializeReplicaStatus]
    ):
        if not isinstance(selection, common_pb2.ArtifactSelection):
            raise ValueError("selection is required")
        if not selection.artifact_id:
            raise ValueError("selection.artifact_id is required")
        logger.debug(
            "materialize_by_artifact_id_v2: %s, %s, wait_for_completion=%s",
            selection.artifact_id,
            replica_uuid,
            wait_for_completion,
        )
        pid = self._get_effective_pid()
        with self._client_span("Client/MaterializeReplicaV2") as span:
            resolved_source_policy = _normalize_source_policy(source_policy)
            device_uuid_value = (
                ""
                if target_device_type == store_daemon_pb2.DeviceType.DEVICE_TYPE_CPU
                else device_uuid
            )
            request = store_daemon_pb2.MaterializeReplicaRequest(
                pid=pid,
                selection=selection,
                replica_uuid=replica_uuid,
                device_uuid=device_uuid_value,
                target_device_type=target_device_type,
                pinned_allocation_timeout_ms=pinned_allocation_timeout_ms,
                lease_mode=lease_mode,
            )
            if collective_load_group is not None:
                request.collective_load_group.CopyFrom(collective_load_group)
            if wait_for_shared_disk_ms:
                request.wait_for_shared_disk_ms = int(wait_for_shared_disk_ms)
            request.source_policy.CopyFrom(resolved_source_policy)
            if serving_runtime_policy is not None:
                request.serving_artifact_policy.CopyFrom(
                    serving_runtime_policy.to_proto()
                )
            if export_policy is not None:
                request.export_policy = export_policy
            if not need_view_data_hash:
                request.need_view_data_hash = False
            if placement is not None:
                request.placement = placement
            try:
                rpc_start = time.perf_counter()
                response: store_daemon_pb2.MaterializeReplicaResponse = (
                    self._unary_call(
                        self.stub_v2.MaterializeReplica,
                        request,
                        timeout=60 if timeout_s is None else timeout_s,
                        span=span,
                        retries=1,
                    )
                )
                if timing_out is not None:
                    timing_out["rpc_roundtrip_sec"] = time.perf_counter() - rpc_start
            except grpc.RpcError as e:  # noqa: BLE001
                span.record_exception(e)
                code = e.code()
                if code == grpc.StatusCode.UNAVAILABLE:
                    raise RuntimeError(
                        f"Local StoreDaemon ({self.server_address}) is not available."
                    ) from e
                if code == grpc.StatusCode.NOT_FOUND:
                    raise RuntimeError(
                        f"Artifact id '{selection.artifact_id}' was not found by StoreDaemon at {self.server_address}."
                    ) from e
                raise RuntimeError(
                    _grpc_message(e, fallback="MaterializeReplicaV2 RPC failed")
                ) from e

        load_status = response.status
        if (
            load_status
            == store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_FAILED
        ):
            raise RuntimeError(
                f"Artifact allocation failed for {selection.artifact_id}"
            )

        if not wait_for_completion:
            logger.info(
                "Artifact allocation initiated (async): %s, %s",
                selection.artifact_id,
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

        logger.info("Artifact loaded: %s, %s", selection.artifact_id, replica_uuid)
        if (
            response.status
            == store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_ALLOCATED
        ):
            confirm_start = time.perf_counter()
            materialize_timeout_s = 60.0 if timeout_s is None else float(timeout_s)
            confirm_timeout_s = max(300.0, materialize_timeout_s)
            success = self.confirm_replica_loaded(
                response.disk_path or "",
                replica_uuid,
                target_device_type=target_device_type,
                timeout_s=confirm_timeout_s,
            )
            if timing_out is not None:
                timing_out["confirm_replica_sec"] = time.perf_counter() - confirm_start
            if not success:
                raise RuntimeError(
                    "Failed to confirm artifact loading: "
                    f"artifact_id={selection.artifact_id}, "
                    f"replica_uuid={replica_uuid}, "
                    f"disk_path={response.disk_path or ''}"
                )
        if return_response:
            return response
        if target_device_type != store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU:
            raise ValueError(
                "materialize_by_artifact_id_v2 must use return_response=True for non-GPU targets"
            )
        assert response.mem_handle is not None
        return response.mem_handle.cuda_ipc_handle

    def import_artifact_from_path_v2(
        self, *, path: str, verify_checksums: bool = True
    ) -> store_daemon_pb2.ImportArtifactFromPathResponse:
        if not path:
            raise ValueError("path is required")
        with self._client_span("Client/ImportArtifactFromPathV2") as span:
            request = store_daemon_pb2.ImportArtifactFromPathRequest(
                path=path, verify_checksums=bool(verify_checksums)
            )
            try:
                return self._unary_call(
                    self.stub_v2.ImportArtifactFromPath,
                    request,
                    timeout=_import_artifact_from_path_timeout_seconds(),
                    span=span,
                    retries=_import_artifact_from_path_retries(),
                )
            except grpc.RpcError as e:  # noqa: BLE001
                span.record_exception(e)
                _raise_import_artifact_from_path_rpc_error(self.server_address, e)

    def import_artifact_from_path_stream_v2(
        self, *, path: str, verify_checksums: bool = True
    ) -> Iterator[store_daemon_pb2.ImportArtifactFromPathStreamEvent]:
        if not path:
            raise ValueError("path is required")
        request = store_daemon_pb2.ImportArtifactFromPathRequest(
            path=path,
            verify_checksums=bool(verify_checksums),
        )
        timeout_s = _import_artifact_from_path_timeout_seconds()

        def _event_iter() -> Iterator[
            store_daemon_pb2.ImportArtifactFromPathStreamEvent
        ]:
            with self._client_span("Client/ImportArtifactFromPathStreamV2") as span:
                try:
                    stream = self.stub_v2.ImportArtifactFromPathStream(
                        request,
                        timeout=timeout_s,
                    )
                    for event in stream:
                        yield event
                except grpc.RpcError as e:  # noqa: BLE001
                    span.record_exception(e)
                    _raise_import_artifact_from_path_rpc_error(self.server_address, e)

        return _event_iter()

    def resolve_public_disk_source(
        self,
        *,
        path: str,
        verify_checksums: bool = True,
    ) -> store_daemon_pb2.ResolvePublicDiskSourceResponse:
        if not path:
            raise ValueError("path is required")
        with self._client_span("Client/ResolvePublicDiskSource") as span:
            request = store_daemon_pb2.ResolvePublicDiskSourceRequest(
                path=path,
                verify_checksums=bool(verify_checksums),
            )
            try:
                response: store_daemon_pb2.ResolvePublicDiskSourceResponse = (
                    self._unary_call(
                        self.stub_v2.ResolvePublicDiskSource,
                        request,
                        timeout=_import_artifact_from_path_timeout_seconds(),
                        span=span,
                        retries=_import_artifact_from_path_retries(),
                    )
                )
            except grpc.RpcError as e:  # noqa: BLE001
                span.record_exception(e)
                _raise_import_artifact_from_path_rpc_error(self.server_address, e)
            return response

    def confirm_replica_loaded(
        self,
        disk_path: str,
        replica_uuid: str,
        *,
        target_device_type: store_daemon_pb2.DeviceType = store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
        timeout_s: float = 30.0,
    ) -> bool:
        confirm_retries = 5
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
                    timeout=timeout_s,
                    span=span,
                    retries=confirm_retries,
                )
                logger.info(
                    "ConfirmReplica succeeded: replica_uuid=%s, disk_path=%s",
                    replica_uuid,
                    disk_path,
                )
                return True
            except grpc.RpcError as e:
                span.record_exception(e)
                code = e.code()
                code_name = code.name if code is not None else "UNKNOWN"
                logger.error(
                    "ConfirmReplica failed: code=%s, details=%s, replica_uuid=%s, "
                    "disk_path=%s, target_device_type=%s, timeout_s=%.1f, retries=%d, daemon=%s",
                    code_name,
                    _grpc_details(e),
                    replica_uuid,
                    disk_path,
                    int(target_device_type),
                    timeout_s,
                    confirm_retries,
                    self.server_address,
                )
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
                # Older daemons may omit optional local socket metadata.
                local_handle_socket_path = str(
                    getattr(response, "local_handle_socket_path", "") or ""
                )
                cpu_shared_memory_enabled = bool(
                    getattr(response, "cpu_shared_memory_enabled", True)
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
                    source_bound_capability_flags=int(
                        getattr(response, "source_bound_capability_flags", 0)
                    ),
                    source_bound_contract_version=int(
                        getattr(response, "source_bound_contract_version", 0)
                    ),
                )

    def execute_plan(
        self,
        *,
        plan: Any,
        execution_class: str = "terminal_only",
        dry_run: bool = False,
        timeout_s: float = 30.0,
    ) -> "node_agent_pb2.ExecutePlanResponse":
        execution_class_value = {
            "terminal_only": store_daemon_pb2.PLAN_EXECUTION_CLASS_TERMINAL_ONLY,
            "public_continuation_required": (
                store_daemon_pb2.PLAN_EXECUTION_CLASS_PUBLIC_CONTINUATION_REQUIRED
            ),
        }.get(str(execution_class))
        if execution_class_value is None:
            raise ValueError(
                "execution_class must be 'terminal_only' or 'public_continuation_required'"
            )
        request = store_daemon_pb2.ExecutePlanRequest(
            execution_class=execution_class_value,
            dry_run=bool(dry_run),
        )
        request.plan.CopyFrom(plan)
        with self._client_span("Client/ExecutePlan") as span:
            response: store_daemon_pb2.ExecutePlanResponse = self._unary_call(
                self.stub.ExecutePlan,
                request,
                timeout=timeout_s,
                span=span,
                retries=0,
            )
        if not response.terminal_result:
            return node_agent_pb2.ExecutePlanResponse(
                request_id=str(response.request_id),
                ok=bool(response.ok),
            )
        terminal = node_agent_pb2.ExecutePlanResponse()
        terminal.ParseFromString(response.terminal_result)
        if not terminal.request_id:
            terminal.request_id = str(response.request_id)
        terminal.ok = bool(response.ok)
        return terminal

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

    def list_directory_workers(
        self,
        *,
        include_unavailable: bool = False,
        required_capability_flags: int = 0,
        max_staleness_ms: int | None = None,
        timeout_s: float = 5.0,
    ) -> store_daemon_pb2.ListDirectoryWorkersResponse:
        with self._client_span("Client/ListDirectoryWorkers") as span:
            request = store_daemon_pb2.ListDirectoryWorkersRequest(
                include_unavailable=bool(include_unavailable),
                required_capability_flags=int(required_capability_flags),
            )
            if max_staleness_ms is not None:
                request.max_staleness_ms = int(max_staleness_ms)
            try:
                response: store_daemon_pb2.ListDirectoryWorkersResponse = (
                    self._unary_call(
                        self.stub_v2.ListDirectoryWorkers,
                        request,
                        timeout=timeout_s,
                        span=span,
                        retries=1,
                    )
                )
            except grpc.RpcError as e:  # noqa: BLE001
                span.record_exception(e)
                raise
        return response

    def list_directory_instances(
        self,
        *,
        include_unavailable: bool = False,
        required_capability_flags: int = 0,
        max_staleness_ms: int | None = None,
        timeout_s: float = 5.0,
    ) -> store_daemon_pb2.ListDirectoryInstancesResponse:
        with self._client_span("Client/ListDirectoryInstances") as span:
            request = store_daemon_pb2.ListDirectoryInstancesRequest(
                include_unavailable=bool(include_unavailable),
                required_capability_flags=int(required_capability_flags),
            )
            if max_staleness_ms is not None:
                request.max_staleness_ms = int(max_staleness_ms)
            try:
                response: store_daemon_pb2.ListDirectoryInstancesResponse = (
                    self._unary_call(
                        self.stub_v2.ListDirectoryInstances,
                        request,
                        timeout=timeout_s,
                        span=span,
                        retries=1,
                    )
                )
            except grpc.RpcError as e:  # noqa: BLE001
                span.record_exception(e)
                raise
        return response

    def resolve_instance_execution(
        self,
        *,
        instance_id: str,
        max_staleness_ms: int | None = None,
        timeout_s: float = 5.0,
    ) -> store_daemon_pb2.ResolveInstanceExecutionResponse:
        with self._client_span("Client/ResolveInstanceExecution") as span:
            request = store_daemon_pb2.ResolveInstanceExecutionRequest(
                instance_id=str(instance_id)
            )
            if max_staleness_ms is not None:
                request.max_staleness_ms = int(max_staleness_ms)
            try:
                response: store_daemon_pb2.ResolveInstanceExecutionResponse = (
                    self._unary_call(
                        self.stub_v2.ResolveInstanceExecution,
                        request,
                        timeout=timeout_s,
                        span=span,
                        retries=1,
                    )
                )
            except grpc.RpcError as e:  # noqa: BLE001
                span.record_exception(e)
                raise
        return response

    def get_detailed_status(self) -> store_daemon_pb2.GetDetailedStatusResponse:
        with self._client_span("Client/GetDetailedStatus") as span:
            request = store_daemon_pb2.GetDetailedStatusRequest()
            try:
                response: store_daemon_pb2.GetDetailedStatusResponse = self._unary_call(
                    self.stub_v2.GetDetailedStatus,
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
                publish_size = 0
                publish_offset = 0
                if resp.stable_dram.HasField("publish_cpu_memfd"):
                    publish_size = int(resp.stable_dram.publish_cpu_memfd.size_bytes)
                    publish_offset = int(
                        resp.stable_dram.publish_cpu_memfd.offset_bytes
                    )
                handshake = StableDramHandshake(
                    staging_cuda_ipc_handle=bytes(
                        resp.stable_dram.staging_cuda_ipc_handle
                    ),
                    publish_cpu_memfd_size_bytes=publish_size,
                    publish_cpu_memfd_offset_bytes=publish_offset,
                    publish_cpu_memfd_lease_token=bytes(
                        resp.stable_dram.publish_cpu_memfd_lease_token
                    ),
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
            registration_kind = (
                "piece"
                if resp.registration_kind
                == store_daemon_pb2.VIEW_REGISTRATION_KIND_PIECE
                else "canonical"
            )
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

    def register_region(
        self,
        *,
        memory_kind: RegionMemoryKind,
        size_bytes: int,
        ttl_ms: int,
        device_id: int | None = None,
        cuda_ipc_handle: bytes | None = None,
        host_shared_attach_token: bytes | None = None,
        daemon_managed: bool = False,
        host_shared_region_class: HostSharedRegionClass | None = None,
        session_id: str | None = None,
        region_name: str | None = None,
        timeout_s: float = 10.0,
    ) -> LocalRegionHandle:
        if size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        if ttl_ms < 0:
            raise ValueError("ttl_ms must be non-negative (0 disables TTL)")

        if memory_kind is RegionMemoryKind.VRAM:
            if device_id is None or int(device_id) < 0:
                raise ValueError("device_id must be non-negative for VRAM regions")
            if not cuda_ipc_handle:
                raise ValueError("cuda_ipc_handle must not be empty for VRAM regions")
            req = store_daemon_pb2.RegisterRegionRequest(
                memory_kind=store_daemon_pb2.REGION_MEMORY_KIND_VRAM,
                device_id=int(device_id),
                size_bytes=int(size_bytes),
                ttl_ms=int(ttl_ms),
                owner_pid=int(self._get_effective_pid()),
                cuda_ipc_handle=bytes(cuda_ipc_handle),
            )
        else:
            req = store_daemon_pb2.RegisterRegionRequest(
                memory_kind=store_daemon_pb2.REGION_MEMORY_KIND_HOST_SHARED,
                device_id=-1,
                size_bytes=int(size_bytes),
                ttl_ms=int(ttl_ms),
                owner_pid=int(self._get_effective_pid()),
            )
            req.host_shared.attach_token = bytes(host_shared_attach_token or b"")
            req.host_shared.daemon_managed = bool(daemon_managed)
            region_class = host_shared_region_class or HostSharedRegionClass.SCRATCH
            req.host_shared.region_class = (
                store_daemon_pb2.HOST_SHARED_REGION_CLASS_ALLOCATOR
                if region_class is HostSharedRegionClass.ALLOCATOR
                else store_daemon_pb2.HOST_SHARED_REGION_CLASS_SCRATCH
            )
        if session_id:
            req.session_id = session_id
        if region_name:
            req.region_name = region_name

        with self._client_span("Client/RegisterRegion") as span:
            set_span_attributes(
                {
                    "tc.region.size_bytes": int(size_bytes),
                    "tc.region.ttl_ms": int(ttl_ms),
                    "tc.region.memory_kind": memory_kind.value,
                }
            )
            if device_id is not None:
                set_span_attributes({"tc.device.id": int(device_id)})
            try:
                resp = self._unary_call(
                    self.stub.RegisterRegion,
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
                    f"RegisterRegion failed: {_grpc_message(e, fallback='rpc failed')}"
                ) from e

        expires_at = None
        if resp.HasField("expires_at"):
            expires_at = resp.expires_at.ToDatetime(tzinfo=timezone.utc)
        resolved_device_id = int(resp.region.device_id)
        resolved_region_class: HostSharedRegionClass | None = None
        if resp.HasField("host_shared"):
            if (
                resp.host_shared.region_class
                == store_daemon_pb2.HOST_SHARED_REGION_CLASS_ALLOCATOR
            ):
                resolved_region_class = HostSharedRegionClass.ALLOCATOR
            elif (
                resp.host_shared.region_class
                == store_daemon_pb2.HOST_SHARED_REGION_CLASS_SCRATCH
            ):
                resolved_region_class = HostSharedRegionClass.SCRATCH
        return LocalRegionHandle(
            region_id=str(resp.region.region_id),
            memory_kind=(
                RegionMemoryKind.HOST_SHARED
                if resp.region.memory_kind
                == store_daemon_pb2.REGION_MEMORY_KIND_HOST_SHARED
                else RegionMemoryKind.VRAM
            ),
            ttl_ms=int(resp.ttl_ms),
            size_bytes=int(resp.region.size_bytes),
            device_id=None if resolved_device_id < 0 else resolved_device_id,
            attach_token=(
                bytes(resp.host_shared.attach_token)
                if resp.HasField("host_shared")
                else b""
            ),
            daemon_managed=(
                bool(resp.host_shared.daemon_managed)
                if resp.HasField("host_shared")
                else False
            ),
            host_shared_region_class=resolved_region_class,
            expires_at=expires_at,
        )

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
        handle = self.register_region(
            memory_kind=RegionMemoryKind.VRAM,
            device_id=device_id,
            size_bytes=size_bytes,
            ttl_ms=ttl_ms,
            cuda_ipc_handle=cuda_ipc_handle,
            session_id=session_id,
            region_name=region_name,
            timeout_s=timeout_s,
        )
        return VramRegionHandle(
            region_id=handle.region_id,
            ttl_ms=handle.ttl_ms,
            expires_at=handle.expires_at,
        )

    def unregister_region(
        self,
        region_id: str,
        *,
        session_id: str | None = None,
        force: bool | None = None,
        timeout_s: float = 10.0,
    ) -> bool:
        if not region_id:
            raise ValueError("region_id is required")
        req = store_daemon_pb2.UnregisterRegionRequest(
            region_id=region_id,
            owner_pid=int(self._get_effective_pid()),
        )
        if session_id:
            req.session_id = session_id
        if force is not None:
            req.force = bool(force)

        with self._client_span("Client/UnregisterRegion") as span:
            set_span_attributes({"tc.region.id": region_id})
            try:
                resp = self._unary_call(
                    self.stub.UnregisterRegion,
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
                    "UnregisterRegion failed: "
                    f"{_grpc_message(e, fallback='rpc failed')}"
                ) from e

        return bool(resp.released)

    def activate_stable_local_backing(
        self,
        region_id: str,
        *,
        slot_bytes: int,
        session_id: str | None = None,
        timeout_s: float = 180.0,
    ) -> None:
        if not region_id:
            raise ValueError("region_id is required")
        if slot_bytes <= 0:
            raise ValueError("slot_bytes must be positive")
        req = store_daemon_pb2.ActivateStableLocalBackingRequest(
            region_id=region_id,
            owner_pid=int(self._get_effective_pid()),
            slot_bytes=int(slot_bytes),
        )
        if session_id:
            req.session_id = session_id

        with self._client_span("Client/ActivateStableLocalBacking") as span:
            set_span_attributes(
                {
                    "tc.region.id": region_id,
                    "tc.region.slot_bytes": int(slot_bytes),
                }
            )
            try:
                self._unary_call(
                    self.stub.ActivateStableLocalBacking,
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
                raise RuntimeError(
                    "ActivateStableLocalBacking failed: "
                    f"{_grpc_message(e, fallback='rpc failed')}"
                ) from e

    def unregister_vram_region(
        self,
        region_id: str,
        *,
        session_id: str | None = None,
        force: bool | None = None,
        timeout_s: float = 10.0,
    ) -> bool:
        return self.unregister_region(
            region_id,
            session_id=session_id,
            force=force,
            timeout_s=timeout_s,
        )

    def attach_host_shared_region(
        self,
        handle: LocalRegionHandle,
        *,
        timeout_s: float = 5.0,
    ) -> HostSharedRegionAttachment:
        if handle.memory_kind is not RegionMemoryKind.HOST_SHARED:
            raise ValueError("attach_host_shared_region requires HOST_SHARED")
        if not handle.attach_token:
            raise ValueError("HOST_SHARED region is missing attach_token")
        fd = self._request_local_handle_fd(
            opcode=3,
            token=handle.attach_token,
            local_handle_socket_path=self._local_handle_socket_path(),
            timeout_s=timeout_s,
        )
        return HostSharedRegionAttachment(
            region_id=handle.region_id,
            size_bytes=handle.size_bytes,
            attach_token=handle.attach_token,
            fd=fd,
        )

    def release_host_shared_region(
        self,
        handle: LocalRegionHandle,
        *,
        timeout_s: float = 5.0,
    ) -> bool:
        if handle.memory_kind is not RegionMemoryKind.HOST_SHARED:
            raise ValueError("release_host_shared_region requires HOST_SHARED")
        if not handle.attach_token:
            raise ValueError("HOST_SHARED region is missing attach_token")
        return self._release_local_handle_token(
            token=handle.attach_token,
            local_handle_socket_path=self._local_handle_socket_path(),
            timeout_s=timeout_s,
        )

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
        keep_shared_disk_copy: bool = False,
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
        if keep_shared_disk_copy:
            req.keep_shared_disk_copy = True
        if operation_id:
            req.operation_id = str(operation_id)

        with self._client_span("Client/DeregisterArtifact") as span:
            set_span_attributes(
                {
                    "tc.artifact.id": artifact_id,
                    "tc.deregister.wait": bool(wait_for_drain),
                    "tc.deregister.keep_shared_disk_copy": bool(keep_shared_disk_copy),
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
        target_publication_token: bytes,
        byte_space: common_pb2.ByteSpaceRef,
        ttl_ms: int | None = None,
        owner_pid: int | None = None,
        operation_id: str | None = None,
        timeout_s: float = 60.0,
    ) -> store_daemon_pb2.PublishTargetReplicaResponse:
        if not target_publication_token:
            raise ValueError("target_publication_token is required")
        if byte_space is None:
            raise ValueError("byte_space is required")
        req = store_daemon_pb2.PublishTargetReplicaRequest(
            target_publication_token=bytes(target_publication_token),
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

    def start_publish_target_replica(
        self,
        *,
        target_publication_token: bytes,
        byte_space: common_pb2.ByteSpaceRef,
        ttl_ms: int | None = None,
        owner_pid: int | None = None,
        operation_id: str | None = None,
        timeout_s: float = 10.0,
    ) -> store_daemon_pb2.StartPublishTargetReplicaResponse:
        if not target_publication_token:
            raise ValueError("target_publication_token is required")
        if byte_space is None:
            raise ValueError("byte_space is required")
        req = store_daemon_pb2.PublishTargetReplicaRequest(
            target_publication_token=bytes(target_publication_token),
            byte_space=byte_space,
        )
        if ttl_ms is not None:
            req.ttl_ms = int(ttl_ms)
        if owner_pid is not None:
            req.owner_pid = int(owner_pid)
        if operation_id:
            req.operation_id = str(operation_id)

        with self._client_span("Client/StartPublishTargetReplica") as span:
            try:
                return self._unary_call(
                    self.stub.StartPublishTargetReplica,
                    req,
                    timeout=timeout_s,
                    span=span,
                    retries=0,
                )
            except grpc.RpcError as e:
                code = e.code()
                if code == grpc.StatusCode.INVALID_ARGUMENT:
                    raise ValueError(
                        _grpc_message(e, fallback="invalid argument")
                    ) from e
                if code == grpc.StatusCode.FAILED_PRECONDITION:
                    raise RuntimeError(
                        _grpc_message(e, fallback="failed precondition")
                    ) from e
                if code == grpc.StatusCode.DEADLINE_EXCEEDED:
                    raise TimeoutError(
                        _grpc_message(e, fallback="deadline exceeded")
                    ) from e
                raise RuntimeError(
                    "StartPublishTargetReplica failed: "
                    f"{_grpc_message(e, fallback='rpc failed')}"
                ) from e

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
        chunk_bytes: int | None = None,
        base_offset: int = 0,
        timeout_s: float | None = None,
    ) -> bool:
        return self.feed_register_artifact_view_spans(
            registration_id,
            ((int(base_offset), data),),
            chunk_bytes=chunk_bytes,
            timeout_s=timeout_s,
        )

    def _resolve_feed_view_chunk_bytes(self, requested_chunk_bytes: int | None) -> int:
        safe_limit = int(_feed_view_chunk_safe_max_bytes())
        if requested_chunk_bytes is None:
            return safe_limit
        return min(safe_limit, max(1, int(requested_chunk_bytes)))

    def feed_register_artifact_view_spans(
        self,
        registration_id: str,
        spans: Iterable[tuple[int, bytes | bytearray | memoryview]],
        *,
        chunk_bytes: int | None = None,
        timeout_s: float | None = None,
    ) -> bool:
        safe_chunk_bytes = self._resolve_feed_view_chunk_bytes(chunk_bytes)
        resolved_timeout = (
            _feed_view_stream_timeout_seconds()
            if timeout_s is None
            else (None if float(timeout_s) == 0.0 else float(timeout_s))
        )
        progress_interval_bytes = _feed_view_progress_log_interval_bytes()
        emitted_bytes = 0
        emitted_chunks = 0
        next_progress_log_bytes = (
            progress_interval_bytes if progress_interval_bytes > 0 else 0
        )
        stream_start = time.monotonic()

        def _iter():
            nonlocal emitted_bytes, emitted_chunks, next_progress_log_bytes
            for base_offset, data in spans:
                mv = memoryview(data)
                relative_offset = 0
                while relative_offset < mv.nbytes:
                    chunk = mv[relative_offset : relative_offset + safe_chunk_bytes]
                    req = store_daemon_pb2.FeedRegisterArtifactStreamRequest(
                        registration_id=registration_id
                    )
                    req.view_chunk.view_offset = int(base_offset + relative_offset)
                    req.view_chunk.data = chunk.tobytes()
                    emitted_chunks += 1
                    emitted_bytes += int(len(chunk))
                    if (
                        next_progress_log_bytes > 0
                        and emitted_bytes >= next_progress_log_bytes
                    ):
                        elapsed_s = max(1e-6, time.monotonic() - stream_start)
                        gib = emitted_bytes / float(1024**3)
                        logger.info(
                            "feed_register_artifact_stream progress "
                            "registration_id=%s chunks=%d bytes=%d "
                            "elapsed_s=%.3f avg_gibps=%.3f",
                            registration_id,
                            emitted_chunks,
                            emitted_bytes,
                            elapsed_s,
                            gib / elapsed_s,
                        )
                        next_progress_log_bytes += progress_interval_bytes
                    yield req
                    relative_offset += len(chunk)

        try:
            self._unary_call(
                self.stub.FeedRegisterArtifactStream,
                _iter(),
                timeout=resolved_timeout,
                retries=0,
            )
            if progress_interval_bytes > 0:
                elapsed_s = max(1e-6, time.monotonic() - stream_start)
                gib = emitted_bytes / float(1024**3)
                logger.info(
                    "feed_register_artifact_stream done "
                    "registration_id=%s chunks=%d bytes=%d elapsed_s=%.3f avg_gibps=%.3f",
                    registration_id,
                    emitted_chunks,
                    emitted_bytes,
                    elapsed_s,
                    gib / elapsed_s,
                )
            return True
        except grpc.RpcError as e:  # noqa: BLE001
            if progress_interval_bytes > 0:
                elapsed_s = max(1e-6, time.monotonic() - stream_start)
                gib = emitted_bytes / float(1024**3)
                logger.error(
                    "FeedRegisterArtifactStream(view) failed: %s "
                    "(registration_id=%s chunks=%d bytes=%d elapsed_s=%.3f avg_gibps=%.3f)",
                    e,
                    registration_id,
                    emitted_chunks,
                    emitted_bytes,
                    elapsed_s,
                    gib / elapsed_s,
                )
            else:
                logger.error(
                    "FeedRegisterArtifactStream(view) failed: %s (registration_id=%s)",
                    e,
                    registration_id,
                )
            return False

    def feed_register_artifact_stable_dram_write_ranges(
        self,
        registration_id: str,
        ranges: Iterable[tuple[int, int]],
        *,
        timeout_s: float | None = None,
    ) -> bool:
        resolved_timeout = (
            _feed_view_stream_timeout_seconds()
            if timeout_s is None
            else (None if float(timeout_s) == 0.0 else float(timeout_s))
        )
        max_ranges_per_request = 1024

        def _iter():
            req: store_daemon_pb2.FeedRegisterArtifactStreamRequest | None = None
            ranges_in_req = 0
            for canonical_offset, length in ranges:
                offset_value = int(canonical_offset)
                length_value = int(length)
                if offset_value < 0:
                    raise ValueError(
                        f"stable_dram write range offset must be >= 0, got {offset_value}"
                    )
                if length_value < 0:
                    raise ValueError(
                        f"stable_dram write range length must be >= 0, got {length_value}"
                    )
                if length_value == 0:
                    continue
                if req is None:
                    req = store_daemon_pb2.FeedRegisterArtifactStreamRequest(
                        registration_id=registration_id
                    )
                    ranges_in_req = 0
                dst = req.stable_dram_write_progress.ranges.add()
                dst.canonical_offset = offset_value
                dst.length = length_value
                ranges_in_req += 1
                if ranges_in_req >= max_ranges_per_request:
                    yield req
                    req = None
            if req is None:
                req = store_daemon_pb2.FeedRegisterArtifactStreamRequest(
                    registration_id=registration_id
                )
            req.stable_dram_write_progress.upload_complete = True
            yield req

        try:
            self._unary_call(
                self.stub.FeedRegisterArtifactStream,
                _iter(),
                timeout=resolved_timeout,
                retries=0,
            )
            return True
        except grpc.RpcError as e:  # noqa: BLE001
            logger.error(
                "FeedRegisterArtifactStream(stable_dram_write_progress) failed: %s (registration_id=%s)",
                e,
                registration_id,
            )
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
        fail_if_exists: bool = True,
        timeout_s: float = 5.0,
    ) -> bool:
        from tensorcast.proto.common.v1 import common_pb2 as common_pb2

        req = store_daemon_pb2.PublishReplicaKeyRequest(
            key=key,
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

    def list_artifact_layouts(
        self, artifact_id: str, *, timeout_s: float = 10.0
    ) -> list[str]:
        """Fetch layout ids currently attached to an artifact via daemon."""
        if not artifact_id:
            raise ValueError("artifact_id is required")
        with self._client_span("Client/ListArtifactLayouts") as span:
            request = store_daemon_pb2.ListArtifactLayoutsRequest(
                artifact_id=artifact_id
            )
            resp = self._unary_call(
                self.stub.ListArtifactLayouts,
                request,
                timeout=timeout_s,
                span=span,
                retries=1,
            )
            return [str(item) for item in resp.layout_ids]

    def start_assembly_attempt(
        self,
        *,
        layout_id: str | None = None,
        requirements: AssemblyRequirementSetRef | None = None,
        readiness_policy: AssemblyReadinessPolicy | None = None,
        closeout_contract: AssemblyCloseoutContract | None = None,
        representation_publish_spec: RepresentationPublishSpec | None = None,
        timeout_s: float = 30.0,
    ) -> AssemblyAttemptRef:
        if representation_publish_spec is not None:
            req = store_daemon_pb2.StartAssemblyAttemptRequest()
            spec_proto = representation_publish_spec.to_proto()
            resolved_layout_id = layout_id or getattr(
                representation_publish_spec, "layout_id", None
            )
            if resolved_layout_id and not spec_proto.layout_id:
                spec_proto.layout_id = str(resolved_layout_id)
            resolved_requirements = requirements or getattr(
                representation_publish_spec, "requirements", None
            )
            if resolved_requirements is not None and not spec_proto.HasField(
                "requirements"
            ):
                spec_proto.requirements.CopyFrom(
                    resolved_requirements.to_publication_proto()
                )
            resolved_readiness_policy = readiness_policy or getattr(
                representation_publish_spec, "readiness_policy", None
            )
            if resolved_readiness_policy is not None and not spec_proto.HasField(
                "readiness_policy"
            ):
                spec_proto.readiness_policy.CopyFrom(
                    resolved_readiness_policy.to_publication_proto()
                )
            req.representation_publish_spec.CopyFrom(spec_proto)
        else:
            if not layout_id:
                raise ValueError("layout_id is required")
            if requirements is None:
                raise ValueError(
                    "requirements are required; construct them explicitly with "
                    "AssemblyRequirementSetRef.pp_from_structural_views(...), "
                    "AssemblyRequirementSetRef.ep_from_structural_views(...), "
                    "or AssemblyRequirementSetRef.canonical_full()"
                )
            req = store_daemon_pb2.StartAssemblyAttemptRequest(layout_id=str(layout_id))
            req.requirements.CopyFrom(requirements.to_proto())
            if readiness_policy is not None:
                req.readiness_policy.CopyFrom(readiness_policy.to_proto())
            if closeout_contract is not None:
                req.closeout_contract.CopyFrom(closeout_contract.to_proto())
        with self._client_span("Client/StartAssemblyAttempt") as span:
            resp = self._unary_call(
                self.stub.StartAssemblyAttempt,
                req,
                timeout=timeout_s,
                span=span,
                retries=1,
            )
        attempt = resp.attempt
        coordinator_operation = operation_pb2.OperationRef()
        if attempt.HasField("coordinator_operation"):
            coordinator_operation.CopyFrom(attempt.coordinator_operation)
        elif attempt.coordinator_operation_id:
            coordinator_operation.operation_id = str(attempt.coordinator_operation_id)
        return AssemblyAttemptRef(
            attempt_id=str(attempt.attempt_id or ""),
            workspace_assembly_id=str(attempt.workspace_assembly_id),
            layout_id=str(attempt.layout_id),
            attempt_intent_digest=str(attempt.attempt_intent_digest),
            coordinator_generation=int(attempt.coordinator_generation),
            coordinator_operation=coordinator_operation,
        )

    def submit_binding_contribution(
        self,
        *,
        attempt_id: str,
        workspace_assembly_id: str,
        binding_id: str,
        binding_value_id: str,
        coverage_plan_hash: str,
        contribution_kind: store_daemon_pb2.BindingContributionKind,
        coordinator_operation_id: str,
        coordinator_generation: int,
        attempt_intent_digest: str = "",
        view_id: str | None = None,
        timeout_s: float = 30.0,
    ) -> store_daemon_pb2.SubmitBindingContributionResponse:
        req = store_daemon_pb2.SubmitBindingContributionRequest(
            attempt_id=str(attempt_id),
            workspace_assembly_id=str(workspace_assembly_id),
            binding_id=str(binding_id),
            binding_value_id=str(binding_value_id),
            coverage_plan_hash=str(coverage_plan_hash),
            contribution_kind=contribution_kind,
            coordinator_operation_id=str(coordinator_operation_id),
            coordinator_generation=int(coordinator_generation),
            attempt_intent_digest=str(attempt_intent_digest),
        )
        if view_id:
            req.view_id = str(view_id)
        with self._client_span("Client/SubmitBindingContribution") as span:
            return self._unary_call(
                self.stub.SubmitBindingContribution,
                req,
                timeout=timeout_s,
                span=span,
                retries=1,
            )

    def seal_assembly_attempt(
        self,
        *,
        attempt_id: str,
        timeout_s: float = 10.0,
    ) -> store_daemon_pb2.SealAssemblyAttemptResponse:
        if not attempt_id:
            raise ValueError("attempt_id is required")
        req = store_daemon_pb2.SealAssemblyAttemptRequest(attempt_id=attempt_id)
        with self._client_span("Client/SealAssemblyAttempt") as span:
            return self._unary_call(
                self.stub.SealAssemblyAttempt,
                req,
                timeout=timeout_s,
                span=span,
                retries=0,
            )

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
        self,
        operation_id: str,
        *,
        operation_ref: "operation_pb2.OperationRef | None" = None,
        timeout_s: float = 10.0,
    ) -> "operation_pb2.GetOperationResponse":
        if not operation_id:
            raise ValueError("operation_id is required")
        from tensorcast.proto.operation.v1 import operation_pb2

        req = operation_pb2.GetOperationRequest(operation_id=operation_id)
        if operation_ref is not None:
            req.ref.CopyFrom(operation_ref)
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
        operation_ref: "operation_pb2.OperationRef | None" = None,
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
        if operation_ref is not None:
            req.ref.CopyFrom(operation_ref)
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
        key_hint: str | None = None,
        policy: StorePolicy | dict[str, object] | str | None = None,
        timeout_s: float = 10.0,
    ) -> store_daemon_pb2.StartPersistenceResponse:
        if not artifact_id:
            raise ValueError("artifact_id is required")
        req = store_daemon_pb2.StartPersistenceRequest(artifact_id=artifact_id)
        if key_hint:
            req.key_hint = str(key_hint)
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
