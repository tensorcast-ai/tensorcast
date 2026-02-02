#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import time
import uuid
import weakref
from dataclasses import dataclass
from typing import (
    Callable,
    Generic,
    Literal,
    Mapping,
    NoReturn,
    Protocol,
    TypeVar,
    cast,
)

import grpc
from google.protobuf import timestamp_pb2

from tensorcast.api.context import CallContext
from tensorcast.api.errors import ArtifactError, ArtifactStatusCode
from tensorcast.error_reporting import debug_errors_enabled, debug_errors_hint
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.proto.operation.v1 import operation_pb2

T = TypeVar("T")

OperationState = Literal[
    "pending",
    "running",
    "success",
    "failed",
    "cancelled",
    "degraded",
]


@dataclass(frozen=True, slots=True)
class TimeoutErrorDetails:
    kind: Literal["ctx_deadline", "wait_timeout"]
    deadline_ms: int | None = None
    timeout_s: float | None = None
    elapsed_ms: int | None = None


@dataclass(frozen=True, slots=True)
class OperationError:
    status_code: str
    message: str
    retryable: bool
    timeout: TimeoutErrorDetails | None = None
    context: Mapping[str, str] | None = None


@dataclass(frozen=True, slots=True)
class OperationStatus:
    state: OperationState
    message: str | None = None
    progress: float | None = None
    as_of_ms: int | None = None
    error: OperationError | None = None


class OperationTimeoutError(ArtifactError):
    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message, status_code="DEADLINE_EXCEEDED", retryable=retryable)


class Operation(Generic[T]):
    operation_id: str

    def done(self) -> bool:
        status = self.status()
        return status.state in {"success", "failed", "cancelled"}

    def status(self) -> OperationStatus:
        raise NotImplementedError

    def result(self, *, timeout_s: float | None = None) -> T:
        return self.wait(timeout_s=timeout_s)

    def wait(self, *, timeout_s: float | None = None) -> T:
        raise NotImplementedError

    def cancel(self) -> bool:
        raise NotImplementedError


class _DaemonReplicaClient(Protocol):
    def query_replica_status(
        self, ticket: store_daemon_pb2.ReplicaTicket
    ) -> store_daemon_pb2.QueryReplicaStatusResponse: ...

    def wait_replica_status(
        self, ticket: store_daemon_pb2.ReplicaTicket, *, timeout_ms: int | None
    ) -> store_daemon_pb2.WaitReplicaStatusResponse: ...

    def release_replica(
        self, ticket: store_daemon_pb2.ReplicaTicket
    ) -> store_daemon_pb2.ReleaseReplicaResponse: ...


class _Runtime(Protocol):
    daemon_endpoint: str
    closed: bool

    def ensure_client(self) -> _DaemonReplicaClient: ...


_REPLICA_STATE_TO_OP_STATE: dict[int, OperationState] = {
    store_daemon_pb2.REPLICA_OPERATION_STATE_PENDING: "pending",
    store_daemon_pb2.REPLICA_OPERATION_STATE_RUNNING: "running",
    store_daemon_pb2.REPLICA_OPERATION_STATE_SUCCESS: "success",
    store_daemon_pb2.REPLICA_OPERATION_STATE_FAILED: "failed",
    store_daemon_pb2.REPLICA_OPERATION_STATE_CANCELLED: "cancelled",
    store_daemon_pb2.REPLICA_OPERATION_STATE_DEGRADED: "degraded",
}

_GLOBAL_STATE_TO_OP_STATE: dict[int, OperationState] = {
    operation_pb2.OPERATION_STATE_PENDING: "pending",
    operation_pb2.OPERATION_STATE_RUNNING: "running",
    operation_pb2.OPERATION_STATE_SUCCESS: "success",
    operation_pb2.OPERATION_STATE_FAILED: "failed",
    operation_pb2.OPERATION_STATE_CANCELLED: "cancelled",
    operation_pb2.OPERATION_STATE_DEGRADED: "degraded",
}


def _timestamp_to_epoch_ms(ts: timestamp_pb2.Timestamp | None) -> int | None:
    if ts is None:
        return None
    try:
        return int(ts.seconds * 1000 + ts.nanos // 1_000_000)
    except Exception:  # noqa: BLE001
        return None


def _coerce_grpc_status_code(code: object) -> ArtifactStatusCode:
    if isinstance(code, str):
        name = code
    else:
        try:
            name = str(code.name)  # type: ignore[attr-defined]
        except Exception:
            name = "UNKNOWN"
    allowed: set[str] = {
        "OK",
        "CANCELLED",
        "UNKNOWN",
        "INVALID_ARGUMENT",
        "DEADLINE_EXCEEDED",
        "NOT_FOUND",
        "ALREADY_EXISTS",
        "PERMISSION_DENIED",
        "RESOURCE_EXHAUSTED",
        "FAILED_PRECONDITION",
        "ABORTED",
        "OUT_OF_RANGE",
        "UNIMPLEMENTED",
        "INTERNAL",
        "UNAVAILABLE",
        "DATA_LOSS",
        "UNAUTHENTICATED",
    }
    return name if name in allowed else "UNKNOWN"  # type: ignore[return-value]


def _retryable_for_grpc(code: object) -> bool:
    return code in {
        grpc.StatusCode.UNAVAILABLE,
        grpc.StatusCode.DEADLINE_EXCEEDED,
        grpc.StatusCode.INTERNAL,
        grpc.StatusCode.UNKNOWN,
    }


def _raise_api_error(err: Exception, *, cause: BaseException | None) -> NoReturn:
    if debug_errors_enabled() and cause is not None:
        raise err from cause
    raise err from None


def _append_debug_hint(message: str) -> str:
    if debug_errors_enabled():
        return message
    return f"{message}\nDebug: {debug_errors_hint()}"


def _grpc_details(exc: grpc.RpcError) -> str:
    try:
        details = exc.details()
    except Exception:  # noqa: BLE001
        return ""
    return str(details) if details else ""


def _format_rpc_message(action: str, exc: grpc.RpcError) -> str:
    details = _grpc_details(exc)
    base = f"{action} failed"
    if details:
        return f"{base}: {details}"
    return _append_debug_hint(base)


def _daemon_status_to_operation_status(
    status: store_daemon_pb2.ReplicaOperationStatus,
    *,
    context: Mapping[str, str] | None,
) -> OperationStatus:
    state = _REPLICA_STATE_TO_OP_STATE.get(int(status.state), "running")
    error: OperationError | None = None
    if status.HasField("error"):
        error = OperationError(
            status_code=str(status.error.status_code or "UNKNOWN"),
            message=str(status.error.message or ""),
            retryable=bool(status.error.retryable),
            context=context,
        )
    return OperationStatus(
        state=state,
        message=str(status.message) if status.message else None,
        progress=float(status.progress) if status.progress else None,
        as_of_ms=_timestamp_to_epoch_ms(
            status.as_of if status.HasField("as_of") else None
        ),
        error=error,
    )


def _global_store_status_to_operation_status(
    status: operation_pb2.OperationStatus,
    *,
    context: Mapping[str, str] | None,
) -> OperationStatus:
    state = _GLOBAL_STATE_TO_OP_STATE.get(int(status.state), "running")
    error: OperationError | None = None
    if status.HasField("error"):
        error = OperationError(
            status_code=str(status.error.status_code or "UNKNOWN"),
            message=str(status.error.message or ""),
            retryable=bool(status.error.retryable),
            context=context,
        )
    return OperationStatus(
        state=state,
        message=str(status.message) if status.message else None,
        progress=float(status.progress) if status.progress else None,
        as_of_ms=_timestamp_to_epoch_ms(
            status.as_of if status.HasField("as_of") else None
        ),
        error=error,
    )


class DaemonReplicaOperation(Operation[T]):
    """Daemon-scoped operation backed by replica_uuid + Wait/Query/Release RPCs."""

    def __init__(
        self,
        *,
        operation_id: str,
        runtime_ref: "weakref.ReferenceType[object]",
        ctx: CallContext | None,
        result_factory: Callable[[], T],
    ) -> None:
        self.operation_id = str(operation_id)
        self._runtime_ref = runtime_ref
        self._ctx = ctx
        self._request_id = (ctx.request_id if ctx else None) or uuid.uuid4().hex
        self._created_at = time.monotonic()
        self._result_factory = result_factory

    def _runtime(self) -> _Runtime:
        runtime = self._runtime_ref()
        if runtime is None or getattr(runtime, "closed", False):
            raise ArtifactError(
                "Store runtime is closed",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        return cast(_Runtime, runtime)

    def _ctx_remaining_timeout_s(self) -> float | None:
        if self._ctx is None or self._ctx.deadline_ms is None:
            return None
        elapsed_s = time.monotonic() - self._created_at
        remaining_s = (float(self._ctx.deadline_ms) / 1000.0) - elapsed_s
        return max(0.0, remaining_s)

    def status(self) -> OperationStatus:
        runtime = self._runtime()
        client = runtime.ensure_client()
        ticket = store_daemon_pb2.ReplicaTicket(replica_uuid=self.operation_id)
        try:
            resp = client.query_replica_status(ticket)
        except grpc.RpcError as exc:  # noqa: BLE001
            code = exc.code()
            error = ArtifactError(
                _format_rpc_message("QueryReplicaStatus", exc),
                status_code=_coerce_grpc_status_code(code),
                retryable=_retryable_for_grpc(code),
            )
            _raise_api_error(error, cause=exc)

        context = {
            "request_id": self._request_id,
            "operation_id": self.operation_id,
            "daemon_endpoint": str(getattr(runtime, "daemon_endpoint", "")),
        }
        return _daemon_status_to_operation_status(resp.status, context=context)

    def wait(self, *, timeout_s: float | None = None) -> T:
        runtime = self._runtime()
        client = runtime.ensure_client()

        overall_timeout_s = timeout_s
        ctx_remaining = self._ctx_remaining_timeout_s()
        if ctx_remaining is not None:
            overall_timeout_s = (
                ctx_remaining
                if overall_timeout_s is None
                else max(0.0, min(float(overall_timeout_s), ctx_remaining))
            )
            if overall_timeout_s <= 0:
                raise OperationTimeoutError(
                    "Operation deadline exceeded (ctx.deadline_ms)",
                    retryable=True,
                )

        deadline = (
            None
            if overall_timeout_s is None
            else time.monotonic() + float(overall_timeout_s)
        )

        ticket = store_daemon_pb2.ReplicaTicket(replica_uuid=self.operation_id)
        context = {
            "request_id": self._request_id,
            "operation_id": self.operation_id,
            "daemon_endpoint": str(getattr(runtime, "daemon_endpoint", "")),
        }

        while True:
            per_call_timeout_s: float | None = None
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise OperationTimeoutError(
                        "Operation wait timeout expired",
                        retryable=True,
                    )
                per_call_timeout_s = min(remaining, 600.0)

            try:
                resp = client.wait_replica_status(
                    ticket,
                    timeout_ms=None
                    if per_call_timeout_s is None
                    else int(per_call_timeout_s * 1000),
                )
            except grpc.RpcError as exc:  # noqa: BLE001
                code = exc.code()
                error = ArtifactError(
                    _format_rpc_message("WaitReplicaStatus", exc),
                    status_code=_coerce_grpc_status_code(code),
                    retryable=_retryable_for_grpc(code),
                )
                _raise_api_error(error, cause=exc)

            status = _daemon_status_to_operation_status(resp.status, context=context)
            if status.state == "success":
                return self._result_factory()
            if status.state in {"failed", "cancelled"}:
                if status.error is not None:
                    raise ArtifactError(
                        status.error.message or "Operation failed",
                        status_code=_coerce_grpc_status_code(status.error.status_code),
                        retryable=bool(status.error.retryable),
                    )
                raise ArtifactError(
                    "Operation failed",
                    status_code="UNKNOWN",
                    retryable=False,
                )
            if status.state == "degraded":
                if deadline is None:
                    continue
                raise OperationTimeoutError(
                    status.error.message
                    if status.error is not None and status.error.message
                    else "Operation wait timeout expired",
                    retryable=True,
                )

    def cancel(self) -> bool:
        runtime = self._runtime_ref()
        if runtime is None or getattr(runtime, "closed", False):
            return False
        client = runtime.ensure_client()
        ticket = store_daemon_pb2.ReplicaTicket(replica_uuid=self.operation_id)
        try:
            resp = client.release_replica(ticket)
        except grpc.RpcError:
            return False
        return bool(resp.released)


class _DaemonGlobalStoreClient(Protocol):
    def get_operation(
        self, operation_id: str, *, timeout_s: float = 10.0
    ) -> operation_pb2.GetOperationResponse: ...

    def wait_operation(
        self,
        operation_id: str,
        *,
        timeout_ms: int,
        timeout_s: float,
    ) -> operation_pb2.GetOperationResponse: ...


class DaemonGlobalStoreOperation(Operation[T]):
    """Daemon-scoped operation backed by Global Store Operation[T]."""

    def __init__(
        self,
        *,
        operation_id: str,
        runtime_ref: "weakref.ReferenceType[object]",
        ctx: CallContext | None,
        context: Mapping[str, str] | None,
        result_factory: Callable[[operation_pb2.GetOperationResponse], T],
    ) -> None:
        self.operation_id = str(operation_id)
        self._runtime_ref = runtime_ref
        self._ctx = ctx
        self._created_at = time.monotonic()
        self._context = context
        self._result_factory = result_factory

    def _runtime(self) -> _Runtime:
        runtime = self._runtime_ref()
        if runtime is None or getattr(runtime, "closed", False):
            raise ArtifactError(
                "Store runtime is closed",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        return cast(_Runtime, runtime)

    def _client(self) -> _DaemonGlobalStoreClient:
        return cast(_DaemonGlobalStoreClient, self._runtime().ensure_client())

    def _ctx_remaining_timeout_s(self) -> float | None:
        if self._ctx is None or self._ctx.deadline_ms is None:
            return None
        elapsed_s = time.monotonic() - self._created_at
        remaining_s = (float(self._ctx.deadline_ms) / 1000.0) - elapsed_s
        return max(0.0, remaining_s)

    def status(self) -> OperationStatus:
        timeout_s = self._ctx_remaining_timeout_s()
        if timeout_s is not None and timeout_s <= 0:
            return OperationStatus(
                state="degraded",
                message="CallContext deadline exceeded",
                progress=0.0,
                as_of_ms=int(time.time() * 1000),
                error=OperationError(
                    status_code="DEADLINE_EXCEEDED",
                    message="CallContext deadline exceeded",
                    retryable=True,
                    context=self._context,
                ),
            )

        try:
            resp = self._client().get_operation(
                self.operation_id,
                timeout_s=timeout_s if timeout_s is not None else 10.0,
            )
        except grpc.RpcError as exc:
            code = exc.code()
            message = _format_rpc_message("GetOperation", exc)
            return OperationStatus(
                state="degraded",
                message=message,
                progress=0.0,
                as_of_ms=int(time.time() * 1000),
                error=OperationError(
                    status_code=_coerce_grpc_status_code(code),
                    message=message,
                    retryable=_retryable_for_grpc(code),
                    context=self._context,
                ),
            )

        return _global_store_status_to_operation_status(
            resp.status, context=self._context
        )

    def wait(self, *, timeout_s: float | None = None) -> T:
        overall_timeout_s = timeout_s
        ctx_remaining = self._ctx_remaining_timeout_s()
        if ctx_remaining is not None:
            overall_timeout_s = (
                ctx_remaining
                if overall_timeout_s is None
                else max(0.0, min(float(overall_timeout_s), ctx_remaining))
            )
            if overall_timeout_s <= 0:
                raise OperationTimeoutError(
                    "Operation deadline exceeded (ctx.deadline_ms)",
                    retryable=True,
                )

        deadline = (
            None
            if overall_timeout_s is None
            else time.monotonic() + float(overall_timeout_s)
        )
        wait_chunk_s = 30.0
        while True:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise OperationTimeoutError(
                        "Operation wait timeout expired", retryable=True
                    )
                wait_chunk_s = max(0.05, min(wait_chunk_s, remaining))

            wait_ms = int(wait_chunk_s * 1000)
            try:
                resp = self._client().wait_operation(
                    self.operation_id,
                    timeout_ms=wait_ms,
                    timeout_s=float(wait_chunk_s) + 5.0,
                )
            except grpc.RpcError as exc:
                code = exc.code()
                if code == grpc.StatusCode.DEADLINE_EXCEEDED:
                    error = OperationTimeoutError(
                        "Operation wait timeout expired", retryable=True
                    )
                    _raise_api_error(error, cause=exc)
                error = ArtifactError(
                    _format_rpc_message("WaitOperation", exc),
                    status_code=_coerce_grpc_status_code(code),
                    retryable=_retryable_for_grpc(code),
                )
                _raise_api_error(error, cause=exc)

            status = _global_store_status_to_operation_status(
                resp.status, context=self._context
            )
            if status.state == "success":
                return self._result_factory(resp)
            if status.state in {"failed", "cancelled"}:
                if status.error is not None:
                    raise ArtifactError(
                        status.error.message or "Operation failed",
                        status_code=_coerce_grpc_status_code(status.error.status_code),
                        retryable=bool(status.error.retryable),
                    )
                raise ArtifactError(
                    "Operation failed",
                    status_code="UNKNOWN",
                    retryable=False,
                )
            if deadline is None:
                continue

    def cancel(self) -> bool:
        return False


class PollingOperation(Operation[T]):
    def __init__(
        self,
        *,
        operation_id: str,
        status_fn: Callable[[], OperationStatus],
        result_fn: Callable[[], T],
        cancel_fn: Callable[[], bool] | None = None,
        ctx: CallContext | None = None,
    ) -> None:
        self.operation_id = str(operation_id)
        self._status_fn = status_fn
        self._result_fn = result_fn
        self._cancel_fn = cancel_fn
        self._ctx = ctx
        self._created_at = time.monotonic()

    def _ctx_remaining_timeout_s(self) -> float | None:
        if self._ctx is None or self._ctx.deadline_ms is None:
            return None
        elapsed_s = time.monotonic() - self._created_at
        remaining_s = (float(self._ctx.deadline_ms) / 1000.0) - elapsed_s
        return max(0.0, remaining_s)

    def status(self) -> OperationStatus:
        return self._status_fn()

    def wait(self, *, timeout_s: float | None = None) -> T:
        overall_timeout_s = timeout_s
        ctx_remaining = self._ctx_remaining_timeout_s()
        if ctx_remaining is not None:
            overall_timeout_s = (
                ctx_remaining
                if overall_timeout_s is None
                else max(0.0, min(float(overall_timeout_s), ctx_remaining))
            )
            if overall_timeout_s <= 0:
                raise OperationTimeoutError(
                    "Operation deadline exceeded (ctx.deadline_ms)",
                    retryable=True,
                )

        deadline = (
            None
            if overall_timeout_s is None
            else time.monotonic() + float(overall_timeout_s)
        )
        sleep_s = 0.05
        while True:
            status = self.status()
            if status.state == "success":
                return self._result_fn()
            if status.state in {"failed", "cancelled"}:
                if status.error is not None:
                    raise ArtifactError(
                        status.error.message or "Operation failed",
                        status_code=_coerce_grpc_status_code(status.error.status_code),
                        retryable=bool(status.error.retryable),
                    )
                raise ArtifactError(
                    "Operation failed",
                    status_code="UNKNOWN",
                    retryable=False,
                )
            if status.state == "degraded":
                if deadline is None:
                    continue
                raise OperationTimeoutError(
                    status.error.message
                    if status.error is not None and status.error.message
                    else "Operation wait timeout expired",
                    retryable=True,
                )
            if deadline is not None and time.monotonic() > deadline:
                raise OperationTimeoutError(
                    "Operation wait timeout expired", retryable=True
                )
            time.sleep(sleep_s)
            sleep_s = min(sleep_s * 1.2, 0.5)

    def cancel(self) -> bool:
        if self._cancel_fn is None:
            return False
        return bool(self._cancel_fn())


__all__ = [
    "DaemonReplicaOperation",
    "DaemonGlobalStoreOperation",
    "Operation",
    "OperationError",
    "OperationState",
    "OperationStatus",
    "OperationTimeoutError",
    "PollingOperation",
    "TimeoutErrorDetails",
]
