#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import weakref

import pytest

import tensorcast.api as api
from tensorcast.api import operation as operation_module
from tensorcast.api.context import CallContext
from tensorcast.api.operation import (
    DaemonGlobalStoreOperation,
    DaemonReplicaOperation,
    OperationStatus,
    OperationTimeoutError,
    PollingOperation,
)
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.proto.operation.v1 import operation_pb2


class _Client:
    def __init__(self) -> None:
        self.query_state = store_daemon_pb2.REPLICA_OPERATION_STATE_RUNNING
        self.wait_state = store_daemon_pb2.REPLICA_OPERATION_STATE_SUCCESS
        self.wait_error: store_daemon_pb2.ReplicaOperationError | None = None
        self.released: list[str] = []

    def query_replica_status(self, ticket: store_daemon_pb2.ReplicaTicket):
        resp = store_daemon_pb2.QueryReplicaStatusResponse()
        resp.ticket.replica_uuid = ticket.replica_uuid
        resp.status.state = self.query_state
        return resp

    def wait_replica_status(
        self, ticket: store_daemon_pb2.ReplicaTicket, *, timeout_ms: int | None
    ):
        del timeout_ms
        resp = store_daemon_pb2.WaitReplicaStatusResponse()
        resp.ticket.replica_uuid = ticket.replica_uuid
        resp.status.state = self.wait_state
        if self.wait_error is not None:
            resp.status.error.CopyFrom(self.wait_error)
        return resp

    def release_replica(self, ticket: store_daemon_pb2.ReplicaTicket):
        self.released.append(ticket.replica_uuid)
        return store_daemon_pb2.ReleaseReplicaResponse(released=True)


class _Runtime:
    daemon_endpoint = "daemon"
    closed = False

    def __init__(self, client: object) -> None:
        self._client = client

    def ensure_client(self) -> object:
        return self._client


class _GlobalClient:
    def __init__(self) -> None:
        self.get_state = operation_pb2.OPERATION_STATE_RUNNING
        self.wait_state = operation_pb2.OPERATION_STATE_SUCCESS
        self.error: operation_pb2.OperationError | None = None
        self.kind = "seal_assembly"
        self.target_artifact_id = "assembly-1"

    def get_operation(self, operation_id: str, *, timeout_s: float = 10.0):
        del timeout_s
        resp = operation_pb2.GetOperationResponse()
        resp.ref.operation_id = operation_id
        resp.ref.kind = self.kind
        resp.ref.target_artifact_id = self.target_artifact_id
        resp.status.state = self.get_state
        if self.error is not None:
            resp.status.error.CopyFrom(self.error)
        return resp

    def wait_operation(
        self, operation_id: str, *, timeout_ms: int, timeout_s: float
    ):
        del timeout_ms, timeout_s
        resp = operation_pb2.GetOperationResponse()
        resp.ref.operation_id = operation_id
        resp.ref.kind = self.kind
        resp.ref.target_artifact_id = self.target_artifact_id
        resp.status.state = self.wait_state
        if self.error is not None:
            resp.status.error.CopyFrom(self.error)
        return resp


def test_daemon_replica_operation_success_wait() -> None:
    client = _Client()
    runtime = _Runtime(client)
    op = DaemonReplicaOperation(
        operation_id="op-1",
        runtime_ref=weakref.ref(runtime),
        ctx=None,
        result_factory=lambda: "ok",
    )
    assert op.wait(timeout_s=0.1) == "ok"


def test_daemon_replica_operation_failed_wait_raises() -> None:
    client = _Client()
    client.wait_state = store_daemon_pb2.REPLICA_OPERATION_STATE_FAILED
    client.wait_error = store_daemon_pb2.ReplicaOperationError(
        status_code="INTERNAL",
        message="boom",
        retryable=False,
    )
    runtime = _Runtime(client)
    op = DaemonReplicaOperation(
        operation_id="op-2",
        runtime_ref=weakref.ref(runtime),
        ctx=None,
        result_factory=lambda: "unreachable",
    )
    with pytest.raises(Exception) as excinfo:
        op.wait(timeout_s=0.1)
    assert "boom" in str(excinfo.value)


def test_daemon_replica_operation_degraded_wait_times_out() -> None:
    client = _Client()
    client.wait_state = store_daemon_pb2.REPLICA_OPERATION_STATE_DEGRADED
    client.wait_error = store_daemon_pb2.ReplicaOperationError(
        status_code="DEADLINE_EXCEEDED",
        message="timeout",
        retryable=True,
    )
    runtime = _Runtime(client)
    op = DaemonReplicaOperation(
        operation_id="op-3",
        runtime_ref=weakref.ref(runtime),
        ctx=None,
        result_factory=lambda: "unreachable",
    )
    with pytest.raises(OperationTimeoutError):
        op.wait(timeout_s=0.001)


def test_daemon_replica_operation_cancel() -> None:
    client = _Client()
    runtime = _Runtime(client)
    op = DaemonReplicaOperation(
        operation_id="op-4",
        runtime_ref=weakref.ref(runtime),
        ctx=None,
        result_factory=lambda: "unreachable",
    )
    assert op.cancel() is True
    assert client.released == ["op-4"]


def test_polling_operation_respects_ctx_deadline() -> None:
    ctx = CallContext(deadline_ms=0)
    op = PollingOperation(
        operation_id="op-5",
        status_fn=lambda: OperationStatus(state="running"),
        result_fn=lambda: "ok",
        ctx=ctx,
    )
    with pytest.raises(OperationTimeoutError):
        op.wait()


def test_public_operation_surface_excludes_internal_routed_carriers() -> None:
    forbidden = {
        "AuthorityAttachmentRef",
        "DelegationEnvelope",
        "OwnerStageReply",
        "RoutedAuthorityRequest",
    }

    for name in forbidden:
        assert name not in api.__all__
        assert not hasattr(operation_module, name)


def test_daemon_global_store_operation_retains_explicit_operation_ref() -> None:
    client = _GlobalClient()
    client.get_state = operation_pb2.OPERATION_STATE_FAILED
    client.error = operation_pb2.OperationError(
        status_code="FAILED_PRECONDITION",
        message="no replay target",
        retryable=False,
    )
    client.kind = "seal_assembly"
    client.target_artifact_id = "assembly-6"
    runtime = _Runtime(client)
    operation_ref = operation_pb2.OperationRef(
        operation_id="op-6",
        kind="seal_assembly",
        target_artifact_id="assembly-6",
    )
    op = DaemonGlobalStoreOperation(
        operation_id="op-6",
        runtime_ref=weakref.ref(runtime),
        ctx=None,
        context={"request_id": "req-6"},
        result_factory=lambda _: "unreachable",
        operation_ref=operation_ref,
    )

    status = op.status()

    assert status.state == "failed"
    assert status.error is not None
    assert status.error.context is not None
    assert status.error.context["request_id"] == "req-6"
    assert status.error.context["operation_kind"] == "seal_assembly"
    assert status.error.context["target_artifact_id"] == "assembly-6"
    assert status.error.context["daemon_endpoint"] == "daemon"
    descriptor = op._operation_ref
    assert descriptor.kind == "seal_assembly"
    assert descriptor.target_artifact_id == "assembly-6"


def test_daemon_global_store_operation_refreshes_operation_ref_from_backend() -> None:
    client = _GlobalClient()
    client.get_state = operation_pb2.OPERATION_STATE_FAILED
    client.kind = "seal_assembly"
    client.target_artifact_id = "assembly-7"
    client.error = operation_pb2.OperationError(
        status_code="FAILED_PRECONDITION",
        message="owner unavailable",
        retryable=True,
    )
    runtime = _Runtime(client)
    op = DaemonGlobalStoreOperation(
        operation_id="op-7",
        runtime_ref=weakref.ref(runtime),
        ctx=None,
        context={},
        result_factory=lambda _: "unreachable",
    )

    status = op.status()

    assert status.state == "failed"
    assert status.error is not None
    assert status.error.context is not None
    assert status.error.context["operation_kind"] == "seal_assembly"
    assert status.error.context["target_artifact_id"] == "assembly-7"
    descriptor = op._operation_ref
    assert descriptor.operation_id == "op-7"
    assert descriptor.kind == "seal_assembly"
    assert descriptor.target_artifact_id == "assembly-7"
