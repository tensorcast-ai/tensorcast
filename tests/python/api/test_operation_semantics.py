#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import weakref

import pytest

from tensorcast.api.operation import DaemonReplicaOperation, OperationTimeoutError
from tensorcast.proto.daemon.v2 import store_daemon_pb2


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

    def __init__(self, client: _Client) -> None:
        self._client = client

    def ensure_client(self) -> _Client:
        return self._client


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

