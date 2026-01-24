#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import hashlib
import uuid
import weakref
from typing import Any

import tensorcast as tc

from tensorcast.api._materialize import MaterializationPayload
from tensorcast.api.store.artifact import Artifact
from tensorcast.proto.daemon.v2 import store_daemon_pb2


class _Client:
    def query_replica_status(self, ticket: store_daemon_pb2.ReplicaTicket):
        resp = store_daemon_pb2.QueryReplicaStatusResponse()
        resp.ticket.replica_uuid = ticket.replica_uuid
        resp.status.state = store_daemon_pb2.REPLICA_OPERATION_STATE_SUCCESS
        return resp

    def wait_replica_status(
        self, ticket: store_daemon_pb2.ReplicaTicket, *, timeout_ms: int | None
    ):
        del timeout_ms
        resp = store_daemon_pb2.WaitReplicaStatusResponse()
        resp.ticket.replica_uuid = ticket.replica_uuid
        resp.status.state = store_daemon_pb2.REPLICA_OPERATION_STATE_SUCCESS
        return resp

    def release_replica(self, ticket: store_daemon_pb2.ReplicaTicket):
        del ticket
        return store_daemon_pb2.ReleaseReplicaResponse(released=True)


class _Runtime:
    daemon_endpoint = "daemon"
    daemon_id = "daemon-1"
    session_id = "sess"
    closed = False

    def __init__(self) -> None:
        self._client = _Client()
        self.cached: list[Any] = []

    def ensure_client(self) -> _Client:
        return self._client

    def cache_artifact_index(self, entry: object) -> None:
        self.cached.append(entry)


class _Pipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def materialize_subset(self, **kwargs):
        self.calls.append(kwargs)
        replica_uuid = str(kwargs.get("replica_uuid") or "")
        payload = MaterializationPayload(
            artifact_id=str(kwargs.get("artifact_id") or ""),
            canonical_index_bytes=b"{}",
            descriptors=(),
            payload_iter=lambda: iter(()),
            replica_uuid=replica_uuid,
            ticket_replica_uuid=replica_uuid,
        )
        return payload, 0


class _Store:
    def __init__(self) -> None:
        self._runtime = _Runtime()
        self._materialization = _Pipeline()
        self.closed = False


def test_prefetch_uses_deterministic_operation_id() -> None:
    store = _Store()
    artifact = Artifact(store_ref=weakref.ref(store), artifact_id="aid")
    ctx = tc.context(request_id="req-1", idempotency_key="idem-1")

    op = artifact.prefetch(device="cuda:0", ctx=ctx)

    daemon_id = store._runtime.daemon_id
    selection_hash = hashlib.sha256("aid|".encode("utf-8")).hexdigest()
    action_fingerprint = (
        f"prefetch|daemon={daemon_id}|selection={selection_hash}|device=0|lease=NO_LEASE|v1"
    )
    ns = uuid.uuid5(uuid.NAMESPACE_DNS, "tensorcast.op.v1")
    expected = str(uuid.uuid5(ns, f"{ctx.idempotency_key}|{action_fingerprint}"))

    assert store._materialization.calls
    assert store._materialization.calls[0]["replica_uuid"] == expected
    assert (
        store._materialization.calls[0]["lease_mode"]
        == store_daemon_pb2.LeaseMode.LEASE_MODE_NO_LEASE
    )
    assert op.operation_id == expected

    replica = op.result(timeout_s=1.0)
    assert replica.operation_id == expected
    assert replica.daemon_id == daemon_id

