#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import uuid
import weakref
from hashlib import sha256
from typing import Any

import tensorcast as tc

from tensorcast.api._materialize import MaterializationPayload
from tensorcast.api.store.artifact import Artifact
from tensorcast.common.selection_identity import (
    compute_logical_layout_hash,
    compute_selection_hash,
)
from tensorcast.proto.daemon.v2 import store_daemon_pb2


class _Client:
    def get_artifact_index_by_id(self, artifact_id: str) -> bytes:  # noqa: ARG002
        return b"{}"

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
        self.invalidated: list[tuple[str, str]] = []

    def ensure_client(self) -> _Client:
        return self._client

    def get_artifact_index_cached(self, artifact_id: str):  # noqa: ANN001, ARG002
        return None

    def invalidate_artifact(self, artifact_id: str, *, reason: str) -> None:
        self.invalidated.append((str(artifact_id), str(reason)))

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
    idempotency_key_hex = sha256(ctx.idempotency_key.encode("utf-8")).hexdigest()
    selection_hash = compute_selection_hash(
        view_id="",
        view_subset_hash=None,
    ).hex()
    logical_layout_hash = compute_logical_layout_hash(
        index_bytes=b"{}", needs_view_index=False
    ).hex()
    action_fingerprint = (
        f"prefetch|daemon={daemon_id}|artifact=aid|layout={logical_layout_hash}|selection={selection_hash}"
        f"|device=0|lease=NO_LEASE|v1"
    )
    ns = uuid.uuid5(uuid.NAMESPACE_DNS, "tensorcast.op.v1")
    expected = str(uuid.uuid5(ns, f"{idempotency_key_hex}|{action_fingerprint}"))

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


def test_prefetch_without_ctx_generates_operation_id() -> None:
    store = _Store()
    artifact = Artifact(store_ref=weakref.ref(store), artifact_id="aid")

    op = artifact.prefetch(device="cuda:0")

    assert store._materialization.calls
    replica_uuid = str(store._materialization.calls[0]["replica_uuid"] or "")
    assert replica_uuid
    assert op.operation_id == replica_uuid
