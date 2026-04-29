#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import hashlib
import importlib
import uuid
import weakref
from typing import Any

import tensorcast as tc
from tensorcast.api._materialize import MaterializationPayload
from tensorcast.api.context import TransportSchedulingGroup
from tensorcast.api.store.artifact import Artifact
from tensorcast.common.selection_identity import (
    compute_selection_hash,
)
from tensorcast.proto.daemon.v2 import store_daemon_pb2

artifact_module = importlib.import_module("tensorcast.api.store.artifact")


class _Client:
    def get_artifact_index_by_id(self, artifact_id: str) -> bytes:
        del artifact_id
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


def test_transport_scheduling_group_rejects_invalid_values() -> None:
    invalid_cases = [
        {"group_kind": "", "group_id": "model:v1", "total_parts": 2, "part_id": "d0"},
        {"group_kind": "weight_broadcast", "group_id": "", "total_parts": 2, "part_id": "d0"},
        {"group_kind": "weight_broadcast", "group_id": "model:v1", "total_parts": 0, "part_id": "d0"},
        {"group_kind": "weight_broadcast", "group_id": "model:v1", "total_parts": 2, "part_id": ""},
        {"group_kind": "weight_broadcast", "group_id": "model:v1", "total_parts": 2, "part_id": "d0", "priority": -1},
        {"group_kind": "weight_broadcast", "group_id": "model:v1", "total_parts": 2, "part_id": "d0", "epoch": -1},
    ]

    for kwargs in invalid_cases:
        try:
            TransportSchedulingGroup(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected invalid transport group: {kwargs}")


def test_context_accepts_typed_transport_group() -> None:
    group = tc.TransportSchedulingGroup(
        group_kind="weight_broadcast",
        group_id="model-a:v42",
        epoch=42,
        total_parts=8,
        part_id="daemon-3",
    )

    ctx = tc.context(request_id="req-1", transport_group=group)

    assert ctx.transport_group == group
    assert tc.TransportSchedulingGroup is TransportSchedulingGroup


def test_prefetch_uses_deterministic_operation_id(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(artifact_module, "device_uuid_for", lambda _device_id: "uuid-0")
    store = _Store()
    artifact = Artifact(store_ref=weakref.ref(store), artifact_id="aid")
    ctx = tc.context(request_id="req-1", idempotency_key="idem-1")

    op = artifact.prefetch(device="cuda:0", ctx=ctx)

    daemon_id = store._runtime.daemon_id
    selection_hash = compute_selection_hash(
        view_id="",
        view_subset_hash=None,
    ).hex()
    logical_layout_hash = artifact._build_artifact_selection().logical_layout_hash.hex()
    device_uuid = "uuid-0"
    action_fingerprint = (
        f"prefetch|daemon={daemon_id}|artifact=aid|layout={logical_layout_hash}"
        f"|selection={selection_hash}|device=0|device_uuid={device_uuid}|lease=NO_LEASE|v2"
    )
    ns = uuid.uuid5(uuid.NAMESPACE_DNS, "tensorcast.op.v1")
    idempotency_key_hex = hashlib.sha256(ctx.idempotency_key.encode("utf-8")).hexdigest()
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


def test_prefetch_forwards_typed_transport_group_hint() -> None:
    store = _Store()
    artifact = Artifact(store_ref=weakref.ref(store), artifact_id="aid")
    group = tc.TransportSchedulingGroup(
        group_kind="weight_broadcast",
        group_id="model-a:v42",
        epoch=42,
        total_parts=16,
        part_id="daemon-1",
        priority=7,
        request_id="explicit-transport-req",
    )

    artifact.prefetch(device="cuda:0", ctx=tc.context(transport_group=group))

    call = store._materialization.calls[0]
    assert call["transport_request_id"] == "explicit-transport-req"
    forwarded = call["transport_scheduling_group"]
    assert forwarded.group_kind == "weight_broadcast"
    assert forwarded.group_id == "model-a:v42"
    assert forwarded.epoch == 42
    assert forwarded.total_parts == 16
    assert forwarded.part_id == "daemon-1"
    assert forwarded.priority == 7


def test_prefetch_derives_stable_transport_request_id_for_group(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(artifact_module, "device_uuid_for", lambda _device_id: "uuid-0")
    group = tc.TransportSchedulingGroup(
        group_kind="weight_broadcast",
        group_id="model-a:v42",
        epoch=42,
        total_parts=16,
        part_id="daemon-1",
    )
    ctx = tc.context(transport_group=group)
    first_store = _Store()
    second_store = _Store()
    first = Artifact(store_ref=weakref.ref(first_store), artifact_id="aid")
    second = Artifact(store_ref=weakref.ref(second_store), artifact_id="aid")

    first.prefetch(device="cuda:0", ctx=ctx)
    second.prefetch(device="cuda:0", ctx=ctx)

    first_request_id = first_store._materialization.calls[0]["transport_request_id"]
    second_request_id = second_store._materialization.calls[0]["transport_request_id"]
    assert first_request_id
    assert first_request_id == second_request_id
    assert first_request_id.startswith("prefetch:")


def test_prefetch_without_group_sends_no_transport_hint() -> None:
    store = _Store()
    artifact = Artifact(store_ref=weakref.ref(store), artifact_id="aid")

    artifact.prefetch(device="cuda:0")

    call = store._materialization.calls[0]
    assert call["transport_request_id"] is None
    assert call["transport_scheduling_group"] is None


def test_prefetch_forwards_broadcast_context_hint() -> None:
    store = _Store()
    artifact = Artifact(store_ref=weakref.ref(store), artifact_id="aid")
    ctx = tc.context(
        broadcast=tc.BroadcastContext(
            session_id="broadcast-session-1",
            strict_parent=True,
        )
    )

    artifact.prefetch(device="cuda:0", ctx=ctx)

    call = store._materialization.calls[0]
    assert call["broadcast_session_id"] == "broadcast-session-1"
    assert call["broadcast_strict_parent"] is True
