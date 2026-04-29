#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import pytest

from tensorcast.api.store import Store
from tensorcast.api.store.types import ArtifactError
from tensorcast.daemon_ctl import DaemonCtl
from tensorcast.proto.daemon.v2 import store_daemon_pb2


class _DaemonClient:
    def __init__(self, response: store_daemon_pb2.CreateBroadcastSessionResponse) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def create_broadcast_session(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class _Runtime:
    daemon_endpoint = "daemon"
    daemon_id = "daemon-1"
    session_id = "sess"
    closed = False

    def __init__(self, client: _DaemonClient) -> None:
        self._client = client

    def ensure_client(self) -> _DaemonClient:
        return self._client


def test_store_create_broadcast_session_uses_daemon_client() -> None:
    client = _DaemonClient(
        store_daemon_pb2.CreateBroadcastSessionResponse(
            status=store_daemon_pb2.BROADCAST_SESSION_STATUS_OK,
            session_id="session-a",
        )
    )
    store = Store("daemon", runtime=_Runtime(client))

    handle = store.create_broadcast_session(
        session_id="session-a",
        artifact_id="artifact-a",
        target_daemon_ids=["daemon-a", "daemon-b"],
        target_worker_ids=["worker-a"],
        requested_view_id="view-a",
        epoch=7,
        fanout=2,
        root_replica_id="replica-root",
        strict_parent=True,
        max_attempts=3,
    )

    assert handle.session_id == "session-a"
    assert client.calls == [
        {
            "session_id": "session-a",
            "artifact_id": "artifact-a",
            "requested_view_id": "view-a",
            "epoch": 7,
            "fanout": 2,
            "target_worker_ids": ["worker-a"],
            "target_daemon_ids": ["daemon-a", "daemon-b"],
            "root_replica_id": "replica-root",
            "strict_parent": True,
            "max_attempts": 3,
        }
    ]


def test_store_create_broadcast_session_validates_required_fields() -> None:
    client = _DaemonClient(
        store_daemon_pb2.CreateBroadcastSessionResponse(
            status=store_daemon_pb2.BROADCAST_SESSION_STATUS_OK,
            session_id="unused",
        )
    )
    store = Store("daemon", runtime=_Runtime(client))

    with pytest.raises(ArtifactError, match="session_id is required"):
        store.create_broadcast_session(
            session_id="",
            artifact_id="artifact-a",
            fanout=2,
            target_daemon_ids=["daemon-a"],
        )

    with pytest.raises(ArtifactError, match="fanout must be > 0"):
        store.create_broadcast_session(
            session_id="session-a",
            artifact_id="artifact-a",
            fanout=0,
            target_daemon_ids=["daemon-a"],
        )

    assert client.calls == []


def test_daemon_ctl_create_broadcast_session_builds_request() -> None:
    client = DaemonCtl.__new__(DaemonCtl)
    client.server_address = "127.0.0.1:1"
    captured: list[store_daemon_pb2.CreateBroadcastSessionRequest] = []

    class _Stub:
        def CreateBroadcastSession(self):  # pragma: no cover - marker only
            raise AssertionError("fake unary call should intercept this")

    def _unary_call(method, request, *, timeout=None, retries=0, span=None):  # noqa: ANN001
        del method, timeout, retries, span
        captured.append(request)
        return store_daemon_pb2.CreateBroadcastSessionResponse(
            status=store_daemon_pb2.BROADCAST_SESSION_STATUS_OK,
            session_id="session-a",
        )

    client.stub_v2 = _Stub()
    client._unary_call = _unary_call

    response = client.create_broadcast_session(
        session_id="session-a",
        artifact_id="artifact-a",
        requested_view_id="view-a",
        epoch=9,
        fanout=4,
        target_worker_ids=["worker-a"],
        target_daemon_ids=["daemon-a", "daemon-b"],
        root_replica_id="replica-root",
        strict_parent=False,
        max_attempts=5,
        timeout_s=12.0,
    )

    assert response.session_id == "session-a"
    assert len(captured) == 1
    request = captured[0]
    assert request.session_id == "session-a"
    assert request.artifact_id == "artifact-a"
    assert request.requested_view_id == "view-a"
    assert request.epoch == 9
    assert request.fanout == 4
    assert list(request.target_worker_ids) == ["worker-a"]
    assert list(request.target_daemon_ids) == ["daemon-a", "daemon-b"]
    assert request.root_replica_id == "replica-root"
    assert request.strict_parent is False
    assert request.max_attempts == 5


def test_daemon_ctl_create_broadcast_session_defaults_max_attempts() -> None:
    client = DaemonCtl.__new__(DaemonCtl)
    client.server_address = "127.0.0.1:1"
    captured: list[store_daemon_pb2.CreateBroadcastSessionRequest] = []

    class _Stub:
        def CreateBroadcastSession(self):  # pragma: no cover - marker only
            raise AssertionError("fake unary call should intercept this")

    def _unary_call(method, request, *, timeout=None, retries=0, span=None):  # noqa: ANN001
        del method, timeout, retries, span
        captured.append(request)
        return store_daemon_pb2.CreateBroadcastSessionResponse(
            status=store_daemon_pb2.BROADCAST_SESSION_STATUS_OK,
            session_id="session-a",
        )

    client.stub_v2 = _Stub()
    client._unary_call = _unary_call

    client.create_broadcast_session(
        session_id="session-a",
        artifact_id="artifact-a",
        fanout=2,
        target_daemon_ids=["daemon-a"],
    )

    assert len(captured) == 1
    assert captured[0].max_attempts == 3


@pytest.mark.parametrize(
    ("status", "status_code"),
    [
        (store_daemon_pb2.BROADCAST_SESSION_STATUS_ERROR, "FAILED_PRECONDITION"),
        (store_daemon_pb2.BROADCAST_SESSION_STATUS_NOT_FOUND, "NOT_FOUND"),
    ],
)
def test_store_create_broadcast_session_maps_daemon_errors(
    status: int,
    status_code: str,
) -> None:
    client = _DaemonClient(
        store_daemon_pb2.CreateBroadcastSessionResponse(
            status=status,
            session_id="",
        )
    )
    store = Store("daemon", runtime=_Runtime(client))

    with pytest.raises(ArtifactError) as exc_info:
        store.create_broadcast_session(
            session_id="session-a",
            artifact_id="artifact-a",
            fanout=2,
            target_daemon_ids=["daemon-a"],
        )

    assert exc_info.value.status_code == status_code
