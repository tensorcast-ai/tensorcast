#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

from tensorcast.daemon_ctl import DaemonCtl
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2


class _FakeUnary:
    _method = b"/tensorcast.daemon.v2.StoreDaemonService/MaterializeReplica"

    def __init__(self) -> None:
        self.requests: list[store_daemon_pb2.MaterializeReplicaRequest] = []

    def __call__(self, request, timeout=None):  # noqa: ANN001, ANN204
        del timeout
        self.requests.append(request)
        response = store_daemon_pb2.MaterializeReplicaResponse()
        response.status = store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_ALLOCATED
        response.ticket.replica_uuid = request.replica_uuid
        return response


class _FakeStub:
    def __init__(self) -> None:
        self.MaterializeReplica = _FakeUnary()


def test_daemon_ctl_copies_broadcast_hint_to_materialize_request(monkeypatch) -> None:  # noqa: ANN001
    ctl = DaemonCtl.__new__(DaemonCtl)
    ctl.server_address = "fake-daemon"
    fake_stub = _FakeStub()
    ctl.stub_v2 = fake_stub
    ctl.stub = fake_stub
    monkeypatch.setattr(ctl, "_get_effective_pid", lambda: 123)
    monkeypatch.setattr(
        ctl,
        "_unary_call",
        lambda method, request, **kwargs: method(
            request,
            timeout=kwargs.get("timeout"),
        ),
    )
    selection = common_pb2.ArtifactSelection(artifact_id="aid")

    ctl.materialize_by_artifact_id_v2(
        selection=selection,
        replica_uuid="replica-1",
        device_uuid="device-uuid",
        wait_for_completion=False,
        return_response=True,
        broadcast_session_id="session-a",
        broadcast_strict_parent=True,
    )

    request = fake_stub.MaterializeReplica.requests[0]
    assert request.broadcast.session_id == "session-a"
    assert request.broadcast.strict_parent is True
