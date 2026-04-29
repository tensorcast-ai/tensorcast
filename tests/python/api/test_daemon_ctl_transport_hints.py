#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import torch

from tensorcast.daemon_ctl import DaemonCtl
from tensorcast.api._config import GetArtifactOptions
from tensorcast.api._materialize import materialize_artifact_v2
from tensorcast.api.context import TransportSchedulingGroup
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


def test_daemon_ctl_forwards_materialize_transport_hints(monkeypatch) -> None:  # noqa: ANN001
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
    group = store_daemon_pb2.TransportSchedulingGroupHint(
        group_kind="weight_broadcast",
        group_id="model-a:v42",
        total_parts=16,
        part_id="daemon-1",
        priority=7,
        epoch=42,
    )

    ctl.materialize_by_artifact_id_v2(
        selection=selection,
        replica_uuid="replica-1",
        device_uuid="device-uuid",
        wait_for_completion=False,
        return_response=True,
        transport_request_id="transport-req-1",
        transport_scheduling_group=group,
    )

    request = fake_stub.MaterializeReplica.requests[0]
    assert request.transport_request_id == "transport-req-1"
    assert request.transport_scheduling_group.group_kind == "weight_broadcast"
    assert request.transport_scheduling_group.group_id == "model-a:v42"
    assert request.transport_scheduling_group.total_parts == 16
    assert request.transport_scheduling_group.part_id == "daemon-1"
    assert request.transport_scheduling_group.priority == 7
    assert request.transport_scheduling_group.epoch == 42


class _FakeMaterializeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get_artifact_index_by_id(self, artifact_id: str) -> bytes:
        del artifact_id
        return b"{}"

    def materialize_by_artifact_id_v2(self, **kwargs):
        self.calls.append(kwargs)
        response = store_daemon_pb2.MaterializeReplicaResponse()
        response.status = store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_ALLOCATED
        response.artifact_id = "aid"
        response.canonical_index_bytes = b"{}"
        return response


def test_materialize_artifact_v2_converts_transport_group_to_daemon_proto() -> None:
    client = _FakeMaterializeClient()
    group = TransportSchedulingGroup(
        group_kind="weight_broadcast",
        group_id="model-a:v42",
        total_parts=16,
        part_id="daemon-1",
        priority=7,
        epoch=42,
    )

    materialize_artifact_v2(
        client=client,
        daemon_address="daemon",
        device_id=torch.device("cpu"),
        artifact_id="aid",
        key=None,
        options=GetArtifactOptions(
            wait_for_completion=False,
            enable_verification=False,
        ),
        transport_request_id="transport-req-1",
        transport_scheduling_group=group,
    )

    request = client.calls[0]
    assert request["transport_request_id"] == "transport-req-1"
    forwarded = request["transport_scheduling_group"]
    assert isinstance(forwarded, store_daemon_pb2.TransportSchedulingGroupHint)
    assert forwarded.group_kind == "weight_broadcast"
    assert forwarded.group_id == "model-a:v42"
    assert forwarded.total_parts == 16
    assert forwarded.part_id == "daemon-1"
    assert forwarded.priority == 7
    assert forwarded.epoch == 42
