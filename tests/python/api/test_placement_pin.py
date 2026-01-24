#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import weakref

from google.protobuf import timestamp_pb2

from tensorcast.api.store.artifact import Artifact
from tensorcast.proto.daemon.v2 import store_daemon_pb2


class _Client:
    def __init__(self) -> None:
        self.created: list[tuple[str, str, int, int | None]] = []
        self.renewed: list[tuple[bytes, int]] = []
        self.released: list[bytes] = []

    def create_placement_lease(
        self,
        *,
        artifact_id: str,
        view_id: str,
        device_id: int,
        ttl_ms: int | None,
        timeout_s: float | None = None,
    ) -> store_daemon_pb2.CreatePlacementLeaseResponse:
        del timeout_s
        self.created.append((artifact_id, view_id, int(device_id), ttl_ms))
        resp = store_daemon_pb2.CreatePlacementLeaseResponse(
            lease_id=123,
            lease_token=b"token",
        )
        ts = timestamp_pb2.Timestamp()
        ts.FromSeconds(10)
        resp.expires_at.CopyFrom(ts)
        return resp

    def renew_placement_lease(
        self,
        *,
        lease_token: bytes,
        ttl_ms: int,
        timeout_s: float | None = None,
    ) -> store_daemon_pb2.RenewPlacementLeaseResponse:
        del timeout_s
        self.renewed.append((bytes(lease_token), int(ttl_ms)))
        resp = store_daemon_pb2.RenewPlacementLeaseResponse(lease_id=123)
        ts = timestamp_pb2.Timestamp()
        ts.FromSeconds(20)
        resp.expires_at.CopyFrom(ts)
        return resp

    def release_placement_lease(
        self, *, lease_token: bytes, timeout_s: float | None = None
    ) -> store_daemon_pb2.ReleasePlacementLeaseResponse:
        del timeout_s
        self.released.append(bytes(lease_token))
        return store_daemon_pb2.ReleasePlacementLeaseResponse(released=True)


class _Runtime:
    daemon_endpoint = "daemon"
    daemon_id = "daemon-1"
    closed = False

    def __init__(self) -> None:
        self._client = _Client()

    def ensure_client(self) -> _Client:
        return self._client


class _Store:
    def __init__(self) -> None:
        self._runtime = _Runtime()
        self._materialization = object()
        self.closed = False


def test_pin_device_residency_returns_operation_and_pin() -> None:
    store = _Store()
    artifact = Artifact(store_ref=weakref.ref(store), artifact_id="aid")
    op = artifact.pin_device_residency(device="cuda:0", ttl_ms=1000)
    pin = op.result(timeout_s=1.0)

    assert pin.pin_id == 123
    assert pin.daemon_id == "daemon-1"
    assert pin.artifact_id == "aid"
    assert pin.device_id == 0
    assert pin.expires_at_ms == 10_000

    renewed = pin.renew(ttl_ms=2000)
    assert renewed.pin_id == 123
    assert renewed.expires_at_ms == 20_000

    assert op.cancel() is True
    pin.release()
