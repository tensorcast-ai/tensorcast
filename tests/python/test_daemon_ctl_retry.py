#  Copyright (c) 2025, TensorCast Team.

import grpc
from types import SimpleNamespace

from tensorcast.daemon_ctl import DaemonCtl


class DummyRpcError(grpc.RpcError):
    def __init__(self, code: grpc.StatusCode):
        self._code = code

    def code(self) -> grpc.StatusCode:
        return self._code

    def details(self) -> str:
        return "dummy"


class BadStub:
    def GetServerConfig(self, request, timeout=None):  # noqa: N802
        raise DummyRpcError(grpc.StatusCode.UNAVAILABLE)


class GoodStub:
    def GetServerConfig(self, request, timeout=None):  # noqa: N802
        return SimpleNamespace(chunk_size=123, mem_pool_size=456)


def test_unary_call_rebinds_stub_after_refresh(monkeypatch):
    ctl = DaemonCtl("127.0.0.1:65535")
    ctl.stub = BadStub() # type: ignore[assignment]

    refreshed_count = {"n": 0}

    def fake_refresh_channel():
        refreshed_count["n"] += 1
        ctl.stub = GoodStub() # type: ignore[assignment]

    # Swap refresh to our fake which installs a working stub
    monkeypatch.setattr(ctl, "_refresh_channel", fake_refresh_channel)

    # Pass the bound method from the bad stub; retries=1 allows one retry after failure
    resp = ctl._unary_call(ctl.stub.GetServerConfig, request=None, retries=1)

    assert resp.chunk_size == 123
    assert resp.mem_pool_size == 456
    assert refreshed_count["n"] == 1


