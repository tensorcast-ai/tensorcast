#  Copyright (c) 2025-2026, TensorCast Team.

from types import SimpleNamespace

import grpc
import pytest

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
        # Mimic real GetServerConfigResponse field names
        return SimpleNamespace(tx_slice_bytes=123, mem_pool_size=456)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return float(self.now)

    def sleep(self, seconds: float) -> None:
        self.now += max(0.0, float(seconds))

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


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

    assert resp.tx_slice_bytes == 123
    assert resp.mem_pool_size == 456
    assert refreshed_count["n"] == 1


def test_unary_call_does_not_retry_after_budget_exhausted(monkeypatch):
    class SlowDeadlineStub:
        def __init__(self, clock: FakeClock) -> None:
            self.clock = clock
            self.calls = 0

        def GetServerConfig(self, request, timeout=None):  # noqa: N802
            self.calls += 1
            # Consume more than the total unary timeout budget.
            self.clock.advance(1.2)
            raise DummyRpcError(grpc.StatusCode.DEADLINE_EXCEEDED)

    clock = FakeClock()
    ctl = DaemonCtl("127.0.0.1:65535")
    stub = SlowDeadlineStub(clock)
    ctl.stub = stub # type: ignore[assignment]
    monkeypatch.setattr("tensorcast.daemon_ctl.time.monotonic", clock.monotonic)
    monkeypatch.setattr("tensorcast.daemon_ctl.time.sleep", clock.sleep)
    monkeypatch.setattr("tensorcast.daemon_ctl.random.random", lambda: 0.0)
    monkeypatch.setattr(ctl, "_refresh_channel", lambda: None)

    with pytest.raises(DummyRpcError):
        _ = ctl._unary_call(
            ctl.stub.GetServerConfig,
            request=None,
            timeout=1.0,
            retries=1,
        )

    assert stub.calls == 1


def test_unary_call_retry_uses_remaining_timeout_budget(monkeypatch):
    class FlakyStub:
        def __init__(self, clock: FakeClock) -> None:
            self.clock = clock
            self.calls = 0
            self.timeouts: list[float | int | None] = []

        def GetServerConfig(self, request, timeout=None):  # noqa: N802
            self.calls += 1
            self.timeouts.append(timeout)
            if self.calls == 1:
                self.clock.advance(0.3)
                raise DummyRpcError(grpc.StatusCode.UNAVAILABLE)
            return SimpleNamespace(tx_slice_bytes=1, mem_pool_size=2)

    clock = FakeClock()
    ctl = DaemonCtl("127.0.0.1:65535")
    stub = FlakyStub(clock)
    ctl.stub = stub # type: ignore[assignment]
    monkeypatch.setattr("tensorcast.daemon_ctl.time.monotonic", clock.monotonic)
    monkeypatch.setattr("tensorcast.daemon_ctl.time.sleep", clock.sleep)
    monkeypatch.setattr("tensorcast.daemon_ctl.random.random", lambda: 0.0)
    monkeypatch.setattr(ctl, "_refresh_channel", lambda: None)

    resp = ctl._unary_call(
        ctl.stub.GetServerConfig,
        request=None,
        timeout=1.0,
        retries=1,
    )

    assert resp.tx_slice_bytes == 1
    assert resp.mem_pool_size == 2
    assert stub.calls == 2
    assert len(stub.timeouts) == 2
    assert stub.timeouts[0] == pytest.approx(1.0, rel=0.0, abs=1e-3)
    # 1.0 total budget - 0.3 first attempt - 0.05 retry backoff.
    assert stub.timeouts[1] == pytest.approx(0.65, rel=0.0, abs=1e-3)
