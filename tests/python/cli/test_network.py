#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import socket

from tensorcast.cli_utils import network


class _FakeSocket:
    def __init__(self, attempt: int):
        self.attempt = attempt
        self.closed = False

    def __enter__(self) -> "_FakeSocket":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()

    def close(self) -> None:  # noqa: D401
        self.closed = True


def test_wait_daemon_listening_retries_tcp_connect_until_success(
    monkeypatch,
) -> None:
    attempts: list[_FakeSocket] = []

    def _fake_create_connection(
        address: tuple[str, int], timeout: float
    ) -> _FakeSocket:
        assert address == ("127.0.0.1", 50052)
        assert timeout == 0.2
        sock = _FakeSocket(len(attempts))
        attempts.append(sock)
        if sock.attempt < 2:
            raise RuntimeError("daemon not listening yet")
        return sock

    monkeypatch.setattr(socket, "create_connection", _fake_create_connection)
    monkeypatch.setattr(
        network.time,
        "sleep",
        lambda _seconds: None,
    )

    ready_host = network.wait_daemon_listening("127.0.0.1", 50052, timeout=1.0)

    assert ready_host == "127.0.0.1"
    assert len(attempts) == 3
    assert attempts[-1].closed is True
