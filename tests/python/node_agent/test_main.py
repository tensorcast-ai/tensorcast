#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tensorcast.node_agent import __main__ as node_agent_main


class _DaemonClient:
    def __init__(self, *, worker_id: str = "", daemon_id: str = "") -> None:
        self._response = SimpleNamespace(worker_id=worker_id, daemon_id=daemon_id)

    def get_worker_status(self) -> object:
        return self._response


def test_resolve_worker_id_from_daemon_returns_registered_worker_id() -> None:
    client = _DaemonClient(worker_id="worker-a", daemon_id="daemon-a")

    worker_id = node_agent_main._resolve_worker_id_from_daemon(
        client, daemon_id="daemon-a"
    )

    assert worker_id == "worker-a"


def test_resolve_worker_id_from_daemon_returns_none_when_unregistered() -> None:
    client = _DaemonClient(worker_id="", daemon_id="daemon-a")

    worker_id = node_agent_main._resolve_worker_id_from_daemon(
        client, daemon_id="daemon-a"
    )

    assert worker_id is None


def test_resolve_worker_id_from_daemon_fails_on_identity_mismatch() -> None:
    client = _DaemonClient(worker_id="worker-a", daemon_id="daemon-b")

    with pytest.raises(RuntimeError, match="Connected daemon identity does not match"):
        node_agent_main._resolve_worker_id_from_daemon(client, daemon_id="daemon-a")
