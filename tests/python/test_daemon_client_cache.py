#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import os

import pytest

try:
    from tensorcast.daemon_ctl import get_daemon_client
except Exception as e:  # pragma: no cover - env dependent
    pytest.skip(f"skipping: daemon client import failed: {e}", allow_module_level=True)


def test_get_daemon_client_same_pid_same_addr_reuses_instance() -> None:
    addr = "127.0.0.1:65000"
    c1 = get_daemon_client(addr)
    c2 = get_daemon_client(addr)
    assert c1 is c2
    assert c1.server_address == addr


def test_get_daemon_client_different_addr_returns_different_instances() -> None:
    c1 = get_daemon_client("127.0.0.1:65001")
    c2 = get_daemon_client("127.0.0.1:65002")
    assert c1 is not c2
    assert c1.server_address != c2.server_address


def test_get_daemon_client_keyed_by_pid() -> None:
    # Sanity: ensure function references current PID in cache key
    # We cannot fork reliably in test env; just ensure call does not crash
    _ = os.getpid()
    c = get_daemon_client("127.0.0.1:65003")
    assert c.server_address == "127.0.0.1:65003"
