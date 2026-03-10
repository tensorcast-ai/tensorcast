#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorcast import runtime, startup
from tensorcast.cli_utils.paths import session_paths
from tensorcast.cli_utils.process import read_json_default


@pytest.fixture(autouse=True)
def _isolate_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TENSORCAST_HOME", str(tmp_path))
    monkeypatch.setattr(startup, "discover_daemon_config", lambda: None)


def test_auto_mode_connects_existing_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    daemon_address = "127.0.0.1:61001"
    dummy_client = object()

    monkeypatch.setattr(
        startup.runtime,
        "status",
        lambda _session_id=None: SimpleNamespace(daemon_address=daemon_address),
    )
    monkeypatch.setattr(startup, "ping_daemon", lambda _address: True)
    monkeypatch.setattr(
        "tensorcast.daemon_ctl.get_daemon_client", lambda _address: dummy_client
    )

    ctx = startup.init(mode="auto")
    try:
        assert ctx.is_owner is False
        assert ctx.address == daemon_address
        assert ctx.client is dummy_client
    finally:
        startup.shutdown()


def test_auto_mode_creates_daemon_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    started: dict[str, object] = {}
    daemon_address = "127.0.0.1:61002"

    def _fake_start(**kwargs):
        started.update(kwargs)
        return runtime.RuntimeSession(
            session_id="sess-auto",
            daemon_pid=4242,
            daemon_address=daemon_address,
            daemon_p2p_address="127.0.0.1:61003",
            logs_dir=session_paths("sess-auto").logs,
            started_at=time.time(),
            owner=True,
        )

    monkeypatch.setattr(startup.runtime, "status", lambda _session_id=None: None)
    monkeypatch.setattr(startup.runtime, "start", _fake_start)
    monkeypatch.setattr(startup, "ping_daemon", lambda _address: True)
    monkeypatch.setattr(
        "tensorcast.daemon_ctl.get_daemon_client", lambda _address: object()
    )
    monkeypatch.setattr(startup.runtime, "stop", lambda **_kwargs: None)

    ctx = startup.init(mode="auto", show_daemon_logs=False)
    try:
        assert ctx.is_owner is True
        assert ctx.address == daemon_address
        assert ctx.session_id == "sess-auto"
        assert started["session_id"] is None
        assert started["register_current"] is True
        assert started["ephemeral"] is False
        auto_state = read_json_default(startup._auto_state_path(), {})
        assert auto_state["status"] == "READY"
        assert auto_state["session_id"] == "sess-auto"
        assert auto_state["address"] == daemon_address
    finally:
        startup.shutdown()


def test_auto_mode_recovers_from_stale_starting_owner_dead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon_address = "127.0.0.1:64002"
    started: dict[str, object] = {}

    def _fake_start(**kwargs):
        started.update(kwargs)
        return runtime.RuntimeSession(
            session_id="sess-starting-recovered",
            daemon_pid=5454,
            daemon_address=daemon_address,
            daemon_p2p_address="127.0.0.1:64003",
            logs_dir=session_paths("sess-starting-recovered").logs,
            started_at=time.time(),
            owner=True,
        )

    monkeypatch.setattr(startup.runtime, "status", lambda _session_id=None: None)
    monkeypatch.setattr(startup.runtime, "start", _fake_start)
    monkeypatch.setattr(
        startup, "ping_daemon", lambda addr: bool(addr == daemon_address)
    )
    monkeypatch.setattr(
        "tensorcast.daemon_ctl.get_daemon_client", lambda _address: object()
    )
    monkeypatch.setattr(startup.runtime, "stop", lambda **_kwargs: None)

    startup.atomic_write_json(
        startup._auto_state_path(),
        {
            "schema_version": 1,
            "status": "STARTING",
            "epoch": 3,
            "owner_pid": 0,
            "owner_fingerprint": {},
            "config_hash": "stale-config-hash",
            "session_id": "sess-dead",
            "started_at": time.time(),
            "address": "",
            "logs_dir": "/tmp/daemon-logs",
            "error_code": "",
            "error_message": "",
        },
    )

    ctx = startup.init(mode="auto", show_daemon_logs=False)
    try:
        assert ctx.is_owner is True
        assert ctx.address == daemon_address
        assert started["session_id"] is None
        auto_state = read_json_default(startup._auto_state_path(), {})
        assert auto_state["status"] == "READY"
        assert auto_state["session_id"] == "sess-starting-recovered"
        assert auto_state["address"] == daemon_address
    finally:
        startup.shutdown()


def test_auto_mode_rejects_config_mismatch_when_owner_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(startup.runtime, "status", lambda _session_id=None: None)
    monkeypatch.setattr(startup, "ping_daemon", lambda _address: False)
    monkeypatch.setattr(
        startup.runtime,
        "start",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not start")),
    )

    startup.atomic_write_json(
        startup._auto_state_path(),
        {
            "schema_version": 1,
            "status": "STARTING",
            "epoch": 7,
            "owner_pid": os.getpid(),
            "owner_fingerprint": startup.instance_fingerprint(),
            "config_hash": "different-config-hash",
            "session_id": "sess-live",
            "started_at": time.time(),
            "address": "",
            "logs_dir": "/tmp/live-logs",
            "error_code": "",
            "error_message": "",
        },
    )

    with pytest.raises(RuntimeError, match="AUTO_CONFIG_MISMATCH"):
        startup.init(mode="auto")


def test_auto_mode_records_failed_state_on_start_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(startup.runtime, "status", lambda _session_id=None: None)
    monkeypatch.setattr(startup, "ping_daemon", lambda _address: False)
    monkeypatch.setattr(
        startup.runtime,
        "start",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("start boom")),
    )

    with pytest.raises(RuntimeError, match="start boom"):
        startup.init(mode="auto")

    auto_state = read_json_default(startup._auto_state_path(), {})
    assert auto_state["status"] == "FAILED"
    assert auto_state["error_code"] == "AUTO_START_FAILED"
    assert "start boom" in auto_state["error_message"]


def test_auto_mode_recovers_from_stale_ready_owner_dead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon_address = "127.0.0.1:62002"
    started: dict[str, object] = {}

    def _fake_start(**kwargs):
        started.update(kwargs)
        return runtime.RuntimeSession(
            session_id="sess-recovered",
            daemon_pid=5252,
            daemon_address=daemon_address,
            daemon_p2p_address="127.0.0.1:62003",
            logs_dir=session_paths("sess-recovered").logs,
            started_at=time.time(),
            owner=True,
        )

    monkeypatch.setattr(startup.runtime, "status", lambda _session_id=None: None)
    monkeypatch.setattr(startup.runtime, "start", _fake_start)
    monkeypatch.setattr(
        startup,
        "ping_daemon",
        lambda addr: bool(addr == daemon_address),
    )
    monkeypatch.setattr(
        "tensorcast.daemon_ctl.get_daemon_client", lambda _address: object()
    )
    monkeypatch.setattr(startup.runtime, "stop", lambda **_kwargs: None)
    startup.atomic_write_json(
        startup._auto_state_path(),
        {
            "schema_version": 1,
            "status": "READY",
            "epoch": 9,
            "owner_pid": 99999999,
            "owner_fingerprint": {},
            "config_hash": startup._compute_auto_config_hash(
                daemon_config_path=None,
                global_store_mode="none",
                global_store_address=None,
                global_store_config_path=None,
                cluster_id=None,
                allow_gs_fallback=False,
                session_id=None,
            ),
            "session_id": "sess-stale",
            "address": "127.0.0.1:50052",
            "logs_dir": "/tmp/stale",
            "started_at": time.time() - 30,
        },
    )

    ctx = startup.init(mode="auto", show_daemon_logs=False)
    try:
        assert ctx.is_owner is True
        assert ctx.address == daemon_address
        assert started["session_id"] is None
        auto_state = read_json_default(startup._auto_state_path(), {})
        assert auto_state["status"] == "READY"
        assert auto_state["session_id"] == "sess-recovered"
        assert auto_state["address"] == daemon_address
    finally:
        startup.shutdown()


def test_auto_mode_recovers_from_stale_failed_owner_dead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon_address = "127.0.0.1:63002"

    monkeypatch.setattr(startup.runtime, "status", lambda _session_id=None: None)
    monkeypatch.setattr(
        startup,
        "ping_daemon",
        lambda addr: bool(addr == daemon_address),
    )
    monkeypatch.setattr(
        startup.runtime,
        "start",
        lambda **_kwargs: runtime.RuntimeSession(
            session_id="sess-failed-recover",
            daemon_pid=5353,
            daemon_address=daemon_address,
            daemon_p2p_address="127.0.0.1:63003",
            logs_dir=session_paths("sess-failed-recover").logs,
            started_at=time.time(),
            owner=True,
        ),
    )
    monkeypatch.setattr(
        "tensorcast.daemon_ctl.get_daemon_client", lambda _address: object()
    )
    monkeypatch.setattr(startup.runtime, "stop", lambda **_kwargs: None)
    startup.atomic_write_json(
        startup._auto_state_path(),
        {
            "schema_version": 1,
            "status": "FAILED",
            "epoch": 11,
            "owner_pid": 99999998,
            "owner_fingerprint": {},
            "config_hash": startup._compute_auto_config_hash(
                daemon_config_path=None,
                global_store_mode="none",
                global_store_address=None,
                global_store_config_path=None,
                cluster_id=None,
                allow_gs_fallback=False,
                session_id=None,
            ),
            "session_id": "sess-failed",
            "address": "127.0.0.1:50052",
            "logs_dir": "/tmp/stale-failed",
            "started_at": time.time() - 30,
            "error_code": "AUTO_START_FAILED",
            "error_message": "old error",
        },
    )

    ctx = startup.init(mode="auto", show_daemon_logs=False)
    try:
        assert ctx.is_owner is True
        assert ctx.address == daemon_address
        auto_state = read_json_default(startup._auto_state_path(), {})
        assert auto_state["status"] == "READY"
        assert auto_state["session_id"] == "sess-failed-recover"
    finally:
        startup.shutdown()
