#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tensorcast import runtime
from tensorcast.cli_utils.errors import ServiceError
from tensorcast.cli_utils.process import (
    instance_fingerprint,
    update_runtime_daemon,
    write_session_state,
    write_runtime_state,
)
from tensorcast.cli_utils.paths import runtime_state_path, session_paths


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch, tmp_path):
    monkeypatch.setenv("TENSORCAST_HOME", str(tmp_path))


def _stub_daemon_start(monkeypatch, gs_state: dict[str, Any]):
    started: dict[str, Any] = {}

    def _fake_start_service(**kwargs):
        inst = session_paths(kwargs.get("session_id"))
        started["ha_endpoints"] = kwargs.get("ha_endpoints")
        started["global_store"] = kwargs.get("global_store")
        update_runtime_daemon(
            path=runtime_state_path(),
            session_id=inst.id,
            pid=4242,
            address="127.0.0.1:61000",
            p2p_address="127.0.0.1:61001",
            owner=True,
            fingerprint=instance_fingerprint(4242),
        )
        write_session_state(
            inst.session_state_json,
            {
                "session_id": inst.id,
                "daemon": {
                    "pid": 4242,
                    "address": "127.0.0.1:61000",
                    "p2p_address": "127.0.0.1:61001",
                },
                "global_store": gs_state,
                "logs_dir": str(inst.logs),
            },
        )
        return SimpleNamespace(id=inst.id, logs=inst.logs)

    monkeypatch.setattr(runtime.service_manager, "start_service", _fake_start_service)
    monkeypatch.setattr(runtime.service_manager, "stop_service", lambda **_kwargs: None)
    monkeypatch.setattr(runtime, "is_process_alive", lambda pid: True)
    monkeypatch.setattr(runtime, "ping_daemon", lambda address: True)
    return started


def test_runtime_start_rejects_existing_daemon(monkeypatch):
    inst = session_paths("sess-existing")
    update_runtime_daemon(
        path=runtime_state_path(),
        session_id=inst.id,
        pid=4242,
        address="127.0.0.1:61000",
        p2p_address="127.0.0.1:61001",
        owner=True,
        fingerprint=instance_fingerprint(4242),
    )
    monkeypatch.setattr(runtime, "is_process_alive", lambda _pid: True)
    monkeypatch.setattr(runtime, "ping_daemon", lambda _address: True)

    def _fail_start(**_kwargs):
        raise AssertionError("start_service should not be called")

    monkeypatch.setattr(runtime.service_manager, "start_service", _fail_start)

    with pytest.raises(ServiceError, match="already running"):
        runtime.start(session_id="sess-new")


def test_reconcile_keeps_alive_daemon_when_ping_fails(monkeypatch):
    inst = session_paths("sess-existing")
    update_runtime_daemon(
        path=runtime_state_path(),
        session_id=inst.id,
        pid=4242,
        address="127.0.0.1:61000",
        p2p_address="127.0.0.1:61001",
        owner=True,
        fingerprint=instance_fingerprint(4242),
    )
    monkeypatch.setattr(runtime, "is_process_alive", lambda _pid: True)
    monkeypatch.setattr(runtime, "ping_daemon", lambda _address: False)

    session = runtime.reconcile()
    assert session is not None
    assert session.session_id == inst.id


def test_runtime_stop_uses_runtime_state(monkeypatch):
    inst = session_paths("sess-running")
    update_runtime_daemon(
        path=runtime_state_path(),
        session_id=inst.id,
        pid=4242,
        address="127.0.0.1:61000",
        p2p_address="127.0.0.1:61001",
        owner=True,
        fingerprint=instance_fingerprint(4242),
    )
    monkeypatch.setattr(runtime, "is_process_alive", lambda _pid: True)
    monkeypatch.setattr(runtime, "ping_daemon", lambda _address: False)

    stopped: dict[str, Any] = {}

    def _fake_stop_service(**kwargs):
        stopped.update(kwargs)

    monkeypatch.setattr(runtime.service_manager, "stop_service", _fake_stop_service)

    runtime.stop()

    assert stopped["session_id"] == inst.id


def test_runtime_start_injects_global_store(monkeypatch):
    captured: dict[str, Any] = {}

    def _fake_start_global_store(**kwargs):
        captured["cluster_token"] = kwargs.get("cluster_token")
        return SimpleNamespace(
            id="gs-123",
            pid=1111,
            address="127.0.0.1:50051",
            listen_host="127.0.0.1",
            listen_port=50051,
            metrics_port=8000,
            logs_dir=None,
            cluster_token="tok-1",
            db_file=None,
            owner=True,
        )

    monkeypatch.setattr(runtime, "ping_global_store", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runtime.global_store_manager, "start_global_store", _fake_start_global_store
    )

    gs_state = {
        "mode": "start",
        "address": "127.0.0.1:50051",
        "session": "gs-123",
        "owner": True,
    }
    started = _stub_daemon_start(monkeypatch, gs_state)

    session = runtime.start(global_store_mode="start")

    assert session.global_store_address == "127.0.0.1:50051"
    assert started["ha_endpoints"] == ["127.0.0.1:50051"]
    assert captured["cluster_token"] is None
    assert session.global_store_session == "gs-123"
    assert session.global_store_mode == "start"


def test_runtime_stop_cascades_global_store(monkeypatch):
    stop_calls: dict[str, Any] = {}

    def _fake_start_global_store(**_kwargs):
        return SimpleNamespace(
            id="gs-own",
            pid=1111,
            address="127.0.0.1:50051",
            listen_host="127.0.0.1",
            listen_port=50051,
            metrics_port=8000,
            logs_dir=None,
            cluster_token="tok-1",
            db_file=None,
            owner=True,
        )

    def _fake_stop_global_store(**kwargs):
        stop_calls["session_id"] = kwargs.get("session_id")
        stop_calls["force"] = kwargs.get("force")

    monkeypatch.setattr(
        runtime.global_store_manager, "start_global_store", _fake_start_global_store
    )
    monkeypatch.setattr(
        runtime.global_store_manager, "stop_global_store", _fake_stop_global_store
    )
    gs_state = {
        "mode": "start",
        "address": "127.0.0.1:50051",
        "session": "gs-own",
        "owner": True,
    }
    _stub_daemon_start(monkeypatch, gs_state)

    session = runtime.start(global_store_mode="start")
    runtime.stop(session_id=session.session_id)

    assert stop_calls["session_id"] == "gs-own"
    assert stop_calls["force"] is False


def test_runtime_none_mode_skips_global_store(monkeypatch):
    called = {"gs": False}

    def _fake_start_global_store(**_kwargs):
        called["gs"] = True
        raise AssertionError("Global Store should not start in none mode")

    monkeypatch.setattr(
        runtime.global_store_manager, "start_global_store", _fake_start_global_store
    )
    monkeypatch.setattr(runtime, "ping_global_store", lambda *_args, **_kwargs: None)
    gs_state: dict[str, Any] = {"mode": "none", "required": False}
    _stub_daemon_start(monkeypatch, gs_state)

    session = runtime.start(global_store_mode="none")

    assert session.global_store_mode == "none"
    assert called["gs"] is False


def test_runtime_rejects_mismatched_cluster_token(monkeypatch):
    write_runtime_state(
        runtime_state_path(),
        {
            "global_store": {
                "address": "127.0.0.1:50051",
                "cluster_token": "cluster-abc",
            }
        },
    )

    def _fake_ping_global_store(*_args, **_kwargs):
        return SimpleNamespace(
            listen_host="127.0.0.1",
            listen_port=50052,
            metrics_port=None,
            cluster_token="cluster-def",
            version="v1",
            db_file=None,
        )

    monkeypatch.setattr(runtime, "ping_global_store", _fake_ping_global_store)

    with pytest.raises(ServiceError, match="mismatched cluster token"):
        runtime._resolve_global_store(
            mode="connect",
            address=None,
            allow_gs_fallback=False,
            cluster_id=None,
        )
