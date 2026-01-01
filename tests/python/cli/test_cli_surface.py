#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import json
from types import SimpleNamespace

from click.testing import CliRunner

from tensorcast import cli as cli_mod
from tensorcast import runtime


def test_daemon_status_json(monkeypatch):
    session = runtime.RuntimeSession(
        session_id="sess-1",
        daemon_pid=123,
        daemon_address="127.0.0.1:6000",
        daemon_p2p_address="127.0.0.1:7000",
        logs_dir=None,
        started_at=1.0,
        owner=True,
        global_store_mode="connect",
        global_store_address="127.0.0.1:50051",
        global_store_session="gs-1",
        global_store_owner=False,
        cluster_token="tok",
    )
    monkeypatch.setattr(cli_mod.runtime, "status", lambda _sid=None: session)

    runner = CliRunner()
    res = runner.invoke(cli_mod.cli, ["daemon", "status", "--json"])
    assert res.exit_code == 0
    payload = json.loads(res.output)
    assert payload["session_id"] == "sess-1"
    assert payload["global_store"]["address"] == "127.0.0.1:50051"


def test_daemon_start_passes_options(monkeypatch):
    captured = {}

    def _fake_start(**kwargs):
        captured.update(kwargs)
        return runtime.RuntimeSession(
            session_id="sess-start",
            daemon_pid=321,
            daemon_address="1.2.3.4:5",
            daemon_p2p_address=None,
            logs_dir=None,
            started_at=2.0,
            owner=True,
            global_store_mode=kwargs.get("global_store_mode"),
            global_store_address=kwargs.get("global_store_address"),
            global_store_session=None,
            global_store_owner=False,
            cluster_token=None,
        )

    monkeypatch.setattr(cli_mod.runtime, "start", _fake_start)

    runner = CliRunner()
    res = runner.invoke(
        cli_mod.cli,
        [
            "daemon",
            "start",
            "--global-store-endpoints",
            "10.0.0.1:50051",
            "--global-store-endpoints",
            "10.0.0.2:50051",
            "--session",
            "sess-x",
            "--stable-bytes",
            "4GB",
            "--mem-pool-size-bytes",
            "8GB",
            "--enable-rdma",
            "--log-level",
            "warn",
            "--set",
            "engine.tx_slice_bytes=64MB",
            "--json",
        ],
    )
    assert res.exit_code == 0
    assert captured["global_store_mode"] == "connect"
    assert captured["global_store_address"] == "10.0.0.1:50051"
    assert captured["ha_endpoints"] == ["10.0.0.1:50051", "10.0.0.2:50051"]
    assert captured["session_id"] == "sess-x"
    assert captured["fate_share"] is False
    assert set(captured["config_overrides"]) == {
        "engine.memory_tiers.stable_bytes=4GB",
        "engine.mem_pool_size_bytes=8GB",
        "communicator.enable_rdma=true",
        "observability.logging.level=warn",
        "engine.tx_slice_bytes=64MB",
    }
    assert "wait" not in captured
    assert "timeout" not in captured


def test_daemon_start_blocking_passes_flags(monkeypatch):
    captured = {}

    def _fake_start(**kwargs):
        captured.update(kwargs)
        return runtime.RuntimeSession(
            session_id="sess-blocking",
            daemon_pid=999,
            daemon_address="127.0.0.1:50052",
            daemon_p2p_address=None,
            logs_dir=None,
            started_at=2.0,
            owner=True,
            global_store_mode=kwargs.get("global_store_mode"),
            global_store_address=kwargs.get("global_store_address"),
            global_store_session=None,
            global_store_owner=False,
            cluster_token=None,
        )

    monkeypatch.setattr(cli_mod.runtime, "start", _fake_start)
    monkeypatch.setattr(cli_mod.runtime, "status", lambda _sid=None: None)

    runner = CliRunner()
    res = runner.invoke(cli_mod.cli, ["daemon", "start", "--blocking"])
    assert res.exit_code == 0
    assert captured["blocking"] is True
    assert captured["fate_share"] is True


def test_daemon_start_reports_existing(monkeypatch):
    session = runtime.RuntimeSession(
        session_id="sess-1",
        daemon_pid=123,
        daemon_address="127.0.0.1:6000",
        daemon_p2p_address="127.0.0.1:7000",
        logs_dir=None,
        started_at=1.0,
        owner=True,
        global_store_mode="connect",
        global_store_address="127.0.0.1:50051",
        global_store_session="gs-1",
        global_store_owner=False,
        cluster_token="tok",
    )
    monkeypatch.setattr(cli_mod.runtime, "status", lambda _sid=None: session)

    def _fail_start(**_kwargs):
        raise AssertionError("runtime.start should not be called")

    monkeypatch.setattr(cli_mod.runtime, "start", _fail_start)

    runner = CliRunner()
    res = runner.invoke(cli_mod.cli, ["daemon", "start"])
    assert res.exit_code == 1
    assert "Daemon session: sess-1" in res.output


def test_global_status_json(monkeypatch):
    def _fake_payload(_sid):
        return (
            "gs-1",
            {
                "address": "127.0.0.1:50051",
                "state": {"global_store": {"address": "127.0.0.1:50051"}},
            },
        )

    class _Health(SimpleNamespace):
        pass

    monkeypatch.setattr(cli_mod, "_global_status_payload", _fake_payload)
    monkeypatch.setattr(
        cli_mod, "ping_global_store", lambda *_args, **_kwargs: _Health(address="127.0.0.1:50051", listen_host="127.0.0.1", listen_port=50051, metrics_port=8000, cluster_token="tok", version="v1", db_file="/tmp/db.duckdb")
    )

    runner = CliRunner()
    res = runner.invoke(cli_mod.cli, ["global", "status", "--json"])
    assert res.exit_code == 0
    payload = json.loads(res.output)
    assert payload["session_id"] == "gs-1"
    assert payload["health"]["listen_port"] == 50051


def test_global_start_echo(monkeypatch):
    inst = SimpleNamespace(
        id="gs-2",
        pid=222,
        address="127.0.0.1:50052",
        metrics_port=8001,
        logs_dir="/logs",
    )
    monkeypatch.setattr(cli_mod.global_store_manager, "start_global_store", lambda **_kwargs: inst)

    runner = CliRunner()
    res = runner.invoke(
        cli_mod.cli, ["global", "start", "--listen-port", "0", "--json"]
    )
    assert res.exit_code == 0
    payload = json.loads(res.output)
    assert payload["session_id"] == "gs-2"
    assert payload["address"] == "127.0.0.1:50052"


def test_global_start_blocking_passes_flags(monkeypatch):
    captured = {}
    inst = SimpleNamespace(
        id="gs-3",
        pid=333,
        address="127.0.0.1:50053",
        metrics_port=8002,
        logs_dir="/logs",
    )

    def _fake_start_global_store(**kwargs):
        captured.update(kwargs)
        return inst

    monkeypatch.setattr(
        cli_mod.global_store_manager, "start_global_store", _fake_start_global_store
    )

    runner = CliRunner()
    res = runner.invoke(cli_mod.cli, ["global", "start", "--blocking"])
    assert res.exit_code == 0
    assert captured["blocking"] is True
    assert captured["fate_share"] is True
