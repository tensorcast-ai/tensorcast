#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import time
from pathlib import Path

import pytest
from pydantic import ValidationError

import tensorcast.runtime as runtime_module
from tensorcast import startup


class _RecordingLogger:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str, *args: object) -> None:
        self.warnings.append(message % args if args else message)

    def info(self, _message: str, *_args: object) -> None:
        return


@pytest.fixture(autouse=True)
def _isolate_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TENSORCAST_HOME", str(tmp_path))
    monkeypatch.setattr(startup, "_current_ctx", None)
    monkeypatch.setattr(startup, "discover_daemon_config", lambda: None)


def test_port_config_validation_rejects_invalid_port() -> None:
    with pytest.raises(ValidationError, match="port values must be integers"):
        startup.PortConfig(daemon_listen_port=70000)


def test_connect_mode_warns_port_config_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _RecordingLogger()
    called: dict[str, object] = {}

    def _fake_connect_context(**kwargs: object) -> object:
        called.update(kwargs)
        return object()

    monkeypatch.setattr(startup, "init_logger", lambda _name: logger)
    monkeypatch.setattr(startup, "_connect_context", _fake_connect_context)

    startup.init(
        mode="connect",
        address="127.0.0.1:50052",
        port_config=startup.PortConfig(daemon_listen_port=50060),
    )

    assert called["target_address"] == "127.0.0.1:50052"
    assert len(logger.warnings) == 1
    assert "port_config" in logger.warnings[0]
    assert "are ignored" in logger.warnings[0]


def test_create_mode_forwards_port_config_to_runtime_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: dict[str, object] = {}
    daemon_address = "127.0.0.1:61002"

    def _fake_start(**kwargs: object) -> runtime_module.RuntimeSession:
        started.update(kwargs)
        return runtime_module.RuntimeSession(
            session_id="sess-port-config",
            daemon_pid=4242,
            daemon_address=daemon_address,
            daemon_p2p_address="127.0.0.1:62002",
            logs_dir=None,
            started_at=time.time(),
            owner=True,
        )

    monkeypatch.setattr(startup.runtime, "start", _fake_start)
    monkeypatch.setattr(startup.runtime, "stop", lambda **_kwargs: None)
    monkeypatch.setattr(
        "tensorcast.daemon_ctl.get_daemon_client", lambda _address: object()
    )

    port_config = startup.PortConfig(
        daemon_listen_port=50060,
        daemon_p2p_port=50061,
        global_store_listen_port=50062,
        global_store_metrics_port=18008,
    )

    ctx = startup.init(
        mode="create",
        global_store_mode="start",
        show_daemon_logs=False,
        port_config=port_config,
    )
    try:
        assert ctx.address == daemon_address
        assert started["listen_port"] == 50060
        assert started["p2p_listen_port"] == 50061
        assert started["global_store_listen_port"] == 50062
        assert started["global_store_metrics_port"] == 18008
    finally:
        startup.shutdown()


def test_auto_config_hash_includes_port_config() -> None:
    hash_a = startup._compute_auto_config_hash(
        daemon_config_path=None,
        global_store_mode="start",
        global_store_address=None,
        global_store_config_path=None,
        cluster_id=None,
        allow_gs_fallback=False,
        session_id=None,
        port_config=startup.PortConfig(daemon_listen_port=50052),
    )
    hash_b = startup._compute_auto_config_hash(
        daemon_config_path=None,
        global_store_mode="start",
        global_store_address=None,
        global_store_config_path=None,
        cluster_id=None,
        allow_gs_fallback=False,
        session_id=None,
        port_config=startup.PortConfig(daemon_listen_port=50053),
    )

    assert hash_a != hash_b
