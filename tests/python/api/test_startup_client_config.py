#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorcast import startup
from tensorcast.client_config_loader import discover_client_config


@pytest.fixture(autouse=True)
def _isolate_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TENSORCAST_HOME", str(tmp_path))


def test_discover_client_config_prefers_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    env_cfg = cfg_dir / "client.yaml"
    env_cfg.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("TENSORCAST_CLIENT_CONFIG", str(env_cfg))

    discovered = discover_client_config()

    assert discovered == env_cfg


def test_discover_client_config_falls_back_to_home(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config" / "client.yml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text("{}", encoding="utf-8")

    discovered = discover_client_config()

    assert discovered == cfg_path


def test_init_from_client_config_prefers_runtime_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "client.json"
    cfg_path.write_text("{}", encoding="utf-8")

    captured: dict[str, str] = {}
    monkeypatch.setattr(startup, "set_client_config", lambda _cfg: None)
    monkeypatch.setattr(startup, "daemon_target_default", lambda: None)
    monkeypatch.setattr(
        startup.runtime, "status", lambda _sid=None: SimpleNamespace(daemon_address="127.0.0.1:50052")
    )

    def _set_addr(addr: str) -> None:
        captured["addr"] = addr

    monkeypatch.setattr(startup, "set_daemon_address", _set_addr)

    startup.init_from_client_config(cfg_path)

    assert captured["addr"] == "127.0.0.1:50052"


def test_init_from_client_config_prefers_config_target(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "client.json"
    cfg_path.write_text('{"daemon": {"target": {"host": "10.0.0.1", "port": 6000}}}', encoding="utf-8")

    captured: dict[str, str] = {}
    monkeypatch.setattr(startup, "set_client_config", lambda _cfg: None)
    monkeypatch.setattr(startup, "daemon_target_default", lambda: "10.0.0.1:6000")
    monkeypatch.setattr(
        startup.runtime, "status", lambda _sid=None: SimpleNamespace(daemon_address="127.0.0.1:50052")
    )

    def _set_addr(addr: str) -> None:
        captured["addr"] = addr

    monkeypatch.setattr(startup, "set_daemon_address", _set_addr)

    startup.init_from_client_config(cfg_path)

    assert captured["addr"] == "10.0.0.1:6000"
