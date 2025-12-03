#  Copyright (c) 2025, TensorCast Team.

# Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

from pathlib import Path

import pytest

from tensorcast import startup
from tensorcast.daemon_runtime_config import load_daemon_config


def test_embedded_daemon_config_is_materialized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(startup, "_EMBEDDED_CONFIG_PATH", None)
    monkeypatch.setattr(startup, "_select_cache_root", lambda: tmp_path)

    cfg_path = startup._ensure_embedded_daemon_config_path()
    cfg = load_daemon_config(cfg_path)

    assert cfg.server.listen.host == "127.0.0.1"
    assert int(cfg.server.listen.port) == 0
    assert cfg.high_availability.enabled is False
    assert str(tmp_path) in cfg.server.storage_path
