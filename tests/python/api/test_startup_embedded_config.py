#  Copyright (c) 2025, TensorCast Team.

# Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

from pathlib import Path

import pytest

from tensorcast.cli_utils import config as cli_config
from tensorcast.cli_utils.paths import session_paths
from tensorcast.daemon_runtime_config import load_daemon_config


def test_embedded_daemon_config_is_materialized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TENSORCAST_HOME", str(tmp_path))
    inst = session_paths()

    cfg_path = cli_config.materialize_daemon_config(inst, None, restrict_to_localhost=True)
    cfg = load_daemon_config(cfg_path)

    assert cfg.server.listen.host == "127.0.0.1"
    assert int(cfg.server.listen.port) == 0
    assert cfg.high_availability.enabled is False
    assert str(inst.root) in cfg.server.storage_path
