#  Copyright (c) 2025-2026, TensorCast Team.

# Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

from pathlib import Path

import pytest

from tensorcast.cli_utils import config as cli_config
from tensorcast.cli_utils.errors import ServiceError
from tensorcast.cli_utils.paths import session_paths


def test_example_daemon_config_is_used_when_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TENSORCAST_HOME", str(tmp_path))
    monkeypatch.delenv("TENSORCAST_DAEMON_CONFIG", raising=False)
    inst = session_paths()

    repo_root = Path(__file__).resolve().parents[3]
    example_cfg = repo_root / "examples" / "config" / "store_daemon_config.yaml"
    assert example_cfg.exists()

    cfg_path = cli_config.materialize_daemon_config(inst, None, restrict_to_localhost=True)
    assert cfg_path.resolve() == example_cfg.resolve()


def test_missing_daemon_config_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TENSORCAST_HOME", str(tmp_path))
    monkeypatch.delenv("TENSORCAST_DAEMON_CONFIG", raising=False)
    monkeypatch.setattr(cli_config, "_discover_repo_example_config", lambda _: None)
    monkeypatch.setattr(cli_config, "_discover_packaged_example_config", lambda _: None)
    inst = session_paths()

    with pytest.raises(ServiceError, match="No Store Daemon config found"):
        cli_config.materialize_daemon_config(
            inst, None, restrict_to_localhost=True
        )
