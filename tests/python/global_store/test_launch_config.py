#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

from pathlib import Path

import pytest

from tensorcast.global_store import launch_config


def test_resolve_global_store_config_path_explicit(tmp_path: Path) -> None:
    cfg = tmp_path / "global_store.yaml"
    cfg.write_text("server: {}", encoding="utf-8")
    resolved = launch_config.resolve_global_store_config_path(cfg)
    assert resolved == cfg


def test_resolve_global_store_config_path_discovered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    discovered = tmp_path / "global_store.yaml"
    discovered.write_text("server: {}", encoding="utf-8")
    monkeypatch.setattr(
        launch_config,
        "discover_global_store_config",
        lambda: discovered,
    )
    resolved = launch_config.resolve_global_store_config_path(None)
    assert resolved == discovered


def test_resolve_global_store_config_path_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        launch_config,
        "discover_global_store_config",
        lambda: None,
    )
    with pytest.raises(FileNotFoundError, match="No Global Store config found"):
        launch_config.resolve_global_store_config_path(None)
