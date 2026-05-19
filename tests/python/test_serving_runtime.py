#  Copyright (c) 2026, TensorCast Team.

from pathlib import Path

import pytest

from tensorcast.serving import (
    DEFAULT_RUNTIME_PROFILE,
    RuntimeSettings,
    resolve_runtime_config_profile,
)


def test_runtime_settings_resolves_default_packaged_profile() -> None:
    kwargs = RuntimeSettings().to_init_kwargs()

    assert kwargs["mode"] == "auto"
    assert kwargs["global_store_mode"] == "start"
    daemon_path = Path(kwargs["daemon_config_path"])
    global_store_path = Path(kwargs["global_store_config_path"])
    assert daemon_path.name == "store_daemon_config.yaml"
    assert global_store_path.name == "global_store_config.yaml"
    assert "config" in daemon_path.parts
    assert "profiles" in daemon_path.parts
    assert DEFAULT_RUNTIME_PROFILE in daemon_path.parts
    assert daemon_path.is_file()
    assert global_store_path.is_file()


def test_runtime_settings_resolves_named_profile() -> None:
    profile = resolve_runtime_config_profile("local_dev")

    assert profile.name == "local_dev"
    assert Path(profile.daemon_config_path).is_file()
    assert Path(profile.global_store_config_path).is_file()

    kwargs = RuntimeSettings(profile="local_dev").to_init_kwargs()

    assert Path(kwargs["daemon_config_path"]).parent.name == "local_dev"
    assert Path(kwargs["global_store_config_path"]).parent.name == "local_dev"


def test_runtime_settings_rejects_missing_profile_for_create_modes() -> None:
    with pytest.raises(ValueError, match="runtime.mode requires"):
        RuntimeSettings(profile=None).to_init_kwargs()

    with pytest.raises(ValueError, match="Unknown TensorCast runtime"):
        RuntimeSettings(profile="missing").to_init_kwargs()


def test_runtime_settings_connect_mode_does_not_require_profile() -> None:
    kwargs = RuntimeSettings(profile=None, mode="connect").to_init_kwargs()

    assert kwargs["mode"] == "connect"
    assert "daemon_config_path" not in kwargs
    assert "global_store_mode" not in kwargs


def test_runtime_settings_explicit_paths_override_profile(tmp_path: Path) -> None:
    daemon_config = tmp_path / "daemon.yaml"
    global_store_config = tmp_path / "global-store.yaml"
    daemon_config.write_text("meta:\n  description: explicit\n", encoding="utf-8")
    global_store_config.write_text("meta:\n  description: explicit\n",
                                   encoding="utf-8")

    kwargs = RuntimeSettings(
        profile="local_dev",
        daemon={"config_path": str(daemon_config)},
        global_store={"mode": "start", "config_path": str(global_store_config)},
    ).to_init_kwargs()

    assert kwargs["daemon_config_path"] == str(daemon_config)
    assert kwargs["global_store_config_path"] == str(global_store_config)
