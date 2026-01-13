#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import json
import socket
from pathlib import Path

from tensorcast.cli_utils import config as cfg_utils
from tensorcast.cli_utils.paths import global_session_paths, runtime_root


def test_cluster_token_persisted_with_hmac(monkeypatch, tmp_path):
    monkeypatch.setenv("TENSORCAST_HOME", str(tmp_path))
    token_path = runtime_root() / "cluster_token"

    token = cfg_utils.load_or_create_cluster_token()
    payload = json.loads(token_path.read_text(encoding="utf-8"))
    assert payload["token"] == token
    assert payload["hmac"]
    assert (token_path.stat().st_mode & 0o777) == 0o600

    token_again = cfg_utils.load_or_create_cluster_token()
    assert token_again == token

    token_path.unlink()
    preset = cfg_utils.load_or_create_cluster_token("preset-token")
    payload = json.loads(token_path.read_text(encoding="utf-8"))
    assert preset == "preset-token"
    assert payload["token"] == "preset-token"


def test_build_embedded_global_store_config(monkeypatch, tmp_path):
    monkeypatch.setenv("TENSORCAST_HOME", str(tmp_path))
    session = global_session_paths("gs-test")
    cfg = cfg_utils.build_embedded_global_store_config(
        session, cluster_token="cluster-1", listen_port=0, metrics_port=0
    )

    assert cfg.server.listen.host == "0.0.0.0"
    assert cfg.server.listen.port == 0
    assert cfg.server.metrics_port == 0
    assert cfg.database.db_file == str(session.root / "global_store.duckdb")
    assert cfg.observability.logging.file == str(session.logs / "global_store.out")
    assert cfg.meta.cluster_token == "cluster-1"


def test_select_free_port_falls_back_when_taken():
    host = "127.0.0.1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        in_use = s.getsockname()[1]
        chosen = cfg_utils.select_free_port(in_use, host=host, probe_span=4)
        assert chosen != in_use


def test_discover_global_store_config_prefers_env(monkeypatch, tmp_path):
    monkeypatch.setenv("TENSORCAST_HOME", str(tmp_path))
    env_cfg = tmp_path / "env_cfg.yaml"
    env_cfg.write_text("server: {}", encoding="utf-8")
    monkeypatch.setenv("TENSORCAST_GLOBAL_STORE_CONFIG", str(env_cfg))

    discovered = cfg_utils.discover_global_store_config()
    assert discovered == env_cfg


def test_discover_global_store_config_falls_back_to_examples(monkeypatch, tmp_path):
    monkeypatch.setenv("TENSORCAST_HOME", str(tmp_path))
    monkeypatch.delenv("TENSORCAST_GLOBAL_STORE_CONFIG", raising=False)

    example_cfg = None
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "examples" / "config" / "global_store_config.yaml"
        if candidate.exists():
            example_cfg = candidate
            break

    assert example_cfg is not None
    discovered = cfg_utils.discover_global_store_config()
    assert discovered == example_cfg
