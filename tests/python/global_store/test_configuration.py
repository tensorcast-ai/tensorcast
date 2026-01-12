#  Copyright (c) 2025-2026, TensorCast Team.

"""Tests for Global Store configuration management."""

import pytest
from pydantic import ValidationError

from tensorcast.global_store.config import GlobalStoreConfig
import yaml


class TestConfiguration:
    """Test configuration management."""

    def test_config_defaults(self):
        """Test default configuration."""
        config = GlobalStoreConfig()
        assert config.port == 50051
        assert config.max_workers == 10
        assert config.heartbeat_timeout_ms == 30000
        assert config.db_file is None

    def test_config_from_file(self, tmp_path):
        """Test configuration from YAML file."""
        cfg = {
            "database": {"db_file": "/tmp/test.db"},
            "server": {"listen": {"host": "127.0.0.1", "port": 50052}, "max_workers": 20},
        }
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        config = GlobalStoreConfig.from_file(str(p))
        assert config.port == 50052
        assert config.max_workers == 20
        assert str(config.db_file) == "/tmp/test.db"

    def test_config_partial_file(self, tmp_path):
        """Test configuration with partial fields in YAML."""
        cfg = {"server": {"listen": {"port": 50053}}}
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        config = GlobalStoreConfig.from_file(str(p))
        assert config.port == 50053
        assert config.max_workers == 10  # Default value
        assert config.heartbeat_timeout_ms == 30000  # Default value

    def test_config_invalid_port_raises(self, tmp_path):
        """Invalid types should raise during strict parsing."""
        cfg = {"server": {"listen": {"port": "invalid"}}}
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        with pytest.raises(Exception):
            _ = GlobalStoreConfig.from_file(str(p))

    def test_config_zero_max_workers(self, tmp_path):
        """Explicit zero should be accepted from file config."""
        cfg = {"server": {"max_workers": 0}}
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        config = GlobalStoreConfig.from_file(str(p))
        assert config.max_workers == 0

    def test_config_negative_timeout(self, tmp_path):
        """Negative duration clamps to 0ms."""
        cfg = {"worker_policy": {"heartbeat_timeout": "-1s"}}
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        config = GlobalStoreConfig.from_file(str(p))
        assert config.heartbeat_timeout_ms == 0

    def test_config_all_fields_from_file(self, tmp_path):
        """Test configuration with key fields from YAML."""
        cfg = {
            "database": {"db_file": "/custom/path/models.db"},
            "server": {"listen": {"port": 50054}, "max_workers": 25},
            "worker_policy": {
                "heartbeat_timeout": "45s",
                "cleanup_interval": "90s",
                "default_heartbeat_interval": "7s",
            },
        }
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        config = GlobalStoreConfig.from_file(str(p))
        assert config.port == 50054
        assert config.max_workers == 25
        assert str(config.db_file) == "/custom/path/models.db"
        assert config.heartbeat_timeout_ms == 45000
        assert config.cleanup_interval_ms == 90000
        assert config.default_heartbeat_interval_ms == 7000
        # Fields not present in file remain defaults
        assert config.optimize_interval_ms == 3_600_000
        assert config.metrics_port == 8000

    def test_config_empty_db_path(self, tmp_path):
        """Empty db_file becomes None."""
        cfg = {"database": {"db_file": ""}}
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        config = GlobalStoreConfig.from_file(str(p))
        assert config.db_file is None  # Empty string should result in None

    def test_config_simple_values(self, tmp_path):
        cfg = {"server": {"listen": {"port": 50055}}, "database": {"db_file": "/path/with/spaces.db"}}
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        config = GlobalStoreConfig.from_file(str(p))
        assert config.port == 50055
        assert str(config.db_file) == "/path/with/spaces.db"

    def test_config_expands_user_db_path(self, tmp_path, monkeypatch):
        cfg = {"database": {"db_file": "~/global_store.db"}}
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        monkeypatch.setenv("HOME", str(tmp_path))
        config = GlobalStoreConfig.from_file(str(p))
        assert str(config.db_file) == str(tmp_path / "global_store.db")

    def test_config_repr(self):
        """Test configuration string representation."""
        config = GlobalStoreConfig(
            listen_port=50051,
            max_workers=10,
            heartbeat_timeout_ms=30000,
        )

        repr_str = repr(config)
        assert "GlobalStoreConfig" in repr_str
        assert "listen_port=50051" in repr_str
        assert "max_workers=10" in repr_str

    def test_config_equality(self):
        """Test configuration equality comparison."""
        config1 = GlobalStoreConfig(listen_port=50051, max_workers=10)
        config2 = GlobalStoreConfig(listen_port=50051, max_workers=10)
        config3 = GlobalStoreConfig(listen_port=50052, max_workers=10)

        assert config1 == config2
        assert config1 != config3

    def test_config_immutability(self):
        """Test that configuration is immutable after creation."""
        config = GlobalStoreConfig(listen_port=50051)

        # The pydantic model is frozen, so attempting to modify attributes should raise ValidationError
        with pytest.raises((ValidationError)):
            config.port = 50052

    def test_config_transport_wait_retry_interval(self, tmp_path):
        """transport_wait_retry_interval_ms remains default (not file-configured)."""
        cfg = {}
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        config = GlobalStoreConfig.from_file(str(p))
        assert config.transport_wait_retry_interval_ms == 200

    def test_config_large_port_values(self, tmp_path):
        """Port values beyond typical range are accepted by parser (validation elsewhere)."""
        cfg = {"server": {"listen": {"port": 99999}}}
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        config = GlobalStoreConfig.from_file(str(p))
        assert config.port == 99999

    def test_config_unicode_db_path(self, tmp_path):
        """Unicode paths supported in file config."""
        unicode_path = "/data/测试/моделі/データ.db"
        cfg = {"database": {"db_file": unicode_path}}
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        config = GlobalStoreConfig.from_file(str(p))
        assert str(config.db_file) == unicode_path
