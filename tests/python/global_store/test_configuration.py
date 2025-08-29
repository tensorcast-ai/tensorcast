#  Copyright (c) 2025, TensorCast Team.

"""Tests for Global Store configuration management."""

import pytest
from pydantic import ValidationError

from scstore.global_store.config import GlobalStoreConfig


class TestConfiguration:
    """Test configuration management."""

    def test_config_defaults(self):
        """Test default configuration."""
        config = GlobalStoreConfig()
        assert config.port == 50051
        assert config.max_workers == 10
        assert config.heartbeat_timeout_ms == 30000
        assert config.db_file is None

    def test_config_from_env(self, monkeypatch):
        """Test configuration from environment variables."""
        monkeypatch.setenv("GLOBAL_STORE_PORT", "50052")
        monkeypatch.setenv("GLOBAL_STORE_MAX_WORKERS", "20")
        monkeypatch.setenv("GLOBAL_STORE_DB_PATH", "/tmp/test.db")

        config = GlobalStoreConfig.from_env()
        assert config.port == 50052
        assert config.max_workers == 20
        assert str(config.db_file) == "/tmp/test.db"

    def test_config_partial_env(self, monkeypatch):
        """Test configuration with partial environment variables."""
        monkeypatch.setenv("GLOBAL_STORE_PORT", "50053")
        # Leave other values as defaults

        config = GlobalStoreConfig.from_env()
        assert config.port == 50053
        assert config.max_workers == 10  # Default value
        assert config.heartbeat_timeout_ms == 30000  # Default value

    def test_config_invalid_port(self, monkeypatch):
        """Test configuration with invalid port."""
        monkeypatch.setenv("GLOBAL_STORE_PORT", "invalid")

        # Should fall back to default on invalid value
        config = GlobalStoreConfig.from_env()
        assert config.port == 50051  # Default value

    def test_config_zero_max_workers(self, monkeypatch):
        """Test configuration with zero max workers."""
        monkeypatch.setenv("GLOBAL_STORE_MAX_WORKERS", "0")

        config = GlobalStoreConfig.from_env()
        assert config.max_workers == 0

    def test_config_negative_timeout(self, monkeypatch):
        """Test configuration with negative timeout."""
        monkeypatch.setenv("GLOBAL_STORE_HEARTBEAT_TIMEOUT_MS", "-1000")

        # Should handle negative values gracefully
        config = GlobalStoreConfig.from_env()
        # Implementation specific: might clamp to minimum or use default
        assert config.heartbeat_timeout_ms >= 0

    def test_config_all_environment_variables(self, monkeypatch):
        """Test configuration with all supported environment variables."""
        env_vars = {
            "GLOBAL_STORE_PORT": "50054",
            "GLOBAL_STORE_MAX_WORKERS": "25",
            "GLOBAL_STORE_DB_PATH": "/custom/path/models.db",
            "GLOBAL_STORE_HEARTBEAT_TIMEOUT_MS": "45000",
            "GLOBAL_STORE_CLEANUP_INTERVAL_MS": "90000",
            "GLOBAL_STORE_HEARTBEAT_INTERVAL_MS": "7000",
            "GLOBAL_STORE_OPTIMIZE_INTERVAL_MS": "7200000",
            "GLOBAL_STORE_METRICS_PORT": "8001",
        }

        for key, value in env_vars.items():
            monkeypatch.setenv(key, value)

        config = GlobalStoreConfig.from_env()
        assert config.port == 50054
        assert config.max_workers == 25
        assert str(config.db_file) == "/custom/path/models.db"
        assert config.heartbeat_timeout_ms == 45000
        assert config.cleanup_interval_ms == 90000
        assert config.default_heartbeat_interval_ms == 7000
        assert config.optimize_interval_ms == 7200000
        assert config.metrics_port == 8001

    def test_config_empty_db_path(self, monkeypatch):
        """Test configuration with empty database path."""
        monkeypatch.setenv("GLOBAL_STORE_DB_PATH", "")

        config = GlobalStoreConfig.from_env()
        assert config.db_file is None  # Empty string should result in None

    def test_config_whitespace_values(self, monkeypatch):
        """Test configuration with whitespace values."""
        monkeypatch.setenv("GLOBAL_STORE_PORT", "  50055  ")
        monkeypatch.setenv("GLOBAL_STORE_DB_PATH", "  /path/with/spaces.db  ")

        config = GlobalStoreConfig.from_env()
        assert config.port == 50055  # Should strip whitespace
        assert str(config.db_file) == "/path/with/spaces.db"  # Should strip whitespace

    def test_config_repr(self):
        """Test configuration string representation."""
        config = GlobalStoreConfig(
            port=50051,
            max_workers=10,
            heartbeat_timeout_ms=30000,
        )

        repr_str = repr(config)
        assert "GlobalStoreConfig" in repr_str
        assert "port=50051" in repr_str
        assert "max_workers=10" in repr_str

    def test_config_equality(self):
        """Test configuration equality comparison."""
        config1 = GlobalStoreConfig(port=50051, max_workers=10)
        config2 = GlobalStoreConfig(port=50051, max_workers=10)
        config3 = GlobalStoreConfig(port=50052, max_workers=10)

        assert config1 == config2
        assert config1 != config3

    def test_config_immutability(self):
        """Test that configuration is immutable after creation."""
        config = GlobalStoreConfig(port=50051)

        # The pydantic model is frozen, so attempting to modify attributes should raise ValidationError
        with pytest.raises((ValidationError)):
            config.port = 50052

    def test_config_transport_wait_retry_interval(self, monkeypatch):
        """Test that transport_wait_retry_interval_ms uses default."""
        # This value is not configurable via environment variables
        monkeypatch.setenv("GLOBAL_STORE_TRANSPORT_WAIT_RETRY_INTERVAL_MS", "500")

        config = GlobalStoreConfig.from_env()
        # Should always use the default value of 200
        assert config.transport_wait_retry_interval_ms == 200

    def test_config_large_port_values(self, monkeypatch):
        """Test configuration with port values at limits."""
        # Test port value above valid range
        monkeypatch.setenv("GLOBAL_STORE_PORT", "99999")

        config = GlobalStoreConfig.from_env()
        # Should accept the value as-is (validation happens elsewhere)
        assert config.port == 99999

    def test_config_unicode_db_path(self, monkeypatch):
        """Test configuration with unicode characters in path."""
        unicode_path = "/data/测试/моделі/データ.db"
        monkeypatch.setenv("GLOBAL_STORE_DB_PATH", unicode_path)

        config = GlobalStoreConfig.from_env()
        assert str(config.db_file) == unicode_path
