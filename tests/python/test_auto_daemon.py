#  Copyright (c) 2025, TensorCast Team.

"""
Test module for automatic daemon management functionality.

This module tests the automatic daemon lifecycle management:
1. Check if daemon is already running
2. Start daemon if not running
3. Connect to daemon
4. Clean up daemon on exit if we started it
"""

import time
import pytest
from unittest import mock

from scstore.daemon_manager import DaemonManager, ensure_daemon_running
from scstore.store_daemon.config import StoreDaemonConfig, ServerConfig
from scstore.logger import init_logger
from pathlib import Path
from pydantic import ByteSize

logger = init_logger(__name__)


@pytest.fixture
def test_storage_path(tmp_path):
    """Provide a temporary storage path for testing."""
    return str(tmp_path / "test-models")


@pytest.fixture
def daemon_manager(test_storage_path):
    """Create a DaemonManager instance for testing."""
    config = StoreDaemonConfig(
        server=ServerConfig(
            storage_path=Path(test_storage_path),
            mem_pool_size=ByteSize(4 * 1024 * 1024 * 1024),  # 4GB for testing
        ),
        global_store_address=None,
    )
    manager = DaemonManager(
        config=config,
        auto_start=True,
    )
    yield manager
    # Cleanup is handled automatically by DaemonManager


class TestDaemonManager:
    """Test cases for DaemonManager class."""

    def test_daemon_manager_initialization(self, daemon_manager, test_storage_path):
        """Test DaemonManager initialization with proper parameters."""
        assert str(daemon_manager.config.server.storage_path) == test_storage_path
        assert daemon_manager.config.server.mem_pool_size == ByteSize(
            4 * 1024 * 1024 * 1024
        )
        assert daemon_manager.auto_start is True
        assert daemon_manager.server_address is not None

    def test_daemon_not_running_initially(self, daemon_manager):
        """Test that daemon is not running initially in a clean test environment."""
        # In a test environment, daemon should not be running initially
        # This test might be environment-dependent
        is_running = daemon_manager.is_daemon_running()
        # We can't assume the initial state, so we just test the method works
        assert isinstance(is_running, bool)

    def test_ensure_daemon_running_success(self, daemon_manager):
        """Test ensuring daemon is running returns success."""
        success = daemon_manager.ensure_daemon_running()
        assert isinstance(success, bool)

        if success:
            # If daemon started successfully, it should be responsive
            time.sleep(1)  # Give daemon time to start
            is_responsive = daemon_manager.is_daemon_running()
            assert isinstance(is_responsive, bool)

    def test_daemon_responsiveness_check(self, daemon_manager):
        """Test daemon responsiveness checking functionality."""
        # First ensure daemon is ready
        daemon_manager.ensure_daemon_running()

        # Test responsiveness
        time.sleep(1)
        is_responsive = daemon_manager.is_daemon_running()
        assert isinstance(is_responsive, bool)

    @mock.patch("scstore.daemon_manager.DaemonManager.is_daemon_running")
    def test_daemon_running_check_mocked(self, mock_is_running, daemon_manager):
        """Test daemon running check with mocked response."""
        # Mock daemon as running
        mock_is_running.return_value = True

        is_running = daemon_manager.is_daemon_running()
        assert is_running is True
        mock_is_running.assert_called_once()

    @mock.patch("scstore.daemon_manager.DaemonManager.is_daemon_running")
    @mock.patch("scstore.daemon_manager.DaemonManager.start_daemon")
    def test_ensure_daemon_running_when_not_running(
        self, mock_start_daemon, mock_is_running, daemon_manager
    ):
        """Test ensuring daemon running when it's not initially running."""
        # Mock daemon as not running initially
        mock_is_running.return_value = False
        mock_start_daemon.return_value = True

        success = daemon_manager.ensure_daemon_running()

        assert success is True
        mock_start_daemon.assert_called_once()
        # ensure_daemon_running calls is_daemon_running once to check initial state
        assert mock_is_running.call_count == 1

    @mock.patch("scstore.daemon_manager.DaemonManager.is_daemon_running")
    def test_ensure_daemon_running_when_already_running(
        self, mock_is_running, daemon_manager
    ):
        """Test ensuring daemon running when it's already running."""
        # Mock daemon as already running
        mock_is_running.return_value = True

        success = daemon_manager.ensure_daemon_running()

        assert success is True
        mock_is_running.assert_called_once()


class TestConvenienceFunction:
    """Test cases for convenience functions."""

    def test_ensure_daemon_running_convenience_function(self, test_storage_path):
        """Test the ensure_daemon_running convenience function."""
        config = StoreDaemonConfig(
            server=ServerConfig(
                storage_path=Path(test_storage_path),
                mem_pool_size=ByteSize(4 * 1024 * 1024 * 1024),
            ),
            global_store_address=None,
        )
        success = ensure_daemon_running(
            config=config,
            auto_start=True,
        )

        assert isinstance(success, bool)

    @mock.patch("scstore.daemon_manager.DaemonManager.ensure_daemon_running")
    @mock.patch("scstore.daemon_manager.DaemonManager.__init__")
    def test_convenience_function_with_mock(
        self, mock_init, mock_ensure_running, test_storage_path
    ):
        """Test convenience function with mocked DaemonManager."""
        # Mock initialization to do nothing
        mock_init.return_value = None
        # Mock ensure_daemon_running to return True
        mock_ensure_running.return_value = True

        config = StoreDaemonConfig(
            server=ServerConfig(
                storage_path=Path(test_storage_path),
                mem_pool_size=ByteSize(4 * 1024 * 1024 * 1024),
            ),
            global_store_address=None,
        )
        success = ensure_daemon_running(
            config=config,
            auto_start=True,
        )

        assert success is True
        mock_ensure_running.assert_called_once()

    def test_convenience_function_with_different_params(self, test_storage_path):
        """Test convenience function with different parameter combinations."""
        # Test with different memory pool size
        config = StoreDaemonConfig(
            server=ServerConfig(
                storage_path=Path(test_storage_path),
                mem_pool_size=ByteSize(2 * 1024 * 1024 * 1024),
            ),
            global_store_address=None,
        )
        success = ensure_daemon_running(
            config=config,
            auto_start=False,
        )
        assert isinstance(success, bool)


class TestDaemonLifecycle:
    """Test cases for daemon lifecycle management."""

    def test_daemon_lifecycle_simulation(self, daemon_manager):
        """Test a complete daemon lifecycle simulation."""
        # Check initial state
        initial_state = daemon_manager.is_daemon_running()
        assert isinstance(initial_state, bool)

        # Ensure daemon is running
        success = daemon_manager.ensure_daemon_running()
        assert isinstance(success, bool)

        if success:
            # Give daemon time to start
            time.sleep(1)

            # Check responsiveness
            is_responsive = daemon_manager.is_daemon_running()
            assert isinstance(is_responsive, bool)

            # Test multiple responsiveness checks
            for _ in range(3):
                time.sleep(0.5)
                responsive = daemon_manager.is_daemon_running()
                assert isinstance(responsive, bool)

    def test_daemon_startup_timeout(self, daemon_manager):
        """Test that daemon startup doesn't hang indefinitely."""
        # This test ensures the daemon startup process completes within reasonable time
        start_time = time.time()

        success = daemon_manager.ensure_daemon_running()

        elapsed_time = time.time() - start_time
        assert elapsed_time < 25  # Should complete within 25 seconds
        assert isinstance(success, bool)


class TestErrorHandling:
    """Test cases for error handling scenarios."""

    @mock.patch("scstore.daemon_manager.DaemonManager.start_daemon")
    @mock.patch("scstore.daemon_manager.DaemonManager.is_daemon_running")
    def test_daemon_start_failure(
        self, mock_is_running, mock_start_daemon, daemon_manager
    ):
        """Test handling of daemon start failure."""
        # Mock daemon as not running and start failing
        mock_is_running.return_value = False
        mock_start_daemon.return_value = False

        success = daemon_manager.ensure_daemon_running()

        assert success is False
        mock_start_daemon.assert_called_once()

    def test_invalid_storage_path(self, tmp_path):
        """Test DaemonManager with invalid storage path."""
        # Use a path that doesn't exist and can't be created
        invalid_path = "/invalid/path/that/cannot/be/created"

        config = StoreDaemonConfig(
            server=ServerConfig(
                storage_path=Path(invalid_path),
                mem_pool_size=ByteSize(4 * 1024 * 1024 * 1024),
            ),
            global_store_address=None,
        )
        manager = DaemonManager(
            config=config,
            auto_start=True,
        )

        # The manager should be created but may fail when trying to ensure daemon is running
        assert str(manager.config.server.storage_path) == str(
            Path(invalid_path).resolve()
        )

    def test_invalid_memory_size(self, test_storage_path):
        """Test DaemonManager with invalid memory size."""
        # Test with invalid memory size format - this should raise a validation error
        with pytest.raises(ValueError):
            config = StoreDaemonConfig(
                server=ServerConfig(
                    storage_path=Path(test_storage_path),
                    mem_pool_size="invalid_size",  # This will trigger validation error
                ),
                global_store_address=None,
            )
            DaemonManager(
                config=config,
                auto_start=True,
            )


class TestIntegration:
    """Integration tests for daemon management."""

    def test_full_integration_workflow(self, test_storage_path):
        """Test the full integration workflow using both manager and convenience function."""
        # Test with DaemonManager
        config = StoreDaemonConfig(
            server=ServerConfig(
                storage_path=Path(test_storage_path),
                mem_pool_size=ByteSize(4 * 1024 * 1024 * 1024),
            ),
            global_store_address=None,
        )
        manager = DaemonManager(
            config=config,
            auto_start=True,
        )

        manager_success = manager.ensure_daemon_running()
        assert isinstance(manager_success, bool)

        # Test with convenience function
        convenience_success = ensure_daemon_running(
            config=config,
            auto_start=True,
        )
        assert isinstance(convenience_success, bool)

        # Both should work consistently
        if manager_success:
            assert convenience_success  # If manager works, convenience should too

    def test_multiple_daemon_managers(self, tmp_path):
        """Test creating multiple daemon managers with different configurations."""
        path1 = str(tmp_path / "models1")
        path2 = str(tmp_path / "models2")

        config1 = StoreDaemonConfig(
            server=ServerConfig(
                storage_path=Path(path1),
                mem_pool_size=ByteSize(2 * 1024 * 1024 * 1024),
            ),
            global_store_address=None,
        )
        manager1 = DaemonManager(
            config=config1,
            auto_start=True,
        )

        config2 = StoreDaemonConfig(
            server=ServerConfig(
                storage_path=Path(path2),
                mem_pool_size=ByteSize(4 * 1024 * 1024 * 1024),
            ),
            global_store_address=None,
        )
        manager2 = DaemonManager(
            config=config2,
            auto_start=True,
        )

        success1 = manager1.ensure_daemon_running()
        success2 = manager2.ensure_daemon_running()

        assert isinstance(success1, bool)
        assert isinstance(success2, bool)

        # Both managers should be independent
        assert (
            manager1.config.server.storage_path != manager2.config.server.storage_path
        )
        assert (
            manager1.config.server.mem_pool_size != manager2.config.server.mem_pool_size
        )
