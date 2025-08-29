#  Copyright (c) 2025, TensorCast Team.

"""Tests for lifecycle worker module."""

import time
import threading
from unittest.mock import Mock, MagicMock, patch

import pytest

from tensorcast.store_daemon.lifecycle_worker import LifecycleWorker
from tensorcast.store_daemon.replica_ref import ReplicaKey


class TestLifecycleWorker:
    """Test cases for LifecycleWorker."""

    def test_lifecycle_worker_creation(self):
        """Test creating a lifecycle worker."""
        process_watcher = Mock()
        replica_manager = Mock()

        worker = LifecycleWorker(
            process_watcher=process_watcher,
            replica_manager=replica_manager,
            eviction_check_interval_seconds=10.0,
            gpu_memory_limit_fraction=0.8,
        )

        assert worker.process_watcher == process_watcher
        assert worker.replica_manager == replica_manager
        assert worker.eviction_interval == 10.0
        assert worker.gpu_memory_threshold == 0.8

    def test_start_stop(self):
        """Test starting and stopping the worker."""
        process_watcher = Mock()
        replica_manager = Mock()

        worker = LifecycleWorker(
            process_watcher=process_watcher,
            replica_manager=replica_manager,
            eviction_check_interval_seconds=0.1,
        )

        # Start worker
        worker.start()
        process_watcher.start.assert_called_once()
        assert worker._eviction_thread is not None
        assert worker._eviction_thread.is_alive()

        # Stop worker
        worker.stop()
        process_watcher.stop.assert_called_once()
        assert worker._eviction_thread is None

    def test_context_manager(self):
        """Test using worker as context manager."""
        process_watcher = Mock()
        replica_manager = Mock()

        with LifecycleWorker(
            process_watcher=process_watcher,
            replica_manager=replica_manager,
            eviction_check_interval_seconds=0.1,
        ) as worker:
            process_watcher.start.assert_called_once()
            assert worker._eviction_thread is not None

        process_watcher.stop.assert_called_once()

    def test_eviction_check_no_gpus(self):
        """Test eviction check when no GPUs are present."""
        process_watcher = Mock()
        replica_manager = Mock()
        replica_manager.get_gpu_memory_stats.return_value = {}

        worker = LifecycleWorker(
            process_watcher=process_watcher,
            replica_manager=replica_manager,
            gpu_memory_limit_fraction=0.9,
        )

        # Should not crash when no GPUs
        worker._check_and_evict()
        replica_manager.periodic_evict.assert_not_called()

    def test_eviction_check_below_threshold(self):
        """Test eviction check when memory usage is below threshold."""
        process_watcher = Mock()
        replica_manager = Mock()
        replica_manager.get_gpu_memory_stats.return_value = {
            0: {"total": 10 * 1024**3, "used": 5 * 1024**3, "free": 5 * 1024**3}
        }
        # Mock the device cache bytes to return 5GB (50% of 10GB)
        replica_manager.get_device_cache_bytes.return_value = int(5 * 1024**3)

        worker = LifecycleWorker(
            process_watcher=process_watcher,
            replica_manager=replica_manager,
            gpu_memory_limit_fraction=0.9,  # 90% threshold
        )

        # 50% usage is below 90% threshold
        worker._check_and_evict()
        replica_manager.periodic_evict.assert_not_called()

    def test_eviction_check_above_threshold(self):
        """Test eviction check when memory usage exceeds threshold."""
        process_watcher = Mock()
        replica_manager = Mock()
        replica_manager.get_gpu_memory_stats.return_value = {
            0: {"total": 10 * 1024**3, "used": 9.5 * 1024**3, "free": 0.5 * 1024**3}
        }
        # Mock the device cache bytes to return 9.5GB (95% of 10GB)
        replica_manager.get_device_cache_bytes.return_value = int(9.5 * 1024**3)

        worker = LifecycleWorker(
            process_watcher=process_watcher,
            replica_manager=replica_manager,
            gpu_memory_limit_fraction=0.9,  # 90% threshold
        )

        # 95% usage exceeds 90% threshold
        worker._check_and_evict()
        replica_manager.periodic_evict.assert_called_once_with(
            device_id=0,
            bytes_needed=int(0.5 * 1024**3),  # 9.5GB - 9GB = 0.5GB
        )

    def test_eviction_check_multiple_gpus(self):
        """Test eviction check with multiple GPUs."""
        process_watcher = Mock()
        replica_manager = Mock()
        replica_manager.get_gpu_memory_stats.return_value = {
            0: {"total": 10 * 1024**3, "used": 9.5 * 1024**3, "free": 0.5 * 1024**3},
            1: {"total": 8 * 1024**3, "used": 4 * 1024**3, "free": 4 * 1024**3},
            2: {"total": 12 * 1024**3, "used": 11 * 1024**3, "free": 1 * 1024**3},
        }

        # Mock the device cache bytes for each GPU
        def get_device_cache_bytes(device_id):
            if device_id == 0:
                return int(9.5 * 1024**3)  # 95% of 10GB
            elif device_id == 1:
                return int(4 * 1024**3)  # 50% of 8GB
            elif device_id == 2:
                return int(11 * 1024**3)  # ~92% of 12GB
            return 0

        replica_manager.get_device_cache_bytes.side_effect = get_device_cache_bytes

        worker = LifecycleWorker(
            process_watcher=process_watcher,
            replica_manager=replica_manager,
            gpu_memory_limit_fraction=0.9,
        )

        worker._check_and_evict()

        # Should trigger eviction for GPU 0 and 2
        assert replica_manager.periodic_evict.call_count == 2
        calls = replica_manager.periodic_evict.call_args_list

        # Check GPU 0 eviction (9.5GB - 9GB = 0.5GB)
        assert any(call[1]["device_id"] == 0 for call in calls)
        assert any(call[1]["bytes_needed"] == int(0.5 * 1024**3) for call in calls)

        # Check GPU 2 eviction (11GB - 10.8GB = 0.2GB)
        assert any(call[1]["device_id"] == 2 for call in calls)
        assert any(call[1]["bytes_needed"] == int(0.2 * 1024**3) for call in calls)

    def test_eviction_check_exception_handling(self):
        """Test that exceptions in eviction check are handled."""
        process_watcher = Mock()
        replica_manager = Mock()
        replica_manager.get_gpu_memory_stats.side_effect = Exception("Test error")

        worker = LifecycleWorker(
            process_watcher=process_watcher, replica_manager=replica_manager
        )

        # Should not crash
        worker._check_and_evict()

    def test_trigger_eviction_check(self):
        """Test manual eviction trigger."""
        process_watcher = Mock()
        replica_manager = Mock()
        replica_manager.get_gpu_memory_stats.return_value = {
            0: {"total": 10 * 1024**3, "used": 9.5 * 1024**3, "free": 0.5 * 1024**3}
        }
        # Mock the device cache bytes to return 9.5GB (95% of 10GB)
        replica_manager.get_device_cache_bytes.return_value = int(9.5 * 1024**3)

        worker = LifecycleWorker(
            process_watcher=process_watcher,
            replica_manager=replica_manager,
            gpu_memory_limit_fraction=0.9,
        )

        # Manual trigger
        worker.trigger_eviction_check()
        replica_manager.periodic_evict.assert_called_once()

    def test_background_eviction_loop(self):
        """Test that background eviction loop runs periodically."""
        process_watcher = Mock()
        replica_manager = Mock()
        replica_manager.get_gpu_memory_stats.return_value = {
            0: {"total": 10 * 1024**3, "used": 9.5 * 1024**3, "free": 0.5 * 1024**3}
        }
        # Mock the device cache bytes to return 9.5GB (95% of 10GB)
        replica_manager.get_device_cache_bytes.return_value = int(9.5 * 1024**3)

        check_count = threading.Event()
        original_check = replica_manager.periodic_evict

        def counting_check(*args, **kwargs):
            original_check(*args, **kwargs)
            check_count.set()

        replica_manager.periodic_evict = Mock(side_effect=counting_check)

        worker = LifecycleWorker(
            process_watcher=process_watcher,
            replica_manager=replica_manager,
            eviction_check_interval_seconds=0.1,
            gpu_memory_limit_fraction=0.9,
        )

        # Start and wait for at least one check
        worker.start()
        try:
            assert check_count.wait(timeout=1.0)
            assert replica_manager.periodic_evict.call_count >= 1
        finally:
            worker.stop()

    def test_multiple_start_stop(self):
        """Test multiple start/stop cycles."""
        process_watcher = Mock()
        replica_manager = Mock()

        worker = LifecycleWorker(
            process_watcher=process_watcher,
            replica_manager=replica_manager,
            eviction_check_interval_seconds=0.1,
        )

        # Multiple starts should be safe
        worker.start()
        worker.start()  # Should log warning
        worker.stop()

        # Should be able to restart
        worker.start()
        assert worker._eviction_thread is not None
        worker.stop()
