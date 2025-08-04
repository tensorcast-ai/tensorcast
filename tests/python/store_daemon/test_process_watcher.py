#  Copyright (c) 2025, StepCast Team.

"""Tests for process watcher module."""

import os
import threading
import time
import subprocess
import pytest

from scstore.store_daemon.process_watcher import ProcessWatcher


class TestProcessWatcher:
    """Test cases for ProcessWatcher."""

    def test_process_watcher_creation(self):
        """Test creating a process watcher."""
        watcher = ProcessWatcher(check_interval_seconds=1.0)
        assert watcher.check_interval == 1.0
        assert watcher.on_pid_dead is None
        assert len(watcher.get_monitored_pids()) == 0

    def test_add_remove_pid(self):
        """Test adding and removing PIDs."""
        watcher = ProcessWatcher()

        # Add PIDs
        watcher.add_pid(1234)
        watcher.add_pid(5678)
        assert 1234 in watcher.get_monitored_pids()
        assert 5678 in watcher.get_monitored_pids()

        # Remove PID
        watcher.remove_pid(1234)
        assert 1234 not in watcher.get_monitored_pids()
        assert 5678 in watcher.get_monitored_pids()

        # Remove non-existent PID (should not raise)
        watcher.remove_pid(9999)

    def test_invalid_pid_handling(self):
        """Test handling of invalid PIDs."""
        watcher = ProcessWatcher()

        # Invalid PIDs should not be added
        watcher.add_pid(0)
        watcher.add_pid(-1)
        assert len(watcher.get_monitored_pids()) == 0

    def test_is_process_alive(self):
        """Test process alive detection."""
        watcher = ProcessWatcher()

        # Test with current process (should be alive)
        current_pid = os.getpid()
        # Get current process start time
        start_time = watcher._get_process_start_time(current_pid)
        assert (
            watcher._is_process_alive_with_start_time(current_pid, start_time) is True
        )

        # Test with invalid PIDs
        assert watcher._is_process_alive_with_start_time(0, None) is False
        assert watcher._is_process_alive_with_start_time(-1, None) is False

        # Test with likely non-existent PID
        assert watcher._is_process_alive_with_start_time(999999999, None) is False

        # Test PID reuse detection (with mismatched start time)
        if start_time is not None:
            # Use a different start time - should return False
            assert (
                watcher._is_process_alive_with_start_time(
                    current_pid, start_time + 1000
                )
                is False
            )

    def test_scan_dead_processes(self):
        """Test scanning for dead processes."""
        dead_pids = []

        def on_dead(pid):
            dead_pids.append(pid)

        watcher = ProcessWatcher(on_pid_dead=on_dead)

        # Add current process (alive)
        current_pid = os.getpid()
        watcher.add_pid(current_pid)

        # Add non-existent PIDs
        watcher.add_pid(999999997)
        watcher.add_pid(999999998)

        # Scan should detect dead PIDs
        detected = watcher.scan()
        assert 999999997 in detected
        assert 999999998 in detected
        assert current_pid not in detected

        # Dead PIDs should be removed from monitoring
        assert current_pid in watcher.get_monitored_pids()
        assert 999999997 not in watcher.get_monitored_pids()
        assert 999999998 not in watcher.get_monitored_pids()

        # Callback should be called
        assert 999999997 in dead_pids
        assert 999999998 in dead_pids

    def test_callback_exception_handling(self):
        """Test that exceptions in callback don't crash the watcher."""

        def bad_callback(pid):
            raise ValueError(f"Test error for PID {pid}")

        watcher = ProcessWatcher(on_pid_dead=bad_callback)
        watcher.add_pid(999999999)  # Non-existent PID

        # Should not raise despite callback error
        detected = watcher.scan()
        assert 999999999 in detected

    def test_start_stop(self):
        """Test starting and stopping the watcher thread."""
        watcher = ProcessWatcher(check_interval_seconds=0.1)

        # Start watcher
        watcher.start()
        assert watcher._thread is not None
        assert watcher._thread.is_alive()

        # Stop watcher
        watcher.stop()
        assert watcher._thread is None

        # Multiple starts should be safe
        watcher.start()
        watcher.start()  # Should log warning but not crash
        watcher.stop()

    def test_context_manager(self):
        """Test using watcher as context manager."""
        with ProcessWatcher(check_interval_seconds=0.1) as watcher:
            assert watcher._thread is not None
            assert watcher._thread.is_alive()

        # Thread should be stopped after exiting context
        assert watcher._thread is None

    def test_background_monitoring(self):
        """Test that background monitoring detects process death."""
        dead_pids = []
        detected_event = threading.Event()

        def on_dead(pid):
            dead_pids.append(pid)
            detected_event.set()

        # Start a subprocess that will exit quickly
        proc = subprocess.Popen(
            ["python", "-c", "import time; time.sleep(0.1)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        proc_pid = proc.pid

        with ProcessWatcher(
            check_interval_seconds=0.05, on_pid_dead=on_dead
        ) as watcher:
            watcher.add_pid(proc_pid)

            # Wait for process to exit and be detected
            proc.wait()  # Ensure process exits
            detected = detected_event.wait(timeout=2.0)

            assert detected is True
            assert proc_pid in dead_pids

    def test_concurrent_access(self):
        """Test thread-safe operations."""
        watcher = ProcessWatcher()
        errors = []

        def add_pids():
            try:
                for i in range(100):
                    watcher.add_pid(1000 + i)
            except Exception as e:
                errors.append(e)

        def remove_pids():
            try:
                for i in range(100):
                    watcher.remove_pid(1000 + i)
            except Exception as e:
                errors.append(e)

        def scan_pids():
            try:
                for _ in range(10):
                    watcher.scan()
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)

        # Run operations concurrently
        threads = [
            threading.Thread(target=add_pids),
            threading.Thread(target=remove_pids),
            threading.Thread(target=scan_pids),
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Should complete without errors
        assert len(errors) == 0
