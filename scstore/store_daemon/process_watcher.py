#  Copyright (c) 2025, TensorCast Team.

"""Process watcher for monitoring PID lifecycle."""

import logging
import os
import threading
from typing import Callable, Dict, Optional, Set

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)


class ProcessWatcher:
    """Monitors process lifecycle by checking PID existence.

    Periodically polls /proc/<pid> to detect when processes have terminated.
    When a PID is detected as dead, calls the configured callback.
    """

    def __init__(
        self,
        check_interval_seconds: float = 5.0,
        on_pid_dead: Callable[[int], None] | None = None,
    ):
        """Initialize the process watcher.

        Args:
            check_interval_seconds: How often to check PIDs
            on_pid_dead: Callback when a PID is detected as dead
        """
        self.check_interval = check_interval_seconds
        self.on_pid_dead = on_pid_dead
        self._pids: Set[int] = set()
        # Track PID start times to prevent PID reuse issues
        self._pid_start_times: Dict[int, float] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def add_pid(self, pid: int) -> None:
        """Add a PID to monitor."""
        with self._lock:
            if pid > 0:  # Valid PID
                self._pids.add(pid)
                # Get and cache process start time
                start_time = self._get_process_start_time(pid)
                if start_time is not None:
                    self._pid_start_times[pid] = start_time
                logger.debug(f"Added PID {pid} to process watcher")

    def remove_pid(self, pid: int) -> None:
        """Remove a PID from monitoring."""
        with self._lock:
            self._pids.discard(pid)
            self._pid_start_times.pop(pid, None)
            logger.debug(f"Removed PID {pid} from process watcher")

    def get_monitored_pids(self) -> Set[int]:
        """Get copy of currently monitored PIDs."""
        with self._lock:
            return self._pids.copy()

    def start(self) -> None:
        """Start the background monitoring thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Process watcher already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._monitor_loop, name="ProcessWatcher", daemon=True
        )
        self._thread.start()
        logger.info(f"Started process watcher with interval {self.check_interval}s")

    def stop(self) -> None:
        """Stop the background monitoring thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("Stopped process watcher")

    def _monitor_loop(self) -> None:
        """Main monitoring loop running in background thread."""
        while not self._stop_event.is_set():
            try:
                self.scan()
            except Exception:
                logger.exception("Error in process watcher scan")

            # Sleep with interruptible wait
            self._stop_event.wait(self.check_interval)

    def scan(self) -> Set[int]:
        """Scan all monitored PIDs and detect dead ones.

        Returns:
            Set of dead PIDs detected
        """
        dead_pids = set()

        # Get snapshot of PIDs to check with their expected start times
        with self._lock:
            pids_to_check = list(self._pids)
            expected_start_times = self._pid_start_times.copy()

        # Batch check if we have psutil
        if HAS_PSUTIL:
            alive_pids = self._batch_check_pids(pids_to_check, expected_start_times)
            dead_pids = set(pids_to_check) - alive_pids
        else:
            # Fallback to individual checks
            for pid in pids_to_check:
                expected_start = expected_start_times.get(pid)
                if not self._is_process_alive_with_start_time(pid, expected_start):
                    dead_pids.add(pid)

        # Process dead PIDs
        for pid in dead_pids:
            # Remove from monitoring
            with self._lock:
                self._pids.discard(pid)
                self._pid_start_times.pop(pid, None)

            # Notify callback
            if self.on_pid_dead:
                try:
                    self.on_pid_dead(pid)
                except Exception:
                    logger.exception(f"Error in on_pid_dead callback for PID {pid}")

        if dead_pids:
            logger.info(f"Detected {len(dead_pids)} dead PIDs: {dead_pids}")

        return dead_pids

    def _is_process_alive_with_start_time(
        self, pid: int, expected_start_time: Optional[float]
    ) -> bool:
        """Check if a process is still alive and has the expected start time.

        This prevents false positives from PID reuse.
        """
        if pid <= 0:
            return False

        try:
            # Check if process exists
            if not os.path.exists(f"/proc/{pid}"):
                return False

            # If we have an expected start time, verify it matches
            if expected_start_time is not None:
                actual_start_time = self._get_process_start_time(pid)
                if (
                    actual_start_time is None
                    or actual_start_time != expected_start_time
                ):
                    logger.warning(
                        f"PID {pid} start time mismatch: expected {expected_start_time}, "
                        f"got {actual_start_time}. Likely PID reuse."
                    )
                    return False

            return True
        except Exception:
            # If we can't determine, assume it's dead to be safe
            return False

    @staticmethod
    def _get_process_start_time(pid: int) -> Optional[float]:
        """Get the start time of a process.

        Returns the start time in seconds since boot, or None if not available.
        """
        try:
            # Read /proc/[pid]/stat
            with open(f"/proc/{pid}/stat", "r") as f:
                stat_line = f.read()

            # The start time is the 22nd field (0-indexed: field 21)
            # We need to handle the case where the process name contains spaces/parens
            # Find the last ')' which closes the process name
            end_paren = stat_line.rfind(")")
            if end_paren == -1:
                return None

            # Fields after the process name
            fields = stat_line[end_paren + 1 :].split()
            if len(fields) < 20:  # We need at least 20 fields after the name
                return None

            # Field 19 (0-indexed) after the name is start time in clock ticks
            start_time_ticks = int(fields[19])

            # Convert to seconds (assumes USER_HZ = 100, which is typical)
            # This gives us a unique identifier for the process
            return start_time_ticks / 100.0

        except Exception:
            return None

    def _batch_check_pids(
        self, pids: list[int], expected_start_times: Dict[int, float]
    ) -> Set[int]:
        """Batch check multiple PIDs for efficiency.

        Returns the set of PIDs that are still alive.
        """
        alive_pids = set()

        if not HAS_PSUTIL:
            # Fallback to individual checks
            for pid in pids:
                expected_start = expected_start_times.get(pid)
                if self._is_process_alive_with_start_time(pid, expected_start):
                    alive_pids.add(pid)
            return alive_pids

        # Use psutil for batch checking if available
        if psutil is not None:
            for pid in pids:
                try:
                    proc = psutil.Process(pid)
                    # Check if process is still running
                    if proc.is_running():
                        # Verify start time if we have it
                        expected_start = expected_start_times.get(pid)
                        if expected_start is not None:
                            # psutil gives create_time as seconds since epoch
                            # We need to compare with our start time format
                            actual_start = self._get_process_start_time(pid)
                            if actual_start == expected_start:
                                alive_pids.add(pid)
                            else:
                                logger.warning(
                                    f"PID {pid} start time mismatch in batch check"
                                )
                        else:
                            alive_pids.add(pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    # Process doesn't exist or we can't access it
                    pass
        else:
            # Fallback to /proc-based checking
            for pid in pids:
                if os.path.exists(f"/proc/{pid}"):
                    alive_pids.add(pid)

        return alive_pids

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
