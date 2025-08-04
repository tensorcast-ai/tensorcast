#  Copyright (c) 2025, StepCast Team.

"""PID file management utilities for daemon processes."""

import os
import signal
from pathlib import Path

from scstore.logger import init_logger

logger = init_logger(__name__)


class PidManagerError(Exception):
    """Base exception for PID manager operations."""

    pass


def write_pid_file(pid_file: Path) -> None:
    """
    Write the current process PID to a file.

    Args:
        pid_file: Path to the PID file

    Raises:
        PidManagerError: If unable to write PID file
    """
    try:
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(str(os.getpid()))
        logger.info(f"PID {os.getpid()} written to {pid_file}")
    except Exception as e:
        error_msg = f"Failed to write PID file: {e}"
        logger.error(error_msg)
        raise PidManagerError(error_msg) from e


def read_pid_file(pid_file: Path) -> int | None:
    """
    Read PID from file.

    Args:
        pid_file: Path to the PID file

    Returns:
        The PID if file exists and is valid, None otherwise
    """
    if not pid_file.exists():
        return None

    try:
        pid_str = pid_file.read_text().strip()
        if not pid_str:
            return None
        return int(pid_str)
    except (ValueError, OSError) as e:
        logger.error(f"Failed to read PID file: {e}")
        return None


def is_process_running(pid: int) -> bool:
    """
    Check if a process with given PID is running.

    Args:
        pid: Process ID to check

    Returns:
        True if process is running, False otherwise
    """
    if pid <= 0:
        return False

    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def cleanup_pid_file(pid_file: Path) -> None:
    """
    Remove PID file if it exists.

    Args:
        pid_file: Path to the PID file
    """
    if not pid_file.exists():
        return

    try:
        pid_file.unlink()
        logger.info(f"Removed PID file: {pid_file}")
    except Exception as e:
        logger.error(f"Failed to remove PID file: {e}")


def stop_process(pid: int, force: bool = False, timeout: int = 10) -> bool:
    """
    Stop a process by PID.

    Args:
        pid: Process ID to stop
        force: If True, use SIGKILL instead of SIGTERM
        timeout: Seconds to wait for graceful shutdown (ignored if force=True)

    Returns:
        True if process was stopped successfully, False otherwise
    """
    if not is_process_running(pid):
        return True

    sig = signal.SIGKILL if force else signal.SIGTERM

    try:
        os.kill(pid, sig)

        if force:
            return True

        # Wait for graceful shutdown
        import time

        for _ in range(timeout):
            time.sleep(1)
            if not is_process_running(pid):
                return True

        # Force kill if still running
        logger.warning(f"Process {pid} didn't stop gracefully, sending SIGKILL")
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.5)
        return not is_process_running(pid)

    except ProcessLookupError:
        return True
    except PermissionError:
        logger.error(f"Permission denied to stop process {pid}")
        return False
    except Exception as e:
        logger.error(f"Error stopping process {pid}: {e}")
        return False


def get_process_info(pid: int) -> dict[str, str | float] | None:
    """
    Get information about a running process.

    Args:
        pid: Process ID

    Returns:
        Dictionary with process info or None if process not found
    """
    if not is_process_running(pid):
        return None

    try:
        import time

        import psutil

        proc = psutil.Process(pid)
        return {
            "pid": pid,
            "started": time.ctime(proc.create_time()),
            "cpu_percent": proc.cpu_percent(),
            "memory_mb": proc.memory_info().rss / 1024 / 1024,
        }
    except ImportError:
        # psutil not available, return minimal info
        return {"pid": pid}
    except Exception as e:
        logger.debug(f"Failed to get process info: {e}")
        return {"pid": pid}
