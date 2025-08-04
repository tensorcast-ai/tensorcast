#  Copyright (c) 2025, StepCast Team.

"""Unix daemon process utilities."""

import os
import sys
from pathlib import Path

from scstore.logger import init_logger

logger = init_logger(__name__)


class DaemonizationError(Exception):
    """Base exception for daemonization operations."""

    pass


def daemonize(log_file: Path | None = None) -> None:
    """
    Daemonize the current process.

    This will detach from terminal and run in background following
    the Unix double-fork magic.

    Args:
        log_file: Optional path to redirect stdout/stderr to

    Raises:
        DaemonizationError: If daemonization fails
    """
    # First fork - decouple from parent environment
    try:
        pid = os.fork()
        if pid > 0:
            # Exit parent process
            sys.exit(0)
    except OSError as e:
        raise DaemonizationError(f"First fork failed: {e}") from e

    # Decouple from parent environment
    os.chdir("/")
    os.setsid()
    os.umask(0)

    # Second fork - prevent acquiring controlling TTY
    try:
        pid = os.fork()
        if pid > 0:
            # Exit second parent
            sys.exit(0)
    except OSError as e:
        raise DaemonizationError(f"Second fork failed: {e}") from e

    # Redirect file descriptors
    _redirect_file_descriptors(log_file)


def _redirect_file_descriptors(log_file: Path | None) -> None:
    """
    Redirect stdin, stdout, and stderr for daemon process.

    Args:
        log_file: Optional path to redirect stdout/stderr to
    """
    # Flush any pending output
    sys.stdout.flush()
    sys.stderr.flush()

    # Redirect stdin to /dev/null
    with open("/dev/null", "r") as devnull:
        os.dup2(devnull.fileno(), sys.stdin.fileno())

    # Redirect stdout and stderr
    if log_file:
        _redirect_to_file(log_file)
    else:
        _redirect_to_devnull()


def _redirect_to_file(log_file: Path) -> None:
    """
    Redirect stdout and stderr to a log file.

    Args:
        log_file: Path to the log file
    """
    try:
        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Open log file in append mode with line buffering
        with open(log_file, "a", buffering=1) as log_fd:
            os.dup2(log_fd.fileno(), sys.stdout.fileno())
            os.dup2(log_fd.fileno(), sys.stderr.fileno())
    except Exception as e:
        logger.error(f"Failed to redirect to log file: {e}")
        # Fall back to /dev/null
        _redirect_to_devnull()


def _redirect_to_devnull() -> None:
    """Redirect stdout and stderr to /dev/null."""
    with open("/dev/null", "w") as devnull:
        os.dup2(devnull.fileno(), sys.stdout.fileno())
        os.dup2(devnull.fileno(), sys.stderr.fileno())
