#  Copyright (c) 2025, TensorCast Team.

"""CLI utilities for tensorcast."""

from tensorcast.cli_utils.config_loader import (
    ConfigError,
    load_config,
    print_config_summary,
    validate_config,
)
from tensorcast.cli_utils.daemon import DaemonizationError, daemonize
from tensorcast.cli_utils.pid_manager import (
    PidManagerError,
    cleanup_pid_file,
    get_process_info,
    is_process_running,
    read_pid_file,
    stop_process,
    write_pid_file,
)
from tensorcast.cli_utils.service_manager import (
    ServiceError,
    check_service_status,
    start_service,
    stop_service,
)

# Default paths
DEFAULT_PID_FILE = "/tmp/tensorcast_daemon.pid"
DEFAULT_LOG_FILE = "/tmp/tensorcast-daemon.log"

__all__ = [
    # Exceptions
    "ConfigError",
    "DaemonizationError",
    "PidManagerError",
    "ServiceError",
    # Config functions
    "load_config",
    "validate_config",
    "print_config_summary",
    # Daemon functions
    "daemonize",
    # PID functions
    "write_pid_file",
    "read_pid_file",
    "is_process_running",
    "cleanup_pid_file",
    "stop_process",
    "get_process_info",
    # Service functions
    "start_service",
    "stop_service",
    "check_service_status",
    # Constants
    "DEFAULT_PID_FILE",
    "DEFAULT_LOG_FILE",
]
