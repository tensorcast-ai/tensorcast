#  Copyright (c) 2025, TensorCast Team.

"""CLI utilities for tensorcast.

Exports Linux-only service management primitives for the daemon.
"""

from tensorcast.cli_utils.service_manager import (
    ServiceError,
    check_service_status,
    get_current_instance_id,
    list_instances,
    logs_tail,
    start_service,
    stop_service,
)

__all__ = [
    "ServiceError",
    "start_service",
    "stop_service",
    "check_service_status",
    "logs_tail",
    "get_current_instance_id",
    "list_instances",
]
