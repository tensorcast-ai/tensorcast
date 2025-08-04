#  Copyright (c) 2025, StepCast Team.

"""Utilities for managing ports in tests."""

import socket
from typing import List, Tuple


def get_free_port() -> int:
    """Get a free port from the OS."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def get_free_ports(count: int) -> List[int]:
    """Get multiple free ports from the OS."""
    ports = []
    for _ in range(count):
        ports.append(get_free_port())
    return ports


def get_free_port_pair() -> Tuple[int, int]:
    """Get a pair of free ports for RPC and metrics."""
    return get_free_port(), get_free_port()