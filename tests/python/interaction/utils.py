#  Copyright (c) 2025, TensorCast Team.

"""Shared testing utilities for interaction tests."""

from __future__ import annotations

import socket
import grpc


class FakeContext(grpc.ServicerContext):
    """Minimal stub for :class:`grpc.ServicerContext` used by in-process calls.

    Only implements the small subset of methods that tests rely on, namely
    ``set_code`` and ``set_details``.  All other abstract methods raise
    ``NotImplementedError`` if accessed – they are not needed for the current
    tests."""

    def __init__(self) -> None:
        self.code: grpc.StatusCode | None = None
        self.details: str | None = None

    # Required overrides -----------------------------------------------------

    def set_code(self, code: grpc.StatusCode) -> None:  # noqa: D401
        self.code = code

    def set_details(self, details: str) -> None:  # noqa: D401
        self.details = details

    # The remaining abstract methods of ``ServicerContext`` are not needed for
    # these unit tests.  Implement them as simple no-ops or properties that
    # satisfy the abstract base but provide no functionality.

    def abort(self, code, details):
        raise NotImplementedError

    def abort_with_status(self, status):
        raise NotImplementedError

    def add_callback(self, callback):
        raise NotImplementedError

    def auth_context(self):
        raise NotImplementedError

    def cancel(self):
        raise NotImplementedError

    def invocation_metadata(self):
        raise NotImplementedError

    def is_active(self):
        return True

    def peer(self):
        raise NotImplementedError

    def peer_identities(self):
        raise NotImplementedError

    def peer_identity_key(self):
        raise NotImplementedError

    def send_initial_metadata(self, initial_metadata):
        raise NotImplementedError

    def set_trailing_metadata(self, trailing_metadata):
        raise NotImplementedError

    def time_remaining(self):
        return None


def get_free_port() -> int:
    """Get a free port from the OS."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def get_free_port_pair() -> tuple[int, int]:
    """Get a pair of free ports for RPC and P2P/metrics."""
    return get_free_port(), get_free_port()