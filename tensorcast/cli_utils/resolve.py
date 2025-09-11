#  Copyright (c) 2025, TensorCast Team.

"""Shared address resolution utilities for CLI and library.

Implements unified resolve(mode) semantics per LINUX_INIT_CONNECT_CLI_BEST_PRACTICES:
 - address=None: env TENSORCAST_ADDRESS → current_session → launch
 - address="auto": env/current_session or error (do not launch)
 - address="local": force launch
 - address="host:port": connect-only; error if unreachable
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import grpc

from tensorcast.cli_utils.service_manager import get_session_address
from tensorcast.proto.daemon.v1 import store_daemon_pb2, store_daemon_pb2_grpc


def ping_daemon(address: str, timeout: float = 1.0) -> bool:
    try:
        ch = grpc.insecure_channel(address)
        stub = store_daemon_pb2_grpc.StoreDaemonServiceStub(ch)
        stub.GetServerConfig(store_daemon_pb2.GetServerConfigRequest(), timeout=timeout)
        try:  # noqa: SIM105
            ch.close()
        except Exception:
            pass
        return True
    except Exception:
        return False


@dataclass(slots=True, frozen=True)
class ResolvedAddressMode:
    mode: str  # "connect" | "launch"
    address: Optional[str]


def resolve_address_mode(address: Optional[str]) -> ResolvedAddressMode:
    """Resolve the desired mode and target address.

    Behavior:
      - None: env TENSORCAST_ADDRESS → current_session (if alive) → launch
      - "auto": env/current_session (if alive) → error
      - "local": launch
      - host:port: connect; must be reachable else error
    """
    env_addr = os.environ.get("TENSORCAST_ADDRESS")

    if address is None:
        if env_addr and ping_daemon(env_addr):
            return ResolvedAddressMode("connect", env_addr)
        auto_addr = get_session_address()
        if auto_addr and ping_daemon(auto_addr):
            return ResolvedAddressMode("connect", auto_addr)
        return ResolvedAddressMode("launch", None)

    s = address.strip()
    s_l = s.lower()
    if s_l == "local":
        return ResolvedAddressMode("launch", None)
    if s_l == "auto":
        auto_addr = env_addr or get_session_address()
        if auto_addr and ping_daemon(auto_addr):
            return ResolvedAddressMode("connect", auto_addr)
        raise RuntimeError(
            "address=auto but no running daemon found in env/current_session"
        )

    # host:port or other literal address; attempt reachability
    # We avoid eager normalization here; rely on caller to validate schemas like TLS if needed.
    if ":" in s:
        if ping_daemon(s):
            return ResolvedAddressMode("connect", s)
        raise RuntimeError(f"No daemon reachable at {s}")

    # Fallback: treat any other literal as host:port missing ':'
    raise RuntimeError(f"Invalid address literal: {address}")


__all__ = ["ResolvedAddressMode", "resolve_address_mode", "ping_daemon"]
