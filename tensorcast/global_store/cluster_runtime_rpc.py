#  Copyright (c) 2026, TensorCast Team.

"""Shared client helpers for ClusterRuntimeService RPC calls."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from typing import Any, TypeVar

T = TypeVar("T")


def call_cluster_runtime_rpc(
    *,
    gs_addr: str,
    ready_timeout_sec: float,
    rpc_name: str,
    grpc_module: Any,
    stub_factory: Callable[[Any], Any],
    rpc_call: Callable[[Any], T],
) -> tuple[T | None, str | None]:
    """Run one ClusterRuntimeService RPC with channel-ready timeout handling."""

    address = str(gs_addr).strip()
    if not address:
        return None, "gs_addr is empty"

    ready_timeout = max(0.1, float(ready_timeout_sec))
    channel = grpc_module.insecure_channel(address)
    try:
        grpc_module.channel_ready_future(channel).result(timeout=ready_timeout)
        stub = stub_factory(channel)
        return rpc_call(stub), None
    except grpc_module.FutureTimeoutError:
        return (
            None,
            f"{rpc_name} channel ready timeout after {ready_timeout:.1f}s",
        )
    except grpc_module.RpcError as exc:
        detail = exc.details() if hasattr(exc, "details") else str(exc)
        return None, f"{rpc_name} RPC failed: {detail}"
    finally:
        with suppress(Exception):
            channel.close()
