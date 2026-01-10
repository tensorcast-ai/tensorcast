#  Copyright (c) 2025-2026, TensorCast Team.

"""Health probes for Global Store and daemon services."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import Any

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

from tensorcast.cli_utils.network import resolve_connect_host
from tensorcast.proto.daemon.v2 import store_daemon_pb2, store_daemon_pb2_grpc
from tensorcast.proto.global_store.v1 import (
    global_store_pb2,
    global_store_pb2_grpc,
)


@dataclass(frozen=True)
class GlobalStoreHealth:
    address: str
    listen_host: str | None
    listen_port: int | None
    metrics_port: int | None
    cluster_token: str | None
    version: str | None
    db_file: str | None


def ping_daemon(address: str, timeout: float = 1.0) -> bool:
    """Ping the daemon via GetServerConfig; returns True on success."""

    channel = grpc.insecure_channel(address)
    try:
        stub = store_daemon_pb2_grpc.StoreDaemonServiceStub(channel)
        stub.GetServerConfig(store_daemon_pb2.GetServerConfigRequest(), timeout=timeout)
        return True
    except Exception:
        return False


def get_daemon_config(address: str, timeout: float = 1.0):
    """Fetch the daemon's server config; returns response or None on failure."""

    channel = grpc.insecure_channel(address)
    try:
        stub = store_daemon_pb2_grpc.StoreDaemonServiceStub(channel)
        return stub.GetServerConfig(
            store_daemon_pb2.GetServerConfigRequest(), timeout=timeout
        )
    except Exception:
        return None


def wait_for_daemon(
    address: str, *, timeout: float | None = 10.0, interval: float = 0.2
) -> bool:
    """Wait for daemon readiness using GetServerConfig."""

    wait_interval = max(0.05, interval)
    deadline = None if timeout is None else time.time() + max(timeout, 0.0)
    while deadline is None or time.time() < deadline:
        rpc_timeout = (
            min(wait_interval, timeout) if timeout is not None else wait_interval
        )
        if ping_daemon(address, timeout=rpc_timeout):
            return True
        time.sleep(wait_interval)
    return False


def _split_host_port(address: str) -> tuple[str | None, int | None]:
    try:
        host, port_s = address.rsplit(":", 1)
        return host, int(port_s)
    except Exception:
        return None, None


def ping_global_store(
    address: str, *, timeout: float = 1.0
) -> GlobalStoreHealth | None:
    """Ping Global Store via health RPC (with grpc-health fallback)."""

    host_hint, port_hint = _split_host_port(address)
    connect_host = resolve_connect_host(host_hint)
    target = f"{connect_host}:{port_hint}" if port_hint else address
    channel = grpc.insecure_channel(target)
    try:
        stub = global_store_pb2_grpc.GlobalStoreServiceStub(channel)
        resp = stub.HealthCheck(global_store_pb2.HealthCheckRequest(), timeout=timeout)
        if resp.status == global_store_pb2.Status.STATUS_OK:
            listen_port = int(resp.listen_port) if resp.listen_port else port_hint
            metrics_port = int(resp.metrics_port) if resp.metrics_port else None
            listen_host = resp.listen_host or host_hint or connect_host
            listen_addr = resp.listen_address or None
            if listen_addr and ":" in listen_addr:
                try:
                    lh, lp = listen_addr.rsplit(":", 1)
                    listen_host = lh or listen_host
                    listen_port = listen_port or int(lp)
                except Exception:
                    pass
            return GlobalStoreHealth(
                address=target,
                listen_host=listen_host,
                listen_port=listen_port,
                metrics_port=metrics_port,
                cluster_token=resp.cluster_token or None,
                version=resp.version or None,
                db_file=resp.db_file or None,
            )
    except Exception:
        pass

    # Fallback to standard grpc-health
    try:
        health_stub = health_pb2_grpc.HealthStub(channel)
        resp = health_stub.Check(
            health_pb2.HealthCheckRequest(
                service="tensorcast.global_store.v1.GlobalStoreService"
            ),
            timeout=timeout,
        )
        if resp.status == health_pb2.HealthCheckResponse.SERVING:
            return GlobalStoreHealth(
                address=target,
                listen_host=connect_host,
                listen_port=port_hint,
                metrics_port=None,
                cluster_token=None,
                version=None,
                db_file=None,
            )
    except Exception:
        return None
    return None


def wait_for_global_store(
    address: str,
    *,
    timeout: float | None = 10.0,
    interval: float = 0.2,
    proc: subprocess.Popen[Any] | None = None,
) -> GlobalStoreHealth | None:
    """Wait for Global Store to become healthy."""

    wait_interval = max(0.05, interval)
    deadline = None if timeout is None else time.time() + max(timeout, 0.0)
    while deadline is None or time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return None
        rpc_timeout = (
            min(wait_interval, timeout) if timeout is not None else wait_interval
        )
        health = ping_global_store(address, timeout=rpc_timeout)
        if health is not None:
            return health
        time.sleep(wait_interval)
    return None


__all__ = [
    "GlobalStoreHealth",
    "ping_global_store",
    "wait_for_global_store",
    "ping_daemon",
    "wait_for_daemon",
    "get_daemon_config",
]
