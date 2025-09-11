#  Copyright (c) 2025, TensorCast Team.

"""Network helpers: port selection, readiness probes, and address normalization."""

from __future__ import annotations

import socket
import time

import grpc

from tensorcast.proto.daemon.v1 import store_daemon_pb2, store_daemon_pb2_grpc


def resolve_connect_host(listen_host: str | None) -> str:
    if not listen_host:
        return "127.0.0.1"
    s = str(listen_host).strip().lower()
    if s in {"0.0.0.0", "::", "[::]", "*"}:
        return "127.0.0.1"
    return listen_host


def pick_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(s.getsockname()[1])


def tcp_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            return s.connect_ex((host, port)) == 0
    except OSError:
        return False


def wait_daemon_ready(host: str, port: int, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    addr = f"{host}:{port}"
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            channel = grpc.insecure_channel(addr)
            stub = store_daemon_pb2_grpc.StoreDaemonServiceStub(channel)
            stub.GetServerConfig(store_daemon_pb2.GetServerConfigRequest(), timeout=0.8)
            return True
        except Exception as e:  # noqa: BLE001
            last_err = e
            if tcp_port_open(host, port, timeout=0.4):
                return True
            time.sleep(0.2)
    # For debugging purposes; caller can log if desired
    _ = last_err
    return False
