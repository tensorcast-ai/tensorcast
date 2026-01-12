#  Copyright (c) 2025-2026, TensorCast Team.

"""Network helpers: port selection, readiness probes, and address normalization."""

from __future__ import annotations

import contextlib
import socket
import subprocess
import time
from typing import Any, Callable, Iterable

import grpc

from tensorcast.proto.daemon.v2 import store_daemon_pb2, store_daemon_pb2_grpc


def resolve_connect_host(listen_host: str | None) -> str:
    if not listen_host:
        return "127.0.0.1"
    s = str(listen_host).strip().lower()
    if s in {"0.0.0.0", "::", "[::]", "*"}:
        return "127.0.0.1"
    return listen_host


def is_unspecified_host(host: str | None) -> bool:
    if not host:
        return True
    s = str(host).strip().lower()
    return s in {"0.0.0.0", "::", "[::]", "*"}


def pick_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(s.getsockname()[1])


def wait_daemon_ready(
    host: str,
    port: int,
    timeout: float | None = 20.0,
    *,
    proc: subprocess.Popen[Any] | None = None,
    progress: Callable[[float, Exception | None], None] | None = None,
    progress_interval: float = 5.0,
    extra_hosts: Iterable[str] | None = None,
) -> str | None:
    deadline = None if timeout is None else time.time() + timeout
    start_ts = time.time()
    last_report = start_ts
    last_err: Exception | None = None
    candidates: list[str] = []
    seen: set[str] = set()
    for candidate in (host, *(extra_hosts or [])):
        if not candidate:
            continue
        if candidate in seen:
            continue
        candidates.append(candidate)
        seen.add(candidate)
    if proc is not None and proc.poll() is not None:
        return None
    stubs: list[
        tuple[str, grpc.Channel, store_daemon_pb2_grpc.StoreDaemonServiceStub]
    ] = []
    for candidate in candidates:
        addr = f"{candidate}:{port}"
        channel = grpc.insecure_channel(addr)
        stubs.append(
            (candidate, channel, store_daemon_pb2_grpc.StoreDaemonServiceStub(channel))
        )
    try:
        while deadline is None or time.time() < deadline:
            if proc is not None and proc.poll() is not None:
                return None
            for candidate, _channel, stub in stubs:
                try:
                    stub.GetServerConfig(
                        store_daemon_pb2.GetServerConfigRequest(), timeout=0.8
                    )
                    return candidate
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    if proc is not None and proc.poll() is not None:
                        return None
            now = time.time()
            if progress is not None and now - last_report >= max(
                progress_interval, 0.5
            ):
                progress(now - start_ts, last_err)
                last_report = now
            time.sleep(0.2)
    finally:
        for _candidate, channel, _stub in stubs:
            with contextlib.suppress(Exception):
                channel.close()
    # For debugging purposes; caller can log if desired
    _ = last_err
    return None
