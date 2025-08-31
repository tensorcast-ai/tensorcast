#  Copyright (c) 2025, TensorCast Team.

# Copyright (c) 2025, TensorCast Team.

"""
Integration test for the C++ StoreDaemon lifecycle with Global Store.

This test launches an in-process Python Global Store gRPC server and starts the
compiled C++ daemon binary pointing at it. It verifies:
 - /health endpoint responds 200
 - Worker registers and appears in ListActiveWorkers with expected fields

The test is skipped if the C++ daemon binary is not available.
"""

from __future__ import annotations

import contextlib
import socket
import subprocess
import time
from pathlib import Path

import grpc
import pytest

from tensorcast.cli_utils.service_manager import (
    ServiceError,
    _ensure_cpp_daemon_binary,
)
from tensorcast.global_store.grpc_service import GlobalStoreServicer
from tensorcast.proto import global_store_pb2, global_store_pb2_grpc


def _get_free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def gs_server():
    from concurrent.futures import ThreadPoolExecutor

    servicer = GlobalStoreServicer()
    server = grpc.server(ThreadPoolExecutor(max_workers=8))
    global_store_pb2_grpc.add_GlobalStoreServicer_to_server(servicer, server)
    port = _get_free_port()
    server.add_insecure_port(f"127.0.0.1:{port}")
    server.start()
    try:
        yield (server, port)
    finally:
        server.stop(grace=None)


def _wait_http_ok(host: str, port: int, path: str, timeout_s: float = 15.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0) as s:
                req = f"GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n".encode()
                s.sendall(req)
                data = s.recv(256)
                if b"200 OK" in data:
                    return True
        except OSError:
            time.sleep(0.2)
    return False


@pytest.mark.integration
def test_cpp_daemon_registers_with_global_store(gs_server):
    import os
    if os.environ.get("TENSORCAST_RUN_CPP_DAEMON_IT") != "1":
        pytest.skip("Set TENSORCAST_RUN_CPP_DAEMON_IT=1 to run daemon integration test")
    # Locate daemon binary or skip
    try:
        bin_path: Path = _ensure_cpp_daemon_binary()
    except ServiceError:
        pytest.skip("C++ daemon binary not available; skipping integration test")

    _, gs_port = gs_server

    # Allocate ports
    listen_port = _get_free_port()
    metrics_port = _get_free_port()

    # Launch daemon with small pool and fake comm settings
    args = [
        str(bin_path),
        f"--listen_addr=127.0.0.1:{listen_port}",
        f"--p2p_port=0",
        f"--metrics_port={metrics_port}",
        f"--mem_pool_size={64 * 1024 * 1024}",
        f"--chunk_size={1 * 1024 * 1024}",
        f"--io_threads=2",
        f"--global_store_addr=127.0.0.1:{gs_port}",
        "--enable_p2p_engine=false",
        "--enable_p2p_access=true",
        "--heartbeat_interval_ms=500",
        "--chunk_sync_interval_ms=0",
    ]

    proc = subprocess.Popen(args)
    try:
        # Wait for /health
        assert _wait_http_ok("127.0.0.1", metrics_port, "/health", timeout_s=20.0)

        # Poll ListActiveWorkers until daemon registers
        channel = grpc.insecure_channel(f"127.0.0.1:{gs_port}")
        stub = global_store_pb2_grpc.GlobalStoreStub(channel)

        deadline = time.time() + 15.0
        found = None
        while time.time() < deadline and not found:
            resp = stub.ListActiveWorkers(global_store_pb2.ListActiveWorkersRequest(include_unavailable=True))
            for w in resp.workers:
                if w.grpc_port == listen_port:
                    found = w
                    break
            if not found:
                time.sleep(0.25)
        assert found is not None, "daemon did not register in time"
        assert found.mem_pool_total_size == 64 * 1024 * 1024
        assert found.accepting_new_requests is True
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
