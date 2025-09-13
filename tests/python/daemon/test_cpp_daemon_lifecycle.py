#  Copyright (c) 2025, TensorCast Team.

# Copyright (c) 2025, TensorCast Team.

"""
Integration test for the C++ StoreDaemon lifecycle with Global Store.

This test launches an in-process Python Global Store gRPC server and starts the
compiled C++ daemon binary pointing at it. It verifies:
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

from tensorcast.cli_utils.service_manager import ServiceError
from tensorcast.cli_utils.proc import ensure_cpp_daemon_binary
from tensorcast.global_store.grpc_service import GlobalStoreServicer
from tensorcast.global_store.config.settings import (
    GlobalStoreConfig,
    set_config as set_gs_config,
)
from tensorcast.proto.global_store.v1 import (
    global_store_pb2,
    global_store_pb2_grpc,
)


def _get_free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def gs_server():
    from concurrent.futures import ThreadPoolExecutor

    # Initialize Global Store config (file-less, in-memory DB, defaults)
    set_gs_config(GlobalStoreConfig())
    servicer = GlobalStoreServicer()
    server = grpc.server(ThreadPoolExecutor(max_workers=8))
    global_store_pb2_grpc.add_GlobalStoreServiceServicer_to_server(servicer, server)
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
        bin_path: Path = ensure_cpp_daemon_binary()
    except ServiceError:
        pytest.skip("C++ daemon binary not available; skipping integration test")

    import tempfile
    import yaml

    _, gs_port = gs_server

    # Allocate ports and temp storage dir
    listen_port = _get_free_port()
    storage_dir = Path(tempfile.mkdtemp(prefix="tc_daemon_it_"))

    # Build minimal unified DaemonConfig (YAML)
    log_path = storage_dir / "daemon.log"
    cfg = {
        "server": {
            "listen": {"host": "localhost", "port": listen_port},
            "p2p_listen": {"host": "localhost", "port": 65090},
            "storage_path": str(storage_dir),
            "num_threads": 2,
            "grpc": {"max_message_size_mb": 16, "tcp_nodelay": True, "so_reuseport": False},
        },
        "engine": {
            "mem_pool_size_bytes": 64 * 1024 * 1024,
            "chunk_bytes": 1 * 1024 * 1024,
            "cpu_chunk_size_bytes": 1 * 1024 * 1024,
            "streaming_buffer_max_concurrent_sessions": 1,
        },
        "high_availability": {
            "enabled": True,
            "global_store_endpoints": [{"host": "localhost", "port": gs_port}],
        },
        "communicator": {"enable_rdma": False},
        "observability": {
            "otel": {"enabled": False},
            "logging": {"level": "INFO", "otel_context_enabled": False, "file": str(log_path)},
            "tracing": {"chrome_trace_dir": ""},
        },
        "debug": {"cuda": {"enable_same_process_ipc_fallback": False}},
    }

    with tempfile.NamedTemporaryFile(prefix="tc_daemon_cfg_", suffix=".yaml", mode="w", delete=False) as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
        cfg_path = Path(f.name)

    # Launch daemon with config-only init, capture output for debugging
    proc_log = storage_dir / "daemon_proc.log"
    log_fd = open(proc_log, "a")
    proc = subprocess.Popen([str(bin_path), f"--config={cfg_path}"], stdout=log_fd, stderr=log_fd)
    try:
        # Poll ListActiveWorkers until daemon registers
        channel = grpc.insecure_channel(f"127.0.0.1:{gs_port}")
        stub = global_store_pb2_grpc.GlobalStoreServiceStub(channel)

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
        if found is None:
            try:
                tail = ""
                if log_path.exists():
                    tail = log_path.read_text()[-2000:]
                proc_tail = ""
                try:
                    proc_tail = proc_log.read_text()[-2000:]
                except Exception:
                    pass
                pytest.fail(
                    "daemon did not register in time;\n"
                    + "daemon log tail:\n" + tail + "\n"
                    + "proc output tail:\n" + proc_tail
                )
            finally:
                pass
        assert found is not None
        assert found.mem_pool_total_size == 64 * 1024 * 1024
        assert found.accepting_new_requests is True
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
