#  Copyright (c) 2025-2026, TensorCast Team.

"""
Integration test for the C++ StoreDaemon lifecycle with Global Store.

This test launches an in-process Python Global Store gRPC server and starts the
compiled C++ daemon binary pointing at it. It verifies:
 - Worker registers and appears in ListActiveWorkers with expected fields

The test is skipped if the C++ daemon binary is not available.
"""

from __future__ import annotations

import contextlib
import os
import socket
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc
import pytest
import yaml

from tensorcast.cli_utils.proc import (
    build_daemon_process_env,
    ensure_cpp_daemon_binary,
)
from tensorcast.global_store.composite_stub import GlobalStoreCompositeStub
from tensorcast.global_store.config.settings import (
    GlobalStoreConfig,
)
from tensorcast.global_store.config.settings import (
    set_config as set_gs_config,
)
from tensorcast.global_store.grpc_service import (
    GlobalStoreServicer,
    register_global_store_servicers,
)
from tensorcast.proto.global_store.v1 import global_store_pb2

pytestmark = pytest.mark.requires_cuda_or_fake


def _get_free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def gs_server():
    # Initialize Global Store config (file-less, in-memory DB, defaults)
    set_gs_config(GlobalStoreConfig())
    servicer = GlobalStoreServicer()
    server = grpc.server(ThreadPoolExecutor(max_workers=8))
    register_global_store_servicers(server, servicer)
    port = server.add_insecure_port("127.0.0.1:0")
    if port <= 0:
        raise RuntimeError("failed to bind Global Store server port")
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
    if os.environ.get("TENSORCAST_RUN_CPP_DAEMON_IT", "1") != "1":
        pytest.skip("Set TENSORCAST_RUN_CPP_DAEMON_IT=1 to run daemon integration test")

    bin_path: Path = ensure_cpp_daemon_binary()

    _, gs_port = gs_server

    # Allocate ports and temp storage dir
    listen_port = _get_free_port()
    storage_dir = Path(tempfile.mkdtemp(prefix="tc_daemon_it_"))
    daemon_id = f"daemon_it_{listen_port}"

    # Build minimal unified DaemonConfig (YAML)
    log_path = storage_dir / "daemon.log"
    cfg = {
        "server": {
            "listen": {"host": "localhost", "port": listen_port},
            "p2p_listen": {"host": "localhost", "port": 65090},
            "storage_path": str(storage_dir),
            "num_threads": 2,
            "grpc": {"tcp_nodelay": True, "so_reuseport": False},
        },
        "daemon_id": daemon_id,
        "engine": {
            "artifact_chunk_bytes": 1 * 1024 * 1024,
            "streaming_buffer_chunks": 4,
        },
        "pinned_memory": {
            "allocation_timeout": "30s",
            "classes": [
                {
                    "name": "engine",
                    "slice_bytes": 1 * 1024 * 1024,
                    "pool_bytes": 64 * 1024 * 1024,
                },
                {
                    "name": "comm_gpu",
                    "slice_bytes": 1 * 1024 * 1024,
                    "pool_bytes": 4 * 1024 * 1024,
                },
                {
                    "name": "comm_cpu",
                    "slice_bytes": 1 * 1024 * 1024,
                    "pool_bytes": 1 * 1024 * 1024,
                },
            ],
        },
        "high_availability": {
            "enabled": True,
            "global_store_endpoints": [{"host": "localhost", "port": gs_port}],
        },
        "communicator": {
            "enable_rdma": False,
            "stager": {"buffers_per_flow": 1},
            "transport": {"tcp_conn_count": 2},
        },
        "observability": {
            "otel": {"enabled": False},
            "logging": {
                "level": "INFO",
                "otel_context_enabled": False,
                "file": str(log_path),
            },
            "tracing": {"chrome_trace_dir": ""},
        },
        "debug": {"cuda": {"enable_same_process_ipc_fallback": False}},
    }

    with tempfile.NamedTemporaryFile(
        prefix="tc_daemon_cfg_", suffix=".yaml", mode="w", delete=False
    ) as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
        cfg_path = Path(f.name)

    # Launch daemon with config-only init, capture output for debugging
    proc_log = storage_dir / "daemon_proc.log"
    with open(proc_log, "a") as log_fd:
        env = build_daemon_process_env()
        proc = subprocess.Popen(
            [str(bin_path), f"--config={cfg_path}"],
            stdout=log_fd,
            stderr=log_fd,
            env=env,
        )
        try:
            # Poll ListActiveWorkers until daemon registers
            channel = grpc.insecure_channel(f"127.0.0.1:{gs_port}")
            stub = GlobalStoreCompositeStub(channel)

            deadline = time.time() + 15.0
            found = None
            while time.time() < deadline and not found:
                resp = stub.ListActiveWorkers(
                    global_store_pb2.ListActiveWorkersRequest(include_unavailable=True)
                )
                for w in resp.workers:
                    if w.grpc_port == listen_port:
                        found = w
                        break
                if not found:
                    time.sleep(0.25)
            if found is None:
                tail = ""
                if log_path.exists():
                    tail = log_path.read_text()[-2000:]
                proc_tail = ""
                with contextlib.suppress(Exception):
                    proc_tail = proc_log.read_text()[-2000:]
                pytest.fail(
                    "daemon did not register in time;\n"
                    + "daemon log tail:\n"
                    + tail
                    + "\n"
                    + "proc output tail:\n"
                    + proc_tail
                )
            assert found is not None
            assert found.mem_pool_total_size == 64 * 1024 * 1024
            assert found.accepting_new_requests is True
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
