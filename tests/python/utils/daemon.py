#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import BinaryIO, Literal, TextIO

import grpc
import yaml

from tensorcast.cli_utils.proc import build_daemon_process_env
from tensorcast.proto.daemon.v2 import store_daemon_pb2 as _pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2_grpc as _pb2_grpc


def _resolve_daemon_binary(repo_root: Path) -> Path:
    override = os.environ.get("TENSORCAST_DAEMON_BIN")
    if override:
        p = Path(override)
        if p.exists() and os.access(p, os.X_OK):
            return p
    bazel_bin = repo_root / "bazel-bin" / "daemon" / "tensorcast_daemon"
    if bazel_bin.exists() and os.access(bazel_bin, os.X_OK):
        return bazel_bin
    wheel_bin = repo_root / "tensorcast" / "bin" / "tensorcast_daemon"
    if wheel_bin.exists() and os.access(wheel_bin, os.X_OK):
        return wheel_bin
    raise FileNotFoundError("tensorcast_daemon binary not found")


def _wait_ready(
    listen_addr: str, proc: subprocess.Popen, timeout_s: float = 10.0
) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            chan = grpc.insecure_channel(listen_addr)
            stub = _pb2_grpc.StoreDaemonServiceStub(chan)
            stub.GetServerConfig(_pb2.GetServerConfigRequest(), timeout=1.0)
            chan.close()
            return
        except Exception:
            time.sleep(0.2)
    try:
        out, err = proc.communicate(timeout=1)
    except Exception:
        out = b""
        err = b""
    raise RuntimeError(f"daemon failed to start: out={out.decode()} err={err.decode()}")


def start_daemon_binary(
    listen_addr: str,
    storage_path: Path,
    config_mode: Literal["yaml", "inline_json"] = "yaml",
    enable_same_process_ipc_fallback: bool = True,
    *,
    cpu_shared_memory_enabled: bool = False,
    local_handle_socket_path: str | None = None,
    stable_bytes: int | None = None,
    handle_lease_ttl: str = "10m",
    daemon_id: str | None = None,
) -> subprocess.Popen:
    """Start the C++ daemon with a unified minimal config for tests.

    - Ensures libtorch libs are on LD_LIBRARY_PATH
    - Sets debug.cuda.enable_same_process_ipc_fallback according to flag
    - Waits for readiness by issuing GetServerConfig
    """
    repo_root = Path(__file__).resolve().parents[3]
    bin_path = _resolve_daemon_binary(repo_root)
    host, port_s = listen_addr.split(":", 1)
    port = int(port_s)
    storage_path.mkdir(parents=True, exist_ok=True)
    env = build_daemon_process_env(os.environ)
    fake_cuda = os.environ.get("TENSORCAST_CUDA_BACKEND") == "fake"
    if not daemon_id:
        daemon_id = f"test-daemon-{os.getpid()}-{port}-{int(time.time() * 1000)}"

    engine_pool_bytes = 268435456 if fake_cuda else 67108864
    comm_gpu_pool_bytes = 268435456 if fake_cuda else 67108864
    cfg: dict[str, object] = {
        "server": {
            "listen": {"host": host, "port": port},
            "p2p_listen": {"host": host, "port": 9090},
            "storage_path": str(storage_path),
            "num_threads": 2,
            "grpc": {"tcp_nodelay": True, "so_reuseport": False},
        },
        "daemon_id": daemon_id,
        "engine": {
            "artifact_chunk_bytes": 8388608,
            "streaming_buffer_chunks": 4,
        },
        "pinned_memory": {
            "allocation_timeout": "30s",
            "classes": [
                {
                    "name": "engine",
                    "slice_bytes": 8388608,
                    "pool_bytes": engine_pool_bytes,
                },
                {
                    "name": "comm_gpu",
                    "slice_bytes": 16777216,
                    "pool_bytes": comm_gpu_pool_bytes,
                    "rdma_preregister": False,
                },
                {
                    "name": "comm_cpu",
                    "slice_bytes": 4194304,
                    "pool_bytes": 8388608,
                    "rdma_preregister": False,
                },
            ],
        },
        "communicator": {
            "enable_rdma": False,
            "stager": {"buffers_per_flow": 1},
            "transport": {"tcp_conn_count": 2},
        },
        "observability": {
            "otel": {"enabled": False},
            "logging": {"level": "INFO"},
            "tracing": {"chrome_trace_dir": ""},
        },
        "debug": {
            "cuda": {
                "enable_same_process_ipc_fallback": bool(
                    enable_same_process_ipc_fallback
                )
            }
        },
    }
    if cpu_shared_memory_enabled:
        if not local_handle_socket_path:
            raise ValueError(
                "local_handle_socket_path is required when cpu_shared_memory_enabled is True"
            )
        if stable_bytes is None:
            stable_bytes = 64 * 1024 * 1024
        cfg["engine"]["cpu_shared_memory"] = {"enabled": True}
        cfg["engine"]["memory_tiers"] = {
            "enable_preemptible": False,
            "stable_bytes": int(stable_bytes),
        }
        cfg["lifecycle"] = {
            "handle_leases": {
                "local_handle_socket_path": str(local_handle_socket_path),
                "ttl": str(handle_lease_ttl),
            }
        }

    cfg_suffix = ".yaml" if config_mode == "yaml" else ".json"
    with tempfile.NamedTemporaryFile(
        prefix="tc_daemon_cfg_", suffix=cfg_suffix, mode="w", delete=False
    ) as f:
        if config_mode == "yaml":
            yaml.safe_dump(cfg, f, sort_keys=False)
        else:
            json.dump(cfg, f)
        cfg_path = Path(f.name)
    args = [str(bin_path), f"--config={cfg_path}"]

    proc = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
    )

    # Stream daemon logs to current process stdout/stderr for debugging
    def _pump(stream: BinaryIO | None, sink: TextIO) -> None:
        if stream is None:
            return
        try:
            for chunk in iter(stream.readline, b""):
                try:
                    sink.write(chunk.decode("utf-8", errors="replace"))
                    sink.flush()
                except Exception:
                    # Best-effort streaming; ignore sink write errors
                    pass
        finally:
            with contextlib.suppress(Exception):
                stream.close()

    threading.Thread(target=_pump, args=(proc.stdout, sys.stdout), daemon=True).start()
    threading.Thread(target=_pump, args=(proc.stderr, sys.stderr), daemon=True).start()
    _wait_ready(listen_addr, proc, timeout_s=10.0)
    return proc
