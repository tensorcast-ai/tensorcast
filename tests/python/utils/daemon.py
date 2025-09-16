#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Literal, TextIO, BinaryIO
import sys
import threading
import torch

import grpc

from tensorcast.proto.daemon.v1 import store_daemon_pb2 as _pb2
from tensorcast.proto.daemon.v1 import store_daemon_pb2_grpc as _pb2_grpc


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


def _augment_env_with_torch_libs(env: dict[str, str]) -> dict[str, str]:
    try:
        torch_libdir = Path(torch.__file__).resolve().parent / "lib"
        if torch_libdir.exists():
            ld_path = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{torch_libdir}:{ld_path}" if ld_path else str(torch_libdir)
    except Exception:
        pass
    return env


def _wait_ready(listen_addr: str, proc: subprocess.Popen, timeout_s: float = 10.0) -> None:
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
        out = b""; err = b""
    raise RuntimeError(f"daemon failed to start: out={out.decode()} err={err.decode()}")


def start_daemon_binary(
    listen_addr: str,
    storage_path: Path,
    config_mode: Literal["yaml", "inline_json"] = "yaml",
    enable_same_process_ipc_fallback: bool = True,
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
    env = _augment_env_with_torch_libs(os.environ.copy())

    if config_mode == "yaml":
        import tempfile
        import yaml
        cfg = {
            "server": {
                "listen": {"host": host, "port": port},
                "p2p_listen": {"host": host, "port": 9090},
                "storage_path": str(storage_path),
                "num_threads": 2,
                "grpc": {"tcp_nodelay": True, "so_reuseport": False},
            },
            "engine": {
                "mem_pool_size_bytes": 268435456,
                "tx_slice_bytes": 8388608,
                "artifact_chunk_bytes": 8388608,
                "streaming_buffer_max_concurrent_sessions": 1,
            },
            "communicator": {"enable_rdma": False},
            "observability": {
                "otel": {"enabled": False},
                "logging": {"level": "INFO"},
                "tracing": {"chrome_trace_dir": ""},
            },
            "debug": {"cuda": {"enable_same_process_ipc_fallback": bool(enable_same_process_ipc_fallback)}},
        }
        with tempfile.NamedTemporaryFile(prefix="tc_daemon_cfg_", suffix=".yaml", mode="w", delete=False) as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
            cfg_path = Path(f.name)
        args = [str(bin_path), f"--config={cfg_path}"]
    else:
        config_text = (
            "{"
            f"\"server\": {{\"listen\": {{\"host\": \"{host}\", \"port\": {port}}}, "
            f"\"storage_path\": \"{str(storage_path)}\", \"num_threads\": 2}}, "
            "\"engine\": {\"mem_pool_size_bytes\": 268435456, \"tx_slice_bytes\": 8388608, \"artifact_chunk_bytes\": 268435456}, "
            f"\"debug\": {{\"cuda\": {{\"enable_same_process_ipc_fallback\": {str(enable_same_process_ipc_fallback).lower()} }}}}"
            "}"
        )
        args = [str(bin_path), f"--config_text={config_text}"]

    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
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
            try:
                stream.close()
            except Exception:
                pass

    threading.Thread(target=_pump, args=(proc.stdout, sys.stdout), daemon=True).start()
    threading.Thread(target=_pump, args=(proc.stderr, sys.stderr), daemon=True).start()
    _wait_ready(listen_addr, proc, timeout_s=10.0)
    return proc


