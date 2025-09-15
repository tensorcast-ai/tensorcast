#  Copyright (c) 2025, TensorCast Team.

import os
import time
from pathlib import Path

import grpc
import pytest
import torch

from tensorcast.api import RegisterArtifactOptions, register_artifact
from tensorcast.daemon_ctl import DaemonCtl
from tensorcast.proto.daemon.v1 import store_daemon_pb2 as _pb2
from tensorcast.proto.daemon.v1 import store_daemon_pb2_grpc as _pb2_grpc


def _start_daemon_binary(listen_addr: str, storage_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    override = os.environ.get("TENSORCAST_DAEMON_BIN")
    if override:
        bin_path = Path(override)
    else:
        bazel_bin = repo_root / "bazel-bin" / "daemon" / "tensorcast_daemon"
        if bazel_bin.exists() and os.access(bazel_bin, os.X_OK):
            bin_path = bazel_bin
        else:
            bin_path = repo_root / "tensorcast" / "bin" / "tensorcast_daemon"
    assert bin_path.exists() and os.access(bin_path, os.X_OK)
    # Ensure libtorch libs available to daemon
    torch_libdir = Path(torch.__file__).resolve().parent / "lib"
    env = os.environ.copy()
    if torch_libdir.exists():
        ld_path = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{torch_libdir}:{ld_path}" if ld_path else str(torch_libdir)
    # Minimal config
    import tempfile, yaml
    host, port_s = listen_addr.split(":", 1)
    port = int(port_s)
    storage_path.mkdir(parents=True, exist_ok=True)
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
        "observability": {"otel": {"enabled": False}, "logging": {"level": "INFO"}},
        "debug": {"cuda": {"enable_same_process_ipc_fallback": True}},
    }
    with tempfile.NamedTemporaryFile(prefix="tc_daemon_cfg_", suffix=".yaml", mode="w", delete=False) as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
        cfg_path = Path(f.name)
    proc = __import__("subprocess").Popen([str(bin_path), f"--config={cfg_path}"], stdout=__import__("subprocess").PIPE, stderr=__import__("subprocess").PIPE, env=env)
    # Wait for readiness
    deadline = time.time() + 10
    ok = False
    while time.time() < deadline:
        try:
            chan = grpc.insecure_channel(listen_addr)
            stub = _pb2_grpc.StoreDaemonServiceStub(chan)
            stub.GetServerConfig(_pb2.GetServerConfigRequest(), timeout=1.0)
            chan.close()
            ok = True
            break
        except Exception:
            time.sleep(0.2)
    if not ok:
        try:
            out, err = proc.communicate(timeout=1)
        except Exception:
            out = b""; err = b""
        raise RuntimeError(f"daemon failed to start: out={out.decode()} err={err.decode()}")
    return proc


def _skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available – skipping LIP helper test")


@pytest.mark.timeout(60)
def test_register_artifact_lease_in_place_helper(tmp_path: Path):
    _skip_if_no_cuda()
    os.environ.setdefault("TENSORCAST_ENABLE_IPC_SAME_PROCESS_FALLBACK", "1")
    listen = "127.0.0.1:50741"
    try:
        proc = _start_daemon_binary(listen, tmp_path / "models")
    except RuntimeError as e:
        pytest.fail(str(e))
    try:
        dev = torch.device("cuda", 0)
        a = torch.arange(0, 32, dtype=torch.uint8, device=dev)
        b = torch.full((64,), 0x77, dtype=torch.uint8, device=dev)
        state = {"a": a, "b": b}
        opts = RegisterArtifactOptions(plan="vram_leased", lease_in_place=True)
        res = register_artifact(state, options=opts, ttl_ms=2000, daemon_address=listen, create_post_commit_lease=True)
        desc, lease = res.descriptor, res.lease
        assert desc.artifact_id.startswith("mi2:")
        # Keepalive thread should be running; sleep to allow a keepalive tick
        time.sleep(0.5)

        assert lease is not None
        # Context revoke
        with lease:
            pass
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            pass
