#!/usr/bin/env python3
#  Copyright (c) 2025, TensorCast Team.

# Minimal end-to-end observability smoke test.
# - Starts OpenTelemetry Collector (if Docker available)
# - Starts Global Store (Python, with OTel)
# - Builds and Starts StoreDaemon (C++, with fake CUDA to avoid GPU requirements)
# - Issues a small client RPC to the Daemon to generate spans

from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _wait_for_port(host: str, port: int, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            try:
                s.connect((host, port))
                return True
            except OSError:
                time.sleep(0.2)
    return False


def _start_proc(cmd: list[str], env: dict[str, str] | None, log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "wb")
    return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)


def _run(cmd: list[str], env: dict[str, str] | None = None, check: bool = True) -> int:
    print("$", " ".join(cmd))
    return subprocess.run(cmd, env=env, check=check).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="TensorCast OTel E2E smoke test")
    parser.add_argument("--collector", action="store_true", help="Start OTel Collector via Docker")
    parser.add_argument("--otlp", default="http://127.0.0.1:4317", help="OTLP endpoint")
    parser.add_argument("--gs-port", type=int, default=50051, help="Global Store port")
    parser.add_argument("--gs-metrics", type=int, default=8001, help="Global Store metrics port")
    parser.add_argument("--daemon-addr", default="127.0.0.1:8073", help="StoreDaemon listen address")
    parser.add_argument("--workspace", default=str(Path.cwd()), help="Repository root")
    args = parser.parse_args()

    repo = Path(args.workspace)
    log_dir = repo / "build" / "e2e-smoke-logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    procs: list[tuple[str, subprocess.Popen]] = []
    docker_ok = False

    try:
        # 1) Collector (optional)
        if args.collector:
            if shutil.which("docker") is None:
                print("[warn] docker not found; skipping collector")
            else:
                cfg = repo / "tools" / "otel" / "collector-dev.yaml"
                if not cfg.exists():
                    print(f"[warn] collector config not found: {cfg}; skipping")
                else:
                    docker_cmd = [
                        "docker",
                        "run",
                        "--rm",
                        "--network",
                        "host",
                        "-v",
                        f"{cfg}:/etc/otelcol/config.yaml:ro",
                        "otel/opentelemetry-collector:latest",
                    ]
                    env = dict(os.environ)
                    p = _start_proc(docker_cmd, env, log_dir / "collector.log")
                    procs.append(("collector", p))
                    docker_ok = True
                    print("[info] collector starting (docker)")
                    time.sleep(2)

        # 2) Build daemon with fake CUDA to avoid GPU dependency
        print("[info] building daemon (fake CUDA)")
        _run(["bazel", "build", "--define", "use_fake_cuda=true", "//daemon:tensorcast_daemon"])  # may take time
        daemon_bin = repo / "bazel-bin" / "daemon" / "tensorcast_daemon"
        if not daemon_bin.exists():
            print(f"[error] daemon binary not found: {daemon_bin}")
            return 2

        # 3) Start Global Store
        env_gs = dict(os.environ)
        env_gs["OTEL_EXPORTER_OTLP_ENDPOINT"] = args.otlp
        env_gs["OTEL_TRACES_SAMPLER"] = "parentbased_traceidratio"
        env_gs["OTEL_TRACES_SAMPLER_ARG"] = "1.0"
        gs_cmd = [
            "uv",
            "run",
            "-m",
            "tensorcast.global_store",
            "--port",
            str(args.gs_port),
            "--metrics-port",
            str(args.gs_metrics),
        ]
        p_gs = _start_proc(gs_cmd, env_gs, log_dir / "global_store.log")
        procs.append(("global_store", p_gs))
        print("[info] global store starting")
        if not _wait_for_port("127.0.0.1", args.gs_port, timeout=15.0):
            print("[error] global store did not open port in time")
            return 3

        # 4) Start Daemon (points to GS)
        env_daemon = dict(os.environ)
        env_daemon["OTEL_EXPORTER_OTLP_ENDPOINT"] = args.otlp
        daemon_cmd = [
            str(daemon_bin),
            "--listen_addr",
            args.daemon_addr,
            "--global_store_addr",
            f"127.0.0.1:{args.gs_port}",
            "--p2p_port",
            "9090",
        ]
        p_daemon = _start_proc(daemon_cmd, env_daemon, log_dir / "daemon.log")
        procs.append(("daemon", p_daemon))
        print("[info] daemon starting")
        host, port_str = args.daemon_addr.split(":", 1)
        if not _wait_for_port(host, int(port_str), timeout=10.0):
            print("[error] daemon did not open port in time")
            return 4

        # 5) Client action: query server config (and rely on daemon → GS worker register)
        print("[info] issuing client RPCs")
        # Use in-repo imports; ensure client auto-init for OTel
        os.environ.setdefault("TC_OTEL_CLIENT_AUTO_INIT", "1")
        from tensorcast.daemon_ctl import get_daemon_client

        ctl = get_daemon_client(args.daemon_addr)
        resp = ctl.get_server_config()
        print("GetServerConfig:", resp)

        # A small grace period to flush spans
        print("[info] waiting to allow spans to export (2s)")
        time.sleep(2.0)

        print("[OK] smoke test completed. Logs in:", str(log_dir))
        if docker_ok:
            print("- Collector logs:", log_dir / "collector.log")
        print("- Global Store logs:", log_dir / "global_store.log")
        print("- Daemon logs:", log_dir / "daemon.log")
        return 0

    finally:
        # Teardown in reverse order
        for name, p in reversed(procs):
            try:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
