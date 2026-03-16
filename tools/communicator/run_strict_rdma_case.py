#!/usr/bin/env python3
#  Copyright (c) 2026, TensorCast Team.

"""Run a strict two-node communicator RDMA validation case."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def run_command(command: list[str], log_path: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        command,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    log_path.write_text(
        json.dumps(
            {
                "command": command,
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return proc


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def collect_host_facts(node_dir: Path) -> None:
    commands = {
        "hostname.txt": ["bash", "-lc", "hostname"],
        "hostname_i.txt": ["bash", "-lc", "hostname -i"],
        "env.txt": ["bash", "-lc", "env | sort"],
        "memlock.txt": ["bash", "-lc", "ulimit -l"],
        "nvidia_smi_topo.txt": ["bash", "-lc", "nvidia-smi topo -m"],
        "rdma_link.txt": ["bash", "-lc", "command -v rdma >/dev/null 2>&1 && rdma link || true"],
        "ibv_devinfo.txt": ["bash", "-lc", "command -v ibv_devinfo >/dev/null 2>&1 && ibv_devinfo || true"],
        "modules.txt": ["bash", "-lc", "cat /proc/modules | egrep 'nvidia_peermem|nv_peer_mem|mlx5' || true"],
    }
    node_dir.mkdir(parents=True, exist_ok=True)
    for name, command in commands.items():
        run_command(command, node_dir / name)


def current_host_ip() -> str:
    proc = subprocess.run(
        ["bash", "-lc", "hostname -i | awk '{print $1}'"],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def wait_for_file(path: Path, timeout_sec: int) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if path.exists():
            return True
        time.sleep(1)
    return False


def wait_for_ready_line(path: Path, timeout_sec: int) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if path.exists():
            content = path.read_text(encoding="utf-8", errors="ignore")
            if "READY role=target" in content:
                return True
        time.sleep(1)
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--listen-port", type=int, default=6606)
    parser.add_argument("--memory", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--target-nic", required=True)
    parser.add_argument("--initiator-nic", required=True)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--warmup-iterations", type=int, default=0)
    parser.add_argument("--direct-rdma", action="store_true")
    parser.add_argument("--strict-direct-rdma", action="store_true")
    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    node_rank = int(os.getenv("NODE_RANK", "-1"))
    master_addr = os.getenv("MASTER_ADDR", "")
    host = socket.gethostname()
    host_ip = current_host_ip()

    if node_rank not in (0, 1):
        print(f"expected NODE_RANK to be 0 or 1, got {node_rank}", file=sys.stderr)
        return 2
    if not master_addr:
        print("MASTER_ADDR is empty", file=sys.stderr)
        return 2
    if not host_ip:
        print("hostname -i did not return a usable host IP", file=sys.stderr)
        return 2

    repo_root = Path("/data/workspace/tensorcast-280")
    bench_binary = repo_root / "bazel-bin/core/communicator/communicator_bench_binary"
    if not bench_binary.exists():
        print(f"bench binary does not exist: {bench_binary}", file=sys.stderr)
        return 2

    node_dir = case_dir / "nodes" / f"rank-{node_rank}-{host}"
    collect_host_facts(node_dir)

    role = "target" if node_rank == 0 else "initiator"
    nic = args.target_nic if node_rank == 0 else args.initiator_nic
    target_ready = case_dir / "target_ready.json"
    client_done = case_dir / "client_done.json"

    inspect_cmd = [
        str(bench_binary),
        "--role",
        "inspect",
        "--memory",
        args.memory,
        "--bytes",
        str(args.bytes),
        "--gpu-id",
        str(args.gpu_id),
        "--rdma",
        "--rdma-nic",
        nic,
        "--strict-nic",
    ]
    if args.direct_rdma or args.strict_direct_rdma:
        inspect_cmd.append("--probe-gpu-mr")
    inspect_proc = run_command(inspect_cmd, node_dir / "inspect.json")
    if inspect_proc.returncode != 0:
        return inspect_proc.returncode

    if role == "target":
        target_cmd = [
            str(bench_binary),
            "--role",
            "target",
            "--listen-ip",
            "0.0.0.0",
            "--listen-port",
            str(args.listen_port),
            "--tensor-key",
            "strict-rdma-tensor",
            "--memory",
            args.memory,
            "--bytes",
            str(args.bytes),
            "--gpu-id",
            str(args.gpu_id),
            "--rdma",
            "--rdma-nic",
            args.target_nic,
            "--strict-nic",
        ]
        if args.direct_rdma:
            target_cmd.append("--direct-rdma")
        if args.strict_direct_rdma:
            target_cmd.append("--strict-direct-rdma")

        target_log = node_dir / "target.log"
        with target_log.open("w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(target_cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)
        if not wait_for_ready_line(target_log, timeout_sec=120):
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            print("target did not print READY before timeout", file=sys.stderr)
            return 4
        write_json(
            target_ready,
            {
                "host": host,
                "listen_ip": host_ip,
                "listen_port": args.listen_port,
                "target_nic": args.target_nic,
                "memory": args.memory,
                "gpu_id": args.gpu_id,
            },
        )
        if not wait_for_file(client_done, timeout_sec=300):
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            print("client_done.json did not appear before timeout", file=sys.stderr)
            return 4
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        result = json.loads(client_done.read_text(encoding="utf-8"))
        return int(result.get("returncode", 1))

    if not wait_for_file(target_ready, timeout_sec=300):
        print("target_ready.json did not appear before timeout", file=sys.stderr)
        return 4

    target_info = json.loads(target_ready.read_text(encoding="utf-8"))
    initiator_cmd = [
        str(bench_binary),
        "--role",
        "initiator",
        "--listen-ip",
        "0.0.0.0",
        "--listen-port",
        str(args.listen_port + 1),
        "--peer-ip",
        str(target_info["listen_ip"]),
        "--peer-port",
        str(target_info["listen_port"]),
        "--tensor-key",
        "strict-rdma-tensor",
        "--memory",
        args.memory,
        "--bytes",
        str(args.bytes),
        "--gpu-id",
        str(args.gpu_id),
        "--rdma",
        "--rdma-nic",
        args.initiator_nic,
        "--strict-nic",
        "--expected-remote-nic",
        str(target_info["target_nic"]),
        "--iterations",
        str(args.iterations),
        "--warmup-iterations",
        str(args.warmup_iterations),
    ]
    if args.direct_rdma:
        initiator_cmd.append("--direct-rdma")
    if args.strict_direct_rdma:
        initiator_cmd.append("--strict-direct-rdma")

    result = run_command(initiator_cmd, node_dir / "initiator.json")
    write_json(
        client_done,
        {
            "host": host,
            "returncode": result.returncode,
            "initiator_nic": args.initiator_nic,
            "target_nic": target_info["target_nic"],
        },
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
