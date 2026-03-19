#!/usr/bin/env python3
#  Copyright (c) 2026, TensorCast Team.

"""Run one two-node communicator benchmark case."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time


KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")
NUMA_WRAPPER = Path(__file__).resolve().parent / "run_with_numa_policy.py"


def run_command(
    command: list[str],
    log_path: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
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


def parse_keyvals(line: str) -> dict[str, str]:
    return {match.group(1): match.group(2) for match in KV_RE.finditer(line)}


def parse_bench_output(stdout: str) -> dict:
    parsed: dict[str, object] = {
        "precheck_line": None,
        "precheck": {},
        "gpu_mr_probe_line": None,
        "gpu_mr_probe": {},
        "ready_line": None,
        "ready": {},
        "iterations": [],
        "summary_line": None,
        "summary": {},
    }
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("PRECHECK "):
            parsed["precheck_line"] = line
            parsed["precheck"] = parse_keyvals(line)
            continue
        if line.startswith("GPU_MR_PROBE "):
            parsed["gpu_mr_probe_line"] = line
            parsed["gpu_mr_probe"] = parse_keyvals(line)
            continue
        if line.startswith("READY "):
            parsed["ready_line"] = line
            parsed["ready"] = parse_keyvals(line)
            continue
        if line.startswith("ITER "):
            parsed["iterations"].append(
                {
                    "line": line,
                    "fields": parse_keyvals(line),
                }
            )
            continue
        if line.startswith("SUMMARY "):
            parsed["summary_line"] = line
            parsed["summary"] = parse_keyvals(line)
    return parsed


def collect_host_facts(node_dir: Path) -> None:
    commands = {
        "hostname.txt": ["bash", "-lc", "hostname"],
        "hostname_i.txt": ["bash", "-lc", "hostname -i"],
        "env.txt": ["bash", "-lc", "env | sort"],
        "lscpu.txt": ["bash", "-lc", "lscpu"],
        "memlock.txt": ["bash", "-lc", "ulimit -l"],
        "nvidia_smi_topo.txt": ["bash", "-lc", "nvidia-smi topo -m"],
        "rdma_link.txt": ["bash", "-lc", "command -v rdma >/dev/null 2>&1 && rdma link || true"],
        "ibv_devinfo.txt": ["bash", "-lc", "command -v ibv_devinfo >/dev/null 2>&1 && ibv_devinfo || true"],
        "numactl_hardware.txt": ["bash", "-lc", "command -v numactl >/dev/null 2>&1 && numactl --hardware || true"],
        "infiniband_numa.txt": [
            "bash",
            "-lc",
            "for d in /sys/class/infiniband/*; do "
            "name=$(basename \"$d\"); "
            "node=$(cat \"$d/device/numa_node\" 2>/dev/null || echo -1); "
            "echo \"$name $node\"; "
            "done | sort",
        ],
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
            if "role=target" in content and "listen_port=" in content:
                return True
        time.sleep(1)
    return False


def nic_numa_node(nic: str) -> int:
    proc = subprocess.run(
        ["bash", "-lc", f"cat /sys/class/infiniband/{nic}/device/numa_node"],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"could not read NUMA node for {nic}: {proc.stderr.strip()}")
    try:
        return int(proc.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"invalid NUMA node for {nic}: {proc.stdout!r}") from exc


def wrap_with_numa(command: list[str], nic: str, bind_numa: bool) -> tuple[list[str], int | None]:
    if not bind_numa:
        return command, None
    node = nic_numa_node(nic)
    if node < 0:
        raise RuntimeError(f"--bind-numa requested but {nic} reported NUMA node {node}")
    return [sys.executable, str(NUMA_WRAPPER), "--numa-node", str(node), "--"] + command, node


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--listen-port", type=int, default=6606)
    parser.add_argument("--memory", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--target-gpu-id", type=int, default=0)
    parser.add_argument("--initiator-gpu-id", type=int, default=0)
    parser.add_argument("--target-nic", required=True)
    parser.add_argument("--initiator-nic", required=True)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--warmup-iterations", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--qp-count", type=int, default=1)
    parser.add_argument("--duration-sec", type=int, default=0)
    parser.add_argument("--case-timeout-sec", type=int, default=300)
    parser.add_argument("--direct-rdma", action="store_true")
    parser.add_argument("--strict-direct-rdma", action="store_true")
    parser.add_argument("--bind-numa", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

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
    if args.strict_direct_rdma and not args.direct_rdma:
        print("--strict-direct-rdma requires --direct-rdma", file=sys.stderr)
        return 2
    if args.memory != "gpu" and (args.direct_rdma or args.strict_direct_rdma):
        print("direct RDMA requires --memory gpu", file=sys.stderr)
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
    gpu_id = args.target_gpu_id if node_rank == 0 else args.initiator_gpu_id
    target_ready = case_dir / "target_ready.json"
    client_done = case_dir / "client_done.json"
    result_json = case_dir / "result.json"

    write_json(
        case_dir / "case_config.json",
        {
            "case_name": args.case_name,
            "memory": args.memory,
            "bytes": args.bytes,
            "target_gpu_id": args.target_gpu_id,
            "initiator_gpu_id": args.initiator_gpu_id,
            "target_nic": args.target_nic,
            "initiator_nic": args.initiator_nic,
            "iterations": args.iterations,
            "warmup_iterations": args.warmup_iterations,
            "threads": args.threads,
            "batch_size": args.batch_size,
            "qp_count": args.qp_count,
            "duration_sec": args.duration_sec,
            "case_timeout_sec": args.case_timeout_sec,
            "direct_rdma": args.direct_rdma,
            "strict_direct_rdma": args.strict_direct_rdma,
            "bind_numa": args.bind_numa,
            "verify": not args.no_verify,
        },
    )

    inspect_cmd = [
        str(bench_binary),
        "--role",
        "inspect",
        "--memory",
        args.memory,
        "--bytes",
        str(args.bytes),
        "--gpu-id",
        str(gpu_id),
        "--batch-size",
        str(args.batch_size),
        "--qp-count",
        str(args.qp_count),
        "--rdma",
        "--rdma-nic",
        nic,
        "--strict-nic",
    ]
    if args.direct_rdma:
        inspect_cmd.append("--direct-rdma")
    if args.strict_direct_rdma:
        inspect_cmd.append("--strict-direct-rdma")
    if args.memory == "gpu":
        inspect_cmd.append("--probe-gpu-mr")

    try:
        wrapped_inspect_cmd, bound_numa_node = wrap_with_numa(inspect_cmd, nic, args.bind_numa)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 4

    inspect_proc = run_command(wrapped_inspect_cmd, node_dir / "inspect.json")
    inspect_parsed = parse_bench_output(inspect_proc.stdout)
    write_json(
        node_dir / "inspect_summary.json",
        {
            "command": wrapped_inspect_cmd,
            "returncode": inspect_proc.returncode,
            "bound_numa_node": bound_numa_node,
            "parsed": inspect_parsed,
        },
    )
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
            args.case_name,
            "--memory",
            args.memory,
            "--bytes",
            str(args.bytes),
            "--gpu-id",
            str(args.target_gpu_id),
            "--threads",
            str(args.threads),
            "--batch-size",
            str(args.batch_size),
            "--qp-count",
            str(args.qp_count),
            "--rdma",
            "--rdma-nic",
            args.target_nic,
            "--strict-nic",
        ]
        if args.direct_rdma:
            target_cmd.append("--direct-rdma")
        if args.strict_direct_rdma:
            target_cmd.append("--strict-direct-rdma")

        try:
            wrapped_target_cmd, target_numa_node = wrap_with_numa(target_cmd, args.target_nic, args.bind_numa)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 4

        target_log = node_dir / "target.log"
        with target_log.open("w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(wrapped_target_cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)
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
                "target_gpu_id": args.target_gpu_id,
                "memory": args.memory,
                "bound_numa_node": target_numa_node,
            },
        )
        if not wait_for_file(client_done, timeout_sec=args.case_timeout_sec):
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

    if not wait_for_file(target_ready, timeout_sec=args.case_timeout_sec):
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
        args.case_name,
        "--memory",
        args.memory,
        "--bytes",
        str(args.bytes),
        "--gpu-id",
        str(args.initiator_gpu_id),
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
        "--threads",
        str(args.threads),
        "--batch-size",
        str(args.batch_size),
        "--qp-count",
        str(args.qp_count),
    ]
    if args.duration_sec > 0:
        initiator_cmd.extend(["--duration-sec", str(args.duration_sec)])
    if args.direct_rdma:
        initiator_cmd.append("--direct-rdma")
    if args.strict_direct_rdma:
        initiator_cmd.append("--strict-direct-rdma")
    if args.no_verify:
        initiator_cmd.append("--no-verify")

    try:
        wrapped_initiator_cmd, initiator_numa_node = wrap_with_numa(initiator_cmd, args.initiator_nic, args.bind_numa)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 4

    result = run_command(wrapped_initiator_cmd, node_dir / "initiator.json")
    parsed = parse_bench_output(result.stdout)
    summary = {
        "case_name": args.case_name,
        "returncode": result.returncode,
        "host": host,
        "target_host": target_info["host"],
        "target_listen_ip": target_info["listen_ip"],
        "memory": args.memory,
        "bytes": args.bytes,
        "target_nic": target_info["target_nic"],
        "initiator_nic": args.initiator_nic,
        "target_gpu_id": target_info["target_gpu_id"],
        "initiator_gpu_id": args.initiator_gpu_id,
        "direct_rdma": args.direct_rdma,
        "strict_direct_rdma": args.strict_direct_rdma,
        "verify": not args.no_verify,
        "duration_sec": args.duration_sec,
        "iterations_requested": args.iterations,
        "warmup_iterations": args.warmup_iterations,
        "threads": args.threads,
        "batch_size": args.batch_size,
        "qp_count": args.qp_count,
        "bind_numa": args.bind_numa,
        "initiator_bound_numa_node": initiator_numa_node,
        "target_bound_numa_node": target_info.get("bound_numa_node"),
        "parsed": parsed,
    }
    write_json(result_json, summary)
    write_json(
        client_done,
        {
            "case_name": args.case_name,
            "host": host,
            "returncode": result.returncode,
            "result_json": str(result_json),
        },
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
