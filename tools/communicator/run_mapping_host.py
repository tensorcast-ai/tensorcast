#!/usr/bin/env python3
#  Copyright (c) 2026, TensorCast Team.

"""Run one side of a multi-lane communicator GPU RDMA mapping case."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time


REPO_ROOT = Path("./")
BENCH_BINARY = REPO_ROOT / "bazel-bin/core/communicator/communicator_bench_binary"
NUMA_WRAPPER = REPO_ROOT / "tools/communicator/run_with_numa_policy.py"


def parse_csv_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_csv_strings(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_summary(log_path: Path) -> dict[str, str] | None:
    if not log_path.exists():
        return None
    for raw_line in reversed(log_path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        line = raw_line.strip()
        if not line.startswith("SUMMARY "):
            continue
        fields: dict[str, str] = {}
        for token in line.split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            fields[key] = value
        return fields
    return None


def run_command(command: list[str], log_path: Path) -> subprocess.Popen[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    except Exception:
        log_file.close()
        raise
    return proc


def wait_for_ready(log_path: Path, timeout_sec: int) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="ignore")
            if "READY role=target" in text:
                return
        time.sleep(1)
    raise TimeoutError(f"target log did not print READY: {log_path}")


def wait_for_file(path: Path, timeout_sec: int) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(1)
    raise TimeoutError(f"timed out waiting for {path}")


def maybe_wrap_with_numa(cmd: list[str], nic: str, bind_numa: bool) -> list[str]:
    if not bind_numa:
        return cmd
    numa_node_path = Path(f"/sys/class/infiniband/{nic}/device/numa_node")
    if not numa_node_path.exists():
        raise RuntimeError(f"missing NUMA node file for NIC {nic}")
    numa_node = int(numa_node_path.read_text(encoding="utf-8").strip())
    if numa_node < 0:
        raise RuntimeError(f"invalid NUMA node for NIC {nic}: {numa_node}")
    return [sys.executable or "python3", str(NUMA_WRAPPER), "--numa-node", str(numa_node), "--", *cmd]


def build_target_command(args: argparse.Namespace, lane: int, gpu_id: int, nic: str, port: int) -> list[str]:
    cmd = [
        str(BENCH_BINARY),
        "--role",
        "target",
        "--listen-ip",
        "0.0.0.0",
        "--listen-port",
        str(port),
        "--tensor-key",
        f"{args.tensor_prefix}-lane{lane}",
        "--memory",
        "gpu",
        "--bytes",
        str(args.bytes),
        "--gpu-id",
        str(gpu_id),
        "--threads",
        "1",
        "--batch-size",
        "1",
        "--qp-count",
        str(args.qp_count),
        "--outstanding-wr",
        str(args.outstanding_wr),
        "--rdma",
        "--rdma-nic",
        nic,
        "--strict-nic",
        "--direct-rdma",
        "--strict-direct-rdma",
    ]
    return maybe_wrap_with_numa(cmd, nic, args.bind_numa)


def build_initiator_command(
    args: argparse.Namespace,
    lane: int,
    gpu_id: int,
    nic: str,
    port: int,
    expected_remote_nic: str,
) -> list[str]:
    cmd = [
        str(BENCH_BINARY),
        "--role",
        "initiator",
        "--listen-ip",
        "0.0.0.0",
        "--listen-port",
        str(args.base_port + 100 + lane),
        "--peer-ip",
        args.peer_ip,
        "--peer-port",
        str(port),
        "--tensor-key",
        f"{args.tensor_prefix}-lane{lane}",
        "--memory",
        "gpu",
        "--bytes",
        str(args.bytes),
        "--gpu-id",
        str(gpu_id),
        "--threads",
        "1",
        "--batch-size",
        "1",
        "--qp-count",
        str(args.qp_count),
        "--outstanding-wr",
        str(args.outstanding_wr),
        "--iterations",
        str(args.iterations),
        "--warmup-iterations",
        str(args.warmup_iterations),
        "--duration-sec",
        str(args.duration_sec),
        "--rdma",
        "--rdma-nic",
        nic,
        "--strict-nic",
        "--expected-remote-nic",
        expected_remote_nic,
        "--direct-rdma",
        "--strict-direct-rdma",
    ]
    if args.no_verify:
        cmd.append("--no-verify")
    return maybe_wrap_with_numa(cmd, nic, args.bind_numa)


def terminate_processes(processes: list[subprocess.Popen[str]]) -> None:
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()
    deadline = time.time() + 10
    for proc in processes:
        if proc.poll() is None:
            timeout = max(0.0, deadline - time.time())
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()


def collect_host_facts(case_dir: Path) -> None:
    commands = {
        "hostname.txt": ["hostname"],
        "hostname_i.txt": ["bash", "-lc", "hostname -i | awk '{print $1}'"],
        "nvidia_smi_topo.txt": ["nvidia-smi", "topo", "-m"],
        "rdma_link.txt": ["bash", "-lc", "command -v rdma >/dev/null 2>&1 && rdma link || true"],
        "ibv_devinfo.txt": ["bash", "-lc", "command -v ibv_devinfo >/dev/null 2>&1 && ibv_devinfo || true"],
    }
    for filename, command in commands.items():
        proc = subprocess.run(command, text=True, capture_output=True, check=False)
        (case_dir / filename).write_text(proc.stdout + proc.stderr, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["target", "initiator"], required=True)
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--tensor-prefix", required=True)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--nics", required=True)
    parser.add_argument("--target-nics", default="")
    parser.add_argument("--base-port", type=int, required=True)
    parser.add_argument("--peer-ip", default="")
    parser.add_argument("--bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--warmup-iterations", type=int, default=0)
    parser.add_argument("--duration-sec", type=int, default=5)
    parser.add_argument("--qp-count", type=int, default=4)
    parser.add_argument("--outstanding-wr", type=int, default=256)
    parser.add_argument("--bind-numa", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--ready-timeout-sec", type=int, default=180)
    parser.add_argument("--case-timeout-sec", type=int, default=1800)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gpu_ids = parse_csv_ints(args.gpu_ids)
    nics = parse_csv_strings(args.nics)
    if len(gpu_ids) != len(nics):
      print("gpu_ids and nics length mismatch", file=sys.stderr)
      return 2

    target_nics = parse_csv_strings(args.target_nics) if args.target_nics else []
    if args.role == "initiator" and len(target_nics) != len(gpu_ids):
      print("target_nics length mismatch", file=sys.stderr)
      return 2
    if args.role == "initiator" and not args.peer_ip:
      print("initiator requires --peer-ip", file=sys.stderr)
      return 2

    case_dir = Path(args.case_dir)
    role_dir = case_dir / "nodes" / f"{args.role}-{socket.gethostname()}"
    role_dir.mkdir(parents=True, exist_ok=True)
    collect_host_facts(role_dir)

    ready_path = case_dir / "target_ready.json"
    done_path = case_dir / "client_done.json"

    if args.role == "target":
        processes: list[subprocess.Popen[str]] = []
        lane_info: list[dict[str, object]] = []
        try:
            for lane, (gpu_id, nic) in enumerate(zip(gpu_ids, nics)):
                port = args.base_port + lane
                log_path = role_dir / f"target-lane{lane}.log"
                cmd = build_target_command(args, lane, gpu_id, nic, port)
                proc = run_command(cmd, log_path)
                processes.append(proc)
                wait_for_ready(log_path, args.ready_timeout_sec)
                lane_info.append(
                    {
                        "lane": lane,
                        "gpu_id": gpu_id,
                        "nic": nic,
                        "port": port,
                        "log_path": str(log_path),
                    }
                )
            write_json(
                ready_path,
                {
                    "host": socket.gethostname(),
                    "host_ip": socket.gethostbyname(socket.gethostname()),
                    "lanes": lane_info,
                },
            )
            wait_for_file(done_path, args.case_timeout_sec)
            result = json.loads(done_path.read_text(encoding="utf-8"))
            return int(result.get("returncode", 1))
        finally:
            terminate_processes(processes)

    wait_for_file(ready_path, args.case_timeout_sec)
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    start = time.time()
    processes: list[subprocess.Popen[str]] = []
    lane_logs: list[Path] = []
    try:
        for lane, (gpu_id, nic, remote_nic) in enumerate(zip(gpu_ids, nics, target_nics)):
            port = args.base_port + lane
            log_path = role_dir / f"initiator-lane{lane}.log"
            cmd = build_initiator_command(args, lane, gpu_id, nic, port, remote_nic)
            proc = run_command(cmd, log_path)
            processes.append(proc)
            lane_logs.append(log_path)
        lane_returncodes = [proc.wait(timeout=args.case_timeout_sec) for proc in processes]
    finally:
        terminate_processes(processes)
    end = time.time()

    lane_summaries: list[dict[str, object]] = []
    total_bytes = 0
    ok = True
    for lane, log_path, returncode in zip(range(len(gpu_ids)), lane_logs, lane_returncodes):
        summary = parse_summary(log_path)
        if returncode != 0 or summary is None:
            ok = False
        lane_summaries.append(
            {
                "lane": lane,
                "returncode": returncode,
                "summary": summary,
                "log_path": str(log_path),
            }
        )
        if summary is not None:
            total_bytes += int(float(summary.get("bytes", "0")))

    wall_us = (end - start) * 1.0e6
    max_lane_wall_us = 0.0
    for lane_info in lane_summaries:
        summary = lane_info.get("summary")
        if isinstance(summary, dict):
            try:
                max_lane_wall_us = max(max_lane_wall_us, float(summary.get("wall_us", "0")))
            except ValueError:
                pass
    aggregate_bw_case_wall_GBps = 0.0 if wall_us <= 0 else (float(total_bytes) / 1.0e9) / (wall_us / 1.0e6)
    aggregate_bw_data_plane_GBps = (
        0.0 if max_lane_wall_us <= 0 else (float(total_bytes) / 1.0e9) / (max_lane_wall_us / 1.0e6)
    )
    result = {
        "returncode": 0 if ok else 1,
        "host": socket.gethostname(),
        "peer_ip": args.peer_ip,
        "bytes": args.bytes,
        "iterations": args.iterations,
        "warmup_iterations": args.warmup_iterations,
        "duration_sec": args.duration_sec,
        "qp_count": args.qp_count,
        "outstanding_wr": args.outstanding_wr,
        "lanes": lane_summaries,
        "aggregate": {
            "lane_count": len(gpu_ids),
            "total_bytes": total_bytes,
            "wall_us": wall_us,
            "max_lane_wall_us": max_lane_wall_us,
            "bw_GBps_case_wall": aggregate_bw_case_wall_GBps,
            "bw_GBps_data_plane": aggregate_bw_data_plane_GBps,
            "bw_gbps_case_wall": aggregate_bw_case_wall_GBps,
            "bw_gbps_data_plane": aggregate_bw_data_plane_GBps,
        },
    }
    write_json(case_dir / "result.json", result)
    write_json(done_path, {"returncode": result["returncode"]})
    return int(result["returncode"])


if __name__ == "__main__":
    sys.exit(main())
