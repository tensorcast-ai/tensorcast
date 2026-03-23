#!/usr/bin/env python3
#  Copyright (c) 2026, TensorCast Team.

"""Launch one communicator benchmark case with two detached single-node workers."""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/data/workspace/tensorcast-280")
BENCH_BINARY = REPO_ROOT / "bazel-bin/core/communicator/communicator_bench_binary"
NAMESPACE = "shai-core"
NUMA_WRAPPER = REPO_ROOT / "tools/communicator/run_with_numa_policy.py"
RUN_AS_USER = (
    os.environ.get("TENSORCAST_RUN_AS_USER")
    or os.environ.get("BRAINCTL_RUN_AS_USER")
    or subprocess.check_output(["id", "-un"], text=True).strip()
).strip()


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
        fields = {}
        for token in line.split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            fields[key] = value
        if line.startswith("PRECHECK "):
            parsed["precheck_line"] = line
            parsed["precheck"] = fields
            continue
        if line.startswith("GPU_MR_PROBE "):
            parsed["gpu_mr_probe_line"] = line
            parsed["gpu_mr_probe"] = fields
            continue
        if line.startswith("READY "):
            parsed["ready_line"] = line
            parsed["ready"] = fields
            continue
        if line.startswith("ITER "):
            parsed["iterations"].append(
                {
                    "line": line,
                    "fields": fields,
                }
            )
            continue
        if line.startswith("SUMMARY "):
            parsed["summary_line"] = line
            parsed["summary"] = fields
    return parsed


def run_command(command: list[str], log_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    if log_path is not None:
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


def wrap_remote_shell_command(shell_command: str) -> str:
    encoded = base64.b64encode(shell_command.encode("utf-8")).decode("ascii")
    return (
        "set -euo pipefail\n"
        f"run_as_user={shlex.quote(RUN_AS_USER)}\n"
        'if [ "$run_as_user" = "root" ]; then echo "refuse to run remote workload as root" >&2; exit 2; fi\n'
        'getent passwd "$run_as_user" >/dev/null\n'
        "python3 - <<'PY'\n"
        "import base64\n"
        "import os\n"
        "import subprocess\n"
        "import sys\n"
        f"run_as_user = {RUN_AS_USER!r}\n"
        f"script = base64.b64decode({encoded!r}).decode('utf-8')\n"
        "if os.geteuid() == 0:\n"
        "    cmd = ['su', '-', run_as_user, '-s', '/bin/bash', '-c', script]\n"
        "else:\n"
        "    cmd = ['bash', '-lc', script]\n"
        "raise SystemExit(subprocess.run(cmd, check=False).returncode)\n"
        "PY\n"
    )


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
    parser.add_argument("--outstanding-wr", type=int, default=64)
    parser.add_argument("--duration-sec", type=int, default=0)
    parser.add_argument("--case-timeout-sec", type=int, default=600)
    parser.add_argument("--direct-rdma", action="store_true")
    parser.add_argument("--strict-direct-rdma", action="store_true")
    parser.add_argument("--bind-numa", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--worker-max-wait-duration", default="20m")
    parser.add_argument("--charged-group", default="tensorcast_dev")
    parser.add_argument("--worker-gpu", type=int, default=8)
    parser.add_argument("--worker-cpu", type=int, default=8)
    parser.add_argument("--worker-memory-mib", type=int, default=32768)
    parser.add_argument("--private-machine", default="group")
    parser.add_argument("--pool", default="")
    parser.add_argument("--positive-tags", default="")
    parser.add_argument("--negative-tags", default="")
    parser.add_argument("--require-target-host", default="")
    parser.add_argument("--require-initiator-host", default="")
    parser.add_argument("--keep-workers", action="store_true")
    parser.add_argument("--target-worker-id", default="")
    parser.add_argument("--initiator-worker-id", default="")
    return parser.parse_args()


def launch_flags(args: argparse.Namespace, comment: str) -> list[str]:
    flags = [
        "brainctl",
        "launch",
        "--charged-group",
        args.charged_group,
        "--gpu",
        str(args.worker_gpu),
        "--cpu",
        str(args.worker_cpu),
        "--memory",
        str(args.worker_memory_mib),
        "--host-network=true",
        "--custom-resources",
        "rdma/mlnx_shared=8",
        "--custom-resources",
        "mellanox.com/mlnx_rdma=1",
        "--max-wait-duration",
        args.worker_max_wait_duration,
        "--comment",
        comment,
    ]
    if args.private_machine:
        flags.extend(["--private-machine", args.private_machine])
    if args.pool:
        flags.extend(["--pool", args.pool])
    if args.positive_tags:
        flags.extend(["--positive-tags", args.positive_tags])
    if args.negative_tags:
        flags.extend(["--negative-tags", args.negative_tags])
    return flags


def predict_reports_no_capacity(proc: subprocess.CompletedProcess[str]) -> bool:
    return "no machine available" in proc.stdout.lower()


def predict_worker(args: argparse.Namespace, role: str, log_path: Path) -> subprocess.CompletedProcess[str]:
    return run_command(launch_flags(args, f"{args.case_name}-{role}") + ["--predict-only"], log_path)


def launch_worker(args: argparse.Namespace, role: str, log_path: Path) -> str:
    keepalive_cmd = wrap_remote_shell_command("set -euo pipefail\necho START\nhostname\nid -un\nsleep 7200\n")
    proc = run_command(
        launch_flags(args, f"{args.case_name}-{role}")
        + [
            "-d",
            "--replica-restart",
            "never",
            "--",
            "bash",
            "-lc",
            keepalive_cmd,
        ],
        log_path,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"launch {role} failed: {proc.stderr.strip()}")
    worker_id = proc.stdout.strip().splitlines()[-1].strip()
    if not worker_id:
        raise RuntimeError(f"launch {role} did not return a worker id")
    return worker_id


def brainctl_exec(worker_id: str, shell_command: str, log_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    return run_command(
        [
            "brainctl",
            "exec",
            f"process/{worker_id}",
            "-n",
            NAMESPACE,
            "--",
            "bash",
            "-lc",
            wrap_remote_shell_command(shell_command),
        ],
        log_path,
    )


def worker_status(worker_id: str) -> str:
    proc = run_command(["brainctl", "get", "process", worker_id, "-n", NAMESPACE])
    if proc.returncode != 0:
        return "UNKNOWN"
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        return "UNKNOWN"
    parts = lines[1].split()
    if len(parts) < 5:
        return "UNKNOWN"
    return parts[4]


def wait_worker_running(worker_id: str, timeout_sec: int) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        status = worker_status(worker_id)
        if status == "Running":
            return
        if status in {"Failed", "Succeeded"}:
            raise RuntimeError(f"worker {worker_id} entered terminal state {status}")
        time.sleep(5)
    raise RuntimeError(f"worker {worker_id} did not become Running before timeout")


def delete_worker(worker_id: str, log_path: Path) -> None:
    run_command(["brainctl", "delete", "process", worker_id, "-n", NAMESPACE], log_path)


def remote_stdout(worker_id: str, shell_command: str, log_path: Path) -> str:
    proc = brainctl_exec(worker_id, shell_command, log_path)
    if proc.returncode != 0:
        raise RuntimeError(f"remote command failed on {worker_id}: {proc.stderr.strip()}")
    return proc.stdout.strip()


def collect_host_facts(worker_id: str, node_dir: Path) -> dict[str, str]:
    commands = {
        "hostname.txt": "hostname",
        "hostname_i.txt": "hostname -i",
        "env.txt": "env | sort",
        "lscpu.txt": "lscpu",
        "memlock.txt": "ulimit -l",
        "nvidia_smi_topo.txt": "nvidia-smi topo -m",
        "rdma_link.txt": "command -v rdma >/dev/null 2>&1 && rdma link || true",
        "ibv_devinfo.txt": "command -v ibv_devinfo >/dev/null 2>&1 && ibv_devinfo || true",
        "numactl_hardware.txt": "command -v numactl >/dev/null 2>&1 && numactl --hardware || true",
        "infiniband_numa.txt": (
            "for d in /sys/class/infiniband/*; do "
            "name=$(basename \"$d\"); "
            "node=$(cat \"$d/device/numa_node\" 2>/dev/null || echo -1); "
            "echo \"$name $node\"; "
            "done | sort"
        ),
        "modules.txt": "cat /proc/modules | egrep 'nvidia_peermem|nv_peer_mem|mlx5' || true",
    }
    node_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, str] = {}
    for name, command in commands.items():
        proc = brainctl_exec(worker_id, command, node_dir / name)
        if proc.returncode == 0:
            summary[name] = proc.stdout.strip()
    return summary


def base_remote_preamble() -> str:
    return (
        "set -euo pipefail\n"
        f"cd {shlex.quote(str(REPO_ROOT))}\n"
        "source .venv/bin/activate\n"
        "export LD_LIBRARY_PATH=/data/cuda/compat:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}\n"
    )


def bench_shell_command(
    args: argparse.Namespace,
    role: str,
    nic: str,
    gpu_id: int,
    peer_ip: str = "",
    peer_port: int = 0,
) -> str:
    cmd = [
        str(BENCH_BINARY),
        "--role",
        role,
        "--listen-ip",
        "0.0.0.0",
        "--listen-port",
        str(args.listen_port if role == "target" else args.listen_port + 1),
        "--tensor-key",
        args.case_name,
        "--memory",
        args.memory,
        "--bytes",
        str(args.bytes),
        "--gpu-id",
        str(gpu_id),
        "--threads",
        str(args.threads),
        "--batch-size",
        str(args.batch_size),
        "--qp-count",
        str(args.qp_count),
        "--outstanding-wr",
        str(args.outstanding_wr),
        "--rdma",
        "--rdma-nic",
        nic,
        "--strict-nic",
    ]
    if role == "initiator":
        cmd.extend(
            [
                "--peer-ip",
                peer_ip,
                "--peer-port",
                str(peer_port),
                "--expected-remote-nic",
                args.target_nic,
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
                "--outstanding-wr",
                str(args.outstanding_wr),
            ]
        )
        if args.duration_sec > 0:
            cmd.extend(["--duration-sec", str(args.duration_sec)])
        if args.no_verify:
            cmd.append("--no-verify")
    if role == "inspect" and args.memory == "gpu":
        cmd.append("--probe-gpu-mr")
    if args.direct_rdma:
        cmd.append("--direct-rdma")
    if args.strict_direct_rdma:
        cmd.append("--strict-direct-rdma")

    bench_cmd = shlex.join(cmd)
    nic_path = shlex.quote(nic)
    shell = base_remote_preamble()
    if args.bind_numa:
        shell += (
            f"numa_node=\"$(cat /sys/class/infiniband/{nic_path}/device/numa_node)\"\n"
            "[ \"$numa_node\" -ge 0 ] || { echo \"invalid numa node for NIC\" >&2; exit 4; }\n"
            f"exec python {shlex.quote(str(NUMA_WRAPPER))} --numa-node \"$numa_node\" -- {bench_cmd}\n"
        )
    else:
        shell += f"exec {bench_cmd}\n"
    return shell


def wait_for_ready_line(log_path: Path, timeout_sec: int) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if log_path.exists():
            content = log_path.read_text(encoding="utf-8", errors="ignore")
            if "role=target" in content and "listen_port=" in content:
                return
        time.sleep(1)
    raise RuntimeError(f"target log did not reach READY before timeout: {log_path}")


def main() -> int:
    if RUN_AS_USER == "root":
        print("refuse to run remote workload as root", file=sys.stderr)
        return 2
    args = parse_args()
    reuse_workers = bool(args.target_worker_id or args.initiator_worker_id)
    if reuse_workers and (not args.target_worker_id or not args.initiator_worker_id):
        print("both --target-worker-id and --initiator-worker-id are required when reusing workers", file=sys.stderr)
        return 2
    case_dir = Path(args.case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        case_dir / "case_config.json",
        {
            "case_name": args.case_name,
            "charged_group": args.charged_group,
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
            "outstanding_wr": args.outstanding_wr,
            "duration_sec": args.duration_sec,
            "case_timeout_sec": args.case_timeout_sec,
            "direct_rdma": args.direct_rdma,
            "strict_direct_rdma": args.strict_direct_rdma,
            "bind_numa": args.bind_numa,
            "verify": not args.no_verify,
            "worker_gpu": args.worker_gpu,
            "worker_cpu": args.worker_cpu,
            "worker_memory_mib": args.worker_memory_mib,
            "private_machine": args.private_machine,
            "pool": args.pool,
            "positive_tags": args.positive_tags,
            "negative_tags": args.negative_tags,
            "keep_workers": args.keep_workers,
            "reuse_workers": reuse_workers,
            "target_worker_id": args.target_worker_id,
            "initiator_worker_id": args.initiator_worker_id,
        },
    )

    if not BENCH_BINARY.exists():
        print(f"bench binary does not exist: {BENCH_BINARY}", file=sys.stderr)
        return 2

    worker_ids: list[str] = []
    launched_worker_ids: list[str] = []
    try:
        if reuse_workers:
            target_worker = args.target_worker_id
            initiator_worker = args.initiator_worker_id
            worker_ids.extend([target_worker, initiator_worker])
            wait_worker_running(target_worker, timeout_sec=600)
            wait_worker_running(initiator_worker, timeout_sec=600)
        else:
            predict_target = predict_worker(args, "target", case_dir / "launch_target_predict.json")
            predict_initiator = predict_worker(args, "initiator", case_dir / "launch_initiator_predict.json")
            if predict_target.returncode != 0 or predict_initiator.returncode != 0:
                raise RuntimeError("predict-only failed for target or initiator")
            if predict_reports_no_capacity(predict_target) or predict_reports_no_capacity(predict_initiator):
                raise RuntimeError(
                    "predict-only reported no machine available for target or initiator; "
                    "adjust scheduler constraints (private-machine/pool/positive-tags) or wait for capacity"
                )

            target_worker = launch_worker(args, "target", case_dir / "launch_target.json")
            initiator_worker = launch_worker(args, "initiator", case_dir / "launch_initiator.json")
            worker_ids.extend([target_worker, initiator_worker])
            launched_worker_ids.extend([target_worker, initiator_worker])

            wait_worker_running(target_worker, timeout_sec=600)
            wait_worker_running(initiator_worker, timeout_sec=600)

        target_host = remote_stdout(target_worker, "hostname", case_dir / "target_hostname_exec.json")
        initiator_host = remote_stdout(initiator_worker, "hostname", case_dir / "initiator_hostname_exec.json")
        if target_host == initiator_host:
            raise RuntimeError(f"target and initiator landed on the same host: {target_host}")
        if args.require_target_host and args.require_target_host not in target_host:
            raise RuntimeError(
                f"target host mismatch: expected contains {args.require_target_host}, got {target_host}"
            )
        if args.require_initiator_host and args.require_initiator_host not in initiator_host:
            raise RuntimeError(
                f"initiator host mismatch: expected contains {args.require_initiator_host}, got {initiator_host}"
            )

        target_ip = remote_stdout(
            target_worker,
            "hostname -i | awk '{print $1}'",
            case_dir / "target_ip_exec.json",
        )
        initiator_ip = remote_stdout(
            initiator_worker,
            "hostname -i | awk '{print $1}'",
            case_dir / "initiator_ip_exec.json",
        )

        target_node_dir = case_dir / "nodes" / f"target-{target_host}"
        initiator_node_dir = case_dir / "nodes" / f"initiator-{initiator_host}"
        collect_host_facts(target_worker, target_node_dir)
        collect_host_facts(initiator_worker, initiator_node_dir)

        target_inspect = brainctl_exec(
            target_worker,
            bench_shell_command(args, "inspect", args.target_nic, args.target_gpu_id),
            target_node_dir / "inspect.json",
        )
        initiator_inspect = brainctl_exec(
            initiator_worker,
            bench_shell_command(args, "inspect", args.initiator_nic, args.initiator_gpu_id),
            initiator_node_dir / "inspect.json",
        )
        if target_inspect.returncode != 0 or initiator_inspect.returncode != 0:
            raise RuntimeError("inspect failed on target or initiator")

        target_log = case_dir / "target.log"
        target_pid = case_dir / "target.pid"
        target_shell = (
            base_remote_preamble()
            + f": > {shlex.quote(str(target_log))}\n"
            + f"(\n{bench_shell_command(args, 'target', args.target_nic, args.target_gpu_id)}) >> {shlex.quote(str(target_log))} 2>&1 &\n"
            + f"echo $! > {shlex.quote(str(target_pid))}\n"
            + f"cat {shlex.quote(str(target_pid))}\n"
        )
        target_start = brainctl_exec(target_worker, target_shell, case_dir / "target_start.json")
        if target_start.returncode != 0:
            raise RuntimeError(f"failed to start target bench: {target_start.stderr.strip()}")
        wait_for_ready_line(target_log, timeout_sec=120)

        initiator_result = brainctl_exec(
            initiator_worker,
            bench_shell_command(
                args,
                "initiator",
                args.initiator_nic,
                args.initiator_gpu_id,
                peer_ip=target_ip,
                peer_port=args.listen_port,
            ),
            initiator_node_dir / "initiator.json",
        )
        parsed = parse_bench_output(initiator_result.stdout)
        write_json(
            case_dir / "result.json",
            {
                "case_name": args.case_name,
                "returncode": initiator_result.returncode,
                "target_worker": target_worker,
                "initiator_worker": initiator_worker,
                "target_host": target_host,
                "initiator_host": initiator_host,
                "target_ip": target_ip,
                "initiator_ip": initiator_ip,
                "target_nic": args.target_nic,
                "initiator_nic": args.initiator_nic,
                "target_gpu_id": args.target_gpu_id,
                "initiator_gpu_id": args.initiator_gpu_id,
                "memory": args.memory,
                "bytes": args.bytes,
                "direct_rdma": args.direct_rdma,
                "strict_direct_rdma": args.strict_direct_rdma,
                "bind_numa": args.bind_numa,
                "verify": not args.no_verify,
                "duration_sec": args.duration_sec,
                "iterations_requested": args.iterations,
                "warmup_iterations": args.warmup_iterations,
                "threads": args.threads,
                "batch_size": args.batch_size,
                "qp_count": args.qp_count,
                "outstanding_wr": args.outstanding_wr,
                "reused_workers": reuse_workers,
                "parsed": parsed,
            },
        )
        return initiator_result.returncode
    finally:
        if len(worker_ids) > 0:
            with contextlib.suppress(Exception):
                brainctl_exec(
                    worker_ids[0],
                    (
                        f"if [ -f {shlex.quote(str(case_dir / 'target.pid'))} ]; then "
                        f"kill $(cat {shlex.quote(str(case_dir / 'target.pid'))}) >/dev/null 2>&1 || true; "
                        "fi"
                    ),
                    case_dir / "pre_cleanup_target_process.json",
                )
        if args.keep_workers:
            write_json(
                case_dir / "kept_workers.json",
                {
                    "target_worker": worker_ids[0] if len(worker_ids) > 0 else "",
                    "initiator_worker": worker_ids[1] if len(worker_ids) > 1 else "",
                },
            )
        else:
            for worker_id, log_name in (
                (launched_worker_ids[0], "cleanup_target_worker.json") if len(launched_worker_ids) > 0 else (None, None),
                (
                    launched_worker_ids[1],
                    "cleanup_initiator_worker.json",
                )
                if len(launched_worker_ids) > 1
                else (None, None),
            ):
                if worker_id is None or log_name is None:
                    continue
                with contextlib.suppress(Exception):
                    brainctl_exec(
                        worker_id,
                        (
                            f"if [ -f {shlex.quote(str(case_dir / 'target.pid'))} ]; then "
                            f"kill $(cat {shlex.quote(str(case_dir / 'target.pid'))}) >/dev/null 2>&1 || true; "
                            "fi"
                        ),
                        case_dir / f"pre_{log_name}",
                    )
                delete_worker(worker_id, case_dir / log_name)


if __name__ == "__main__":
    sys.exit(main())
