#!/usr/bin/env python3
#  Copyright (c) 2026, TensorCast Team.

"""Launch one multi-lane communicator GPU RDMA mapping case."""

from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


LOCAL_REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_REL = Path("tools/communicator/run_mapping_host.py")
RUNNER = LOCAL_REPO_ROOT / RUNNER_REL
NAMESPACE = "tensorcast"
RUN_AS_USER = subprocess.check_output(["id", "-un"], text=True).strip()
DEFAULT_REMOTE_REPO_ROOT = os.environ.get(
    "TENSORCAST_REMOTE_REPO_ROOT", str(LOCAL_REPO_ROOT)
)


def run_command(
    command: list[str], log_path: Path | None = None
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(command, text=True, capture_output=True, check=False)
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


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def parse_csv_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_csv_strings(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


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
        f"run_as_user = {RUN_AS_USER!r}\n"
        f"script = base64.b64decode({encoded!r}).decode('utf-8')\n"
        "if os.geteuid() == 0:\n"
        "    cmd = ['su', '-', run_as_user, '-s', '/bin/bash', '-c', script]\n"
        "else:\n"
        "    cmd = ['bash', '-lc', script]\n"
        "raise SystemExit(subprocess.run(cmd, check=False).returncode)\n"
        "PY\n"
    )


def orchestratorctl_exec(
    worker_id: str, shell_command: str, log_path: Path | None = None
) -> subprocess.CompletedProcess[str]:
    return run_command(
        [
            "orchestratorctl",
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


def launch_flags(args: argparse.Namespace, comment: str) -> list[str]:
    return [
        "orchestratorctl",
        "launch",
        "--charged-group",
        args.charged_group,
        "--gpu",
        "8",
        "--cpu",
        "8",
        "--memory",
        "32768",
        "--private-machine",
        "group",
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


def predict_worker(
    args: argparse.Namespace, role: str, log_path: Path
) -> subprocess.CompletedProcess[str]:
    return run_command(
        launch_flags(args, f"{args.label}-{role}") + ["--predict-only"], log_path
    )


def launch_worker(args: argparse.Namespace, role: str, log_path: Path) -> str:
    keepalive_cmd = wrap_remote_shell_command(
        "set -euo pipefail\necho START\nhostname\nid -un\nsleep 7200\n"
    )
    proc = run_command(
        launch_flags(args, f"{args.label}-{role}")
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
        raise RuntimeError("launch did not return worker id")
    return worker_id


def worker_status(worker_id: str) -> str:
    proc = run_command(
        ["orchestratorctl", "get", "process", worker_id, "-n", NAMESPACE]
    )
    if proc.returncode != 0:
        return "UNKNOWN"
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        return "UNKNOWN"
    parts = lines[1].split()
    return parts[4] if len(parts) >= 5 else "UNKNOWN"


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


def remote_stdout(worker_id: str, shell_command: str, log_path: Path) -> str:
    proc = orchestratorctl_exec(worker_id, shell_command, log_path)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip())
    return proc.stdout.strip()


def delete_worker(worker_id: str, log_path: Path) -> None:
    run_command(
        ["orchestratorctl", "delete", "process", worker_id, "-n", NAMESPACE], log_path
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--source-gpus", required=True)
    parser.add_argument("--source-nics", required=True)
    parser.add_argument("--target-gpus", required=True)
    parser.add_argument("--target-nics", required=True)
    parser.add_argument("--base-port", type=int, default=6800)
    parser.add_argument("--bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--warmup-iterations", type=int, default=0)
    parser.add_argument("--duration-sec", type=int, default=5)
    parser.add_argument("--qp-count", type=int, default=4)
    parser.add_argument("--outstanding-wr", type=int, default=256)
    parser.add_argument("--bind-numa", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--case-timeout-sec", type=int, default=1800)
    parser.add_argument("--worker-max-wait-duration", default="20m")
    parser.add_argument("--charged-group", default="tensorcast-dev")
    parser.add_argument("--require-target-host", default="")
    parser.add_argument("--require-initiator-host", default="")
    parser.add_argument(
        "--remote-repo-root",
        default=DEFAULT_REMOTE_REPO_ROOT,
        help="Absolute TensorCast checkout path on remote workers.",
    )
    return parser.parse_args()


def validate_remote_repo_root(remote_repo_root: str) -> str:
    resolved = remote_repo_root.strip()
    if not resolved or not resolved.startswith("/"):
        raise ValueError(
            "--remote-repo-root must be an absolute path on remote workers"
        )
    return resolved


def remote_preamble(args: argparse.Namespace) -> str:
    remote_repo_root = validate_remote_repo_root(args.remote_repo_root)
    return (
        f"remote_repo_root={shlex.quote(remote_repo_root)} && "
        '[ -d "$remote_repo_root" ] || { echo "remote repo root not found: $remote_repo_root" >&2; exit 3; }; '
        'cd "$remote_repo_root" && '
        '[ -f .venv/bin/activate ] || { echo "missing .venv at $remote_repo_root" >&2; exit 3; }; '
        "source .venv/bin/activate && "
        "export LD_LIBRARY_PATH=/data/cuda/compat:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
    )


def main() -> int:
    if RUN_AS_USER == "root":
        print("refuse to run remote workload as root", file=sys.stderr)
        return 2
    args = parse_args()

    source_gpus = parse_csv_ints(args.source_gpus)
    source_nics = parse_csv_strings(args.source_nics)
    target_gpus = parse_csv_ints(args.target_gpus)
    target_nics = parse_csv_strings(args.target_nics)
    lane_count = len(source_gpus)
    if (
        lane_count == 0
        or lane_count != len(source_nics)
        or lane_count != len(target_gpus)
        or lane_count != len(target_nics)
    ):
        print("lane mapping lengths must match and be non-zero", file=sys.stderr)
        return 2
    if not RUNNER.exists():
        print(f"missing runner: {RUNNER}", file=sys.stderr)
        return 2

    case_dir = Path(args.case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        case_dir / "case_config.json",
        {
            "label": args.label,
            "source_gpus": source_gpus,
            "source_nics": source_nics,
            "target_gpus": target_gpus,
            "target_nics": target_nics,
            "bytes": args.bytes,
            "iterations": args.iterations,
            "warmup_iterations": args.warmup_iterations,
            "duration_sec": args.duration_sec,
            "qp_count": args.qp_count,
            "outstanding_wr": args.outstanding_wr,
            "bind_numa": args.bind_numa,
            "verify": not args.no_verify,
            "remote_repo_root": args.remote_repo_root,
        },
    )

    worker_ids: list[str] = []
    try:
        if (
            predict_worker(
                args, "target", case_dir / "launch_target_predict.json"
            ).returncode
            != 0
        ):
            raise RuntimeError("target predict failed")
        if (
            predict_worker(
                args, "initiator", case_dir / "launch_initiator_predict.json"
            ).returncode
            != 0
        ):
            raise RuntimeError("initiator predict failed")

        target_worker = launch_worker(args, "target", case_dir / "launch_target.json")
        initiator_worker = launch_worker(
            args, "initiator", case_dir / "launch_initiator.json"
        )
        worker_ids.extend([target_worker, initiator_worker])
        wait_worker_running(target_worker, 900)
        wait_worker_running(initiator_worker, 900)

        target_host = remote_stdout(
            target_worker, "hostname", case_dir / "target_hostname_exec.json"
        )
        initiator_host = remote_stdout(
            initiator_worker, "hostname", case_dir / "initiator_hostname_exec.json"
        )
        if target_host == initiator_host:
            raise RuntimeError(f"workers landed on same host: {target_host}")
        if args.require_target_host and args.require_target_host not in target_host:
            raise RuntimeError(
                f"target host mismatch: expected contains {args.require_target_host}, got {target_host}"
            )
        if (
            args.require_initiator_host
            and args.require_initiator_host not in initiator_host
        ):
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

        target_args = [
            "python",
            str(RUNNER_REL),
            "--role",
            "target",
            "--case-dir",
            str(case_dir),
            "--tensor-prefix",
            args.label,
            "--gpu-ids",
            ",".join(str(item) for item in target_gpus),
            "--nics",
            ",".join(target_nics),
            "--base-port",
            str(args.base_port),
            "--bytes",
            str(args.bytes),
            "--iterations",
            str(args.iterations),
            "--warmup-iterations",
            str(args.warmup_iterations),
            "--duration-sec",
            str(args.duration_sec),
            "--qp-count",
            str(args.qp_count),
            "--outstanding-wr",
            str(args.outstanding_wr),
            "--case-timeout-sec",
            str(args.case_timeout_sec),
        ]
        if args.bind_numa:
            target_args.append("--bind-numa")
        if args.no_verify:
            target_args.append("--no-verify")
        target_shell = f"{remote_preamble(args)} && {shell_join(target_args)}"
        target_bg = (
            f"set -euo pipefail\nmkdir -p {shlex.quote(str(case_dir))}\n"
            f"({target_shell}) > {shlex.quote(str(case_dir / 'target_exec.log'))} 2>&1 &\n"
            f"echo $! > {shlex.quote(str(case_dir / 'target_exec.pid'))}\n"
        )
        target_proc = orchestratorctl_exec(
            target_worker, target_bg, case_dir / "target_exec_start.json"
        )
        if target_proc.returncode != 0:
            raise RuntimeError("failed to start target mapping host runner")

        initiator_args = [
            "python",
            str(RUNNER_REL),
            "--role",
            "initiator",
            "--case-dir",
            str(case_dir),
            "--tensor-prefix",
            args.label,
            "--gpu-ids",
            ",".join(str(item) for item in source_gpus),
            "--nics",
            ",".join(source_nics),
            "--target-nics",
            ",".join(target_nics),
            "--base-port",
            str(args.base_port),
            "--peer-ip",
            target_ip,
            "--bytes",
            str(args.bytes),
            "--iterations",
            str(args.iterations),
            "--warmup-iterations",
            str(args.warmup_iterations),
            "--duration-sec",
            str(args.duration_sec),
            "--qp-count",
            str(args.qp_count),
            "--outstanding-wr",
            str(args.outstanding_wr),
            "--case-timeout-sec",
            str(args.case_timeout_sec),
        ]
        if args.bind_numa:
            initiator_args.append("--bind-numa")
        if args.no_verify:
            initiator_args.append("--no-verify")
        initiator_shell = f"{remote_preamble(args)} && {shell_join(initiator_args)}"
        initiator_proc = orchestratorctl_exec(
            initiator_worker,
            initiator_shell,
            case_dir / "initiator_exec.json",
        )
        if initiator_proc.returncode != 0:
            raise RuntimeError("initiator mapping host runner failed")

        result = json.loads((case_dir / "result.json").read_text(encoding="utf-8"))
        result["target_host"] = target_host
        result["initiator_host"] = initiator_host
        result["target_ip"] = target_ip
        result["initiator_ip"] = initiator_ip
        write_json(case_dir / "result.json", result)
        print(json.dumps(result, indent=2))
        return int(result.get("returncode", 1))
    finally:
        for worker_id, name in (
            (worker_ids[0], "cleanup_target_worker.json")
            if len(worker_ids) > 0
            else (None, None),
            (worker_ids[1], "cleanup_initiator_worker.json")
            if len(worker_ids) > 1
            else (None, None),
        ):
            if worker_id is None:
                continue
            delete_worker(worker_id, case_dir / name)


if __name__ == "__main__":
    sys.exit(main())
