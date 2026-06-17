#!/usr/bin/env python
#  Copyright (c) 2026, TensorCast Team.

# Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import argparse
import json
import random
import shlex
import statistics
import string
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_ROOT = REPO_ROOT.as_posix()
SCRIPT_DIR = (REPO_ROOT / "examples" / "cross_host").resolve()
PUT_HELPER = (SCRIPT_DIR / "cross_host_put_once.py").as_posix()
GET_HELPER = (SCRIPT_DIR / "cross_host_get_once.py").as_posix()
DEREGISTER_HELPER = (SCRIPT_DIR / "cross_host_deregister_once.py").as_posix()


def run(cmd: str) -> str:
    proc = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if proc.returncode != 0:
        sys.stderr.write(f"[cmd failed] {cmd}\n")
        if proc.stdout:
            sys.stderr.write(proc.stdout)
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        raise RuntimeError(f"command failed with rc={proc.returncode}")
    return proc.stdout


def run_remote(process_id: str, inner_cmd: str) -> str:
    cmd = (
        f"orchestratorctl exec process/{process_id} -n tensorcast -- bash -lc "
        f"{shlex.quote(inner_cmd)}"
    )
    return run(cmd)


def extract_last_json(text: str) -> dict[str, Any]:
    for line in reversed([ln.strip() for ln in text.splitlines()]):
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError(f"no json object found in output:\n{text}")


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    return float(ordered[idx])


def random_suffix(n: int = 10) -> str:
    return "".join(random.choice(string.hexdigits.lower()) for _ in range(n))


def restart_daemon(
    *,
    process_id: str,
    daemon_config: str,
    daemon_session: str,
    daemon_id: str,
    storage_path: str,
    grpc_port: int,
    p2p_port: int,
    advertise_ip: str,
    gs_addr: str,
    conn: int,
    buffers: int,
    maxw: int,
    expected_gpu_channels: int,
    home: str,
) -> None:
    config_flag = f"-c {daemon_config}" if daemon_config else ""
    cmd = (
        "set -euo pipefail; "
        f"cd {BENCH_ROOT}; "
        "source .venv/bin/activate; "
        f"mkdir -p {shlex.quote(home)} {shlex.quote(storage_path)}; "
        f"export TENSORCAST_HOME={shlex.quote(home)}; "
        "LD_LIBRARY_PATH=/data/cuda/compat tensorcast-cli daemon stop >/dev/null 2>&1 || true; "
        "LD_LIBRARY_PATH=/data/cuda/compat tensorcast-cli daemon start "
        f"{config_flag} "
        f"--session {shlex.quote(daemon_session)} "
        "--global-store-mode connect "
        f"--global-store-address {shlex.quote(gs_addr)} "
        f"--set daemon_id={shlex.quote(daemon_id)} "
        f"--set server.storage_path={shlex.quote(storage_path)} "
        "--set server.listen.host=0.0.0.0 "
        f"--set server.listen.port={int(grpc_port)} "
        "--set server.p2p_listen.host=0.0.0.0 "
        f"--set server.p2p_listen.port={int(p2p_port)} "
        f"--set server.advertise.host={shlex.quote(advertise_ip)} "
        "--set high_availability.enabled=true "
        "--set high_availability.heartbeat_interval=5s "
        "--set high_availability.periodic_sync_interval=5s "
        f"--set communicator.transport.tcp_conn_count={int(conn)} "
        f"--set communicator.stager.buffers_per_flow={int(buffers)} "
        f"--set communicator.stager.max_window_segments={int(maxw)} "
        f"--set communicator.stager.expected_gpu_channels={int(expected_gpu_channels)} "
        "--set observability.logging.level=warn "
        "--json"
    )
    run_remote(process_id, cmd)


def restart_daemons(args: argparse.Namespace) -> None:
    restart_daemon(
        process_id=str(args.put_proc),
        daemon_config=str(args.daemon_config),
        daemon_session=str(args.put_session),
        daemon_id=str(args.put_daemon_id),
        storage_path=str(args.put_storage),
        grpc_port=int(args.put_grpc_port),
        p2p_port=int(args.put_p2p_port),
        advertise_ip=str(args.put_adv_ip),
        gs_addr=str(args.gs_addr),
        conn=int(args.conn),
        buffers=int(args.buffers),
        maxw=int(args.maxw),
        expected_gpu_channels=int(args.expected_gpu_channels),
        home=str(args.put_home),
    )
    restart_daemon(
        process_id=str(args.get_proc),
        daemon_config=str(args.daemon_config),
        daemon_session=str(args.get_session),
        daemon_id=str(args.get_daemon_id),
        storage_path=str(args.get_storage),
        grpc_port=int(args.get_grpc_port),
        p2p_port=int(args.get_p2p_port),
        advertise_ip=str(args.get_adv_ip),
        gs_addr=str(args.gs_addr),
        conn=int(args.conn),
        buffers=int(args.buffers),
        maxw=int(args.maxw),
        expected_gpu_channels=int(args.expected_gpu_channels),
        home=str(args.get_home),
    )


def run_put(
    *,
    process_id: str,
    daemon_addr: str,
    key: str,
    size_mib: int,
    seed: int,
    put_policy: str,
    tensor_count: int,
    dtype: str,
    put_device: str,
) -> dict[str, Any]:
    inner_cmd = (
        "set -euo pipefail; "
        f"cd {BENCH_ROOT}; "
        "source .venv/bin/activate; "
        "LD_LIBRARY_PATH=/data/cuda/compat "
        f"python {PUT_HELPER} "
        f"--daemon {shlex.quote(daemon_addr)} "
        f"--key {shlex.quote(key)} "
        f"--size-mib {int(size_mib)} "
        f"--seed {int(seed)} "
        f"--put-policy {shlex.quote(put_policy)} "
        f"--tensor-count {int(tensor_count)} "
        f"--dtype {shlex.quote(dtype)} "
        f"--put-device {shlex.quote(put_device)}"
    )
    return extract_last_json(run_remote(process_id, inner_cmd))


def run_get(
    *,
    process_id: str,
    daemon_addr: str,
    key: str,
    artifact_id: str,
    lookup_mode: str,
    get_device: str,
    visibility_timeout_sec: float,
    visibility_retry_sec: float,
) -> dict[str, Any]:
    inner_cmd = (
        "set -euo pipefail; "
        f"cd {BENCH_ROOT}; "
        "source .venv/bin/activate; "
        "LD_LIBRARY_PATH=/data/cuda/compat "
        f"python {GET_HELPER} "
        f"--daemon {shlex.quote(daemon_addr)} "
        f"--key {shlex.quote(key)} "
        f"--lookup-mode {shlex.quote(lookup_mode)} "
        f"--get-device {shlex.quote(get_device)} "
        f"--visibility-timeout-sec {float(visibility_timeout_sec)} "
        f"--visibility-retry-sec {float(visibility_retry_sec)} "
        "--prefer p2p "
        "--capture-comm-stats"
    )
    if lookup_mode == "artifact_id":
        inner_cmd += f" --artifact-id {shlex.quote(artifact_id)}"
    return extract_last_json(run_remote(process_id, inner_cmd))


def run_deregister(
    *,
    process_id: str,
    daemon_addr: str,
    artifact_id: str,
) -> dict[str, Any]:
    inner_cmd = (
        "set -euo pipefail; "
        f"cd {BENCH_ROOT}; "
        "source .venv/bin/activate; "
        "LD_LIBRARY_PATH=/data/cuda/compat "
        f"python {DEREGISTER_HELPER} "
        f"--daemon {shlex.quote(daemon_addr)} "
        f"--artifact-id {shlex.quote(artifact_id)}"
    )
    return extract_last_json(run_remote(process_id, inner_cmd))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-host split-process benchmark runner for put/get tuning."
    )
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--conn", type=int, required=True)
    parser.add_argument("--buffers", type=int, required=True)
    parser.add_argument("--maxw", type=int, required=True)
    parser.add_argument("--expected-gpu-channels", type=int, default=0)
    parser.add_argument("--size-mib", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--tensor-count", type=int, default=1)
    parser.add_argument(
        "--dtype", choices=("float16", "float32", "bfloat16"), default="float16"
    )
    parser.add_argument("--lookup-mode", choices=("key", "artifact_id"), default="key")
    parser.add_argument("--put-policy", default="pinned")
    parser.add_argument("--put-device", default="cuda:0")
    parser.add_argument("--get-device", default="cuda:0")
    parser.add_argument("--out-dir", default="/tmp/tensorcast/cross_host/results")
    parser.add_argument(
        "--require-p2p", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--cleanup-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--put-proc", required=True)
    parser.add_argument("--get-proc", required=True)
    parser.add_argument("--put-adv-ip", required=True)
    parser.add_argument("--get-adv-ip", required=True)
    parser.add_argument("--put-daemon-addr", required=True)
    parser.add_argument("--get-daemon-addr", required=True)
    parser.add_argument("--gs-addr", required=True)

    parser.add_argument("--put-grpc-port", type=int, default=62001)
    parser.add_argument("--get-grpc-port", type=int, default=62011)
    parser.add_argument("--put-p2p-port", type=int, default=63001)
    parser.add_argument("--get-p2p-port", type=int, default=63011)

    parser.add_argument("--put-home", default="/tmp/tensorcast/cross_host/put_home")
    parser.add_argument("--get-home", default="/tmp/tensorcast/cross_host/get_home")
    parser.add_argument(
        "--put-storage", default="/tmp/tensorcast/cross_host/put_storage"
    )
    parser.add_argument(
        "--get-storage", default="/tmp/tensorcast/cross_host/get_storage"
    )
    parser.add_argument("--put-session", default="tc-cross-put")
    parser.add_argument("--get-session", default="tc-cross-get")
    parser.add_argument("--put-daemon-id", default="tc-cross-put")
    parser.add_argument("--get-daemon-id", default="tc-cross-get")

    parser.add_argument("--visibility-timeout-sec", type=float, default=30.0)
    parser.add_argument("--visibility-retry-sec", type=float, default=0.05)
    parser.add_argument("--daemon-config", default="")

    args = parser.parse_args()
    out_dir = Path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[case] {args.case_name} conn={args.conn} buffers={args.buffers} maxw={args.maxw} "
        f"egc={args.expected_gpu_channels} size_mib={args.size_mib} warmup={args.warmup} "
        f"iter={args.iterations} lookup_mode={args.lookup_mode}",
        flush=True,
    )

    restart_daemons(args)

    total = int(args.warmup) + int(args.iterations)
    records: list[dict[str, Any]] = []
    run_tag = f"{int(time.time())}-{random_suffix(8)}"

    for iteration in range(total):
        warmup = iteration < int(args.warmup)
        key = (
            f"bench:cross:{args.case_name}:run{run_tag}:iter{iteration}:"
            f"{uuid.uuid4().hex}"
        )
        put_info = run_put(
            process_id=str(args.put_proc),
            daemon_addr=str(args.put_daemon_addr),
            key=key,
            size_mib=int(args.size_mib),
            seed=1000 + iteration,
            put_policy=str(args.put_policy),
            tensor_count=int(args.tensor_count),
            dtype=str(args.dtype),
            put_device=str(args.put_device),
        )
        artifact_id = str(put_info.get("artifact_id", ""))
        get_info: dict[str, Any] | None = None
        cleanup_put: dict[str, Any] | None = None
        cleanup_get: dict[str, Any] | None = None
        try:
            get_info = run_get(
                process_id=str(args.get_proc),
                daemon_addr=str(args.get_daemon_addr),
                key=key,
                artifact_id=artifact_id,
                lookup_mode=str(args.lookup_mode),
                get_device=str(args.get_device),
                visibility_timeout_sec=float(args.visibility_timeout_sec),
                visibility_retry_sec=float(args.visibility_retry_sec),
            )
        finally:
            if bool(args.cleanup_artifacts) and artifact_id:
                try:
                    cleanup_put = run_deregister(
                        process_id=str(args.put_proc),
                        daemon_addr=str(args.put_daemon_addr),
                        artifact_id=artifact_id,
                    )
                except Exception:  # noqa: BLE001
                    cleanup_put = {"removed": False, "error": "put_deregister_failed"}
                try:
                    cleanup_get = run_deregister(
                        process_id=str(args.get_proc),
                        daemon_addr=str(args.get_daemon_addr),
                        artifact_id=artifact_id,
                    )
                except Exception:  # noqa: BLE001
                    cleanup_get = {"removed": False, "error": "get_deregister_failed"}
        if get_info is None:
            raise RuntimeError("get_info is missing after run_get")

        record = {
            "iter": int(iteration),
            "warmup": bool(warmup),
            "key": key,
            "put": put_info,
            "get": get_info,
            "cleanup": {
                "put": cleanup_put,
                "get": cleanup_get,
            },
        }
        records.append(record)
        print(
            "  "
            f"iter={iteration:02d} warmup={warmup} src={get_info.get('source')} "
            f"put={float(put_info.get('put_sec', 0.0)):.4f}s "
            f"e2e={float(get_info.get('e2e_gibps', 0.0)):.3f}GiB/s "
            f"xfer={float(get_info.get('transfer_gibps', 0.0)):.3f}GiB/s "
            f"wait={float(get_info.get('visibility_wait_sec', 0.0)):.4f}s "
            f"attempts={int(get_info.get('attempts', 1))}",
            flush=True,
        )

    measured = [item for item in records if not bool(item["warmup"])]
    if not measured:
        raise RuntimeError("no measured records")

    bad_sources = [
        str(item["get"].get("source"))
        for item in measured
        if str(item["get"].get("source")) != "p2p"
    ]
    if bool(args.require_p2p) and bad_sources:
        raise RuntimeError(f"measured records include non-p2p sources: {bad_sources}")

    put_vals = [float(item["put"]["put_sec"]) for item in measured]
    e2e_vals = [float(item["get"]["e2e_gibps"]) for item in measured]
    transfer_vals = [float(item["get"]["transfer_gibps"]) for item in measured]
    wait_vals = [float(item["get"]["visibility_wait_sec"]) for item in measured]
    attempts_vals = [float(item["get"].get("attempts", 1)) for item in measured]

    summary: dict[str, Any] = {
        "case_name": str(args.case_name),
        "conn": int(args.conn),
        "buffers": int(args.buffers),
        "maxw": int(args.maxw),
        "expected_gpu_channels": int(args.expected_gpu_channels),
        "size_mib": int(args.size_mib),
        "warmup": int(args.warmup),
        "iterations": int(args.iterations),
        "lookup_mode": str(args.lookup_mode),
        "put_policy": str(args.put_policy),
        "p2p_ratio": 1.0
        if not bad_sources
        else 1.0 - (len(bad_sources) / len(measured)),
        "put_sec_mean": float(statistics.mean(put_vals)),
        "put_sec_p90": percentile(put_vals, 0.9),
        "e2e_gibps_mean": float(statistics.mean(e2e_vals)),
        "e2e_gibps_p50": percentile(e2e_vals, 0.5),
        "e2e_gibps_p90": percentile(e2e_vals, 0.9),
        "transfer_gibps_mean": float(statistics.mean(transfer_vals)),
        "transfer_gibps_p50": percentile(transfer_vals, 0.5),
        "transfer_gibps_p90": percentile(transfer_vals, 0.9),
        "visibility_wait_sec_p90": percentile(wait_vals, 0.9),
        "visibility_attempts_p90": percentile(attempts_vals, 0.9),
    }

    out_path = out_dir / f"{args.case_name}.json"
    payload = {
        "summary": summary,
        "records": records,
        "generated_at_epoch": time.time(),
        "params": {
            "put_proc": str(args.put_proc),
            "get_proc": str(args.get_proc),
            "put_daemon_addr": str(args.put_daemon_addr),
            "get_daemon_addr": str(args.get_daemon_addr),
            "gs_addr": str(args.gs_addr),
            "daemon_config": str(args.daemon_config),
        },
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("SUMMARY", json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"OUTPUT {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
