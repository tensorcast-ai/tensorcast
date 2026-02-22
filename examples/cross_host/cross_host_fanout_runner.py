#!/usr/bin/env python
#  Copyright (c) 2026, TensorCast Team.

# Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import shlex
import statistics
import string
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_ROOT = REPO_ROOT.as_posix()
SCRIPT_DIR = (REPO_ROOT / "examples" / "cross_host").resolve()
PUT_HELPER = (SCRIPT_DIR / "cross_host_put_once.py").as_posix()
GET_HELPER = (SCRIPT_DIR / "cross_host_get_once.py").as_posix()
DEREGISTER_HELPER = (SCRIPT_DIR / "cross_host_deregister_once.py").as_posix()


@dataclass(frozen=True)
class WorkerSpec:
    name: str
    process_id: str
    daemon_addr: str
    advertise_ip: str
    grpc_port: int
    p2p_port: int
    daemon_session: str
    daemon_id: str
    home: str
    storage: str


class GsDbUnavailableError(RuntimeError):
    """Raised when benchmark DB probes cannot access the GS duckdb file."""


def run(cmd: str, *, timeout_sec: float | None = None) -> str:
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        stdout_tail = (exc.stdout or "")[-4000:]
        stderr_tail = (exc.stderr or "")[-4000:]
        raise RuntimeError(
            "command timeout "
            f"(timeout_sec={timeout_sec}): {cmd}\n"
            f"[stdout tail]\n{stdout_tail}\n"
            f"[stderr tail]\n{stderr_tail}"
        ) from exc
    if proc.returncode != 0:
        sys.stderr.write(f"[cmd failed] {cmd}\n")
        if proc.stdout:
            sys.stderr.write(proc.stdout)
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        raise RuntimeError(f"command failed with rc={proc.returncode}")
    return proc.stdout


def run_remote(
    process_id: str,
    inner_cmd: str,
    *,
    timeout_sec: float,
) -> str:
    cmd = (
        f"brainctl exec process/{process_id} -n shai-core -- bash -lc "
        f"{shlex.quote(inner_cmd)}"
    )
    return run(cmd, timeout_sec=timeout_sec)


def extract_last_json(text: str) -> dict[str, Any]:
    for line in reversed([ln.strip() for ln in text.splitlines()]):
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError(f"no json object found in output:\n{text}")


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    return float(ordered[idx])


def random_suffix(n: int = 10) -> str:
    return "".join(random.choice(string.hexdigits.lower()) for _ in range(n))


def default_daemon_addr(advertise_ip: str, grpc_port: int) -> str:
    return f"{advertise_ip}:{grpc_port}"


def restart_daemon(
    *,
    worker: WorkerSpec,
    daemon_config: str,
    gs_addr: str,
    conn: int,
    buffers: int,
    maxw: int,
    expected_gpu_channels: int,
    timeout_sec: float,
) -> None:
    print(
        f"[restart] {worker.name} proc={worker.process_id} "
        f"grpc={worker.grpc_port} p2p={worker.p2p_port}",
        flush=True,
    )
    config_flag = f"-c {daemon_config}" if daemon_config else ""
    cmd = (
        "set -euo pipefail; "
        f"cd {BENCH_ROOT}; "
        "source .venv/bin/activate; "
        f"mkdir -p {shlex.quote(worker.home)} {shlex.quote(worker.storage)}; "
        f"export TENSORCAST_HOME={shlex.quote(worker.home)}; "
        "LD_LIBRARY_PATH=/data/cuda/compat tensorcast-cli daemon stop >/dev/null 2>&1 || true; "
        "for pid in $(pgrep -f '[t]ensorcast_daemon --config=' || true); do "
        'kill -TERM "$pid" >/dev/null 2>&1 || true; '
        "done; "
        "sleep 1; "
        "for pid in $(pgrep -f '[t]ensorcast_daemon --config=' || true); do "
        'kill -KILL "$pid" >/dev/null 2>&1 || true; '
        "done; "
        "LD_LIBRARY_PATH=/data/cuda/compat tensorcast-cli daemon start "
        f"{config_flag} "
        f"--session {shlex.quote(worker.daemon_session)} "
        "--global-store-mode connect "
        f"--global-store-address {shlex.quote(gs_addr)} "
        f"--set daemon_id={shlex.quote(worker.daemon_id)} "
        f"--set server.storage_path={shlex.quote(worker.storage)} "
        "--set server.listen.host=0.0.0.0 "
        f"--set server.listen.port={int(worker.grpc_port)} "
        "--set server.p2p_listen.host=0.0.0.0 "
        f"--set server.p2p_listen.port={int(worker.p2p_port)} "
        f"--set server.advertise.host={shlex.quote(worker.advertise_ip)} "
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
    run_remote(worker.process_id, cmd, timeout_sec=timeout_sec)

    # Ensure daemon process and GS connectivity are observable before traffic.
    wait_daemon_ready(worker=worker, timeout_sec=timeout_sec)


def assert_worker_has_gpu(worker: WorkerSpec, *, timeout_sec: float) -> None:
    """Fail fast when a worker does not expose a CUDA device."""
    inner_cmd = "set -euo pipefail; nvidia-smi -L 2>&1 || true"
    output = run_remote(
        worker.process_id,
        inner_cmd,
        timeout_sec=min(60.0, max(5.0, float(timeout_sec))),
    )
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    has_gpu = any(line.startswith("GPU ") for line in lines)
    if has_gpu:
        return
    tail = "\n".join(lines[-20:])
    raise RuntimeError(
        f"worker={worker.name} process_id={worker.process_id} has no detectable CUDA GPU; "
        f"nvidia-smi output:\n{tail}"
    )


def wait_daemon_ready(*, worker: WorkerSpec, timeout_sec: float) -> None:
    deadline = time.perf_counter() + max(1.0, float(timeout_sec))
    last_status: dict[str, Any] = {}
    while time.perf_counter() < deadline:
        cmd = (
            "set -euo pipefail; "
            f"export TENSORCAST_HOME={shlex.quote(worker.home)}; "
            f"cd {BENCH_ROOT}; "
            "source .venv/bin/activate; "
            "tensorcast-cli daemon status "
            f"--session {shlex.quote(worker.daemon_session)} --json"
        )
        try:
            status_out = run_remote(
                worker.process_id,
                cmd,
                timeout_sec=min(20.0, max(5.0, float(timeout_sec))),
            )
            left = status_out.find("{")
            right = status_out.rfind("}")
            if left < 0 or right <= left:
                raise RuntimeError("daemon status output missing json object")
            status_payload = json.loads(status_out[left : right + 1])
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
            continue

        last_status = status_payload
        daemon = status_payload.get("daemon", {})
        global_store = status_payload.get("global_store", {})
        daemon_addr = str(daemon.get("address") or "")
        p2p_addr = str(daemon.get("p2p_address") or "")
        gs_mode = str(global_store.get("mode") or "")
        gs_addr = str(global_store.get("address") or "")
        if daemon_addr and p2p_addr and gs_mode == "connect" and gs_addr:
            return
        time.sleep(0.5)

    raise RuntimeError(
        "daemon did not become ready before timeout "
        f"for worker={worker.name} session={worker.daemon_session}; "
        f"last_status={last_status}"
    )


def stop_daemon(worker: WorkerSpec, *, timeout_sec: float) -> dict[str, Any]:
    print(f"[stop] {worker.name} proc={worker.process_id}", flush=True)
    start_ts = time.perf_counter()
    cmd = (
        "set -euo pipefail; "
        f"export TENSORCAST_HOME={shlex.quote(worker.home)}; "
        f"cd {BENCH_ROOT}; "
        "source .venv/bin/activate; "
        "LD_LIBRARY_PATH=/data/cuda/compat tensorcast-cli daemon stop "
        f"--session {shlex.quote(worker.daemon_session)}"
    )
    output = run_remote(worker.process_id, cmd, timeout_sec=timeout_sec)
    return {
        "worker": worker.name,
        "elapsed_sec": float(time.perf_counter() - start_ts),
        "output_tail": output.strip().splitlines()[-1] if output.strip() else "",
    }


def run_put(
    *,
    worker: WorkerSpec,
    key: str,
    size_mib: int,
    seed: int,
    put_policy: str,
    tensor_count: int,
    dtype: str,
    put_device: str,
    timeout_sec: float,
) -> dict[str, Any]:
    print(f"[put] worker={worker.name} key={key}", flush=True)
    inner_cmd = (
        "set -euo pipefail; "
        f"cd {BENCH_ROOT}; "
        "source .venv/bin/activate; "
        "LD_LIBRARY_PATH=/data/cuda/compat "
        f"python {PUT_HELPER} "
        f"--daemon {shlex.quote(worker.daemon_addr)} "
        f"--key {shlex.quote(key)} "
        f"--size-mib {int(size_mib)} "
        f"--seed {int(seed)} "
        f"--put-policy {shlex.quote(put_policy)} "
        f"--tensor-count {int(tensor_count)} "
        f"--dtype {shlex.quote(dtype)} "
        f"--put-device {shlex.quote(put_device)}"
    )
    output = run_remote(worker.process_id, inner_cmd, timeout_sec=timeout_sec)
    payload = extract_last_json(output)
    payload["worker"] = worker.name
    return payload


def run_get(
    *,
    worker: WorkerSpec,
    key: str,
    artifact_id: str,
    lookup_mode: str,
    get_device: str,
    visibility_timeout_sec: float,
    visibility_retry_sec: float,
    payload_sample_verify: bool,
    timeout_sec: float,
) -> dict[str, Any]:
    print(
        f"[get] worker={worker.name} key={key} lookup={lookup_mode} device={get_device}",
        flush=True,
    )
    inner_cmd = (
        "set -euo pipefail; "
        f"cd {BENCH_ROOT}; "
        "source .venv/bin/activate; "
        "LD_LIBRARY_PATH=/data/cuda/compat "
        f"python {GET_HELPER} "
        f"--daemon {shlex.quote(worker.daemon_addr)} "
        f"--key {shlex.quote(key)} "
        f"--lookup-mode {shlex.quote(lookup_mode)} "
        f"--get-device {shlex.quote(get_device)} "
        f"--visibility-timeout-sec {float(visibility_timeout_sec)} "
        f"--visibility-retry-sec {float(visibility_retry_sec)} "
        "--prefer p2p "
        "--capture-comm-stats"
    )
    inner_cmd += (
        " --payload-sample-verify"
        if bool(payload_sample_verify)
        else " --no-payload-sample-verify"
    )
    if lookup_mode == "artifact_id":
        inner_cmd += f" --artifact-id {shlex.quote(artifact_id)}"
    output = run_remote(worker.process_id, inner_cmd, timeout_sec=timeout_sec)
    payload = extract_last_json(output)
    payload["worker"] = worker.name
    return payload


def run_deregister(
    worker: WorkerSpec,
    artifact_id: str,
    *,
    device_id: int | None,
    timeout_sec: float,
) -> dict[str, Any]:
    print(f"[deregister] worker={worker.name} artifact_id={artifact_id}", flush=True)
    inner_cmd = (
        "set -euo pipefail; "
        f"cd {BENCH_ROOT}; "
        "source .venv/bin/activate; "
        "LD_LIBRARY_PATH=/data/cuda/compat "
        f"python {DEREGISTER_HELPER} "
        f"--daemon {shlex.quote(worker.daemon_addr)} "
        f"--artifact-id {shlex.quote(artifact_id)}"
    )
    if device_id is not None:
        inner_cmd += f" --device-id {int(device_id)}"
    output = run_remote(worker.process_id, inner_cmd, timeout_sec=timeout_sec)
    payload = extract_last_json(output)
    payload["worker"] = worker.name
    return payload


def retire_source(
    *,
    worker: WorkerSpec,
    artifact_id: str,
    retire_mode: str,
    deregister_device_id: int | None,
    stop_timeout_sec: float,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "worker": worker.name,
        "mode": retire_mode,
        "artifact_id": artifact_id,
    }
    if retire_mode in {"deregister", "deregister_then_stop"}:
        dereg = run_deregister(
            worker,
            artifact_id,
            device_id=deregister_device_id,
            timeout_sec=stop_timeout_sec,
        )
        event["deregister"] = dereg
        if not bool(dereg.get("removed")):
            raise RuntimeError(
                f"source retire requires one-shot deregister success: {dereg}"
            )
    if retire_mode in {"stop", "deregister_then_stop"}:
        event["stop"] = stop_daemon(worker, timeout_sec=stop_timeout_sec)
    return event


def gs_query_one(
    db_file: str,
    sql: str,
    params: tuple[Any, ...],
) -> tuple[Any, ...] | None:
    try:
        con = duckdb.connect(database=db_file, read_only=True)
    except duckdb.IOException as exc:
        message = str(exc)
        if "Could not set lock on file" in message:
            raise GsDbUnavailableError(message) from exc
        raise
    try:
        row = con.execute(sql, params).fetchone()
        if row is None:
            return None
        return tuple(row)
    finally:
        con.close()


def gs_probe_replica(
    *,
    db_file: str,
    artifact_id: str,
    worker: WorkerSpec,
) -> dict[str, Any]:
    sql = """
        SELECT
          COALESCE(memory_type, '') AS memory_type,
          COALESCE(export_state, '') AS export_state,
          COALESCE(is_available, FALSE) AS is_available,
          COALESCE(device_id, -1) AS device_id,
          COALESCE(source_process_id, '') AS source_process_id
        FROM artifact_replicas
        WHERE artifact_id = ?
          AND node_address = ?
          AND node_port = ?
        ORDER BY updated_at DESC
        LIMIT 1
    """
    row = gs_query_one(
        db_file,
        sql,
        (str(artifact_id), str(worker.advertise_ip), int(worker.grpc_port)),
    )
    if row is None:
        return {
            "present": False,
            "memory_type": "",
            "export_state": "",
            "is_available": False,
            "device_id": -1,
            "source_process_id": "",
            "is_gpu_exportable": False,
        }
    memory_type = str(row[0] or "")
    export_state = str(row[1] or "")
    is_available = bool(row[2])
    return {
        "present": True,
        "memory_type": memory_type,
        "export_state": export_state,
        "is_available": is_available,
        "device_id": int(row[3]),
        "source_process_id": str(row[4] or ""),
        "is_gpu_exportable": (
            memory_type.upper() == "GPU"
            and export_state.upper() == "EXPORTABLE"
            and is_available
        ),
    }


def gs_probe_transport_source(
    *,
    db_file: str,
    artifact_id: str,
    target_worker: WorkerSpec,
) -> dict[str, Any]:
    sql = """
        SELECT
          COALESCE(t.source_node_id, '') AS source_node_id,
          COALESCE(t.source_address, '') AS source_address,
          COALESCE(t.source_port, 0) AS source_port,
          COALESCE(t.status, '') AS status
        FROM artifact_transports t
        JOIN artifact_replicas r
          ON t.replica_id = r.replica_id
        WHERE t.artifact_id = ?
          AND r.node_address = ?
          AND r.node_port = ?
        ORDER BY t.created_at DESC
        LIMIT 1
    """
    row = gs_query_one(
        db_file,
        sql,
        (
            str(artifact_id),
            str(target_worker.advertise_ip),
            int(target_worker.grpc_port),
        ),
    )
    if row is None:
        return {
            "present": False,
            "source_node_id": "",
            "source_address": "",
            "source_port": 0,
            "status": "",
        }
    return {
        "present": True,
        "source_node_id": str(row[0] or ""),
        "source_address": str(row[1] or ""),
        "source_port": int(row[2] or 0),
        "status": str(row[3] or ""),
    }


def restart_workers(
    *,
    workers: list[WorkerSpec],
    daemon_config: str,
    gs_addr: str,
    conn: int,
    buffers: int,
    maxw: int,
    expected_gpu_channels: int,
    timeout_sec: float,
) -> None:
    for worker in workers:
        assert_worker_has_gpu(worker, timeout_sec=timeout_sec)
        restart_daemon(
            worker=worker,
            daemon_config=daemon_config,
            gs_addr=gs_addr,
            conn=conn,
            buffers=buffers,
            maxw=maxw,
            expected_gpu_channels=expected_gpu_channels,
            timeout_sec=timeout_sec,
        )


def summarize_wave(
    *,
    gets: list[dict[str, Any]],
    wall_sec: float,
) -> dict[str, Any]:
    if not gets:
        return {
            "nodes": 0,
            "wall_sec": float(wall_sec),
            "e2e_gibps_mean": 0.0,
            "transfer_gibps_mean": 0.0,
            "cluster_gibps": 0.0,
            "p2p_ratio": 0.0,
            "comm_error_count": 0,
            "comm_bytes_mismatch_count": 0,
            "failed_count": 0,
            "failed_workers": [],
            "all_get_complete": True,
        }

    failed_workers = sorted(
        str(item.get("worker", "unknown")) for item in gets if bool(item.get("failed"))
    )
    successful = [item for item in gets if not bool(item.get("failed"))]
    if not successful:
        return {
            "nodes": len(gets),
            "wall_sec": float(wall_sec),
            "e2e_gibps_mean": 0.0,
            "transfer_gibps_mean": 0.0,
            "cluster_gibps": 0.0,
            "p2p_ratio": 0.0,
            "comm_error_count": 0,
            "comm_bytes_mismatch_count": 0,
            "failed_count": len(failed_workers),
            "failed_workers": failed_workers,
            "all_get_complete": False,
        }

    e2e_vals = [float(item["e2e_gibps"]) for item in successful]
    transfer_vals = [float(item["transfer_gibps"]) for item in successful]
    total_bytes = sum(int(item["total_bytes"]) for item in successful)
    p2p_count = sum(1 for item in successful if str(item.get("source")) == "p2p")
    comm_errors = sum(
        1 for item in successful if int(item.get("comm_errors_delta") or 0) > 0
    )
    comm_mismatch = sum(
        1
        for item in successful
        if int(item.get("comm_bytes_delta") or 0) != int(item.get("total_bytes") or 0)
    )

    cluster_gibps = (
        float(total_bytes) / float(1024**3) / float(wall_sec) if wall_sec > 0 else 0.0
    )
    return {
        "nodes": len(successful),
        "wall_sec": float(wall_sec),
        "e2e_gibps_mean": float(statistics.mean(e2e_vals)),
        "transfer_gibps_mean": float(statistics.mean(transfer_vals)),
        "cluster_gibps": float(cluster_gibps),
        "p2p_ratio": float(p2p_count / len(successful)),
        "comm_error_count": int(comm_errors),
        "comm_bytes_mismatch_count": int(comm_mismatch),
        "failed_count": len(failed_workers),
        "failed_workers": failed_workers,
        "all_get_complete": len(failed_workers) == 0,
    }


def run_wave_gets(
    *,
    workers: list[WorkerSpec],
    key: str,
    artifact_id: str,
    lookup_mode: str,
    get_device: str,
    visibility_timeout_sec: float,
    visibility_retry_sec: float,
    payload_sample_verify: bool,
    remote_timeout_sec: float,
) -> tuple[list[dict[str, Any]], float]:
    if not workers:
        return [], 0.0

    start_ts = time.perf_counter()
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(workers)) as pool:
        future_map = {
            pool.submit(
                run_get,
                worker=worker,
                key=key,
                artifact_id=artifact_id,
                lookup_mode=lookup_mode,
                get_device=get_device,
                visibility_timeout_sec=visibility_timeout_sec,
                visibility_retry_sec=visibility_retry_sec,
                payload_sample_verify=payload_sample_verify,
                timeout_sec=remote_timeout_sec,
            ): worker
            for worker in workers
        }
        for future in concurrent.futures.as_completed(future_map):
            worker = future_map[future]
            try:
                payload = future.result()
            except Exception as exc:  # noqa: BLE001
                results.append(
                    {
                        "worker": worker.name,
                        "failed": True,
                        "error": str(exc),
                        "final_error_code": "UNKNOWN",
                        "attempts": 0,
                    }
                )
                continue
            payload["failed"] = False
            results.append(payload)
    wall_sec = float(time.perf_counter() - start_ts)
    results.sort(key=lambda item: str(item.get("worker")))
    return results, wall_sec


def classify_failure(error_message: str) -> str:
    lowered = error_message.lower()
    infra_tokens = (
        "cannot exec into",
        "stopped",
        "connection refused",
        "connection reset",
        "timed out",
        "timeout",
        "temporary failure",
        "network",
    )
    product_tokens = (
        "invalid_argument",
        "failed_precondition",
        "not found",
        "not_found",
        "permission denied",
        "resource exhausted",
    )
    if any(token in lowered for token in infra_tokens):
        return "infra"
    if any(token in lowered for token in product_tokens):
        return "product"
    return "unknown"


def build_source_cardinality_entry(
    *,
    iteration: int,
    wave: str,
    wave_gets: list[dict[str, Any]],
    artifact_id: str,
    worker_by_name: dict[str, WorkerSpec],
    gs_db_file: str,
    db_probe_enabled: bool,
) -> dict[str, Any]:
    source_by_target: dict[str, str] = {}
    db_probe_errors: list[str] = []
    for item in wave_gets:
        if bool(item.get("failed")):
            continue
        worker_name = str(item.get("worker", ""))
        source_value = str(item.get("source", "unknown")) or "unknown"
        if db_probe_enabled and worker_name in worker_by_name:
            try:
                probe = gs_probe_transport_source(
                    db_file=gs_db_file,
                    artifact_id=artifact_id,
                    target_worker=worker_by_name[worker_name],
                )
                source_address = str(probe.get("source_address", ""))
                source_port = int(probe.get("source_port") or 0)
                if source_address:
                    source_value = f"{source_address}:{source_port}"
            except Exception as exc:  # noqa: BLE001
                db_probe_errors.append(f"{worker_name}: {exc}")
        source_by_target[worker_name] = source_value
    unique_sources = sorted(set(source_by_target.values()))
    return {
        "iter": int(iteration),
        "wave": wave,
        "unique_sources": unique_sources,
        "unique_source_count": int(len(unique_sources)),
        "source_by_target": source_by_target,
        "db_probe_enabled": bool(db_probe_enabled),
        "db_probe_errors": db_probe_errors,
    }


def aggregate_retry_reason_buckets(gets: list[dict[str, Any]]) -> dict[str, int]:
    buckets: dict[str, int] = {}
    for item in gets:
        raw = item.get("retry_reason_buckets", {})
        if not isinstance(raw, dict):
            continue
        for reason, value in raw.items():
            count = int(value or 0)
            buckets[str(reason)] = buckets.get(str(reason), 0) + count
    return buckets


def payload_hash_mismatches(
    *,
    expected_hash: str,
    gets: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    mismatches: list[str] = []
    missing: list[str] = []
    for item in gets:
        if bool(item.get("failed")):
            continue
        worker = str(item.get("worker", "unknown"))
        observed = str(item.get("payload_sample_hash", ""))
        if not observed:
            missing.append(worker)
            continue
        if expected_hash and observed != expected_hash:
            mismatches.append(worker)
    return sorted(mismatches), sorted(missing)


def run_cascade_mode(
    *,
    args: argparse.Namespace,
    seed: WorkerSpec,
    getters: list[WorkerSpec],
    deregister_device_id: int | None,
) -> dict[str, Any]:
    if len(getters) < 2:
        raise RuntimeError("cascade mode requires at least 2 get workers")

    run_tag = f"{int(time.time())}-{random_suffix(8)}"
    key = f"bench:fanout:cascade:{args.case_name}:{run_tag}:{uuid.uuid4().hex}"

    put = run_put(
        worker=seed,
        key=key,
        size_mib=int(args.size_mib),
        seed=int(args.put_seed_base),
        put_policy=str(args.put_policy),
        tensor_count=int(args.tensor_count),
        dtype=str(args.dtype),
        put_device=str(args.put_device),
        timeout_sec=float(args.remote_timeout_sec),
    )
    artifact_id = str(put.get("artifact_id", ""))
    if not artifact_id:
        raise RuntimeError("put returned empty artifact_id")

    hops: list[dict[str, Any]] = []
    retire_events: list[dict[str, Any]] = []
    cleanup: list[dict[str, Any]] = []
    db_probe_enabled = bool(str(args.gs_db_file))
    db_probe_disabled_reason = ""
    if db_probe_enabled:
        try:
            _ = gs_query_one(
                str(args.gs_db_file),
                "SELECT 1",
                (),
            )
        except GsDbUnavailableError as exc:
            db_probe_enabled = False
            db_probe_disabled_reason = f"duckdb_lock_conflict: {exc}"
            print(
                "[warn] disable GS DB probes; falling back to functional "
                f"single-source validation ({db_probe_disabled_reason})",
                flush=True,
            )

    has_db_probe = db_probe_enabled
    source_path_expected: list[str] = [seed.advertise_ip]
    source_path_observed: list[str] = []
    chain_source_match_failures: list[str] = []
    vram_exportable_failures: list[str] = []

    def replica_probe_or_empty(worker: WorkerSpec) -> dict[str, Any]:
        if not db_probe_enabled:
            return {}
        return gs_probe_replica(
            db_file=str(args.gs_db_file),
            artifact_id=artifact_id,
            worker=worker,
        )

    def transport_probe_or_empty(target_worker: WorkerSpec) -> dict[str, Any]:
        if not db_probe_enabled:
            return {}
        return gs_probe_transport_source(
            db_file=str(args.gs_db_file),
            artifact_id=artifact_id,
            target_worker=target_worker,
        )

    first = run_get(
        worker=getters[0],
        key=key,
        artifact_id=artifact_id,
        lookup_mode=str(args.lookup_mode),
        get_device=str(args.get_device),
        visibility_timeout_sec=float(args.visibility_timeout_sec),
        visibility_retry_sec=float(args.visibility_retry_sec),
        payload_sample_verify=bool(args.payload_sample_verify),
        timeout_sec=float(args.remote_timeout_sec),
    )
    first_replica = replica_probe_or_empty(getters[0])
    first_transport = transport_probe_or_empty(getters[0])
    expected_source = seed.advertise_ip
    observed_source = str(first_transport.get("source_address", ""))
    source_path_observed.append(observed_source)
    if observed_source and observed_source != expected_source:
        chain_source_match_failures.append(
            f"hop1 target={getters[0].name} expected={expected_source} observed={observed_source}"
        )
    hops.append(
        {
            "hop": 1,
            "target_worker": getters[0].name,
            "source": first.get("source"),
            "get": first,
            "target_replica_probe": first_replica,
            "transport_probe": first_transport,
            "expected_source_address": expected_source,
        }
    )
    retire_events.append(
        retire_source(
            worker=seed,
            artifact_id=artifact_id,
            retire_mode=str(args.source_retire_mode),
            deregister_device_id=deregister_device_id,
            stop_timeout_sec=float(args.stop_timeout_sec),
        )
    )
    if float(args.source_stop_settle_sec) > 0:
        print(
            f"[cascade] settle after retiring {seed.name}: "
            f"{float(args.source_stop_settle_sec):.1f}s",
            flush=True,
        )
        time.sleep(float(args.source_stop_settle_sec))

    prev_worker = getters[0]
    for idx in range(1, len(getters)):
        payload = run_get(
            worker=getters[idx],
            key=key,
            artifact_id=artifact_id,
            lookup_mode=str(args.lookup_mode),
            get_device=str(args.get_device),
            visibility_timeout_sec=float(args.visibility_timeout_sec),
            visibility_retry_sec=float(args.visibility_retry_sec),
            payload_sample_verify=bool(args.payload_sample_verify),
            timeout_sec=float(args.remote_timeout_sec),
        )
        prev_probe = replica_probe_or_empty(prev_worker)
        if prev_probe and not bool(prev_probe.get("is_gpu_exportable")):
            vram_exportable_failures.append(prev_worker.name)
        target_probe = replica_probe_or_empty(getters[idx])
        transport_probe = transport_probe_or_empty(getters[idx])
        expected_source = prev_worker.advertise_ip
        observed_source = str(transport_probe.get("source_address", ""))
        source_path_expected.append(expected_source)
        source_path_observed.append(observed_source)
        if observed_source and observed_source != expected_source:
            chain_source_match_failures.append(
                "hop"
                f"{idx + 1} target={getters[idx].name} "
                f"expected={expected_source} observed={observed_source}"
            )
        hops.append(
            {
                "hop": idx + 1,
                "target_worker": getters[idx].name,
                "source": payload.get("source"),
                "get": payload,
                "target_replica_probe": target_probe,
                "upstream_probe_before_stop": prev_probe,
                "transport_probe": transport_probe,
                "expected_source_address": expected_source,
            }
        )
        retire_events.append(
            retire_source(
                worker=prev_worker,
                artifact_id=artifact_id,
                retire_mode=str(args.source_retire_mode),
                deregister_device_id=deregister_device_id,
                stop_timeout_sec=float(args.stop_timeout_sec),
            )
        )
        if float(args.source_stop_settle_sec) > 0:
            print(
                f"[cascade] settle after retiring {prev_worker.name}: "
                f"{float(args.source_stop_settle_sec):.1f}s",
                flush=True,
            )
            time.sleep(float(args.source_stop_settle_sec))
        prev_worker = getters[idx]

    alive_workers = [prev_worker]
    for worker in alive_workers:
        try:
            cleanup.append(
                run_deregister(
                    worker,
                    artifact_id,
                    device_id=deregister_device_id,
                    timeout_sec=float(args.stop_timeout_sec),
                )
            )
        except Exception as exc:  # noqa: BLE001
            cleanup.append(
                {
                    "worker": worker.name,
                    "artifact_id": artifact_id,
                    "removed": False,
                    "error": str(exc),
                }
            )

    bad_sources = [
        hop["target_worker"] for hop in hops if str(hop.get("source")) != "p2p"
    ]
    comm_mismatch = [
        hop["target_worker"]
        for hop in hops
        if int(hop["get"].get("comm_bytes_delta") or 0)
        != int(hop["get"].get("total_bytes") or 0)
    ]
    comm_error_workers = [
        hop["target_worker"]
        for hop in hops
        if int(hop["get"].get("comm_errors_delta") or 0) > 0
    ]
    if has_db_probe and not vram_exportable_failures and hops:
        last_probe = hops[-1].get("target_replica_probe", {})
        if isinstance(last_probe, dict) and not bool(
            last_probe.get("is_gpu_exportable")
        ):
            vram_exportable_failures.append(str(hops[-1]["target_worker"]))
    functional_chain_ok = (
        len(hops) == len(getters)
        and not bad_sources
        and not comm_mismatch
        and not comm_error_workers
    )

    return {
        "mode": "cascade",
        "key": key,
        "artifact_id": artifact_id,
        "put": put,
        "hops": hops,
        "retire_events": retire_events,
        "cleanup": cleanup,
        "summary": {
            "hop_count": len(hops),
            "p2p_ratio": 1.0
            if not hops
            else float(
                sum(1 for hop in hops if str(hop.get("source")) == "p2p") / len(hops)
            ),
            "vram_validation_mode": (
                "db_probe" if has_db_probe else "functional_single_source"
            ),
            "db_probe_enabled": has_db_probe,
            "db_probe_disabled_reason": db_probe_disabled_reason,
            "single_source_chain_enforced": True,
            "functional_chain_ok": functional_chain_ok,
            "source_retire_mode": str(args.source_retire_mode),
            "source_stop_settle_sec": float(args.source_stop_settle_sec),
            "source_path_expected": source_path_expected,
            "source_path_observed": source_path_observed,
            "chain_source_match_failures": chain_source_match_failures,
            "vram_exportable_failures": sorted(set(vram_exportable_failures)),
            "bad_source_workers": bad_sources,
            "comm_bytes_mismatch_workers": comm_mismatch,
            "comm_error_workers": comm_error_workers,
        },
    }


def run_fanout_mode(
    *,
    args: argparse.Namespace,
    seed: WorkerSpec,
    getters: list[WorkerSpec],
    deregister_device_id: int | None,
) -> dict[str, Any]:
    if len(getters) < 2:
        raise RuntimeError("fanout mode requires at least 2 get workers")

    wave_size = int(args.wave_size)
    if wave_size <= 0:
        wave_size = max(1, len(getters) // 2)
    wave_size = min(max(1, wave_size), len(getters) - 1)

    wave1_workers = getters[:wave_size]
    wave2_workers = getters[wave_size:]
    total = int(args.warmup) + int(args.iterations)
    run_tag = f"{int(time.time())}-{random_suffix(8)}"
    worker_by_name = {worker.name: worker for worker in getters}

    db_probe_enabled = bool(str(args.gs_db_file))
    db_probe_disabled_reason = ""
    if db_probe_enabled:
        try:
            _ = gs_query_one(str(args.gs_db_file), "SELECT 1", ())
        except GsDbUnavailableError as exc:
            db_probe_enabled = False
            db_probe_disabled_reason = f"duckdb_lock_conflict: {exc}"
            print(
                "[warn] disable GS DB probes in fanout mode; "
                f"fallback to API-level source labels ({db_probe_disabled_reason})",
                flush=True,
            )

    records: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    source_cardinality_timeline: list[dict[str, Any]] = []
    for iteration in range(total):
        warmup = iteration < int(args.warmup)
        key = (
            f"bench:fanout:{args.case_name}:run{run_tag}:iter{iteration}:"
            f"{uuid.uuid4().hex}"
        )
        events.append(
            {
                "ts_epoch": float(time.time()),
                "iter": int(iteration),
                "type": "iteration_start",
                "warmup": bool(warmup),
                "key": key,
            }
        )
        put = run_put(
            worker=seed,
            key=key,
            size_mib=int(args.size_mib),
            seed=int(args.put_seed_base) + int(iteration),
            put_policy=str(args.put_policy),
            tensor_count=int(args.tensor_count),
            dtype=str(args.dtype),
            put_device=str(args.put_device),
            timeout_sec=float(args.remote_timeout_sec),
        )
        artifact_id = str(put.get("artifact_id", ""))
        if not artifact_id:
            raise RuntimeError(f"iter={iteration} put returned empty artifact_id")

        iter_start = time.perf_counter()
        wave1_gets, wave1_wall = run_wave_gets(
            workers=wave1_workers,
            key=key,
            artifact_id=artifact_id,
            lookup_mode=str(args.lookup_mode),
            get_device=str(args.get_device),
            visibility_timeout_sec=float(args.visibility_timeout_sec),
            visibility_retry_sec=float(args.visibility_retry_sec),
            payload_sample_verify=bool(args.payload_sample_verify),
            remote_timeout_sec=float(args.remote_timeout_sec),
        )
        wave2_gets, wave2_wall = run_wave_gets(
            workers=wave2_workers,
            key=key,
            artifact_id=artifact_id,
            lookup_mode=str(args.lookup_mode),
            get_device=str(args.get_device),
            visibility_timeout_sec=float(args.visibility_timeout_sec),
            visibility_retry_sec=float(args.visibility_retry_sec),
            payload_sample_verify=bool(args.payload_sample_verify),
            remote_timeout_sec=float(args.remote_timeout_sec),
        )
        iter_wall = float(time.perf_counter() - iter_start)
        all_wave_gets = [*wave1_gets, *wave2_gets]

        failed_gets = [item for item in all_wave_gets if bool(item.get("failed"))]
        classification_counts: dict[str, int] = {"infra": 0, "product": 0, "unknown": 0}
        for failed in failed_gets:
            worker = str(failed.get("worker", "unknown"))
            reason = str(failed.get("error", ""))
            failure_type = classify_failure(reason)
            classification_counts[failure_type] = (
                int(classification_counts.get(failure_type, 0)) + 1
            )
            events.append(
                {
                    "ts_epoch": float(time.time()),
                    "iter": int(iteration),
                    "type": "get_failed",
                    "worker": worker,
                    "classification": failure_type,
                    "error": reason,
                }
            )

        cleanup: list[dict[str, Any]] = []
        if bool(args.cleanup_artifacts):
            for worker in [seed, *getters]:
                try:
                    cleanup.append(
                        run_deregister(
                            worker,
                            artifact_id,
                            device_id=deregister_device_id,
                            timeout_sec=float(args.stop_timeout_sec),
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    cleanup.append(
                        {
                            "worker": worker.name,
                            "artifact_id": artifact_id,
                            "removed": False,
                            "error": str(exc),
                        }
                    )

        wave1_summary = summarize_wave(gets=wave1_gets, wall_sec=wave1_wall)
        wave2_summary = summarize_wave(gets=wave2_gets, wall_sec=wave2_wall)
        wave1_source_timeline = build_source_cardinality_entry(
            iteration=iteration,
            wave="wave1",
            wave_gets=wave1_gets,
            artifact_id=artifact_id,
            worker_by_name=worker_by_name,
            gs_db_file=str(args.gs_db_file),
            db_probe_enabled=db_probe_enabled,
        )
        wave2_source_timeline = build_source_cardinality_entry(
            iteration=iteration,
            wave="wave2",
            wave_gets=wave2_gets,
            artifact_id=artifact_id,
            worker_by_name=worker_by_name,
            gs_db_file=str(args.gs_db_file),
            db_probe_enabled=db_probe_enabled,
        )
        source_cardinality_timeline.extend(
            (wave1_source_timeline, wave2_source_timeline)
        )

        successful_gets = [
            item for item in all_wave_gets if not bool(item.get("failed"))
        ]
        cluster_total_bytes = sum(int(item["total_bytes"]) for item in successful_gets)
        cluster_gibps = (
            float(cluster_total_bytes) / float(1024**3) / float(iter_wall)
            if iter_wall > 0
            else 0.0
        )
        retry_reason_buckets = aggregate_retry_reason_buckets(all_wave_gets)
        payload_hash_expected = str(put.get("payload_sample_hash") or "")
        payload_hash_mismatch_workers, payload_hash_missing_workers = (
            payload_hash_mismatches(
                expected_hash=payload_hash_expected,
                gets=all_wave_gets,
            )
        )
        incomplete_workers = sorted(
            str(item.get("worker", "unknown")) for item in failed_gets
        )
        all_get_complete = len(incomplete_workers) == 0

        record = {
            "iter": int(iteration),
            "warmup": bool(warmup),
            "key": key,
            "artifact_id": artifact_id,
            "put": put,
            "wave1": {
                "workers": [worker.name for worker in wave1_workers],
                "gets": wave1_gets,
                "summary": wave1_summary,
            },
            "wave2": {
                "workers": [worker.name for worker in wave2_workers],
                "gets": wave2_gets,
                "summary": wave2_summary,
            },
            "iter_wall_sec": iter_wall,
            "cluster_gibps": cluster_gibps,
            "all_get_complete": bool(all_get_complete),
            "incomplete_workers": incomplete_workers,
            "retry_reason_buckets": retry_reason_buckets,
            "payload_hash_expected": payload_hash_expected,
            "payload_hash_mismatch_workers": payload_hash_mismatch_workers,
            "payload_hash_missing_workers": payload_hash_missing_workers,
            "source_cardinality_timeline": [
                wave1_source_timeline,
                wave2_source_timeline,
            ],
            "classification_counts": classification_counts,
            "cleanup": cleanup,
        }
        records.append(record)
        events.append(
            {
                "ts_epoch": float(time.time()),
                "iter": int(iteration),
                "type": "iteration_end",
                "warmup": bool(warmup),
                "all_get_complete": bool(all_get_complete),
                "incomplete_workers": incomplete_workers,
                "cluster_gibps": float(cluster_gibps),
            }
        )
        print(
            "  "
            f"iter={iteration:02d} warmup={warmup} "
            f"put={float(put.get('put_sec', 0.0)):.4f}s "
            f"w1_xfer={float(wave1_summary['transfer_gibps_mean']):.3f}GiB/s "
            f"w2_xfer={float(wave2_summary['transfer_gibps_mean']):.3f}GiB/s "
            f"cluster={float(cluster_gibps):.3f}GiB/s "
            f"all_get_complete={all_get_complete}",
            flush=True,
        )

    measured = [item for item in records if not bool(item["warmup"])]
    if not measured:
        raise RuntimeError("no measured records in fanout mode")

    all_gets = [
        payload
        for item in measured
        for payload in [*item["wave1"]["gets"], *item["wave2"]["gets"]]
    ]
    successful_gets = [item for item in all_gets if not bool(item.get("failed"))]
    bad_sources = sorted(
        {
            str(payload.get("worker"))
            for payload in successful_gets
            if str(payload.get("source")) != "p2p"
        }
    )
    if bool(args.require_p2p) and bad_sources:
        raise RuntimeError(f"measured records include non-p2p sources: {bad_sources}")

    put_vals = [float(item["put"]["put_sec"]) for item in measured]
    wave1_xfer_vals = [
        float(item["wave1"]["summary"]["transfer_gibps_mean"]) for item in measured
    ]
    wave2_xfer_vals = [
        float(item["wave2"]["summary"]["transfer_gibps_mean"]) for item in measured
    ]
    wave1_e2e_vals = [
        float(item["wave1"]["summary"]["e2e_gibps_mean"]) for item in measured
    ]
    wave2_e2e_vals = [
        float(item["wave2"]["summary"]["e2e_gibps_mean"]) for item in measured
    ]
    cluster_vals = [float(item["cluster_gibps"]) for item in measured]

    wave1_wall_vals = [float(item["wave1"]["summary"]["wall_sec"]) for item in measured]
    wave2_wall_vals = [float(item["wave2"]["summary"]["wall_sec"]) for item in measured]
    comm_error_count = sum(
        int(item["wave1"]["summary"]["comm_error_count"])
        + int(item["wave2"]["summary"]["comm_error_count"])
        for item in measured
    )
    comm_mismatch_count = sum(
        int(item["wave1"]["summary"]["comm_bytes_mismatch_count"])
        + int(item["wave2"]["summary"]["comm_bytes_mismatch_count"])
        for item in measured
    )
    payload_hash_mismatch_count = sum(
        len(item.get("payload_hash_mismatch_workers", [])) for item in measured
    )
    payload_hash_missing_count = sum(
        len(item.get("payload_hash_missing_workers", [])) for item in measured
    )
    all_get_complete = all(bool(item.get("all_get_complete")) for item in measured)
    incomplete_workers = sorted(
        {
            str(worker)
            for item in measured
            for worker in item.get("incomplete_workers", [])
        }
    )

    p2p_ratio = (
        float(
            sum(1 for payload in successful_gets if str(payload.get("source")) == "p2p")
        )
        / float(len(successful_gets))
        if successful_gets
        else 0.0
    )
    wave2_over_wave1 = (
        float(statistics.mean(wave2_xfer_vals))
        / float(statistics.mean(wave1_xfer_vals))
        if statistics.mean(wave1_xfer_vals) > 0
        else 0.0
    )
    total_gets_expected = int(len(getters) * len(measured))
    total_gets_success = int(len(successful_gets))
    get_success_rate = (
        float(total_gets_success) / float(total_gets_expected)
        if total_gets_expected > 0
        else 0.0
    )
    put_success_rate = 1.0
    retry_reason_buckets_total = aggregate_retry_reason_buckets(all_gets)
    budget_exit_reason_buckets: dict[str, int] = {}
    for payload in successful_gets:
        budget = payload.get("budget_trace", {})
        if not isinstance(budget, dict):
            continue
        exit_reason = str(budget.get("exit_reason") or "")
        if not exit_reason:
            continue
        budget_exit_reason_buckets[exit_reason] = (
            budget_exit_reason_buckets.get(exit_reason, 0) + 1
        )

    classification_totals: dict[str, int] = {"infra": 0, "product": 0, "unknown": 0}
    for item in measured:
        record_class = item.get("classification_counts", {})
        if not isinstance(record_class, dict):
            continue
        for label in ("infra", "product", "unknown"):
            classification_totals[label] = classification_totals.get(label, 0) + int(
                record_class.get(label, 0)
            )

    recover_time_sec = 0.0
    first_failure_idx: int | None = None
    for idx, item in enumerate(measured):
        if not bool(item.get("all_get_complete")):
            first_failure_idx = idx
            break
    if first_failure_idx is not None:
        for idx in range(first_failure_idx + 1, len(measured)):
            if bool(measured[idx].get("all_get_complete")):
                recover_time_sec = float(
                    sum(
                        float(row.get("iter_wall_sec", 0.0))
                        for row in measured[first_failure_idx + 1 : idx + 1]
                    )
                )
                break

    summary: dict[str, Any] = {
        "mode": "fanout",
        "case_name": str(args.case_name),
        "warmup": int(args.warmup),
        "iterations": int(args.iterations),
        "size_mib": int(args.size_mib),
        "conn": int(args.conn),
        "buffers": int(args.buffers),
        "maxw": int(args.maxw),
        "expected_gpu_channels": int(args.expected_gpu_channels),
        "getters": int(len(getters)),
        "wave1_getters": int(len(wave1_workers)),
        "wave2_getters": int(len(wave2_workers)),
        "put_sec_mean": float(statistics.mean(put_vals)),
        "put_sec_p90": percentile(put_vals, 0.9),
        "wave1_transfer_gibps_mean": float(statistics.mean(wave1_xfer_vals)),
        "wave1_transfer_gibps_p90": percentile(wave1_xfer_vals, 0.9),
        "wave2_transfer_gibps_mean": float(statistics.mean(wave2_xfer_vals)),
        "wave2_transfer_gibps_p90": percentile(wave2_xfer_vals, 0.9),
        "wave1_e2e_gibps_mean": float(statistics.mean(wave1_e2e_vals)),
        "wave2_e2e_gibps_mean": float(statistics.mean(wave2_e2e_vals)),
        "wave2_over_wave1_transfer_ratio": float(wave2_over_wave1),
        "cluster_gibps_mean": float(statistics.mean(cluster_vals)),
        "cluster_gibps_p90": percentile(cluster_vals, 0.9),
        "wave1_wall_sec_p90": percentile(wave1_wall_vals, 0.9),
        "wave2_wall_sec_p90": percentile(wave2_wall_vals, 0.9),
        "p2p_ratio": float(p2p_ratio),
        "all_get_complete": bool(all_get_complete),
        "incomplete_workers": incomplete_workers,
        "source_cardinality_timeline": source_cardinality_timeline,
        "recover_time_sec": float(recover_time_sec),
        "put_success_rate": float(put_success_rate),
        "get_success_rate": float(get_success_rate),
        "comm_bytes_delta": int(
            sum(int(item.get("comm_bytes_delta") or 0) for item in successful_gets)
        ),
        "comm_errors_delta": int(
            sum(int(item.get("comm_errors_delta") or 0) for item in successful_gets)
        ),
        "retry_reason_buckets": retry_reason_buckets_total,
        "budget_exit_reason_buckets": budget_exit_reason_buckets,
        "payload_hash_mismatch_count": int(payload_hash_mismatch_count),
        "payload_hash_missing_count": int(payload_hash_missing_count),
        "failure_classification_counts": classification_totals,
        "bad_source_workers": bad_sources,
        "comm_error_count": int(comm_error_count),
        "comm_bytes_mismatch_count": int(comm_mismatch_count),
        "db_probe_enabled": bool(db_probe_enabled),
        "db_probe_disabled_reason": db_probe_disabled_reason,
        "events_count": int(len(events)),
    }
    return {
        "mode": "fanout",
        "summary": summary,
        "records": records,
        "events": events,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-host fanout benchmark runner: validate cascade source promotion "
            "and measure wave-based scale-out throughput."
        )
    )
    parser.add_argument("--mode", choices=("cascade", "fanout"), required=True)
    parser.add_argument("--case-name", required=True)

    parser.add_argument("--seed-proc", required=True)
    parser.add_argument("--seed-adv-ip", required=True)
    parser.add_argument("--seed-daemon-addr", default="")

    parser.add_argument(
        "--get-procs", required=True, help="Comma-separated process ids"
    )
    parser.add_argument(
        "--get-adv-ips", required=True, help="Comma-separated advertise ips"
    )
    parser.add_argument(
        "--get-daemon-addrs",
        default="",
        help="Comma-separated daemon addresses; default uses <adv_ip>:<seed-grpc-port>",
    )

    parser.add_argument("--seed-grpc-port", type=int, default=62001)
    parser.add_argument("--seed-p2p-port", type=int, default=63001)
    parser.add_argument("--get-grpc-port", type=int, default=62001)
    parser.add_argument("--get-p2p-port", type=int, default=63001)
    parser.add_argument(
        "--get-port-step",
        type=int,
        default=1,
        help=(
            "Per-getter port increment to avoid collisions when multiple workers "
            "share one host; getter i uses base_port + i*step"
        ),
    )
    parser.add_argument("--gs-addr", required=True)

    parser.add_argument("--conn", type=int, required=True)
    parser.add_argument("--buffers", type=int, required=True)
    parser.add_argument("--maxw", type=int, required=True)
    parser.add_argument("--expected-gpu-channels", type=int, default=0)
    parser.add_argument("--daemon-config", default="")

    parser.add_argument("--size-mib", type=int, default=1024)
    parser.add_argument(
        "--put-seed-base",
        type=int,
        default=1000,
        help="Base random seed for payload generation; fanout iter i uses base+i",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--wave-size", type=int, default=0)
    parser.add_argument("--tensor-count", type=int, default=1)
    parser.add_argument(
        "--dtype", choices=("float16", "float32", "bfloat16"), default="float16"
    )
    parser.add_argument("--lookup-mode", choices=("key", "artifact_id"), default="key")
    parser.add_argument("--put-policy", default="pinned")
    parser.add_argument("--put-device", default="cuda:0")
    parser.add_argument("--get-device", default="cuda:0")
    parser.add_argument(
        "--payload-sample-verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable payload sample hash checks in get helper. "
            "Disable for pure performance-only phases."
        ),
    )
    parser.add_argument("--deregister-device-id", type=int, default=None)
    parser.add_argument("--visibility-timeout-sec", type=float, default=30.0)
    parser.add_argument("--visibility-retry-sec", type=float, default=0.05)
    parser.add_argument("--remote-timeout-sec", type=float, default=900.0)
    parser.add_argument("--daemon-start-timeout-sec", type=float, default=600.0)
    parser.add_argument("--stop-timeout-sec", type=float, default=180.0)
    parser.add_argument(
        "--source-stop-settle-sec",
        type=float,
        default=0.0,
        help=(
            "Cascade only: wait after stopping previous source before next hop get; "
            "used to allow GS heartbeat-based source filtering to converge"
        ),
    )
    parser.add_argument(
        "--source-retire-mode",
        choices=("deregister", "stop", "deregister_then_stop"),
        default="deregister",
        help=(
            "Cascade only: how to retire each previous source before next hop. "
            "'deregister' enforces one-shot replica removal without killing daemon."
        ),
    )
    parser.add_argument(
        "--gs-db-file",
        default="",
        help="Optional Global Store duckdb file for transport/replica source validation",
    )
    parser.add_argument(
        "--require-vram-source", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--require-p2p", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--cleanup-artifacts", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--out-dir", default="/tmp/tc_cross_20260221/results_multi_host"
    )
    return parser.parse_args()


def build_workers(args: argparse.Namespace) -> tuple[WorkerSpec, list[WorkerSpec]]:
    get_procs = split_csv(str(args.get_procs))
    get_adv_ips = split_csv(str(args.get_adv_ips))
    if len(get_procs) != len(get_adv_ips):
        raise RuntimeError("get-procs count must match get-adv-ips count")
    if len(get_procs) < 1:
        raise RuntimeError("at least one get worker is required")

    get_daemon_addrs_input = split_csv(str(args.get_daemon_addrs))
    if get_daemon_addrs_input and len(get_daemon_addrs_input) != len(get_procs):
        raise RuntimeError("get-daemon-addrs count must match get-procs count")
    if int(args.get_port_step) < 0:
        raise RuntimeError("get-port-step must be >= 0")
    case_token = hashlib.sha1(str(args.case_name).encode("utf-8")).hexdigest()[:10]
    case_root = f"/tmp/tc_cross_20260221/{case_token}"

    seed_daemon_addr = (
        str(args.seed_daemon_addr)
        if str(args.seed_daemon_addr).strip()
        else default_daemon_addr(str(args.seed_adv_ip), int(args.seed_grpc_port))
    )
    seed = WorkerSpec(
        name="seed",
        process_id=str(args.seed_proc),
        daemon_addr=seed_daemon_addr,
        advertise_ip=str(args.seed_adv_ip),
        grpc_port=int(args.seed_grpc_port),
        p2p_port=int(args.seed_p2p_port),
        daemon_session=f"tc-fanout-{args.case_name}-seed",
        daemon_id=f"tc-fanout-{args.case_name}-seed",
        home=f"{case_root}/seed_h",
        storage=f"{case_root}/seed_s",
    )

    getters: list[WorkerSpec] = []
    for idx, (proc, adv_ip) in enumerate(zip(get_procs, get_adv_ips, strict=True)):
        grpc_port = int(args.get_grpc_port) + idx * int(args.get_port_step)
        p2p_port = int(args.get_p2p_port) + idx * int(args.get_port_step)
        daemon_addr = (
            get_daemon_addrs_input[idx]
            if get_daemon_addrs_input
            else default_daemon_addr(adv_ip, grpc_port)
        )
        getters.append(
            WorkerSpec(
                name=f"get{idx + 1}",
                process_id=str(proc),
                daemon_addr=daemon_addr,
                advertise_ip=str(adv_ip),
                grpc_port=grpc_port,
                p2p_port=p2p_port,
                daemon_session=f"tc-fanout-{args.case_name}-get{idx + 1}",
                daemon_id=f"tc-fanout-{args.case_name}-get{idx + 1}",
                home=f"{case_root}/g{idx + 1}_h",
                storage=f"{case_root}/g{idx + 1}_s",
            )
        )
    return seed, getters


def infer_deregister_device_id(args: argparse.Namespace) -> int | None:
    if args.deregister_device_id is not None:
        return int(args.deregister_device_id)
    get_device = str(args.get_device).strip().lower()
    if get_device.startswith("cuda:"):
        suffix = get_device.split(":", 1)[1]
        if suffix.isdigit():
            return int(suffix)
    return None


def main() -> int:
    args = parse_args()
    out_dir = Path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    seed, getters = build_workers(args)
    deregister_device_id = infer_deregister_device_id(args)
    print(
        f"[case] {args.case_name} mode={args.mode} seed={seed.process_id} "
        f"getters={len(getters)} conn={args.conn} buffers={args.buffers} maxw={args.maxw} "
        f"egc={args.expected_gpu_channels} size_mib={args.size_mib}",
        flush=True,
    )

    restart_workers(
        workers=[seed, *getters],
        daemon_config=str(args.daemon_config),
        gs_addr=str(args.gs_addr),
        conn=int(args.conn),
        buffers=int(args.buffers),
        maxw=int(args.maxw),
        expected_gpu_channels=int(args.expected_gpu_channels),
        timeout_sec=float(args.daemon_start_timeout_sec),
    )

    if str(args.mode) == "cascade":
        payload = run_cascade_mode(
            args=args,
            seed=seed,
            getters=getters,
            deregister_device_id=deregister_device_id,
        )
        summary = payload["summary"]
        if bool(args.require_p2p) and summary["bad_source_workers"]:
            raise RuntimeError(
                f"cascade includes non-p2p hops: {summary['bad_source_workers']}"
            )
        if bool(args.require_vram_source):
            validation_mode = str(summary.get("vram_validation_mode", ""))
            if validation_mode == "db_probe":
                if summary["vram_exportable_failures"]:
                    raise RuntimeError(
                        "cascade source workers without gpu exportable replica: "
                        f"{summary['vram_exportable_failures']}"
                    )
                if summary["chain_source_match_failures"]:
                    raise RuntimeError(
                        "cascade transport source mismatch: "
                        f"{summary['chain_source_match_failures']}"
                    )
            elif not bool(summary.get("functional_chain_ok")):
                raise RuntimeError(
                    "cascade functional single-source validation failed: "
                    "expected p2p success for every hop under enforced source chain"
                )
    else:
        payload = run_fanout_mode(
            args=args,
            seed=seed,
            getters=getters,
            deregister_device_id=deregister_device_id,
        )
        summary = payload["summary"]

    out_path = out_dir / f"{args.case_name}.json"
    payload["generated_at_epoch"] = time.time()
    payload["params"] = {
        "mode": str(args.mode),
        "seed_proc": str(args.seed_proc),
        "seed_daemon_addr": str(seed.daemon_addr),
        "get_procs": [worker.process_id for worker in getters],
        "get_daemon_addrs": [worker.daemon_addr for worker in getters],
        "get_grpc_ports": [int(worker.grpc_port) for worker in getters],
        "get_p2p_ports": [int(worker.p2p_port) for worker in getters],
        "get_port_step": int(args.get_port_step),
        "gs_addr": str(args.gs_addr),
        "daemon_config": str(args.daemon_config),
        "conn": int(args.conn),
        "buffers": int(args.buffers),
        "maxw": int(args.maxw),
        "expected_gpu_channels": int(args.expected_gpu_channels),
        "size_mib": int(args.size_mib),
        "put_seed_base": int(args.put_seed_base),
        "lookup_mode": str(args.lookup_mode),
        "payload_sample_verify": bool(args.payload_sample_verify),
        "gs_db_file": str(args.gs_db_file),
        "source_stop_settle_sec": float(args.source_stop_settle_sec),
        "deregister_device_id": deregister_device_id,
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("SUMMARY", json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"OUTPUT {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
