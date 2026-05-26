#!/usr/bin/env python
#  Copyright (c) 2026, TensorCast Team.

# Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import pwd
import random
import re
import shlex
import statistics
import string
import subprocess
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import grpc
from google.protobuf import timestamp_pb2

from tensorcast.global_store.cluster_runtime_rpc import call_cluster_runtime_rpc
from tensorcast.proto.global_store.v1 import global_store_pb2, global_store_pb2_grpc

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_ROOT = REPO_ROOT.as_posix()
SCRIPT_DIR = (REPO_ROOT / "examples" / "cross_host").resolve()
PUT_HELPER = (SCRIPT_DIR / "cross_host_put_once.py").as_posix()
GET_HELPER = (SCRIPT_DIR / "cross_host_get_once.py").as_posix()
DEREGISTER_HELPER = (SCRIPT_DIR / "cross_host_deregister_once.py").as_posix()
DETAILED_STATUS_HELPER = (SCRIPT_DIR / "cross_host_detailed_status_once.py").as_posix()
REMOTE_USER_RE = re.compile(r"^[a-z_][a-z0-9_.-]{0,63}$")
_REMOTE_RUN_AS_USER = ""
DAEMON_START_MAX_ATTEMPTS = 4
DAEMON_START_RETRY_BACKOFF_SEC = 2.0


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


def normalize_non_root_user(raw: str) -> str:
    value = str(raw).strip()
    if not value:
        raise ValueError("resolved empty workspace user for remote execution")
    if not REMOTE_USER_RE.fullmatch(value):
        raise ValueError(f"invalid workspace user for remote execution: {value!r}")
    if value == "root":
        raise ValueError(
            "workspace user resolved to root; refusing remote execution as root"
        )
    return value


def resolve_workspace_user() -> str:
    env_user = str(os.environ.get("USER", "")).strip()
    if env_user:
        try:
            return normalize_non_root_user(env_user)
        except ValueError:
            pass
    try:
        uid_user = pwd.getpwuid(os.getuid()).pw_name
    except KeyError as exc:
        raise RuntimeError(
            f"failed to resolve workspace user from uid={os.getuid()}"
        ) from exc
    return normalize_non_root_user(uid_user)


def configure_remote_run_as_user(run_as_user: str) -> str:
    normalized = normalize_non_root_user(run_as_user)
    global _REMOTE_RUN_AS_USER
    _REMOTE_RUN_AS_USER = normalized
    return normalized


def _resolved_remote_run_as_user() -> str:
    run_as_user = str(_REMOTE_RUN_AS_USER).strip()
    if not run_as_user:
        raise RuntimeError("remote run-as user is not configured")
    return run_as_user


def _wrap_remote_inner_cmd_for_user(
    *,
    inner_cmd: str,
    run_as_user: str,
) -> str:
    quoted_inner = shlex.quote(str(inner_cmd))
    quoted_user = shlex.quote(str(run_as_user))
    return (
        "set -euo pipefail; "
        f"run_as_user={quoted_user}; "
        'if ! getent passwd "${run_as_user}" >/dev/null 2>&1; then '
        'echo "remote run-as user not found: ${run_as_user}" >&2; '
        "exit 97; "
        "fi; "
        'if [[ "$(id -un)" == "${run_as_user}" ]]; then '
        f"bash -lc {quoted_inner}; "
        "else "
        f'su - "${{run_as_user}}" -s /bin/bash -c {quoted_inner}; '
        "fi"
    )


def _tail_text(raw: str | None, *, max_chars: int = 4000) -> str:
    if raw is None:
        return ""
    text = str(raw)
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _extract_section(text: str, *, begin: str, end: str) -> str:
    start = text.find(begin)
    if start < 0:
        return ""
    start += len(begin)
    stop = text.find(end, start)
    if stop < 0:
        stop = len(text)
    return text[start:stop]


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
        stdout_tail = _tail_text(proc.stdout)
        stderr_tail = _tail_text(proc.stderr)
        raise RuntimeError(
            "command failed "
            f"(rc={proc.returncode}, timeout_sec={timeout_sec}): {cmd}\n"
            f"[stdout tail]\n{stdout_tail}\n"
            f"[stderr tail]\n{stderr_tail}"
        )
    return proc.stdout


def run_remote(
    process_id: str,
    inner_cmd: str,
    *,
    timeout_sec: float,
) -> str:
    wrapped_cmd = _wrap_remote_inner_cmd_for_user(
        inner_cmd=inner_cmd,
        run_as_user=_resolved_remote_run_as_user(),
    )
    cmd = (
        f"orchestratorctl exec process/{process_id} -n tensorcast -- bash -lc "
        f"{shlex.quote(wrapped_cmd)}"
    )
    return run(cmd, timeout_sec=timeout_sec)


def verify_remote_run_as_user(
    process_id: str,
    *,
    timeout_sec: float,
) -> None:
    target = _resolved_remote_run_as_user()
    output = run_remote(
        process_id,
        "set -euo pipefail; id -un",
        timeout_sec=min(20.0, max(5.0, float(timeout_sec))),
    )
    current = output.strip().splitlines()[-1].strip() if output.strip() else ""
    if current != target:
        raise RuntimeError(
            "remote run-as verification failed: "
            f"process={process_id} current_user={current!r} target={target!r}"
        )


def collect_worker_failure_probe(
    *,
    worker: WorkerSpec,
    timeout_sec: float,
) -> dict[str, Any]:
    probe: dict[str, Any] = {
        "process_id": worker.process_id,
        "daemon_addr": worker.daemon_addr,
        "daemon_session": worker.daemon_session,
        "daemon_id": worker.daemon_id,
        "advertise_ip": worker.advertise_ip,
    }
    per_call_timeout = max(5.0, min(60.0, float(timeout_sec)))

    get_cmd = (
        f"orchestratorctl get process {worker.process_id} -n tensorcast --no-headers"
    )
    try:
        probe["orchestratorctl_get_process"] = _tail_text(
            run(get_cmd, timeout_sec=per_call_timeout)
        )
    except Exception as exc:  # noqa: BLE001
        probe["orchestratorctl_get_process_error"] = str(exc)

    describe_cmd = f"orchestratorctl describe process/{worker.process_id} -n tensorcast | sed -n '1,140p'"
    try:
        probe["orchestratorctl_describe_head"] = _tail_text(
            run(describe_cmd, timeout_sec=per_call_timeout),
            max_chars=6000,
        )
    except Exception as exc:  # noqa: BLE001
        probe["orchestratorctl_describe_error"] = str(exc)

    remote_probe_cmd = (
        "set -euo pipefail; "
        'echo "__tc_probe_user__=$(id -un)"; '
        "nvidia-smi -L | head -n 8 || true; "
        f"export TENSORCAST_HOME={shlex.quote(worker.home)}; "
        f"cd {BENCH_ROOT}; "
        "source .venv/bin/activate; "
        'status_json="$(mktemp)"; '
        "LD_LIBRARY_PATH=/data/cuda/compat tensorcast-cli daemon status "
        f'--session {shlex.quote(worker.daemon_session)} --json >"${{status_json}}" || true; '
        'cat "${status_json}" || true; '
        'logs_dir="$(sed -n \'s/.*\\"logs_dir\\"[[:space:]]*:[[:space:]]*\\"\\([^\\"]*\\)\\".*/\\1/p\' "${status_json}" | head -n 1)"; '
        'if [[ -n "${logs_dir}" ]]; then '
        'echo "__tc_daemon_logs_tail_begin__"; '
        "for log_name in daemon.err daemon.INFO daemon.log daemon.out; do "
        'if [[ -f "${logs_dir}/${log_name}" ]]; then '
        'echo "__tc_daemon_log_file__=${log_name}"; '
        'tail -n 240 "${logs_dir}/${log_name}" 2>/dev/null || true; '
        "fi; "
        "done; "
        'echo "__tc_daemon_logs_tail_end__"; '
        "fi; "
        'rm -f "${status_json}"'
    )
    try:
        remote_out = run_remote(
            worker.process_id,
            remote_probe_cmd,
            timeout_sec=per_call_timeout,
        )
        probe["remote_probe_tail"] = _tail_text(remote_out, max_chars=20000)
        daemon_logs_excerpt = _extract_section(
            remote_out,
            begin="__tc_daemon_logs_tail_begin__",
            end="__tc_daemon_logs_tail_end__",
        ).strip()
        if daemon_logs_excerpt:
            compact_logs_excerpt = _tail_text(
                daemon_logs_excerpt,
                max_chars=12000,
            )
            probe["daemon_logs_excerpt"] = compact_logs_excerpt
            # Backward-compatible key retained for downstream parsers.
            probe["daemon_err_excerpt"] = compact_logs_excerpt
    except Exception as exc:  # noqa: BLE001
        probe["remote_probe_error"] = str(exc)

    return probe


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
    promotion_max_concurrency: int,
    timeout_sec: float,
) -> None:
    print(
        f"[restart] {worker.name} proc={worker.process_id} "
        f"grpc={worker.grpc_port} p2p={worker.p2p_port}",
        flush=True,
    )
    config_flag = f"-c {daemon_config}" if daemon_config else ""
    promo_concurrency_set = ""
    if int(promotion_max_concurrency) > 0:
        promo_concurrency_set = (
            f" --set promotion.max_concurrency={int(promotion_max_concurrency)}"
        )
    pre_start_cmd = (
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
    )
    start_cmd = (
        "set -euo pipefail; "
        f"cd {BENCH_ROOT}; "
        "source .venv/bin/activate; "
        f"mkdir -p {shlex.quote(worker.home)} {shlex.quote(worker.storage)}; "
        f"export TENSORCAST_HOME={shlex.quote(worker.home)}; "
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
        f"{promo_concurrency_set} "
        "--set observability.logging.level=warn "
        "--json"
    )
    last_error: RuntimeError | None = None
    for attempt in range(1, DAEMON_START_MAX_ATTEMPTS + 1):
        run_remote(worker.process_id, pre_start_cmd, timeout_sec=timeout_sec)
        try:
            run_remote(worker.process_id, start_cmd, timeout_sec=timeout_sec)
            last_error = None
            break
        except RuntimeError as exc:
            if (
                not _is_transient_daemon_start_error(exc)
                or attempt >= DAEMON_START_MAX_ATTEMPTS
            ):
                raise
            last_error = exc
            backoff_sec = DAEMON_START_RETRY_BACKOFF_SEC * float(attempt)
            compact_error = (
                _tail_text(str(exc), max_chars=320)
                .replace("\r", " ")
                .replace("\n", " | ")
            )
            print(
                "[restart] transient daemon start failure "
                f"worker={worker.name} attempt={attempt}/{DAEMON_START_MAX_ATTEMPTS} "
                f"backoff={backoff_sec:.1f}s error={compact_error}",
                flush=True,
            )
            time.sleep(backoff_sec)
    if last_error is not None:
        raise last_error

    # Ensure daemon process and GS connectivity are observable before traffic.
    wait_daemon_ready(worker=worker, timeout_sec=timeout_sec)


def _is_transient_daemon_start_error(error: Exception) -> bool:
    message = str(error).lower()
    markers = (
        "global store connect mode requires a reachable address",
        "failed to connect to global store",
        "statuscode.unavailable",
        "statuscode.deadline_exceeded",
        "grpc_status:14",
        "grpc_status:4",
        "deadline exceeded",
        "timed out",
    )
    return any(marker in message for marker in markers)


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
            try:
                _probe_daemon_global_store_connectivity(
                    worker=worker, timeout_sec=timeout_sec
                )
                return
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
                continue
        time.sleep(0.5)

    raise RuntimeError(
        "daemon did not become ready before timeout "
        f"for worker={worker.name} session={worker.daemon_session}; "
        f"last_status={last_status}"
    )


def _probe_daemon_global_store_connectivity(
    *,
    worker: WorkerSpec,
    timeout_sec: float,
) -> None:
    probe_key = f"__tc_probe__:{uuid.uuid4().hex}"
    probe_py = f"""
from tensorcast.daemon_ctl import DaemonCtl as _D

client = _D(server_address={worker.daemon_addr!r})
try:
    client.resolve_key_mapping({probe_key!r}, timeout_s=5.0)
except Exception as exc:  # noqa: BLE001
    message = str(exc).lower()
    if (
        "key not found" in message
        or "statuscode.not_found" in message
        or "grpc_status:5" in message
    ):
        pass
    else:
        raise
print("gs-ready")
"""
    probe_cmd = (
        "set -euo pipefail; "
        f"cd {BENCH_ROOT}; "
        "source .venv/bin/activate; "
        f"python -c {shlex.quote(probe_py)}"
    )
    run_remote(
        worker.process_id,
        probe_cmd,
        timeout_sec=min(30.0, max(5.0, float(timeout_sec))),
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


def query_daemon_status_best_effort(
    worker: WorkerSpec,
    *,
    timeout_sec: float,
) -> dict[str, Any]:
    cmd = (
        "set -euo pipefail; "
        f"export TENSORCAST_HOME={shlex.quote(worker.home)}; "
        f"cd {BENCH_ROOT}; "
        "source .venv/bin/activate; "
        "tensorcast-cli daemon status "
        f"--session {shlex.quote(worker.daemon_session)} --json || true"
    )
    try:
        output = run_remote(
            worker.process_id,
            cmd,
            timeout_sec=min(20.0, max(5.0, float(timeout_sec))),
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "worker": worker.name,
            "session": worker.daemon_session,
            "status_found": False,
            "stopped": True,
            "error": str(exc),
        }

    parsed: dict[str, Any] | None = None
    try:
        parsed = extract_last_json(output)
    except Exception:  # noqa: BLE001
        parsed = None

    daemon_payload = parsed.get("daemon", {}) if isinstance(parsed, dict) else {}
    has_daemon = bool(isinstance(daemon_payload, dict) and daemon_payload.get("pid"))
    return {
        "worker": worker.name,
        "session": worker.daemon_session,
        "status_found": bool(parsed is not None),
        "stopped": not has_daemon,
        "status": parsed,
    }


def teardown_workers(
    *,
    workers: list[WorkerSpec],
    stop_timeout_sec: float,
    verify_timeout_sec: float,
) -> dict[str, Any]:
    stop_results: list[dict[str, Any]] = []
    verify_results: list[dict[str, Any]] = []
    errors: list[str] = []
    for worker in workers:
        try:
            stop_results.append(
                stop_daemon(worker, timeout_sec=float(stop_timeout_sec))
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"stop worker={worker.name}: {exc}")
            stop_results.append(
                {
                    "worker": worker.name,
                    "elapsed_sec": 0.0,
                    "output_tail": "",
                    "error": str(exc),
                }
            )
        verify_results.append(
            query_daemon_status_best_effort(
                worker,
                timeout_sec=float(verify_timeout_sec),
            )
        )
    all_stopped = all(bool(item.get("stopped")) for item in verify_results)
    if not all_stopped:
        still_running = [
            str(item.get("worker", "unknown"))
            for item in verify_results
            if not bool(item.get("stopped"))
        ]
        errors.append(f"daemon session still running after teardown: {still_running}")
    return {
        "attempted": True,
        "all_stopped": bool(all_stopped),
        "stop_results": stop_results,
        "verify_results": verify_results,
        "errors": errors,
    }


def run_detailed_status(
    worker: WorkerSpec,
    *,
    timeout_sec: float,
) -> dict[str, Any]:
    inner_cmd = (
        "set -euo pipefail; "
        f"cd {BENCH_ROOT}; "
        "source .venv/bin/activate; "
        "LD_LIBRARY_PATH=/data/cuda/compat "
        f"python {DETAILED_STATUS_HELPER} "
        f"--daemon {shlex.quote(worker.daemon_addr)}"
    )
    output = run_remote(worker.process_id, inner_cmd, timeout_sec=timeout_sec)
    payload = extract_last_json(output)
    payload["worker"] = worker.name
    payload["process_id"] = worker.process_id
    payload["daemon_addr"] = worker.daemon_addr
    return payload


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
    payload["process_id"] = worker.process_id
    payload["daemon_addr"] = worker.daemon_addr
    return payload


def run_get(
    *,
    worker: WorkerSpec,
    key: str,
    artifact_id: str,
    lookup_mode: str,
    get_device: str,
    pinned_allocation_timeout_ms: int,
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
        f"--pinned-allocation-timeout-ms {int(pinned_allocation_timeout_ms)} "
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
    payload["process_id"] = worker.process_id
    payload["daemon_addr"] = worker.daemon_addr
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


def _to_utc_sql_timestamp(ts: datetime) -> str:
    aware_ts = ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
    return aware_ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f+00:00")


def _to_proto_timestamp(ts: datetime) -> timestamp_pb2.Timestamp:
    aware_ts = ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
    proto_ts = timestamp_pb2.Timestamp()
    proto_ts.FromDatetime(aware_ts.astimezone(timezone.utc))
    return proto_ts


def query_transport_rows_via_gs_rpc(
    *,
    gs_addr: str,
    started_at_utc: datetime,
    finished_at_utc: datetime,
    limit: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "enabled": True,
        "audit_method": "gs_rpc",
        "gs_addr": str(gs_addr),
        "window_start_utc": _to_utc_sql_timestamp(started_at_utc),
        "window_end_utc": _to_utc_sql_timestamp(finished_at_utc),
        "error": None,
        "row_count": 0,
        "rows": [],
    }
    if not str(gs_addr).strip():
        payload["error"] = "gs_addr is empty"
        return payload

    request = global_store_pb2.QueryTransportWindowRequest(
        created_at_start=_to_proto_timestamp(started_at_utc),
        created_at_end=_to_proto_timestamp(finished_at_utc),
        limit=max(1, int(limit)),
    )
    response, rpc_error = call_cluster_runtime_rpc(
        gs_addr=str(gs_addr),
        ready_timeout_sec=5.0,
        rpc_name="QueryTransportWindow",
        grpc_module=grpc,
        stub_factory=global_store_pb2_grpc.ClusterRuntimeServiceStub,
        rpc_call=lambda stub: stub.QueryTransportWindow(request, timeout=20.0),
    )
    if rpc_error is not None:
        payload["error"] = rpc_error
        return payload
    if response is None:
        payload["error"] = "QueryTransportWindow returned empty response"
        return payload

    if response.status != global_store_pb2.Status.STATUS_OK:
        payload["error"] = (
            f"QueryTransportWindow returned non-OK status: {int(response.status)}"
        )
        return payload

    rows: list[dict[str, Any]] = []
    for row in response.rows:
        created_dt = (
            row.created_at.ToDatetime(tzinfo=timezone.utc)
            if row.HasField("created_at")
            else None
        )
        completed_dt = (
            row.completed_at.ToDatetime(tzinfo=timezone.utc)
            if row.HasField("completed_at")
            else None
        )
        rows.append(
            {
                "transport_id": str(row.transport_id),
                "replica_id": str(row.replica_id),
                "artifact_id": str(row.artifact_id),
                "status": str(row.status),
                "completion_outcome": str(row.completion_outcome),
                "request_id": str(row.request_id),
                "requester_worker_id": str(row.requester_worker_id),
                "group_id": str(row.group_id),
                "group_kind": str(row.group_kind),
                "group_part_id": str(row.group_part_id),
                "group_total_parts": int(row.group_total_parts),
                "created_at_utc": (
                    created_dt.astimezone(timezone.utc).isoformat()
                    if created_dt is not None
                    else ""
                ),
                "completed_at_utc": (
                    completed_dt.astimezone(timezone.utc).isoformat()
                    if completed_dt is not None
                    else ""
                ),
                "created_at_epoch_s": (
                    float(created_dt.timestamp()) if created_dt is not None else 0.0
                ),
                "completed_at_epoch_s": (
                    float(completed_dt.timestamp()) if completed_dt is not None else 0.0
                ),
                "replica_memory_size_bytes": int(row.replica_memory_size_bytes),
            }
        )
    payload["rows"] = rows
    payload["row_count"] = int(len(rows))
    return payload


def compute_replica_id_diffusion(
    *,
    transport_rows_payload: dict[str, Any],
    artifact_ids: set[str],
) -> dict[str, Any]:
    diffusion: dict[str, Any] = {
        "source_key": "replica_id",
        "error": None,
        "artifact_filter_enabled": bool(artifact_ids),
        "artifact_filter_count": int(len(artifact_ids)),
        "transport_rows_scanned": 0,
        "transport_rows_in_scope": 0,
        "total_transports": 0,
        "unique_sources": 0,
        "top1_share": 0.0,
        "hhi": 0.0,
        "source_counts": [],
    }
    error = transport_rows_payload.get("error")
    if error:
        diffusion["error"] = str(error)
        return diffusion

    rows_raw = transport_rows_payload.get("rows", [])
    rows: list[dict[str, Any]] = []
    if isinstance(rows_raw, list):
        rows.extend(row for row in rows_raw if isinstance(row, dict))
    diffusion["transport_rows_scanned"] = int(len(rows))

    source_counter: Counter[str] = Counter()
    for row in rows:
        artifact_id = str(row.get("artifact_id", "")).strip()
        if artifact_ids and artifact_id not in artifact_ids:
            continue
        replica_id = str(row.get("replica_id", "")).strip()
        diffusion["transport_rows_in_scope"] = (
            int(diffusion["transport_rows_in_scope"]) + 1
        )
        if not replica_id:
            continue
        source_counter[replica_id] += 1

    total_sources = int(sum(source_counter.values()))
    diffusion["total_transports"] = int(total_sources)
    if total_sources <= 0:
        return diffusion

    top_count = max(source_counter.values())
    shares = [float(count) / float(total_sources) for count in source_counter.values()]
    source_counts = [
        {
            "replica_id": replica_id,
            "count": int(count),
            "share": float(float(count) / float(total_sources)),
        }
        for replica_id, count in sorted(
            source_counter.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    diffusion["unique_sources"] = int(len(source_counter))
    diffusion["top1_share"] = float(float(top_count) / float(total_sources))
    diffusion["hhi"] = float(sum(share * share for share in shares))
    diffusion["source_counts"] = source_counts
    return diffusion


def restart_workers(
    *,
    workers: list[WorkerSpec],
    daemon_config: str,
    gs_addr: str,
    conn: int,
    buffers: int,
    maxw: int,
    expected_gpu_channels: int,
    promotion_max_concurrency: int,
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
            promotion_max_concurrency=promotion_max_concurrency,
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


def summarize_completion_profile(
    *,
    gets: list[dict[str, Any]],
    offset_key: str,
) -> dict[str, Any]:
    offsets: list[float] = []
    for item in gets:
        if bool(item.get("failed")):
            continue
        raw = item.get(offset_key)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value < 0.0:
            continue
        offsets.append(value)

    if not offsets:
        return {
            "sample_count": 0,
            "completion_makespan_s": 0.0,
            "completion_mean_s": 0.0,
            "completion_p50_s": 0.0,
            "completion_p90_s": 0.0,
            "completion_stddev_s": 0.0,
            "completion_by_half_ratio": 0.0,
            "completion_curve_auc_s": 0.0,
            "completion_curve_auc_norm": 0.0,
        }

    ordered = sorted(offsets)
    sample_count = len(ordered)
    makespan = float(ordered[-1])
    completion_mean = float(statistics.mean(ordered))
    completion_p50 = percentile(ordered, 0.5)
    completion_p90 = percentile(ordered, 0.9)
    completion_stddev = float(statistics.pstdev(ordered)) if sample_count > 1 else 0.0

    if makespan <= 0.0:
        completion_by_half_ratio = 1.0
        completion_curve_auc_s = 0.0
        completion_curve_auc_norm = 1.0
    else:
        completion_by_half_ratio = float(
            sum(1 for value in ordered if value <= (makespan * 0.5))
        ) / float(sample_count)
        area = 0.0
        prev_t = 0.0
        for completed, value in enumerate(ordered):
            area += (float(completed) / float(sample_count)) * (value - prev_t)
            prev_t = value
        completion_curve_auc_s = float(area)
        completion_curve_auc_norm = float(area / makespan)

    return {
        "sample_count": int(sample_count),
        "completion_makespan_s": float(makespan),
        "completion_mean_s": float(completion_mean),
        "completion_p50_s": float(completion_p50),
        "completion_p90_s": float(completion_p90),
        "completion_stddev_s": float(completion_stddev),
        "completion_by_half_ratio": float(completion_by_half_ratio),
        "completion_curve_auc_s": float(completion_curve_auc_s),
        "completion_curve_auc_norm": float(completion_curve_auc_norm),
    }


def run_wave_gets(
    *,
    workers: list[WorkerSpec],
    key: str,
    artifact_id: str,
    lookup_mode: str,
    get_device: str,
    pinned_allocation_timeout_ms: int,
    visibility_timeout_sec: float,
    visibility_retry_sec: float,
    payload_sample_verify: bool,
    remote_timeout_sec: float,
    enable_failure_diag: bool,
    failure_diag_timeout_sec: float,
    iteration_start_ts: float | None = None,
    wave_name: str = "",
) -> tuple[list[dict[str, Any]], float]:
    if not workers:
        return [], 0.0

    start_ts = time.perf_counter()
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(workers)) as pool:
        future_map: dict[concurrent.futures.Future, tuple[WorkerSpec, float]] = {}
        for worker in workers:
            submitted_at = time.perf_counter()
            future = pool.submit(
                run_get,
                worker=worker,
                key=key,
                artifact_id=artifact_id,
                lookup_mode=lookup_mode,
                get_device=get_device,
                pinned_allocation_timeout_ms=pinned_allocation_timeout_ms,
                visibility_timeout_sec=visibility_timeout_sec,
                visibility_retry_sec=visibility_retry_sec,
                payload_sample_verify=payload_sample_verify,
                timeout_sec=remote_timeout_sec,
            )
            future_map[future] = (worker, submitted_at)
        for future in concurrent.futures.as_completed(future_map):
            worker, submitted_at = future_map[future]
            completed_at = time.perf_counter()
            wave_submit_offset_sec = float(max(0.0, submitted_at - start_ts))
            wave_wait_to_complete_sec = float(max(0.0, completed_at - start_ts))
            dispatch_to_complete_sec = float(max(0.0, completed_at - submitted_at))
            iteration_wait_to_complete_sec = (
                float(max(0.0, completed_at - iteration_start_ts))
                if iteration_start_ts is not None
                else wave_wait_to_complete_sec
            )
            try:
                payload = future.result()
            except Exception as exc:  # noqa: BLE001
                failure_probe: dict[str, Any] | None = None
                if enable_failure_diag:
                    try:
                        failure_probe = collect_worker_failure_probe(
                            worker=worker,
                            timeout_sec=float(failure_diag_timeout_sec),
                        )
                    except Exception as probe_exc:  # noqa: BLE001
                        failure_probe = {
                            "process_id": worker.process_id,
                            "daemon_addr": worker.daemon_addr,
                            "probe_error": str(probe_exc),
                        }
                error_text = str(exc)
                failure_classification = classify_failure(error_text)
                timeout_root = classify_timeout_root(
                    error_message=error_text,
                    failure_probe=failure_probe,
                )
                results.append(
                    {
                        "worker": worker.name,
                        "process_id": worker.process_id,
                        "daemon_addr": worker.daemon_addr,
                        "failed": True,
                        "error": error_text,
                        "failure_classification": failure_classification,
                        "timeout_root": timeout_root,
                        "final_error_code": "UNKNOWN",
                        "attempts": 0,
                        "failure_probe": failure_probe,
                        "wave": wave_name,
                        "wave_submit_offset_sec": wave_submit_offset_sec,
                        "wave_wait_to_complete_sec": wave_wait_to_complete_sec,
                        "dispatch_to_complete_sec": dispatch_to_complete_sec,
                        "iteration_wait_to_complete_sec": iteration_wait_to_complete_sec,
                    }
                )
                continue
            payload["failed"] = False
            payload["wave"] = wave_name
            payload["wave_submit_offset_sec"] = wave_submit_offset_sec
            payload["wave_wait_to_complete_sec"] = wave_wait_to_complete_sec
            payload["dispatch_to_complete_sec"] = dispatch_to_complete_sec
            payload["iteration_wait_to_complete_sec"] = iteration_wait_to_complete_sec
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
        "unavailable",
        "deadline exceeded",
        "context deadline exceeded",
        "no route to host",
        "command failed with rc=",
        "command terminated with exit code",
        "exit code 137",
        "exit code 143",
        "terminated",
        "killed",
    )
    product_tokens = (
        "invalid_argument",
        "failed_precondition",
        "not found",
        "not_found",
        "permission denied",
        "resource exhausted",
        "statuscode.resource_exhausted",
        "out of memory",
        "cuda out of memory",
        "grpc_status:8",
    )
    if any(token in lowered for token in infra_tokens):
        return "infra"
    if any(token in lowered for token in product_tokens):
        return "product"
    return "unknown"


TIMEOUT_ROOT_LABELS = (
    "none",
    "wait_timeout",
    "transport_timeout",
    "wait_then_transport",
    "timeout_unknown",
)


def classify_timeout_root(
    *,
    error_message: str,
    failure_probe: dict[str, Any] | None,
) -> str:
    probe_text = ""
    if isinstance(failure_probe, dict):
        probe_text = " ".join(
            str(failure_probe.get(key, ""))
            for key in (
                "remote_probe_tail",
                "daemon_err_excerpt",
                "daemon_logs_excerpt",
            )
        )
    lowered = f"{error_message} {probe_text}".lower()
    timeout_trigger_tokens = (
        "deadline exceeded",
        "timed out",
        "timeout",
    )
    if not any(token in lowered for token in timeout_trigger_tokens):
        return "none"

    wait_tokens = (
        "no available replicas",
        "retrying source selection",
        "remaining_budget_ms",
        "read still pending",
        "waiting for staging credit",
        "staging_credit",
    )
    transport_tokens = (
        "failed to read chunk",
        "peer closed connection",
        "connection reset",
        "connection refused",
        "epollrdhup",
        "failed to connect",
        "streamingpinnedbuffer::release timed out",
    )
    has_wait = any(token in lowered for token in wait_tokens)
    has_transport = any(token in lowered for token in transport_tokens)
    if has_wait and has_transport:
        return "wait_then_transport"
    if has_wait:
        return "wait_timeout"
    if has_transport:
        return "transport_timeout"
    return "timeout_unknown"


def resolve_get_pinned_allocation_timeout_ms(args: argparse.Namespace) -> int:
    configured = int(args.get_pinned_allocation_timeout_ms)
    if configured > 0:
        return configured
    # Under large fanout, pinned allocation can exceed the legacy 30s timeout.
    visibility_ms = int(max(0.0, float(args.visibility_timeout_sec)) * 1000.0)
    return max(30000, visibility_ms)


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


def summarize_worker_transfer_stats(
    *,
    gets: list[dict[str, Any]],
    worker_by_name: dict[str, WorkerSpec],
) -> dict[str, dict[str, Any]]:
    per_worker: dict[str, list[float]] = {}
    for item in gets:
        if bool(item.get("failed")):
            continue
        worker = str(item.get("worker", "")).strip()
        if not worker:
            continue
        per_worker.setdefault(worker, []).append(float(item.get("transfer_gibps", 0.0)))

    summary: dict[str, dict[str, Any]] = {}
    for worker in sorted(per_worker):
        values = per_worker[worker]
        spec = worker_by_name.get(worker)
        summary[worker] = {
            "samples": int(len(values)),
            "transfer_gibps_mean": float(statistics.mean(values)),
            "transfer_gibps_p90": percentile(values, 0.9),
            "transfer_gibps_min": float(min(values)),
            "transfer_gibps_max": float(max(values)),
            "process_id": spec.process_id if spec else "",
            "daemon_addr": spec.daemon_addr if spec else "",
            "advertise_ip": spec.advertise_ip if spec else "",
        }
    return summary


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
    pinned_allocation_timeout_ms = resolve_get_pinned_allocation_timeout_ms(args)
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
        pinned_allocation_timeout_ms=pinned_allocation_timeout_ms,
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
            pinned_allocation_timeout_ms=pinned_allocation_timeout_ms,
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
    if len(getters) < 1:
        raise RuntimeError("fanout mode requires at least 1 get worker")

    if len(getters) == 1:
        wave_size = 1
    else:
        wave_size = int(args.wave_size)
        if wave_size <= 0:
            wave_size = max(1, len(getters) // 2)
        wave_size = min(max(1, wave_size), len(getters) - 1)

    wave_assignment = str(args.wave_assignment).strip().lower()
    wave_assignment_seed = int(args.wave_assignment_seed)
    total = int(args.warmup) + int(args.iterations)
    run_tag = f"{int(time.time())}-{random_suffix(8)}"
    worker_by_name = {worker.name: worker for worker in getters}
    pinned_allocation_timeout_ms = resolve_get_pinned_allocation_timeout_ms(args)
    case_started_at_utc = datetime.now(timezone.utc)
    artifact_ids_seen: set[str] = set()
    size_bytes = int(args.size_mib) * 1024 * 1024
    cleanup_leak_threshold_bytes = int(args.cleanup_leak_threshold_bytes)
    if cleanup_leak_threshold_bytes <= 0:
        cleanup_leak_threshold_bytes = max(64 * 1024 * 1024, size_bytes // 2)
    cleanup_leak_streak_threshold = max(1, int(args.cleanup_leak_streak_threshold))
    cleanup_leak_streak = 0
    pre_case_seed_status = run_detailed_status(
        seed,
        timeout_sec=min(45.0, max(10.0, float(args.remote_timeout_sec))),
    )
    baseline_seed_artifact_bytes = int(
        pre_case_seed_status.get("total_artifact_size_bytes", 0)
    )
    baseline_seed_replicas = int(pre_case_seed_status.get("total_replicas_loaded", 0))

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

    def select_wave_workers(
        iteration_index: int,
    ) -> tuple[list[WorkerSpec], list[WorkerSpec]]:
        if len(getters) == 1:
            return getters[:1], []
        ordered = list(getters)
        if wave_assignment == "rotate":
            offset = int(iteration_index) % len(ordered)
            ordered = ordered[offset:] + ordered[:offset]
        elif wave_assignment == "shuffle":
            rng = random.Random(wave_assignment_seed + int(iteration_index))
            rng.shuffle(ordered)
        return ordered[:wave_size], ordered[wave_size:]

    for iteration in range(total):
        warmup = iteration < int(args.warmup)
        wave1_workers, wave2_workers = select_wave_workers(iteration)
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
                "wave_assignment": wave_assignment,
                "wave1_workers": [worker.name for worker in wave1_workers],
                "wave2_workers": [worker.name for worker in wave2_workers],
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
        artifact_ids_seen.add(artifact_id)

        iter_start = time.perf_counter()
        wave1_gets, wave1_wall = run_wave_gets(
            workers=wave1_workers,
            key=key,
            artifact_id=artifact_id,
            lookup_mode=str(args.lookup_mode),
            get_device=str(args.get_device),
            pinned_allocation_timeout_ms=pinned_allocation_timeout_ms,
            visibility_timeout_sec=float(args.visibility_timeout_sec),
            visibility_retry_sec=float(args.visibility_retry_sec),
            payload_sample_verify=bool(args.payload_sample_verify),
            remote_timeout_sec=float(args.remote_timeout_sec),
            enable_failure_diag=bool(args.failure_diag),
            failure_diag_timeout_sec=float(args.failure_diag_timeout_sec),
            iteration_start_ts=iter_start,
            wave_name="wave1",
        )
        wave2_gets, wave2_wall = run_wave_gets(
            workers=wave2_workers,
            key=key,
            artifact_id=artifact_id,
            lookup_mode=str(args.lookup_mode),
            get_device=str(args.get_device),
            pinned_allocation_timeout_ms=pinned_allocation_timeout_ms,
            visibility_timeout_sec=float(args.visibility_timeout_sec),
            visibility_retry_sec=float(args.visibility_retry_sec),
            payload_sample_verify=bool(args.payload_sample_verify),
            remote_timeout_sec=float(args.remote_timeout_sec),
            enable_failure_diag=bool(args.failure_diag),
            failure_diag_timeout_sec=float(args.failure_diag_timeout_sec),
            iteration_start_ts=iter_start,
            wave_name="wave2",
        )
        iter_wall = float(time.perf_counter() - iter_start)
        all_wave_gets = [*wave1_gets, *wave2_gets]

        failed_gets = [item for item in all_wave_gets if bool(item.get("failed"))]
        classification_counts: dict[str, int] = {"infra": 0, "product": 0, "unknown": 0}
        timeout_root_counts: dict[str, int] = dict.fromkeys(
            TIMEOUT_ROOT_LABELS,
            0,
        )
        for failed in failed_gets:
            worker = str(failed.get("worker", "unknown"))
            process_id = str(failed.get("process_id", ""))
            daemon_addr = str(failed.get("daemon_addr", ""))
            reason = str(failed.get("error", ""))
            failure_type = str(
                failed.get("failure_classification") or classify_failure(reason)
            )
            timeout_root = str(
                failed.get("timeout_root")
                or classify_timeout_root(
                    error_message=reason,
                    failure_probe=(
                        failed.get("failure_probe")
                        if isinstance(failed.get("failure_probe"), dict)
                        else None
                    ),
                )
            )
            failed["failure_classification"] = failure_type
            failed["timeout_root"] = timeout_root
            classification_counts[failure_type] = (
                int(classification_counts.get(failure_type, 0)) + 1
            )
            timeout_root_counts[timeout_root] = (
                int(timeout_root_counts.get(timeout_root, 0)) + 1
            )
            events.append(
                {
                    "ts_epoch": float(time.time()),
                    "iter": int(iteration),
                    "type": "get_failed",
                    "worker": worker,
                    "process_id": process_id,
                    "daemon_addr": daemon_addr,
                    "classification": failure_type,
                    "timeout_root": timeout_root,
                    "error": reason,
                    "failure_probe": failed.get("failure_probe"),
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
        post_cleanup_probe: dict[str, Any] | None = None
        if bool(args.cleanup_artifacts) and bool(args.cleanup_leak_sentinel):
            post_cleanup_probe = run_detailed_status(
                seed,
                timeout_sec=min(45.0, max(10.0, float(args.remote_timeout_sec))),
            )
            observed_artifact_bytes = int(
                post_cleanup_probe.get("total_artifact_size_bytes", 0)
            )
            observed_replicas = int(post_cleanup_probe.get("total_replicas_loaded", 0))
            leak_guard_limit = int(baseline_seed_artifact_bytes) + int(
                cleanup_leak_threshold_bytes
            )
            if (
                observed_artifact_bytes > leak_guard_limit
                or observed_replicas > baseline_seed_replicas + 1
            ):
                cleanup_leak_streak += 1
            else:
                cleanup_leak_streak = 0
            if cleanup_leak_streak >= cleanup_leak_streak_threshold:
                raise RuntimeError(
                    "cleanup leak sentinel triggered: "
                    f"iter={iteration} streak={cleanup_leak_streak} "
                    f"baseline_artifact_bytes={baseline_seed_artifact_bytes} "
                    f"observed_artifact_bytes={observed_artifact_bytes} "
                    f"threshold_bytes={cleanup_leak_threshold_bytes} "
                    f"baseline_replicas={baseline_seed_replicas} "
                    f"observed_replicas={observed_replicas}"
                )
        else:
            cleanup_leak_streak = 0

        wave1_summary = summarize_wave(gets=wave1_gets, wall_sec=wave1_wall)
        wave2_summary = summarize_wave(gets=wave2_gets, wall_sec=wave2_wall)
        wave1_completion_profile = summarize_completion_profile(
            gets=wave1_gets,
            offset_key="wave_wait_to_complete_sec",
        )
        wave2_completion_profile = summarize_completion_profile(
            gets=wave2_gets,
            offset_key="wave_wait_to_complete_sec",
        )
        iteration_completion_profile = summarize_completion_profile(
            gets=all_wave_gets,
            offset_key="iteration_wait_to_complete_sec",
        )
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
                "completion_profile": wave1_completion_profile,
            },
            "wave2": {
                "workers": [worker.name for worker in wave2_workers],
                "gets": wave2_gets,
                "summary": wave2_summary,
                "completion_profile": wave2_completion_profile,
            },
            "iter_wall_sec": iter_wall,
            "task_load_complete_sec": float(
                iteration_completion_profile.get("completion_makespan_s", 0.0)
            ),
            "task_total_sec": float(float(put.get("put_sec", 0.0)) + iter_wall),
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
            "timeout_root_counts": timeout_root_counts,
            "cleanup": cleanup,
            "post_cleanup_probe": post_cleanup_probe,
            "cleanup_leak_streak": int(cleanup_leak_streak),
            "completion_profile": iteration_completion_profile,
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
            f"task={float(record['task_load_complete_sec']):.3f}s "
            f"auc={float(iteration_completion_profile.get('completion_curve_auc_norm', 0.0)):.3f} "
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
    worker_transfer_stats = summarize_worker_transfer_stats(
        gets=all_gets,
        worker_by_name=worker_by_name,
    )
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
    task_load_complete_vals = [
        float(item.get("task_load_complete_sec", 0.0)) for item in measured
    ]
    task_total_vals = [float(item.get("task_total_sec", 0.0)) for item in measured]
    completion_auc_vals = [
        float(item.get("completion_profile", {}).get("completion_curve_auc_norm", 0.0))
        for item in measured
    ]
    completion_by_half_vals = [
        float(item.get("completion_profile", {}).get("completion_by_half_ratio", 0.0))
        for item in measured
    ]
    completion_mean_vals = [
        float(item.get("completion_profile", {}).get("completion_mean_s", 0.0))
        for item in measured
    ]
    completion_p90_vals = [
        float(item.get("completion_profile", {}).get("completion_p90_s", 0.0))
        for item in measured
    ]
    instance_completion_profile = summarize_completion_profile(
        gets=all_gets,
        offset_key="iteration_wait_to_complete_sec",
    )
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
    if len(wave2_workers) == 0:
        wave2_over_wave1 = 1.0
    else:
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
    timeout_root_totals: dict[str, int] = dict.fromkeys(TIMEOUT_ROOT_LABELS, 0)
    for item in measured:
        record_class = item.get("classification_counts", {})
        if not isinstance(record_class, dict):
            record_class = {}
        for label in ("infra", "product", "unknown"):
            classification_totals[label] = classification_totals.get(label, 0) + int(
                record_class.get(label, 0)
            )
        record_timeout = item.get("timeout_root_counts", {})
        if not isinstance(record_timeout, dict):
            continue
        for label, value in record_timeout.items():
            timeout_root_totals[str(label)] = timeout_root_totals.get(
                str(label), 0
            ) + int(value)

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

    measured_artifact_ids = {
        str(item.get("artifact_id", "")).strip()
        for item in measured
        if str(item.get("artifact_id", "")).strip()
    }
    if not measured_artifact_ids:
        measured_artifact_ids = set(artifact_ids_seen)
    case_finished_at_utc = datetime.now(timezone.utc)
    transport_rows_payload = query_transport_rows_via_gs_rpc(
        gs_addr=str(args.gs_addr),
        started_at_utc=case_started_at_utc,
        finished_at_utc=case_finished_at_utc,
        limit=int(args.transport_window_limit),
    )
    replica_id_diffusion = compute_replica_id_diffusion(
        transport_rows_payload=transport_rows_payload,
        artifact_ids=measured_artifact_ids,
    )

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
        "wave1_getters": int(wave_size),
        "wave2_getters": int(len(getters) - wave_size),
        "wave_assignment": wave_assignment,
        "wave_assignment_seed": int(wave_assignment_seed),
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
        "task_load_complete_sec_mean": float(statistics.mean(task_load_complete_vals)),
        "task_load_complete_sec_p90": percentile(task_load_complete_vals, 0.9),
        "task_total_sec_mean": float(statistics.mean(task_total_vals)),
        "task_total_sec_p90": percentile(task_total_vals, 0.9),
        "completion_curve_auc_norm_mean": float(statistics.mean(completion_auc_vals)),
        "completion_curve_auc_norm_p90": percentile(completion_auc_vals, 0.9),
        "completion_by_half_ratio_mean": float(
            statistics.mean(completion_by_half_vals)
        ),
        "completion_by_half_ratio_p90": percentile(completion_by_half_vals, 0.9),
        "completion_mean_s_mean": float(statistics.mean(completion_mean_vals)),
        "completion_mean_s_p90": percentile(completion_mean_vals, 0.9),
        "completion_p90_s_mean": float(statistics.mean(completion_p90_vals)),
        "completion_p90_s_p90": percentile(completion_p90_vals, 0.9),
        "instance_wait_to_complete_profile": instance_completion_profile,
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
        "failure_timeout_root_counts": timeout_root_totals,
        "bad_source_workers": bad_sources,
        "comm_error_count": int(comm_error_count),
        "comm_bytes_mismatch_count": int(comm_mismatch_count),
        "db_probe_enabled": bool(db_probe_enabled),
        "db_probe_disabled_reason": db_probe_disabled_reason,
        "transport_rows_error": transport_rows_payload.get("error"),
        "transport_rows_count": int(transport_rows_payload.get("row_count", 0)),
        "replica_id_diffusion": replica_id_diffusion,
        "source_cardinality_mode": "address_label",
        "worker_transfer_stats": worker_transfer_stats,
        "cleanup_leak_sentinel_enabled": bool(args.cleanup_leak_sentinel),
        "cleanup_leak_threshold_bytes": int(cleanup_leak_threshold_bytes),
        "cleanup_leak_streak_threshold": int(cleanup_leak_streak_threshold),
        "cleanup_leak_streak_final": int(cleanup_leak_streak),
        "pre_case_seed_status": pre_case_seed_status,
        "events_count": int(len(events)),
    }
    return {
        "mode": "fanout",
        "summary": summary,
        "records": records,
        "events": events,
        "transport_rows_payload": transport_rows_payload,
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
    parser.add_argument(
        "--promotion-max-concurrency",
        type=int,
        default=0,
        help=(
            "Override daemon promotion.max_concurrency during restart. "
            "0 keeps daemon-config/default value."
        ),
    )
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
    parser.add_argument(
        "--wave-assignment",
        choices=("fixed", "rotate", "shuffle"),
        default="rotate",
        help=(
            "How to assign getters into wave1/wave2 each iteration. "
            "'rotate' reduces fixed wave bias for heterogeneous nodes."
        ),
    )
    parser.add_argument(
        "--wave-assignment-seed",
        type=int,
        default=20260227,
        help="Deterministic seed used by wave-assignment=shuffle.",
    )
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
    parser.add_argument(
        "--get-pinned-allocation-timeout-ms",
        type=int,
        default=0,
        help=(
            "Pinned allocation timeout for each get helper call. "
            "0 means adaptive: max(30000, visibility-timeout-sec*1000)."
        ),
    )
    parser.add_argument("--visibility-timeout-sec", type=float, default=30.0)
    parser.add_argument("--visibility-retry-sec", type=float, default=0.05)
    parser.add_argument("--transport-window-limit", type=int, default=200000)
    parser.add_argument("--remote-timeout-sec", type=float, default=900.0)
    parser.add_argument(
        "--failure-diag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Collect on-failure worker diagnostics (orchestratorctl get/describe + "
            "remote daemon status probe)."
        ),
    )
    parser.add_argument(
        "--failure-diag-timeout-sec",
        type=float,
        default=45.0,
        help="Per-failure diagnostics probe timeout in seconds.",
    )
    parser.add_argument("--daemon-start-timeout-sec", type=float, default=600.0)
    parser.add_argument("--stop-timeout-sec", type=float, default=180.0)
    parser.add_argument(
        "--teardown-verify-timeout-sec",
        type=float,
        default=30.0,
        help="Per-worker timeout for teardown daemon status verification.",
    )
    parser.add_argument(
        "--teardown-strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail the case when any daemon session is still running after teardown.",
    )
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
        "--runtime-root",
        default="/tmp/tensorcast/cross_host/runtime",
        help=(
            "Remote runtime root for daemon home/storage state; "
            "override for persistent scratch storage on your workers."
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
        "--cleanup-leak-sentinel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After each cleanup, probe seed daemon status and fail-fast when "
            "artifact residency keeps growing."
        ),
    )
    parser.add_argument(
        "--cleanup-leak-threshold-bytes",
        type=int,
        default=0,
        help=(
            "Absolute bytes above pre-case baseline tolerated by cleanup leak "
            "sentinel. 0 uses adaptive threshold."
        ),
    )
    parser.add_argument(
        "--cleanup-leak-streak-threshold",
        type=int,
        default=2,
        help="Consecutive suspicious cleanup iterations required to fail-fast.",
    )
    parser.add_argument(
        "--out-dir", default="/tmp/tensorcast/cross_host/results_multi_host_scaleout"
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
    runtime_root = str(args.runtime_root).strip()
    if not runtime_root:
        raise RuntimeError("runtime-root must be non-empty")
    case_root = f"{runtime_root.rstrip('/')}/{case_token}"

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
    return None


def main() -> int:
    args = parse_args()
    if int(args.transport_window_limit) <= 0:
        raise RuntimeError("transport-window-limit must be > 0")
    if float(args.failure_diag_timeout_sec) <= 0:
        raise RuntimeError("failure-diag-timeout-sec must be > 0")
    if int(args.cleanup_leak_streak_threshold) <= 0:
        raise RuntimeError("cleanup-leak-streak-threshold must be > 0")
    if int(args.cleanup_leak_threshold_bytes) < 0:
        raise RuntimeError("cleanup-leak-threshold-bytes must be >= 0")
    if float(args.teardown_verify_timeout_sec) <= 0:
        raise RuntimeError("teardown-verify-timeout-sec must be > 0")
    out_dir = Path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    seed, getters = build_workers(args)
    deregister_device_id = infer_deregister_device_id(args)
    remote_run_as_user = configure_remote_run_as_user(resolve_workspace_user())
    print(
        "[preflight] enforce non-root remote execution "
        f"run_as_user={remote_run_as_user}",
        flush=True,
    )
    all_workers = [seed, *getters]
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, len(all_workers))
    ) as pool:
        futures = {
            pool.submit(
                verify_remote_run_as_user,
                worker.process_id,
                timeout_sec=min(30.0, float(args.remote_timeout_sec)),
            ): worker
            for worker in all_workers
        }
        for future, worker in futures.items():
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"remote run-as preflight failed for worker={worker.name} "
                    f"process={worker.process_id}: {exc}"
                ) from exc
    print(
        f"[case] {args.case_name} mode={args.mode} seed={seed.process_id} "
        f"getters={len(getters)} conn={args.conn} buffers={args.buffers} maxw={args.maxw} "
        f"egc={args.expected_gpu_channels} size_mib={args.size_mib}",
        flush=True,
    )
    payload: dict[str, Any] | None = None
    summary: dict[str, Any] = {}
    case_error: Exception | None = None

    try:
        restart_workers(
            workers=[seed, *getters],
            daemon_config=str(args.daemon_config),
            gs_addr=str(args.gs_addr),
            conn=int(args.conn),
            buffers=int(args.buffers),
            maxw=int(args.maxw),
            expected_gpu_channels=int(args.expected_gpu_channels),
            promotion_max_concurrency=int(args.promotion_max_concurrency),
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
    except Exception as exc:  # noqa: BLE001
        case_error = exc
    finally:
        teardown = teardown_workers(
            workers=all_workers,
            stop_timeout_sec=float(args.stop_timeout_sec),
            verify_timeout_sec=float(args.teardown_verify_timeout_sec),
        )
        if payload is None:
            payload = {
                "mode": str(args.mode),
                "summary": {},
                "records": [],
                "events": [],
            }
        payload["teardown"] = teardown
        payload_summary = payload.get("summary")
        if not isinstance(payload_summary, dict):
            payload_summary = {}
            payload["summary"] = payload_summary
        payload_summary["teardown_all_stopped"] = bool(teardown.get("all_stopped"))
        payload_summary["teardown_errors"] = list(teardown.get("errors", []))
        if (
            bool(args.teardown_strict)
            and not bool(teardown.get("all_stopped"))
            and case_error is None
        ):
            case_error = RuntimeError(
                "teardown verification failed: daemon session still running"
            )
        if case_error is not None:
            payload["fatal_error"] = str(case_error)

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
        "promotion_max_concurrency": int(args.promotion_max_concurrency),
        "size_mib": int(args.size_mib),
        "warmup": int(args.warmup),
        "iterations": int(args.iterations),
        "wave_size": int(args.wave_size),
        "wave_assignment": str(args.wave_assignment),
        "wave_assignment_seed": int(args.wave_assignment_seed),
        "put_seed_base": int(args.put_seed_base),
        "lookup_mode": str(args.lookup_mode),
        "visibility_timeout_sec": float(args.visibility_timeout_sec),
        "visibility_retry_sec": float(args.visibility_retry_sec),
        "remote_timeout_sec": float(args.remote_timeout_sec),
        "daemon_start_timeout_sec": float(args.daemon_start_timeout_sec),
        "stop_timeout_sec": float(args.stop_timeout_sec),
        "get_pinned_allocation_timeout_ms": int(
            resolve_get_pinned_allocation_timeout_ms(args)
        ),
        "payload_sample_verify": bool(args.payload_sample_verify),
        "failure_diag_enabled": bool(args.failure_diag),
        "failure_diag_timeout_sec": float(args.failure_diag_timeout_sec),
        "require_p2p": bool(args.require_p2p),
        "require_vram_source": bool(args.require_vram_source),
        "cleanup_artifacts": bool(args.cleanup_artifacts),
        "cleanup_leak_sentinel": bool(args.cleanup_leak_sentinel),
        "cleanup_leak_threshold_bytes": int(args.cleanup_leak_threshold_bytes),
        "cleanup_leak_streak_threshold": int(args.cleanup_leak_streak_threshold),
        "source_retire_mode": str(args.source_retire_mode),
        "gs_db_file": str(args.gs_db_file),
        "transport_window_limit": int(args.transport_window_limit),
        "runtime_root": str(args.runtime_root),
        "source_stop_settle_sec": float(args.source_stop_settle_sec),
        "teardown_strict": bool(args.teardown_strict),
        "teardown_verify_timeout_sec": float(args.teardown_verify_timeout_sec),
        "deregister_device_id": deregister_device_id,
        "remote_run_as_user": remote_run_as_user,
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    final_summary = summary
    if not final_summary and isinstance(payload.get("summary"), dict):
        final_summary = payload["summary"]
    print("SUMMARY", json.dumps(final_summary, ensure_ascii=False), flush=True)
    print(f"OUTPUT {out_path}", flush=True)
    if case_error is not None:
        raise case_error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
