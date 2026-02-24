#!/usr/bin/env python3
#  Copyright (c) 2026, TensorCast Team.

# Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import argparse
import atexit
import concurrent.futures
import json
import re
import shlex
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import grpc
import yaml

from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.global_store.v1 import global_store_pb2, global_store_pb2_grpc

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REMOTE_TIMEOUT_SEC = 1800.0
BYTE_UNITS: dict[str, int] = {
    "b": 1,
    "kb": 1024,
    "mb": 1024**2,
    "gb": 1024**3,
    "tb": 1024**4,
}


@dataclass(frozen=True)
class RoleSpec:
    process_id: str
    output_json: str
    inner_cmd: str
    log_file: str
    ready_file: str | None = None


def estimate_remote_timeout_floor_sec(args: argparse.Namespace) -> float:
    # Conservative lower bound for full-case wall time to avoid timing out
    # long-running large payload runs while roles are still healthy.
    payload_gib = float(max(0, int(args.tp_total_bytes))) / float(1024**3)
    if str(args.payload_mode) == "tp_ranked" and payload_gib > 0.0:
        # cpu_stream registration throughput can be well below 1 GiB/s
        # under high fanout; use a conservative 0.6 GiB/s model.
        upload_seconds = payload_gib / 0.6
    else:
        upload_seconds = 0.0

    per_version_seconds = max(
        float(args.publish_interval_s) + upload_seconds,
        float(args.publish_interval_s) + 20.0,
        60.0,
    )
    tail_guard = max(
        float(args.receiver_timeout_s),
        float(args.tp_materialize_deadline_s),
        float(args.retention_timeout_s),
    )
    return float(int(int(args.num_versions) * per_version_seconds + tail_guard + 600.0))


def split_csv(raw: str) -> list[str]:
    values = [item.strip() for item in str(raw).split(",")]
    return [item for item in values if item]


def parse_host_port(address: str) -> tuple[str, int]:
    value = str(address).strip()
    if ":" not in value:
        raise ValueError(f"address must be host:port, got {address}")
    host, port_text = value.rsplit(":", 1)
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError(f"address port is not integer: {address}") from exc
    if not host:
        raise ValueError(f"address host is empty: {address}")
    if port <= 0:
        raise ValueError(f"address port must be positive: {address}")
    return host, port


def _parse_bytes_literal(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        raise ValueError(f"invalid byte value type: {type(value)}")
    if isinstance(value, (int, float)):
        number = int(value)
        if number < 0:
            raise ValueError(f"byte value must be >= 0, got {value}")
        return number
    if not isinstance(value, str):
        raise ValueError(f"invalid byte value type: {type(value)}")
    text = value.strip().lower()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([kmgt]?b)?", text)
    if not match:
        raise ValueError(f"invalid byte literal: {value!r}")
    number = float(match.group(1))
    unit = match.group(2) or "b"
    if unit not in BYTE_UNITS:
        raise ValueError(f"unsupported byte unit in {value!r}")
    parsed = int(number * BYTE_UNITS[unit])
    if parsed < 0:
        raise ValueError(f"byte literal must be >= 0, got {value!r}")
    return parsed


def _load_daemon_memory_hints(daemon_config_path: str) -> dict[str, int]:
    config_path = Path(str(daemon_config_path)).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"daemon config not found: {config_path}")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"daemon config must be a mapping object: path={config_path}")
    engine = loaded.get("engine", {})
    memory_tiers = engine.get("memory_tiers", {}) if isinstance(engine, dict) else {}
    stable_bytes = _parse_bytes_literal(
        memory_tiers.get("stable_bytes", 0) if isinstance(memory_tiers, dict) else 0
    )

    pinned = loaded.get("pinned_memory", {})
    classes = pinned.get("classes", []) if isinstance(pinned, dict) else []
    pinned_total = 0
    if isinstance(classes, list):
        for item in classes:
            if not isinstance(item, dict):
                continue
            pinned_total += _parse_bytes_literal(item.get("pool_bytes", 0))
    return {
        "stable_bytes": int(stable_bytes),
        "pinned_total_bytes": int(pinned_total),
    }


def _query_remote_memory_limit_bytes(
    process_id: str,
    *,
    timeout_sec: float,
) -> int | None:
    cmd = (
        "set -euo pipefail; "
        "if [[ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]]; then "
        "cat /sys/fs/cgroup/memory/memory.limit_in_bytes; "
        "elif [[ -f /sys/fs/cgroup/memory.max ]]; then "
        "cat /sys/fs/cgroup/memory.max; "
        "else echo 0; fi"
    )
    output = run_remote(process_id, cmd, timeout_sec=max(5.0, timeout_sec)).strip()
    if not output:
        return None
    line = output.splitlines()[-1].strip()
    if not line or line == "max":
        return None
    try:
        value = int(line)
    except ValueError:
        return None
    if value <= 0 or value >= (1 << 60):
        return None
    return int(value)


def _parse_cuda_device_index(device: str) -> int | None:
    text = str(device).strip().lower()
    if text in {"cuda", "cuda:"}:
        return 0
    if not text.startswith("cuda:"):
        return None
    index_text = text.split(":", 1)[1].strip()
    if not index_text:
        return 0
    try:
        index = int(index_text)
    except ValueError:
        return None
    if index < 0:
        return None
    return index


def _query_remote_gpu_free_memory_bytes(
    process_id: str,
    *,
    device_index: int,
    timeout_sec: float,
) -> int | None:
    cmd = (
        "set -euo pipefail; "
        "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits "
        '| awk -F\',\' \'{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1":"$2}\''
    )
    output = run_remote(process_id, cmd, timeout_sec=max(5.0, timeout_sec)).strip()
    if not output:
        return None
    free_mib: int | None = None
    for raw in output.splitlines():
        row = raw.strip()
        if not row or ":" not in row:
            continue
        idx_text, mem_text = row.split(":", 1)
        try:
            idx = int(idx_text)
            mib = int(mem_text)
        except ValueError:
            continue
        if idx == int(device_index):
            free_mib = mib
            break
    if free_mib is None or free_mib <= 0:
        return None
    return int(free_mib) * 1024 * 1024


def _format_gib(bytes_value: int) -> float:
    return float(bytes_value) / float(1024**3)


def _validate_publisher_memory_budget(
    *,
    publisher_process_id: str,
    daemon_config_path: str,
    payload_mode: str,
    tp_total_bytes: int,
    keep_last: int,
    publish_device: str,
    timeout_sec: float,
) -> dict[str, Any]:
    enabled = str(payload_mode).strip() == "tp_ranked" and int(tp_total_bytes) > 0
    report: dict[str, Any] = {
        "enabled": bool(enabled),
        "publisher_process_id": str(publisher_process_id),
        "daemon_config_path": str(daemon_config_path),
        "payload_mode": str(payload_mode),
        "tp_total_bytes": int(tp_total_bytes),
        "keep_last": int(keep_last),
        "publish_device": str(publish_device),
    }
    if not enabled:
        return report

    daemon_hints = _load_daemon_memory_hints(daemon_config_path)
    memory_limit_bytes = _query_remote_memory_limit_bytes(
        publisher_process_id,
        timeout_sec=timeout_sec,
    )
    per_version_bytes = int(tp_total_bytes)
    keep_last_versions = max(0, int(keep_last))
    # e2e publisher enables pre-publish trim for large tp_ranked payloads:
    # before putting version N, drop oldest so retained window becomes keep_last-1.
    pre_publish_trim_enabled = enabled and keep_last_versions > 0
    retained_before_publish_versions = (
        max(0, keep_last_versions - 1)
        if pre_publish_trim_enabled
        else keep_last_versions
    )
    stable_window_bytes = retained_before_publish_versions * per_version_bytes
    steady_state_window_bytes = keep_last_versions * per_version_bytes
    inflight_registration_bytes = per_version_bytes
    publisher_source_bytes = (
        per_version_bytes if str(publish_device).strip().lower() == "cpu" else 0
    )
    pinned_total_bytes = int(daemon_hints.get("pinned_total_bytes", 0))
    estimated_base_bytes = (
        stable_window_bytes
        + inflight_registration_bytes
        + publisher_source_bytes
        + pinned_total_bytes
    )
    guard_headroom_bytes = max(2 * 1024**3, int(estimated_base_bytes * 0.10))
    estimated_peak_bytes = estimated_base_bytes + guard_headroom_bytes

    report.update(
        {
            "memory_limit_bytes": memory_limit_bytes,
            "stable_bytes_configured": int(daemon_hints.get("stable_bytes", 0)),
            "pinned_total_bytes_configured": pinned_total_bytes,
            "pre_publish_trim_enabled": bool(pre_publish_trim_enabled),
            "retained_before_publish_versions": int(retained_before_publish_versions),
            "steady_state_versions": int(keep_last_versions),
            "estimated_stable_window_bytes": stable_window_bytes,
            "estimated_steady_state_window_bytes": steady_state_window_bytes,
            "estimated_inflight_registration_bytes": inflight_registration_bytes,
            "estimated_publisher_source_bytes": publisher_source_bytes,
            "estimated_headroom_bytes": guard_headroom_bytes,
            "estimated_peak_bytes": estimated_peak_bytes,
        }
    )

    warnings: list[str] = []
    violations: list[str] = []
    publish_cuda_device = _parse_cuda_device_index(str(publish_device))
    if publish_cuda_device is not None:
        # DRAM_STABLE currently stages on GPU in daemon-side registration, so
        # a CUDA publish path requires (source payload + staging payload) on one
        # device. For large payloads this can exceed per-GPU memory even when
        # host memory is sufficient.
        gpu_free_bytes = _query_remote_gpu_free_memory_bytes(
            publisher_process_id,
            device_index=publish_cuda_device,
            timeout_sec=timeout_sec,
        )
        gpu_double_buffer_bytes = int(per_version_bytes) * 2
        gpu_headroom_bytes = max(2 * 1024**3, int(gpu_double_buffer_bytes * 0.10))
        gpu_required_peak_bytes = gpu_double_buffer_bytes + gpu_headroom_bytes
        report.update(
            {
                "publish_cuda_device": int(publish_cuda_device),
                "gpu_free_bytes": gpu_free_bytes,
                "estimated_gpu_double_buffer_bytes": gpu_double_buffer_bytes,
                "estimated_gpu_headroom_bytes": gpu_headroom_bytes,
                "estimated_gpu_peak_bytes": gpu_required_peak_bytes,
            }
        )
        if gpu_free_bytes is not None and gpu_required_peak_bytes > int(
            float(gpu_free_bytes) * 0.98
        ):
            violations.append(
                "publisher CUDA path is infeasible for current payload size: "
                "DRAM_STABLE registration needs source+staging double buffer on one GPU "
                f"(required≈{_format_gib(gpu_required_peak_bytes):.1f}GiB, "
                f"free≈{_format_gib(gpu_free_bytes):.1f}GiB). "
                "Use publish_device=cpu for this benchmark, or implement stage_on_gpu=false streaming."
            )
    stable_bytes = int(daemon_hints.get("stable_bytes", 0))
    if stable_bytes > 0 and stable_bytes < steady_state_window_bytes:
        violations.append(
            "stable_bytes is smaller than steady-state keep_last window "
            f"(stable={_format_gib(stable_bytes):.1f}GiB, "
            f"steady_state_window={_format_gib(steady_state_window_bytes):.1f}GiB)"
        )
    if memory_limit_bytes is not None:
        soft_limit = int(float(memory_limit_bytes) * 0.90)
        hard_limit = int(float(memory_limit_bytes) * 0.98)
        if estimated_peak_bytes > soft_limit:
            warnings.append(
                "estimated host-memory peak exceeds 90% of cgroup limit "
                f"(estimated_peak={_format_gib(estimated_peak_bytes):.1f}GiB, "
                f"limit={_format_gib(memory_limit_bytes):.1f}GiB)"
            )
        if estimated_peak_bytes > hard_limit:
            violations.append(
                "estimated host-memory peak exceeds 98% of cgroup limit "
                f"(estimated_peak={_format_gib(estimated_peak_bytes):.1f}GiB, "
                f"limit={_format_gib(memory_limit_bytes):.1f}GiB)"
            )
    report["warnings"] = warnings
    report["violations"] = violations
    report["safe"] = not violations
    if violations:
        raise RuntimeError(
            "publisher memory preflight failed: "
            + "; ".join(violations)
            + ". suggestions: increase worker memory, reduce tp_total_bytes, "
            "or keep publish_device=cpu, enable pre-publish trim, or implement stage_on_gpu=false streaming for DRAM_STABLE."
        )
    return report


def _validate_receiver_memory_budget(
    *,
    receiver_process_id: str,
    daemon_config_path: str,
    payload_mode: str,
    tp_world_size: int,
    tp_total_bytes: int,
    max_concurrency: int,
    timeout_sec: float,
) -> dict[str, Any]:
    enabled = str(payload_mode).strip() == "tp_ranked" and int(tp_total_bytes) > 0
    report: dict[str, Any] = {
        "enabled": bool(enabled),
        "receiver_process_id": str(receiver_process_id),
        "daemon_config_path": str(daemon_config_path),
        "payload_mode": str(payload_mode),
        "tp_world_size": int(tp_world_size),
        "tp_total_bytes": int(tp_total_bytes),
        "max_concurrency": int(max_concurrency),
    }
    if not enabled:
        return report

    world_size = max(1, int(tp_world_size))
    daemon_hints = _load_daemon_memory_hints(daemon_config_path)
    memory_limit_bytes = _query_remote_memory_limit_bytes(
        receiver_process_id,
        timeout_sec=timeout_sec,
    )
    stable_bytes = int(daemon_hints.get("stable_bytes", 0))
    pinned_total_bytes = int(daemon_hints.get("pinned_total_bytes", 0))
    # Receiver estimate:
    # - one stable resident version (stable_bytes configured budget)
    # - transient transfer/materialization windows scaled by transport concurrency
    # - TP rank local target tensors (~1/world_size of full payload)
    rank_target_bytes = int(tp_total_bytes) // world_size
    transient_window_factor = max(1, min(2, int(max_concurrency)))
    transient_materialization_bytes = int(tp_total_bytes) * transient_window_factor
    estimated_base_bytes = (
        stable_bytes
        + pinned_total_bytes
        + rank_target_bytes
        + transient_materialization_bytes
    )
    guard_headroom_bytes = max(2 * 1024**3, int(estimated_base_bytes * 0.10))
    estimated_peak_bytes = estimated_base_bytes + guard_headroom_bytes

    report.update(
        {
            "memory_limit_bytes": memory_limit_bytes,
            "stable_bytes_configured": stable_bytes,
            "pinned_total_bytes_configured": pinned_total_bytes,
            "estimated_rank_target_bytes": rank_target_bytes,
            "estimated_transient_window_factor": transient_window_factor,
            "estimated_transient_materialization_bytes": transient_materialization_bytes,
            "estimated_headroom_bytes": guard_headroom_bytes,
            "estimated_peak_bytes": estimated_peak_bytes,
        }
    )

    violations: list[str] = []
    if stable_bytes < int(tp_total_bytes):
        violations.append(
            "stable_bytes is smaller than one full payload version "
            f"(stable={_format_gib(stable_bytes):.1f}GiB, "
            f"payload={_format_gib(int(tp_total_bytes)):.1f}GiB)"
        )
    if memory_limit_bytes is not None and estimated_peak_bytes > int(
        float(memory_limit_bytes) * 0.95
    ):
        violations.append(
            "estimated receiver peak exceeds 95% of cgroup limit "
            f"(estimated_peak={_format_gib(estimated_peak_bytes):.1f}GiB, "
            f"limit={_format_gib(memory_limit_bytes):.1f}GiB)"
        )
    report["violations"] = violations
    report["safe"] = not violations
    if violations:
        raise RuntimeError(
            "receiver memory preflight failed: "
            + "; ".join(violations)
            + ". suggestions: lower stable_bytes for receiver profile, "
            "increase worker memory, or reduce tp_total_bytes."
        )
    return report


def run_local(cmd: list[str], *, timeout_sec: float) -> str:
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=max(1.0, timeout_sec),
    )
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        stdout = proc.stdout.strip()
        detail = stderr or stdout
        raise RuntimeError(
            f"local command failed rc={proc.returncode}: {' '.join(cmd)}\n{detail}"
        )
    return proc.stdout


def run_remote(process_id: str, inner_cmd: str, *, timeout_sec: float) -> str:
    cmd = [
        "brainctl",
        "exec",
        f"process/{process_id}",
        "-n",
        "shai-core",
        "--",
        "bash",
        "-lc",
        inner_cmd,
    ]
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=max(1.0, timeout_sec),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "remote command failed: "
            f"process={process_id} rc={proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc.stdout


def extract_last_json_object(output: str) -> dict[str, Any]:
    text = str(output).strip()
    decoder = json.JSONDecoder()
    last_object: dict[str, Any] | None = None
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            loaded, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            last_object = loaded
    if last_object is not None:
        return last_object
    raise RuntimeError(f"failed to find json object in output:\n{output}")


def discover_remote_advertise_ip(
    *,
    process_id: str,
    timeout_sec: float,
) -> str:
    detect_cmd = (
        "set -euo pipefail; "
        "ip -4 route get 1.1.1.1 2>/dev/null | awk "
        "'{for (i=1;i<=NF;i++) if ($i==\"src\") {print $(i+1); exit}}'"
    )
    output = run_remote(process_id, detect_cmd, timeout_sec=max(10.0, timeout_sec))
    candidate = str(output).strip().splitlines()
    if candidate:
        host = candidate[-1].strip()
        if host and host != "127.0.0.1":
            return host
    fallback_cmd = (
        "set -euo pipefail; "
        "hostname -I | awk '{for (i=1;i<=NF;i++) if ($i != \"127.0.0.1\") {print $i; exit}}'"
    )
    fallback = run_remote(
        process_id,
        fallback_cmd,
        timeout_sec=max(10.0, timeout_sec),
    )
    host = str(fallback).strip()
    if not host:
        raise RuntimeError(f"failed to detect advertise ip on process={process_id}")
    return host


def start_remote_daemon(
    *,
    process_id: str,
    repo_root: str,
    daemon_config: str,
    gs_addr: str,
    daemon_session: str,
    daemon_id: str,
    daemon_connect_address: str,
    daemon_p2p_port: int,
    advertise_host: str,
    cluster_id: str,
    heartbeat_interval: str,
    periodic_sync_interval: str,
    max_concurrency: int,
    timeout_sec: float,
) -> dict[str, Any]:
    _, daemon_port = daemon_connect_address.split(":", 1)
    start_args: list[str] = [
        "tensorcast-cli daemon start",
        f"--config {shlex.quote(daemon_config)}",
        f"--session {shlex.quote(daemon_session)}",
        "--global-store-mode connect",
        f"--global-store-address {shlex.quote(gs_addr)}",
        f"--set daemon_id={shlex.quote(daemon_id)}",
        "--set high_availability.enabled=true",
        (
            "--set high_availability.heartbeat_interval="
            f"{shlex.quote(str(heartbeat_interval).strip())}"
        ),
        (
            "--set high_availability.periodic_sync_interval="
            f"{shlex.quote(str(periodic_sync_interval).strip())}"
        ),
        f"--set promotion.max_concurrency={int(max_concurrency)}",
        "--set server.listen.host=0.0.0.0",
        f"--set server.listen.port={int(daemon_port)}",
        "--set server.p2p_listen.host=0.0.0.0",
        f"--set server.p2p_listen.port={int(daemon_p2p_port)}",
        f"--set server.advertise.host={shlex.quote(advertise_host)}",
    ]
    if cluster_id:
        start_args.append(f"--set meta.cluster_token={shlex.quote(cluster_id)}")
    start_args.append("--json")
    start_expr = " ".join(start_args)
    start_cmd = [
        "set -euo pipefail",
        f"cd {shlex.quote(repo_root)}",
        "source .venv/bin/activate",
        "export LD_LIBRARY_PATH=/data/cuda/compat:${LD_LIBRARY_PATH:-}",
        (
            "if command -v timeout >/dev/null 2>&1; then "
            "timeout 30s tensorcast-cli daemon stop --force >/dev/null 2>&1 || true; "
            "else tensorcast-cli daemon stop --force >/dev/null 2>&1 || true; fi"
        ),
        "for pid in $(pgrep -f '[t]ensorcast_daemon --config=' || true); do kill -TERM \"$pid\" >/dev/null 2>&1 || true; done",
        "sleep 1",
        (
            "if command -v timeout >/dev/null 2>&1; then "
            f"timeout 30s tensorcast-cli daemon stop --session {shlex.quote(daemon_session)} >/dev/null 2>&1 || true; "
            f"else tensorcast-cli daemon stop --session {shlex.quote(daemon_session)} >/dev/null 2>&1 || true; fi"
        ),
        (
            "if command -v timeout >/dev/null 2>&1; then "
            f"timeout 180s {start_expr} || true; "
            f"else {start_expr} || true; fi"
        ),
        f"tensorcast-cli daemon status --session {shlex.quote(daemon_session)} --json",
    ]
    output = run_remote(
        process_id,
        "; ".join(start_cmd),
        timeout_sec=max(60.0, timeout_sec),
    )
    status = extract_last_json_object(output)
    return status


def stop_remote_daemon(
    *,
    process_id: str,
    repo_root: str,
    daemon_session: str,
    timeout_sec: float,
) -> None:
    daemon_pattern = shlex.quote(
        f"sessions/{daemon_session}/session/effective_daemon_config.yaml"
    )
    stop_cmd = [
        "set +e",
        f"cd {shlex.quote(repo_root)}",
        "source .venv/bin/activate",
        (
            "if command -v timeout >/dev/null 2>&1; then "
            "timeout 30s tensorcast-cli daemon stop "
            f"--session {shlex.quote(daemon_session)} >/dev/null 2>&1 || true; "
            "else tensorcast-cli daemon stop "
            f"--session {shlex.quote(daemon_session)} >/dev/null 2>&1 || true; fi"
        ),
        (
            f"for pid in $(pgrep -f -- {daemon_pattern} || true); do "
            'kill -TERM "$pid" >/dev/null 2>&1 || true; '
            "done"
        ),
        "sleep 1",
        (
            f"for pid in $(pgrep -f -- {daemon_pattern} || true); do "
            'kill -KILL "$pid" >/dev/null 2>&1 || true; '
            "done"
        ),
        "exit 0",
    ]
    run_remote(
        process_id,
        "; ".join(stop_cmd),
        timeout_sec=max(20.0, timeout_sec),
    )


def preclean_remote_role_processes(
    *,
    process_id: str,
    repo_root: str,
    timeout_sec: float,
) -> None:
    clean_cmd = [
        "set +e",
        f"cd {shlex.quote(repo_root)}",
        "pkill -TERM -f '[w]eight_publisher_e2e.py' >/dev/null 2>&1 || true",
        "sleep 1",
        "pkill -KILL -f '[w]eight_publisher_e2e.py' >/dev/null 2>&1 || true",
        "source .venv/bin/activate >/dev/null 2>&1 || true",
        (
            "if command -v timeout >/dev/null 2>&1; then "
            "timeout 30s tensorcast-cli daemon stop --force >/dev/null 2>&1 || true; "
            "else tensorcast-cli daemon stop --force >/dev/null 2>&1 || true; fi"
        ),
        "pkill -TERM -f '[t]ensorcast_daemon --config=' >/dev/null 2>&1 || true",
        "sleep 1",
        "pkill -KILL -f '[t]ensorcast_daemon --config=' >/dev/null 2>&1 || true",
        "exit 0",
    ]
    run_remote(
        process_id,
        "; ".join(clean_cmd),
        timeout_sec=max(20.0, timeout_sec),
    )


def read_remote_json(
    process_id: str, path: str, *, timeout_sec: float
) -> dict[str, Any]:
    output = run_remote(
        process_id,
        f"cat {shlex.quote(path)}",
        timeout_sec=timeout_sec,
    )
    try:
        loaded = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"failed to parse remote json: process={process_id} path={path}"
        ) from exc
    if not isinstance(loaded, dict):
        raise RuntimeError(
            f"unexpected json payload type from process={process_id} path={path}"
        )
    return loaded


def build_e2e_command(
    *,
    repo_root: str,
    mode: str,
    daemon_connect_address: str,
    model_name: str,
    start_version: int,
    num_versions: int,
    keep_last: int,
    publish_interval_s: float,
    poll_interval_s: float,
    receiver_timeout_s: float,
    retention_timeout_s: float,
    payload_mode: str,
    tp_world_size: int,
    tp_total_bytes: int,
    tp_device_base_index: int,
    tp_materialize_deadline_s: float,
    publish_device: str,
    output_json: str,
    weights_root: str,
    run_id: str,
    receiver_apply_mode: str,
    fallback_prefer: str,
    materialize_device: str,
    allow_version_skip: bool,
    hold_after_finish_s: float,
    log_file: str,
    ready_file: str | None,
) -> str:
    cli_args: list[str] = [
        "python",
        "./tensorcast/tools/weight_publisher_e2e.py",
        mode,
        "--init-mode",
        "connect",
        "--connect-address",
        daemon_connect_address,
        "--model-name",
        model_name,
        "--start-version",
        str(start_version),
        "--num-versions",
        str(num_versions),
        "--poll-interval-s",
        str(poll_interval_s),
        "--receiver-timeout-s",
        str(receiver_timeout_s),
        "--payload-mode",
        payload_mode,
        "--tp-world-size",
        str(tp_world_size),
        "--tp-total-bytes",
        str(tp_total_bytes),
        "--tp-device-base-index",
        str(tp_device_base_index),
        "--tp-materialize-deadline-s",
        str(tp_materialize_deadline_s),
        "--publish-device",
        publish_device,
        "--receiver-apply-mode",
        receiver_apply_mode,
        "--fallback-prefer",
        fallback_prefer,
        "--materialize-device",
        materialize_device,
        "--output-json",
        output_json,
        "--hold-after-finish-s",
        str(hold_after_finish_s),
    ]
    if mode in {"publisher", "single-host"}:
        cli_args.extend(
            [
                "--publish-interval-s",
                str(publish_interval_s),
                "--keep-last",
                str(keep_last),
                "--retention-timeout-s",
                str(retention_timeout_s),
                "--weights-root",
                weights_root,
                "--run-id",
                run_id,
            ]
        )
    if ready_file:
        cli_args.extend(["--ready-file", ready_file])
    if mode == "receiver" and allow_version_skip:
        cli_args.append("--allow-version-skip")
    quoted_cli = " ".join(shlex.quote(part) for part in cli_args)
    quoted_log = shlex.quote(log_file)
    script = [
        "set -euo pipefail",
        f"cd {shlex.quote(repo_root)}",
        "source .venv/bin/activate",
        "export LD_LIBRARY_PATH=/data/cuda/compat:${LD_LIBRARY_PATH:-}",
        f"mkdir -p {shlex.quote(str(Path(output_json).parent))}",
        f"mkdir -p {shlex.quote(str(Path(log_file).parent))}",
        f"{quoted_cli} 2>&1 | tee -a {quoted_log}",
    ]
    return "; ".join(script)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    lower = int(k)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return float(ordered[lower])
    weight = k - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def summarize_series(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0.0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
        }
    return {
        "count": float(len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(statistics.fmean(values)),
        "p50": percentile(values, 50.0),
        "p95": percentile(values, 95.0),
    }


def discover_global_cluster_token() -> str | None:
    cmd = ["tensorcast-cli", "global", "status", "--json"]
    try:
        output = run_local(cmd, timeout_sec=10.0)
    except Exception:
        return None
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    health = payload.get("health", {})
    if not isinstance(health, dict):
        return None
    cluster_token = health.get("cluster_token")
    if isinstance(cluster_token, str) and cluster_token.strip():
        return cluster_token.strip()
    return None


def query_replica_counts_via_gs_rpc(
    *,
    gs_addr: str,
    artifact_ids: list[str],
    timeout_sec: float,
) -> tuple[dict[str, dict[str, int]], str]:
    normalized_ids = [str(artifact_id).strip() for artifact_id in artifact_ids]
    unique_ids = [
        artifact_id for artifact_id in dict.fromkeys(normalized_ids) if artifact_id
    ]
    if not unique_ids:
        return {}, "gs_rpc"

    channel = grpc.insecure_channel(str(gs_addr))
    try:
        grpc.channel_ready_future(channel).result(timeout=max(1.0, timeout_sec))
        stub = global_store_pb2_grpc.ClusterRuntimeServiceStub(channel)
        try:
            response = stub.BatchGetReplicaCounts(
                global_store_pb2.BatchGetReplicaCountsRequest(artifact_ids=unique_ids),
                timeout=max(1.0, timeout_sec),
            )
        except grpc.RpcError as exc:
            detail = exc.details() if hasattr(exc, "details") else str(exc)
            error_text = str(detail).lower()
            if (
                exc.code() == grpc.StatusCode.UNIMPLEMENTED
                or "method not found" in error_text
            ):
                counts: dict[str, dict[str, int]] = {}
                for artifact_id in unique_ids:
                    page_token = ""
                    replica_count = 0
                    while True:
                        list_response = stub.ListReplicasV2(
                            global_store_pb2.ListReplicasV2Request(
                                artifact_id=artifact_id,
                                pagination=common_pb2.Pagination(
                                    page_size=500,
                                    page_token=page_token,
                                ),
                            ),
                            timeout=max(1.0, timeout_sec),
                        )
                        replica_count += len(list_response.replicas)
                        page_token = str(
                            list_response.page_info.next_page_token or ""
                        ).strip()
                        if not page_token:
                            break
                    counts[artifact_id] = {
                        "replica_count": int(replica_count),
                        "available_count": int(replica_count),
                    }
                return counts, "gs_rpc_list_replicas_v2_fallback"
            raise RuntimeError(f"BatchGetReplicaCounts RPC failed: {detail}") from exc
    except grpc.RpcError as exc:
        detail = exc.details() if hasattr(exc, "details") else str(exc)
        raise RuntimeError(f"BatchGetReplicaCounts RPC failed: {detail}") from exc
    finally:
        channel.close()

    if response.status != global_store_pb2.Status.STATUS_OK:
        raise RuntimeError(
            f"BatchGetReplicaCounts returned non-OK status: {int(response.status)}"
        )

    result = {
        artifact_id: {"replica_count": 0, "available_count": 0}
        for artifact_id in unique_ids
    }
    for row in response.counts:
        artifact_id = str(row.artifact_id).strip()
        if not artifact_id:
            continue
        result[artifact_id] = {
            "replica_count": int(row.replica_count),
            "available_count": int(row.available_count),
        }
    return result, "gs_rpc"


VERSION_TOKEN_RE = re.compile(r"\bversion=(\d+)\b")


def _extract_latest_logged_version(log_text: str, marker: str) -> tuple[int, int]:
    latest = 0
    count = 0
    for line in str(log_text).splitlines():
        if marker not in line:
            continue
        match = VERSION_TOKEN_RE.search(line)
        if not match:
            continue
        version = int(match.group(1))
        latest = max(latest, version)
        count += 1
    return latest, count


def probe_remote_role_progress(
    *,
    process_id: str,
    log_file: str,
    timeout_sec: float,
) -> dict[str, Any]:
    inner_cmd = (
        "set -euo pipefail; "
        f"if [[ -f {shlex.quote(log_file)} ]]; then "
        f"tail -n 120 {shlex.quote(log_file)}; "
        "else echo __NO_LOG__; fi"
    )
    output = run_remote(
        process_id,
        inner_cmd,
        timeout_sec=max(5.0, timeout_sec),
    )
    if "__NO_LOG__" in output:
        return {
            "published_latest": 0,
            "published_count": 0,
            "received_latest": 0,
            "received_count": 0,
            "skipped_count": 0,
        }
    published_latest, published_count = _extract_latest_logged_version(
        output, "[publisher] published"
    )
    received_latest, received_count = _extract_latest_logged_version(
        output, "[receiver] received"
    )
    _, skipped_count = _extract_latest_logged_version(output, "[receiver] skipped")
    return {
        "published_latest": int(published_latest),
        "published_count": int(published_count),
        "received_latest": int(received_latest),
        "received_count": int(received_count),
        "skipped_count": int(skipped_count),
    }


def start_progress_monitor(
    *,
    publisher_spec: RoleSpec,
    receiver_specs: list[RoleSpec],
    run_futures: dict[str, concurrent.futures.Future[str]],
    num_versions: int,
    poll_interval_s: float,
) -> tuple[threading.Event, threading.Thread]:
    stop_event = threading.Event()

    def _monitor_loop() -> None:
        tracked_specs = [publisher_spec] + receiver_specs
        worker_count = max(1, min(16, len(tracked_specs)))
        publisher_latest_seen = 0
        publisher_events_seen = 0
        receiver_latest_seen = {spec.process_id: 0 for spec in receiver_specs}
        receiver_received_seen = {spec.process_id: 0 for spec in receiver_specs}
        receiver_skipped_seen = {spec.process_id: 0 for spec in receiver_specs}
        while not stop_event.is_set():
            progress_by_process: dict[str, dict[str, Any]] = {}
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=worker_count
            ) as pool:
                poll_futures = {
                    spec.process_id: pool.submit(
                        probe_remote_role_progress,
                        process_id=spec.process_id,
                        log_file=spec.log_file,
                        timeout_sec=10.0,
                    )
                    for spec in tracked_specs
                }
                for process_id, future in poll_futures.items():
                    try:
                        progress_by_process[process_id] = future.result()
                    except Exception as exc:  # noqa: BLE001
                        progress_by_process[process_id] = {"error": str(exc)}

            publisher_progress = progress_by_process.get(
                publisher_spec.process_id,
                {},
            )
            publisher_latest_seen = max(
                publisher_latest_seen,
                int(publisher_progress.get("published_latest", 0)),
            )
            publisher_events_seen = max(
                publisher_events_seen,
                int(publisher_progress.get("published_count", 0)),
                publisher_latest_seen,
            )

            receiver_latest_values: list[int] = []
            receiver_received_count = 0
            receiver_skipped_count = 0
            receiver_done = 0
            for spec in receiver_specs:
                role_progress = progress_by_process.get(spec.process_id, {})
                latest_received = int(role_progress.get("received_latest", 0))
                receiver_latest_seen[spec.process_id] = max(
                    int(receiver_latest_seen.get(spec.process_id, 0)),
                    latest_received,
                )
                receiver_received_seen[spec.process_id] = max(
                    int(receiver_received_seen.get(spec.process_id, 0)),
                    int(role_progress.get("received_count", 0)),
                    int(receiver_latest_seen.get(spec.process_id, 0)),
                )
                receiver_skipped_seen[spec.process_id] = max(
                    int(receiver_skipped_seen.get(spec.process_id, 0)),
                    int(role_progress.get("skipped_count", 0)),
                )
                receiver_latest_values.append(
                    int(receiver_latest_seen.get(spec.process_id, 0))
                )
                receiver_received_count += int(
                    receiver_received_seen.get(spec.process_id, 0)
                )
                receiver_skipped_count += int(
                    receiver_skipped_seen.get(spec.process_id, 0)
                )
                run_future = run_futures.get(spec.process_id)
                if run_future is not None and run_future.done():
                    receiver_done += 1

            receiver_latest_min = (
                min(receiver_latest_values) if receiver_latest_values else 0
            )
            receiver_latest_max = (
                max(receiver_latest_values) if receiver_latest_values else 0
            )
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(
                "[progress]",
                f"time={timestamp}",
                f"publisher_latest={publisher_latest_seen}/{num_versions}",
                f"publisher_events={publisher_events_seen}",
                f"receiver_done={receiver_done}/{len(receiver_specs)}",
                f"receiver_latest_min={receiver_latest_min}/{num_versions}",
                f"receiver_latest_max={receiver_latest_max}/{num_versions}",
                f"receiver_received_events={receiver_received_count}",
                f"receiver_skipped_events={receiver_skipped_count}",
                flush=True,
            )

            if stop_event.wait(max(1.0, poll_interval_s)):
                break

    monitor_thread = threading.Thread(
        target=_monitor_loop,
        name="weight-publisher-progress-monitor",
        daemon=True,
    )
    monitor_thread.start()
    return stop_event, monitor_thread


def wait_for_remote_file(
    process_id: str,
    path: str,
    *,
    timeout_sec: float,
    poll_sec: float = 1.0,
) -> None:
    deadline = time.monotonic() + max(1.0, timeout_sec)
    check_cmd = (
        "set -euo pipefail; "
        f"if [[ -f {shlex.quote(path)} ]]; then echo READY; else echo WAIT; fi"
    )
    while time.monotonic() < deadline:
        output = run_remote(process_id, check_cmd, timeout_sec=10.0).strip()
        if "READY" in output:
            return
        time.sleep(max(0.1, poll_sec))
    raise TimeoutError(
        f"timed out waiting remote file: process={process_id} path={path}"
    )


def wait_for_remote_file_or_fail_fast(
    process_id: str,
    path: str,
    *,
    future: concurrent.futures.Future[str],
    timeout_sec: float,
    poll_sec: float = 1.0,
) -> None:
    deadline = time.monotonic() + max(1.0, timeout_sec)
    check_cmd = (
        "set -euo pipefail; "
        f"if [[ -f {shlex.quote(path)} ]]; then echo READY; else echo WAIT; fi"
    )
    while time.monotonic() < deadline:
        if future.done():
            try:
                _ = future.result()
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "receiver role exited before producing output summary: "
                    f"process={process_id}, output={path}"
                ) from exc
        output = run_remote(process_id, check_cmd, timeout_sec=10.0).strip()
        if "READY" in output:
            return
        time.sleep(max(0.1, poll_sec))
    if future.done():
        try:
            _ = future.result()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "receiver role exited and output summary is missing: "
                f"process={process_id}, output={path}"
            ) from exc
    raise TimeoutError(
        f"timed out waiting remote file: process={process_id} path={path}"
    )


def terminate_remote_e2e_roles(
    *,
    process_ids: list[str],
    model_name: str,
    timeout_sec: float,
) -> list[dict[str, str]]:
    model_token = f"--model-name {str(model_name).strip()}"

    def _terminate_one(process_id: str) -> None:
        kill_cmd = (
            "set +e; "
            f"pkill -TERM -f -- {shlex.quote(model_token)} >/dev/null 2>&1 || true; "
            "sleep 2; "
            f"pkill -KILL -f -- {shlex.quote(model_token)} >/dev/null 2>&1 || true; "
            "exit 0"
        )
        _ = run_remote(
            process_id,
            kill_cmd,
            timeout_sec=max(5.0, timeout_sec),
        )

    if not process_ids:
        return []
    errors: list[dict[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, len(process_ids))
    ) as pool:
        futures = {
            process_id: pool.submit(_terminate_one, process_id)
            for process_id in process_ids
        }
        for process_id, future in futures.items():
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                errors.append({"process_id": process_id, "error": str(exc)})
    return errors


def probe_artifacts_on_process(
    *,
    process_id: str,
    repo_root: str,
    daemon_connect_address: str,
    artifact_ids: list[str],
    timeout_sec: float,
) -> dict[str, Any]:
    encoded_ids = json.dumps(artifact_ids, ensure_ascii=False)
    encoded_connect_addr = json.dumps(str(daemon_connect_address), ensure_ascii=False)
    python_script = f"""
import json

import tensorcast as tc
from tensorcast.api.store import artifact as resolve_artifact

artifact_ids = json.loads({encoded_ids!r})
connect_addr = json.loads({encoded_connect_addr!r})
results = {{}}
tc.init(mode="connect", address=connect_addr)
try:
    for artifact_id in artifact_ids:
        resolved = False
        exists = False
        try:
            artifact = resolve_artifact(artifact_id=artifact_id)
            resolved = True
            try:
                exists = bool(artifact.exists())
            except Exception:
                exists = False
        except Exception:
            resolved = False
        # Keep cluster audit lightweight for large payloads: materializable is
        # approximated by exists() and end-to-end apply correctness is covered
        # by receiver summaries.
        results[artifact_id] = {{
            "resolved": bool(resolved),
            "exists": bool(exists),
            "materializable": bool(exists),
        }}
    print(
        json.dumps(
            {{
                "init_mode": "connect",
                "probe_mode": "exists_only",
                "artifacts": results,
            }},
            ensure_ascii=False,
        )
    )
finally:
    tc.shutdown()
"""
    inner_cmd = (
        "set -euo pipefail; "
        f"cd {shlex.quote(repo_root)}; "
        "source .venv/bin/activate; "
        "python - <<'PY'\n"
        f"{python_script}\n"
        "PY"
    )
    output = run_remote(
        process_id,
        inner_cmd,
        timeout_sec=max(10.0, timeout_sec),
    ).strip()
    loaded = extract_last_json_object(output)
    if not isinstance(loaded, dict):
        raise RuntimeError(
            f"unexpected probe payload type for process={process_id}: {type(loaded)}"
        )
    return loaded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-host WeightPublisher runner: one publisher repeatedly publishes "
            "versions while multiple receivers update via binding.swap."
        )
    )
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--publisher-proc", required=True)
    parser.add_argument("--receiver-procs", required=True)
    parser.add_argument("--gs-addr", required=True)
    parser.add_argument(
        "--daemon-config",
        default="examples/config/store_daemon_config_cross_host_bench.yaml",
    )
    parser.add_argument(
        "--publisher-daemon-config",
        default="",
        help="Daemon config for publisher role. Defaults to --daemon-config when unset.",
    )
    parser.add_argument(
        "--receiver-daemon-config",
        default="",
        help="Daemon config for receiver roles. Defaults to --daemon-config when unset.",
    )
    parser.add_argument(
        "--daemon-connect-address",
        default="127.0.0.1:50052",
        help="SDK connect address for role/probe scripts. Must point to local daemon.",
    )
    parser.add_argument(
        "--daemon-p2p-port-base",
        type=int,
        default=65090,
        help="Daemon P2P listen base port; each role gets +index.",
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--model-name", default="")
    parser.add_argument("--start-version", type=int, default=1)
    parser.add_argument("--num-versions", type=int, default=6)
    parser.add_argument("--keep-last", type=int, default=2)
    parser.add_argument("--publish-interval-s", type=float, default=3.0)
    parser.add_argument("--poll-interval-s", type=float, default=0.5)
    parser.add_argument(
        "--daemon-heartbeat-interval",
        default="5s",
        help="Daemon high_availability.heartbeat_interval value (e.g. 5s, 60s).",
    )
    parser.add_argument(
        "--daemon-periodic-sync-interval",
        default="5s",
        help=("Daemon high_availability.periodic_sync_interval value (e.g. 5s, 60s)."),
    )
    parser.add_argument("--receiver-timeout-s", type=float, default=300.0)
    parser.add_argument(
        "--payload-mode",
        choices=["probe", "version_fill", "tp_ranked"],
        default="probe",
        help=(
            "Payload mode forwarded to weight_publisher_e2e. "
            "Use version_fill for scalar full-value checks, tp_ranked for TP rank-tagged payloads."
        ),
    )
    parser.add_argument(
        "--tp-world-size",
        type=int,
        default=1,
        help="TP world size passed to weight_publisher_e2e.",
    )
    parser.add_argument(
        "--tp-total-bytes",
        type=int,
        default=0,
        help="Total tp_ranked payload bytes passed to weight_publisher_e2e.",
    )
    parser.add_argument(
        "--tp-device-base-index",
        type=int,
        default=0,
        help="TP rank base device index (rank i uses cuda:{base+i}).",
    )
    parser.add_argument(
        "--tp-materialize-deadline-s",
        type=float,
        default=600.0,
        help=(
            "TP bind_into/swap deadline in seconds passed to weight_publisher_e2e. "
            "Use larger values for large payload fanout."
        ),
    )
    parser.add_argument(
        "--publish-device",
        default="auto",
        help="Publisher payload device passed to weight_publisher_e2e.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
        help=(
            "Global Store per-replica max_concurrency propagated via "
            "daemon unified config (promotion.max_concurrency)."
        ),
    )
    parser.add_argument(
        "--progress-poll-s",
        type=float,
        default=10.0,
        help="Progress heartbeat polling interval (seconds).",
    )
    parser.add_argument(
        "--max-publish-to-apply-s",
        type=float,
        default=30.0,
        help="Upper bound for publish->receiver-apply latency (seconds).",
    )
    parser.add_argument("--retention-timeout-s", type=float, default=90.0)
    parser.add_argument("--receiver-apply-mode", default="binding_swap")
    parser.add_argument(
        "--allow-receiver-skips",
        action="store_true",
        help=(
            "Allow receiver to skip dropped intermediate versions. "
            "Default is strict (no skip)."
        ),
    )
    parser.add_argument("--fallback-prefer", default="p2p")
    parser.add_argument("--materialize-device", default="cuda:0")
    parser.add_argument("--receiver-hold-after-finish-s", type=float, default=25.0)
    parser.add_argument("--publisher-hold-after-finish-s", type=float, default=30.0)
    parser.add_argument("--receiver-warmup-s", type=float, default=2.0)
    parser.add_argument(
        "--remote-timeout-sec",
        type=float,
        default=DEFAULT_REMOTE_TIMEOUT_SEC,
    )
    parser.add_argument(
        "--weights-root",
        default="/data/tensorcast_weight_publisher_e2e",
    )
    parser.add_argument(
        "--out-dir",
        default="/data/tc_cross_20260223/results_weight_publisher",
    )
    parser.add_argument("--cluster-id", default="")
    parser.add_argument(
        "--keep-daemons",
        action="store_true",
        help="Keep remote daemons running after case completion for manual debugging.",
    )
    parser.add_argument(
        "--preclean-roles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Kill stale weight_publisher_e2e and daemon processes on selected "
            "workers before starting this case."
        ),
    )
    return parser


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    receiver_procs = split_csv(args.receiver_procs)
    if not receiver_procs:
        raise ValueError("receiver-procs is empty")
    if args.num_versions <= 0:
        raise ValueError("num-versions must be > 0")
    if args.start_version <= 0:
        raise ValueError("start-version must be > 0")
    if args.keep_last < 0:
        raise ValueError("keep-last must be >= 0")
    if args.keep_last > args.num_versions:
        raise ValueError("keep-last must be <= num-versions")
    if int(args.tp_world_size) <= 0:
        raise ValueError("tp-world-size must be > 0")
    if int(args.tp_total_bytes) < 0:
        raise ValueError("tp-total-bytes must be >= 0")
    if int(args.tp_total_bytes) > 0 and str(args.payload_mode) != "tp_ranked":
        raise ValueError("tp-total-bytes requires payload-mode=tp_ranked")
    if int(args.tp_device_base_index) < 0:
        raise ValueError("tp-device-base-index must be >= 0")
    if float(args.tp_materialize_deadline_s) < 0.0:
        raise ValueError("tp-materialize-deadline-s must be >= 0")
    if int(args.max_concurrency) <= 0:
        raise ValueError("max-concurrency must be > 0")
    if float(args.progress_poll_s) <= 0:
        raise ValueError("progress-poll-s must be > 0")
    if float(args.remote_timeout_sec) <= 0:
        raise ValueError("remote-timeout-sec must be > 0")
    estimated_remote_timeout_sec = estimate_remote_timeout_floor_sec(args)
    configured_remote_timeout_sec = float(args.remote_timeout_sec)
    if (
        abs(configured_remote_timeout_sec - DEFAULT_REMOTE_TIMEOUT_SEC) < 1e-6
        and configured_remote_timeout_sec < estimated_remote_timeout_sec
    ):
        args.remote_timeout_sec = estimated_remote_timeout_sec
        print(
            "[preflight] auto-adjust remote-timeout-sec "
            f"{configured_remote_timeout_sec:.1f} -> {estimated_remote_timeout_sec:.1f} "
            "(large case conservative floor)"
        )
    elif configured_remote_timeout_sec < estimated_remote_timeout_sec:
        print(
            "[preflight] warning: configured remote-timeout-sec "
            f"{configured_remote_timeout_sec:.1f}s is lower than estimated floor "
            f"{estimated_remote_timeout_sec:.1f}s; role exec may timeout early"
        )
    if not str(args.daemon_heartbeat_interval).strip():
        raise ValueError("daemon-heartbeat-interval must be non-empty")
    if not str(args.daemon_periodic_sync_interval).strip():
        raise ValueError("daemon-periodic-sync-interval must be non-empty")
    daemon_connect_base = str(args.daemon_connect_address).strip()
    daemon_connect_host, daemon_connect_port = parse_host_port(daemon_connect_base)
    daemon_p2p_port_base = int(args.daemon_p2p_port_base)
    if daemon_p2p_port_base <= 0:
        raise ValueError("daemon-p2p-port-base must be > 0")

    run_tag = f"{int(time.time())}"
    run_id = f"{args.case_name}-{run_tag}"
    cluster_id = str(args.cluster_id).strip() or (discover_global_cluster_token() or "")
    out_dir = Path(str(args.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    case_dir = out_dir / run_id
    case_dir.mkdir(parents=True, exist_ok=True)

    model_base = str(args.model_name).strip() or f"wp-mh-{str(args.case_name).strip()}"
    model_name = f"{model_base}-{run_tag}"
    publisher_output = (case_dir / "publisher_summary.json").as_posix()
    publisher_log_file = (case_dir / "publisher.log").as_posix()
    receiver_outputs = {
        proc: (case_dir / f"receiver_{idx + 1}_summary.json").as_posix()
        for idx, proc in enumerate(receiver_procs)
    }
    receiver_log_files = {
        proc: (case_dir / f"receiver_{idx + 1}.log").as_posix()
        for idx, proc in enumerate(receiver_procs)
    }

    all_processes = [str(args.publisher_proc)] + receiver_procs
    if daemon_p2p_port_base + len(all_processes) - 1 > 65535:
        raise ValueError(
            "daemon-p2p-port-base is too large for role count (port > 65535)"
        )
    if bool(args.preclean_roles):
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(all_processes)
        ) as pool:
            preclean_futures = {
                process_id: pool.submit(
                    preclean_remote_role_processes,
                    process_id=process_id,
                    repo_root=str(args.repo_root),
                    timeout_sec=40.0,
                )
                for process_id in all_processes
            }
            for process_id, future in preclean_futures.items():
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        f"failed to pre-clean remote role process: process={process_id}"
                    ) from exc
    publisher_daemon_config = str(args.publisher_daemon_config).strip() or str(
        args.daemon_config
    )
    receiver_daemon_config = str(args.receiver_daemon_config).strip() or str(
        args.daemon_config
    )
    daemon_config_by_process = {
        str(args.publisher_proc): str(publisher_daemon_config),
        **{proc: str(receiver_daemon_config) for proc in receiver_procs},
    }
    publisher_memory_preflight = _validate_publisher_memory_budget(
        publisher_process_id=str(args.publisher_proc),
        daemon_config_path=str(publisher_daemon_config),
        payload_mode=str(args.payload_mode),
        tp_total_bytes=int(args.tp_total_bytes),
        keep_last=int(args.keep_last),
        publish_device=str(args.publish_device),
        timeout_sec=min(30.0, float(args.remote_timeout_sec)),
    )
    receiver_memory_preflight: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, len(receiver_procs))
    ) as pool:
        receiver_preflight_futures = {
            proc: pool.submit(
                _validate_receiver_memory_budget,
                receiver_process_id=proc,
                daemon_config_path=str(receiver_daemon_config),
                payload_mode=str(args.payload_mode),
                tp_world_size=int(args.tp_world_size),
                tp_total_bytes=int(args.tp_total_bytes),
                max_concurrency=int(args.max_concurrency),
                timeout_sec=min(30.0, float(args.remote_timeout_sec)),
            )
            for proc in receiver_procs
        }
        for proc, future in receiver_preflight_futures.items():
            receiver_memory_preflight[proc] = future.result()
    daemon_connect_by_process = {
        process_id: f"{daemon_connect_host}:{daemon_connect_port + index}"
        for index, process_id in enumerate(all_processes)
    }
    daemon_p2p_port_by_process = {
        process_id: int(daemon_p2p_port_base + index)
        for index, process_id in enumerate(all_processes)
    }
    daemon_sessions = {
        process_id: f"wp-{run_tag}-{index + 1}"
        for index, process_id in enumerate(all_processes)
    }
    daemon_ids = {
        process_id: f"weight-publisher-{run_tag}-{index + 1}"
        for index, process_id in enumerate(all_processes)
    }

    remote_advertise_hosts: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(all_processes)) as pool:
        ip_futures = {
            process_id: pool.submit(
                discover_remote_advertise_ip,
                process_id=process_id,
                timeout_sec=20.0,
            )
            for process_id in all_processes
        }
        for process_id, future in ip_futures.items():
            remote_advertise_hosts[process_id] = future.result()

    daemon_status_by_process: dict[str, dict[str, Any]] = {}
    daemon_cleanup_errors: list[dict[str, str]] = []

    def _cleanup_daemons() -> None:
        if bool(args.keep_daemons):
            return
        for process_id in reversed(all_processes):
            daemon_session = daemon_sessions.get(process_id, "")
            if not daemon_session:
                continue
            try:
                stop_remote_daemon(
                    process_id=process_id,
                    repo_root=str(args.repo_root),
                    daemon_session=daemon_session,
                    timeout_sec=40.0,
                )
            except Exception as exc:  # noqa: BLE001
                daemon_cleanup_errors.append(
                    {
                        "process_id": process_id,
                        "session": daemon_session,
                        "error": str(exc),
                    }
                )

    atexit.register(_cleanup_daemons)
    for process_id in all_processes:
        print(
            "[daemon-start] process="
            f"{process_id} session={daemon_sessions[process_id]} "
            f"connect={daemon_connect_by_process[process_id]} "
            f"p2p={daemon_p2p_port_by_process[process_id]}"
        )
        daemon_status_by_process[process_id] = start_remote_daemon(
            process_id=process_id,
            repo_root=str(args.repo_root),
            daemon_config=daemon_config_by_process[process_id],
            gs_addr=str(args.gs_addr),
            daemon_session=daemon_sessions[process_id],
            daemon_id=daemon_ids[process_id],
            daemon_connect_address=daemon_connect_by_process[process_id],
            daemon_p2p_port=daemon_p2p_port_by_process[process_id],
            advertise_host=remote_advertise_hosts[process_id],
            cluster_id=cluster_id,
            heartbeat_interval=str(args.daemon_heartbeat_interval),
            periodic_sync_interval=str(args.daemon_periodic_sync_interval),
            max_concurrency=int(args.max_concurrency),
            timeout_sec=120.0,
        )

    publisher_spec = RoleSpec(
        process_id=str(args.publisher_proc),
        output_json=publisher_output,
        log_file=publisher_log_file,
        inner_cmd=build_e2e_command(
            repo_root=str(args.repo_root),
            mode="publisher",
            daemon_connect_address=daemon_connect_by_process[str(args.publisher_proc)],
            model_name=model_name,
            start_version=int(args.start_version),
            num_versions=int(args.num_versions),
            keep_last=int(args.keep_last),
            publish_interval_s=float(args.publish_interval_s),
            poll_interval_s=float(args.poll_interval_s),
            receiver_timeout_s=float(args.receiver_timeout_s),
            retention_timeout_s=float(args.retention_timeout_s),
            payload_mode=str(args.payload_mode),
            tp_world_size=int(args.tp_world_size),
            tp_total_bytes=int(args.tp_total_bytes),
            tp_device_base_index=int(args.tp_device_base_index),
            tp_materialize_deadline_s=float(args.tp_materialize_deadline_s),
            publish_device=str(args.publish_device),
            output_json=publisher_output,
            weights_root=str(args.weights_root),
            run_id=f"{run_id}-publisher",
            receiver_apply_mode=str(args.receiver_apply_mode),
            fallback_prefer=str(args.fallback_prefer),
            materialize_device=str(args.materialize_device),
            allow_version_skip=bool(args.allow_receiver_skips),
            hold_after_finish_s=float(args.publisher_hold_after_finish_s),
            log_file=publisher_log_file,
            ready_file=None,
        ),
        ready_file=None,
    )

    receiver_specs = [
        RoleSpec(
            process_id=proc,
            output_json=receiver_outputs[proc],
            log_file=receiver_log_files[proc],
            ready_file=(case_dir / f"receiver_{idx + 1}.ready.json").as_posix(),
            inner_cmd=build_e2e_command(
                repo_root=str(args.repo_root),
                mode="receiver",
                daemon_connect_address=daemon_connect_by_process[proc],
                model_name=model_name,
                start_version=int(args.start_version),
                num_versions=int(args.num_versions),
                keep_last=int(args.keep_last),
                publish_interval_s=float(args.publish_interval_s),
                poll_interval_s=float(args.poll_interval_s),
                receiver_timeout_s=float(args.receiver_timeout_s),
                retention_timeout_s=float(args.retention_timeout_s),
                payload_mode=str(args.payload_mode),
                tp_world_size=int(args.tp_world_size),
                tp_total_bytes=int(args.tp_total_bytes),
                tp_device_base_index=int(args.tp_device_base_index),
                tp_materialize_deadline_s=float(args.tp_materialize_deadline_s),
                publish_device=str(args.publish_device),
                output_json=receiver_outputs[proc],
                weights_root=str(args.weights_root),
                run_id=f"{run_id}-receiver",
                receiver_apply_mode=str(args.receiver_apply_mode),
                fallback_prefer=str(args.fallback_prefer),
                materialize_device=str(args.materialize_device),
                allow_version_skip=bool(args.allow_receiver_skips),
                hold_after_finish_s=float(args.receiver_hold_after_finish_s),
                log_file=receiver_log_files[proc],
                ready_file=(case_dir / f"receiver_{idx + 1}.ready.json").as_posix(),
            ),
        )
        for idx, proc in enumerate(receiver_procs)
    ]

    receiver_logs: dict[str, str] = {}
    publisher_stdout = ""
    publisher_summary: dict[str, Any] = {}
    receiver_summaries: dict[str, dict[str, Any]] = {}
    gs_probe: dict[str, Any] = {}
    cluster_artifact_probe: dict[str, dict[str, Any]] = {}

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(receiver_specs) + 1
    ) as pool:
        receiver_futures = {
            spec.process_id: pool.submit(
                run_remote,
                spec.process_id,
                spec.inner_cmd,
                timeout_sec=float(args.remote_timeout_sec),
            )
            for spec in receiver_specs
        }
        for spec in receiver_specs:
            if spec.ready_file:
                wait_for_remote_file_or_fail_fast(
                    spec.process_id,
                    spec.ready_file,
                    future=receiver_futures[spec.process_id],
                    timeout_sec=float(args.receiver_timeout_s),
                )
        time.sleep(max(0.0, float(args.receiver_warmup_s)))

        publisher_future = pool.submit(
            run_remote,
            publisher_spec.process_id,
            publisher_spec.inner_cmd,
            timeout_sec=float(args.remote_timeout_sec),
        )
        run_futures: dict[str, concurrent.futures.Future[str]] = {
            **receiver_futures,
            publisher_spec.process_id: publisher_future,
        }
        progress_stop_event, progress_thread = start_progress_monitor(
            publisher_spec=publisher_spec,
            receiver_specs=receiver_specs,
            run_futures=run_futures,
            num_versions=int(args.num_versions),
            poll_interval_s=float(args.progress_poll_s),
        )
        try:
            publisher_stdout = publisher_future.result()
            publisher_summary = read_remote_json(
                publisher_spec.process_id,
                publisher_spec.output_json,
                timeout_sec=20.0,
            )

            published_events = publisher_summary.get("published", [])
            published_artifact_ids: list[str] = []
            if isinstance(published_events, list):
                for event in published_events:
                    if isinstance(event, dict):
                        artifact_id = str(event.get("artifact_id", "")).strip()
                        if artifact_id:
                            published_artifact_ids.append(artifact_id)
            published_artifact_ids = list(dict.fromkeys(published_artifact_ids))

            if published_artifact_ids:
                try:
                    replica_counts, audit_method = query_replica_counts_via_gs_rpc(
                        gs_addr=str(args.gs_addr),
                        artifact_ids=published_artifact_ids,
                        timeout_sec=15.0,
                    )
                    gs_probe = {
                        "gs_addr": str(args.gs_addr),
                        "audit_method": audit_method,
                        "replica_counts": replica_counts,
                    }
                except Exception as exc:  # noqa: BLE001
                    gs_probe = {
                        "gs_addr": str(args.gs_addr),
                        "audit_method": "gs_rpc",
                        "error": str(exc),
                    }

            for spec in receiver_specs:
                wait_for_remote_file_or_fail_fast(
                    spec.process_id,
                    spec.output_json,
                    future=receiver_futures[spec.process_id],
                    timeout_sec=(
                        float(args.receiver_timeout_s)
                        + float(args.receiver_hold_after_finish_s)
                        + 60.0
                    ),
                )

            if published_artifact_ids:
                probe_processes = [publisher_spec.process_id] + [
                    spec.process_id for spec in receiver_specs
                ]
                for process_id in probe_processes:
                    try:
                        cluster_artifact_probe[process_id] = probe_artifacts_on_process(
                            process_id=process_id,
                            repo_root=str(args.repo_root),
                            daemon_connect_address=daemon_connect_by_process[
                                process_id
                            ],
                            artifact_ids=published_artifact_ids,
                            timeout_sec=30.0,
                        )
                    except Exception as exc:  # noqa: BLE001
                        cluster_artifact_probe[process_id] = {"__error__": str(exc)}

            for spec in receiver_specs:
                receiver_logs[spec.process_id] = receiver_futures[
                    spec.process_id
                ].result()
                receiver_summaries[spec.process_id] = read_remote_json(
                    spec.process_id,
                    spec.output_json,
                    timeout_sec=20.0,
                )
        except Exception as exc:
            terminate_errors = terminate_remote_e2e_roles(
                process_ids=[spec.process_id for spec in receiver_specs]
                + [publisher_spec.process_id],
                model_name=model_name,
                timeout_sec=15.0,
            )
            if terminate_errors:
                raise RuntimeError(
                    "cross-host runner failed and role termination reported errors: "
                    + json.dumps(terminate_errors, ensure_ascii=False)
                ) from exc
            raise
        finally:
            progress_stop_event.set()
            progress_thread.join(timeout=5.0)

    published = publisher_summary.get("published", [])
    published_by_version: dict[int, dict[str, Any]] = {}
    if isinstance(published, list):
        for row in published:
            if not isinstance(row, dict):
                continue
            version = int(row.get("version", 0))
            if version > 0:
                published_by_version[version] = row
    ordered_versions = sorted(published_by_version)
    expected_versions = list(
        range(int(args.start_version), int(args.start_version) + int(args.num_versions))
    )

    keep_last = int(args.keep_last)
    kept_versions = ordered_versions[-keep_last:] if keep_last > 0 else []
    dropped_versions = (
        ordered_versions[:-keep_last] if keep_last > 0 else ordered_versions
    )
    kept_artifact_ids = {
        str(published_by_version[v].get("artifact_id", "")).strip()
        for v in kept_versions
    }
    dropped_artifact_ids = {
        str(published_by_version[v].get("artifact_id", "")).strip()
        for v in dropped_versions
    }
    kept_artifact_ids.discard("")
    dropped_artifact_ids.discard("")

    publish_latencies = [
        float(row.get("publish_latency_s", 0.0))
        for row in published_by_version.values()
        if isinstance(row, dict)
    ]
    apply_latencies: list[float] = []
    propagation_latencies: list[float] = []
    pointer_stability_violations: list[dict[str, Any]] = []
    receiver_sequence_failures: list[dict[str, Any]] = []
    receiver_skips: list[dict[str, Any]] = []
    receiver_mode_failures: list[dict[str, Any]] = []
    propagation_violations: list[dict[str, Any]] = []

    publish_ts_by_version = {
        version: float(row.get("published_at_s", 0.0))
        for version, row in published_by_version.items()
    }

    for proc, summary in receiver_summaries.items():
        received_rows = summary.get("received", [])
        if not isinstance(received_rows, list):
            receiver_sequence_failures.append(
                {
                    "process_id": proc,
                    "reason": "received payload is not a list",
                }
            )
            continue
        versions = [
            int(row.get("version", 0)) for row in received_rows if isinstance(row, dict)
        ]
        expected_set = set(expected_versions)
        unique_sorted_versions = sorted(set(versions))
        has_order_or_dup_issue = versions != unique_sorted_versions
        out_of_expected_range = [
            version for version in versions if version not in expected_set
        ]
        missing_versions = [
            version for version in expected_versions if version not in set(versions)
        ]
        latest_expected = expected_versions[-1] if expected_versions else 0
        reached_latest = bool(versions) and versions[-1] == latest_expected
        if missing_versions:
            receiver_skips.append(
                {
                    "process_id": proc,
                    "missing_versions": missing_versions,
                }
            )
        skip_disallowed_violation = bool(missing_versions) and not bool(
            args.allow_receiver_skips
        )
        if (
            has_order_or_dup_issue
            or out_of_expected_range
            or not reached_latest
            or skip_disallowed_violation
        ):
            receiver_sequence_failures.append(
                {
                    "process_id": proc,
                    "expected_versions": expected_versions,
                    "actual_versions": versions,
                    "missing_versions": missing_versions,
                    "order_or_duplicate_issue": has_order_or_dup_issue,
                    "out_of_expected_range": out_of_expected_range,
                    "reached_latest_version": reached_latest,
                    "allow_receiver_skips": bool(args.allow_receiver_skips),
                }
            )
        for idx, row in enumerate(received_rows):
            if not isinstance(row, dict):
                continue
            apply_latencies.append(float(row.get("materialize_latency_s", 0.0)))
            mode = str(row.get("apply_mode", ""))
            op = str(row.get("apply_operation", ""))
            if mode != str(args.receiver_apply_mode):
                receiver_mode_failures.append(
                    {
                        "process_id": proc,
                        "version": int(row.get("version", 0)),
                        "expected_mode": str(args.receiver_apply_mode),
                        "actual_mode": mode,
                    }
                )
            receiver_mode = str(args.receiver_apply_mode)
            if receiver_mode in {
                "binding_swap",
                "tp_bind_into_swap",
                "tp4_bind_into_swap",
            }:
                if receiver_mode == "binding_swap":
                    expected_op = "bind" if idx == 0 else "swap"
                else:
                    expected_op = "bind_into" if idx == 0 else "swap"
                if op != expected_op:
                    receiver_mode_failures.append(
                        {
                            "process_id": proc,
                            "version": int(row.get("version", 0)),
                            "expected_operation": expected_op,
                            "actual_operation": op,
                        }
                    )
                if idx > 0 and not bool(row.get("pointer_stable", False)):
                    pointer_stability_violations.append(
                        {
                            "process_id": proc,
                            "version": int(row.get("version", 0)),
                            "pointer_stable": row.get("pointer_stable"),
                        }
                    )
            version = int(row.get("version", 0))
            published_at_s = publish_ts_by_version.get(version)
            received_at_s = float(row.get("received_at_s", 0.0))
            if published_at_s is not None and received_at_s > 0:
                delta = received_at_s - published_at_s
                if delta >= 0:
                    propagation_latencies.append(delta)
                    if delta > float(args.max_publish_to_apply_s):
                        propagation_violations.append(
                            {
                                "process_id": proc,
                                "version": version,
                                "publish_to_apply_s": delta,
                                "threshold_s": float(args.max_publish_to_apply_s),
                            }
                        )

    gs_checks: dict[str, Any] = {
        "enabled": bool(published_by_version),
        "audit_method": gs_probe.get("audit_method", "gs_rpc"),
        "error": gs_probe.get("error") if gs_probe else "missing gs probe payload",
        "old_versions_released": None,
        "replica_versions_within_window": None,
        "replica_version_count_within_limit": None,
    }
    if gs_checks["enabled"] and "replica_counts" in gs_probe:
        gs_checks["error"] = None
        counts = gs_probe.get("replica_counts", {})
        if isinstance(counts, dict):
            old_ok = True
            replica_artifacts: set[str] = set()
            for artifact_id, row in counts.items():
                if not isinstance(row, dict):
                    continue
                replica_count = int(row.get("replica_count", 0))
                if replica_count > 0:
                    replica_artifacts.add(artifact_id)
                if artifact_id in dropped_artifact_ids and replica_count != 0:
                    old_ok = False
            gs_checks["old_versions_released"] = old_ok
            gs_checks["replica_versions_within_window"] = replica_artifacts.issubset(
                kept_artifact_ids
            )
            gs_checks["replica_version_count_within_limit"] = (
                len(replica_artifacts) <= keep_last
            )
            gs_checks["replica_artifacts"] = sorted(replica_artifacts)
            gs_checks["kept_artifacts"] = sorted(kept_artifact_ids)
            gs_checks["dropped_artifacts"] = sorted(dropped_artifact_ids)
    elif gs_checks["enabled"]:
        gs_checks["old_versions_released"] = False
        gs_checks["replica_versions_within_window"] = False
        gs_checks["replica_version_count_within_limit"] = False

    probe_checks: dict[str, Any] = {
        "enabled": bool(cluster_artifact_probe),
        "probe_errors": [],
        "dropped_non_materializable": None,
        "dropped_not_exists": None,
        "materializable_versions_within_window": None,
        "materializable_version_count_within_limit": None,
        "materializable_artifacts": [],
        "probe_init_modes": {},
    }
    if cluster_artifact_probe:
        dropped_non_materializable = True
        dropped_not_exists = True
        materializable_artifacts: set[str] = set()
        probe_errors: list[dict[str, str]] = []
        probe_init_modes: dict[str, str] = {}
        for process_id, payload in cluster_artifact_probe.items():
            if "__error__" in payload:
                probe_errors.append(
                    {"process_id": process_id, "error": str(payload["__error__"])}
                )
                continue
            artifact_states: dict[str, Any]
            if isinstance(payload.get("artifacts"), dict):
                artifact_states = payload.get("artifacts", {})
                init_mode = str(payload.get("init_mode", "")).strip()
                if init_mode:
                    probe_init_modes[process_id] = init_mode
            else:
                artifact_states = payload
            for artifact_id, state in artifact_states.items():
                if not isinstance(state, dict):
                    continue
                exists = bool(state.get("exists", False))
                materializable = bool(state.get("materializable", False))
                if materializable:
                    materializable_artifacts.add(artifact_id)
                if artifact_id in dropped_artifact_ids and materializable:
                    dropped_non_materializable = False
                if artifact_id in dropped_artifact_ids and exists:
                    dropped_not_exists = False

        probe_checks["probe_errors"] = probe_errors
        probe_checks["dropped_non_materializable"] = dropped_non_materializable
        probe_checks["dropped_not_exists"] = dropped_not_exists
        probe_checks["materializable_versions_within_window"] = (
            materializable_artifacts.issubset(kept_artifact_ids)
        )
        probe_checks["materializable_version_count_within_limit"] = (
            len(materializable_artifacts) <= keep_last
        )
        probe_checks["materializable_artifacts"] = sorted(materializable_artifacts)
        probe_checks["probe_init_modes"] = probe_init_modes

    summary = {
        "case_name": str(args.case_name),
        "model_name": model_name,
        "publisher_process": str(args.publisher_proc),
        "receiver_processes": receiver_procs,
        "versions": {
            "expected": expected_versions,
            "published": ordered_versions,
            "keep_last": keep_last,
            "kept_versions": kept_versions,
            "dropped_versions": dropped_versions,
        },
        "performance": {
            "publish_latency_s": summarize_series(publish_latencies),
            "apply_latency_s": summarize_series(apply_latencies),
            "publish_to_apply_s": summarize_series(propagation_latencies),
        },
        "stability": {
            "all_receivers_completed": not receiver_sequence_failures,
            "receiver_skips_present": bool(receiver_skips),
            "allow_receiver_skips": bool(args.allow_receiver_skips),
            "binding_pointer_stable": not pointer_stability_violations,
            "receiver_mode_consistent": not receiver_mode_failures,
            "publish_to_apply_within_limit": not propagation_violations,
            "mechanism_keep_last_window_validated": bool(published_by_version),
        },
        "retention_checks": {
            "global_store_probe": gs_checks,
            "cluster_probe": probe_checks,
        },
        "failures": {
            "receiver_sequence_failures": receiver_sequence_failures,
            "receiver_skips": receiver_skips,
            "receiver_mode_failures": receiver_mode_failures,
            "pointer_stability_violations": pointer_stability_violations,
            "propagation_violations": propagation_violations,
        },
    }
    passed = (
        summary["stability"]["all_receivers_completed"]
        and summary["stability"]["binding_pointer_stable"]
        and summary["stability"]["receiver_mode_consistent"]
        and summary["stability"]["publish_to_apply_within_limit"]
    )
    if gs_checks["enabled"]:
        passed = passed and not bool(gs_checks.get("error"))
        passed = passed and bool(gs_checks.get("old_versions_released"))
        passed = passed and bool(gs_checks.get("replica_versions_within_window"))
        passed = passed and bool(gs_checks.get("replica_version_count_within_limit"))
    if probe_checks["enabled"]:
        passed = passed and not bool(probe_checks.get("probe_errors"))
        passed = passed and bool(probe_checks.get("dropped_non_materializable"))
        passed = passed and bool(
            probe_checks.get("materializable_versions_within_window")
        )
        passed = passed and bool(
            probe_checks.get("materializable_version_count_within_limit")
        )
    summary["passed"] = bool(passed)

    payload = {
        "summary": summary,
        "params": {
            "case_name": str(args.case_name),
            "publisher_proc": str(args.publisher_proc),
            "receiver_procs": receiver_procs,
            "gs_addr": str(args.gs_addr),
            "daemon_config": str(args.daemon_config),
            "publisher_daemon_config": str(publisher_daemon_config),
            "receiver_daemon_config": str(receiver_daemon_config),
            "daemon_connect_base": daemon_connect_base,
            "daemon_connect_addresses": daemon_connect_by_process,
            "daemon_p2p_port_base": daemon_p2p_port_base,
            "daemon_p2p_ports": daemon_p2p_port_by_process,
            "repo_root": str(args.repo_root),
            "start_version": int(args.start_version),
            "num_versions": int(args.num_versions),
            "keep_last": int(args.keep_last),
            "publish_interval_s": float(args.publish_interval_s),
            "poll_interval_s": float(args.poll_interval_s),
            "receiver_timeout_s": float(args.receiver_timeout_s),
            "payload_mode": str(args.payload_mode),
            "tp_world_size": int(args.tp_world_size),
            "tp_total_bytes": int(args.tp_total_bytes),
            "tp_device_base_index": int(args.tp_device_base_index),
            "tp_materialize_deadline_s": float(args.tp_materialize_deadline_s),
            "max_concurrency": int(args.max_concurrency),
            "publish_device": str(args.publish_device),
            "progress_poll_s": float(args.progress_poll_s),
            "max_publish_to_apply_s": float(args.max_publish_to_apply_s),
            "retention_timeout_s": float(args.retention_timeout_s),
            "receiver_apply_mode": str(args.receiver_apply_mode),
            "allow_receiver_skips": bool(args.allow_receiver_skips),
            "fallback_prefer": str(args.fallback_prefer),
            "materialize_device": str(args.materialize_device),
            "receiver_hold_after_finish_s": float(args.receiver_hold_after_finish_s),
            "publisher_hold_after_finish_s": float(args.publisher_hold_after_finish_s),
            "receiver_warmup_s": float(args.receiver_warmup_s),
            "remote_timeout_sec": float(args.remote_timeout_sec),
            "cluster_id": cluster_id,
            "daemon_heartbeat_interval": str(args.daemon_heartbeat_interval),
            "daemon_periodic_sync_interval": str(args.daemon_periodic_sync_interval),
            "preclean_roles": bool(args.preclean_roles),
            "publisher_memory_preflight": publisher_memory_preflight,
            "receiver_memory_preflight": receiver_memory_preflight,
        },
        "daemon": {
            "sessions": daemon_sessions,
            "daemon_ids": daemon_ids,
            "advertise_hosts": remote_advertise_hosts,
            "status": daemon_status_by_process,
            "keep_daemons": bool(args.keep_daemons),
            "daemon_config_by_process": daemon_config_by_process,
        },
        "publisher_stdout": publisher_stdout,
        "receiver_logs": receiver_logs,
        "publisher_log_file": publisher_log_file,
        "receiver_log_files": receiver_log_files,
        "publisher_summary": publisher_summary,
        "receiver_summaries": receiver_summaries,
        "cluster_artifact_probe": cluster_artifact_probe,
        "global_store_probe": gs_probe,
    }

    output_path = case_dir / f"{args.case_name}.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        "SUMMARY "
        + json.dumps(
            {
                "case_name": str(args.case_name),
                "passed": summary["passed"],
                "receivers": len(receiver_procs),
                "publish_latency_mean_s": summary["performance"]["publish_latency_s"][
                    "mean"
                ],
                "apply_latency_mean_s": summary["performance"]["apply_latency_s"][
                    "mean"
                ],
                "publish_to_apply_p95_s": summary["performance"]["publish_to_apply_s"][
                    "p95"
                ],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    print(f"OUTPUT {output_path}", flush=True)
    if bool(args.keep_daemons):
        atexit.unregister(_cleanup_daemons)
    else:
        _cleanup_daemons()
        atexit.unregister(_cleanup_daemons)
    if daemon_cleanup_errors:
        print(
            "DAEMON_CLEANUP_WARN "
            + json.dumps(daemon_cleanup_errors, ensure_ascii=False),
            flush=True,
        )
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
