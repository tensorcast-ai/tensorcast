#!/usr/bin/env python3
#  Copyright (c) 2026, TensorCast Team.

# Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import argparse
import atexit
import concurrent.futures
import json
import math
import os
import pwd
import re
import secrets
import shlex
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import grpc
import yaml
from google.protobuf import timestamp_pb2

from tensorcast.global_store.cluster_runtime_rpc import call_cluster_runtime_rpc
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
TRANSPORT_GROUP_KIND_TP_VERSION = "tp_version"
TRANSPORT_GROUP_DEFAULT_PRIORITY = 0
REMOTE_USER_RE = re.compile(r"^[a-z_][a-z0-9_.-]{0,63}$")
_REMOTE_RUN_AS_USER = ""


@dataclass(frozen=True)
class RoleSpec:
    process_id: str
    output_json: str
    inner_cmd: str
    log_file: str
    ready_file: str | None = None


@dataclass(frozen=True)
class TransportGroupPlan:
    mode: str
    kind: str
    total_parts: int
    priority: int


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


def estimate_receiver_timeout_floor_sec(args: argparse.Namespace) -> float:
    apply_mode = str(args.receiver_apply_mode).strip().lower()
    if apply_mode not in {"tp_bind_into_swap", "tp4_bind_into_swap"}:
        return max(30.0, float(args.receiver_timeout_s))
    tp_world_size = max(1, int(args.tp_world_size))
    total_payload_gib = float(max(0, int(args.tp_total_bytes))) / float(1024**3)
    per_rank_payload_gib = (
        total_payload_gib / float(tp_world_size)
        if total_payload_gib > 0.0
        else 0.0
    )
    # Empirical lower bound from multi-host TP4 runs:
    # rank-local bind/swap latency grows with per-rank payload and fanout.
    per_rank_apply_budget_s = max(12.0, per_rank_payload_gib * 2.0 + 8.0)
    receiver_count = max(1, len(split_csv(str(args.receiver_procs))))
    queue_budget_s = max(10.0, float(receiver_count) * 5.0)
    visibility_budget_s = max(20.0, float(args.publish_interval_s) * 2.0)
    guard_s = 20.0
    floor = (
        visibility_budget_s
        + float(tp_world_size) * per_rank_apply_budget_s
        + queue_budget_s
        + guard_s
    )
    return float(int(math.ceil(floor)))


def estimate_keep_last_floor(args: argparse.Namespace) -> int:
    keep_last = max(0, int(args.keep_last))
    if str(args.payload_mode).strip().lower() != "tp_ranked":
        return keep_last
    if int(args.tp_total_bytes) <= 0:
        return keep_last
    apply_mode = str(args.receiver_apply_mode).strip().lower()
    if apply_mode not in {"tp_bind_into_swap", "tp4_bind_into_swap"}:
        return keep_last
    if bool(args.allow_receiver_skips):
        return keep_last
    receiver_count = max(1, len(split_csv(str(args.receiver_procs))))
    if receiver_count <= 1:
        return keep_last
    return min(max(1, int(args.num_versions)), max(2, keep_last))


def estimate_keep_last_stable_cap(args: argparse.Namespace) -> int | None:
    if str(args.payload_mode).strip().lower() != "tp_ranked":
        return None
    per_version_bytes = int(args.tp_total_bytes)
    if per_version_bytes <= 0:
        return None
    publisher_daemon_config = str(
        getattr(args, "publisher_daemon_config", "") or getattr(args, "daemon_config", "")
    ).strip()
    if not publisher_daemon_config:
        return None
    daemon_hints = _load_daemon_memory_hints(publisher_daemon_config)
    stable_bytes = int(daemon_hints.get("stable_bytes", 0))
    if stable_bytes <= 0:
        return None
    return max(0, stable_bytes // per_version_bytes)


def estimate_publish_to_apply_floor_sec(args: argparse.Namespace) -> float:
    apply_mode = str(args.receiver_apply_mode).strip().lower()
    configured = float(args.max_publish_to_apply_s)
    if apply_mode not in {"tp_bind_into_swap", "tp4_bind_into_swap"}:
        return max(20.0, configured)
    if str(args.payload_mode).strip().lower() != "tp_ranked":
        return max(20.0, configured)
    if int(args.tp_total_bytes) <= 0:
        return max(20.0, configured)

    tp_world_size = max(1, int(args.tp_world_size))
    receiver_count = max(1, len(split_csv(str(args.receiver_procs))))
    total_payload_gib = float(max(0, int(args.tp_total_bytes))) / float(1024**3)
    per_rank_payload_gib = total_payload_gib / float(tp_world_size)
    per_rank_apply_budget_s = max(4.0, per_rank_payload_gib * 1.2 + 3.0)
    tp_wave_budget_s = float(tp_world_size) * per_rank_apply_budget_s * 0.4
    queue_budget_s = float(max(0, receiver_count - 1)) * max(
        1.5, per_rank_apply_budget_s * 0.25
    )
    visibility_budget_s = max(8.0, float(args.publish_interval_s) * 2.0)
    guard_s = 10.0 if str(args.transport_group_mode).strip().lower() == "tp_version" else 6.0
    floor = visibility_budget_s + tp_wave_budget_s + queue_budget_s + guard_s
    return float(max(20.0, math.ceil(floor)))


def parse_receiver_skip_events(receiver_log: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in str(receiver_log).splitlines():
        if "[receiver] skipped" not in line:
            continue
        version_match = re.search(r"\bversion=(\d+)\b", line)
        if version_match is None:
            continue
        version = int(version_match.group(1))
        reason_match = re.search(r"\breason=([^\s]+)", line)
        reason = (
            str(reason_match.group(1)).strip().lower()
            if reason_match is not None
            else "unknown"
        )
        newer_version_match = re.search(r"\bnewer_version=([^\s]+)", line)
        newer_version_raw = (
            str(newer_version_match.group(1)).strip()
            if newer_version_match is not None
            else ""
        )
        newer_version: int | None = None
        if newer_version_raw.isdigit():
            newer_version = int(newer_version_raw)
        events.append(
            {
                "version": version,
                "reason": reason,
                "newer_version": newer_version,
                "line": line.strip(),
            }
        )
    return events


def collect_receiver_skip_events_by_process(
    receiver_logs: dict[str, str],
) -> dict[str, list[dict[str, Any]]]:
    return {
        str(process_id): parse_receiver_skip_events(str(log_text))
        for process_id, log_text in receiver_logs.items()
    }


def assess_receiver_sequence(
    *,
    expected_versions: list[int],
    actual_versions: list[int],
    allow_receiver_skips: bool,
    explicit_skipped_versions: set[int],
) -> dict[str, Any]:
    expected_set = set(expected_versions)
    actual_set = set(actual_versions)
    unique_sorted_versions = sorted(actual_set)
    has_order_or_dup_issue = actual_versions != unique_sorted_versions
    out_of_expected_range = [
        version for version in actual_versions if version not in expected_set
    ]
    missing_versions = [
        version for version in expected_versions if version not in actual_set
    ]
    latest_expected = expected_versions[-1] if expected_versions else 0
    reached_latest = bool(actual_versions) and actual_versions[-1] == latest_expected

    effective_explicit_skips = sorted(
        version for version in explicit_skipped_versions if version in expected_set
    )
    effective_explicit_skip_set = set(effective_explicit_skips)
    accounted_missing_versions = (
        [
            version
            for version in missing_versions
            if version in effective_explicit_skip_set
        ]
        if allow_receiver_skips
        else []
    )
    unaccounted_missing_versions = (
        [
            version
            for version in missing_versions
            if version not in effective_explicit_skip_set
        ]
        if allow_receiver_skips
        else list(missing_versions)
    )
    skip_disallowed_violation = bool(missing_versions) and not allow_receiver_skips
    latest_violation = (not reached_latest) and not allow_receiver_skips
    missing_violation = bool(unaccounted_missing_versions)

    is_failure = bool(
        has_order_or_dup_issue
        or out_of_expected_range
        or skip_disallowed_violation
        or latest_violation
        or missing_violation
    )
    return {
        "expected_versions": expected_versions,
        "actual_versions": actual_versions,
        "missing_versions": missing_versions,
        "order_or_duplicate_issue": has_order_or_dup_issue,
        "out_of_expected_range": out_of_expected_range,
        "reached_latest_version": reached_latest,
        "allow_receiver_skips": bool(allow_receiver_skips),
        "explicit_skipped_versions": effective_explicit_skips,
        "accounted_missing_versions": accounted_missing_versions,
        "unaccounted_missing_versions": unaccounted_missing_versions,
        "is_failure": is_failure,
    }


def _infer_timeout_reasons_from_text(text: str) -> tuple[set[str], set[str]]:
    lowered = str(text).lower()
    timeout_tokens = ("deadline exceeded", "timed out", "timeout")
    has_timeout_signal = any(token in lowered for token in timeout_tokens)
    waiting_reasons: set[str] = set()
    transport_reasons: set[str] = set()

    if any(
        token in lowered
        for token in (
            "state=key_mapping_absent",
            "key_mapping_absent",
            "receiver summary missing",
            "timed out waiting remote file",
            "no available replicas",
            "retrying source selection",
        )
    ):
        waiting_reasons.add("queue_or_visibility_wait")
    if any(
        token in lowered
        for token in (
            "version_deregistered",
            "region is poisoned",
            "artifact id",
            "tensor not found",
            "unaccounted_missing_versions",
        )
    ):
        waiting_reasons.add("version_window_evicted")
    if any(
        token in lowered
        for token in (
            "group contract violation",
            "duplicate part_id in transport history",
            "artifact/view/total_parts mismatch",
        )
    ):
        waiting_reasons.add("group_contract_conflict")

    if any(
        token in lowered
        for token in (
            "deadline exceeded",
            "statuscode.deadline_exceeded",
            "grpc_status:4",
            "client_rpc_retry_suppressed",
        )
    ):
        transport_reasons.add("deadline_exceeded")
    if any(
        token in lowered
        for token in (
            "connection reset",
            "connection refused",
            "peer closed connection",
            "failed to read chunk",
            "failed to connect",
            "epollrdhup",
        )
    ):
        transport_reasons.add("connection_or_io_error")

    if has_timeout_signal and not waiting_reasons and not transport_reasons:
        waiting_reasons.add("timeout_unknown")
    return waiting_reasons, transport_reasons


def summarize_timeout_reasons(
    *,
    receiver_logs: dict[str, str],
    receiver_sequence_failures: list[dict[str, Any]],
) -> dict[str, Any]:
    waiting_by_process: dict[str, set[str]] = {}
    transport_by_process: dict[str, set[str]] = {}

    def _ensure_proc(process_id: str) -> None:
        waiting_by_process.setdefault(process_id, set())
        transport_by_process.setdefault(process_id, set())

    for process_id, log_text in receiver_logs.items():
        process_key = str(process_id)
        _ensure_proc(process_key)
        waiting, transport = _infer_timeout_reasons_from_text(str(log_text))
        waiting_by_process[process_key].update(waiting)
        transport_by_process[process_key].update(transport)

    for idx, failure in enumerate(receiver_sequence_failures):
        if not isinstance(failure, dict):
            continue
        process_key = str(failure.get("process_id", "")).strip()
        if not process_key:
            process_key = f"unknown-{idx}"
        _ensure_proc(process_key)
        reason_text = str(failure.get("reason", ""))
        waiting, transport = _infer_timeout_reasons_from_text(reason_text)
        waiting_by_process[process_key].update(waiting)
        transport_by_process[process_key].update(transport)
        unaccounted_missing = failure.get("unaccounted_missing_versions", [])
        if isinstance(unaccounted_missing, list) and unaccounted_missing:
            waiting_by_process[process_key].add("version_window_evicted")

    waiting_counts: Counter[str] = Counter()
    transport_counts: Counter[str] = Counter()
    for reasons in waiting_by_process.values():
        for reason in reasons:
            waiting_counts[str(reason)] += 1
    for reasons in transport_by_process.values():
        for reason in reasons:
            transport_counts[str(reason)] += 1

    return {
        "waiting_timeout_reason_counts": dict(waiting_counts),
        "transport_timeout_reason_counts": dict(transport_counts),
        "waiting_timeout_observed": bool(waiting_counts),
        "transport_timeout_observed": bool(transport_counts),
    }


def split_csv(raw: str) -> list[str]:
    values = [item.strip() for item in str(raw).split(",")]
    return [item for item in values if item]


def normalize_cuda_backend(raw: str) -> str:
    value = str(raw).strip().lower()
    if not value:
        return ""
    if value not in {"real", "fake"}:
        raise ValueError(f"cuda-backend must be one of: real, fake, empty; got {raw!r}")
    return value


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
            # Fall through to uid-based lookup if USER is unavailable or malformed.
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


def derive_transport_group_plan(
    *,
    mode: str,
    receiver_count: int,
    tp_world_size: int,
) -> TransportGroupPlan:
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"none", "tp_version"}:
        raise ValueError(f"unsupported transport-group-mode={normalized_mode!r}")
    if normalized_mode == "none":
        return TransportGroupPlan(
            mode="none",
            kind="",
            total_parts=0,
            priority=TRANSPORT_GROUP_DEFAULT_PRIORITY,
        )

    total_parts = int(receiver_count) * int(tp_world_size)
    if total_parts <= 0:
        raise ValueError(
            "transport-group-mode=tp_version requires receiver_count * tp_world_size > 0"
        )
    return TransportGroupPlan(
        mode="tp_version",
        kind=TRANSPORT_GROUP_KIND_TP_VERSION,
        total_parts=total_parts,
        priority=TRANSPORT_GROUP_DEFAULT_PRIORITY,
    )


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


def _parse_bool_literal(value: Any) -> bool:
    if isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"invalid bool literal: {value!r}")


def _load_daemon_memory_hints(daemon_config_path: str) -> dict[str, Any]:
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
    cpu_shared_memory = (
        engine.get("cpu_shared_memory", {}) if isinstance(engine, dict) else {}
    )
    cpu_shared_memory_enabled = (
        _parse_bool_literal(cpu_shared_memory.get("enabled", False))
        if isinstance(cpu_shared_memory, dict)
        else False
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
        "cpu_shared_memory_enabled": bool(cpu_shared_memory_enabled),
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
            "cpu_shared_memory_enabled": bool(
                daemon_hints.get("cpu_shared_memory_enabled", True)
            ),
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
            "or keep publish_device=cpu, or implement stage_on_gpu=false streaming for DRAM_STABLE."
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
    transient_overlap_hint: int,
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
        "transient_overlap_hint": int(transient_overlap_hint),
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
    # Receiver estimate for tp_ranked:
    # - one stable resident window for local rank payload
    # - transient transfer/materialization windows controlled by overlap hint
    #   (serial by default; not tied to max_concurrency in this runner)
    # - local TP rank target tensors (~1/world_size of full payload)
    rank_target_bytes = int(tp_total_bytes) // world_size
    transient_window_factor = max(1, int(transient_overlap_hint))
    transient_materialization_bytes = rank_target_bytes * transient_window_factor
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
    if stable_bytes < rank_target_bytes:
        violations.append(
            "stable_bytes is smaller than one local TP-rank window "
            f"(stable={_format_gib(stable_bytes):.1f}GiB, "
            f"rank_target={_format_gib(rank_target_bytes):.1f}GiB)"
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
            "increase worker memory, reduce tp_total_bytes, or lower "
            "receiver-preflight-transient-overlap."
        )
    return report


def _validate_daemon_memfd_required(
    *,
    process_id: str,
    role: str,
    daemon_config_path: str,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "required": True,
        "process_id": str(process_id),
        "role": str(role),
        "daemon_config_path": str(daemon_config_path),
        "checked": True,
        "cpu_shared_memory_enabled": None,
        "safe": True,
    }
    daemon_hints = _load_daemon_memory_hints(str(daemon_config_path))
    cpu_shared_memory_enabled = bool(
        daemon_hints.get("cpu_shared_memory_enabled", True)
    )
    report["cpu_shared_memory_enabled"] = bool(cpu_shared_memory_enabled)
    report["safe"] = bool(cpu_shared_memory_enabled)
    if not cpu_shared_memory_enabled:
        raise RuntimeError(
            "daemon memfd preflight failed: "
            "all benchmark roles require daemon config "
            "engine.cpu_shared_memory.enabled=true. "
            f"role={role}, process={process_id}, config={daemon_config_path}"
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


def probe_process_state(process_id: str, *, timeout_sec: float) -> dict[str, Any]:
    cmd = [
        "brainctl",
        "get",
        "process",
        str(process_id),
        "-n",
        "shai-core",
    ]
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=max(1.0, timeout_sec),
    )
    stdout = str(proc.stdout or "")
    stderr = str(proc.stderr or "")
    merged = (stdout + "\n" + stderr).strip()
    lowered = merged.lower()
    detail = merged[-500:] if merged else ""
    if "notfound" in lowered or "not found" in lowered:
        return {
            "exists": False,
            "status": "NotFound",
            "rc": int(proc.returncode),
            "detail_tail": detail,
        }
    if proc.returncode != 0:
        return {
            "exists": None,
            "status": "Unknown",
            "rc": int(proc.returncode),
            "detail_tail": detail,
        }
    status = "Unknown"
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("ID "):
            continue
        tokens = stripped.split()
        if len(tokens) >= 5:
            status = tokens[4]
        break
    return {
        "exists": True,
        "status": status,
        "rc": int(proc.returncode),
        "detail_tail": detail,
    }


def run_remote(process_id: str, inner_cmd: str, *, timeout_sec: float) -> str:
    run_as_user = _resolved_remote_run_as_user()
    remote_cmd = _wrap_remote_inner_cmd_for_user(
        inner_cmd=str(inner_cmd),
        run_as_user=run_as_user,
    )
    cmd = [
        "brainctl",
        "exec",
        f"process/{process_id}",
        "-n",
        "shai-core",
        "--",
        "bash",
        "-lc",
        remote_cmd,
    ]
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=max(1.0, timeout_sec),
    )
    if proc.returncode != 0:
        probe_timeout_sec = max(3.0, min(12.0, float(timeout_sec) * 0.1))
        process_probe = probe_process_state(
            process_id=str(process_id),
            timeout_sec=probe_timeout_sec,
        )
        infra_hint = ""
        if process_probe.get("exists") is False:
            infra_hint = (
                "remote process is NotFound in namespace; likely preempted/terminated "
                "by scheduler or cleaned externally"
            )
        elif int(proc.returncode) in {137, 143}:
            infra_hint = (
                "remote command terminated by signal-like exit code "
                f"{int(proc.returncode)}; check daemon/session logs and worker health"
            )
        raise RuntimeError(
            "remote command failed: "
            f"process={process_id} rc={proc.returncode} "
            f"run_as_user={run_as_user}\n"
            f"infra_hint={infra_hint or 'none'}\n"
            f"process_probe={json.dumps(process_probe, ensure_ascii=False)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc.stdout


def extract_last_json_object(output: str) -> dict[str, Any]:
    text = str(output).strip()
    if not text:
        raise RuntimeError("failed to find json object in empty output")
    last_object: dict[str, Any] | None = None

    # Fast path for newline-delimited JSON where each object fits on one line.
    for line in text.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        try:
            loaded = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            last_object = loaded
    if last_object is not None:
        return last_object

    # Fallback for pretty-printed/mixed output: decode top-level objects in order.
    decoder = json.JSONDecoder()
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start < 0:
            break
        try:
            loaded, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(loaded, dict):
            last_object = loaded
        index = start + max(1, end)
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


def detect_remote_execution_context(
    *,
    process_id: str,
    timeout_sec: float,
) -> dict[str, Any]:
    detect_cmd = (
        "set -euo pipefail; "
        "user=$(id -un); "
        "home=${HOME:-}; "
        "host=$(hostname -f 2>/dev/null || hostname); "
        'printf \'{"user":"%s","home":"%s","host":"%s",'
        '\\"tensorcast_session_root\\":\\"%s/.tensorcast\\"}\\n\' '
        '"$user" "$home" "$host" "$home"'
    )
    output = run_remote(process_id, detect_cmd, timeout_sec=max(10.0, timeout_sec))
    payload = extract_last_json_object(output)
    user = str(payload.get("user", "")).strip()
    home = str(payload.get("home", "")).strip()
    session_root = str(payload.get("tensorcast_session_root", "")).strip()
    if not user:
        raise RuntimeError(
            "failed to detect remote execution user: "
            f"process={process_id}, payload={payload}"
        )
    if not session_root:
        session_root = f"{home}/.tensorcast" if home else ""
        payload["tensorcast_session_root"] = session_root
    return payload


def verify_remote_run_as_user(
    *,
    process_id: str,
    timeout_sec: float,
) -> None:
    run_as_user = _resolved_remote_run_as_user()
    check_cmd = (
        "set -euo pipefail; "
        f"target={shlex.quote(str(run_as_user))}; "
        'if ! getent passwd "${target}" >/dev/null 2>&1; then '
        'echo "remote run-as user not found: ${target}" >&2; '
        "exit 97; "
        "fi; "
        'printf "run_as_user_ready current=%s target=%s\\n" "$(id -un)" "${target}"'
    )
    _ = run_remote(process_id, check_cmd, timeout_sec=max(10.0, timeout_sec))


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
    cuda_backend: str,
    capability_token_secret: str,
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
        "--set capability_tokens.active.version=1",
        f"--set capability_tokens.active.secret={shlex.quote(capability_token_secret)}",
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
        "for pid in $(pgrep -f '[t]ensorcast_daemon --config' || true); do kill -TERM \"$pid\" >/dev/null 2>&1 || true; done",
        "sleep 1",
        (
            "if command -v timeout >/dev/null 2>&1; then "
            f"timeout 30s tensorcast-cli daemon stop --session {shlex.quote(daemon_session)} >/dev/null 2>&1 || true; "
            f"else tensorcast-cli daemon stop --session {shlex.quote(daemon_session)} >/dev/null 2>&1 || true; fi"
        ),
        (
            "if command -v timeout >/dev/null 2>&1; then "
            f"timeout 180s {start_expr}; "
            f"else {start_expr}; fi"
        ),
        f"tensorcast-cli daemon status --session {shlex.quote(daemon_session)} --json",
    ]
    if str(cuda_backend).strip():
        start_cmd.insert(
            4,
            (
                "export TENSORCAST_CUDA_BACKEND="
                f"{shlex.quote(str(cuda_backend).strip())}"
            ),
        )
    if str(cuda_backend).strip() == "fake":
        start_cmd.insert(
            5, "export TEST_TMPDIR=${TEST_TMPDIR:-/tmp/tensorcast_fake_backend}"
        )
        start_cmd.insert(6, 'mkdir -p "$TEST_TMPDIR"')
    output = run_remote(
        process_id,
        "; ".join(start_cmd),
        timeout_sec=max(60.0, timeout_sec),
    )
    status = extract_last_json_object(output)
    daemon_status = status.get("daemon")
    if not isinstance(daemon_status, dict):
        raise RuntimeError(
            "daemon status payload missing daemon field: "
            f"process={process_id}, session={daemon_session}, payload={status}"
        )
    daemon_pid = daemon_status.get("pid")
    daemon_address = daemon_status.get("address")
    daemon_p2p_address = daemon_status.get("p2p_address")
    if (
        daemon_pid is None
        or daemon_address is None
        or daemon_p2p_address is None
        or status.get("started_at") is None
    ):
        raise RuntimeError(
            "daemon failed to start or did not report ready status: "
            f"process={process_id}, session={daemon_session}, status={status}"
        )
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
        "pkill -TERM -f '[t]ensorcast_daemon --config' >/dev/null 2>&1 || true",
        "sleep 1",
        "pkill -KILL -f '[t]ensorcast_daemon --config' >/dev/null 2>&1 || true",
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
    cuda_backend: str,
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
    tp_device_map_policy: str,
    tp_materialize_deadline_s: float,
    publish_device: str,
    output_json: str,
    weights_root: str,
    run_id: str,
    receiver_apply_mode: str,
    materialize_device: str,
    allow_version_skip: bool,
    hold_after_finish_s: float,
    log_file: str,
    ready_file: str | None,
    transport_group_mode: str,
    transport_group_kind: str,
    transport_group_namespace: str,
    transport_group_total_parts: int,
    transport_group_receiver_index: int,
    transport_group_priority: int,
    transport_group_epoch: int,
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
        "--tp-device-map-policy",
        str(tp_device_map_policy),
        "--tp-materialize-deadline-s",
        str(tp_materialize_deadline_s),
        "--transport-group-mode",
        transport_group_mode,
        "--transport-group-kind",
        transport_group_kind,
        "--transport-group-namespace",
        transport_group_namespace,
        "--transport-group-total-parts",
        str(transport_group_total_parts),
        "--transport-group-receiver-index",
        str(transport_group_receiver_index),
        "--transport-group-priority",
        str(transport_group_priority),
        "--transport-group-epoch",
        str(transport_group_epoch),
        "--publish-device",
        publish_device,
        "--receiver-apply-mode",
        receiver_apply_mode,
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
    if str(cuda_backend).strip():
        script.insert(
            4,
            (
                "export TENSORCAST_CUDA_BACKEND="
                f"{shlex.quote(str(cuda_backend).strip())}"
            ),
        )
    if str(cuda_backend).strip() == "fake":
        script.insert(
            5, "export TEST_TMPDIR=${TEST_TMPDIR:-/tmp/tensorcast_fake_backend}"
        )
        script.insert(6, 'mkdir -p "$TEST_TMPDIR"')
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


def discover_global_status_payload() -> dict[str, Any] | None:
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
    return payload


def discover_global_cluster_token() -> str | None:
    payload = discover_global_status_payload()
    if payload is None:
        return None
    health = payload.get("health", {})
    if not isinstance(health, dict):
        return None
    cluster_token = health.get("cluster_token")
    if isinstance(cluster_token, str) and cluster_token.strip():
        return cluster_token.strip()
    return None


def discover_global_db_file() -> str | None:
    payload = discover_global_status_payload()
    if payload is None:
        return None

    health = payload.get("health", {})
    if isinstance(health, dict):
        db_file = str(health.get("db_file", "")).strip()
        if db_file:
            return db_file

    state = payload.get("state", {})
    gs_state = state.get("global_store", state) if isinstance(state, dict) else {}
    if isinstance(gs_state, dict):
        db_file = str(gs_state.get("db_file", "")).strip()
        if db_file:
            return db_file
    return None


def _to_utc_sql_timestamp(ts: datetime) -> str:
    aware_ts = ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
    return aware_ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f+00:00")


def _to_proto_timestamp(ts: datetime) -> timestamp_pb2.Timestamp:
    aware_ts = ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
    proto_ts = timestamp_pb2.Timestamp()
    proto_ts.FromDatetime(aware_ts.astimezone(timezone.utc))
    return proto_ts


def query_transport_rows(
    *,
    gs_addr: str,
    started_at_utc: datetime,
    finished_at_utc: datetime,
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
        limit=200_000,
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


def query_transport_group_probe(
    *,
    gs_addr: str,
    group_mode: str,
    group_kind: str,
    started_at_utc: datetime,
    finished_at_utc: datetime,
) -> dict[str, Any]:
    probe: dict[str, Any] = {
        "enabled": True,
        "mode": str(group_mode),
        "expected_grouped": str(group_mode) == "tp_version",
        "group_kind": str(group_kind),
        "audit_method": "gs_rpc",
        "gs_addr": str(gs_addr),
        "window_start_utc": _to_utc_sql_timestamp(started_at_utc),
        "window_end_utc": _to_utc_sql_timestamp(finished_at_utc),
        "error": None,
        "total_transports": 0,
        "requester_tagged_transports": 0,
        "grouped_transports": 0,
        "kind_matched_transports": 0,
        "group_contract_transports": 0,
        "window_has_transports": False,
        "requester_tagged_complete": False,
        "group_mode_consistent": False,
        "group_contract_consistent": False,
    }
    transport_rows_payload = query_transport_rows(
        gs_addr=str(gs_addr),
        started_at_utc=started_at_utc,
        finished_at_utc=finished_at_utc,
    )
    probe["audit_method"] = str(transport_rows_payload.get("audit_method", "gs_rpc"))
    if bool(transport_rows_payload.get("error")):
        probe["error"] = (
            f"transport group probe failed: {transport_rows_payload.get('error')}"
        )
        return probe

    rows_raw = transport_rows_payload.get("rows", [])
    rows: list[dict[str, Any]] = []
    if isinstance(rows_raw, list):
        rows.extend(row for row in rows_raw if isinstance(row, dict))

    total_transports = int(len(rows))
    requester_tagged = 0
    grouped_transports = 0
    kind_matched_transports = 0
    group_contract_transports = 0
    expected_kind = str(group_kind).strip()
    for row in rows:
        requester = str(row.get("requester_worker_id", "")).strip()
        if requester:
            requester_tagged += 1
        row_group_id = str(row.get("group_id", "")).strip()
        row_group_kind = str(row.get("group_kind", "")).strip()
        grouped = bool(row_group_id and row_group_kind)
        if grouped:
            grouped_transports += 1
            if row_group_kind == expected_kind:
                kind_matched_transports += 1
            part_id = str(row.get("group_part_id", "")).strip()
            total_parts = int(_coerce_int(row.get("group_total_parts", 0)))
            if part_id and total_parts > 0:
                group_contract_transports += 1

    probe["total_transports"] = total_transports
    probe["requester_tagged_transports"] = requester_tagged
    probe["grouped_transports"] = grouped_transports
    probe["kind_matched_transports"] = kind_matched_transports
    probe["group_contract_transports"] = group_contract_transports
    probe["window_has_transports"] = total_transports > 0
    probe["requester_tagged_complete"] = (
        total_transports > 0 and requester_tagged == total_transports
    )
    if str(group_mode) == "tp_version":
        probe["group_mode_consistent"] = grouped_transports > 0
        probe["group_contract_consistent"] = (
            grouped_transports > 0
            and grouped_transports == kind_matched_transports
            and grouped_transports == group_contract_transports
        )
    else:
        probe["group_mode_consistent"] = grouped_transports == 0
        probe["group_contract_consistent"] = grouped_transports == 0
    return probe


def _coerce_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _coerce_int(value: Any) -> int:
    return int(_coerce_float(value))


def evaluate_group_probe_gate_failures(
    *,
    probe: dict[str, Any],
    mode: str,
) -> list[str]:
    reasons: list[str] = []
    if bool(probe.get("error")):
        reasons.append(f"group_probe_error:{probe.get('error')}")
    if not bool(probe.get("window_has_transports")):
        reasons.append("window_has_transports=false")
    if not bool(probe.get("requester_tagged_complete")):
        reasons.append("requester_tagged_complete=false")
    if str(mode) == "tp_version":
        if not bool(probe.get("group_mode_consistent")):
            reasons.append("group_mode_consistent=false")
        if not bool(probe.get("group_contract_consistent")):
            reasons.append("group_contract_consistent=false")
    return reasons


def _extract_group_probe_progress_snapshot(probe: dict[str, Any]) -> dict[str, int]:
    return {
        "total_transports": int(_coerce_int(probe.get("total_transports", 0))),
        "requester_tagged_transports": int(
            _coerce_int(probe.get("requester_tagged_transports", 0))
        ),
        "grouped_transports": int(_coerce_int(probe.get("grouped_transports", 0))),
        "kind_matched_transports": int(
            _coerce_int(probe.get("kind_matched_transports", 0))
        ),
        "group_contract_transports": int(
            _coerce_int(probe.get("group_contract_transports", 0))
        ),
    }


def _progress_snapshot_token(snapshot: dict[str, int]) -> tuple[int, int, int, int, int]:
    return (
        int(snapshot.get("total_transports", 0)),
        int(snapshot.get("requester_tagged_transports", 0)),
        int(snapshot.get("grouped_transports", 0)),
        int(snapshot.get("kind_matched_transports", 0)),
        int(snapshot.get("group_contract_transports", 0)),
    )


def evaluate_waiting_lease(
    *,
    previous_progress_token: tuple[int, int, int, int, int] | None,
    previous_progress_mono: float | None,
    now_mono: float,
    no_progress_limit_s: float,
    probe: dict[str, Any],
) -> dict[str, Any]:
    snapshot = _extract_group_probe_progress_snapshot(probe)
    current_token = _progress_snapshot_token(snapshot)
    progressed = previous_progress_token is None or current_token != previous_progress_token
    if progressed or previous_progress_mono is None:
        effective_progress_mono = float(now_mono)
    else:
        effective_progress_mono = float(previous_progress_mono)
    no_progress_elapsed_s = max(0.0, float(now_mono) - effective_progress_mono)
    waiting_timeout = no_progress_elapsed_s >= max(0.0, float(no_progress_limit_s))
    waiting_timeout_reason = "waiting_no_progress" if waiting_timeout else ""
    return {
        "current_progress_token": current_token,
        "current_progress_snapshot": snapshot,
        "progressed": bool(progressed),
        "effective_progress_mono": float(effective_progress_mono),
        "no_progress_elapsed_s": float(no_progress_elapsed_s),
        "waiting_timeout": bool(waiting_timeout),
        "waiting_timeout_reason": waiting_timeout_reason,
    }


def run_transport_group_p0_guard(
    *,
    enabled: bool,
    mode: str,
    group_kind: str,
    gs_addr: str,
    started_at_utc: datetime,
    grace_s: float,
    poll_interval_s: float,
) -> dict[str, Any]:
    no_progress_limit_s = float(max(0.0, grace_s))
    effective_poll_interval_s = float(max(0.5, poll_interval_s))
    result: dict[str, Any] = {
        "enabled": bool(enabled) and str(mode) == "tp_version",
        "mode": str(mode),
        "group_kind": str(group_kind),
        "grace_s": float(max(0.0, grace_s)),
        "no_progress_limit_s": no_progress_limit_s,
        "poll_interval_s": effective_poll_interval_s,
        "timeout_model": "waiting_lease_no_progress",
        "attempts": 0,
        "triggered": False,
        "triggered_at_utc": None,
        "waiting_timeout_reason": None,
        "lease_renew_count": 0,
        "max_no_progress_elapsed_s": 0.0,
        "last_no_progress_elapsed_s": 0.0,
        "last_progress_snapshot": {},
        "reasons": [],
        "probe": {},
        "terminate_errors": [],
    }
    if not bool(result["enabled"]):
        return result

    previous_progress_token: tuple[int, int, int, int, int] | None = None
    previous_progress_mono: float | None = None
    probe: dict[str, Any] = {}
    reasons: list[str] = []
    while True:
        result["attempts"] = int(result["attempts"]) + 1
        now_utc = datetime.now(timezone.utc)
        probe = query_transport_group_probe(
            gs_addr=str(gs_addr),
            group_mode=str(mode),
            group_kind=str(group_kind),
            started_at_utc=started_at_utc,
            finished_at_utc=now_utc,
        )
        reasons = evaluate_group_probe_gate_failures(
            probe=probe,
            mode=str(mode),
        )
        now_mono = time.monotonic()
        lease_eval = evaluate_waiting_lease(
            previous_progress_token=previous_progress_token,
            previous_progress_mono=previous_progress_mono,
            now_mono=float(now_mono),
            no_progress_limit_s=no_progress_limit_s,
            probe=probe,
        )
        previous_progress_token = lease_eval["current_progress_token"]
        previous_progress_mono = float(lease_eval["effective_progress_mono"])
        result["last_progress_snapshot"] = dict(
            lease_eval.get("current_progress_snapshot", {})
        )
        result["last_no_progress_elapsed_s"] = float(
            lease_eval.get("no_progress_elapsed_s", 0.0)
        )
        result["max_no_progress_elapsed_s"] = max(
            float(result.get("max_no_progress_elapsed_s", 0.0)),
            float(lease_eval.get("no_progress_elapsed_s", 0.0)),
        )
        if bool(lease_eval.get("progressed")):
            result["lease_renew_count"] = int(result.get("lease_renew_count", 0)) + 1
        if not reasons:
            break
        if bool(lease_eval.get("waiting_timeout")):
            result["triggered"] = True
            result["triggered_at_utc"] = _to_utc_sql_timestamp(now_utc)
            result["waiting_timeout_reason"] = str(
                lease_eval.get("waiting_timeout_reason") or "waiting_no_progress"
            )
            break
        sleep_s = effective_poll_interval_s
        if sleep_s <= 0.0:
            continue
        time.sleep(sleep_s)

    result["probe"] = probe
    result["reasons"] = reasons
    return result


def merge_timeout_analysis_with_waiting_guard(
    *,
    timeout_analysis: dict[str, Any],
    p0_guard: dict[str, Any],
) -> dict[str, Any]:
    waiting_counts = Counter(
        {
            str(k): int(v)
            for k, v in dict(
                timeout_analysis.get("waiting_timeout_reason_counts", {})
            ).items()
        }
    )
    transport_counts = Counter(
        {
            str(k): int(v)
            for k, v in dict(
                timeout_analysis.get("transport_timeout_reason_counts", {})
            ).items()
        }
    )
    waiting_reason = str(p0_guard.get("waiting_timeout_reason") or "").strip()
    if bool(p0_guard.get("triggered")) and waiting_reason:
        waiting_counts[waiting_reason] += 1

    merged = dict(timeout_analysis)
    merged["waiting_timeout_reason_counts"] = dict(waiting_counts)
    merged["transport_timeout_reason_counts"] = dict(transport_counts)
    merged["waiting_timeout_observed"] = bool(waiting_counts)
    merged["transport_timeout_observed"] = bool(transport_counts)
    merged["waiting_lease"] = {
        "renew_count": int(p0_guard.get("lease_renew_count", 0)),
        "no_progress_limit_s": float(max(0.0, p0_guard.get("no_progress_limit_s", 0.0))),
        "max_no_progress_elapsed_s": float(
            max(0.0, p0_guard.get("max_no_progress_elapsed_s", 0.0))
        ),
        "last_no_progress_elapsed_s": float(
            max(0.0, p0_guard.get("last_no_progress_elapsed_s", 0.0))
        ),
        "last_progress_snapshot": (
            dict(p0_guard.get("last_progress_snapshot", {}))
            if isinstance(p0_guard.get("last_progress_snapshot"), dict)
            else {}
        ),
    }
    return merged


def compute_transport_metrics(
    *,
    transport_rows_payload: dict[str, Any],
    sample_interval_s: float,
    max_samples: int,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "enabled": True,
        "error": None,
        "gs_addr": transport_rows_payload.get("gs_addr"),
        "audit_method": str(transport_rows_payload.get("audit_method", "gs_rpc")),
        "window_start_utc": transport_rows_payload.get("window_start_utc"),
        "window_end_utc": transport_rows_payload.get("window_end_utc"),
        "transport_count": 0,
        "completed_transport_count": 0,
        "invalid_completed_bytes_count": 0,
        "throughput": {
            "sample_interval_s": float(max(0.1, sample_interval_s)),
            "sample_count": 0,
            "active_sample_count": 0,
            "peak_active_throughput_gib_s": 0.0,
            "p95_active_throughput_gib_s": 0.0,
            "mean_active_throughput_gib_s": 0.0,
            "active_transport_peak": 0.0,
            "active_transport_mean": 0.0,
            "window_start_epoch_s": 0.0,
            "window_end_epoch_s": 0.0,
            "series": [],
        },
        "diffusion": {
            "source_key": "replica_id",
            "total_transports": 0,
            "unique_sources": 0,
            "top1_share": 0.0,
            "hhi": 0.0,
            "source_counts": [],
        },
        "per_transport_records": [],
    }
    error = transport_rows_payload.get("error")
    if error:
        metrics["error"] = str(error)
        return metrics

    rows_raw = transport_rows_payload.get("rows", [])
    rows: list[dict[str, Any]] = []
    if isinstance(rows_raw, list):
        rows.extend(row for row in rows_raw if isinstance(row, dict))
    metrics["transport_count"] = int(len(rows))
    if not rows:
        return metrics

    step_s = float(max(0.1, sample_interval_s))
    cap_samples = int(max(10, max_samples))
    source_counter: Counter[str] = Counter()
    intervals: list[tuple[float, float, float]] = []
    per_transport_records: list[dict[str, Any]] = []
    invalid_completed_transport_ids: list[str] = []

    for row in rows:
        created_at_epoch_s = _coerce_float(row.get("created_at_epoch_s"))
        completed_at_epoch_s = _coerce_float(row.get("completed_at_epoch_s"))
        duration_s = (
            max(0.0, completed_at_epoch_s - created_at_epoch_s)
            if created_at_epoch_s > 0.0 and completed_at_epoch_s > 0.0
            else 0.0
        )
        bytes_value = max(0, _coerce_int(row.get("replica_memory_size_bytes")))
        bytes_source = "replica_memory_size_bytes"
        if duration_s > 0.0 and bytes_value <= 0:
            invalid_completed_transport_ids.append(str(row.get("transport_id", "")))
        throughput_gib_s = (
            float(bytes_value) / float(1024**3) / float(duration_s)
            if bytes_value > 0 and duration_s > 0.0
            else 0.0
        )
        if duration_s > 0.0 and throughput_gib_s > 0.0:
            intervals.append(
                (
                    float(created_at_epoch_s),
                    float(completed_at_epoch_s),
                    float(throughput_gib_s),
                )
            )
            metrics["completed_transport_count"] = (
                int(metrics["completed_transport_count"]) + 1
            )

        replica_id = str(row.get("replica_id", "")).strip()
        if replica_id:
            source_counter[replica_id] += 1

        per_transport_records.append(
            {
                "transport_id": str(row.get("transport_id", "")),
                "replica_id": replica_id,
                "artifact_id": str(row.get("artifact_id", "")),
                "status": str(row.get("status", "")),
                "completion_outcome": str(row.get("completion_outcome", "")),
                "request_id": str(row.get("request_id", "")),
                "requester_worker_id": str(row.get("requester_worker_id", "")),
                "group_id": str(row.get("group_id", "")),
                "group_kind": str(row.get("group_kind", "")),
                "group_part_id": str(row.get("group_part_id", "")),
                "group_total_parts": int(_coerce_int(row.get("group_total_parts"))),
                "created_at_utc": str(row.get("created_at_utc", "")),
                "completed_at_utc": str(row.get("completed_at_utc", "")),
                "created_at_epoch_s": float(created_at_epoch_s),
                "completed_at_epoch_s": float(completed_at_epoch_s),
                "duration_s": float(duration_s),
                "bytes": int(bytes_value),
                "bytes_source": str(bytes_source),
                "throughput_gib_s": float(throughput_gib_s),
                "bandwidth_gib_s": float(throughput_gib_s),
                "included_in_sampling": bool(
                    duration_s > 0.0 and throughput_gib_s > 0.0
                ),
            }
        )

    metrics["per_transport_records"] = per_transport_records

    total_sources = int(sum(source_counter.values()))
    if total_sources > 0:
        top_count = max(source_counter.values())
        shares = [
            float(count) / float(total_sources) for count in source_counter.values()
        ]
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
        metrics["diffusion"] = {
            "source_key": "replica_id",
            "total_transports": int(total_sources),
            "unique_sources": int(len(source_counter)),
            "top1_share": float(float(top_count) / float(total_sources)),
            "hhi": float(sum(share * share for share in shares)),
            "source_counts": source_counts,
        }

    if invalid_completed_transport_ids:
        metrics["invalid_completed_bytes_count"] = int(
            len(invalid_completed_transport_ids)
        )
        sample_transport_ids = [
            transport_id
            for transport_id in invalid_completed_transport_ids
            if transport_id
        ][:5]
        sample_suffix = ""
        if len(invalid_completed_transport_ids) > len(sample_transport_ids):
            sample_suffix = ",..."
        sample_segment = (
            f" sample_transport_ids={','.join(sample_transport_ids)}{sample_suffix}."
            if sample_transport_ids
            else ""
        )
        metrics["error"] = (
            "invalid transport metrics input: missing replica_memory_size_bytes "
            "for completed transports;"
            f" count={len(invalid_completed_transport_ids)}.{sample_segment}"
        )
        return metrics

    if not intervals:
        return metrics

    sample_start = min(item[0] for item in intervals)
    sample_end = max(item[1] for item in intervals)
    if sample_end <= sample_start:
        sample_end = sample_start + step_s
    estimated_samples = int(math.floor((sample_end - sample_start) / step_s)) + 1
    if estimated_samples > cap_samples:
        step_s = max(step_s, float(sample_end - sample_start) / float(cap_samples - 1))
        estimated_samples = int(math.floor((sample_end - sample_start) / step_s)) + 1
    sample_count = max(1, estimated_samples)

    rate_diff = [0.0] * (sample_count + 1)
    active_diff = [0] * (sample_count + 1)
    for started_at_s, ended_at_s, rate_gib_s in intervals:
        first_index = int(math.ceil((started_at_s - sample_start) / step_s - 1e-9))
        last_index = int(math.floor((ended_at_s - sample_start) / step_s + 1e-9))
        if last_index < 0 or first_index >= sample_count:
            continue
        first_index = max(0, first_index)
        last_index = min(sample_count - 1, last_index)
        if first_index > last_index:
            continue
        rate_diff[first_index] += float(rate_gib_s)
        rate_diff[last_index + 1] -= float(rate_gib_s)
        active_diff[first_index] += 1
        active_diff[last_index + 1] -= 1

    throughput_series: list[dict[str, float]] = []
    active_throughputs: list[float] = []
    active_counts: list[float] = []
    running_rate = 0.0
    running_active = 0
    for index in range(sample_count):
        running_rate += rate_diff[index]
        running_active += active_diff[index]
        point_rate = max(0.0, float(running_rate))
        point_active = max(0, int(running_active))
        ts_epoch_s = float(sample_start + float(index) * step_s)
        throughput_series.append(
            {
                "ts_epoch_s": float(ts_epoch_s),
                "throughput_gib_s": float(point_rate),
                "active_transports": float(point_active),
            }
        )
        if point_active > 0 and point_rate > 0.0:
            active_throughputs.append(float(point_rate))
            active_counts.append(float(point_active))

    throughput_payload = metrics["throughput"]
    throughput_payload["sample_interval_s"] = float(step_s)
    throughput_payload["sample_count"] = float(len(throughput_series))
    throughput_payload["active_sample_count"] = float(len(active_throughputs))
    throughput_payload["window_start_epoch_s"] = float(sample_start)
    throughput_payload["window_end_epoch_s"] = float(sample_end)
    throughput_payload["peak_active_throughput_gib_s"] = (
        float(max(active_throughputs)) if active_throughputs else 0.0
    )
    throughput_payload["p95_active_throughput_gib_s"] = (
        float(percentile(active_throughputs, 95.0)) if active_throughputs else 0.0
    )
    throughput_payload["mean_active_throughput_gib_s"] = (
        float(statistics.fmean(active_throughputs)) if active_throughputs else 0.0
    )
    throughput_payload["active_transport_peak"] = (
        float(max(active_counts)) if active_counts else 0.0
    )
    throughput_payload["active_transport_mean"] = (
        float(statistics.fmean(active_counts)) if active_counts else 0.0
    )
    throughput_payload["series"] = throughput_series
    return metrics


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


def query_replica_counts_until_old_released(
    *,
    gs_addr: str,
    artifact_ids: list[str],
    dropped_artifact_ids: set[str],
    timeout_sec: float,
    poll_interval_s: float,
) -> dict[str, Any]:
    if not artifact_ids:
        return {
            "gs_addr": str(gs_addr),
            "audit_method": "gs_rpc",
            "replica_counts": {},
            "poll_attempts": 0,
            "poll_elapsed_s": 0.0,
            "old_versions_released_observed": True,
            "dropped_artifact_ids": [],
        }

    effective_timeout_s = max(1.0, float(timeout_sec))
    start_time = time.time()
    deadline = start_time + effective_timeout_s
    attempts = 0
    last_counts: dict[str, dict[str, int]] = {}
    last_audit_method = "gs_rpc"
    while True:
        attempts += 1
        counts, audit_method = query_replica_counts_via_gs_rpc(
            gs_addr=str(gs_addr),
            artifact_ids=artifact_ids,
            timeout_sec=max(1.0, min(10.0, float(timeout_sec))),
        )
        last_counts = counts
        last_audit_method = audit_method
        released = True
        for artifact_id in dropped_artifact_ids:
            row = counts.get(artifact_id, {})
            replica_count = int(row.get("replica_count", 0))
            if replica_count != 0:
                released = False
                break
        now = time.time()
        if released or now >= deadline:
            return {
                "gs_addr": str(gs_addr),
                "audit_method": str(last_audit_method),
                "replica_counts": last_counts,
                "poll_attempts": int(attempts),
                "poll_elapsed_s": float(max(0.0, now - start_time)),
                "old_versions_released_observed": bool(released),
                "dropped_artifact_ids": sorted(dropped_artifact_ids),
            }
        sleep_s = min(
            max(0.1, float(poll_interval_s)),
            max(0.0, deadline - now),
        )
        if sleep_s <= 0.0:
            continue
        time.sleep(sleep_s)


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
        "--receiver-timeout-auto-adjust",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Auto-adjust receiver-timeout-s upward for TP bind_into/swap cases "
            "using a conservative scaleout timeout floor."
        ),
    )
    parser.add_argument(
        "--keep-last-auto-adjust",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Auto-adjust keep-last upward for TP bind_into/swap scaleout cases "
            "to avoid dropped-version false failures under queueing."
        ),
    )
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
        "--tp-device-map-policy",
        choices=["auto", "strict", "modulo"],
        default="auto",
        help=(
            "TP rank->CUDA mapping policy forwarded to weight_publisher_e2e. "
            "auto: contiguous when possible else modulo fallback."
        ),
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
        default="cpu",
        help=(
            "Publisher payload device passed to weight_publisher_e2e. "
            "Must be explicit (e.g., cpu or cuda:0); auto is disallowed."
        ),
    )
    parser.add_argument(
        "--cuda-backend",
        default=os.environ.get("TENSORCAST_CUDA_BACKEND", ""),
        help=(
            "Optional CUDA backend override for daemon/role commands. "
            "Allowed values: real, fake. Empty keeps process default."
        ),
    )
    parser.add_argument(
        "--transport-group-mode",
        choices=["none", "tp_version"],
        default="none",
        help=(
            "Receiver transport scheduling-group mode passed to weight_publisher_e2e."
        ),
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
        "--receiver-preflight-transient-overlap",
        type=int,
        default=1,
        help=(
            "Receiver memory preflight transient overlap multiplier. "
            "Unified request chain defaults to 1 (serial apply path)."
        ),
    )
    parser.add_argument(
        "--progress-poll-s",
        type=float,
        default=10.0,
        help="Progress heartbeat polling interval (seconds).",
    )
    parser.add_argument(
        "--p0-early-stop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable P0 early-stop for group case. If group metadata contract is "
            "missing after publisher finishes, terminate roles early and fail fast."
        ),
    )
    parser.add_argument(
        "--p0-early-stop-grace-s",
        type=float,
        default=20.0,
        help="P0 early-stop grace window after publisher completion (seconds).",
    )
    parser.add_argument(
        "--throughput-sample-interval-s",
        type=float,
        default=1.0,
        help="Sampling interval for cluster active throughput series (seconds).",
    )
    parser.add_argument(
        "--throughput-max-samples",
        type=int,
        default=20000,
        help="Max throughput samples kept in payload (adaptive interval when exceeded).",
    )
    parser.add_argument(
        "--max-publish-to-apply-s",
        type=float,
        default=30.0,
        help="Upper bound for publish->receiver-apply latency (seconds).",
    )
    parser.add_argument(
        "--max-publish-to-apply-auto-adjust",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Auto-adjust max-publish-to-apply-s upward for TP bind scaleout "
            "so queueing-heavy runs are evaluated against a realistic floor."
        ),
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
    publish_device = str(args.publish_device).strip().lower()
    if publish_device == "auto":
        raise ValueError(
            "publish-device must be explicit (cpu/cuda:<index>); auto is not allowed"
        )
    materialize_device = str(args.materialize_device).strip().lower()
    if materialize_device == "auto":
        raise ValueError(
            "materialize-device must be explicit (cpu/cuda:<index>); auto is not allowed"
        )
    if int(args.max_concurrency) <= 0:
        raise ValueError("max-concurrency must be > 0")
    cuda_backend = normalize_cuda_backend(str(args.cuda_backend))
    if str(args.transport_group_mode) == "tp_version" and int(args.tp_world_size) <= 0:
        raise ValueError("transport-group-mode=tp_version requires tp-world-size > 0")
    if int(args.receiver_preflight_transient_overlap) <= 0:
        raise ValueError("receiver-preflight-transient-overlap must be > 0")
    if float(args.progress_poll_s) <= 0:
        raise ValueError("progress-poll-s must be > 0")
    if float(args.p0_early_stop_grace_s) < 0.0:
        raise ValueError("p0-early-stop-grace-s must be >= 0")
    if float(args.throughput_sample_interval_s) <= 0.0:
        raise ValueError("throughput-sample-interval-s must be > 0")
    if int(args.throughput_max_samples) <= 0:
        raise ValueError("throughput-max-samples must be > 0")
    if float(args.receiver_timeout_s) <= 0:
        raise ValueError("receiver-timeout-s must be > 0")
    if float(args.max_publish_to_apply_s) <= 0:
        raise ValueError("max-publish-to-apply-s must be > 0")
    estimated_keep_last = estimate_keep_last_floor(args)
    keep_last_stable_cap = estimate_keep_last_stable_cap(args)
    effective_keep_last_floor = estimated_keep_last
    if (
        keep_last_stable_cap is not None
        and keep_last_stable_cap > 0
        and keep_last_stable_cap < estimated_keep_last
    ):
        effective_keep_last_floor = keep_last_stable_cap
        print(
            "[preflight] warning: keep-last floor capped by publisher stable budget "
            f"(estimated_floor={estimated_keep_last}, "
            f"stable_cap={keep_last_stable_cap}); "
            "no-skip runs may observe dropped versions under queueing"
        )

    configured_keep_last = int(args.keep_last)
    if bool(args.keep_last_auto_adjust) and configured_keep_last < effective_keep_last_floor:
        args.keep_last = effective_keep_last_floor
        print(
            "[preflight] auto-adjust keep-last "
            f"{configured_keep_last} -> {effective_keep_last_floor} "
            "(tp_bind scaleout no-skip floor)"
        )
    elif (
        not bool(args.keep_last_auto_adjust)
        and configured_keep_last < effective_keep_last_floor
    ):
        print(
            "[preflight] warning: configured keep-last "
            f"{configured_keep_last} is lower than estimated floor "
            f"{effective_keep_last_floor}; receiver may observe dropped versions"
        )
    estimated_receiver_timeout_sec = estimate_receiver_timeout_floor_sec(args)
    configured_receiver_timeout_sec = float(args.receiver_timeout_s)
    if (
        bool(args.receiver_timeout_auto_adjust)
        and configured_receiver_timeout_sec < estimated_receiver_timeout_sec
    ):
        args.receiver_timeout_s = estimated_receiver_timeout_sec
        print(
            "[preflight] auto-adjust receiver-timeout-s "
            f"{configured_receiver_timeout_sec:.1f} -> "
            f"{estimated_receiver_timeout_sec:.1f} "
            "(tp_bind scaleout conservative floor)"
        )
    elif (
        not bool(args.receiver_timeout_auto_adjust)
        and configured_receiver_timeout_sec < estimated_receiver_timeout_sec
    ):
        print(
            "[preflight] warning: configured receiver-timeout-s "
            f"{configured_receiver_timeout_sec:.1f}s is lower than estimated floor "
            f"{estimated_receiver_timeout_sec:.1f}s; TP bind may timeout under scaleout"
        )
    estimated_publish_to_apply_sec = estimate_publish_to_apply_floor_sec(args)
    configured_publish_to_apply_sec = float(args.max_publish_to_apply_s)
    publish_to_apply_auto_adjusted = False
    if (
        bool(args.max_publish_to_apply_auto_adjust)
        and configured_publish_to_apply_sec < estimated_publish_to_apply_sec
    ):
        args.max_publish_to_apply_s = estimated_publish_to_apply_sec
        publish_to_apply_auto_adjusted = True
        print(
            "[preflight] auto-adjust max-publish-to-apply-s "
            f"{configured_publish_to_apply_sec:.1f} -> "
            f"{estimated_publish_to_apply_sec:.1f} "
            "(tp_bind scaleout propagation floor)"
        )
    elif (
        not bool(args.max_publish_to_apply_auto_adjust)
        and configured_publish_to_apply_sec < estimated_publish_to_apply_sec
    ):
        print(
            "[preflight] warning: configured max-publish-to-apply-s "
            f"{configured_publish_to_apply_sec:.1f}s is lower than estimated floor "
            f"{estimated_publish_to_apply_sec:.1f}s; queueing may produce false violations"
        )
    args._max_publish_to_apply_preflight = {  # noqa: SLF001
        "configured_s": configured_publish_to_apply_sec,
        "estimated_floor_s": estimated_publish_to_apply_sec,
        "auto_adjusted": publish_to_apply_auto_adjusted,
        "effective_s": float(args.max_publish_to_apply_s),
    }
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
    remote_run_as_user = configure_remote_run_as_user(resolve_workspace_user())
    print(
        "[preflight] enforce non-root remote execution "
        f"run_as_user={remote_run_as_user}"
    )

    run_tag = f"{int(time.time())}"
    capability_token_secret = secrets.token_hex(32)
    run_id = f"{args.case_name}-{run_tag}"
    transport_group_plan = derive_transport_group_plan(
        mode=str(args.transport_group_mode),
        receiver_count=len(receiver_procs),
        tp_world_size=int(args.tp_world_size),
    )
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
    remote_exec_context_by_process: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(all_processes)) as pool:
        ctx_futures = {
            process_id: pool.submit(
                detect_remote_execution_context,
                process_id=process_id,
                timeout_sec=20.0,
            )
            for process_id in all_processes
        }
        for process_id, future in ctx_futures.items():
            remote_exec_context_by_process[process_id] = future.result()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(all_processes)) as pool:
        verify_futures = {
            process_id: pool.submit(
                verify_remote_run_as_user,
                process_id=process_id,
                timeout_sec=20.0,
            )
            for process_id in all_processes
        }
        for process_id, future in verify_futures.items():
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "remote run-as user preflight failed: "
                    f"process={process_id}, run_as_user={remote_run_as_user}"
                ) from exc
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
    memfd_preflight_by_process: dict[str, dict[str, Any]] = {}
    for process_id, daemon_config_path in daemon_config_by_process.items():
        role = "publisher" if process_id == str(args.publisher_proc) else "receiver"
        memfd_preflight_by_process[process_id] = _validate_daemon_memfd_required(
            process_id=process_id,
            role=role,
            daemon_config_path=str(daemon_config_path),
        )
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
                transient_overlap_hint=int(args.receiver_preflight_transient_overlap),
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
        process_context = remote_exec_context_by_process.get(process_id, {})
        process_user = str(process_context.get("user", "unknown")).strip() or "unknown"
        process_session_root = (
            str(process_context.get("tensorcast_session_root", "")).strip()
            or "<unknown>"
        )
        print(
            "[daemon-start] process="
            f"{process_id} session={daemon_sessions[process_id]} "
            f"connect={daemon_connect_by_process[process_id]} "
            f"p2p={daemon_p2p_port_by_process[process_id]} "
            f"user={process_user} session_root={process_session_root}"
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
            cuda_backend=cuda_backend,
            capability_token_secret=capability_token_secret,
            timeout_sec=120.0,
        )

    transport_group_namespace_base = f"{str(args.case_name).strip()}-{run_tag}"
    transport_group_epoch = int(time.time())

    publisher_spec = RoleSpec(
        process_id=str(args.publisher_proc),
        output_json=publisher_output,
        log_file=publisher_log_file,
        inner_cmd=build_e2e_command(
            repo_root=str(args.repo_root),
            mode="publisher",
            cuda_backend=cuda_backend,
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
            tp_device_map_policy=str(args.tp_device_map_policy),
            tp_materialize_deadline_s=float(args.tp_materialize_deadline_s),
            publish_device=str(args.publish_device),
            output_json=publisher_output,
            weights_root=str(args.weights_root),
            run_id=f"{run_id}-publisher",
            receiver_apply_mode=str(args.receiver_apply_mode),
            materialize_device=str(args.materialize_device),
            allow_version_skip=bool(args.allow_receiver_skips),
            hold_after_finish_s=float(args.publisher_hold_after_finish_s),
            log_file=publisher_log_file,
            ready_file=None,
            transport_group_mode="none",
            transport_group_kind=TRANSPORT_GROUP_KIND_TP_VERSION,
            transport_group_namespace=f"{transport_group_namespace_base}:publisher",
            transport_group_total_parts=0,
            transport_group_receiver_index=0,
            transport_group_priority=TRANSPORT_GROUP_DEFAULT_PRIORITY,
            transport_group_epoch=transport_group_epoch,
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
                cuda_backend=cuda_backend,
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
                tp_device_map_policy=str(args.tp_device_map_policy),
                tp_materialize_deadline_s=float(args.tp_materialize_deadline_s),
                publish_device=str(args.publish_device),
                output_json=receiver_outputs[proc],
                weights_root=str(args.weights_root),
                run_id=f"{run_id}-receiver",
                receiver_apply_mode=str(args.receiver_apply_mode),
                materialize_device=str(args.materialize_device),
                allow_version_skip=bool(args.allow_receiver_skips),
                hold_after_finish_s=float(args.receiver_hold_after_finish_s),
                log_file=receiver_log_files[proc],
                ready_file=(case_dir / f"receiver_{idx + 1}.ready.json").as_posix(),
                transport_group_mode=transport_group_plan.mode,
                transport_group_kind=transport_group_plan.kind,
                transport_group_namespace=(
                    f"{transport_group_namespace_base}:receiver"
                ),
                transport_group_total_parts=transport_group_plan.total_parts,
                transport_group_receiver_index=idx,
                transport_group_priority=transport_group_plan.priority,
                transport_group_epoch=transport_group_epoch,
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
    transport_group_probe: dict[str, Any] = {}
    p0_early_stop: dict[str, Any] = {
        "enabled": bool(args.p0_early_stop)
        and transport_group_plan.mode == "tp_version",
        "mode": transport_group_plan.mode,
        "grace_s": float(args.p0_early_stop_grace_s),
        "triggered": False,
        "reasons": [],
        "attempts": 0,
        "triggered_at_utc": None,
        "probe": {},
        "terminate_errors": [],
    }
    transport_rows_payload: dict[str, Any] = {}
    transport_metrics: dict[str, Any] = {}
    case_transport_window_start = datetime.now(timezone.utc)

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

            p0_result = run_transport_group_p0_guard(
                enabled=bool(args.p0_early_stop),
                mode=transport_group_plan.mode,
                group_kind=transport_group_plan.kind or TRANSPORT_GROUP_KIND_TP_VERSION,
                gs_addr=str(args.gs_addr),
                started_at_utc=case_transport_window_start,
                grace_s=float(args.p0_early_stop_grace_s),
                poll_interval_s=max(
                    float(args.poll_interval_s), float(args.progress_poll_s)
                ),
            )
            p0_early_stop.update(p0_result)

            if bool(p0_early_stop.get("triggered")):
                terminate_errors = terminate_remote_e2e_roles(
                    process_ids=[spec.process_id for spec in receiver_specs]
                    + [publisher_spec.process_id],
                    model_name=model_name,
                    timeout_sec=15.0,
                )
                p0_early_stop["terminate_errors"] = terminate_errors
                for spec in receiver_specs:
                    future = receiver_futures.get(spec.process_id)
                    if future is None:
                        continue
                    try:
                        receiver_logs[spec.process_id] = future.result(timeout=30.0)
                    except Exception as exc:  # noqa: BLE001
                        receiver_logs[spec.process_id] = (
                            f"receiver terminated or failed after p0 early stop: {exc}"
                        )
                    try:
                        receiver_summaries[spec.process_id] = read_remote_json(
                            spec.process_id,
                            spec.output_json,
                            timeout_sec=10.0,
                        )
                    except Exception as exc:  # noqa: BLE001
                        receiver_summaries[spec.process_id] = {"__error__": str(exc)}
            else:
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
                            cluster_artifact_probe[process_id] = (
                                probe_artifacts_on_process(
                                    process_id=process_id,
                                    repo_root=str(args.repo_root),
                                    daemon_connect_address=daemon_connect_by_process[
                                        process_id
                                    ],
                                    artifact_ids=published_artifact_ids,
                                    timeout_sec=30.0,
                                )
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
    case_transport_window_end = datetime.now(timezone.utc)

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

    published_artifact_ids: list[str] = []
    for version in ordered_versions:
        row = published_by_version.get(version, {})
        if not isinstance(row, dict):
            continue
        artifact_id = str(row.get("artifact_id", "")).strip()
        if artifact_id:
            published_artifact_ids.append(artifact_id)
    published_artifact_ids = list(dict.fromkeys(published_artifact_ids))
    if published_artifact_ids:
        try:
            gs_probe = query_replica_counts_until_old_released(
                gs_addr=str(args.gs_addr),
                artifact_ids=published_artifact_ids,
                dropped_artifact_ids=dropped_artifact_ids,
                timeout_sec=max(1.0, float(args.retention_timeout_s)),
                poll_interval_s=max(0.2, float(args.poll_interval_s)),
            )
        except Exception as exc:  # noqa: BLE001
            gs_probe = {
                "gs_addr": str(args.gs_addr),
                "audit_method": "gs_rpc",
                "error": str(exc),
                "dropped_artifact_ids": sorted(dropped_artifact_ids),
            }
    transport_group_probe = query_transport_group_probe(
        gs_addr=str(args.gs_addr),
        group_mode=transport_group_plan.mode,
        group_kind=transport_group_plan.kind or TRANSPORT_GROUP_KIND_TP_VERSION,
        started_at_utc=case_transport_window_start,
        finished_at_utc=case_transport_window_end,
    )
    transport_rows_payload = query_transport_rows(
        gs_addr=str(args.gs_addr),
        started_at_utc=case_transport_window_start,
        finished_at_utc=case_transport_window_end,
    )
    transport_metrics = compute_transport_metrics(
        transport_rows_payload=transport_rows_payload,
        sample_interval_s=float(args.throughput_sample_interval_s),
        max_samples=int(args.throughput_max_samples),
    )

    publish_latencies = [
        float(row.get("publish_latency_s", 0.0))
        for row in published_by_version.values()
        if isinstance(row, dict)
    ]
    publish_bandwidths = [
        float(_coerce_float(row.get("publish_throughput_gib_s", 0.0)))
        for row in published_by_version.values()
        if isinstance(row, dict)
        and float(_coerce_float(row.get("publish_throughput_gib_s", 0.0))) > 0.0
    ]
    put_bandwidths = [
        float(_coerce_float(row.get("put_throughput_gib_s", 0.0)))
        for row in published_by_version.values()
        if isinstance(row, dict)
        and float(_coerce_float(row.get("put_throughput_gib_s", 0.0))) > 0.0
    ]
    apply_latencies: list[float] = []
    propagation_latencies: list[float] = []
    pointer_stability_violations: list[dict[str, Any]] = []
    receiver_sequence_failures: list[dict[str, Any]] = []
    receiver_skips: list[dict[str, Any]] = []
    receiver_skip_events_by_process = collect_receiver_skip_events_by_process(
        receiver_logs
    )
    receiver_mode_failures: list[dict[str, Any]] = []
    propagation_violations: list[dict[str, Any]] = []
    publish_to_apply_limit_s = float(args.max_publish_to_apply_s)

    publish_ts_by_version = {
        version: float(row.get("published_at_s", 0.0))
        for version, row in published_by_version.items()
    }

    missing_receiver_summaries = [
        {"process_id": proc, "reason": "receiver summary missing"}
        for proc in receiver_procs
        if proc not in receiver_summaries
    ]
    receiver_sequence_failures.extend(missing_receiver_summaries)

    for proc, summary in receiver_summaries.items():
        if "__error__" in summary:
            receiver_sequence_failures.append(
                {
                    "process_id": proc,
                    "reason": str(summary.get("__error__", "receiver summary error")),
                }
            )
            continue
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
        skip_events = receiver_skip_events_by_process.get(str(proc), [])
        explicit_skipped_versions = {
            int(event.get("version", 0))
            for event in skip_events
            if isinstance(event, dict) and int(event.get("version", 0)) > 0
        }
        sequence_assessment = assess_receiver_sequence(
            expected_versions=expected_versions,
            actual_versions=versions,
            allow_receiver_skips=bool(args.allow_receiver_skips),
            explicit_skipped_versions=explicit_skipped_versions,
        )
        missing_versions = list(sequence_assessment.get("missing_versions", []))
        if missing_versions:
            receiver_skips.append(
                {
                    "process_id": proc,
                    "missing_versions": missing_versions,
                    "explicit_skipped_versions": sequence_assessment.get(
                        "explicit_skipped_versions", []
                    ),
                    "accounted_missing_versions": sequence_assessment.get(
                        "accounted_missing_versions", []
                    ),
                    "unaccounted_missing_versions": sequence_assessment.get(
                        "unaccounted_missing_versions", []
                    ),
                }
            )
        if bool(sequence_assessment.get("is_failure", False)):
            receiver_sequence_failures.append(
                {
                    "process_id": proc,
                    **sequence_assessment,
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
                    if delta > publish_to_apply_limit_s:
                        propagation_violations.append(
                            {
                                "process_id": proc,
                                "version": version,
                                "publish_to_apply_s": delta,
                                "threshold_s": publish_to_apply_limit_s,
                            }
                        )

    timeout_analysis = summarize_timeout_reasons(
        receiver_logs=receiver_logs,
        receiver_sequence_failures=receiver_sequence_failures,
    )
    timeout_analysis = merge_timeout_analysis_with_waiting_guard(
        timeout_analysis=timeout_analysis,
        p0_guard=p0_early_stop,
    )

    gs_checks: dict[str, Any] = {
        "enabled": bool(published_by_version),
        "audit_method": gs_probe.get("audit_method", "gs_rpc"),
        "error": gs_probe.get("error") if gs_probe else "missing gs probe payload",
        "old_versions_released": None,
        "old_versions_released_observed": gs_probe.get(
            "old_versions_released_observed"
        ),
        "poll_attempts": int(gs_probe.get("poll_attempts", 0)),
        "poll_elapsed_s": float(gs_probe.get("poll_elapsed_s", 0.0)),
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
            observed_release = bool(
                gs_probe.get("old_versions_released_observed", old_ok)
            )
            gs_checks["old_versions_released_observed"] = observed_release
            gs_checks["old_versions_released"] = bool(old_ok and observed_release)
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
        "probe_modes": [],
        "strict_materializable_checks_enforced": True,
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
        probe_modes: set[str] = set()
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
                mode = str(payload.get("probe_mode", "")).strip()
                if mode:
                    probe_modes.add(mode)
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
        probe_checks["probe_modes"] = sorted(probe_modes)
        # `exists_only` lightweight probe intentionally does not assert true
        # materializability; avoid turning it into a hard pass/fail gate.
        strict_probe = not probe_modes or any(
            mode != "exists_only" for mode in probe_modes
        )
        probe_checks["strict_materializable_checks_enforced"] = bool(strict_probe)

    transport_throughput_summary = dict(transport_metrics.get("throughput", {}))
    if "series" in transport_throughput_summary:
        transport_throughput_summary["series_sample_count"] = float(
            len(transport_throughput_summary.get("series", []))
        )
        transport_throughput_summary.pop("series", None)

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
            "publish_bandwidth_gib_s": summarize_series(publish_bandwidths),
            "put_bandwidth_gib_s": summarize_series(put_bandwidths),
            "apply_latency_s": summarize_series(apply_latencies),
            "publish_to_apply_s": summarize_series(propagation_latencies),
            "transport_throughput_gib_s": transport_throughput_summary,
        },
        "distribution": {
            "transport_diffusion": transport_metrics.get("diffusion", {}),
        },
        "stability": {
            "all_receivers_completed": not receiver_sequence_failures,
            "receiver_skips_present": bool(receiver_skips),
            "allow_receiver_skips": bool(args.allow_receiver_skips),
            "binding_pointer_stable": not pointer_stability_violations,
            "receiver_mode_consistent": not receiver_mode_failures,
            "publish_to_apply_within_limit": not propagation_violations,
            "publish_to_apply_limit_s": publish_to_apply_limit_s,
            "mechanism_keep_last_window_validated": bool(published_by_version),
            "p0_early_stop_triggered": bool(p0_early_stop.get("triggered")),
            "waiting_timeout_observed": bool(
                timeout_analysis.get("waiting_timeout_observed")
            ),
            "transport_timeout_observed": bool(
                timeout_analysis.get("transport_timeout_observed")
            ),
        },
        "retention_checks": {
            "global_store_probe": gs_checks,
            "cluster_probe": probe_checks,
        },
        "timeout_analysis": timeout_analysis,
        "transport_checks": {
            "p0_early_stop": p0_early_stop,
            "group_metadata_probe": transport_group_probe,
            "transport_metrics_error": transport_metrics.get("error"),
            "transport_rows_error": transport_rows_payload.get("error"),
        },
        "failures": {
            "receiver_sequence_failures": receiver_sequence_failures,
            "receiver_skips": receiver_skips,
            "receiver_skip_events": receiver_skip_events_by_process,
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
        and not summary["stability"]["p0_early_stop_triggered"]
    )
    if gs_checks["enabled"]:
        passed = passed and not bool(gs_checks.get("error"))
        passed = passed and bool(gs_checks.get("old_versions_released"))
        passed = passed and bool(gs_checks.get("replica_versions_within_window"))
        passed = passed and bool(gs_checks.get("replica_version_count_within_limit"))
    if probe_checks["enabled"]:
        passed = passed and not bool(probe_checks.get("probe_errors"))
        if bool(probe_checks.get("strict_materializable_checks_enforced", True)):
            passed = passed and bool(probe_checks.get("dropped_non_materializable"))
            passed = passed and bool(
                probe_checks.get("materializable_versions_within_window")
            )
            passed = passed and bool(
                probe_checks.get("materializable_version_count_within_limit")
            )
    if bool(transport_group_probe.get("enabled")):
        passed = passed and not bool(transport_group_probe.get("error"))
        passed = passed and bool(transport_group_probe.get("window_has_transports"))
        passed = passed and bool(transport_group_probe.get("requester_tagged_complete"))
        passed = passed and bool(transport_group_probe.get("group_mode_consistent"))
        passed = passed and bool(transport_group_probe.get("group_contract_consistent"))
    if bool(transport_metrics.get("enabled")):
        passed = passed and not bool(transport_metrics.get("error"))
        passed = passed and int(transport_metrics.get("transport_count", 0)) > 0
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
            "remote_run_as_user": remote_run_as_user,
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
            "tp_device_map_policy": str(args.tp_device_map_policy),
            "tp_materialize_deadline_s": float(args.tp_materialize_deadline_s),
            "transport_group_mode": transport_group_plan.mode,
            "transport_group_kind": transport_group_plan.kind,
            "transport_group_total_parts": transport_group_plan.total_parts,
            "transport_group_priority": transport_group_plan.priority,
            "transport_group_total_parts_formula": (
                "receiver_count*tp_world_size when mode=tp_version, else 0"
            ),
            "max_concurrency": int(args.max_concurrency),
            "cuda_backend": cuda_backend or "process-default",
            "receiver_preflight_transient_overlap": int(
                args.receiver_preflight_transient_overlap
            ),
            "publish_device": str(args.publish_device),
            "progress_poll_s": float(args.progress_poll_s),
            "p0_early_stop": bool(args.p0_early_stop),
            "p0_early_stop_grace_s": float(args.p0_early_stop_grace_s),
            "throughput_sample_interval_s": float(args.throughput_sample_interval_s),
            "throughput_max_samples": int(args.throughput_max_samples),
            "max_publish_to_apply_s": float(args.max_publish_to_apply_s),
            "max_publish_to_apply_auto_adjust": bool(
                args.max_publish_to_apply_auto_adjust
            ),
            "max_publish_to_apply_preflight": dict(
                getattr(args, "_max_publish_to_apply_preflight", {})
            ),
            "retention_timeout_s": float(args.retention_timeout_s),
            "receiver_apply_mode": str(args.receiver_apply_mode),
            "allow_receiver_skips": bool(args.allow_receiver_skips),
            "materialize_device": str(args.materialize_device),
            "receiver_hold_after_finish_s": float(args.receiver_hold_after_finish_s),
            "publisher_hold_after_finish_s": float(args.publisher_hold_after_finish_s),
            "receiver_warmup_s": float(args.receiver_warmup_s),
            "remote_timeout_sec": float(args.remote_timeout_sec),
            "case_transport_window_start_utc": _to_utc_sql_timestamp(
                case_transport_window_start
            ),
            "case_transport_window_end_utc": _to_utc_sql_timestamp(
                case_transport_window_end
            ),
            "cluster_id": cluster_id,
            "daemon_heartbeat_interval": str(args.daemon_heartbeat_interval),
            "daemon_periodic_sync_interval": str(args.daemon_periodic_sync_interval),
            "preclean_roles": bool(args.preclean_roles),
            "publisher_memory_preflight": publisher_memory_preflight,
            "memfd_preflight_by_process": memfd_preflight_by_process,
            "receiver_memory_preflight": receiver_memory_preflight,
        },
        "daemon": {
            "sessions": daemon_sessions,
            "daemon_ids": daemon_ids,
            "advertise_hosts": remote_advertise_hosts,
            "remote_exec_context": remote_exec_context_by_process,
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
        "transport_group_probe": transport_group_probe,
        "p0_early_stop": p0_early_stop,
        "transport_rows_payload": transport_rows_payload,
        "transport_metrics": transport_metrics,
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
                "publish_bandwidth_mean_gib_s": summary["performance"][
                    "publish_bandwidth_gib_s"
                ]["mean"],
                "put_bandwidth_mean_gib_s": summary["performance"][
                    "put_bandwidth_gib_s"
                ]["mean"],
                "apply_latency_mean_s": summary["performance"]["apply_latency_s"][
                    "mean"
                ],
                "publish_to_apply_p95_s": summary["performance"]["publish_to_apply_s"][
                    "p95"
                ],
                "grouped_transports": int(
                    transport_group_probe.get("grouped_transports", 0)
                ),
                "group_probe_error": transport_group_probe.get("error"),
                "p0_early_stop_triggered": bool(
                    summary["stability"].get("p0_early_stop_triggered")
                ),
                "waiting_timeout_reasons": summary.get("timeout_analysis", {}).get(
                    "waiting_timeout_reason_counts", {}
                ),
                "transport_timeout_reasons": summary.get(
                    "timeout_analysis", {}
                ).get("transport_timeout_reason_counts", {}),
                "throughput_peak_gib_s": summary["performance"][
                    "transport_throughput_gib_s"
                ].get("peak_active_throughput_gib_s", 0.0),
                "diffusion_top1_share": summary["distribution"][
                    "transport_diffusion"
                ].get("top1_share", 0.0),
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
