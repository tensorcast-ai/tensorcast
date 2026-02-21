#!/usr/bin/env python3
#  Copyright (c) 2026, TensorCast Team.

"""TensorCast SDK P2P benchmark: one put followed by one get.

This script measures get-side bandwidth using the basic SDK surfaces:
`Store.put(...)` and `Artifact.tensor_dict(...)`.

It is intended for MTCP tuning loops where each trial should report:
- effective bytes
- get latency / throughput
- resolved materialization source (p2p/disk/local)
- replica/ticket diagnostics for root-cause analysis

Recommended runtime topology for meaningful P2P numbers:
1. Start Global Store.
2. Start two Store Daemons connected to the same Global Store.
3. Use daemon A as put endpoint and daemon B as get endpoint.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

import tensorcast as tc
from tensorcast import startup
from tensorcast.api.store.artifact import TensorDictMaterializationResult

DTYPE_BY_NAME: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
}


@dataclass(frozen=True, slots=True)
class TrialRecord:
    case_mib: int
    iteration: int
    warmup: bool
    key: str
    artifact_id: str
    tensor_count: int
    total_bytes: int
    put_sec: float
    get_sec: float
    get_gibps: float
    transfer_sec: float
    transfer_gibps: float
    visibility_wait_sec: float
    source: str
    source_code: int | None
    replica_uuid: str
    ticket_replica_uuid: str | None
    ticket_status: str | None
    materialize_sec: float
    tensor_bind_sec: float
    materialize_total_sec: float
    visibility_attempts: int
    comm_total_transfers_before: int | None
    comm_total_transfers_after: int | None
    comm_transfers_delta: int | None
    comm_bytes_delta: int | None
    comm_errors_delta: int | None


@dataclass(frozen=True, slots=True)
class CaseSummary:
    size_mib: int
    requested_bytes: int
    effective_bytes: int
    warmup: int
    measured: int
    source_counts: dict[str, int]
    p2p_ratio: float
    get_sec_p50: float
    get_sec_p90: float
    get_sec_p99: float
    gibps_p50: float
    gibps_p90: float
    gibps_p99: float
    gibps_mean: float
    transfer_sec_p50: float
    transfer_sec_p90: float
    transfer_sec_p99: float
    transfer_gibps_p50: float
    transfer_gibps_p90: float
    transfer_gibps_p99: float
    transfer_gibps_mean: float
    visibility_wait_sec_p50: float
    visibility_wait_sec_p90: float
    visibility_wait_sec_p99: float
    comm_transfers_delta_p50: float
    comm_transfers_delta_p90: float
    comm_bytes_delta_p50: float
    comm_bytes_delta_p90: float
    comm_errors_delta_sum: int


@dataclass(frozen=True, slots=True)
class DaemonReadyInfo:
    role: str
    daemon_address: str
    daemon_id: str
    worker_id: str
    wait_sec: float
    attempts: int


@dataclass(frozen=True, slots=True)
class CommSnapshot:
    total_transfers: int
    total_bytes_transferred: int
    total_transfer_errors: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TensorCast SDK P2P benchmark (put -> get bandwidth)",
    )
    parser.add_argument(
        "--put-daemon",
        required=True,
        help="Put-side daemon endpoint, e.g. 127.0.0.1:62001",
    )
    parser.add_argument(
        "--get-daemon",
        required=True,
        help="Get-side daemon endpoint, e.g. 127.0.0.1:62011",
    )
    parser.add_argument(
        "--put-device",
        default="cuda:0",
        help="Device for put tensors (default: cuda:0)",
    )
    parser.add_argument(
        "--get-device",
        default="cuda:0",
        help="Device for get materialization (default: cuda:0)",
    )
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPE_BY_NAME.keys()),
        default="float16",
        help="Tensor dtype used for benchmark payload",
    )
    parser.add_argument(
        "--size-mib",
        type=int,
        nargs="+",
        default=[256],
        help="Payload size list in MiB for each case, e.g. --size-mib 64 128 256",
    )
    parser.add_argument(
        "--tensor-count",
        type=int,
        default=1,
        help="Number of tensors in each artifact payload",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warmup iterations per case (excluded from summary)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Measured iterations per case",
    )
    parser.add_argument(
        "--put-policy",
        default="pinned",
        help="Store policy for put (default: pinned)",
    )
    parser.add_argument(
        "--lookup-mode",
        choices=("artifact_id", "key"),
        default="key",
        help="Lookup mode for get-side handle resolution (default: key)",
    )
    parser.add_argument(
        "--prefer",
        choices=("auto", "local", "p2p", "disk"),
        default="p2p",
        help="Fallback preference for get-side artifact handle",
    )
    parser.add_argument(
        "--allow-disk",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow disk fallback during get",
    )
    parser.add_argument(
        "--verify-checksums",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable checksum verification during get",
    )
    parser.add_argument(
        "--export-policy",
        choices=("never", "auto", "force"),
        default="force",
        help="GetArtifactOptions.export_policy",
    )
    parser.add_argument(
        "--pinned-allocation-timeout-ms",
        type=int,
        default=30000,
        help="GetArtifactOptions.pinned_allocation_timeout_ms",
    )
    parser.add_argument(
        "--transport-hold-ms",
        type=int,
        default=None,
        help="GetArtifactOptions.transport_hold_ms hint",
    )
    parser.add_argument(
        "--sync-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Synchronize CUDA device after put/get timing boundaries",
    )
    parser.add_argument(
        "--require-p2p",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail the run if any measured iteration is not sourced from p2p",
    )
    parser.add_argument(
        "--capture-comm-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture get-daemon communication counters before/after each get",
    )
    parser.add_argument(
        "--cleanup-artifacts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Best-effort deregister artifact on both daemons after each trial "
            "(default: false, to reduce metadata churn during throughput tests)"
        ),
    )
    parser.add_argument(
        "--visibility-timeout-sec",
        type=float,
        default=20.0,
        help="Max wait for artifact visibility on get daemon",
    )
    parser.add_argument(
        "--visibility-retry-interval-sec",
        type=float,
        default=0.05,
        help="Retry interval when get sees NOT_FOUND",
    )
    parser.add_argument(
        "--ready-timeout-sec",
        type=float,
        default=30.0,
        help="Max wait for daemon worker registration/health before benchmark starts",
    )
    parser.add_argument(
        "--ready-poll-interval-sec",
        type=float,
        default=0.2,
        help="Polling interval for daemon readiness checks",
    )
    parser.add_argument(
        "--comm-config-path",
        default=None,
        help="Optional communicator YAML/JSON path to snapshot into output",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional JSON output path",
    )
    parser.add_argument(
        "--key-prefix",
        default="bench:sdk:p2p:put-get",
        help="Prefix for generated benchmark keys",
    )
    return parser.parse_args()


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    dtype = DTYPE_BY_NAME.get(dtype_name)
    if dtype is None:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return dtype


def _bytes_to_gib_per_sec(total_bytes: int, elapsed_sec: float) -> float:
    if elapsed_sec <= 0:
        return 0.0
    return float(total_bytes) / elapsed_sec / float(1024**3)


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * quantile
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    frac = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * frac


def _sync_if_needed(device: str, enabled: bool) -> None:
    if not enabled:
        return
    if not device.startswith("cuda"):
        return
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize(torch.device(device))


def _build_payload(
    *,
    total_bytes: int,
    tensor_count: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
) -> tuple[dict[str, torch.Tensor], int]:
    if tensor_count <= 0:
        raise ValueError("tensor_count must be > 0")

    element_size = int(torch.empty((), dtype=dtype).element_size())
    if element_size <= 0:
        raise ValueError("dtype element size must be > 0")

    total_elems = total_bytes // element_size
    if total_elems < tensor_count:
        raise ValueError(
            "Requested payload too small for tensor_count and dtype; "
            f"total_bytes={total_bytes}, tensor_count={tensor_count}, dtype={dtype}"
        )

    base_elems = total_elems // tensor_count
    extra = total_elems % tensor_count

    payload: dict[str, torch.Tensor] = {}
    generated_elems = 0
    floating = bool(torch.empty((), dtype=dtype).is_floating_point())
    for idx in range(tensor_count):
        elems = base_elems + (1 if idx < extra else 0)
        generated_elems += elems
        name = f"tensor_{idx:04d}"
        tensor = torch.empty((int(elems),), dtype=dtype, device=device)
        if floating:
            tensor.fill_(float((seed + idx) % 37) + 0.25)
            if elems > 0:
                tensor[0] = float(seed)
        else:
            tensor.fill_(int((seed + idx) % 97))
            if elems > 0:
                tensor[0] = int(seed)
        payload[name] = tensor

    effective_bytes = generated_elems * element_size
    return payload, int(effective_bytes)


def _materialize_with_retry(
    *,
    artifact_id: str | None,
    key: str,
    fallback: tc.FallbackOptions,
    device: str,
    options: tc.GetArtifactOptions,
    visibility_timeout_sec: float,
    visibility_retry_interval_sec: float,
) -> tuple[TensorDictMaterializationResult, int]:
    deadline = time.monotonic() + visibility_timeout_sec
    attempt = 1
    while True:
        try:
            if artifact_id:
                handle = tc.artifact(artifact_id=artifact_id, fallback=fallback)
            else:
                handle = tc.artifact(key=key, fallback=fallback)
            materialized = handle.tensor_dict_with_diagnostics(
                device=device,
                options=options,
            )
            return materialized, attempt
        except Exception as exc:  # noqa: BLE001
            status_code = str(getattr(exc, "status_code", ""))
            message = str(exc).lower()
            status_code_text = status_code.lower()
            retryable = (
                status_code
                in {
                    "NOT_FOUND",
                    "FAILED_PRECONDITION",
                    "UNAVAILABLE",
                }
                or status_code_text
                in {
                    "not_found",
                    "failed_precondition",
                    "unavailable",
                    "statuscode.not_found",
                    "statuscode.failed_precondition",
                    "statuscode.unavailable",
                }
                or "artifact index not found" in message
                or "key not found" in message
                or "statuscode.not_found" in message
            )
            if not retryable and "not found" in message:
                retryable = (
                    "artifact id" in message
                    or "storedaemon" in message
                    or "artifact" in message
                )
            if not retryable:
                raise
            if time.monotonic() >= deadline:
                raise
            time.sleep(max(0.001, visibility_retry_interval_sec))
            attempt += 1


def _with_daemon_connection(
    daemon_address: str,
    callback,
):
    startup.init(mode="connect", address=daemon_address)
    try:
        return callback()
    finally:
        with contextlib.suppress(Exception):
            startup.shutdown()


def _get_worker_status_once(daemon_address: str):
    ctx = startup.init(mode="connect", address=daemon_address)
    try:
        return ctx.client.get_worker_status()
    finally:
        with contextlib.suppress(Exception):
            startup.shutdown()


def _get_detailed_status_once(daemon_address: str):
    ctx = startup.init(mode="connect", address=daemon_address)
    try:
        return ctx.client.get_detailed_status()
    finally:
        with contextlib.suppress(Exception):
            startup.shutdown()


def _probe_key_visibility(
    *,
    daemon_address: str,
    key: str,
) -> dict[str, Any]:
    probe: dict[str, Any] = {
        "resolved_artifact_id": None,
        "index_bytes": None,
        "resolve_error": None,
        "index_error": None,
    }
    ctx = startup.init(mode="connect", address=daemon_address)
    try:
        client = ctx.client
        try:
            mapping = client.resolve_key_mapping(key)
            resolved_artifact_id = str(mapping.artifact_id or "")
            if resolved_artifact_id:
                probe["resolved_artifact_id"] = resolved_artifact_id
                try:
                    canonical_index = client.get_artifact_index_by_id(
                        resolved_artifact_id
                    )
                    probe["index_bytes"] = int(len(canonical_index))
                except Exception as exc:  # noqa: BLE001
                    probe["index_error"] = str(exc)
        except Exception as exc:  # noqa: BLE001
            probe["resolve_error"] = str(exc)
    finally:
        with contextlib.suppress(Exception):
            startup.shutdown()
    return probe


def _read_comm_snapshot(daemon_address: str) -> CommSnapshot | None:
    try:
        response = _get_detailed_status_once(daemon_address)
    except Exception:  # noqa: BLE001
        return None
    comm = response.communication_info
    return CommSnapshot(
        total_transfers=int(comm.total_transfers),
        total_bytes_transferred=int(comm.total_bytes_transferred),
        total_transfer_errors=int(comm.total_transfer_errors),
    )


def _wait_daemon_ready(
    *,
    daemon_address: str,
    role: str,
    timeout_sec: float,
    poll_interval_sec: float,
) -> DaemonReadyInfo:
    if timeout_sec <= 0:
        raise ValueError("ready timeout must be > 0")
    if poll_interval_sec <= 0:
        raise ValueError("ready poll interval must be > 0")

    deadline = time.monotonic() + timeout_sec
    attempts = 0
    last_status = "uninitialized"
    start = time.monotonic()

    while time.monotonic() < deadline:
        attempts += 1
        try:
            response = _get_worker_status_once(daemon_address)
            is_registered = bool(response.is_registered)
            is_healthy = bool(response.is_healthy)
            worker_id = str(response.worker_id)
            daemon_id = str(response.daemon_id)
            last_status = (
                f"is_registered={is_registered}, "
                f"is_healthy={is_healthy}, worker_id={worker_id or '<empty>'}"
            )
            if is_registered and is_healthy and worker_id:
                wait_sec = time.monotonic() - start
                print(
                    f"[ready] role={role} daemon={daemon_address} worker_id={worker_id} "
                    f"daemon_id={daemon_id} wait={wait_sec:.3f}s attempts={attempts}"
                )
                return DaemonReadyInfo(
                    role=role,
                    daemon_address=daemon_address,
                    daemon_id=daemon_id,
                    worker_id=worker_id,
                    wait_sec=float(wait_sec),
                    attempts=int(attempts),
                )
        except Exception as exc:  # noqa: BLE001
            last_status = f"rpc_error={exc}"
        time.sleep(poll_interval_sec)

    raise RuntimeError(
        "Timed out waiting for daemon readiness: "
        f"role={role}, daemon={daemon_address}, timeout_sec={timeout_sec}, "
        f"attempts={attempts}, last_status={last_status}"
    )


def _put_once(
    *,
    daemon_address: str,
    payload: dict[str, torch.Tensor],
    key: str,
    policy: str,
    put_device: str,
    sync_cuda: bool,
) -> tuple[str, float]:
    def _work() -> tuple[str, float]:
        _sync_if_needed(put_device, sync_cuda)
        put_start = time.perf_counter()
        registered = tc.put(
            payload,
            key=key,
            policy=policy,
        )
        _sync_if_needed(put_device, sync_cuda)
        put_end = time.perf_counter()
        return str(registered.artifact_id), float(put_end - put_start)

    return _with_daemon_connection(daemon_address, _work)


def _get_once(
    *,
    daemon_address: str,
    artifact_id: str | None,
    key: str,
    fallback: tc.FallbackOptions,
    device: str,
    options: tc.GetArtifactOptions,
    visibility_timeout_sec: float,
    visibility_retry_interval_sec: float,
    sync_cuda: bool,
) -> tuple[TensorDictMaterializationResult, int, float]:
    def _work() -> tuple[TensorDictMaterializationResult, int, float]:
        get_start = time.perf_counter()
        materialized, visibility_attempts = _materialize_with_retry(
            artifact_id=artifact_id,
            key=key,
            fallback=fallback,
            device=device,
            options=options,
            visibility_timeout_sec=visibility_timeout_sec,
            visibility_retry_interval_sec=visibility_retry_interval_sec,
        )
        _sync_if_needed(device, sync_cuda)
        get_end = time.perf_counter()
        return materialized, int(visibility_attempts), float(get_end - get_start)

    return _with_daemon_connection(daemon_address, _work)


def _best_effort_deregister(
    *,
    daemon_address: str,
    artifact_id: str,
) -> None:
    def _work() -> None:
        with contextlib.suppress(Exception):
            tc.deregister_artifact(artifact_id, wait=False)

    with contextlib.suppress(Exception):
        _with_daemon_connection(daemon_address, _work)


def _load_comm_config_snapshot(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    from google.protobuf.json_format import MessageToDict

    from tensorcast.communicator.config_io import from_yaml

    cfg = from_yaml(path)
    return MessageToDict(cfg, preserving_proto_field_name=True)


def _summarize_case(
    *,
    size_mib: int,
    requested_bytes: int,
    records: list[TrialRecord],
    warmup: int,
) -> CaseSummary:
    measured_records = [item for item in records if not item.warmup]
    source_counts = Counter(item.source for item in measured_records)
    latencies = [item.get_sec for item in measured_records]
    throughputs = [item.get_gibps for item in measured_records]
    transfer_latencies = [item.transfer_sec for item in measured_records]
    transfer_throughputs = [item.transfer_gibps for item in measured_records]
    visibility_waits = [item.visibility_wait_sec for item in measured_records]
    comm_transfer_deltas = [
        float(item.comm_transfers_delta)
        for item in measured_records
        if item.comm_transfers_delta is not None
    ]
    comm_bytes_deltas = [
        float(item.comm_bytes_delta)
        for item in measured_records
        if item.comm_bytes_delta is not None
    ]
    comm_errors_delta_sum = sum(
        int(item.comm_errors_delta)
        for item in measured_records
        if item.comm_errors_delta is not None
    )
    p2p_count = source_counts.get("p2p", 0)
    measured_count = max(1, len(measured_records))
    effective_bytes = measured_records[0].total_bytes if measured_records else 0
    return CaseSummary(
        size_mib=int(size_mib),
        requested_bytes=int(requested_bytes),
        effective_bytes=int(effective_bytes),
        warmup=int(warmup),
        measured=int(len(measured_records)),
        source_counts=dict(source_counts),
        p2p_ratio=float(p2p_count) / float(measured_count),
        get_sec_p50=_percentile(latencies, 0.50),
        get_sec_p90=_percentile(latencies, 0.90),
        get_sec_p99=_percentile(latencies, 0.99),
        gibps_p50=_percentile(throughputs, 0.50),
        gibps_p90=_percentile(throughputs, 0.90),
        gibps_p99=_percentile(throughputs, 0.99),
        gibps_mean=(sum(throughputs) / float(len(throughputs)) if throughputs else 0.0),
        transfer_sec_p50=_percentile(transfer_latencies, 0.50),
        transfer_sec_p90=_percentile(transfer_latencies, 0.90),
        transfer_sec_p99=_percentile(transfer_latencies, 0.99),
        transfer_gibps_p50=_percentile(transfer_throughputs, 0.50),
        transfer_gibps_p90=_percentile(transfer_throughputs, 0.90),
        transfer_gibps_p99=_percentile(transfer_throughputs, 0.99),
        transfer_gibps_mean=(
            sum(transfer_throughputs) / float(len(transfer_throughputs))
            if transfer_throughputs
            else 0.0
        ),
        visibility_wait_sec_p50=_percentile(visibility_waits, 0.50),
        visibility_wait_sec_p90=_percentile(visibility_waits, 0.90),
        visibility_wait_sec_p99=_percentile(visibility_waits, 0.99),
        comm_transfers_delta_p50=_percentile(comm_transfer_deltas, 0.50),
        comm_transfers_delta_p90=_percentile(comm_transfer_deltas, 0.90),
        comm_bytes_delta_p50=_percentile(comm_bytes_deltas, 0.50),
        comm_bytes_delta_p90=_percentile(comm_bytes_deltas, 0.90),
        comm_errors_delta_sum=int(comm_errors_delta_sum),
    )


def _run_case(
    *,
    case_index: int,
    total_cases: int,
    size_mib: int,
    args: argparse.Namespace,
    put_daemon: str,
    get_daemon: str,
    dtype: torch.dtype,
    run_nonce: int,
) -> tuple[CaseSummary, list[TrialRecord]]:
    requested_bytes = int(size_mib) * 1024 * 1024
    records: list[TrialRecord] = []
    seen_keys: set[str] = set()

    total_iterations = int(args.warmup) + int(args.iterations)
    print(
        f"[case {case_index}/{total_cases}] size_mib={size_mib} "
        f"tensor_count={args.tensor_count} dtype={args.dtype} "
        f"lookup_mode={args.lookup_mode} total_iters={total_iterations}"
    )

    for iteration in range(total_iterations):
        warmup = iteration < int(args.warmup)
        unique_seed = int(run_nonce) + (case_index * 1_000_000) + iteration
        benchmark_key = (
            f"{args.key_prefix}:run{int(run_nonce):x}:"
            f"case{case_index}:mib{size_mib}:iter{iteration}:{uuid.uuid4().hex}"
        )
        if benchmark_key in seen_keys:
            raise RuntimeError(
                f"Generated duplicate benchmark key in one run: key={benchmark_key}"
            )
        seen_keys.add(benchmark_key)

        payload, effective_bytes = _build_payload(
            total_bytes=requested_bytes,
            tensor_count=int(args.tensor_count),
            dtype=dtype,
            device=str(args.put_device),
            seed=unique_seed,
        )

        artifact_id, put_sec = _put_once(
            daemon_address=put_daemon,
            payload=payload,
            key=benchmark_key,
            policy=str(args.put_policy),
            put_device=str(args.put_device),
            sync_cuda=bool(args.sync_cuda),
        )

        fallback = tc.FallbackOptions(
            prefer=str(args.prefer),
            allow_p2p=True,
            allow_disk=bool(args.allow_disk),
            verify_checksums=bool(args.verify_checksums),
            replica_uuid=uuid.uuid4().hex,
        )
        get_options = tc.GetArtifactOptions(
            export_policy=str(args.export_policy),
            pinned_allocation_timeout_ms=int(args.pinned_allocation_timeout_ms),
            wait_for_completion=True,
            enable_verification=False,
            transport_hold_ms=(
                int(args.transport_hold_ms)
                if args.transport_hold_ms is not None
                else None
            ),
        )
        comm_before = (
            _read_comm_snapshot(get_daemon) if bool(args.capture_comm_stats) else None
        )

        try:
            materialized, visibility_attempts, get_sec = _get_once(
                daemon_address=get_daemon,
                artifact_id=(
                    artifact_id if str(args.lookup_mode) == "artifact_id" else None
                ),
                key=benchmark_key,
                fallback=fallback,
                device=str(args.get_device),
                options=get_options,
                visibility_timeout_sec=float(args.visibility_timeout_sec),
                visibility_retry_interval_sec=float(args.visibility_retry_interval_sec),
                sync_cuda=bool(args.sync_cuda),
            )
        except Exception as exc:  # noqa: BLE001
            visibility_probe = _probe_key_visibility(
                daemon_address=get_daemon,
                key=benchmark_key,
            )
            raise RuntimeError(
                "Get failed for benchmark iteration with key mapping context: "
                f"iter={iteration}, key={benchmark_key}, put_artifact_id={artifact_id}, "
                f"lookup_mode={args.lookup_mode}, visibility_probe={visibility_probe}"
            ) from exc
        comm_after = (
            _read_comm_snapshot(get_daemon) if bool(args.capture_comm_stats) else None
        )

        diagnostics = materialized.diagnostics
        measured_bytes = int(diagnostics.total_bytes)
        if measured_bytes <= 0:
            measured_bytes = int(effective_bytes)

        get_gibps = _bytes_to_gib_per_sec(measured_bytes, get_sec)
        transfer_sec = max(float(diagnostics.total_sec), 0.0)
        transfer_gibps = _bytes_to_gib_per_sec(measured_bytes, transfer_sec)
        visibility_wait_sec = max(0.0, float(get_sec) - float(transfer_sec))
        source = diagnostics.source or "unknown"
        comm_transfers_delta: int | None = None
        comm_bytes_delta: int | None = None
        comm_errors_delta: int | None = None
        if comm_before is not None and comm_after is not None:
            comm_transfers_delta = max(
                0, comm_after.total_transfers - comm_before.total_transfers
            )
            comm_bytes_delta = max(
                0,
                comm_after.total_bytes_transferred
                - comm_before.total_bytes_transferred,
            )
            comm_errors_delta = max(
                0,
                comm_after.total_transfer_errors - comm_before.total_transfer_errors,
            )

        record = TrialRecord(
            case_mib=int(size_mib),
            iteration=int(iteration),
            warmup=bool(warmup),
            key=benchmark_key,
            artifact_id=artifact_id,
            tensor_count=int(diagnostics.tensor_count),
            total_bytes=int(measured_bytes),
            put_sec=float(put_sec),
            get_sec=get_sec,
            get_gibps=get_gibps,
            transfer_sec=transfer_sec,
            transfer_gibps=transfer_gibps,
            visibility_wait_sec=visibility_wait_sec,
            source=source,
            source_code=diagnostics.source_code,
            replica_uuid=str(diagnostics.replica_uuid),
            ticket_replica_uuid=diagnostics.ticket_replica_uuid,
            ticket_status=diagnostics.ticket_status,
            materialize_sec=float(diagnostics.materialize_sec),
            tensor_bind_sec=float(diagnostics.tensor_bind_sec),
            materialize_total_sec=float(diagnostics.total_sec),
            visibility_attempts=int(visibility_attempts),
            comm_total_transfers_before=(
                comm_before.total_transfers if comm_before is not None else None
            ),
            comm_total_transfers_after=(
                comm_after.total_transfers if comm_after is not None else None
            ),
            comm_transfers_delta=comm_transfers_delta,
            comm_bytes_delta=comm_bytes_delta,
            comm_errors_delta=comm_errors_delta,
        )
        records.append(record)

        tag = "warmup" if warmup else "measure"
        comm_suffix = ""
        if record.comm_transfers_delta is not None:
            comm_suffix = (
                f" comm_delta={{transfers:{record.comm_transfers_delta},"
                f"bytes:{record.comm_bytes_delta},errors:{record.comm_errors_delta}}}"
            )
        print(
            f"  [{tag}] iter={iteration:02d} source={record.source:<13} "
            f"bytes={record.total_bytes} put={record.put_sec:.4f}s "
            f"get={record.get_sec:.4f}s bw={record.get_gibps:.3f} GiB/s "
            f"xfer={record.transfer_sec:.4f}s xfer_bw={record.transfer_gibps:.3f} GiB/s "
            f"vis_wait={record.visibility_wait_sec:.4f}s "
            f"vis_retry={record.visibility_attempts}"
            f"{comm_suffix}"
        )

        if bool(args.require_p2p) and not warmup and record.source != "p2p":
            raise RuntimeError(
                "Measured iteration was not served by p2p. "
                f"artifact_id={record.artifact_id}, source={record.source}, "
                f"source_code={record.source_code}"
            )

        # Release references before optional cleanup.
        del materialized
        del payload

        if bool(args.cleanup_artifacts):
            _best_effort_deregister(
                daemon_address=get_daemon,
                artifact_id=record.artifact_id,
            )
            _best_effort_deregister(
                daemon_address=put_daemon,
                artifact_id=record.artifact_id,
            )

    summary = _summarize_case(
        size_mib=int(size_mib),
        requested_bytes=int(requested_bytes),
        records=records,
        warmup=int(args.warmup),
    )

    print(
        "  [summary] "
        f"e2e_p50={summary.gibps_p50:.3f} GiB/s "
        f"e2e_mean={summary.gibps_mean:.3f} GiB/s "
        f"xfer_p50={summary.transfer_gibps_p50:.3f} GiB/s "
        f"xfer_mean={summary.transfer_gibps_mean:.3f} GiB/s "
        f"vis_wait_p90={summary.visibility_wait_sec_p90:.4f}s "
        f"comm_transfers_p50={summary.comm_transfers_delta_p50:.2f} "
        f"comm_bytes_p50={summary.comm_bytes_delta_p50:.0f} "
        f"p2p_ratio={summary.p2p_ratio:.2%}"
    )
    return summary, records


def main() -> int:
    args = _parse_args()
    dtype = _resolve_dtype(str(args.dtype))
    if str(args.lookup_mode) != "key":
        print(
            "[warn] lookup_mode=artifact_id bypasses key/index resolution; "
            "use only for control comparisons."
        )

    sizes = [int(item) for item in args.size_mib]
    if any(item <= 0 for item in sizes):
        raise ValueError("--size-mib values must be positive")
    if int(args.iterations) <= 0:
        raise ValueError("--iterations must be > 0")
    if int(args.warmup) < 0:
        raise ValueError("--warmup must be >= 0")
    if float(args.ready_timeout_sec) <= 0:
        raise ValueError("--ready-timeout-sec must be > 0")
    if float(args.ready_poll_interval_sec) <= 0:
        raise ValueError("--ready-poll-interval-sec must be > 0")

    comm_config_snapshot = _load_comm_config_snapshot(args.comm_config_path)
    run_nonce = int(time.time_ns() & 0x7FFFFFFF)
    ready_infos = [
        _wait_daemon_ready(
            daemon_address=str(args.put_daemon),
            role="put",
            timeout_sec=float(args.ready_timeout_sec),
            poll_interval_sec=float(args.ready_poll_interval_sec),
        ),
        _wait_daemon_ready(
            daemon_address=str(args.get_daemon),
            role="get",
            timeout_sec=float(args.ready_timeout_sec),
            poll_interval_sec=float(args.ready_poll_interval_sec),
        ),
    ]

    started = time.time()
    case_reports: list[dict[str, Any]] = []

    total_cases = len(sizes)
    for idx, size_mib in enumerate(sizes, start=1):
        summary, records = _run_case(
            case_index=idx,
            total_cases=total_cases,
            size_mib=size_mib,
            args=args,
            put_daemon=str(args.put_daemon),
            get_daemon=str(args.get_daemon),
            dtype=dtype,
            run_nonce=run_nonce,
        )
        case_reports.append(
            {
                "summary": asdict(summary),
                "trials": [asdict(item) for item in records],
            }
        )

    elapsed = time.time() - started
    all_measured: list[TrialRecord] = [
        TrialRecord(**item)
        for case in case_reports
        for item in case["trials"]
        if not item["warmup"]
    ]

    overall_sources = Counter(item.source for item in all_measured)
    overall_throughput = [item.get_gibps for item in all_measured]
    overall_transfer_throughput = [item.transfer_gibps for item in all_measured]
    overall_visibility_wait = [item.visibility_wait_sec for item in all_measured]
    overall_comm_transfer_delta = [
        float(item.comm_transfers_delta)
        for item in all_measured
        if item.comm_transfers_delta is not None
    ]
    overall_comm_bytes_delta = [
        float(item.comm_bytes_delta)
        for item in all_measured
        if item.comm_bytes_delta is not None
    ]
    overall_comm_errors_delta_sum = sum(
        int(item.comm_errors_delta)
        for item in all_measured
        if item.comm_errors_delta is not None
    )

    output = {
        "started_at_unix": started,
        "elapsed_sec": elapsed,
        "put_daemon": str(args.put_daemon),
        "get_daemon": str(args.get_daemon),
        "dtype": str(args.dtype),
        "sizes_mib": sizes,
        "tensor_count": int(args.tensor_count),
        "cuda_backend": os.environ.get("TENSORCAST_CUDA_BACKEND", "real"),
        "args": vars(args),
        "daemon_ready": [asdict(item) for item in ready_infos],
        "communicator_config": comm_config_snapshot,
        "cases": case_reports,
        "overall": {
            "measured_trials": len(all_measured),
            "source_counts": dict(overall_sources),
            "e2e_gibps_p50": _percentile(overall_throughput, 0.50),
            "e2e_gibps_p90": _percentile(overall_throughput, 0.90),
            "e2e_gibps_p99": _percentile(overall_throughput, 0.99),
            "e2e_gibps_mean": (
                sum(overall_throughput) / float(len(overall_throughput))
                if overall_throughput
                else 0.0
            ),
            "transfer_gibps_p50": _percentile(overall_transfer_throughput, 0.50),
            "transfer_gibps_p90": _percentile(overall_transfer_throughput, 0.90),
            "transfer_gibps_p99": _percentile(overall_transfer_throughput, 0.99),
            "transfer_gibps_mean": (
                sum(overall_transfer_throughput)
                / float(len(overall_transfer_throughput))
                if overall_transfer_throughput
                else 0.0
            ),
            "visibility_wait_sec_p50": _percentile(overall_visibility_wait, 0.50),
            "visibility_wait_sec_p90": _percentile(overall_visibility_wait, 0.90),
            "visibility_wait_sec_p99": _percentile(overall_visibility_wait, 0.99),
            "comm_transfers_delta_p50": _percentile(overall_comm_transfer_delta, 0.50),
            "comm_transfers_delta_p90": _percentile(overall_comm_transfer_delta, 0.90),
            "comm_bytes_delta_p50": _percentile(overall_comm_bytes_delta, 0.50),
            "comm_bytes_delta_p90": _percentile(overall_comm_bytes_delta, 0.90),
            "comm_errors_delta_sum": int(overall_comm_errors_delta_sum),
        },
    }

    print("[overall]", json.dumps(output["overall"], indent=2))

    if args.json_out:
        out_path = Path(str(args.json_out))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"[output] wrote benchmark json to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
