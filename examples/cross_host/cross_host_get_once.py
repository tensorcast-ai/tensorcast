#!/usr/bin/env python
#  Copyright (c) 2026, TensorCast Team.

# Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

import torch

import tensorcast as tc
from tensorcast import startup


@dataclass(frozen=True)
class CommSnapshot:
    total_transfers: int
    total_bytes_transferred: int
    total_transfer_errors: int


def _bytes_to_gib_per_sec(num_bytes: int, sec: float) -> float:
    if num_bytes <= 0 or sec <= 0:
        return 0.0
    return float(num_bytes) / float(1024**3) / float(sec)


def _sync_if_needed(device: str, enabled: bool) -> None:
    if not enabled:
        return
    if device.startswith("cuda:") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _sample_positions(numel: int, sample_count: int) -> tuple[int, ...]:
    if numel <= 0:
        return ()
    if sample_count <= 1 or numel == 1:
        return (0,)
    points: set[int] = {0, numel - 1, numel // 2}
    while len(points) < sample_count:
        idx = int((numel - 1) * (len(points) / float(sample_count - 1)))
        points.add(max(0, min(numel - 1, idx)))
    return tuple(sorted(points))


def _payload_sample_hash(
    tensors: dict[str, torch.Tensor],
    *,
    sample_count: int = 3,
) -> str:
    chunks: list[str] = []
    for name in sorted(tensors):
        tensor = tensors[name].detach().reshape(-1)
        numel = int(tensor.numel())
        positions = _sample_positions(numel, sample_count)
        sampled_values: list[str] = []
        for pos in positions:
            value = tensor[pos].item()
            sampled_values.append(repr(value))
        chunks.append(
            "|".join(
                (
                    name,
                    str(tensors[name].dtype),
                    str(numel),
                    ",".join(sampled_values),
                )
            )
        )
    digest = hashlib.sha256("\n".join(chunks).encode("utf-8")).hexdigest()
    return digest


def _read_comm_snapshot() -> CommSnapshot | None:
    try:
        client = startup.current_client()
        response = client.get_detailed_status()
    except Exception:  # noqa: BLE001
        return None
    comm = response.communication_info
    return CommSnapshot(
        total_transfers=int(comm.total_transfers),
        total_bytes_transferred=int(comm.total_bytes_transferred),
        total_transfer_errors=int(comm.total_transfer_errors),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-host helper: run one get from a local worker process."
    )
    parser.add_argument("--daemon", required=True, help="Get daemon address host:port")
    parser.add_argument("--key", required=True)
    parser.add_argument("--artifact-id", default="")
    parser.add_argument("--lookup-mode", choices=("key", "artifact_id"), default="key")
    parser.add_argument("--get-device", default="cuda:0")
    parser.add_argument(
        "--prefer", choices=("auto", "local", "p2p", "disk"), default="p2p"
    )
    parser.add_argument(
        "--allow-disk",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--verify-checksums",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--payload-sample-verify",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--export-policy",
        choices=("never", "auto", "force"),
        default="force",
    )
    parser.add_argument("--pinned-allocation-timeout-ms", type=int, default=30000)
    parser.add_argument("--transport-hold-ms", type=int, default=None)
    parser.add_argument("--visibility-timeout-sec", type=float, default=30.0)
    parser.add_argument("--visibility-retry-sec", type=float, default=0.05)
    parser.add_argument(
        "--sync-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--capture-comm-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    startup.init(mode="connect", address=str(args.daemon))
    try:
        fallback = tc.FallbackOptions(
            prefer=str(args.prefer),
            allow_p2p=True,
            allow_disk=bool(args.allow_disk),
            verify_checksums=bool(args.verify_checksums),
        )
        options = tc.GetArtifactOptions(
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

        ctx = None
        if float(args.visibility_timeout_sec) > 0:
            ctx = tc.CallContext(deadline_ms=int(args.visibility_timeout_sec * 1000))

        comm_before = _read_comm_snapshot() if bool(args.capture_comm_stats) else None

        _sync_if_needed(str(args.get_device), bool(args.sync_cuda))
        get_start = time.perf_counter()
        if str(args.lookup_mode) == "artifact_id":
            handle = tc.artifact(
                artifact_id=str(args.artifact_id),
                fallback=fallback,
            )
        else:
            handle = tc.artifact(
                key=str(args.key),
                fallback=fallback,
            )
        materialized = handle.tensor_dict_with_diagnostics(
            device=str(args.get_device),
            options=options,
            ctx=ctx,
        )
        _sync_if_needed(str(args.get_device), bool(args.sync_cuda))
        get_end = time.perf_counter()

        comm_after = _read_comm_snapshot() if bool(args.capture_comm_stats) else None
    finally:
        with contextlib.suppress(Exception):
            startup.shutdown()

    diagnostics = materialized.diagnostics
    total_bytes = int(diagnostics.total_bytes)
    transfer_sec = max(float(diagnostics.total_sec), 0.0)
    e2e_sec = float(get_end - get_start)

    comm_transfers_delta: int | None = None
    comm_bytes_delta: int | None = None
    comm_errors_delta: int | None = None
    if comm_before is not None and comm_after is not None:
        comm_transfers_delta = max(
            0, comm_after.total_transfers - comm_before.total_transfers
        )
        comm_bytes_delta = max(
            0,
            comm_after.total_bytes_transferred - comm_before.total_bytes_transferred,
        )
        comm_errors_delta = max(
            0,
            comm_after.total_transfer_errors - comm_before.total_transfer_errors,
        )

    payload_sample_hash: str | None = None
    if bool(args.payload_sample_verify):
        payload_sample_hash = _payload_sample_hash(materialized.tensors)

    retry_reason_buckets: dict[str, int] = dict(diagnostics.retry_reason_buckets)
    budget_trace: dict[str, Any] = {
        "deadline_sec": diagnostics.budget_deadline_sec,
        "elapsed_sec": diagnostics.budget_elapsed_sec,
        "remaining_sec": diagnostics.budget_remaining_sec,
        "exit_reason": diagnostics.budget_exit_reason,
    }
    source_reselection_count = max(0, int(diagnostics.retry_attempts) - 1)

    output = {
        "key": str(args.key),
        "artifact_id": str(args.artifact_id),
        "lookup_mode": str(args.lookup_mode),
        "source": diagnostics.source or "unknown",
        "source_code": diagnostics.source_code,
        "ticket_replica_uuid": diagnostics.ticket_replica_uuid,
        "ticket_status": diagnostics.ticket_status,
        "total_bytes": int(total_bytes),
        "attempts": int(diagnostics.retry_attempts),
        "retry_reason_buckets": retry_reason_buckets,
        "source_reselection_count": int(source_reselection_count),
        "final_error_code": "OK",
        "budget_trace": budget_trace,
        "e2e_sec": e2e_sec,
        "e2e_gibps": _bytes_to_gib_per_sec(total_bytes, e2e_sec),
        "transfer_sec": transfer_sec,
        "transfer_gibps": _bytes_to_gib_per_sec(total_bytes, transfer_sec),
        "visibility_wait_sec": max(0.0, e2e_sec - transfer_sec),
        "transport_wait_ms": int(max(0.0, e2e_sec - transfer_sec) * 1000.0),
        "payload_sample_hash": payload_sample_hash,
        "comm_transfers_delta": comm_transfers_delta,
        "comm_bytes_delta": comm_bytes_delta,
        "comm_errors_delta": comm_errors_delta,
    }
    print(json.dumps(output, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
