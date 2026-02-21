#!/usr/bin/env python
#  Copyright (c) 2026, TensorCast Team.

# Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import argparse
import contextlib
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


def _is_retryable(exc: Exception) -> bool:
    status_code = str(getattr(exc, "status_code", "")).lower()
    message = str(exc).lower()
    if status_code in {
        "not_found",
        "failed_precondition",
        "unavailable",
        "statuscode.not_found",
        "statuscode.failed_precondition",
        "statuscode.unavailable",
    }:
        return True
    if "artifact index not found" in message or "key not found" in message:
        return True
    if "statuscode.not_found" in message:
        return True
    return "not found" in message and (
        "artifact id" in message or "storedaemon" in message or "artifact" in message
    )


def _materialize_with_retry(
    *,
    key: str,
    artifact_id: str,
    lookup_mode: str,
    fallback: tc.FallbackOptions,
    device: str,
    options: tc.GetArtifactOptions,
    visibility_timeout_sec: float,
    visibility_retry_sec: float,
) -> tuple[Any, int]:
    deadline = time.monotonic() + visibility_timeout_sec
    attempt = 1
    while True:
        try:
            if lookup_mode == "artifact_id":
                handle = tc.artifact(artifact_id=artifact_id, fallback=fallback)
            else:
                handle = tc.artifact(key=key, fallback=fallback)
            materialized = handle.tensor_dict_with_diagnostics(
                device=device,
                options=options,
            )
            return materialized, attempt
        except Exception as exc:  # noqa: BLE001
            if not _is_retryable(exc):
                raise
            if time.monotonic() >= deadline:
                raise
            time.sleep(max(0.001, visibility_retry_sec))
            attempt += 1


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

        comm_before = _read_comm_snapshot() if bool(args.capture_comm_stats) else None

        _sync_if_needed(str(args.get_device), bool(args.sync_cuda))
        get_start = time.perf_counter()
        materialized, attempts = _materialize_with_retry(
            key=str(args.key),
            artifact_id=str(args.artifact_id),
            lookup_mode=str(args.lookup_mode),
            fallback=fallback,
            device=str(args.get_device),
            options=options,
            visibility_timeout_sec=float(args.visibility_timeout_sec),
            visibility_retry_sec=float(args.visibility_retry_sec),
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

    output = {
        "key": str(args.key),
        "artifact_id": str(args.artifact_id),
        "lookup_mode": str(args.lookup_mode),
        "source": diagnostics.source or "unknown",
        "source_code": diagnostics.source_code,
        "total_bytes": int(total_bytes),
        "attempts": int(attempts),
        "e2e_sec": e2e_sec,
        "e2e_gibps": _bytes_to_gib_per_sec(total_bytes, e2e_sec),
        "transfer_sec": transfer_sec,
        "transfer_gibps": _bytes_to_gib_per_sec(total_bytes, transfer_sec),
        "visibility_wait_sec": max(0.0, e2e_sec - transfer_sec),
        "comm_transfers_delta": comm_transfers_delta,
        "comm_bytes_delta": comm_bytes_delta,
        "comm_errors_delta": comm_errors_delta,
    }
    print(json.dumps(output, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
