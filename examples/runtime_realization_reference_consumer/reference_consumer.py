#!/usr/bin/env python3
#  Copyright (c) 2026, TensorCast Team.

"""Minimal runtime realization consumer for TensorCast prefetch/acquire flows."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from tensorcast.api.store import (
    ReferenceRuntimeTensorSpec,
    acquire_reference_binding,
    build_reference_resolved_spec,
    prefetch_reference_binding,
    release_reference_acquire,
    target_from_reference_cache_record,
    write_reference_resolved_spec_cache_entry,
)
from tensorcast.daemon_ctl import DaemonCtl
from tensorcast.proto.operation.v1 import operation_pb2
from tensorcast.types import PrefetchHandoff, RealizationTarget


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal TensorCast runtime realization prefetch/acquire flow."
    )
    parser.add_argument("--daemon-address", default="127.0.0.1:8073")
    parser.add_argument("--source-artifact-id", required=True)
    parser.add_argument("--device-uuid", required=True)
    parser.add_argument("--cache-root", default="")
    parser.add_argument("--tensor-name", default="alpha")
    parser.add_argument("--tensor-size-bytes", type=int, default=4)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--target-path", default="")
    parser.add_argument("--prefetched-path", default="")
    return parser.parse_args()


def _worker_main(args: argparse.Namespace) -> None:
    if not args.target_path or not args.prefetched_path:
        raise SystemExit("--target-path and --prefetched-path are required for worker")
    target_proto = operation_pb2.ServingBindingTarget()
    target_proto.ParseFromString(Path(args.target_path).read_bytes())
    target = RealizationTarget.from_proto(target_proto)
    prefetched_proto = operation_pb2.PrefetchServingBindingResult()
    prefetched_proto.ParseFromString(Path(args.prefetched_path).read_bytes())
    prefetched = PrefetchHandoff.from_proto(prefetched_proto)
    client = DaemonCtl(args.daemon_address)
    acquired = acquire_reference_binding(
        client,
        prefetched=prefetched,
        target=target,
        caller_pid=os.getpid(),
    )
    try:
        print(
            json.dumps(
                {
                    "binding_value_ref": acquired.binding_value_ref.model_dump(
                        mode="json"
                    ),
                    "lease_token_bytes": len(acquired.lease_token),
                    "has_cuda_ipc_handle": acquired.has_cuda_ipc_handle,
                    "has_cpu_memfd_handle": acquired.has_cpu_memfd_handle,
                },
                sort_keys=True,
            )
        )
    finally:
        release_reference_acquire(client, acquire_result=acquired)


def _parent_main(args: argparse.Namespace) -> None:
    cache_root = Path(args.cache_root) if args.cache_root else Path(tempfile.mkdtemp())
    client = DaemonCtl(args.daemon_address)
    tensor = ReferenceRuntimeTensorSpec(
        name=args.tensor_name,
        size_bytes=args.tensor_size_bytes,
        shape=(args.tensor_size_bytes,),
        stride=(1,),
    )
    resolved = build_reference_resolved_spec(
        source_artifact_id=args.source_artifact_id,
        artifact_selection_digest=args.source_artifact_id,
        device_uuid=args.device_uuid,
        tensor=tensor,
    )
    record = write_reference_resolved_spec_cache_entry(
        cache_root,
        resolved_spec=resolved,
    )
    target = target_from_reference_cache_record(
        record,
        device_uuid=args.device_uuid,
    )
    prefetched = prefetch_reference_binding(
        client,
        source_artifact_id=args.source_artifact_id,
        target=target,
    )
    with tempfile.TemporaryDirectory(prefix="tc_reference_consumer_") as tmp:
        tmp_path = Path(tmp)
        target_path = tmp_path / "target.pb"
        prefetched_path = tmp_path / "prefetched.pb"
        target_path.write_bytes(target.to_proto().SerializeToString(deterministic=True))
        prefetched_path.write_bytes(
            prefetched.to_proto().SerializeToString(deterministic=True)
        )
        worker = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--daemon-address",
                args.daemon_address,
                "--source-artifact-id",
                args.source_artifact_id,
                "--device-uuid",
                args.device_uuid,
                "--target-path",
                str(target_path),
                "--prefetched-path",
                str(prefetched_path),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
    print(
        json.dumps(
            {
                "cache_root": str(cache_root),
                "prefetched": prefetched.model_dump(mode="json"),
                "worker": json.loads(worker.stdout),
            },
            sort_keys=True,
        )
    )


def main() -> None:
    args = _parse_args()
    if args.worker:
        _worker_main(args)
        return
    _parent_main(args)


if __name__ == "__main__":
    main()
