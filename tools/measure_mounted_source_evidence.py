#!/usr/bin/env python3
#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

from tensorcast.api.store import Store
from tensorcast.common.identity import infer_artifact_id_kind


def _artifact_id_kind_label(artifact_id: str) -> str | None:
    kind = infer_artifact_id_kind(artifact_id)
    return kind.value.lower() if kind is not None else None


def _resolve_store(daemon_endpoint: str | None) -> Store:
    if daemon_endpoint:
        return Store(daemon_endpoint)
    from tensorcast.api.store import store as get_store

    return get_store()


def _describe_artifact(artifact) -> dict[str, Any]:
    desc = artifact.describe()
    return {
        "artifact_id": desc.artifact_id,
        "artifact_id_kind": _artifact_id_kind_label(desc.artifact_id),
        "generation": desc.generation,
        "tensor_count": len(desc.tensor_names),
        "total_bytes": desc.total_bytes,
    }


def _materialize_artifact(artifact, device: str) -> dict[str, Any]:
    started = time.perf_counter()
    result = artifact.tensor_dict_with_diagnostics(device=device)
    elapsed = time.perf_counter() - started
    diagnostics_obj = result.diagnostics
    if diagnostics_obj is None:
        diagnostics = None
    elif hasattr(diagnostics_obj, "model_dump"):
        diagnostics = diagnostics_obj.model_dump()
    elif hasattr(diagnostics_obj, "dict"):
        diagnostics = diagnostics_obj.dict()
    elif dataclasses.is_dataclass(diagnostics_obj):
        diagnostics = dataclasses.asdict(diagnostics_obj)
    else:
        diagnostics = {
            name: getattr(diagnostics_obj, name)
            for name in dir(diagnostics_obj)
            if not name.startswith("_") and not callable(getattr(diagnostics_obj, name))
        }
    return {
        "elapsed_sec": elapsed,
        "tensor_count": len(result.tensors),
        "diagnostics": diagnostics,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    target_path = str(Path(args.path).expanduser())
    store = _resolve_store(args.daemon_endpoint)
    output: dict[str, Any] = {
        "path": target_path,
        "mode": args.mode,
        "verify_checksums": bool(args.verify_checksums),
        "show_progress": bool(args.show_progress),
        "daemon_endpoint": args.daemon_endpoint,
        "timeout_s": args.timeout_s,
    }

    try:
        if args.mode == "from_disk":
            started = time.perf_counter()
            artifact = store.from_disk(
                target_path,
                verify_checksums=bool(args.verify_checksums),
                show_progress=bool(args.show_progress),
            )
            output["operation"] = {
                "elapsed_sec": time.perf_counter() - started,
                **_describe_artifact(artifact),
            }
        elif args.mode == "import_from_disk":
            started = time.perf_counter()
            artifact = store.import_from_disk(
                target_path,
                verify_checksums=bool(args.verify_checksums),
                show_progress=bool(args.show_progress),
            )
            output["operation"] = {
                "elapsed_sec": time.perf_counter() - started,
                **_describe_artifact(artifact),
            }
        elif args.mode == "promote_mounted_source":
            resolve_started = time.perf_counter()
            source = store.resolve_public_disk_source(
                target_path,
                verify_checksums=bool(args.verify_checksums),
            )
            resolve_elapsed = time.perf_counter() - resolve_started

            promote_started = time.perf_counter()
            artifact = store.promote_mounted_source(
                source.artifact_id,
                verify_checksums=bool(args.verify_checksums),
                timeout_s=args.timeout_s,
            )
            promote_elapsed = time.perf_counter() - promote_started

            output["resolve"] = {
                "elapsed_sec": resolve_elapsed,
                "artifact_id": source.artifact_id,
                "artifact_id_kind": _artifact_id_kind_label(source.artifact_id),
                "trusted_content_artifact_id": source.trusted_content_artifact_id,
                "metadata_capability": (
                    source.metadata_capability.value
                    if source.metadata_capability is not None
                    else None
                ),
                "resolution_strategy": (
                    source.resolution_strategy.value
                    if source.resolution_strategy is not None
                    else None
                ),
                "policy_id": source.policy_id,
                "exact_size_bytes": source.exact_size_bytes,
            }
            output["promotion"] = {
                "elapsed_sec": promote_elapsed,
                **_describe_artifact(artifact),
            }
        else:
            raise ValueError(f"unsupported mode: {args.mode}")

        if args.device:
            output["materialize"] = _materialize_artifact(
                artifact, device=str(args.device)
            )
        return output
    finally:
        with contextlib.suppress(Exception):
            store.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure mounted-source evidence for from_disk/import_from_disk/"
            "promote_mounted_source and optional materialization."
        )
    )
    parser.add_argument("path", help="Mounted source directory to measure")
    parser.add_argument(
        "--mode",
        choices=("from_disk", "import_from_disk", "promote_mounted_source"),
        default="from_disk",
        help="Which mounted-source path to exercise",
    )
    parser.add_argument(
        "--daemon-endpoint",
        default=None,
        help="Explicit daemon endpoint. Defaults to the current TensorCast session/runtime.",
    )
    parser.add_argument(
        "--verify-checksums",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to require descriptor validation when supported",
    )
    parser.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Request progress output for import_from_disk / from_disk",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional materialization target (for example cuda:0 or cpu)",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=None,
        help="Optional RPC timeout override for promote_mounted_source",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as exc:  # noqa: BLE001
        error = {
            "path": str(Path(args.path).expanduser()),
            "mode": args.mode,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
        print(json.dumps(error, ensure_ascii=False, indent=2 if args.pretty else None))
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
