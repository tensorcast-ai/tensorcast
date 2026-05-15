#!/usr/bin/env python3
#  Copyright (c) 2026, TensorCast Team.

"""Convert internal-vLLM TensorCast trace-plan dumps into benchmark load plans.

The benchmark at
`//core/store/materialization/benchmarks:safetensors_load_strategy_benchmark`
expects a compact JSON shape:

{
  "version": 2,
  "ranks": [
    {
      "tp_rank": 0,
      "tp_world_size": 4,
      "tensors": {
        "ckpt.tensor.name": {
          "shape": [...],
          "dtype": "torch.bfloat16",
          "copies": [
            {"dst_param": "...", "slices": [{"axis": 0, "start": 0, "size": 8}]}
          ]
        }
      }
    }
  ]
}

internal-vLLM already emits trace-plan dumps under
`<diagnostics.debug_path>/trace_plan/tensorcast_trace_plan_tp*.json`, but those
files include fill ops and builder-oriented copy-plan fields. This tool lowers
the trace dump into the benchmark contract by:

- loading source tensor metadata from the safetensors checkpoint directory,
- keeping only source-backed `copy` ops,
- translating `ckpt_range` into benchmark `copies[].slices`,
- grouping copies by source checkpoint tensor,
- and emitting one benchmark rank entry per trace-plan file.
"""

from __future__ import annotations

import argparse
import json
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any


SAFE_TENSORS_TO_TORCH_DTYPE: dict[str, str] = {
    "BOOL": "torch.bool",
    "U8": "torch.uint8",
    "I8": "torch.int8",
    "I16": "torch.int16",
    "I32": "torch.int32",
    "I64": "torch.int64",
    "F16": "torch.float16",
    "BF16": "torch.bfloat16",
    "F32": "torch.float32",
    "F64": "torch.float64",
}


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        header_len_raw = f.read(8)
        if len(header_len_raw) != 8:
            raise RuntimeError(f"Invalid safetensors header in {path}")
        header_len = struct.unpack("<Q", header_len_raw)[0]
        header_raw = f.read(header_len)
        if len(header_raw) != header_len:
            raise RuntimeError(f"Truncated safetensors header in {path}")
    return json.loads(header_raw.decode("utf-8"))


def _load_source_metadata(model_dir: Path) -> dict[str, dict[str, Any]]:
    meta_by_name: dict[str, dict[str, Any]] = {}
    shard_paths = sorted(model_dir.glob("*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(f"No .safetensors files found under {model_dir}")

    for shard_path in shard_paths:
        header = _read_safetensors_header(shard_path)
        for tensor_name, tensor_info in header.items():
            if tensor_name == "__metadata__":
                continue
            if tensor_name in meta_by_name:
                raise RuntimeError(f"Duplicate tensor '{tensor_name}' across safetensors shards")
            safe_dtype = str(tensor_info["dtype"])
            torch_dtype = SAFE_TENSORS_TO_TORCH_DTYPE.get(safe_dtype)
            if torch_dtype is None:
                raise ValueError(
                    f"Unsupported safetensors dtype '{safe_dtype}' for tensor '{tensor_name}'")
            meta_by_name[tensor_name] = {
                "shape": [int(dim) for dim in tensor_info["shape"]],
                "dtype": torch_dtype,
            }
    return meta_by_name


def _load_trace_plan(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Trace plan must be a JSON object: {path}")
    if "copy_plan" not in data or not isinstance(data["copy_plan"], list):
        raise ValueError(f"Trace plan missing copy_plan[]: {path}")
    if "tp_rank" not in data or "tp_world_size" not in data:
        raise ValueError(f"Trace plan missing tp_rank/tp_world_size: {path}")
    return data


def _range_spec_to_slices(range_spec: dict[str, Any] | None) -> list[dict[str, int]]:
    if range_spec is None:
        return []
    if range_spec.get("type") == "multi":
        axes = list(range_spec.get("ranges", []))
    else:
        axes = [range_spec]
    slices: list[dict[str, int]] = []
    for axis in axes:
        start = int(axis["start"])
        end = int(axis["end"])
        slices.append(
            {
                "axis": int(axis["dim"]),
                "start": start,
                "size": end - start,
            })
    return slices


def _copy_sort_key(copy_spec: dict[str, Any]) -> tuple[Any, ...]:
    slices = tuple(
        (int(s["axis"]), int(s["start"]), int(s["size"]))
        for s in copy_spec.get("slices", []))
    return (str(copy_spec["dst_param"]), slices)


def _build_rank_entry(
    trace_plan: dict[str, Any],
    source_meta_by_name: dict[str, dict[str, Any]],
    trace_plan_path: Path,
) -> tuple[dict[str, Any], dict[str, int]]:
    tensors: dict[str, dict[str, Any]] = {}
    skipped_fill_ops = 0
    skipped_other_ops = 0

    for entry in trace_plan["copy_plan"]:
        op = str(entry.get("op", ""))
        ckpt_name = entry.get("ckpt_name")
        if op != "copy" or ckpt_name is None:
            if op == "fill":
                skipped_fill_ops += 1
            else:
                skipped_other_ops += 1
            continue

        ckpt_name = str(ckpt_name)
        source_meta = source_meta_by_name.get(ckpt_name)
        if source_meta is None:
            raise KeyError(
                f"Trace plan {trace_plan_path} references tensor not present in source checkpoint: {ckpt_name}")

        tensor_entry = tensors.setdefault(
            ckpt_name,
            {
                "shape": list(source_meta["shape"]),
                "dtype": str(source_meta["dtype"]),
                "copies": [],
            },
        )
        copy_spec: dict[str, Any] = {"dst_param": str(entry["dst_name"])}
        slices = _range_spec_to_slices(entry.get("ckpt_range"))
        if slices:
            copy_spec["slices"] = slices
        tensor_entry["copies"].append(copy_spec)

    sorted_tensors: dict[str, dict[str, Any]] = {}
    for tensor_name in sorted(tensors):
        tensor_entry = tensors[tensor_name]
        tensor_entry["copies"] = sorted(tensor_entry["copies"], key=_copy_sort_key)
        sorted_tensors[tensor_name] = tensor_entry

    rank_entry = {
        "tp_rank": int(trace_plan["tp_rank"]),
        "tp_world_size": int(trace_plan["tp_world_size"]),
        "tensors": sorted_tensors,
    }
    stats = {
        "trace_copy_plan_entries": len(trace_plan["copy_plan"]),
        "kept_copy_ops": sum(len(t["copies"]) for t in sorted_tensors.values()),
        "unique_source_tensors": len(sorted_tensors),
        "skipped_fill_ops": skipped_fill_ops,
        "skipped_other_ops": skipped_other_ops,
    }
    return rank_entry, stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        required=True,
        type=Path,
        help="Source checkpoint directory containing *.safetensors shards.",
    )
    parser.add_argument(
        "--trace-plan",
        action="append",
        default=[],
        type=Path,
        help="Trace-plan JSON path. May be repeated.",
    )
    parser.add_argument(
        "--trace-plan-dir",
        type=Path,
        help="Directory containing tensorcast_trace_plan_tp*.json files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output benchmark load-plan JSON path.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    trace_paths: list[Path] = list(args.trace_plan)
    if args.trace_plan_dir is not None:
        trace_paths.extend(sorted(args.trace_plan_dir.glob("tensorcast_trace_plan_tp*.json")))
    if not trace_paths:
        raise SystemExit("No trace plans provided. Use --trace-plan or --trace-plan-dir.")

    source_meta_by_name = _load_source_metadata(args.model_dir)
    rank_entries: list[dict[str, Any]] = []
    rank_stats: list[dict[str, Any]] = []

    for trace_path in sorted(set(trace_paths)):
        trace_plan = _load_trace_plan(trace_path)
        rank_entry, stats = _build_rank_entry(trace_plan, source_meta_by_name, trace_path)
        rank_entries.append(rank_entry)
        rank_stats.append(
            {
                "trace_plan_path": str(trace_path),
                "tp_rank": rank_entry["tp_rank"],
                "tp_world_size": rank_entry["tp_world_size"],
                **stats,
            })

    rank_entries.sort(key=lambda item: (int(item["tp_world_size"]), int(item["tp_rank"])))
    output_payload = {
        "version": 2,
        "source": "tensorcast_trace_plan",
        "model_dir": str(args.model_dir),
        "rank_stats": rank_stats,
        "ranks": rank_entries,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rank_count": len(rank_entries),
                "unique_tensor_counts": {
                    str(entry["tp_rank"]): len(entry["tensors"]) for entry in rank_entries
                },
                "copy_counts": {
                    str(stat["tp_rank"]): stat["kept_copy_ops"] for stat in rank_stats
                },
            },
            indent=2,
            sort_keys=True,
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
