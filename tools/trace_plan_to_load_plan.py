#!/usr/bin/env python3
#  Copyright (c) 2026, TensorCast Team.

"""Convert vllm TensorCast trace-plan dumps into benchmark load plans.

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

vllm already emits trace-plan dumps under
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
import re
import struct
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
    "F8_E4M3": "torch.float8_e4m3fn",
    "F8_E5M2": "torch.float8_e5m2",
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
                raise RuntimeError(
                    f"Duplicate tensor '{tensor_name}' across safetensors shards"
                )
            safe_dtype = str(tensor_info["dtype"])
            torch_dtype = SAFE_TENSORS_TO_TORCH_DTYPE.get(safe_dtype)
            if torch_dtype is None:
                raise ValueError(
                    f"Unsupported safetensors dtype '{safe_dtype}' for tensor '{tensor_name}'"
                )
            meta_by_name[tensor_name] = {
                "shape": [int(dim) for dim in tensor_info["shape"]],
                "dtype": torch_dtype,
            }
    return meta_by_name


def _infer_tp_rank_from_name(path: Path) -> int | None:
    match = re.search(r"(?:^|[_-])tp(\d+)(?:[_\.-]|$)", path.name)
    if match is None:
        return None
    return int(match.group(1))


def _load_trace_plan(path: Path) -> tuple[dict[str, Any], int | None, int | None]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Trace plan must be a JSON object: {path}")

    trace_plan = data.get("trace_plan", data)
    if not isinstance(trace_plan, dict):
        raise ValueError(f"Trace plan wrapper must contain an object: {path}")
    copy_plan = trace_plan.get("copy_plan")
    has_copy_plan = isinstance(copy_plan, list)
    has_trace_summary = isinstance(
        trace_plan.get("expected_src_names"), list
    ) and isinstance(trace_plan.get("tensorcast_slices"), dict)
    if not has_copy_plan and not has_trace_summary:
        raise ValueError(
            f"Trace plan missing copy_plan[] and trace summary fields: {path}"
        )
    if not has_copy_plan:
        trace_plan["copy_plan"] = []

    tp_rank = trace_plan.get("tp_rank", data.get("tp_rank"))
    tp_world_size = trace_plan.get("tp_world_size", data.get("tp_world_size"))
    if tp_rank is None:
        tp_rank = _infer_tp_rank_from_name(path)
    return (
        trace_plan,
        int(tp_rank) if tp_rank is not None else None,
        int(tp_world_size) if tp_world_size is not None else None,
    )


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
            }
        )
    return slices


def _summary_range_to_slices(range_spec: dict[str, Any] | None) -> list[dict[str, int]]:
    if range_spec is None:
        return []
    start = int(range_spec["start"])
    end = int(range_spec["end"])
    return [
        {
            "axis": int(range_spec["dim"]),
            "start": start,
            "size": end - start,
        }
    ]


def _copy_sort_key(copy_spec: dict[str, Any]) -> tuple[Any, ...]:
    slices = tuple(
        (int(s["axis"]), int(s["start"]), int(s["size"]))
        for s in copy_spec.get("slices", [])
    )
    return (str(copy_spec["dst_param"]), slices)


def _build_rank_entry(
    trace_plan: dict[str, Any],
    source_meta_by_name: dict[str, dict[str, Any]],
    trace_plan_path: Path,
    tp_rank: int,
    tp_world_size: int,
) -> tuple[dict[str, Any], dict[str, int]]:
    tensors: dict[str, dict[str, Any]] = {}
    skipped_fill_ops = 0
    skipped_other_ops = 0
    used_trace_summary_fallback = False

    copy_plan = trace_plan["copy_plan"]
    if not copy_plan and trace_plan.get("expected_src_names"):
        # Newer trace-cache summaries intentionally omit the full copy_plan to
        # keep cached recipes compact. For benchmark-only path microbenchmarks,
        # approximate each source tensor as one destination with the recorded
        # source hull slice. This preserves the source read volume and shape but
        # does not try to reconstruct exact vLLM destination parameter names.
        used_trace_summary_fallback = True
        tensorcast_slices = trace_plan.get("tensorcast_slices") or {}
        for ckpt_name_raw in sorted(trace_plan.get("expected_src_names") or []):
            ckpt_name = str(ckpt_name_raw)
            source_meta = source_meta_by_name.get(ckpt_name)
            if source_meta is None:
                raise KeyError(
                    f"Trace summary {trace_plan_path} references tensor not "
                    f"present in source checkpoint: {ckpt_name}"
                )
            copy_spec: dict[str, Any] = {"dst_param": ckpt_name}
            slices = _summary_range_to_slices(tensorcast_slices.get(ckpt_name))
            if slices:
                copy_spec["slices"] = slices
            tensors[ckpt_name] = {
                "shape": list(source_meta["shape"]),
                "dtype": str(source_meta["dtype"]),
                "copies": [copy_spec],
            }

    for entry in copy_plan:
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
                f"Trace plan {trace_plan_path} references tensor not present in source checkpoint: {ckpt_name}"
            )

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
        "tp_rank": int(tp_rank),
        "tp_world_size": int(tp_world_size),
        "tensors": sorted_tensors,
    }
    stats = {
        "trace_copy_plan_entries": len(trace_plan["copy_plan"]),
        "kept_copy_ops": sum(len(t["copies"]) for t in sorted_tensors.values()),
        "unique_source_tensors": len(sorted_tensors),
        "skipped_fill_ops": skipped_fill_ops,
        "skipped_other_ops": skipped_other_ops,
        "used_trace_summary_fallback": int(used_trace_summary_fallback),
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
        help="Directory containing TensorCast trace-plan JSON files.",
    )
    parser.add_argument(
        "--tp-world-size",
        type=int,
        help=(
            "TP world size to use when trace plans do not embed it. "
            "Defaults to max inferred tp rank + 1."
        ),
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
        trace_paths.extend(sorted(args.trace_plan_dir.glob("*.json")))
    if not trace_paths:
        raise SystemExit(
            "No trace plans provided. Use --trace-plan or --trace-plan-dir."
        )

    source_meta_by_name = _load_source_metadata(args.model_dir)
    rank_entries: list[dict[str, Any]] = []
    rank_stats: list[dict[str, Any]] = []

    loaded_trace_plans: list[tuple[Path, dict[str, Any], int | None, int | None]] = []
    for trace_path in sorted(set(trace_paths)):
        trace_plan, tp_rank, tp_world_size = _load_trace_plan(trace_path)
        loaded_trace_plans.append((trace_path, trace_plan, tp_rank, tp_world_size))

    inferred_ranks = [rank for _, _, rank, _ in loaded_trace_plans if rank is not None]
    default_tp_world_size = (
        int(args.tp_world_size)
        if args.tp_world_size is not None
        else (max(inferred_ranks) + 1 if inferred_ranks else None)
    )
    if default_tp_world_size is None:
        raise SystemExit(
            "Trace plans do not include tp_world_size and tp rank could not be "
            "inferred from filenames. Pass --tp-world-size and use filenames "
            "containing tp<rank>, or provide trace files with embedded metadata."
        )

    for trace_path, trace_plan, tp_rank, tp_world_size in loaded_trace_plans:
        if tp_rank is None:
            raise SystemExit(
                f"Trace plan {trace_path} does not include tp_rank and the rank "
                "could not be inferred from its filename."
            )
        effective_tp_world_size = tp_world_size or default_tp_world_size
        rank_entry, stats = _build_rank_entry(
            trace_plan,
            source_meta_by_name,
            trace_path,
            tp_rank=tp_rank,
            tp_world_size=effective_tp_world_size,
        )
        rank_entries.append(rank_entry)
        rank_stats.append(
            {
                "trace_plan_path": str(trace_path),
                "tp_rank": rank_entry["tp_rank"],
                "tp_world_size": rank_entry["tp_world_size"],
                **stats,
            }
        )

    rank_entries.sort(
        key=lambda item: (int(item["tp_world_size"]), int(item["tp_rank"]))
    )
    output_payload = {
        "version": 2,
        "source": "tensorcast_trace_plan",
        "model_dir": str(args.model_dir),
        "rank_stats": rank_stats,
        "ranks": rank_entries,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rank_count": len(rank_entries),
                "unique_tensor_counts": {
                    str(entry["tp_rank"]): len(entry["tensors"])
                    for entry in rank_entries
                },
                "copy_counts": {
                    str(stat["tp_rank"]): stat["kept_copy_ops"] for stat in rank_stats
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
