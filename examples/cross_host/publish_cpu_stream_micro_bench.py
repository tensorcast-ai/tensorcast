#!/usr/bin/env python3
#  Copyright (c) 2026, TensorCast Team.

# Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import Sequence

import torch

TP_RANKED_SOURCE_ELEMS_LIMIT = (1 << 31) - 1


def _normalize_total_bytes(total_bytes: int) -> int:
    value = int(total_bytes)
    if value <= 0:
        raise ValueError("total-bytes must be > 0")
    return value


def _normalize_chunk_bytes(chunk_bytes: int) -> int:
    value = int(chunk_bytes)
    if value <= 0:
        raise ValueError("chunk-bytes must be > 0")
    return value


def _tp_ranked_chunk_elems(tp_world_size: int, total_bytes: int) -> list[int]:
    bytes_per_rank_elem = int(tp_world_size) * 2 * 4
    total_rank_elems = int(total_bytes) // bytes_per_rank_elem
    if total_rank_elems <= 0:
        raise ValueError(
            f"total-bytes too small for tp_world_size={tp_world_size}: total-bytes={total_bytes}"
        )
    max_rank_elems_per_chunk = TP_RANKED_SOURCE_ELEMS_LIMIT // int(tp_world_size)
    if max_rank_elems_per_chunk <= 0:
        raise ValueError("invalid tp_world_size")
    out: list[int] = []
    remaining = int(total_rank_elems)
    while remaining > 0:
        chunk = min(remaining, max_rank_elems_per_chunk)
        out.append(int(chunk))
        remaining -= int(chunk)
    return out


def _build_tp_ranked_tensors(
    tp_world_size: int, total_bytes: int
) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    for rank_block_elems in _tp_ranked_chunk_elems(tp_world_size, total_bytes):
        tensors.append(
            torch.empty(
                (tp_world_size, rank_block_elems),
                dtype=torch.float32,
                device="cpu",
            )
        )
        tensors.append(
            torch.empty(
                (1, tp_world_size * rank_block_elems),
                dtype=torch.float32,
                device="cpu",
            )
        )
    for tensor in tensors:
        tensor.fill_(1.0)
    return tensors


def _tensor_views(tensors: Sequence[torch.Tensor]) -> list[memoryview]:
    views: list[memoryview] = []
    for tensor in tensors:
        local = tensor.detach().contiguous()
        views.append(memoryview(local.view(torch.uint8).numpy()).cast("B"))
    return views


@dataclass(frozen=True)
class BenchResult:
    mode: str
    chunk_bytes: int
    total_bytes: int
    elapsed_s: float
    throughput_gib_s: float


def _run_serialize_only(views: Sequence[memoryview], chunk_bytes: int) -> BenchResult:
    total_bytes = sum(int(view.nbytes) for view in views)
    started = time.monotonic()
    consumed = 0
    for view in views:
        cursor = 0
        while cursor < view.nbytes:
            chunk = view[cursor : cursor + chunk_bytes]
            payload = chunk.tobytes()
            consumed += len(payload)
            cursor += len(chunk)
    elapsed_s = max(1e-9, time.monotonic() - started)
    if consumed != total_bytes:
        raise RuntimeError(f"serialize consumed mismatch: {consumed} != {total_bytes}")
    return BenchResult(
        mode="serialize_only",
        chunk_bytes=int(chunk_bytes),
        total_bytes=int(total_bytes),
        elapsed_s=float(elapsed_s),
        throughput_gib_s=float(total_bytes) / float(1024**3) / float(elapsed_s),
    )


def _run_serialize_and_copy(
    views: Sequence[memoryview],
    *,
    chunk_bytes: int,
    sink_bytes: int,
) -> BenchResult:
    total_bytes = sum(int(view.nbytes) for view in views)
    sink_len = max(1, int(sink_bytes))
    sink = bytearray(sink_len)
    sink_view = memoryview(sink)
    started = time.monotonic()
    consumed = 0
    global_offset = 0
    for view in views:
        cursor = 0
        while cursor < view.nbytes:
            chunk = view[cursor : cursor + chunk_bytes]
            payload = chunk.tobytes()
            payload_len = len(payload)
            dst_offset = global_offset % sink_len
            first = min(payload_len, sink_len - dst_offset)
            sink_view[dst_offset : dst_offset + first] = payload[:first]
            remaining = payload_len - first
            if remaining > 0:
                sink_view[:remaining] = payload[first:]
            consumed += payload_len
            global_offset += payload_len
            cursor += len(chunk)
    elapsed_s = max(1e-9, time.monotonic() - started)
    if consumed != total_bytes:
        raise RuntimeError(f"copy consumed mismatch: {consumed} != {total_bytes}")
    return BenchResult(
        mode="serialize_and_copy",
        chunk_bytes=int(chunk_bytes),
        total_bytes=int(total_bytes),
        elapsed_s=float(elapsed_s),
        throughput_gib_s=float(total_bytes) / float(1024**3) / float(elapsed_s),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Micro benchmark for stable_dram CPU-stream publish path. "
            "Measures Python chunk serialization overhead and memcpy-like copy path."
        )
    )
    parser.add_argument("--tp-world-size", type=int, default=8)
    parser.add_argument("--total-bytes", type=int, default=8 * 1024**3)
    parser.add_argument(
        "--chunk-bytes",
        default="4128768,16777216,33554432",
        help="Comma-separated chunk sizes in bytes.",
    )
    parser.add_argument(
        "--sink-bytes",
        type=int,
        default=1024**3,
        help="Size of rolling destination buffer for copy simulation.",
    )
    args = parser.parse_args()

    tp_world_size = int(args.tp_world_size)
    if tp_world_size <= 0:
        raise ValueError("tp-world-size must be > 0")
    total_bytes = _normalize_total_bytes(int(args.total_bytes))
    chunk_sizes = [
        _normalize_chunk_bytes(int(value.strip()))
        for value in str(args.chunk_bytes).split(",")
        if value.strip()
    ]
    if not chunk_sizes:
        raise ValueError("chunk-bytes must include at least one value")

    tensors = _build_tp_ranked_tensors(tp_world_size, total_bytes)
    views = _tensor_views(tensors)
    planned_total_bytes = sum(int(view.nbytes) for view in views)
    payload_gib = float(planned_total_bytes) / float(1024**3)

    print(
        json.dumps(
            {
                "tp_world_size": tp_world_size,
                "planned_total_bytes": int(planned_total_bytes),
                "planned_total_gib": payload_gib,
                "chunks": chunk_sizes,
                "sink_bytes": int(args.sink_bytes),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    results: list[BenchResult] = []
    for chunk_bytes in chunk_sizes:
        results.append(_run_serialize_only(views, chunk_bytes))
        results.append(
            _run_serialize_and_copy(
                views,
                chunk_bytes=chunk_bytes,
                sink_bytes=int(args.sink_bytes),
            )
        )

    output = {
        "meta": {
            "tp_world_size": tp_world_size,
            "planned_total_bytes": int(planned_total_bytes),
            "planned_total_gib": payload_gib,
            "sink_bytes": int(args.sink_bytes),
        },
        "results": [asdict(result) for result in results],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
