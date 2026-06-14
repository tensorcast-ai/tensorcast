#!/usr/bin/env python3
#  Copyright (c) 2026, TensorCast Team.

"""Feasibility checks for TP source-window collective materialization.

This script is intentionally independent from the TensorCast daemon. It uses
real safetensors headers and Qwen/vLLM TP naming conventions to answer two
questions before changing the production loader:

1. Does a source-window collective plan remove the current local-mapped read
   amplification on the target models?
2. Can rank-striped source reads reach InstantTensor-class read throughput
   before adding TensorCast scatter and NCCL work?

The default mode is metadata-only. Use --io-smoke to actually read files.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import multiprocessing as mp
import os
import struct
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


POSIX_FADV_DONTNEED = 4


DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
}


@dataclass(frozen=True)
class TensorEntry:
    name: str
    file_name: str
    file_path: str
    file_offset: int
    payload_offset: int
    nbytes: int
    dtype: str
    shape: list[int]


@dataclass(frozen=True)
class FileSegment:
    file_name: str
    file_path: str
    data_start: int
    payload_bytes: int
    file_size: int


@dataclass
class CategoryStats:
    tensor_count: int = 0
    source_bytes: int = 0
    current_read_bytes_per_rank: int = 0
    target_bytes_per_rank: int = 0
    source_window_disk_bytes_per_rank: int = 0


def _read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path} is too small to be a safetensors file")
        header_len = struct.unpack("<Q", raw)[0]
        header_raw = f.read(header_len)
        if len(header_raw) != header_len:
            raise ValueError(f"{path} has a truncated safetensors header")
    header = json.loads(header_raw)
    if not isinstance(header, dict):
        raise ValueError(f"{path} safetensors header is not a JSON object")
    return 8 + header_len, header


def read_model_safetensors(model_dir: Path) -> tuple[list[FileSegment], list[TensorEntry]]:
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files found under {model_dir}")

    segments: list[FileSegment] = []
    tensors: list[TensorEntry] = []
    for path in files:
        data_start, header = _read_safetensors_header(path)
        file_size = path.stat().st_size
        if data_start > file_size:
            raise ValueError(f"{path} data_start is beyond EOF")
        payload_bytes = file_size - data_start
        segments.append(
            FileSegment(
                file_name=path.name,
                file_path=str(path),
                data_start=data_start,
                payload_bytes=payload_bytes,
                file_size=file_size,
            )
        )
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(meta, dict):
                raise ValueError(f"{path}:{name} metadata is not an object")
            dtype = str(meta["dtype"])
            shape = [int(v) for v in meta["shape"]]
            begin, end = [int(v) for v in meta["data_offsets"]]
            if begin < 0 or end < begin or end > payload_bytes:
                raise ValueError(f"{path}:{name} has invalid data_offsets")
            nbytes = end - begin
            elem_size = DTYPE_BYTES.get(dtype)
            if elem_size is not None:
                expected = elem_size * math.prod(shape)
                if expected != nbytes:
                    raise ValueError(
                        f"{path}:{name} size mismatch: offsets={nbytes} "
                        f"shape*dtype={expected}"
                    )
            tensors.append(
                TensorEntry(
                    name=name,
                    file_name=path.name,
                    file_path=str(path),
                    file_offset=data_start + begin,
                    payload_offset=begin,
                    nbytes=nbytes,
                    dtype=dtype,
                    shape=shape,
                )
            )
    return segments, tensors


def classify_qwen_vllm_tp(entry: TensorEntry) -> str:
    """Classify how vLLM TP typically consumes Qwen dense/MoE weights.

    The categories are deliberately loader-facing, not model-facing:
    - dim1_sharded: current local mapped path reads full rows for a rank's
      column slice.
    - dim0_sharded: rank slice is source-contiguous enough to avoid read
      amplification.
    - replicated: every TP rank consumes the whole tensor.
    """

    name = entry.name
    if name.endswith(".down_proj.weight") or name.endswith(".o_proj.weight"):
        return "dim1_sharded"
    if name.endswith(
        (
            ".q_proj.weight",
            ".k_proj.weight",
            ".v_proj.weight",
            ".gate_proj.weight",
            ".up_proj.weight",
        )
    ):
        return "dim0_sharded"
    if name in ("lm_head.weight", "model.embed_tokens.weight"):
        return "dim0_sharded"
    return "replicated"


def _max_evenish_shard_bytes(entry: TensorEntry, axis: int, tp: int) -> int:
    if not entry.shape:
        return entry.nbytes
    if axis >= len(entry.shape):
        raise ValueError(f"{entry.name} has no axis {axis} in shape {entry.shape}")
    dim = entry.shape[axis]
    if dim <= 0:
        raise ValueError(f"{entry.name} has invalid sharded dimension {dim}")
    max_shard = (dim + tp - 1) // tp
    return math.ceil(entry.nbytes * max_shard / dim)


def build_feasibility_summary(
    segments: list[FileSegment],
    tensors: list[TensorEntry],
    *,
    tp: int,
    window_bytes: int,
    current_profile_read_bytes_per_rank: int | None,
    current_profile_dst_bytes_per_rank: int | None,
) -> dict[str, Any]:
    categories: dict[str, CategoryStats] = {
        "dim0_sharded": CategoryStats(),
        "dim1_sharded": CategoryStats(),
        "replicated": CategoryStats(),
    }
    total_source_bytes = 0
    current_read_bytes_per_rank = 0
    target_bytes_per_rank = 0

    for entry in tensors:
        category = classify_qwen_vllm_tp(entry)
        stats = categories[category]
        stats.tensor_count += 1
        stats.source_bytes += entry.nbytes
        total_source_bytes += entry.nbytes

        if category == "dim1_sharded":
            current_read = entry.nbytes
            target = _max_evenish_shard_bytes(entry, axis=1, tp=tp)
        elif category == "dim0_sharded":
            current_read = _max_evenish_shard_bytes(entry, axis=0, tp=tp)
            target = current_read
        else:
            current_read = entry.nbytes
            target = entry.nbytes

        stats.current_read_bytes_per_rank += current_read
        stats.target_bytes_per_rank += target
        current_read_bytes_per_rank += current_read
        target_bytes_per_rank += target

    source_window_disk_bytes_per_rank = math.ceil(total_source_bytes / tp)
    for stats in categories.values():
        stats.source_window_disk_bytes_per_rank = math.ceil(stats.source_bytes / tp)

    file_payload_bytes = sum(seg.payload_bytes for seg in segments)
    file_bytes = sum(seg.file_size for seg in segments)
    window_count = sum((seg.payload_bytes + window_bytes - 1) // window_bytes for seg in segments)

    current_read_amp = current_read_bytes_per_rank / max(1, target_bytes_per_rank)
    source_window_read_amp_vs_source = file_payload_bytes / max(1, total_source_bytes)
    source_window_read_vs_current = source_window_disk_bytes_per_rank / max(1, current_read_bytes_per_rank)
    source_window_disk_vs_target = source_window_disk_bytes_per_rank / max(1, target_bytes_per_rank)

    profile_alignment: dict[str, Any] = {}
    if current_profile_read_bytes_per_rank is not None:
        profile_alignment["read_bytes_per_rank_ratio_estimate_to_profile"] = (
            current_read_bytes_per_rank / current_profile_read_bytes_per_rank
        )
        profile_alignment["read_bytes_per_rank_delta"] = (
            current_read_bytes_per_rank - current_profile_read_bytes_per_rank
        )
    if current_profile_dst_bytes_per_rank is not None:
        profile_alignment["dst_bytes_per_rank_ratio_estimate_to_profile"] = (
            target_bytes_per_rank / current_profile_dst_bytes_per_rank
        )
        profile_alignment["dst_bytes_per_rank_delta"] = (
            target_bytes_per_rank - current_profile_dst_bytes_per_rank
        )

    return {
        "tp": tp,
        "tensor_count": len(tensors),
        "file_count": len(segments),
        "file_bytes": file_bytes,
        "file_payload_bytes": file_payload_bytes,
        "tensor_source_bytes": total_source_bytes,
        "window_bytes": window_bytes,
        "window_count": window_count,
        "current_local_mapped_estimate": {
            "read_bytes_per_rank": current_read_bytes_per_rank,
            "target_bytes_per_rank": target_bytes_per_rank,
            "read_amplification": current_read_amp,
        },
        "source_window_collective_estimate": {
            "disk_read_bytes_per_rank": source_window_disk_bytes_per_rank,
            "group_disk_read_bytes": file_payload_bytes,
            "read_amplification_vs_source_payload": source_window_read_amp_vs_source,
            "disk_read_vs_current_local_mapped": source_window_read_vs_current,
            "disk_read_vs_target_bytes_per_rank": source_window_disk_vs_target,
            "estimated_read_saving_bytes_per_rank": (
                current_read_bytes_per_rank - source_window_disk_bytes_per_rank
            ),
        },
        "categories": {name: asdict(stats) for name, stats in categories.items()},
        "profile_alignment": profile_alignment,
    }


def _window_stripes_for_rank(
    segments: list[FileSegment],
    *,
    tp: int,
    rank: int,
    window_bytes: int,
    max_bytes_per_rank: int,
) -> list[tuple[str, int, int]]:
    stripes: list[tuple[str, int, int]] = []
    planned = 0
    for seg in segments:
        cursor = 0
        while cursor < seg.payload_bytes:
            this_window = min(window_bytes, seg.payload_bytes - cursor)
            begin = (this_window * rank) // tp
            end = (this_window * (rank + 1)) // tp
            size = end - begin
            if size > 0:
                if max_bytes_per_rank > 0 and planned + size > max_bytes_per_rank:
                    size = max(0, max_bytes_per_rank - planned)
                if size > 0:
                    stripes.append((seg.file_path, seg.data_start + cursor + begin, size))
                    planned += size
                if max_bytes_per_rank > 0 and planned >= max_bytes_per_rank:
                    return stripes
            cursor += this_window
    return stripes


def fadvise_dontneed(segments: list[FileSegment]) -> list[dict[str, Any]]:
    libc = ctypes.CDLL(None, use_errno=True)
    posix_fadvise = libc.posix_fadvise
    posix_fadvise.argtypes = [ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_int]
    posix_fadvise.restype = ctypes.c_int

    results: list[dict[str, Any]] = []
    for seg in segments:
        fd = os.open(seg.file_path, os.O_RDONLY)
        try:
            rc = posix_fadvise(
                fd,
                ctypes.c_longlong(0),
                ctypes.c_longlong(0),
                POSIX_FADV_DONTNEED,
            )
            results.append(
                {
                    "file_name": seg.file_name,
                    "ok": rc == 0,
                    "returncode": rc,
                    "errno": ctypes.get_errno() if rc != 0 else 0,
                }
            )
        finally:
            os.close(fd)
    return results


def _io_worker(
    rank: int,
    stripes: list[tuple[str, int, int]],
    chunk_bytes: int,
    start_event: mp.synchronize.Event,
    result_queue: mp.Queue,
) -> None:
    try:
        buffer = bytearray(chunk_bytes)
        view = memoryview(buffer)
        start_event.wait()
        started = time.perf_counter()
        read_bytes = 0
        read_calls = 0
        fd_cache: dict[str, int] = {}
        try:
            for path, offset, size in stripes:
                fd = fd_cache.get(path)
                if fd is None:
                    fd = os.open(path, os.O_RDONLY)
                    fd_cache[path] = fd
                remaining = size
                cursor = offset
                while remaining > 0:
                    todo = min(chunk_bytes, remaining)
                    got = os.preadv(fd, [view[:todo]], cursor)
                    if got <= 0:
                        raise OSError(f"short read rank={rank} path={path} offset={cursor}")
                    cursor += got
                    remaining -= got
                    read_bytes += got
                    read_calls += 1
        finally:
            for fd in fd_cache.values():
                os.close(fd)
        elapsed = time.perf_counter() - started
        result_queue.put(
            {
                "rank": rank,
                "ok": True,
                "elapsed_sec": elapsed,
                "read_bytes": read_bytes,
                "read_calls": read_calls,
                "throughput_gb_s": read_bytes / elapsed / 1e9 if elapsed > 0 else 0.0,
            }
        )
    except BaseException as exc:  # noqa: BLE001
        result_queue.put(
            {
                "rank": rank,
                "ok": False,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
        )


def run_io_smoke(
    segments: list[FileSegment],
    *,
    tp: int,
    window_bytes: int,
    chunk_bytes: int,
    max_bytes_per_rank: int,
) -> dict[str, Any]:
    stripes_by_rank = [
        _window_stripes_for_rank(
            segments,
            tp=tp,
            rank=rank,
            window_bytes=window_bytes,
            max_bytes_per_rank=max_bytes_per_rank,
        )
        for rank in range(tp)
    ]
    planned_bytes_by_rank = [sum(size for _, _, size in stripes) for stripes in stripes_by_rank]

    ctx = mp.get_context("spawn")
    start_event = ctx.Event()
    result_queue: mp.Queue = ctx.Queue()
    procs = [
        ctx.Process(
            target=_io_worker,
            args=(rank, stripes_by_rank[rank], chunk_bytes, start_event, result_queue),
        )
        for rank in range(tp)
    ]
    for proc in procs:
        proc.start()
    started = time.perf_counter()
    start_event.set()
    results = [result_queue.get() for _ in range(tp)]
    for proc in procs:
        proc.join()
    elapsed = time.perf_counter() - started

    ok = all(bool(item.get("ok")) for item in results)
    total_read = sum(int(item.get("read_bytes", 0)) for item in results)
    max_rank_elapsed = max((float(item.get("elapsed_sec", 0.0)) for item in results), default=0.0)
    return {
        "ok": ok,
        "tp": tp,
        "window_bytes": window_bytes,
        "chunk_bytes": chunk_bytes,
        "max_bytes_per_rank": max_bytes_per_rank,
        "planned_bytes_by_rank": planned_bytes_by_rank,
        "total_planned_bytes": sum(planned_bytes_by_rank),
        "elapsed_wall_sec": elapsed,
        "max_rank_elapsed_sec": max_rank_elapsed,
        "total_read_bytes": total_read,
        "aggregate_throughput_gb_s_by_wall": total_read / elapsed / 1e9 if elapsed > 0 else 0.0,
        "aggregate_throughput_gb_s_by_max_rank": (
            total_read / max_rank_elapsed / 1e9 if max_rank_elapsed > 0 else 0.0
        ),
        "rank_results": sorted(results, key=lambda item: int(item["rank"])),
    }


def _parse_size(raw: str) -> int:
    text = raw.strip().lower()
    scale = 1
    for suffix, value in (
        ("gib", 1024**3),
        ("gb", 1000**3),
        ("mib", 1024**2),
        ("mb", 1000**2),
        ("kib", 1024),
        ("kb", 1000),
    ):
        if text.endswith(suffix):
            scale = value
            text = text[: -len(suffix)]
            break
    return int(float(text) * scale)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--window-bytes", type=_parse_size, default=512 * 1024 * 1024)
    parser.add_argument("--current-profile-read-bytes-per-rank", type=int, default=None)
    parser.add_argument("--current-profile-dst-bytes-per-rank", type=int, default=None)
    parser.add_argument("--io-smoke", action="store_true")
    parser.add_argument(
        "--fadvise-dontneed",
        action="store_true",
        help="Call posix_fadvise(DONTNEED) on safetensors files before --io-smoke.",
    )
    parser.add_argument("--io-chunk-bytes", type=_parse_size, default=16 * 1024 * 1024)
    parser.add_argument(
        "--io-max-bytes-per-rank",
        type=_parse_size,
        default=0,
        help="Cap IO smoke bytes per rank. 0 means read the whole source-window stripe.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--pretty", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    segments, tensors = read_model_safetensors(args.model_dir)
    result = {
        "model_dir": str(args.model_dir),
        "feasibility": build_feasibility_summary(
            segments,
            tensors,
            tp=args.tp,
            window_bytes=args.window_bytes,
            current_profile_read_bytes_per_rank=args.current_profile_read_bytes_per_rank,
            current_profile_dst_bytes_per_rank=args.current_profile_dst_bytes_per_rank,
        ),
    }
    if args.io_smoke:
        if args.fadvise_dontneed:
            result["fadvise_dontneed"] = fadvise_dontneed(segments)
        result["io_smoke"] = run_io_smoke(
            segments,
            tp=args.tp,
            window_bytes=args.window_bytes,
            chunk_bytes=args.io_chunk_bytes,
            max_bytes_per_rank=args.io_max_bytes_per_rank,
        )

    text = json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
