#  Copyright (c) 2025, StepCast Team.

"""
Content-addressing utilities for RFC-0007 (mi2: index/data multihash).

This module centralizes Python-side helpers to:
- Build canonical index bytes and compute index multihash
- Compute data multihash over a linearized stream of file segments
- Collect standard partition and safetensors segments
- Persist a model descriptor JSON alongside the index

Keeping these helpers separate avoids bloating `scstore/torch_util.py`.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Iterable


def _base32_lower_nopad(data: bytes) -> str:
    """RFC4648 base32 in lowercase without '=' padding."""
    return base64.b32encode(data).decode("ascii").rstrip("=").lower()


def _to_multibase_multihash_sha256(digest: bytes) -> str:
    """Wrap raw sha256 digest into multihash (sha2-256) + multibase (base32-lower).

    Returns: 'b' + base32_lower(multihash_bytes)
    where multihash_bytes = 0x12 | 0x20 | digest(32B)
    """
    mh = bytes([0x12, 0x20]) + digest
    return "b" + _base32_lower_nopad(mh)


def _sha256_bytes(data: bytes) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(data)
    return hasher.digest()


def _compute_tree_hash_sha256(leaves: list[bytes]) -> bytes:
    """Compute Merkle tree (sha256) root given leaf digests (sha256)."""
    if not leaves:
        return bytes(32)
    level = list(leaves)
    while len(level) > 1:
        next_level: list[bytes] = []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                concat = level[i] + level[i + 1]
                next_level.append(_sha256_bytes(concat))
            else:
                next_level.append(level[i])
        level = next_level
    return level[0]


def compute_index_multihash_from_index_bytes(index_bytes: bytes) -> str:
    """Compute index multihash from canonical index bytes.

    Multihash is sha2-256 digest, encoded via multibase base32 (lowercase).
    """
    return _to_multibase_multihash_sha256(_sha256_bytes(index_bytes))


def collect_partition_segments_for_stream(
    model_dir: str | Path, actual_size: int
) -> list[tuple[Path, int, int]]:
    """Collect (path, start, length) segments for standard tensor.data* files.

    Segments are ordered by filename and truncated to 'actual_size'.
    """
    d = Path(str(model_dir))
    multipart = sorted(d.glob("tensor.data_*"))
    segments: list[tuple[Path, int, int]] = []
    remaining = int(actual_size)
    if multipart:
        for p in multipart:
            if remaining <= 0:
                break
            file_size = p.stat().st_size
            take = min(file_size, remaining)
            segments.append((p, 0, int(take)))
            remaining -= take
    else:
        single = d / "tensor.data"
        if single.exists():
            file_size = single.stat().st_size
            take = min(file_size, remaining)
            if take > 0:
                segments.append((single, 0, int(take)))
    return segments


def collect_safetensors_segments_for_stream(
    model_dir: str | Path,
) -> list[tuple[Path, int, int]]:
    """Collect (path, start, length) segments for .safetensors payloads."""
    d = Path(str(model_dir))
    safes = sorted(d.glob("*.safetensors"))
    segments: list[tuple[Path, int, int]] = []
    for p in safes:
        with open(p, "rb") as f:
            header_len_le = f.read(8)
            if len(header_len_le) != 8:
                raise ValueError(f"Invalid safetensors file: {p}")
            header_len = int.from_bytes(header_len_le, byteorder="little", signed=False)
            data_start = 8 + header_len
            file_size = p.stat().st_size
            if data_start > file_size:
                raise ValueError(f"Invalid safetensors layout in {p}")
            segments.append((p, data_start, file_size - data_start))
    return segments


def compute_data_multihash_from_segments(
    segments: Iterable[tuple[Path, int, int]],
    chunk_size: int = 4 * 1024 * 1024,
) -> str:
    """Compute data multihash over a linearized stream represented by file segments.

    We read the concatenated stream in fixed-size leaves (default 4MiB),
    compute sha256 for each leaf, reduce via Merkle tree, then wrap as multibase
    over multihash (sha2-256).
    """
    leaf_digests: list[bytes] = []
    buf = bytearray()
    for path, start, length in segments:
        remain = int(length)
        with open(path, "rb", buffering=0) as f:
            f.seek(start)
            while remain > 0:
                to_read = min(remain, chunk_size - len(buf))
                chunk = f.read(to_read)
                if not chunk:
                    break
                buf.extend(chunk)
                remain -= len(chunk)
                if len(buf) == chunk_size:
                    leaf_digests.append(_sha256_bytes(bytes(buf)))
                    buf.clear()
    if buf:
        leaf_digests.append(_sha256_bytes(bytes(buf)))
    root = _compute_tree_hash_sha256(leaf_digests)
    return _to_multibase_multihash_sha256(root)


def write_model_descriptor(model_dir: str | Path, descriptor: dict) -> None:
    path = Path(str(model_dir)) / "model_descriptor.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(descriptor, f, indent=2, ensure_ascii=False)


# --- Helpers for building canonical index bytes from on-disk safetensors ---


def _torch_dtype_from_safetensors(dtype: str) -> str:
    mapping = {
        "F16": "torch.float16",
        "BF16": "torch.bfloat16",
        "F32": "torch.float32",
        "F64": "torch.float64",
        "I8": "torch.int8",
        "I16": "torch.int16",
        "I32": "torch.int32",
        "I64": "torch.int64",
        "U8": "torch.uint8",
        "BOOL": "torch.uint8",
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported safetensors dtype: {dtype}")
    return mapping[dtype]


def _row_major_stride(shape: list[int]) -> list[int]:
    if not shape:
        return []
    stride = [0] * len(shape)
    acc = 1
    for i in range(len(shape) - 1, -1, -1):
        stride[i] = acc
        acc *= int(shape[i])
    return stride


def _build_canonical_index_bytes_from_safetensors(
    model_dir: str | Path,
) -> tuple[bytes, int]:
    d = Path(str(model_dir))
    files = sorted(d.glob("*.safetensors"))
    if not files:
        raise ValueError(f"No .safetensors files under {d}")

    # name -> [offset, size, shape, stride, dtype, storage_offset]
    entries: dict[str, list] = {}
    total_payload_size = 0
    base_offset = 0
    for p in files:
        with open(p, "rb") as f:
            header_len_le = f.read(8)
            if len(header_len_le) != 8:
                raise ValueError(f"Invalid safetensors file: {p}")
            header_len = int.from_bytes(header_len_le, byteorder="little", signed=False)
            header = f.read(header_len)
            if len(header) != header_len:
                raise ValueError(f"Truncated safetensors header: {p}")
            h = json.loads(header)
        file_size = p.stat().st_size
        data_start = 8 + header_len
        if data_start > file_size:
            raise ValueError(f"Invalid safetensors layout in {p}")
        for name, meta in h.items():
            if name == "__metadata__":
                continue
            if not isinstance(meta, dict):
                raise ValueError(f"Malformed entry for {name} in {p}")
            begin, end = meta["data_offsets"]
            if end < begin:
                raise ValueError(f"Invalid data_offsets for {name} in {p}")
            shape = list(map(int, meta["shape"]))
            entries[name] = [
                int(base_offset + begin),
                int(end - begin),
                shape,
                _row_major_stride(shape),
                _torch_dtype_from_safetensors(str(meta["dtype"])),
                0,
            ]
        payload_size = file_size - data_start
        base_offset += payload_size
        total_payload_size += payload_size

    # Serialize canonical JSON (sorted keys, compact separators)
    canonical = json.dumps(entries, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return canonical, int(total_payload_size)


def generate_model_id_from_path(model_dir: str | Path) -> dict:
    """Generate RFC-0007 mi2 model_id and descriptor from an existing directory.

    Supports:
    - Standard partition format (tensor.data / tensor.data_*, tensor_index.json)
    - Safetensors directories (*.safetensors)
    """
    d = Path(str(model_dir))
    if not d.exists() or not d.is_dir():
        raise ValueError(f"Invalid model_dir: {d}")

    safes = sorted(d.glob("*.safetensors"))
    if safes:
        index_bytes, total_size = _build_canonical_index_bytes_from_safetensors(d)
        segments = collect_safetensors_segments_for_stream(d)
    else:
        index_path = d / "tensor_index.json"
        if not index_path.exists():
            raise ValueError(f"tensor_index.json not found in {d}")
        # Canonicalize JSON bytes
        with open(index_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        index_bytes = json.dumps(obj, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
        # Compute total_size = max(offset + size)
        total_size = 0
        for meta in obj.values():
            off, sz = int(meta[0]), int(meta[1])
            total_size = max(total_size, off + sz)
        segments = collect_partition_segments_for_stream(d, actual_size=int(total_size))

    index_mh = compute_index_multihash_from_index_bytes(index_bytes)
    data_mh = compute_data_multihash_from_segments(segments)
    model_id = f"mi2:{index_mh}:{data_mh}"
    return {
        "model_id": model_id,
        "index_multihash": index_mh,
        "data_multihash": data_mh,
        "schema_version": "v2",
        "encoding": "json",
        "total_size": int(total_size),
        "hash_params": {
            "chunk_size": 4 * 1024 * 1024,
            "fanout": 2,
            "algorithm": "sha2-256",
        },
    }


__all__ = [
    "compute_index_multihash_from_index_bytes",
    "collect_partition_segments_for_stream",
    "collect_safetensors_segments_for_stream",
    "compute_data_multihash_from_segments",
    "write_model_descriptor",
    "generate_model_id_from_path",
]
