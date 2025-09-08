#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from tensorcast._C import build_canonical_index_from_safetensors

from ._config import DEFAULT_ALIGN
from ._errors import IndexParseError

TensorMetaIndex = dict[str, tuple[list[int], list[int], str, int]]
TensorDataIndex = dict[str, tuple[int, int]]
TensorDeviceOffsets = dict[int | torch.device, dict[str, int]]
CopyChunk = tuple[int, int, int, int]


def calculate_tensor_device_offsets(
    tensor_index: dict[str, tuple[int, int]],
    device_id: int | torch.device = 0,
):
    tensor_device_offsets: dict[int | torch.device, dict[str, int]] = {device_id: {}}
    tensor_copy_chunks: dict[int | torch.device, list[tuple[int, int, int, int]]] = {
        device_id: []
    }

    current_offset: int = 0
    ALIGN: int = DEFAULT_ALIGN
    seen: dict[tuple[int, int], int] = {}

    for tensor_name in sorted(tensor_index.keys()):
        src_offset, size = tensor_index[tensor_name]
        if (src_offset, size) in seen:
            dst_offset = seen[(src_offset, size)]
        else:
            if current_offset % ALIGN:
                current_offset = (current_offset + (ALIGN - 1)) // ALIGN * ALIGN
            dst_offset = current_offset
            seen[(src_offset, size)] = dst_offset
            tensor_copy_chunks[device_id].append((src_offset, size, dst_offset, 0))
            current_offset += size

        tensor_device_offsets[device_id][tensor_name] = dst_offset

    return tensor_device_offsets, tensor_copy_chunks


def build_indices_from_safetensors(
    artifact_dir: os.PathLike | Path,
) -> tuple[TensorMetaIndex, TensorDataIndex]:
    artifact_dir = Path(str(artifact_dir))
    index_bytes = build_canonical_index_from_safetensors(str(artifact_dir))
    try:
        index_obj = json.loads(index_bytes)
    except Exception as e:  # noqa: BLE001
        raise IndexParseError(f"Failed to parse canonical index bytes: {e}") from e

    tensor_meta_index: TensorMetaIndex = {}
    tensor_data_index: TensorDataIndex = {}
    for name, meta in index_obj.items():
        offset, size, shape, stride, dtype, storage_offset = meta
        tensor_meta_index[name] = (
            list(shape),
            list(stride),
            str(dtype),
            int(storage_offset),
        )
        tensor_data_index[name] = (int(offset), int(size))
    return tensor_meta_index, tensor_data_index


def load_tensor_indices_from_dir(
    artifact_dir: Path,
) -> tuple[TensorMetaIndex, TensorDataIndex]:
    index_path = artifact_dir / "tensor_index.json"
    safetensors_files: list[Path] = sorted(artifact_dir.glob("*.safetensors"))

    if safetensors_files and not index_path.exists():
        return build_indices_from_safetensors(artifact_dir)

    with open(index_path, "r") as f:
        tensor_index = json.load(f)

    tensor_meta_index: TensorMetaIndex = {}
    tensor_data_index: TensorDataIndex = {}
    for name, meta in tensor_index.items():
        if len(meta) == 5:
            offset, size, shape, stride, dtype = meta
            storage_offset = 0
        else:
            offset, size, shape, stride, dtype, storage_offset = meta

        tensor_meta_index[name] = (shape, stride, dtype, storage_offset)
        tensor_data_index[name] = (offset, size)

    return tensor_meta_index, tensor_data_index


def build_v2_index_bytes(
    tensor_meta_index: TensorMetaIndex,
    tensor_source_index: TensorDataIndex,
    tensor_device_offsets: TensorDeviceOffsets,
    device_id: int,
) -> bytes:
    tensor_index_v2: dict[str, tuple[int, int, list[int], list[int], str, int]] = {}
    for name in sorted(tensor_meta_index.keys()):
        shape, stride, dtype, storage_offset = tensor_meta_index[name]
        _, storage_size = tensor_source_index[name]
        dst_off = int(tensor_device_offsets[device_id][name])
        tensor_index_v2[name] = (
            dst_off,
            int(storage_size),
            list(shape),
            list(stride),
            dtype,
            int(storage_offset),
        )
    return json.dumps(tensor_index_v2, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
