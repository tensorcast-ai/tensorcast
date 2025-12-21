#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import torch

from tensorcast._c_ext import collect_tensor_storage_graph
from tensorcast.api._indices import TensorDataIndex, TensorMetaIndex


@dataclass(frozen=True)
class StorageEntry:
    storage_id: str
    device_id: int
    base_ptr: int
    size_bytes: int


@dataclass(frozen=True)
class TensorAlias:
    name: str
    storage_id: str
    storage_offset: int
    logical_length: int
    shape: list[int]
    stride: list[int]
    dtype: str


@dataclass(frozen=True)
class TensorStorageGraph:
    storages: Dict[str, StorageEntry]
    aliases: Dict[str, TensorAlias]
    tensor_meta_index: TensorMetaIndex
    tensor_source_index: TensorDataIndex


def build_tensor_storage_graph(
    tensors: Mapping[str, torch.Tensor],
) -> TensorStorageGraph:
    tensor_dict = dict(tensors)
    storages: Dict[str, StorageEntry] = {}
    aliases: Dict[str, TensorAlias] = {}
    tensor_meta_index: TensorMetaIndex = {}
    tensor_source_index: TensorDataIndex = {}

    native_graph = collect_tensor_storage_graph(tensor_dict)
    native_storages = native_graph.get("storages", {})
    native_aliases = native_graph.get("aliases", {})
    native_meta = native_graph.get("tensor_meta_index", {})
    native_sources = native_graph.get("tensor_source_index", {})

    for storage_id, payload in native_storages.items():
        device_id = int(payload["device_id"])
        base_ptr = int(payload["base_ptr"])
        size_bytes = int(payload["size_bytes"])
        storages[storage_id] = StorageEntry(
            storage_id=storage_id,
            device_id=device_id,
            base_ptr=base_ptr,
            size_bytes=size_bytes,
        )

    for name, payload in native_aliases.items():
        aliases[name] = TensorAlias(
            name=name,
            storage_id=str(payload["storage_id"]),
            storage_offset=int(payload["storage_offset"]),
            logical_length=int(payload["logical_length"]),
            shape=[int(v) for v in payload["shape"]],
            stride=[int(v) for v in payload["stride"]],
            dtype=str(payload["dtype"]),
        )

    for name, tup in native_meta.items():
        shape, stride, dtype, storage_offset = tup
        tensor_meta_index[name] = (
            [int(v) for v in shape],
            [int(v) for v in stride],
            str(dtype),
            int(storage_offset),
        )

    for name, tup in native_sources.items():
        base_ptr, size_bytes = tup
        tensor_source_index[name] = (int(base_ptr), int(size_bytes))

    return TensorStorageGraph(
        storages=storages,
        aliases=aliases,
        tensor_meta_index=tensor_meta_index,
        tensor_source_index=tensor_source_index,
    )


__all__ = [
    "StorageEntry",
    "TensorAlias",
    "TensorStorageGraph",
    "build_tensor_storage_graph",
]
