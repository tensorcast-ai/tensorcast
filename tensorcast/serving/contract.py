#  Copyright (c) 2026, TensorCast Team.

"""Canonical serving runtime identity and hash helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from tensorcast.api.store.serving_builder import (
    _hash_versioned_payload_to_multihash,
    _normalize_logical_topology_payload,
    compute_serving_tensor_schema_hash,
)
from tensorcast.api.store.types import CanonicalIndex, CanonicalIndexEntry
from tensorcast.types import ServingBindingMemberRef, ServingTopologyRef


@dataclass(frozen=True)
class RuntimeTensorSchemaEntry:
    name: str
    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    element_size: int
    storage_offset: int


def collect_runtime_tensor_schema(
    tensors: Mapping[str, torch.Tensor],
    *,
    remove_duplicate: bool,
) -> tuple[RuntimeTensorSchemaEntry, ...]:
    schema: list[RuntimeTensorSchemaEntry] = []
    seen_ptrs: set[int] = set()
    for name, tensor in sorted(tensors.items()):
        data_ptr = int(tensor.data_ptr())
        if remove_duplicate and data_ptr in seen_ptrs:
            continue
        seen_ptrs.add(data_ptr)
        storage_offset = int(tensor.storage_offset())
        if storage_offset != 0:
            raise ValueError(
                "runtime tensor schema hash requires storage_offset == 0: "
                f"{name} has storage_offset={storage_offset}"
            )
        schema.append(
            RuntimeTensorSchemaEntry(
                name=str(name),
                dtype=str(tensor.dtype),
                shape=tuple(int(dim) for dim in tensor.shape),
                stride=tuple(int(dim) for dim in tensor.stride()),
                element_size=int(tensor.element_size()),
                storage_offset=storage_offset,
            )
        )
    return tuple(schema)


def compute_runtime_tensor_schema_hash(
    schema: Sequence[RuntimeTensorSchemaEntry],
) -> str:
    entries: list[CanonicalIndexEntry] = []
    segment_offset = 0
    for entry in sorted(schema, key=lambda item: item.name):
        if int(entry.storage_offset) != 0:
            raise ValueError(
                "runtime tensor schema hash requires storage_offset == 0: "
                f"{entry.name} has storage_offset={entry.storage_offset}"
            )
        size_bytes = _schema_entry_size_bytes(entry)
        entries.append(
            CanonicalIndexEntry(
                name=entry.name,
                dtype=_torch_dtype_from_name(entry.dtype),
                shape=entry.shape,
                stride=entry.stride,
                storage_offset=0,
                segment_offset=segment_offset,
                size_bytes=size_bytes,
            )
        )
        segment_offset += size_bytes
    return compute_serving_tensor_schema_hash(
        CanonicalIndex(
            entries=tuple(entries),
            total_size_bytes=segment_offset,
            avbs_hash="",
        )
    )


def logical_topology_json(
    topology_ref: ServingTopologyRef,
    *,
    framework_payload: Mapping[str, object],
) -> str:
    del topology_ref
    normalized = _normalize_logical_topology_payload(
        json.dumps(
            dict(framework_payload),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    if normalized is None:
        raise ValueError("framework_payload must define a logical topology")
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def compute_runtime_representation_contract_hash(
    *,
    tensor_schema_hash: str,
    topology_ref: ServingTopologyRef,
    member_ref: ServingBindingMemberRef,
    framework_name: str,
    framework_version: str,
    adapter_version: str,
    serving_abi_version: str,
    source_identity: Mapping[str, object],
) -> str:
    if not tensor_schema_hash:
        raise ValueError("tensor_schema_hash must not be empty")
    payload = {
        "framework": {
            "name": str(framework_name),
            "version": str(framework_version),
            "adapter_version": str(adapter_version),
            "serving_abi_version": str(serving_abi_version),
        },
        "topology_ref": _stable_payload(topology_ref.model_dump(mode="python")),
        "member_ref": _stable_payload(member_ref.model_dump(mode="python")),
        "source_identity": _stable_payload(dict(source_identity)),
        "tensor_schema_hash": str(tensor_schema_hash),
    }
    return _hash_versioned_payload_to_multihash(
        "tensorcast.representation.runtime_contract.v1",
        payload,
    )


def _schema_entry_size_bytes(entry: RuntimeTensorSchemaEntry) -> int:
    elements = 1
    for dim in entry.shape:
        elements *= int(dim)
    return int(elements * entry.element_size)


def _torch_dtype_from_name(dtype_name: str) -> torch.dtype:
    normalized = dtype_name.removeprefix("torch.")
    dtype = getattr(torch, normalized, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported runtime tensor dtype: {dtype_name}")
    return dtype


def _stable_payload(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _stable_payload(value[key])
            for key in sorted(value, key=lambda item: str(item))
            if value[key] is not None
        }
    if isinstance(value, (list, tuple)):
        return [_stable_payload(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "RuntimeTensorSchemaEntry",
    "collect_runtime_tensor_schema",
    "compute_runtime_representation_contract_hash",
    "compute_runtime_tensor_schema_hash",
    "logical_topology_json",
]
