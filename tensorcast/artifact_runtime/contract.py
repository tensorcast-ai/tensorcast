#  Copyright (c) 2026, TensorCast Team.

"""Runtime identity, topology, and source-bound contract helpers."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from tensorcast.api.store.types import CanonicalIndex, CanonicalIndexEntry
from tensorcast.types import (
    SERVING_MANIFEST_TENSOR_NAME,
    RuntimeBindingMemberRef,
    RuntimeTopologyRef,
    SourceBoundCapability,
)

MIN_SOURCE_BOUND_CONTRACT_VERSION = 4
SOURCE_BOUND_CONTRACT_PATH_COLLECTIVE_FIRST_V4 = "collective_first_v4"
REQUIRED_SOURCE_BOUND_CAPABILITIES = (
    SourceBoundCapability.FIRST_CLASS_COLLECTIVE_INGRESS,
    SourceBoundCapability.TYPED_EXECUTION_DIAGNOSTICS,
    SourceBoundCapability.SINGLE_MINT_BINDING_CLOSEOUT,
)


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _multibase_multihash_sha256(digest: bytes) -> str:
    if len(digest) != 32:
        raise ValueError("SHA256 digest must be 32 bytes")
    multihash = b"\x12\x20" + digest
    encoded = base64.b32encode(multihash).decode("ascii").lower().rstrip("=")
    return f"b{encoded}"


def hash_versioned_payload_to_multihash(version: str, payload: object) -> str:
    serialized = _canonical_json_bytes(payload)
    versioned_payload = version.encode("utf-8") + b"\n" + serialized
    return _multibase_multihash_sha256(hashlib.sha256(versioned_payload).digest())


def normalize_logical_topology_payload(
    logical_topology_json: str | None,
) -> dict[str, object] | None:
    if logical_topology_json is None:
        return None
    try:
        payload = json.loads(logical_topology_json)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("logical_topology_json must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("logical_topology_json must encode an object")
    family = str(payload.get("family", "")).strip()
    version = str(payload.get("version", "")).strip()
    raw_dimensions = payload.get("dimensions", [])
    if not family:
        raise ValueError("logical_topology_json.family must not be empty")
    if not version:
        raise ValueError("logical_topology_json.version must not be empty")
    if not isinstance(raw_dimensions, list):
        raise ValueError("logical_topology_json.dimensions must be a list")
    dimensions: list[dict[str, int | str]] = []
    for raw_dimension in raw_dimensions:
        if not isinstance(raw_dimension, dict):
            raise ValueError("logical_topology_json.dimensions items must be objects")
        name = str(raw_dimension.get("name", "")).strip()
        if not name:
            raise ValueError("logical_topology_json dimensions require non-empty name")
        size = raw_dimension.get("size", None)
        if not isinstance(size, int) or size <= 0:
            raise ValueError(
                "logical_topology_json dimensions require positive integer size"
            )
        dimensions.append({"name": name, "size": int(size)})
    dimensions.sort(key=lambda item: (str(item["name"]), int(item["size"])))
    return {
        "family": family,
        "version": version,
        "dimensions": dimensions,
    }


@dataclass(frozen=True)
class RuntimeTensorSchemaEntry:
    name: str
    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    element_size: int
    storage_offset: int


@dataclass(frozen=True)
class SourceBoundContractState:
    server_config_present: bool
    source_bound_contract_version: int
    source_bound_capability_flags: int
    source_bound_capability_names: tuple[str, ...]
    source_bound_contract_ready: bool

    @classmethod
    def unavailable(cls) -> SourceBoundContractState:
        return cls(
            server_config_present=False,
            source_bound_contract_version=0,
            source_bound_capability_flags=0,
            source_bound_capability_names=(),
            source_bound_contract_ready=False,
        )

    @classmethod
    def from_server_config(
        cls,
        server_config: Any | None,
    ) -> SourceBoundContractState:
        if server_config is None:
            return cls.unavailable()
        flags = int(getattr(server_config, "source_bound_capability_flags", 0) or 0)
        version = int(getattr(server_config, "source_bound_contract_version", 0) or 0)
        capability_names = tuple(
            str(capability.name)
            for capability in SourceBoundCapability
            if flags & int(capability)
        )
        contract_ready = version >= MIN_SOURCE_BOUND_CONTRACT_VERSION and all(
            flags & int(capability) for capability in REQUIRED_SOURCE_BOUND_CAPABILITIES
        )
        return cls(
            server_config_present=True,
            source_bound_contract_version=version,
            source_bound_capability_flags=flags,
            source_bound_capability_names=capability_names,
            source_bound_contract_ready=contract_ready,
        )


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
    return compute_canonical_runtime_tensor_schema_hash(
        CanonicalIndex(
            entries=tuple(entries),
            total_size_bytes=segment_offset,
            avbs_hash="",
        )
    )


def compute_canonical_runtime_tensor_schema_hash(
    canonical_index: CanonicalIndex,
    *,
    manifest_tensor_name: str = SERVING_MANIFEST_TENSOR_NAME,
) -> str:
    tensors = [
        {
            "name": str(entry.name),
            "dtype": str(entry.dtype),
            "shape": [int(dim) for dim in entry.shape],
            "stride": [int(dim) for dim in entry.stride],
            "element_size": int(entry.dtype.itemsize),
        }
        for entry in sorted(
            (
                entry
                for entry in canonical_index.entries
                if str(entry.name) != str(manifest_tensor_name)
            ),
            key=lambda entry: str(entry.name),
        )
    ]
    return hash_versioned_payload_to_multihash(
        "tensorcast.representation.tensor_schema.v1",
        {"tensors": tensors},
    )


def logical_topology_json(
    topology_ref: RuntimeTopologyRef,
    *,
    framework_payload: Mapping[str, object],
) -> str:
    del topology_ref
    normalized = normalize_logical_topology_payload(
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
    topology_ref: RuntimeTopologyRef,
    member_ref: RuntimeBindingMemberRef,
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
    return hash_versioned_payload_to_multihash(
        "tensorcast.representation.runtime_contract.v1",
        payload,
    )


def read_source_bound_contract_state(
    *,
    store_fn: Callable[[], Any] | None = None,
) -> SourceBoundContractState:
    try:
        if store_fn is None:
            import tensorcast as tc

            store_fn = tc.store
        store = store_fn()
        capabilities = store.capabilities
        server_config = getattr(capabilities, "server_config", None)
    except Exception:
        return SourceBoundContractState.unavailable()
    return SourceBoundContractState.from_server_config(server_config)


def source_bound_contract_profile_fields(
    state: SourceBoundContractState,
    path: str,
) -> dict[str, object]:
    return {
        "source_bound_contract_version": int(state.source_bound_contract_version),
        "source_bound_capability_flags": list(state.source_bound_capability_names),
        "source_bound_contract_ready": bool(state.source_bound_contract_ready),
        "source_bound_contract_path": path,
    }


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
    "MIN_SOURCE_BOUND_CONTRACT_VERSION",
    "REQUIRED_SOURCE_BOUND_CAPABILITIES",
    "RuntimeTensorSchemaEntry",
    "SOURCE_BOUND_CONTRACT_PATH_COLLECTIVE_FIRST_V4",
    "SourceBoundContractState",
    "collect_runtime_tensor_schema",
    "compute_runtime_representation_contract_hash",
    "compute_runtime_tensor_schema_hash",
    "logical_topology_json",
    "read_source_bound_contract_state",
    "source_bound_contract_profile_fields",
]
