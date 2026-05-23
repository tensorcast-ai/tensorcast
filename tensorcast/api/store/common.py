#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import json
import logging
import time
from typing import Mapping, Sequence

import torch

from tensorcast.api._config import PlanType
from tensorcast.api._register import RegisteredLease, RegistrationResult
from tensorcast.api.store.types import (
    ArtifactError,
    CanonicalIndex,
    CanonicalIndexEntry,
    LeaseHandle,
    ReplicaInfo,
    ReplicaType,
    TensorDict,
)

logger = logging.getLogger(__name__)


def dtype_from_string(value: str) -> torch.dtype:
    canonical = value
    if "." in value:
        _, canonical = value.rsplit(".", 1)
    dtype_obj = getattr(torch, canonical, None)
    if isinstance(dtype_obj, torch.dtype):
        return dtype_obj
    raise ArtifactError(
        f"Unsupported dtype '{value}' in canonical index",
        status_code="DATA_LOSS",
        retryable=False,
    )


def canonical_index_from_bytes(
    index_bytes: bytes, *, avbs_hash: str = ""
) -> CanonicalIndex:
    try:
        raw = json.loads(index_bytes.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ArtifactError(
            "Failed to parse canonical index JSON",
            status_code="DATA_LOSS",
            retryable=False,
        ) from exc

    entries: list[CanonicalIndexEntry] = []
    total = 0
    for name, meta in raw.items():
        if not isinstance(meta, (list, tuple)) or len(meta) != 6:
            raise ArtifactError(
                f"Invalid canonical index entry for '{name}'",
                status_code="DATA_LOSS",
                retryable=False,
            )
        offset, size_bytes, shape, stride, dtype_str, storage_offset = meta
        dtype = dtype_from_string(str(dtype_str))
        entry = CanonicalIndexEntry(
            name=name,
            dtype=dtype,
            shape=tuple(int(x) for x in shape),
            stride=tuple(int(x) for x in stride),
            storage_offset=int(storage_offset),
            segment_offset=int(offset),
            size_bytes=int(size_bytes),
        )
        entries.append(entry)
        total = max(total, int(entry.segment_offset) + int(entry.size_bytes))

    return CanonicalIndex(
        entries=tuple(entries),
        total_size_bytes=total,
        avbs_hash=avbs_hash,
    )


def canonical_entry_storage_span_bytes(entry: CanonicalIndexEntry) -> int:
    element_size = int(torch.empty((), dtype=entry.dtype).element_size())
    if not entry.shape:
        return element_size
    if len(entry.shape) != len(entry.stride):
        raise ArtifactError(
            f"Invalid canonical index entry for '{entry.name}': shape/stride rank mismatch",
            status_code="DATA_LOSS",
            retryable=False,
        )
    max_offset = 0
    for dim, stride in zip(entry.shape, entry.stride, strict=True):
        if int(dim) < 0:
            raise ArtifactError(
                f"Invalid canonical index entry for '{entry.name}': negative shape dimension",
                status_code="DATA_LOSS",
                retryable=False,
            )
        if int(dim) == 0:
            return 0
        max_offset += (int(dim) - 1) * abs(int(stride))
    return (max_offset + 1) * element_size


def canonical_index_storage_extent(entries: Sequence[CanonicalIndexEntry]) -> int:
    total = 0
    for entry in entries:
        total = max(total, int(entry.segment_offset) + int(entry.size_bytes))
    return total


def canonical_index_to_bytes(
    canonical_index: CanonicalIndex, names: Sequence[str] | None = None
) -> bytes:
    selected = canonical_index.entries
    if names is not None:
        requested = set(names)
        selected = tuple(
            entry for entry in canonical_index.entries if entry.name in requested
        )
    data: dict[str, tuple[int, int, list[int], list[int], str, int]] = {}
    for entry in selected:
        data[entry.name] = (
            int(entry.segment_offset),
            int(entry.size_bytes),
            list(entry.shape),
            list(entry.stride),
            str(entry.dtype),
            int(entry.storage_offset),
        )
    return json.dumps(data, separators=(",", ":"), sort_keys=True).encode("utf-8")


def canonical_index_from_result(result: RegistrationResult) -> CanonicalIndex:
    avbs_hash = result.descriptor.data_multihash or ""
    return canonical_index_from_bytes(
        result.index_bytes,
        avbs_hash=avbs_hash,
    )


def replica_type_for_plan(plan: PlanType) -> tuple[ReplicaType, PlanType]:
    if plan is PlanType.DRAM_STABLE:
        return "DRAM_STABLE", PlanType.DRAM_STABLE
    if plan is PlanType.VRAM_COALESCED:
        return "COALESCED_VRAM", PlanType.VRAM_COALESCED
    return "VRAM_LEASE_IN_PLACE", PlanType.VRAM_LEASED


def replica_info_from_result(result: RegistrationResult) -> ReplicaInfo:
    replica_type, plan = replica_type_for_plan(result.plan)
    if plan is PlanType.DRAM_STABLE:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda", int(result.build.device_id))
    size_bytes = int(result.layout.total_size)
    replica_id = result.descriptor.artifact_id
    return ReplicaInfo(
        replica_id=replica_id,
        replica_type=replica_type,
        device=device,
        plan=plan,
        size_bytes=size_bytes,
    )


def lease_handle_from_result(lease: RegisteredLease | None) -> LeaseHandle | None:
    if lease is None:
        return None
    ttl = max(0, int(lease.ttl_ms))
    return LeaseHandle(
        lease_id=lease.registration_id,
        ttl_ms=ttl,
        expires_at_monotonic=time.monotonic() + ttl / 1000.0 if ttl else 0.0,
        owner_pid=int(lease.owner_pid),
    )


def select_device_for_put(tensors: TensorDict) -> int | None:
    device_id: int | None = None
    saw_cpu = False
    saw_cuda = False
    for tensor in tensors.values():
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.is_cuda:
            saw_cuda = True
            idx = tensor.device.index or 0
            if device_id is None:
                device_id = idx
            elif device_id != idx:
                raise ArtifactError(
                    "All CUDA tensors must reside on the same device",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
        else:
            saw_cpu = True
    if saw_cpu and saw_cuda:
        raise ArtifactError(
            "Artifact tensors must be all CPU or all CUDA tensors on the same device",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    if device_id is not None:
        return device_id
    if saw_cpu:
        return None
    return None


def validate_targets(
    *,
    canonical_index: CanonicalIndex,
    target: Mapping[str, torch.Tensor],
    source: Mapping[str, torch.Tensor],
    device_id: int,
    required_names: Sequence[str] | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    validated: list[tuple[torch.Tensor, torch.Tensor]] = []
    device = torch.device("cuda", device_id)
    entries_by_name = {entry.name: entry for entry in canonical_index.entries}
    if required_names is None:
        ordered_names = [entry.name for entry in canonical_index.entries]
    else:
        ordered_names = []
        seen: set[str] = set()
        for name in required_names:
            if name in seen:
                continue
            if name not in entries_by_name:
                raise ArtifactError(
                    f"Tensor '{name}' not found in artifact",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            ordered_names.append(name)
            seen.add(name)
    for name in ordered_names:
        entry = entries_by_name[name]
        if name not in target:
            raise ArtifactError(
                f"Target tensor '{name}' missing",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if name not in source:
            raise ArtifactError(
                f"Source tensor '{name}' missing",
                status_code="DATA_LOSS",
                retryable=False,
            )
        tgt = target[name]
        src = source[name]
        if not isinstance(tgt, torch.Tensor):
            raise ArtifactError(
                f"Target '{entry.name}' must be a tensor",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if tgt.device != device:
            raise ArtifactError(
                (
                    f"Target tensor '{entry.name}' on {tgt.device}, expected cuda:{device.index}"
                ),
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if tgt.dtype != src.dtype:
            raise ArtifactError(
                f"Target tensor '{entry.name}' dtype {tgt.dtype} != {src.dtype}",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if tuple(tgt.shape) != entry.shape:
            raise ArtifactError(
                f"Target tensor '{entry.name}' shape {tuple(tgt.shape)} != {entry.shape}",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        validated.append((tgt, src))
    return validated


__all__ = [
    "canonical_entry_storage_span_bytes",
    "canonical_index_from_bytes",
    "canonical_index_to_bytes",
    "canonical_index_storage_extent",
    "canonical_index_from_result",
    "dtype_from_string",
    "lease_handle_from_result",
    "replica_info_from_result",
    "replica_type_for_plan",
    "select_device_for_put",
    "validate_targets",
]
