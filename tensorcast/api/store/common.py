#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import json
import logging
import time
from typing import Mapping

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
        total += entry.size_bytes

    return CanonicalIndex(
        entries=tuple(entries),
        total_size_bytes=total,
        avbs_hash=avbs_hash,
    )


def canonical_index_from_result(result: RegistrationResult) -> CanonicalIndex:
    avbs_hash = result.descriptor.data_multihash or ""
    return canonical_index_from_bytes(
        result.index_bytes,
        avbs_hash=avbs_hash,
    )


def replica_type_for_plan(plan: PlanType) -> tuple[ReplicaType, PlanType]:
    if plan is PlanType.VRAM_COALESCED:
        return "COALESCED_VRAM", PlanType.VRAM_COALESCED
    return "VRAM_LEASE_IN_PLACE", PlanType.VRAM_LEASED


def replica_info_from_result(result: RegistrationResult) -> ReplicaInfo:
    replica_type, plan = replica_type_for_plan(result.plan)
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


def select_device_for_put(tensors: TensorDict) -> int:
    device_id: int | None = None
    for tensor in tensors.values():
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.is_cuda:
            idx = tensor.device.index or 0
            if device_id is None:
                device_id = idx
            elif device_id != idx:
                raise ArtifactError(
                    "All CUDA tensors must reside on the same device",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
    if device_id is not None:
        return device_id
    if torch.cuda.is_available():
        try:
            return int(torch.cuda.current_device())
        except Exception:  # noqa: BLE001
            return 0
    return 0


def validate_targets(
    *,
    canonical_index: CanonicalIndex,
    target: Mapping[str, torch.Tensor],
    source: Mapping[str, torch.Tensor],
    device_id: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    validated: list[tuple[torch.Tensor, torch.Tensor]] = []
    device = torch.device("cuda", device_id)
    for entry in canonical_index.entries:
        if entry.name not in target:
            raise ArtifactError(
                f"Target tensor '{entry.name}' missing",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if entry.name not in source:
            raise ArtifactError(
                f"Source tensor '{entry.name}' missing",
                status_code="DATA_LOSS",
                retryable=False,
            )
        tgt = target[entry.name]
        src = source[entry.name]
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
    "canonical_index_from_bytes",
    "canonical_index_from_result",
    "dtype_from_string",
    "lease_handle_from_result",
    "replica_info_from_result",
    "replica_type_for_plan",
    "select_device_for_put",
    "validate_targets",
]
