#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from tensorcast.api.store.common import canonical_index_from_bytes, dtype_from_string
from tensorcast.api.store.mapped_binding import Range
from tensorcast.api.store.types import ArtifactError
from tensorcast.profile_utils import (
    emit_tensorcast_profile_event,
    tensorcast_profile_enabled,
)
from tensorcast.proto.daemon.v2 import store_daemon_pb2


@dataclass(frozen=True, slots=True)
class BindingRealizationEntry:
    op: str
    dst_name: str
    source_name: str | None = None
    source_ranges: tuple[Range, ...] = ()
    dst_ranges: tuple[Range, ...] = ()
    fill_value: bytes | int | float | bool | None = None


BindingRealizationPlan = Sequence[BindingRealizationEntry]


def normalize_binding_realization_plan(
    plan: Sequence[object],
) -> tuple[BindingRealizationEntry, ...]:
    if not isinstance(plan, Sequence) or not plan:
        raise ArtifactError(
            "realization_plan must be a non-empty sequence of "
            "BindingRealizationEntry objects",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    entries: list[BindingRealizationEntry] = []
    for raw in plan:
        if isinstance(raw, BindingRealizationEntry):
            entries.append(
                BindingRealizationEntry(
                    op=str(raw.op),
                    dst_name=str(raw.dst_name),
                    source_name=None
                    if raw.source_name is None
                    else str(raw.source_name),
                    source_ranges=_coerce_ranges(raw.source_ranges),
                    dst_ranges=_coerce_ranges(raw.dst_ranges),
                    fill_value=raw.fill_value,
                )
            )
            continue
        if isinstance(raw, Mapping):
            entries.append(
                BindingRealizationEntry(
                    op=str(raw.get("op", "")),
                    dst_name=str(raw.get("dst_name", "")),
                    source_name=(
                        None
                        if raw.get("source_name") is None
                        else str(raw.get("source_name"))
                    ),
                    source_ranges=_coerce_ranges(raw.get("source_ranges")),
                    dst_ranges=_coerce_ranges(raw.get("dst_ranges")),
                    fill_value=raw.get("fill_value"),
                )
            )
            continue
        raise ArtifactError(
            "realization_plan entries must be BindingRealizationEntry objects or dicts",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    return tuple(entries)


def binding_realization_plan_to_proto(
    plan: Sequence[object],
    *,
    target_index_bytes: bytes,
) -> store_daemon_pb2.BindingRealizationPlan:
    if not isinstance(plan, Sequence) or not plan:
        raise ArtifactError(
            "realization_plan must be a non-empty sequence of BindingRealizationEntry objects",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    target_index = canonical_index_from_bytes(target_index_bytes)
    dtype_by_name = {entry.name: entry.dtype for entry in target_index.entries}
    proto = store_daemon_pb2.BindingRealizationPlan(version=1)
    copy_count = 0
    const_fill_count = 0
    scalar_fill_count = 0
    sample_copy_dsts: list[str] = []
    sample_scalar_fill_dsts: list[str] = []
    for idx, raw in enumerate(plan):
        source_ranges: object | None
        dst_ranges: object | None
        if isinstance(raw, BindingRealizationEntry):
            op = str(raw.op).strip().lower()
            dst_name = str(raw.dst_name)
            source_name = None if raw.source_name is None else str(raw.source_name)
            source_ranges = raw.source_ranges
            dst_ranges = raw.dst_ranges
            fill_value = raw.fill_value
        elif isinstance(raw, Mapping):
            op = str(raw.get("op", "")).strip().lower()
            dst_name = str(raw.get("dst_name", ""))
            source_name = (
                None if raw.get("source_name") is None else str(raw.get("source_name"))
            )
            source_ranges = raw.get("source_ranges")
            dst_ranges = raw.get("dst_ranges")
            fill_value = raw.get("fill_value")
        else:
            raise ArtifactError(
                "realization_plan entries must be BindingRealizationEntry "
                "objects or dicts",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if _ranges_describe_noop_for_proto(dst_ranges, idx=idx):
            continue
        if not dst_name:
            raise ArtifactError(
                f"realization_plan[{idx}] dst_name is required",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        dst_dtype = dtype_by_name.get(dst_name)
        if dst_dtype is None:
            raise ArtifactError(
                f"realization_plan[{idx}] dst_name '{dst_name}' is not "
                "present in BindingLayout",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        proto_entry = proto.entries.add(dst_name=dst_name)
        if op == "copy":
            if not source_name:
                raise ArtifactError(
                    f"realization_plan[{idx}] copy requires source_name",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            proto_entry.op_kind = store_daemon_pb2.BINDING_REALIZATION_OP_KIND_COPY
            proto_entry.source_name = source_name
            copy_count += 1
            if len(sample_copy_dsts) < 8:
                sample_copy_dsts.append(dst_name)
        elif op == "fill":
            if source_name:
                proto_entry.op_kind = (
                    store_daemon_pb2.BINDING_REALIZATION_OP_KIND_SCALAR_FILL
                )
                proto_entry.source_name = source_name
                scalar_fill_count += 1
                if len(sample_scalar_fill_dsts) < 8:
                    sample_scalar_fill_dsts.append(dst_name)
            else:
                proto_entry.op_kind = (
                    store_daemon_pb2.BINDING_REALIZATION_OP_KIND_CONST_FILL
                )
                proto_entry.fill_value = _coerce_fill_value_bytes(
                    fill_value,
                    dst_dtype=dst_dtype,
                    idx=idx,
                )
                const_fill_count += 1
        else:
            raise ArtifactError(
                f"realization_plan[{idx}] has unsupported op '{op}'",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        for range_spec in _iter_ranges_for_proto(
            source_ranges, idx=idx, field="source_ranges"
        ):
            proto_entry.source_ranges.add(
                dim=int(range_spec.dim),
                start=int(range_spec.start),
                end=int(range_spec.end),
            )
        for range_spec in _iter_ranges_for_proto(
            dst_ranges, idx=idx, field="dst_ranges"
        ):
            proto_entry.dst_ranges.add(
                dim=int(range_spec.dim),
                start=int(range_spec.start),
                end=int(range_spec.end),
            )
    if tensorcast_profile_enabled():
        emit_tensorcast_profile_event(
            "tensorcast_sdk",
            "binding_realization_plan_to_proto",
            payload={
                "entry_count": len(proto.entries),
                "copy_count": copy_count,
                "const_fill_count": const_fill_count,
                "scalar_fill_count": scalar_fill_count,
                "sample_copy_dsts": sample_copy_dsts,
                "sample_scalar_fill_dsts": sample_scalar_fill_dsts,
            },
        )
    return proto


def _ranges_describe_noop(ranges: Sequence[Range]) -> bool:
    if not ranges:
        return False
    return any(int(rng.end) <= int(rng.start) for rng in ranges)


def _ranges_describe_noop_for_proto(
    ranges: object | None,
    *,
    idx: int,
) -> bool:
    if ranges is None:
        return False
    for rng in _iter_ranges_for_proto(ranges, idx=idx, field="dst_ranges"):
        if int(rng.end) <= int(rng.start):
            return True
    return False


def _iter_ranges_for_proto(
    value: object | None,
    *,
    idx: int,
    field: str,
):
    if value is None:
        return
    if isinstance(value, Range):
        yield value
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            if isinstance(item, Range):
                yield item
                continue
            if isinstance(item, Mapping):
                yield Range(
                    dim=int(item["dim"]),
                    start=int(item["start"]),
                    end=int(item["end"]),
                )
                continue
            raise ArtifactError(
                f"realization_plan[{idx}] {field} entries must be Range "
                "objects or dicts",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        return
    raise ArtifactError(
        f"realization_plan[{idx}] {field} must be a Range or a sequence of "
        "Range objects",
        status_code="INVALID_ARGUMENT",
        retryable=False,
    )


def _coerce_ranges(value: object | None) -> tuple[Range, ...]:
    if value is None:
        return ()
    if isinstance(value, Range):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        ranges: list[Range] = []
        for item in value:
            if isinstance(item, Range):
                ranges.append(item)
                continue
            if isinstance(item, Mapping):
                ranges.append(
                    Range(
                        dim=int(item["dim"]),
                        start=int(item["start"]),
                        end=int(item["end"]),
                    )
                )
                continue
            raise ArtifactError(
                "realization range entries must be Range objects or dicts",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        return tuple(ranges)
    raise ArtifactError(
        "ranges must be a Range or a sequence of Range objects",
        status_code="INVALID_ARGUMENT",
        retryable=False,
    )


def _coerce_fill_value_bytes(
    value: bytes | int | float | bool | None,
    *,
    dst_dtype: torch.dtype,
    idx: int,
) -> bytes:
    if isinstance(value, bytes):
        return bytes(value)
    if value is None:
        raise ArtifactError(
            f"realization_plan[{idx}] const fill requires fill_value",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    tensor = torch.tensor(
        [value], dtype=dtype_from_string(str(dst_dtype)), device="cpu"
    )
    return bytes(tensor.view(torch.uint8).tolist())


__all__ = [
    "BindingRealizationEntry",
    "BindingRealizationPlan",
    "binding_realization_plan_to_proto",
    "normalize_binding_realization_plan",
]
