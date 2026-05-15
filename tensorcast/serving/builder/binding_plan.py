#  Copyright (c) 2026, TensorCast Team.

"""Framework-neutral TracePlan lowering for TensorCast serving bindings."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TypeAlias

from tensorcast.api.store import BindingRealizationEntry
from tensorcast.api.store import CopyPlanEntry as StoreCopyPlanEntry
from tensorcast.api.store import Range as StoreRange
from tensorcast.serving.builder.trace_ir import (
    CopyPlanEntry,
    MultiRange,
    Range,
    RangeSpec,
    TracePlan,
)

TargetShapes: TypeAlias = Mapping[str, Sequence[int]]


def lower_trace_plan_for_binding(
    trace_plan: TracePlan,
    target_shapes: TargetShapes,
    *,
    allow_multirange_lowering: bool = True,
) -> tuple[
    tuple[StoreCopyPlanEntry, ...],
    tuple[CopyPlanEntry, ...],
    tuple[tuple[str, int], ...],
]:
    """Lower copy-only TracePlan entries to mapped-binding copy entries.

    Mapped binding only accepts single-axis destination ranges. MultiRange
    destination slices are lowerable when they select one value from each
    prefix dimension and can therefore be flattened into dim-0 ranges using
    framework-provided target shapes.
    """

    mapped: list[StoreCopyPlanEntry] = []
    fallback: list[CopyPlanEntry] = []
    flattened_prefix_dims: dict[str, int] = {}
    grouped_entries: dict[str, list[CopyPlanEntry]] = {}
    for entry in trace_plan.copy_plan:
        grouped_entries.setdefault(entry.dst_name, []).append(entry)

    for dst_name, entries in grouped_entries.items():
        target_shape = target_shapes.get(dst_name)
        lowered_prefix_dims = (
            None
            if (not allow_multirange_lowering or target_shape is None)
            else (_lowerable_multirange_prefix_dims(entries))
        )
        if lowered_prefix_dims is not None:
            flattened_prefix_dims[dst_name] = lowered_prefix_dims
        for entry in entries:
            if _range_spec_is_noop(entry.dst_range):
                continue
            if entry.op != "copy" or entry.ckpt_name is None:
                fallback.append(entry)
                continue
            if isinstance(entry.ckpt_range, MultiRange):
                fallback.append(entry)
                continue
            lowered_dst_range = entry.dst_range
            if isinstance(entry.dst_range, MultiRange):
                if lowered_prefix_dims is None or target_shape is None:
                    fallback.append(entry)
                    continue
                lowered_dst_range = _lower_multirange_to_range(
                    entry.dst_range,
                    target_shape,
                    prefix_dims=lowered_prefix_dims,
                )
            if isinstance(lowered_dst_range, MultiRange):
                fallback.append(entry)
                continue
            mapped.append(
                StoreCopyPlanEntry(
                    ckpt_name=str(entry.ckpt_name),
                    ckpt_range=None
                    if entry.ckpt_range is None
                    else _range_to_store_range(entry.ckpt_range),
                    dst_name=str(entry.dst_name),
                    dst_range=None
                    if lowered_dst_range is None
                    else _range_to_store_range(lowered_dst_range),
                )
            )
    return (
        tuple(mapped),
        tuple(fallback),
        tuple(sorted(flattened_prefix_dims.items())),
    )


def range_spec_to_tensorcast_ranges(
    range_spec: RangeSpec | None,
) -> tuple[StoreRange, ...]:
    if range_spec is None:
        return ()
    if isinstance(range_spec, MultiRange):
        return tuple(_range_to_store_range(rng) for rng in range_spec.ranges)
    return (_range_to_store_range(range_spec),)


def lower_trace_plan_for_realization(
    trace_plan: TracePlan,
    target_shapes: TargetShapes,
    *,
    allow_multirange_lowering: bool = True,
) -> tuple[
    tuple[BindingRealizationEntry, ...],
    tuple[CopyPlanEntry, ...],
    tuple[tuple[str, int], ...],
]:
    """Lower TracePlan entries to daemon-side binding realization entries."""

    del target_shapes, allow_multirange_lowering
    lowered: list[BindingRealizationEntry] = []
    fallback: list[CopyPlanEntry] = []
    grouped_entries: dict[str, list[CopyPlanEntry]] = {}
    for entry in trace_plan.copy_plan:
        grouped_entries.setdefault(entry.dst_name, []).append(entry)

    for entries in grouped_entries.values():
        for entry in entries:
            if _range_spec_is_noop(entry.dst_range):
                continue
            if entry.op == "copy" and entry.ckpt_name is not None:
                lowered.append(
                    BindingRealizationEntry(
                        op="copy",
                        source_name=str(entry.ckpt_name),
                        source_ranges=range_spec_to_tensorcast_ranges(entry.ckpt_range),
                        dst_name=str(entry.dst_name),
                        dst_ranges=range_spec_to_tensorcast_ranges(entry.dst_range),
                    )
                )
                continue
            if entry.op == "fill":
                if entry.ckpt_name is None:
                    lowered.append(
                        BindingRealizationEntry(
                            op="fill",
                            dst_name=str(entry.dst_name),
                            dst_ranges=range_spec_to_tensorcast_ranges(entry.dst_range),
                            fill_value=entry.fill_value,
                        )
                    )
                    continue
                lowered.append(
                    BindingRealizationEntry(
                        op="fill",
                        source_name=str(entry.ckpt_name),
                        source_ranges=range_spec_to_tensorcast_ranges(entry.ckpt_range),
                        dst_name=str(entry.dst_name),
                        dst_ranges=range_spec_to_tensorcast_ranges(entry.dst_range),
                    )
                )
                continue
            fallback.append(entry)
    return (tuple(lowered), tuple(fallback), ())


def _range_to_store_range(rng: Range) -> StoreRange:
    return StoreRange(
        dim=int(rng.dim),
        start=int(rng.start),
        end=int(rng.end),
    )


def _range_length(rng: Range) -> int:
    return int(rng.end) - int(rng.start)


def _range_spec_is_noop(range_spec: RangeSpec | None) -> bool:
    if range_spec is None:
        return False
    if isinstance(range_spec, MultiRange):
        return any(_range_length(rng) <= 0 for rng in range_spec.ranges)
    return _range_length(range_spec) <= 0


def _lowerable_multirange_prefix_dims(
    entries: Iterable[CopyPlanEntry],
) -> int | None:
    prefix_dims: int | None = None
    for entry in entries:
        if not isinstance(entry.dst_range, MultiRange):
            continue
        ranges = tuple(entry.dst_range.ranges)
        if not ranges:
            return None
        dims = tuple(int(rng.dim) for rng in ranges)
        if dims != tuple(range(len(ranges))):
            return None
        for rng in ranges[:-1]:
            if _range_length(rng) != 1:
                return None
        current_prefix_dims = len(ranges)
        if prefix_dims is None:
            prefix_dims = current_prefix_dims
        elif prefix_dims != current_prefix_dims:
            return None
    return prefix_dims


def _lower_multirange_to_range(
    range_spec: MultiRange,
    target_shape: Sequence[int],
    *,
    prefix_dims: int,
) -> Range:
    ranges = tuple(range_spec.ranges)
    if prefix_dims != len(ranges):
        raise RuntimeError("Flattened prefix dims do not match MultiRange rank")
    shape = tuple(int(dim) for dim in target_shape)
    if prefix_dims > len(shape):
        raise RuntimeError("Flattened prefix dims exceed target tensor rank")
    start = 0
    for idx, rng in enumerate(ranges):
        stride = 1
        for dim in shape[idx + 1 : prefix_dims]:
            stride *= int(dim)
        start += int(rng.start) * stride
    length = _range_length(ranges[-1])
    return Range(dim=0, start=int(start), end=int(start + length))


__all__ = [
    "TargetShapes",
    "lower_trace_plan_for_binding",
    "lower_trace_plan_for_realization",
    "range_spec_to_tensorcast_ranges",
]
