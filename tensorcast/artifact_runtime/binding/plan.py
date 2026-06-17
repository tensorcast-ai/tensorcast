#  Copyright (c) 2026, TensorCast Team.

"""Framework-neutral TracePlan lowering for TensorCast runtime bindings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias

from tensorcast.api.store import BindingRealizationEntry
from tensorcast.api.store import CopyPlanEntry as StoreCopyPlanEntry
from tensorcast.api.store import Range as StoreRange
from tensorcast.artifact_runtime.recipe.trace_ir import (
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
    destination slices stay on the fallback path and should be represented with
    BindingRealizationPlan until the mapped-binding protocol has a first-class
    flattened-layout contract.
    """

    del target_shapes, allow_multirange_lowering
    mapped: list[StoreCopyPlanEntry] = []
    fallback: list[CopyPlanEntry] = []
    for entry in trace_plan.copy_plan:
        if _range_spec_is_noop(entry.dst_range):
            continue
        if entry.op != "copy" or entry.ckpt_name is None:
            fallback.append(entry)
            continue
        if isinstance(entry.ckpt_range, MultiRange) or isinstance(
            entry.dst_range,
            MultiRange,
        ):
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
                if entry.dst_range is None
                else _range_to_store_range(entry.dst_range),
            )
        )
    return (tuple(mapped), tuple(fallback), ())


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


__all__ = [
    "TargetShapes",
    "lower_trace_plan_for_binding",
    "lower_trace_plan_for_realization",
    "range_spec_to_tensorcast_ranges",
]
