#  Copyright (c) 2026, TensorCast Team.

"""Framework-neutral TensorCast runtime trace IR."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True, slots=True)
class Range:
    dim: int
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class MultiRange:
    ranges: tuple[Range, ...]


RangeSpec = Range | MultiRange


@dataclass(frozen=True, slots=True)
class CopyPlanEntry:
    op: str
    ckpt_name: Optional[str]
    ckpt_range: Optional[RangeSpec]
    dst_name: str
    dst_range: Optional[RangeSpec]
    fill_value: Optional[int | float] = None


@dataclass(slots=True)
class TracePlan:
    copy_plan: list[CopyPlanEntry]
    expected_src_names: set[str]
    expected_dst_names: set[str]
    tensorcast_slices: dict[str, Range]
    src_hull: dict[str, Range]


def range_to_dict(range_spec: RangeSpec) -> dict[str, Any]:
    if isinstance(range_spec, Range):
        return {
            "type": "single",
            "dim": range_spec.dim,
            "start": range_spec.start,
            "end": range_spec.end,
        }
    return {
        "type": "multi",
        "ranges": [dataclasses.asdict(rng) for rng in range_spec.ranges],
    }


def range_from_dict(data: dict[str, Any]) -> RangeSpec:
    ranges_data = data.get("ranges")
    if ranges_data is not None:
        ranges = tuple(
            Range(dim=int(r["dim"]), start=int(r["start"]), end=int(r["end"]))
            for r in ranges_data
        )
        if len(ranges) == 1:
            return ranges[0]
        return MultiRange(ranges=ranges)
    return Range(
        dim=int(data["dim"]),
        start=int(data["start"]),
        end=int(data["end"]),
    )


def single_range_from_dict(data: dict[str, Any]) -> Range:
    range_spec = range_from_dict(data)
    if isinstance(range_spec, MultiRange):
        raise RuntimeError(
            "Trace cache has unsupported multi-range for single-range fields"
        )
    return range_spec


def copy_plan_to_dict(entry: CopyPlanEntry) -> dict[str, Any]:
    return {
        "op": entry.op,
        "ckpt_name": entry.ckpt_name,
        "ckpt_range": None
        if entry.ckpt_range is None
        else range_to_dict(entry.ckpt_range),
        "dst_name": entry.dst_name,
        "dst_range": None
        if entry.dst_range is None
        else range_to_dict(entry.dst_range),
        "fill_value": entry.fill_value,
    }


def copy_plan_from_dict(data: dict[str, Any]) -> CopyPlanEntry:
    return CopyPlanEntry(
        op=data.get("op", "copy"),
        ckpt_name=data["ckpt_name"],
        ckpt_range=range_from_dict(data["ckpt_range"])
        if data["ckpt_range"] is not None
        else None,
        dst_name=data["dst_name"],
        dst_range=range_from_dict(data["dst_range"])
        if data["dst_range"] is not None
        else None,
        fill_value=data.get("fill_value"),
    )


def trace_plan_to_dict(trace_plan: TracePlan) -> dict[str, Any]:
    return {
        "copy_plan": [copy_plan_to_dict(e) for e in trace_plan.copy_plan],
        "expected_src_names": sorted(trace_plan.expected_src_names),
        "expected_dst_names": sorted(trace_plan.expected_dst_names),
        "tensorcast_slices": {
            name: dataclasses.asdict(rng)
            for name, rng in trace_plan.tensorcast_slices.items()
        },
        "src_hull": {
            name: dataclasses.asdict(rng) for name, rng in trace_plan.src_hull.items()
        },
    }


def trace_plan_from_dict(data: dict[str, Any]) -> TracePlan:
    return TracePlan(
        copy_plan=[copy_plan_from_dict(item) for item in data["copy_plan"]],
        expected_src_names=set(data["expected_src_names"]),
        expected_dst_names=set(data["expected_dst_names"]),
        tensorcast_slices={
            name: single_range_from_dict(rng)
            for name, rng in data["tensorcast_slices"].items()
        },
        src_hull={
            name: single_range_from_dict(rng) for name, rng in data["src_hull"].items()
        },
    )


__all__ = [
    "CopyPlanEntry",
    "MultiRange",
    "Range",
    "RangeSpec",
    "TracePlan",
    "copy_plan_from_dict",
    "copy_plan_to_dict",
    "range_from_dict",
    "range_to_dict",
    "single_range_from_dict",
    "trace_plan_from_dict",
    "trace_plan_to_dict",
]
