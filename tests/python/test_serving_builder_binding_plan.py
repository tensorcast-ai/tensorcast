#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

from tensorcast.serving.builder.binding_plan import (
    lower_trace_plan_for_binding,
    lower_trace_plan_for_realization,
)
from tensorcast.serving.builder.trace_ir import (
    CopyPlanEntry,
    MultiRange,
    Range,
    TracePlan,
)


def test_lower_trace_plan_for_binding_keeps_dst_multirange_fallback() -> None:
    trace_plan = TracePlan(
        copy_plan=[
            CopyPlanEntry(
                op="copy",
                ckpt_name="gate_proj.weight",
                ckpt_range=Range(dim=0, start=0, end=1),
                dst_name="w13_weight",
                dst_range=MultiRange(
                    ranges=(
                        Range(dim=0, start=0, end=1),
                        Range(dim=1, start=0, end=3),
                    )
                ),
            ),
            CopyPlanEntry(
                op="copy",
                ckpt_name="up_proj.weight",
                ckpt_range=Range(dim=0, start=0, end=1),
                dst_name="w13_weight",
                dst_range=MultiRange(
                    ranges=(
                        Range(dim=0, start=0, end=1),
                        Range(dim=1, start=3, end=6),
                    )
                ),
            ),
        ],
        expected_src_names={"gate_proj.weight", "up_proj.weight"},
        expected_dst_names={"w13_weight"},
        tensorcast_slices={},
        src_hull={},
    )

    mapped, fallback, flattened_dims = lower_trace_plan_for_binding(
        trace_plan,
        {"w13_weight": (2, 6, 4)},
    )

    assert mapped == ()
    assert len(fallback) == 2
    assert flattened_dims == ()


def test_lower_trace_plan_for_binding_keeps_unlowerable_multirange_fallback() -> None:
    entry = CopyPlanEntry(
        op="copy",
        ckpt_name="src",
        ckpt_range=Range(dim=0, start=0, end=1),
        dst_name="w",
        dst_range=MultiRange(
            ranges=(
                Range(dim=1, start=0, end=1),
                Range(dim=0, start=0, end=4),
            )
        ),
    )
    trace_plan = TracePlan(
        copy_plan=[entry],
        expected_src_names={"src"},
        expected_dst_names={"w"},
        tensorcast_slices={},
        src_hull={},
    )

    mapped, fallback, flattened_dims = lower_trace_plan_for_binding(
        trace_plan,
        {"w": (2, 4)},
    )

    assert mapped == ()
    assert fallback == (entry,)
    assert flattened_dims == ()


def test_lower_trace_plan_for_realization_preserves_multiaxis_copy_ranges() -> None:
    trace_plan = TracePlan(
        copy_plan=[
            CopyPlanEntry(
                op="copy",
                ckpt_name="src",
                ckpt_range=Range(dim=0, start=0, end=1),
                dst_name="w",
                dst_range=MultiRange(
                    ranges=(
                        Range(dim=0, start=0, end=1),
                        Range(dim=1, start=0, end=10),
                    )
                ),
            ),
            CopyPlanEntry(
                op="fill",
                ckpt_name=None,
                ckpt_range=None,
                fill_value=0.0,
                dst_name="w",
                dst_range=Range(dim=0, start=4, end=4),
            ),
        ],
        expected_src_names={"src"},
        expected_dst_names={"w"},
        tensorcast_slices={},
        src_hull={},
    )

    realization_plan, fallback, flattened_dims = lower_trace_plan_for_realization(
        trace_plan, {}
    )

    assert fallback == ()
    assert flattened_dims == ()
    assert len(realization_plan) == 1
    entry = realization_plan[0]
    assert entry.op == "copy"
    assert entry.source_name == "src"
    assert len(entry.source_ranges) == 1
    assert entry.source_ranges[0].dim == 0
    assert entry.source_ranges[0].start == 0
    assert entry.source_ranges[0].end == 1
    assert len(entry.dst_ranges) == 2
    assert entry.dst_ranges[0].dim == 0
    assert entry.dst_ranges[0].start == 0
    assert entry.dst_ranges[0].end == 1
    assert entry.dst_ranges[1].dim == 1
    assert entry.dst_ranges[1].start == 0
    assert entry.dst_ranges[1].end == 10
