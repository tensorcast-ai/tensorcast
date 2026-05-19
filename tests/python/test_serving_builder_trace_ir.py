#  Copyright (c) 2026, TensorCast Team.

from tensorcast.serving.builder.trace_ir import (
    CopyPlanEntry,
    MultiRange,
    Range,
    TracePlan,
    copy_plan_from_dict,
    copy_plan_to_dict,
    trace_plan_from_dict,
    trace_plan_to_dict,
)


def test_trace_ir_round_trips_copy_plan_with_multi_range_and_fill() -> None:
    entry = CopyPlanEntry(
        op="fill",
        ckpt_name=None,
        ckpt_range=None,
        dst_name="w",
        dst_range=MultiRange(ranges=(
            Range(dim=0, start=0, end=1),
            Range(dim=1, start=2, end=4),
        )),
        fill_value=3.0,
    )

    assert copy_plan_from_dict(copy_plan_to_dict(entry)) == entry


def test_trace_plan_round_trips_json_dict() -> None:
    plan = TracePlan(
        copy_plan=[
            CopyPlanEntry(
                op="copy",
                ckpt_name="x",
                ckpt_range=Range(dim=0, start=1, end=3),
                dst_name="w",
                dst_range=Range(dim=0, start=0, end=2),
            )
        ],
        expected_src_names={"x"},
        expected_dst_names={"w"},
        tensorcast_slices={"x": Range(dim=0, start=1, end=3)},
        src_hull={"x": Range(dim=0, start=1, end=3)},
    )

    assert trace_plan_from_dict(trace_plan_to_dict(plan)) == plan
