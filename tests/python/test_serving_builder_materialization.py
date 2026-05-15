#  Copyright (c) 2026, TensorCast Team.

import pytest
import torch

from tensorcast.serving.builder.materialization import (
    apply_copy_plan,
    validate_dst_coverage,
)
from tensorcast.serving.builder.trace_ir import (
    CopyPlanEntry,
    MultiRange,
    Range,
    TracePlan,
)


def test_apply_copy_plan_handles_copy_fill_and_multi_range() -> None:
    plan = TracePlan(
        copy_plan=[
            CopyPlanEntry(
                op="copy",
                ckpt_name="x",
                ckpt_range=Range(dim=0, start=0, end=2),
                dst_name="w",
                dst_range=Range(dim=0, start=0, end=2),
            ),
            CopyPlanEntry(
                op="fill",
                ckpt_name=None,
                ckpt_range=None,
                dst_name="w",
                dst_range=MultiRange(ranges=(
                    Range(dim=0, start=2, end=3),
                    Range(dim=1, start=0, end=2),
                )),
                fill_value=7,
            ),
        ],
        expected_src_names={"x"},
        expected_dst_names={"w"},
        tensorcast_slices={},
        src_hull={"x": Range(dim=0, start=0, end=2)},
    )
    source = {"x": torch.tensor([[1, 2], [3, 4]])}
    serving = {"w": torch.zeros((3, 2), dtype=torch.int64)}

    apply_copy_plan(plan, source, serving)

    assert torch.equal(serving["w"], torch.tensor([[1, 2], [3, 4], [7, 7]]))


def test_validate_dst_coverage_detects_gaps() -> None:
    plan = TracePlan(
        copy_plan=[
            CopyPlanEntry(
                op="fill",
                ckpt_name=None,
                ckpt_range=None,
                dst_name="w",
                dst_range=Range(dim=0, start=0, end=1),
                fill_value=0,
            )
        ],
        expected_src_names=set(),
        expected_dst_names={"w"},
        tensorcast_slices={},
        src_hull={},
    )

    with pytest.raises(RuntimeError, match="coverage gap"):
        validate_dst_coverage(plan, {"w": torch.empty((2,))})
