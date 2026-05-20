#  Copyright (c) 2026, TensorCast Team.

from types import SimpleNamespace

import torch

from tensorcast.api.store import BindingRealizationEntry
from tensorcast.api.store import Range as StoreRange
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.serving.builder.tensor_parity import (
    build_tensor_parity_probes_from_realization_plan,
    build_tensor_parity_probes_from_realization_plan_proto,
    build_tensor_parity_probes_from_recipe,
    build_tensor_parity_probes_from_trace_plan,
    evaluate_recipe_tensor_parity,
    evaluate_tensor_parity_probes,
)
from tensorcast.serving.builder.trace_ir import (
    CopyPlanEntry,
    MultiRange,
    Range,
    TracePlan,
)


def _trace_plan() -> TracePlan:
    return TracePlan(
        copy_plan=[
            CopyPlanEntry(
                op="copy",
                ckpt_name="q",
                ckpt_range=Range(dim=0, start=2, end=4),
                dst_name="qkv",
                dst_range=Range(dim=0, start=0, end=2),
            ),
            CopyPlanEntry(
                op="fill",
                ckpt_name=None,
                ckpt_range=None,
                dst_name="qkv",
                dst_range=MultiRange(ranges=(
                    Range(dim=0, start=2, end=3),
                    Range(dim=1, start=0, end=2),
                )),
                fill_value=7,
            ),
        ],
        expected_src_names={"q"},
        expected_dst_names={"qkv"},
        tensorcast_slices={},
        src_hull={"q": Range(dim=0, start=1, end=4)},
    )


def test_tensor_parity_probe_passes_copy_and_fill_entries() -> None:
    plan = _trace_plan()
    source = {"q": torch.tensor([[0, 0], [2, 3], [4, 5]])}
    target = {"qkv": torch.tensor([[2, 3], [4, 5], [7, 7]])}

    probes = build_tensor_parity_probes_from_trace_plan(plan)
    report = evaluate_tensor_parity_probes(
        plan,
        source,
        target,
        probes=probes,
    )

    assert len(probes) == 2
    assert report.passed is True
    assert report.checked == 2
    assert report.to_dict() == {
        "checked": 2,
        "skipped": 0,
        "passed": True,
        "mismatches": [],
    }


def test_tensor_parity_probe_reports_sampled_value_mismatch() -> None:
    plan = _trace_plan()
    source = {"q": torch.tensor([[0, 0], [2, 3], [4, 5]])}
    target = {"qkv": torch.tensor([[2, 3], [4, 99], [7, 7]])}

    report = evaluate_tensor_parity_probes(
        plan,
        source,
        target,
        max_elements_per_probe=4,
    )

    assert report.passed is False
    assert len(report.mismatches) == 1
    mismatch = report.mismatches[0]
    assert mismatch.probe_index == 0
    assert mismatch.ckpt_name == "q"
    assert mismatch.dst_name == "qkv"
    assert mismatch.reason == "value_mismatch"
    assert mismatch.expected_shape == (2, 2)
    assert mismatch.actual_shape == (2, 2)
    assert mismatch.sample_count == 4
    assert mismatch.mismatched_samples == 1
    assert mismatch.max_abs_diff == 94.0


def test_tensor_parity_probe_samples_large_views_without_full_compare(
) -> None:
    plan = TracePlan(
        copy_plan=[
            CopyPlanEntry(
                op="copy",
                ckpt_name="w",
                ckpt_range=None,
                dst_name="w",
                dst_range=None,
            )
        ],
        expected_src_names={"w"},
        expected_dst_names={"w"},
        tensorcast_slices={},
        src_hull={},
    )
    source = {"w": torch.arange(100, dtype=torch.float32).reshape(10, 10)}
    target = {"w": source["w"].clone()}
    target["w"][9, 9] = -1

    report = evaluate_tensor_parity_probes(
        plan,
        source,
        target,
        max_elements_per_probe=2,
    )

    assert report.passed is False
    mismatch = report.mismatches[0]
    assert mismatch.total_elements == 100
    assert mismatch.sample_count == 2
    assert mismatch.mismatched_samples == 1
    assert mismatch.max_abs_diff == 100.0


def test_tensor_parity_probe_reports_missing_source_tensor() -> None:
    plan = _trace_plan()

    report = evaluate_tensor_parity_probes(
        plan,
        {},
        {"qkv": torch.zeros((3, 2), dtype=torch.int64)},
        max_elements_per_probe=2,
    )

    assert report.passed is False
    assert report.mismatches[0].probe_index == 0
    assert report.mismatches[0].reason == "missing_source_tensor"


def test_tensor_parity_builds_probes_from_realization_plan() -> None:
    entries = (
        BindingRealizationEntry(
            op="copy",
            source_name="w",
            source_ranges=(StoreRange(dim=0, start=2, end=4), ),
            dst_name="w_tp",
            dst_ranges=(StoreRange(dim=0, start=0, end=2), ),
        ),
        BindingRealizationEntry(
            op="fill",
            dst_name="w_tp",
            dst_ranges=(
                StoreRange(dim=0, start=2, end=3),
                StoreRange(dim=1, start=0, end=2),
            ),
            fill_value=0,
        ),
    )

    probes = build_tensor_parity_probes_from_realization_plan(entries)

    assert len(probes) == 2
    assert probes[0].ckpt_name == "w"
    assert probes[0].dst_name == "w_tp"
    assert probes[1].fill_value == 0


def test_recipe_tensor_parity_uses_realization_plan_when_trace_is_summary_only(
) -> None:
    trace_plan = TracePlan(
        copy_plan=[],
        expected_src_names={"w"},
        expected_dst_names={"w_tp"},
        tensorcast_slices={},
        src_hull={},
    )
    recipe = SimpleNamespace(
        trace_plan=trace_plan,
        realization_plan=(BindingRealizationEntry(
            op="copy",
            source_name="w",
            source_ranges=(StoreRange(dim=0, start=1, end=3), ),
            dst_name="w_tp",
            dst_ranges=(StoreRange(dim=0, start=0, end=2), ),
        ), ),
    )
    source = {"w": torch.tensor([[1, 2], [3, 4], [5, 6]])}
    target = {"w_tp": torch.tensor([[3, 4], [5, 6]])}

    probes = build_tensor_parity_probes_from_recipe(recipe)
    report = evaluate_recipe_tensor_parity(recipe, source, target)

    assert len(probes) == 1
    assert report.passed is True
    assert report.checked == 1


def test_recipe_tensor_parity_uses_realization_plan_proto_when_plan_is_compact(
) -> None:
    proto = store_daemon_pb2.BindingRealizationPlan(version=1)
    entry = proto.entries.add(
        op_kind=store_daemon_pb2.BINDING_REALIZATION_OP_KIND_COPY,
        source_name="w",
        dst_name="w_tp",
    )
    entry.source_ranges.add(dim=0, start=1, end=3)
    entry.dst_ranges.add(dim=0, start=0, end=2)
    trace_plan = TracePlan(
        copy_plan=[],
        expected_src_names={"w"},
        expected_dst_names={"w_tp"},
        tensorcast_slices={},
        src_hull={},
    )
    recipe = SimpleNamespace(
        trace_plan=trace_plan,
        realization_plan=(),
        realization_plan_proto=proto.SerializeToString(),
    )
    source = {"w": torch.tensor([[1, 2], [3, 4], [5, 6]])}
    target = {"w_tp": torch.tensor([[3, 4], [5, 6]])}

    probes = build_tensor_parity_probes_from_realization_plan_proto(proto)
    report = evaluate_recipe_tensor_parity(recipe, source, target)

    assert len(probes) == 1
    assert build_tensor_parity_probes_from_recipe(recipe)[0].ckpt_name == "w"
    assert report.passed is True
    assert report.checked == 1
