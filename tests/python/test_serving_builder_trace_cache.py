#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import json

from tensorcast.serving.builder.trace_cache import (
    dump_trace_plan_debug,
    load_trace_plan_cache,
    trace_plan_debug_payload,
    write_trace_plan_cache,
)
from tensorcast.serving.builder.trace_ir import CopyPlanEntry, Range, TracePlan


def _trace_plan() -> TracePlan:
    return TracePlan(
        copy_plan=[
            CopyPlanEntry(
                op="copy",
                ckpt_name="x",
                ckpt_range=Range(dim=0, start=0, end=1),
                dst_name="w",
                dst_range=None,
            )
        ],
        expected_src_names={"x"},
        expected_dst_names={"w"},
        tensorcast_slices={},
        src_hull={"x": Range(dim=0, start=0, end=1)},
    )


def test_trace_plan_cache_round_trips_versioned_payload(tmp_path) -> None:
    path = tmp_path / "trace.json"
    trace_plan = _trace_plan()

    write_trace_plan_cache(path, trace_plan)

    assert load_trace_plan_cache(path) == trace_plan
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1


def test_trace_plan_cache_loads_legacy_raw_payload(tmp_path) -> None:
    from tensorcast.serving.builder.trace_ir import trace_plan_to_dict

    path = tmp_path / "legacy.json"
    trace_plan = _trace_plan()
    path.write_text(json.dumps(trace_plan_to_dict(trace_plan)), encoding="utf-8")

    assert load_trace_plan_cache(path) == trace_plan


def test_trace_plan_debug_dump_includes_extra_fields(tmp_path) -> None:
    output = dump_trace_plan_debug(
        _trace_plan(),
        output_dir=tmp_path,
        filename="debug.json",
        cache_path="/tmp/cache.json",
        cache_hit=True,
        trace_cache_schema_version=7,
        extra={"framework": "vllm"},
    )

    assert output == tmp_path / "debug.json"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["framework"] == "vllm"
    assert payload["cache_hit"] is True
    assert payload["trace_cache_schema_version"] == 7
    assert payload["trace_plan_cache_payload_version"] == 1


def test_trace_plan_debug_payload_is_pure_data() -> None:
    payload = trace_plan_debug_payload(
        _trace_plan(),
        cache_path=None,
        cache_hit=False,
        trace_cache_schema_version=8,
    )

    assert payload["cache_path"] is None
    assert payload["expected_src_names"] == ["x"]
