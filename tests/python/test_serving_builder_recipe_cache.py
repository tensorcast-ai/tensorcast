#  Copyright (c) 2026, TensorCast Team.

import json
from dataclasses import replace
from pathlib import Path

from tensorcast.api.store import BindingRealizationEntry
from tensorcast.api.store import Range as StoreRange
from tensorcast.serving.builder.compiler import (
    CompiledServingRecipe,
    SourceHullEntry,
    TensorcastLogicalTopology,
    TensorcastSemanticValidationSpec,
    TensorcastServingFacts,
    TensorSchemaEntry,
)
from tensorcast.serving.builder.recipe_cache import (
    RECIPE_CACHE_PAYLOAD_VERSION,
    load_compiled_recipe_cache,
    write_compiled_recipe_cache,
)
from tensorcast.serving.builder.trace_ir import CopyPlanEntry, Range, TracePlan
from tensorcast.types import FinalizeClass, ServingSupportLevel


def _recipe() -> CompiledServingRecipe:
    trace_plan = TracePlan(
        copy_plan=[
            CopyPlanEntry(
                op="copy",
                ckpt_name="x",
                ckpt_range=Range(dim=0, start=0, end=4),
                dst_name="w",
                dst_range=None,
            )
        ],
        expected_src_names={"x"},
        expected_dst_names={"w"},
        tensorcast_slices={"x": Range(dim=0, start=0, end=4)},
        src_hull={"x": Range(dim=0, start=0, end=4)},
    )
    return CompiledServingRecipe(
        compile_key="compile-key",
        source_artifact_ref="msa1:test-source",
        source_metadata_fingerprint="metadata-fingerprint",
        serving_facts=TensorcastServingFacts(
            framework_name="vllm",
            framework_version="vllm-test",
            adapter_version="adapter-v1",
            serving_abi_version="abi-v1",
            support_level=ServingSupportLevel.RUNTIME_BIND_SWAP_READY,
            runtime_only_tensor_names=("runtime", ),
            process_after_load_class=FinalizeClass.RUNTIME_ONLY,
            post_bind_finalize_class=FinalizeClass.RUNTIME_ONLY,
        ),
        trace_plan=trace_plan,
        tensor_schema=(TensorSchemaEntry(
            name="w",
            dtype="torch.float16",
            shape=(4, ),
            stride=(1, ),
        ), ),
        source_hull=(SourceHullEntry(name="x",
                                     range=Range(dim=0, start=0, end=4)), ),
        realization_plan=(BindingRealizationEntry(
            op="copy",
            source_name="x",
            source_ranges=(StoreRange(dim=0, start=0, end=4), ),
            dst_name="w",
            dst_ranges=(),
        ), ),
        realization_fallback_plan=(),
        logical_topology=TensorcastLogicalTopology(
            tensor_parallel_rank=0,
            tensor_parallel_world_size=2,
        ),
        semantic_validation_spec=TensorcastSemanticValidationSpec(
            kind="explicit",
            payload={"probe": "ok"},
        ),
    )


def test_compiled_recipe_cache_round_trips(tmp_path: Path) -> None:
    cache_path = tmp_path / "recipe.json"
    recipe = _recipe()

    write_compiled_recipe_cache(cache_path, recipe)

    loaded = load_compiled_recipe_cache(cache_path)

    assert loaded == replace(
        recipe,
        trace_plan=replace(recipe.trace_plan, copy_plan=[]),
        realization_plan=(),
        realization_plan_proto=loaded.realization_plan_proto,
        realization_plan_count=1,
    )
    assert loaded.realization_plan_proto
    assert loaded.realization_plan_count == len(recipe.realization_plan)
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["version"] == RECIPE_CACHE_PAYLOAD_VERSION
    assert "realization_plan" not in payload["compiled_recipe"]
    assert "realization_plan_proto" in payload["compiled_recipe"]
    assert "trace_plan" not in payload["compiled_recipe"]
    assert payload["compiled_recipe"]["trace_plan_summary"][
        "expected_dst_names"] == ["w"]
    assert payload["compiled_recipe"]["serving_facts"][
        "framework_version"] == "vllm-test"


def test_compiled_recipe_cache_ignores_missing_and_unknown_version(
    tmp_path: Path, ) -> None:
    missing = tmp_path / "missing.json"
    assert load_compiled_recipe_cache(missing) is None

    bad_version = tmp_path / "bad_version.json"
    bad_version.write_text('{"version": 999, "compiled_recipe": {}}',
                           encoding="utf-8")

    assert load_compiled_recipe_cache(bad_version) is None
