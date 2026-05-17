#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tensorcast.serving.builder.compiler import (
    RecipeCompileIdentity,
    RecipeCompileInputs,
    TensorcastLogicalTopology,
    TensorcastSemanticValidationSpec,
    TensorcastServingFacts,
    TensorSchemaEntry,
    compile_serving_recipe,
)
from tensorcast.serving.builder.source_catalog import SourceCatalog, SourceTensorMeta
from tensorcast.serving.builder.trace_ir import CopyPlanEntry, Range, TracePlan
from tensorcast.types import FinalizeClass, ServingSupportLevel


def _source_catalog(source_artifact_ref: str = "mi2:test:source") -> SourceCatalog:
    return SourceCatalog(
        ordered_names=("x",),
        meta_by_name={
            "x": SourceTensorMeta(
                dtype=torch.float16,
                shape=(4,),
                stride=(1,),
                storage_offset=0,
            )
        },
        selected_files=(),
        source_artifact_ref=source_artifact_ref,
        canonical_index_hash="index-hash",
        metadata_fingerprint="metadata-fingerprint",
        canonical_index_bytes=b"index",
    )


def _trace_plan() -> TracePlan:
    return TracePlan(
        copy_plan=[
            CopyPlanEntry(
                op="copy",
                ckpt_name="x",
                ckpt_range=None,
                dst_name="w",
                dst_range=None,
            )
        ],
        expected_src_names={"x"},
        expected_dst_names={"w"},
        tensorcast_slices={},
        src_hull={"x": Range(dim=0, start=0, end=4)},
    )


def _serving_facts(adapter_version: str = "adapter-v1") -> TensorcastServingFacts:
    return TensorcastServingFacts(
        framework_name="vllm",
        framework_version="vllm-test",
        adapter_version=adapter_version,
        serving_abi_version="abi-v1",
        support_level=ServingSupportLevel.BUILDER_PUBLICATION_READY,
        runtime_only_tensor_names=("runtime_only",),
        process_after_load_class=FinalizeClass.RUNTIME_ONLY,
        post_bind_finalize_class=FinalizeClass.RUNTIME_ONLY,
    )


def _identity() -> RecipeCompileIdentity:
    return RecipeCompileIdentity(
        model_id="fake-model",
        model_revision=None,
        dtype="torch.float16",
        model_hash="model-hash",
        framework_name="vllm",
        adapter_version="adapter-v1",
        serving_abi_version="abi-v1",
        framework_version="vllm-test",
        trace_cache_schema_version=7,
        logical_topology=TensorcastLogicalTopology(
            tensor_parallel_rank=0,
            tensor_parallel_world_size=1,
        ),
    )


def _inputs(**overrides) -> RecipeCompileInputs:
    values = {
        "source_catalog": _source_catalog(),
        "trace_plan": _trace_plan(),
        "serving_facts": _serving_facts(),
        "tensor_schema": (
            TensorSchemaEntry(
                name="w",
                dtype="torch.float16",
                shape=(4,),
                stride=(1,),
            ),
            TensorSchemaEntry(
                name="runtime_buffer",
                dtype="torch.float16",
                shape=(1,),
                stride=(1,),
            ),
        ),
        "semantic_validation_spec": TensorcastSemanticValidationSpec.empty(),
    }
    values.update(overrides)
    return RecipeCompileInputs(**values)


class _Observer:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def event(self, name: str, payload) -> None:
        self.events.append((name, dict(payload)))


def test_compile_serving_recipe_assembles_recipe_from_pure_inputs() -> None:
    observer = _Observer()

    recipe = compile_serving_recipe(
        identity=_identity(),
        inputs=_inputs(),
        observer=observer,
    )

    assert recipe.compile_key
    assert recipe.source_artifact_ref == "mi2:test:source"
    assert recipe.source_metadata_fingerprint == "metadata-fingerprint"
    assert recipe.serving_facts.framework_name == "vllm"
    assert recipe.trace_plan.expected_src_names == {"x"}
    assert [entry.name for entry in recipe.tensor_schema] == ["w"]
    assert [entry.name for entry in recipe.source_hull] == ["x"]
    assert recipe.logical_topology == TensorcastLogicalTopology(
        tensor_parallel_rank=0,
        tensor_parallel_world_size=1,
    )
    assert observer.events == [
        (
            "recipe_compiler.compile",
            {
                "compile_key": recipe.compile_key,
                "source_artifact_ref": "mi2:test:source",
                "source_metadata_fingerprint": "metadata-fingerprint",
                "tensor_schema_count": 1,
                "copy_plan_count": 1,
                "expected_src_count": 1,
                "expected_dst_count": 1,
                "tensorcast_slice_count": 0,
                "realization_plan_count": 1,
                "realization_fallback_count": 0,
            },
        )
    ]


def test_compile_serving_recipe_compile_key_invalidates_on_pure_inputs() -> None:
    recipe_a = compile_serving_recipe(identity=_identity(), inputs=_inputs())
    recipe_b = compile_serving_recipe(
        identity=replace(_identity(), adapter_version="adapter-v2"),
        inputs=_inputs(serving_facts=_serving_facts("adapter-v2")),
    )
    recipe_c = compile_serving_recipe(
        identity=replace(
            _identity(),
            logical_topology=TensorcastLogicalTopology(
                tensor_parallel_rank=1,
                tensor_parallel_world_size=2,
            ),
        ),
        inputs=_inputs(),
    )

    assert recipe_a.compile_key != recipe_b.compile_key
    assert recipe_a.compile_key != recipe_c.compile_key


def test_compile_serving_recipe_rejects_identity_fact_mismatch() -> None:
    with pytest.raises(ValueError, match="RecipeCompileIdentity must match"):
        compile_serving_recipe(
            identity=replace(_identity(), adapter_version="adapter-v2"),
            inputs=_inputs(),
        )


def test_compile_serving_recipe_rejects_missing_destination_schema() -> None:
    with pytest.raises(ValueError, match="tensor_schema is missing"):
        compile_serving_recipe(
            identity=_identity(),
            inputs=_inputs(tensor_schema=()),
        )


def test_compile_serving_recipe_rejects_synthetic_source_identity() -> None:
    with pytest.raises(ValueError, match="real imported source artifact"):
        compile_serving_recipe(
            identity=_identity(),
            inputs=_inputs(source_catalog=_source_catalog("disk:/tmp/fake")),
        )
