#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest
import torch

from tensorcast.artifact_runtime.recipe.compiler import (
    CompiledRuntimeRecipe,
    TensorcastRuntimeFacts,
    TensorcastSemanticValidationSpec,
    TensorSchemaEntry,
)
from tensorcast.artifact_runtime.recipe.semantic_validation import (
    evaluate_semantic_validation_spec,
)
from tensorcast.artifact_runtime.recipe.tensor_schema import (
    validate_tensor_schema_against_tensors,
)
from tensorcast.artifact_runtime.recipe.trace_ir import TracePlan
from tensorcast.artifact_runtime.recipe.validation import (
    validate_recipe_for_builder_mode,
)
from tensorcast.types import BuilderMode, FinalizeClass, ServingSupportLevel


def _recipe() -> CompiledRuntimeRecipe:
    return CompiledRuntimeRecipe(
        compile_key="compile-key",
        source_artifact_ref="mi2:test:source",
        source_metadata_fingerprint="metadata-fingerprint",
        runtime_facts=TensorcastRuntimeFacts(
            framework_name="vllm",
            adapter_version="adapter-v1",
            serving_abi_version="abi-v1",
            support_level=ServingSupportLevel.BUILDER_PUBLICATION_READY,
            runtime_only_tensor_names=("runtime_only",),
            process_after_load_class=FinalizeClass.RUNTIME_ONLY,
            post_bind_finalize_class=FinalizeClass.RUNTIME_ONLY,
        ),
        trace_plan=TracePlan(
            copy_plan=[],
            expected_src_names=set(),
            expected_dst_names=set(),
            tensorcast_slices={},
            src_hull={},
        ),
        tensor_schema=(),
        source_hull=(),
        realization_plan=(),
        realization_fallback_plan=(),
        topology_ref=None,
        member_ref=None,
        semantic_validation_spec=TensorcastSemanticValidationSpec.empty(),
    )


def test_validate_recipe_for_builder_mode_accepts_pure_transform() -> None:
    assert (
        validate_recipe_for_builder_mode(
            _recipe(),
            BuilderMode.PURE_TRANSFORM,
        )
        == _recipe()
    )


def test_validate_recipe_for_builder_mode_rejects_binding_finalize_fact_mismatch() -> (
    None
):
    recipe = _recipe()

    with pytest.raises(ValueError, match="representation_changing"):
        validate_recipe_for_builder_mode(recipe, BuilderMode.BINDING_FINALIZE)


def test_validate_recipe_for_builder_mode_rejects_non_publication_ready() -> None:
    recipe = replace(
        _recipe(),
        runtime_facts=replace(
            _recipe().runtime_facts,
            support_level=ServingSupportLevel.SOURCE_BIND_BOOTSTRAP_ONLY,
        ),
    )

    with pytest.raises(ValueError, match="publication-ready"):
        validate_recipe_for_builder_mode(recipe, BuilderMode.PURE_TRANSFORM)


@dataclass(frozen=True)
class _ProbePayload:
    values: tuple[int, ...]


def test_evaluate_semantic_validation_spec_normalizes_and_compares_payload() -> None:
    spec = TensorcastSemanticValidationSpec(
        kind="explicit",
        payload={"values": [1, 2, 3]},
    )

    assert evaluate_semantic_validation_spec(
        spec,
        _ProbePayload(values=(1, 2, 3)),
    ) == {"values": [1, 2, 3]}


def test_evaluate_semantic_validation_spec_rejects_explicit_mismatch() -> None:
    spec = TensorcastSemanticValidationSpec(
        kind="explicit",
        payload={"values": [1, 2, 3]},
    )

    with pytest.raises(RuntimeError, match="semantic validation failed"):
        evaluate_semantic_validation_spec(spec, {"values": [3, 2, 1]})


def test_validate_tensor_schema_against_tensors_checks_names_shape_stride_dtype() -> (
    None
):
    schema = (
        TensorSchemaEntry(
            name="w",
            dtype=str(torch.float16),
            shape=(2, 3),
            stride=(3, 1),
        ),
    )

    validate_tensor_schema_against_tensors(
        schema,
        {"w": torch.empty((2, 3), dtype=torch.float16)},
    )

    with pytest.raises(RuntimeError, match="shape mismatch"):
        validate_tensor_schema_against_tensors(
            schema,
            {"w": torch.empty((3, 2), dtype=torch.float16)},
        )
