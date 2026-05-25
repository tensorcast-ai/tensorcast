#  Copyright (c) 2026, TensorCast Team.

"""Artifact runtime builder primitives for offline publication workflows."""

from __future__ import annotations

from tensorcast.artifact_runtime.locator import ranked_version_key_for_member
from tensorcast.artifact_runtime.recipe.compiler import (
    CompiledRuntimeRecipe,
    TensorSchemaEntry,
)
from tensorcast.artifact_runtime.recipe.materialization import (
    BindingFinalizeMaterializationResult,
    collect_runtime_tensors_from_model,
    load_source_tensors_for_recipe,
    materialize_binding_finalize_runtime_tensors,
    materialize_pure_transform_runtime_tensors,
    run_binding_finalize_semantic_validation,
    tensorcast_view_slices_from_trace_plan,
    validate_binding_finalize_tensor_schema,
)
from tensorcast.artifact_runtime.recipe.publication import (
    complete_pure_transform_recipe_publication,
)
from tensorcast.artifact_runtime.recipe.validation import (
    validate_recipe_for_builder_mode,
)

LOCAL_READY_BOOTSTRAP_BUILD_PIPELINE_VERSION = "tensorcast-bootstrap-v1"


__all__ = [
    "BindingFinalizeMaterializationResult",
    "CompiledRuntimeRecipe",
    "LOCAL_READY_BOOTSTRAP_BUILD_PIPELINE_VERSION",
    "TensorSchemaEntry",
    "collect_runtime_tensors_from_model",
    "complete_pure_transform_recipe_publication",
    "load_source_tensors_for_recipe",
    "materialize_binding_finalize_runtime_tensors",
    "materialize_pure_transform_runtime_tensors",
    "ranked_version_key_for_member",
    "run_binding_finalize_semantic_validation",
    "tensorcast_view_slices_from_trace_plan",
    "validate_binding_finalize_tensor_schema",
    "validate_recipe_for_builder_mode",
]
