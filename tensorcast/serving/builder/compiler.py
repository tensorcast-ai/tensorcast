#  Copyright (c) 2026, TensorCast Team.

"""Framework-neutral serving recipe compiler primitives."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from tensorcast.serving.builder.trace_ir import Range, TracePlan
from tensorcast.types import FinalizeClass, ServingSupportLevel


@dataclass(frozen=True)
class TensorcastServingFacts:
    framework_name: str
    adapter_version: str
    serving_abi_version: str
    support_level: ServingSupportLevel
    runtime_only_tensor_names: tuple[str, ...]
    process_after_load_class: FinalizeClass
    post_bind_finalize_class: FinalizeClass


ServingFacts = TensorcastServingFacts


@dataclass(frozen=True)
class TensorSchemaEntry:
    name: str
    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]


@dataclass(frozen=True)
class SourceHullEntry:
    name: str
    range: Range


@dataclass(frozen=True)
class TensorcastLogicalTopology:
    tensor_parallel_rank: int
    tensor_parallel_world_size: int


@dataclass(frozen=True)
class TensorcastSemanticValidationSpec:
    kind: str = "none"
    payload: Any = None

    @classmethod
    def empty(cls) -> TensorcastSemanticValidationSpec:
        return cls(kind="none", payload=None)


SemanticValidationSpec = TensorcastSemanticValidationSpec


@dataclass(frozen=True)
class CompiledServingRecipe:
    compile_key: str
    source_artifact_ref: str
    source_metadata_fingerprint: str
    serving_facts: TensorcastServingFacts
    trace_plan: TracePlan
    tensor_schema: tuple[TensorSchemaEntry, ...]
    source_hull: tuple[SourceHullEntry, ...]
    logical_topology: TensorcastLogicalTopology | None
    semantic_validation_spec: TensorcastSemanticValidationSpec


@dataclass(frozen=True)
class RecipeCompileIdentity:
    model_id: str
    model_revision: str | None
    dtype: str
    framework_name: str
    adapter_version: str
    serving_abi_version: str
    trace_cache_schema_version: int
    model_hash: str | None = None
    framework_version: str | None = None
    logical_topology: TensorcastLogicalTopology | None = None
    topology_ref: Any | None = None
    member_ref: Any | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecipeCompileInputs:
    source_catalog: Any
    trace_plan: TracePlan
    serving_facts: TensorcastServingFacts
    tensor_schema: tuple[TensorSchemaEntry, ...]
    semantic_validation_spec: TensorcastSemanticValidationSpec = field(
        default_factory=TensorcastSemanticValidationSpec.empty
    )


class ServingBuildObserver(Protocol):
    def event(self, name: str, payload: Mapping[str, object]) -> None: ...


def compile_serving_recipe(
    *,
    identity: RecipeCompileIdentity,
    inputs: RecipeCompileInputs,
    observer: ServingBuildObserver | None = None,
) -> CompiledServingRecipe:
    """Assemble a serving recipe from framework-collected pure inputs."""

    source_artifact_ref = _resolve_source_artifact_ref(
        inputs.source_catalog.source_artifact_ref
    )
    source_metadata_fingerprint = str(inputs.source_catalog.metadata_fingerprint)
    tensor_schema = filter_tensor_schema_for_trace_plan(
        inputs.tensor_schema,
        inputs.trace_plan,
    )
    compile_key = compute_recipe_compile_key(
        identity=identity,
        source_artifact_ref=source_artifact_ref,
        source_metadata_fingerprint=source_metadata_fingerprint,
        serving_facts=inputs.serving_facts,
        tensor_schema=tensor_schema,
        semantic_validation_spec=inputs.semantic_validation_spec,
    )
    recipe = CompiledServingRecipe(
        compile_key=compile_key,
        source_artifact_ref=source_artifact_ref,
        source_metadata_fingerprint=source_metadata_fingerprint,
        serving_facts=inputs.serving_facts,
        trace_plan=inputs.trace_plan,
        tensor_schema=tensor_schema,
        source_hull=tuple(
            SourceHullEntry(name=name, range=rng)
            for name, rng in sorted(inputs.trace_plan.src_hull.items())
        ),
        logical_topology=identity.logical_topology,
        semantic_validation_spec=inputs.semantic_validation_spec,
    )
    if observer is not None:
        observer.event(
            "recipe_compiler.compile",
            {
                "compile_key": recipe.compile_key,
                "source_artifact_ref": recipe.source_artifact_ref,
                "source_metadata_fingerprint": recipe.source_metadata_fingerprint,
                "tensor_schema_count": len(recipe.tensor_schema),
                "copy_plan_count": len(recipe.trace_plan.copy_plan),
                "expected_src_count": len(recipe.trace_plan.expected_src_names),
                "expected_dst_count": len(recipe.trace_plan.expected_dst_names),
                "tensorcast_slice_count": len(recipe.trace_plan.tensorcast_slices),
            },
        )
    return recipe


def filter_tensor_schema_for_trace_plan(
    tensor_schema: Sequence[TensorSchemaEntry],
    trace_plan: TracePlan,
) -> tuple[TensorSchemaEntry, ...]:
    expected = set(trace_plan.expected_dst_names)
    return tuple(entry for entry in tensor_schema if entry.name in expected)


def compute_recipe_compile_key(
    *,
    identity: RecipeCompileIdentity,
    source_artifact_ref: str,
    source_metadata_fingerprint: str,
    serving_facts: TensorcastServingFacts,
    tensor_schema: Sequence[TensorSchemaEntry],
    semantic_validation_spec: TensorcastSemanticValidationSpec,
) -> str:
    payload = {
        "model_hash": identity.model_hash or identity.model_id,
        "model": identity.model_id,
        "revision": identity.model_revision,
        "dtype": identity.dtype,
        "source_artifact_ref": source_artifact_ref,
        "metadata_fingerprint": source_metadata_fingerprint,
        "framework_name": serving_facts.framework_name,
        "adapter_version": serving_facts.adapter_version,
        "serving_abi_version": serving_facts.serving_abi_version,
        "support_level": str(serving_facts.support_level),
        "runtime_only_tensor_names": list(serving_facts.runtime_only_tensor_names),
        "process_after_load_class": str(serving_facts.process_after_load_class),
        "post_bind_finalize_class": str(serving_facts.post_bind_finalize_class),
        "tensor_schema": [
            {
                "name": item.name,
                "dtype": item.dtype,
                "shape": list(item.shape),
                "stride": list(item.stride),
            }
            for item in tensor_schema
        ],
        "logical_topology": None
        if identity.logical_topology is None
        else {
            "tensor_parallel_rank": identity.logical_topology.tensor_parallel_rank,
            "tensor_parallel_world_size": identity.logical_topology.tensor_parallel_world_size,
        },
        "topology_ref": _jsonable(identity.topology_ref),
        "member_ref": _jsonable(identity.member_ref),
        "semantic_validation_spec": _jsonable_semantic_validation_spec(
            semantic_validation_spec
        ),
        "trace_cache_schema_version": identity.trace_cache_schema_version,
        "framework_version": identity.framework_version,
        "version": identity.framework_version,
        "identity_extra": _jsonable(identity.extra),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _resolve_source_artifact_ref(source_artifact_ref: str) -> str:
    normalized_ref = str(source_artifact_ref).strip()
    if not normalized_ref:
        raise ValueError(
            "Tensorcast bootstrap requires a real imported source artifact identity"
        )
    if normalized_ref.startswith(("disk:", "key:")):
        raise ValueError(
            "Tensorcast bootstrap requires a real imported source artifact "
            "identity, not a synthetic disk/key ref"
        )
    return normalized_ref


def _jsonable_semantic_validation_spec(
    spec: TensorcastSemanticValidationSpec,
) -> dict[str, Any]:
    return {
        "kind": spec.kind,
        "payload": _jsonable(spec.payload),
    }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {key: _jsonable(item) for key, item in dataclasses.asdict(value).items()}
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _jsonable(value.model_dump())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_jsonable(item) for item in value]
    return repr(value)


__all__ = [
    "CompiledServingRecipe",
    "RecipeCompileIdentity",
    "RecipeCompileInputs",
    "SemanticValidationSpec",
    "ServingBuildObserver",
    "ServingFacts",
    "SourceHullEntry",
    "TensorSchemaEntry",
    "TensorcastLogicalTopology",
    "TensorcastSemanticValidationSpec",
    "TensorcastServingFacts",
    "compile_serving_recipe",
    "compute_recipe_compile_key",
    "filter_tensor_schema_for_trace_plan",
]
