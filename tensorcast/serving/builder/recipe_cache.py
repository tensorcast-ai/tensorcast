#  Copyright (c) 2026, TensorCast Team.
"""CompiledServingRecipe cache helpers."""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tensorcast.serving.builder.compiler import (
    CompiledServingRecipe,
    SourceHullEntry,
    TensorcastLogicalTopology,
    TensorcastSemanticValidationSpec,
    TensorcastServingFacts,
    TensorSchemaEntry,
    binding_realization_plan_proto_bytes,
    compiled_recipe_realization_plan_count,
)
from tensorcast.serving.builder.trace_ir import (
    TracePlan,
    copy_plan_from_dict,
    copy_plan_to_dict,
    range_to_dict,
    single_range_from_dict,
)
from tensorcast.types import FinalizeClass, ServingSupportLevel

RECIPE_CACHE_PAYLOAD_VERSION = 4


def _serving_facts_to_dict(facts: TensorcastServingFacts) -> dict[str, Any]:
    return {
        "framework_name": facts.framework_name,
        "framework_version": facts.framework_version,
        "adapter_version": facts.adapter_version,
        "serving_abi_version": facts.serving_abi_version,
        "support_level": facts.support_level.value,
        "runtime_only_tensor_names": list(facts.runtime_only_tensor_names),
        "process_after_load_class": facts.process_after_load_class.value,
        "post_bind_finalize_class": facts.post_bind_finalize_class.value,
    }


def _serving_facts_from_dict(data: Mapping[str, Any]) -> TensorcastServingFacts:
    return TensorcastServingFacts(
        framework_name=str(data["framework_name"]),
        framework_version=(
            None
            if data.get("framework_version") is None
            else str(data["framework_version"])
        ),
        adapter_version=str(data["adapter_version"]),
        serving_abi_version=str(data["serving_abi_version"]),
        support_level=ServingSupportLevel(str(data["support_level"])),
        runtime_only_tensor_names=tuple(
            str(name) for name in data.get("runtime_only_tensor_names", ())
        ),
        process_after_load_class=FinalizeClass(str(data["process_after_load_class"])),
        post_bind_finalize_class=FinalizeClass(str(data["post_bind_finalize_class"])),
    )


def _tensor_schema_to_dict(entry: TensorSchemaEntry) -> dict[str, Any]:
    return {
        "name": entry.name,
        "dtype": entry.dtype,
        "shape": list(entry.shape),
        "stride": list(entry.stride),
    }


def _tensor_schema_from_dict(data: Mapping[str, Any]) -> TensorSchemaEntry:
    return TensorSchemaEntry(
        name=str(data["name"]),
        dtype=str(data["dtype"]),
        shape=tuple(int(dim) for dim in data["shape"]),
        stride=tuple(int(dim) for dim in data["stride"]),
    )


def _source_hull_to_dict(entry: SourceHullEntry) -> dict[str, Any]:
    return {
        "name": entry.name,
        "range": {
            "dim": entry.range.dim,
            "start": entry.range.start,
            "end": entry.range.end,
        },
    }


def _source_hull_from_dict(data: Mapping[str, Any]) -> SourceHullEntry:
    return SourceHullEntry(
        name=str(data["name"]),
        range=single_range_from_dict(dict(data["range"])),
    )


def _trace_plan_summary_to_dict(trace_plan: TracePlan) -> dict[str, Any]:
    return {
        "expected_src_names": sorted(trace_plan.expected_src_names),
        "expected_dst_names": sorted(trace_plan.expected_dst_names),
        "tensorcast_slices": {
            name: range_to_dict(rng)
            for name, rng in trace_plan.tensorcast_slices.items()
        },
        "src_hull": {
            name: range_to_dict(rng) for name, rng in trace_plan.src_hull.items()
        },
    }


def _trace_plan_summary_from_dict(data: Mapping[str, Any]) -> TracePlan:
    return TracePlan(
        copy_plan=[],
        expected_src_names={str(name) for name in data["expected_src_names"]},
        expected_dst_names={str(name) for name in data["expected_dst_names"]},
        tensorcast_slices={
            str(name): single_range_from_dict(dict(rng))
            for name, rng in data["tensorcast_slices"].items()
        },
        src_hull={
            str(name): single_range_from_dict(dict(rng))
            for name, rng in data["src_hull"].items()
        },
    )


def _bytes_to_base64_payload(value: bytes) -> dict[str, str]:
    return {
        "encoding": "base64",
        "data": base64.b64encode(value).decode("ascii"),
    }


def _bytes_from_base64_payload(value: Any, *, field: str) -> bytes:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a base64 payload")
    if value.get("encoding") != "base64":
        raise ValueError(f"unsupported {field} encoding")
    return base64.b64decode(str(value["data"]).encode("ascii"))


def _logical_topology_to_dict(
    topology: TensorcastLogicalTopology | None,
) -> dict[str, Any] | None:
    if topology is None:
        return None
    return {
        "tensor_parallel_rank": topology.tensor_parallel_rank,
        "tensor_parallel_world_size": topology.tensor_parallel_world_size,
    }


def _logical_topology_from_dict(
    data: Mapping[str, Any] | None,
) -> TensorcastLogicalTopology | None:
    if data is None:
        return None
    return TensorcastLogicalTopology(
        tensor_parallel_rank=int(data["tensor_parallel_rank"]),
        tensor_parallel_world_size=int(data["tensor_parallel_world_size"]),
    )


def _semantic_validation_spec_to_dict(
    spec: TensorcastSemanticValidationSpec,
) -> dict[str, Any]:
    return {
        "kind": spec.kind,
        "payload": spec.payload,
    }


def _semantic_validation_spec_from_dict(
    data: Mapping[str, Any],
) -> TensorcastSemanticValidationSpec:
    return TensorcastSemanticValidationSpec(
        kind=str(data.get("kind", "none")),
        payload=data.get("payload"),
    )


def compiled_recipe_to_dict(recipe: CompiledServingRecipe) -> dict[str, Any]:
    realization_plan_proto = bytes(recipe.realization_plan_proto or b"")
    if not realization_plan_proto and recipe.realization_plan:
        realization_plan_proto = binding_realization_plan_proto_bytes(
            recipe.realization_plan,
            recipe.tensor_schema,
        )
    return {
        "compile_key": recipe.compile_key,
        "source_artifact_ref": recipe.source_artifact_ref,
        "source_metadata_fingerprint": recipe.source_metadata_fingerprint,
        "serving_facts": _serving_facts_to_dict(recipe.serving_facts),
        "trace_plan_summary": _trace_plan_summary_to_dict(recipe.trace_plan),
        "tensor_schema": [
            _tensor_schema_to_dict(entry) for entry in recipe.tensor_schema
        ],
        "source_hull": [_source_hull_to_dict(entry) for entry in recipe.source_hull],
        "realization_plan_proto": _bytes_to_base64_payload(realization_plan_proto),
        "realization_plan_count": compiled_recipe_realization_plan_count(recipe),
        "realization_fallback_plan": [
            copy_plan_to_dict(entry) for entry in recipe.realization_fallback_plan
        ],
        "logical_topology": _logical_topology_to_dict(recipe.logical_topology),
        "semantic_validation_spec": _semantic_validation_spec_to_dict(
            recipe.semantic_validation_spec
        ),
    }


def compiled_recipe_from_dict(data: Mapping[str, Any]) -> CompiledServingRecipe:
    realization_plan_proto = _bytes_from_base64_payload(
        data["realization_plan_proto"],
        field="realization_plan_proto",
    )
    return CompiledServingRecipe(
        compile_key=str(data["compile_key"]),
        source_artifact_ref=str(data["source_artifact_ref"]),
        source_metadata_fingerprint=str(data["source_metadata_fingerprint"]),
        serving_facts=_serving_facts_from_dict(data["serving_facts"]),
        trace_plan=_trace_plan_summary_from_dict(data["trace_plan_summary"]),
        tensor_schema=tuple(
            _tensor_schema_from_dict(entry) for entry in data["tensor_schema"]
        ),
        source_hull=tuple(
            _source_hull_from_dict(entry) for entry in data["source_hull"]
        ),
        realization_plan=(),
        realization_fallback_plan=tuple(
            copy_plan_from_dict(dict(entry))
            for entry in data["realization_fallback_plan"]
        ),
        logical_topology=_logical_topology_from_dict(data["logical_topology"]),
        semantic_validation_spec=_semantic_validation_spec_from_dict(
            data["semantic_validation_spec"]
        ),
        realization_plan_proto=realization_plan_proto,
        realization_plan_count=int(data["realization_plan_count"]),
    )


def load_compiled_recipe_cache(
    cache_path: str | os.PathLike[str] | None,
) -> CompiledServingRecipe | None:
    if not cache_path:
        return None
    path = Path(cache_path)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, Mapping):
            return None
        version = int(data.get("version", 0) or 0)
        if version != RECIPE_CACHE_PAYLOAD_VERSION:
            return None
        return compiled_recipe_from_dict(data["compiled_recipe"])
    except Exception:
        return None


def write_compiled_recipe_cache(
    cache_path: str | os.PathLike[str],
    recipe: CompiledServingRecipe,
) -> None:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": RECIPE_CACHE_PAYLOAD_VERSION,
        "compiled_recipe": compiled_recipe_to_dict(recipe),
    }
    tmp_path = path.parent / f".{path.name}.tmp.{os.getpid()}"
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


__all__ = [
    "RECIPE_CACHE_PAYLOAD_VERSION",
    "compiled_recipe_from_dict",
    "compiled_recipe_to_dict",
    "load_compiled_recipe_cache",
    "write_compiled_recipe_cache",
]
