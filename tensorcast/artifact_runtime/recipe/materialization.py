#  Copyright (c) 2026, TensorCast Team.

"""Framework-neutral runtime tensor-dict materialization helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from tensorcast.artifact_runtime.recipe.semantic_validation import (
    evaluate_semantic_validation_spec,
)
from tensorcast.artifact_runtime.recipe.tensor_schema import (
    validate_tensor_schema_against_tensors,
)
from tensorcast.artifact_runtime.recipe.trace_ir import (
    CopyPlanEntry,
    MultiRange,
    Range,
    RangeSpec,
    TracePlan,
)
from tensorcast.artifact_runtime.recipe.validation import (
    validate_recipe_for_builder_mode,
)
from tensorcast.pytorch.module_binding import (
    attach_tensors_to_module,
    collect_module_tensors,
)
from tensorcast.types import BuilderMode


@dataclass(frozen=True)
class BindingFinalizeMaterializationResult:
    model: nn.Module
    runtime_tensors: dict[str, torch.Tensor]
    semantic_probe_result: Any = None


def tensorcast_view_slices_from_trace_plan(
    trace_plan: TracePlan,
) -> dict[str, list[tuple[int, slice]]]:
    return {
        name: [(rng.dim, slice(rng.start, rng.end))]
        for name, rng in trace_plan.tensorcast_slices.items()
    }


def dtype_from_string(dtype_name: str) -> torch.dtype:
    normalized = str(dtype_name).strip()
    if normalized.startswith("torch."):
        normalized = normalized.split(".", 1)[1]
    dtype = getattr(torch, normalized, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported torch dtype string: {dtype_name!r}")
    return dtype


def allocate_tensors_from_schema(
    tensor_schema: Iterable[Any],
    *,
    target_device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        entry.name: torch.empty_strided(
            size=tuple(int(dim) for dim in entry.shape),
            stride=tuple(int(dim) for dim in entry.stride),
            dtype=dtype_from_string(entry.dtype),
            device=target_device,
        )
        for entry in tensor_schema
    }


def validate_source_tensor_names(
    trace_plan: TracePlan,
    source_tensors: Mapping[str, torch.Tensor],
) -> None:
    expected = set(trace_plan.expected_src_names)
    provided = set(source_tensors)
    missing = expected - provided
    unexpected = provided - expected
    if missing or unexpected:
        raise RuntimeError(
            "TensorCast source coverage mismatch. "
            f"Missing={sorted(missing)}, Unexpected={sorted(unexpected)}"
        )


def iter_ranges(range_spec: RangeSpec) -> tuple[Range, ...]:
    if isinstance(range_spec, Range):
        return (range_spec,)
    return range_spec.ranges


def narrow_by_range_spec(
    tensor: torch.Tensor,
    range_spec: RangeSpec,
) -> torch.Tensor:
    view = tensor
    for rng in iter_ranges(range_spec):
        view = view.narrow(rng.dim, rng.start, rng.end - rng.start)
    return view


def narrow_source_view(
    src_base: torch.Tensor,
    ckpt_name: str,
    ckpt_range: RangeSpec | None,
    src_hull: dict[str, Range],
) -> torch.Tensor:
    if ckpt_range is None:
        return src_base
    if isinstance(ckpt_range, Range):
        hull = src_hull.get(ckpt_name)
        if hull is not None:
            if hull.dim != ckpt_range.dim:
                raise RuntimeError(f"Slice dim mismatch for {ckpt_name}")
            rel_start = ckpt_range.start - hull.start
            return src_base.narrow(
                hull.dim,
                rel_start,
                ckpt_range.end - ckpt_range.start,
            )
    return narrow_by_range_spec(src_base, ckpt_range)


def apply_copy_plan(
    trace_plan: TracePlan,
    source_tensors: Mapping[str, torch.Tensor],
    runtime_tensors: Mapping[str, torch.Tensor],
    *,
    entries: Iterable[CopyPlanEntry] | None = None,
) -> None:
    for entry in trace_plan.copy_plan if entries is None else entries:
        dst_base = runtime_tensors.get(entry.dst_name)
        if dst_base is None:
            raise RuntimeError(f"Missing destination tensor {entry.dst_name}")
        dst_view = (
            dst_base
            if entry.dst_range is None
            else narrow_by_range_spec(dst_base, entry.dst_range)
        )

        if entry.op == "copy":
            if entry.ckpt_name is None:
                raise RuntimeError(
                    f"Missing source name for copy into {entry.dst_name}"
                )
            src_base = source_tensors.get(entry.ckpt_name)
            if src_base is None:
                raise RuntimeError(f"Missing source tensor {entry.ckpt_name}")
            src_view = narrow_source_view(
                src_base,
                entry.ckpt_name,
                entry.ckpt_range,
                trace_plan.src_hull,
            )
            if src_view.ndim == 0 and dst_view.numel() == 1:
                src_view = src_view.reshape(dst_view.shape)
            if tuple(src_view.shape) != tuple(dst_view.shape):
                raise RuntimeError(
                    f"Copy shape mismatch for {entry.ckpt_name} -> "
                    f"{entry.dst_name}: {tuple(src_view.shape)} vs "
                    f"{tuple(dst_view.shape)}"
                )
            dst_view.copy_(src_view)
            continue

        if entry.op == "fill":
            if entry.fill_value is not None:
                fill_value = entry.fill_value
            else:
                if entry.ckpt_name is None:
                    raise RuntimeError(f"Missing fill source for {entry.dst_name}")
                src_base = source_tensors.get(entry.ckpt_name)
                if src_base is None:
                    raise RuntimeError(f"Missing source tensor {entry.ckpt_name}")
                src_view = narrow_source_view(
                    src_base,
                    entry.ckpt_name,
                    entry.ckpt_range,
                    trace_plan.src_hull,
                )
                if src_view.numel() != 1:
                    raise RuntimeError(
                        f"Fill source {entry.ckpt_name} is not scalar: "
                        f"numel={src_view.numel()}"
                    )
                fill_value = src_view.reshape(()).item()
            dst_view.fill_(fill_value)
            continue

        raise RuntimeError(
            f"Unknown TensorCast plan op {entry.op} for {entry.dst_name}"
        )


def update_dst_coverage(
    coverage: dict[str, dict[str, Any]],
    entry: CopyPlanEntry,
    dst_base: torch.Tensor,
) -> None:
    name = entry.dst_name
    if entry.dst_range is None:
        coverage[name] = {
            "full": True,
            "dim": None,
            "intervals": None,
            "extent": None,
        }
        return
    if isinstance(entry.dst_range, MultiRange):
        raise RuntimeError(
            "TensorCast destination coverage validation does not yet support "
            f"MultiRange writes for {name}"
        )
    if coverage.get(name, {}).get("full"):
        return
    if int(entry.dst_range.start) >= int(entry.dst_range.end):
        return
    dim = entry.dst_range.dim
    extent = dst_base.shape[dim]
    if name not in coverage or coverage[name].get("full"):
        coverage[name] = {
            "full": False,
            "dim": dim,
            "intervals": [],
            "extent": extent,
        }
    if coverage[name]["dim"] != dim:
        raise RuntimeError(f"Multiple slice dims for dst tensor {name}")
    coverage[name]["intervals"].append((entry.dst_range.start, entry.dst_range.end))


def validate_dst_coverage(
    trace_plan: TracePlan,
    runtime_tensors: Mapping[str, torch.Tensor],
) -> None:
    expected = set(trace_plan.expected_dst_names)
    provided = set(runtime_tensors)
    missing = expected - provided
    unexpected = provided - expected
    if missing or unexpected:
        raise RuntimeError(
            "TensorCast destination coverage mismatch. "
            f"Missing={sorted(missing)}, Unexpected={sorted(unexpected)}"
        )

    coverage: dict[str, dict[str, Any]] = {}
    for entry in trace_plan.copy_plan:
        dst_base = runtime_tensors.get(entry.dst_name)
        if dst_base is None:
            continue
        update_dst_coverage(coverage, entry, dst_base)

    for name in sorted(expected):
        info = coverage.get(name)
        if info is None:
            raise RuntimeError(f"No data written to {name}")
        if info.get("full"):
            continue
        intervals = sorted(info["intervals"])
        cursor = 0
        for start, end in intervals:
            if start > cursor:
                raise RuntimeError(
                    f"Destination tensor {name} has coverage gap "
                    f"[{cursor}, {start}) on dim {info['dim']}"
                )
            cursor = max(cursor, end)
        if cursor < info["extent"]:
            raise RuntimeError(
                f"Destination tensor {name} has coverage gap "
                f"[{cursor}, {info['extent']}) on dim {info['dim']}"
            )


def load_source_tensors_for_recipe(
    recipe: Any,
    source_artifact: Any,
    *,
    target_device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    artifact_view = source_artifact.subset(
        sorted(recipe.trace_plan.expected_src_names)
    ).view(slices=tensorcast_view_slices_from_trace_plan(recipe.trace_plan))
    return {
        str(name): tensor
        for name, tensor in artifact_view.tensor_dict(
            device=str(torch.device(target_device))
        ).items()
    }


def materialize_pure_transform_runtime_tensors(
    recipe: Any,
    source_tensors: Mapping[str, torch.Tensor],
    *,
    target_device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    validate_recipe_for_builder_mode(recipe, BuilderMode.PURE_TRANSFORM)
    return materialize_recipe_copy_plan_tensors(
        recipe,
        source_tensors,
        target_device=target_device,
    )


def materialize_recipe_copy_plan_tensors(
    recipe: Any,
    source_tensors: Mapping[str, torch.Tensor],
    *,
    target_device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    resolved_source_tensors = {
        str(name): tensor for name, tensor in dict(source_tensors).items()
    }
    validate_source_tensor_names(recipe.trace_plan, resolved_source_tensors)
    runtime_tensors = allocate_tensors_from_schema(
        recipe.tensor_schema,
        target_device=torch.device(target_device),
    )
    with torch.no_grad():
        apply_copy_plan(
            recipe.trace_plan,
            resolved_source_tensors,
            runtime_tensors,
        )
    validate_dst_coverage(recipe.trace_plan, runtime_tensors)
    return runtime_tensors


def materialize_binding_finalize_runtime_tensors(
    recipe: Any,
    source_tensors: Mapping[str, torch.Tensor],
    *,
    model_config: Any,
    framework_adapter: Any,
    build_runtime_model: Callable[[torch.device], nn.Module],
    target_device: str | torch.device = "cpu",
) -> BindingFinalizeMaterializationResult:
    validate_recipe_for_builder_mode(recipe, BuilderMode.BINDING_FINALIZE)
    resolved_target_device = torch.device(target_device)
    runtime_tensors = materialize_recipe_copy_plan_tensors(
        recipe,
        source_tensors,
        target_device=resolved_target_device,
    )
    model = build_runtime_model(resolved_target_device)
    attach_tensors_to_module(
        model,
        runtime_tensors,
        replace_meta_params=True,
        skip_reserved_tensor_names=True,
        preserve_aliases=True,
        fail_on_missing=False,
        fail_on_unexpected=True,
    )
    framework_adapter.run_process_after_load(
        model,
        model_config,
        resolved_target_device,
    )
    semantic_probe_result = run_binding_finalize_semantic_validation(
        recipe=recipe,
        model=model,
        model_config=model_config,
        framework_adapter=framework_adapter,
    )
    finalized_tensors = collect_runtime_tensors_from_model(
        model,
        runtime_only_tensor_names=recipe.runtime_facts.runtime_only_tensor_names,
    )
    validate_tensor_schema_against_tensors(recipe.tensor_schema, finalized_tensors)
    return BindingFinalizeMaterializationResult(
        model=model,
        runtime_tensors=finalized_tensors,
        semantic_probe_result=semantic_probe_result,
    )


def run_binding_finalize_semantic_validation(
    *,
    recipe: Any,
    model: nn.Module,
    model_config: Any,
    framework_adapter: Any,
) -> Any:
    spec = recipe.semantic_validation_spec
    if spec.kind == "none":
        return evaluate_semantic_validation_spec(spec, None)
    return evaluate_semantic_validation_spec(
        spec,
        framework_adapter.semantic_probes(model, model_config),
    )


def collect_runtime_tensors_from_model(
    model: nn.Module,
    *,
    runtime_only_tensor_names: Sequence[str],
) -> dict[str, torch.Tensor]:
    return collect_module_tensors(
        model,
        exclude_names=tuple(str(name) for name in runtime_only_tensor_names),
        reject_reserved_tensor_names=True,
        remove_duplicate=True,
    )


def validate_binding_finalize_tensor_schema(
    tensor_schema: Sequence[Any],
    tensors: Mapping[str, torch.Tensor],
) -> None:
    validate_tensor_schema_against_tensors(tensor_schema, tensors)


__all__ = [
    "BindingFinalizeMaterializationResult",
    "allocate_tensors_from_schema",
    "apply_copy_plan",
    "collect_runtime_tensors_from_model",
    "dtype_from_string",
    "iter_ranges",
    "load_source_tensors_for_recipe",
    "materialize_binding_finalize_runtime_tensors",
    "materialize_pure_transform_runtime_tensors",
    "materialize_recipe_copy_plan_tensors",
    "narrow_by_range_spec",
    "narrow_source_view",
    "run_binding_finalize_semantic_validation",
    "tensorcast_view_slices_from_trace_plan",
    "update_dst_coverage",
    "validate_binding_finalize_tensor_schema",
    "validate_dst_coverage",
    "validate_source_tensor_names",
]
