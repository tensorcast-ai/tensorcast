#  Copyright (c) 2026, TensorCast Team.

"""Core local-ready serving helpers used by framework integrations."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, cast

import torch

import tensorcast as tc
from tensorcast.api.store import create_binding as create_tensorcast_binding
from tensorcast.api.store.owned_binding_layout import (
    build_mapped_tensor_spec,
    build_owned_layout,
)
from tensorcast.api.store.serving_builder import prepare_serving_manifest_carrier
from tensorcast.api.store.types import CanonicalIndexEntry
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.serving.builder import materialization as tc_materialization
from tensorcast.serving.builder import publication as tc_publication
from tensorcast.serving.builder.compiler import (
    CompiledServingRecipe,
    TensorSchemaEntry,
)
from tensorcast.serving.builder.compiler import (
    compiled_recipe_realization_plan_count as _compiled_recipe_realization_plan_count,
)
from tensorcast.serving.contract import logical_topology_json
from tensorcast.types import ServingTopologyRef


@dataclass(frozen=True)
class LocalReadyBindingRealizationResult:
    binding: Any
    update_epoch: Any
    layout: Any
    realization_plan: Any
    realization_entry_count: int


def materialized_tensor_schema(
    recipe: CompiledServingRecipe,
) -> tuple[TensorSchemaEntry, ...]:
    expected_names = set(recipe.trace_plan.expected_dst_names)
    return tuple(
        entry for entry in recipe.tensor_schema if entry.name in expected_names
    )


def serving_binding_tensor_schema(
    recipe: CompiledServingRecipe,
    *,
    manifest_tensor_name: str,
    manifest_bytes: bytes | None = None,
) -> tuple[TensorSchemaEntry, ...]:
    schema = list(materialized_tensor_schema(recipe))
    if manifest_bytes:
        schema.append(
            TensorSchemaEntry(
                name=manifest_tensor_name,
                dtype="torch.uint8",
                shape=(len(manifest_bytes),),
                stride=(1,),
            )
        )
    return tuple(schema)


def canonical_index_entries_from_tensor_schema(
    tensor_schema: tuple[TensorSchemaEntry, ...],
) -> tuple[CanonicalIndexEntry, ...]:
    entries: list[CanonicalIndexEntry] = []
    for entry in tensor_schema:
        dtype = tc_materialization.dtype_from_string(entry.dtype)
        numel = 1
        for dim in entry.shape:
            numel *= int(dim)
        entries.append(
            CanonicalIndexEntry(
                name=str(entry.name),
                dtype=dtype,
                shape=tuple(int(dim) for dim in entry.shape),
                stride=tuple(int(dim) for dim in entry.stride),
                storage_offset=0,
                segment_offset=0,
                size_bytes=int(numel * dtype.itemsize),
            )
        )
    return tuple(entries)


def canonical_index_from_recipe(recipe: CompiledServingRecipe) -> tc.CanonicalIndex:
    entries = canonical_index_entries_from_tensor_schema(
        materialized_tensor_schema(recipe)
    )
    total_size_bytes = sum(int(entry.size_bytes) for entry in entries)
    return tc.CanonicalIndex(
        entries=entries,
        total_size_bytes=total_size_bytes,
        avbs_hash="",
    )


def logical_topology_json_from_recipe(
    recipe: CompiledServingRecipe,
    *,
    topology: ServingTopologyRef | None = None,
    framework_payload: dict[str, Any] | None = None,
) -> str | None:
    if topology is None:
        return None if recipe.logical_topology is None else "{}"
    return logical_topology_json(
        topology,
        framework_payload=framework_payload or {},
    )


def publication_context_from_recipe(
    recipe: CompiledServingRecipe,
    *,
    logical_topology_json_payload: str | None = None,
) -> tc_publication.RecipePublicationContext:
    return tc_publication.RecipePublicationContext(
        source_artifact_ref=recipe.source_artifact_ref,
        framework_name=recipe.serving_facts.framework_name,
        adapter_version=recipe.serving_facts.adapter_version,
        serving_abi_version=recipe.serving_facts.serving_abi_version,
        logical_topology_json=logical_topology_json_payload,
    )


def prepare_same_binding_manifest_carrier(
    recipe: CompiledServingRecipe,
    *,
    manifest_tensor_name: str,
    representation_contract_hash: str,
    logical_topology_json_payload: str | None = None,
) -> tuple[str, bytes]:
    base_canonical_index = canonical_index_from_recipe(recipe)
    build_pipeline_version = "tensorcast-bootstrap-v1"
    publication_context = publication_context_from_recipe(
        recipe,
        logical_topology_json_payload=logical_topology_json_payload,
    )
    if (
        recipe.serving_facts.process_after_load_class
        == tc.FinalizeClass.REPRESENTATION_CHANGING
    ):
        build_intent = tc_publication.build_binding_finalize_build_intent(
            publication_context,
            build_pipeline_version=build_pipeline_version,
            representation_contract_hash=representation_contract_hash,
        )
    else:
        build_intent = tc_publication.build_pure_transform_build_intent(
            publication_context,
            build_pipeline_version=build_pipeline_version,
            representation_contract_hash=representation_contract_hash,
        )
    carrier = prepare_serving_manifest_carrier(
        build_intent=build_intent,
        canonical_index=base_canonical_index,
        representation_contract_hash=representation_contract_hash,
        logical_topology_json=logical_topology_json_payload,
        serving_manifest_ref=None,
    )
    return representation_contract_hash, carrier.serving_manifest_bytes


def compute_serving_binding_tensor_schema_hash(
    recipe: CompiledServingRecipe,
    *,
    manifest_tensor_name: str,
    manifest_bytes: bytes | None = None,
) -> str:
    entries = canonical_index_entries_from_tensor_schema(
        serving_binding_tensor_schema(
            recipe,
            manifest_tensor_name=manifest_tensor_name,
            manifest_bytes=manifest_bytes,
        )
    )
    canonical_index = tc.CanonicalIndex(
        entries=entries,
        total_size_bytes=sum(int(entry.size_bytes) for entry in entries),
        avbs_hash="",
    )
    return tc.compute_serving_tensor_schema_hash(
        canonical_index,
        manifest_tensor_name=manifest_tensor_name,
    )


def realization_plan_proto_with_manifest(
    realization_plan_proto: bytes,
    manifest_bytes: bytes | None,
    *,
    manifest_tensor_name: str,
) -> Any:
    proto = store_daemon_pb2.BindingRealizationPlan()
    proto.ParseFromString(bytes(realization_plan_proto))
    if manifest_bytes:
        entry = proto.entries.add(dst_name=manifest_tensor_name)
        entry.op_kind = store_daemon_pb2.BINDING_REALIZATION_OP_KIND_CONST_FILL
        entry.fill_value = bytes(manifest_bytes)
    return proto


def build_binding_layout_for_recipe(
    recipe: CompiledServingRecipe,
    *,
    target_device: torch.device,
    manifest_tensor_name: str,
    manifest_bytes: bytes | None = None,
) -> Any:
    device_index = torch.device(target_device).index
    if device_index is None:
        raise RuntimeError(
            "Tensorcast bootstrap requires an explicit CUDA device index"
        )
    tensor_schema = serving_binding_tensor_schema(
        recipe,
        manifest_tensor_name=manifest_tensor_name,
        manifest_bytes=manifest_bytes,
    )
    dst_specs = tuple(
        build_mapped_tensor_spec(
            name=entry.name,
            shape=tuple(int(dim) for dim in entry.shape),
            stride=tuple(int(dim) for dim in entry.stride),
            dtype=entry.dtype,
            logical_length=int(
                tc_materialization.dtype_from_string(entry.dtype).itemsize
                * int(torch.Size(entry.shape).numel())
            ),
        )
        for entry in tensor_schema
    )
    return build_owned_layout(
        entries=canonical_index_entries_from_tensor_schema(tensor_schema),
        device_id=int(device_index),
        index_kind=cast(
            store_daemon_pb2.TargetLayout.IndexKind,
            store_daemon_pb2.TargetLayout.IndexKind.Value(
                "INDEX_KIND_CANONICAL_UNSPECIFIED"
            ),
        ),
        logical_layout_hash=None,
        ordered_names=tuple(entry.name for entry in tensor_schema),
        dst_specs=dst_specs,
    )


def create_local_ready_binding(
    layout: Any,
    *,
    target_device: torch.device,
    binding_factory: Callable[..., Any] | None = None,
) -> Any:
    factory = binding_factory or create_tensorcast_binding
    return factory(
        layout,
        ownership="daemon",
        device=target_device,
        restore_tensors_async=True,
    )


def realize_local_ready_binding_from_source(
    *,
    recipe: CompiledServingRecipe,
    source_subject: Any,
    target_device: torch.device,
    manifest_tensor_name: str,
    manifest_bytes: bytes | None,
    options: Any | None,
    binding_factory: Callable[..., Any] | None = None,
) -> LocalReadyBindingRealizationResult:
    source_view = source_view_for_recipe(recipe, source_subject)
    layout = build_binding_layout_for_recipe(
        recipe,
        target_device=target_device,
        manifest_tensor_name=manifest_tensor_name,
        manifest_bytes=manifest_bytes,
    )
    binding = create_local_ready_binding(
        layout,
        target_device=target_device,
        binding_factory=binding_factory,
    )
    realization_plan = realization_plan_proto_with_manifest(
        bytes(getattr(recipe, "realization_plan_proto", b"") or b""),
        manifest_bytes,
        manifest_tensor_name=manifest_tensor_name,
    )
    try:
        update_epoch = binding.realize_from(
            source_view,
            realization_plan=realization_plan,
            options=options,
        )
    except Exception:
        with suppress(Exception):
            binding.close()
        raise
    return LocalReadyBindingRealizationResult(
        binding=binding,
        update_epoch=update_epoch,
        layout=layout,
        realization_plan=realization_plan,
        realization_entry_count=len(realization_plan.entries),
    )


def prepare_local_ready_serving(
    *,
    recipe: CompiledServingRecipe,
    source_subject: Any,
    target_device: torch.device,
    manifest_tensor_name: str,
    manifest_bytes: bytes | None,
    options: Any | None,
    binding_factory: Callable[..., Any] | None = None,
) -> LocalReadyBindingRealizationResult:
    return realize_local_ready_binding_from_source(
        recipe=recipe,
        source_subject=source_subject,
        target_device=target_device,
        manifest_tensor_name=manifest_tensor_name,
        manifest_bytes=manifest_bytes,
        options=options,
        binding_factory=binding_factory,
    )


def freeze_local_ready_binding(
    *,
    binding: Any,
    update_epoch: Any,
    source_artifact_ref: str,
) -> Any:
    try:
        return binding.freeze_current(
            update_epoch=update_epoch,
            source_artifact_ref=source_artifact_ref,
        )
    except Exception:
        with suppress(Exception):
            binding.close()
        raise


def source_view_for_recipe(recipe: CompiledServingRecipe, source_subject: Any) -> Any:
    source_view = source_subject
    if not isinstance(source_subject, tc.PublicDiskSourceHandle):
        subset_fn = getattr(source_subject, "subset", None)
        if not callable(subset_fn):
            raise RuntimeError(
                "Tensorcast same-binding bootstrap requires a source subject "
                "that supports subset(...) or is a PublicDiskSourceHandle"
            )
        source_view = source_subject.subset(
            sorted(recipe.trace_plan.expected_src_names)
        )
        slices = tc_materialization.tensorcast_view_slices_from_trace_plan(
            recipe.trace_plan
        )
        if slices:
            view_fn = getattr(source_view, "view", None)
            if not callable(view_fn):
                raise RuntimeError(
                    "Tensorcast same-binding bootstrap requires a source view "
                    "handle with view(...) support for traced slices"
                )
            source_view = view_fn(slices=slices)
    return source_view


def tensorcast_view_slice_count(recipe: CompiledServingRecipe) -> int:
    return len(
        tc_materialization.tensorcast_view_slices_from_trace_plan(recipe.trace_plan)
    )


def compiled_recipe_realization_plan_count(recipe: CompiledServingRecipe) -> int:
    return _compiled_recipe_realization_plan_count(recipe)


def binding_value_verification_state_name(value: Any) -> str:
    raw = int(
        getattr(
            value,
            "verification_state",
            store_daemon_pb2.BINDING_VALUE_VERIFICATION_STATE_UNSPECIFIED,
        )
        or 0
    )
    mapping: dict[int, str] = {
        int(store_daemon_pb2.BINDING_VALUE_VERIFICATION_STATE_PENDING): "pending",
        int(store_daemon_pb2.BINDING_VALUE_VERIFICATION_STATE_VERIFIED): "verified",
        int(store_daemon_pb2.BINDING_VALUE_VERIFICATION_STATE_FAILED): "failed",
        int(store_daemon_pb2.BINDING_VALUE_VERIFICATION_STATE_LOCAL_ONLY): "local_only",
    }
    return mapping.get(raw, "unspecified")


__all__ = [
    "build_binding_layout_for_recipe",
    "binding_value_verification_state_name",
    "canonical_index_entries_from_tensor_schema",
    "canonical_index_from_recipe",
    "compute_serving_binding_tensor_schema_hash",
    "create_local_ready_binding",
    "compiled_recipe_realization_plan_count",
    "freeze_local_ready_binding",
    "logical_topology_json_from_recipe",
    "LocalReadyBindingRealizationResult",
    "materialized_tensor_schema",
    "prepare_local_ready_serving",
    "prepare_same_binding_manifest_carrier",
    "publication_context_from_recipe",
    "realization_plan_proto_with_manifest",
    "realize_local_ready_binding_from_source",
    "serving_binding_tensor_schema",
    "source_view_for_recipe",
    "tensorcast_view_slice_count",
]
