#  Copyright (c) 2026, TensorCast Team.
"""Local-ready manifest, contract, and finalize validation helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import tensorcast as tc
import tensorcast.artifact_runtime.contract as tc_contract
import tensorcast.artifact_runtime.readiness as tc_readiness
import tensorcast.artifact_runtime.recipe.local_ready as tc_local_ready
import tensorcast.artifact_runtime.recipe.tensor_schema as tc_tensor_schema
from tensorcast.artifact_runtime.errors import (
    ArtifactRuntimeIntegrationError,
    ManifestMismatchError,
    SchemaMismatchError,
)
from tensorcast.artifact_runtime.host import FrameworkIdentity
from tensorcast.artifact_runtime.local_ready_projection import (
    LocalReadyBindingContract,
    LocalReadyManifestCarrierResult,
    LocalReadyMaterializationIdentity,
)
from tensorcast.artifact_runtime.recipe.build import RecipeBuildSession
from tensorcast.types import FinalizeClass


def assert_finalize_admitted(
    request: Any,
    *,
    semantic_validation_spec: Any | None,
    requires_binding_finalize: bool,
) -> None:
    if not requires_binding_finalize:
        return
    if not request.run_process_after_load:
        raise ArtifactRuntimeIntegrationError(
            "TensorCast representation-changing local-ready finalize "
            "requires process_after_load execution"
        )
    if not request.run_semantic_validation:
        raise ArtifactRuntimeIntegrationError(
            "TensorCast representation-changing local-ready finalize "
            "requires explicit semantic validation"
        )
    if (
        semantic_validation_spec is None
        or getattr(semantic_validation_spec, "kind", "none") == "none"
    ):
        raise ArtifactRuntimeIntegrationError(
            "TensorCast representation-changing local-ready finalize "
            "requires an explicit semantic validation spec"
        )
    if not request.validate_representation_contract_hash:
        raise ArtifactRuntimeIntegrationError(
            "TensorCast representation-changing local-ready finalize "
            "requires representation contract validation"
        )
    if (
        request.source_bound_contract_state is None
        or not request.source_bound_contract_path
    ):
        raise ArtifactRuntimeIntegrationError(
            "TensorCast representation-changing local-ready finalize "
            "requires same-binding contract proof"
        )
    if not getattr(
        request.source_bound_contract_state,
        "source_bound_contract_ready",
        False,
    ):
        raise ArtifactRuntimeIntegrationError(
            "TensorCast representation-changing local-ready finalize "
            "requires ready same-binding contract proof"
        )


def semantic_validation_spec(request: Any) -> Any | None:
    if request.semantic_validation_spec is not None:
        return request.semantic_validation_spec
    if not request.run_semantic_validation:
        return None
    return getattr(request.recipe, "semantic_validation_spec", None)


def validate_representation_contract_hash(
    request: Any,
    *,
    tensor_schema_hash: str,
    contract_hash_fn: Callable[..., str],
) -> None:
    if not request.validate_representation_contract_hash:
        return
    if request.model_config is None:
        raise ArtifactRuntimeIntegrationError(
            "ArtifactRuntimeIntegration local-ready representation validation "
            "requires model_config"
        )
    if request.placement is None:
        raise ArtifactRuntimeIntegrationError(
            "ArtifactRuntimeIntegration local-ready representation validation "
            "requires placement"
        )
    if request.runtime_binding_schema_version is None:
        raise ArtifactRuntimeIntegrationError(
            "ArtifactRuntimeIntegration local-ready representation validation "
            "requires runtime_binding_schema_version"
        )
    if request.serving_artifact_schema_version is None:
        raise ArtifactRuntimeIntegrationError(
            "ArtifactRuntimeIntegration local-ready representation validation "
            "requires serving_artifact_schema_version"
        )
    actual = contract_hash_fn(
        tensor_schema_hash=tensor_schema_hash,
        model_config=request.model_config,
        placement=request.placement,
        runtime_binding_schema_version=int(request.runtime_binding_schema_version),
        serving_artifact_schema_version=int(request.serving_artifact_schema_version),
        framework_name=request.framework_name,
        framework_version=request.framework_version,
        adapter_version=request.adapter_version,
        serving_abi_version=request.serving_abi_version,
    )
    expected = str(request.representation_contract_hash)
    if actual == expected:
        return
    raise ManifestMismatchError(
        "TensorCast local-ready manifest contract hash drifted after "
        f"finalize: expected={expected}, actual={actual}"
    )


def build_manifest_carrier(
    *,
    recipe: Any,
    manifest_tensor_name: str,
    representation_contract_hash: str,
    logical_topology_json_payload: str | None = None,
    topology_admission_digest: str | None = None,
) -> tuple[str, bytes]:
    return tc_local_ready.prepare_same_binding_manifest_carrier(
        recipe,
        manifest_tensor_name=manifest_tensor_name,
        representation_contract_hash=representation_contract_hash,
        logical_topology_json_payload=logical_topology_json_payload,
        topology_admission_digest=topology_admission_digest,
    )


def build_manifest_carrier_from_contract(
    *,
    recipe: Any,
    manifest_tensor_name: str,
    representation_contract_hash_factory: Any,
    topology: Any | None = None,
    framework_payload: Mapping[str, Any] | None = None,
) -> tuple[str, bytes]:
    base_canonical_index = tc_local_ready.canonical_index_from_recipe(recipe)
    tensor_schema_hash = tc_contract.compute_canonical_runtime_tensor_schema_hash(
        base_canonical_index,
        manifest_tensor_name=manifest_tensor_name,
    )
    representation_contract_hash = representation_contract_hash_factory(
        tensor_schema_hash
    )
    logical_topology_json_payload = tc_local_ready.logical_topology_json_from_recipe(
        recipe,
        topology=topology,
        framework_payload=dict(framework_payload or {}),
    )
    topology_admission_digest = _optional_text(
        getattr(topology, "schema_topology_digest", None)
    )
    return build_manifest_carrier(
        recipe=recipe,
        manifest_tensor_name=manifest_tensor_name,
        representation_contract_hash=representation_contract_hash,
        logical_topology_json_payload=logical_topology_json_payload,
        topology_admission_digest=topology_admission_digest,
    )


def representation_contract_hash(
    *,
    tensor_schema_hash: str,
    model_config: Any,
    placement: Any,
    runtime_binding_schema_version: int,
    serving_artifact_schema_version: int,
    framework_identity: FrameworkIdentity,
) -> str:
    compute_hash = getattr(model_config, "compute_hash", None)
    model_hash = (
        compute_hash()
        if callable(compute_hash)
        else getattr(model_config, "model", "unknown")
    )
    model_name = str(getattr(model_config, "model", "unknown"))
    placement_identity = getattr(placement, "identity_payload", None)
    if placement_identity is None:
        stable_identity_payload = getattr(placement, "stable_identity_payload", None)
        if callable(stable_identity_payload):
            placement_identity = stable_identity_payload()
        else:
            placement_identity = {}
    source_identity = {
        "model_hash": model_hash,
        "model_name": model_name,
        "runtime_binding_schema_version": int(runtime_binding_schema_version),
        "serving_artifact_schema_version": int(serving_artifact_schema_version),
        "placement": placement_identity,
    }
    topology_ref = getattr(placement, "topology", None)
    member_ref = getattr(placement, "member", None)
    if topology_ref is None or member_ref is None:
        raise ArtifactRuntimeIntegrationError(
            "TensorCast local-ready manifest carrier requires placement "
            "topology and member identity"
        )
    return tc_contract.compute_runtime_representation_contract_hash(
        tensor_schema_hash=str(tensor_schema_hash or ""),
        topology_ref=topology_ref,
        member_ref=member_ref,
        framework_name=str(framework_identity.framework_name),
        framework_version=str(framework_identity.framework_version),
        adapter_version=str(framework_identity.adapter_version),
        serving_abi_version=str(framework_identity.serving_abi_version),
        source_identity=source_identity,
    )


def build_manifest_carrier_from_framework_context(
    *,
    recipe: Any,
    manifest_tensor_name: str,
    placement: Any,
    contract_hash_fn: Callable[[str], str],
) -> tuple[str, bytes]:
    return build_manifest_carrier_from_contract(
        recipe=recipe,
        manifest_tensor_name=manifest_tensor_name,
        representation_contract_hash_factory=contract_hash_fn,
        topology=getattr(placement, "topology", None),
        framework_payload=getattr(placement, "framework_payload", {}),
    )


def prepare_manifest_carrier_from_framework_context(
    *,
    build_fn: Callable[..., tuple[str, bytes]],
    manifest_from_bytes: Callable[[bytes], Any],
    recipe: Any,
    manifest_tensor_name: str,
    model_config: Any,
    placement: Any,
    runtime_binding_schema_version: int,
    serving_artifact_schema_version: int,
    framework_name: str | None = None,
    framework_version: str | None = None,
    adapter_version: str | None = None,
    serving_abi_version: str | None = None,
) -> LocalReadyManifestCarrierResult:
    representation_hash, manifest_bytes = build_fn(
        recipe=recipe,
        manifest_tensor_name=manifest_tensor_name,
        model_config=model_config,
        placement=placement,
        runtime_binding_schema_version=runtime_binding_schema_version,
        serving_artifact_schema_version=serving_artifact_schema_version,
        framework_name=framework_name,
        framework_version=framework_version,
        adapter_version=adapter_version,
        serving_abi_version=serving_abi_version,
    )
    manifest = manifest_from_bytes(manifest_bytes)
    return LocalReadyManifestCarrierResult(
        representation_contract_hash=representation_hash,
        manifest_bytes=manifest_bytes,
        serving_manifest_ref=manifest.serving_manifest_ref,
        serving_build_digest=manifest.serving_build_digest,
    )


def tensor_schema_hash(
    *,
    recipe: Any,
    manifest_tensor_name: str,
    manifest_bytes: bytes | None = None,
) -> str:
    return tc_local_ready.compute_runtime_binding_tensor_schema_hash(
        recipe,
        manifest_tensor_name=manifest_tensor_name,
        manifest_bytes=manifest_bytes,
    )


def materialized_tensor_names(recipe: Any) -> tuple[str, ...]:
    return tuple(
        str(entry.name) for entry in tc_local_ready.materialized_tensor_schema(recipe)
    )


def assert_binding_tensor_set(
    *,
    recipe: Any,
    binding: Any,
    manifest_tensor_name: str,
    binding_tensors_fn: Callable[[Any], Mapping[str, Any]],
) -> None:
    expected_names = tuple(sorted(materialized_tensor_names(recipe)))
    actual_names = tuple(
        sorted(
            str(name)
            for name in binding_tensors_fn(binding)
            if str(name) != manifest_tensor_name
        )
    )
    if actual_names == expected_names:
        return
    raise SchemaMismatchError(
        "TensorCast local-ready binding tensor set does not match recipe schema: "
        f"expected={list(expected_names)}, actual={list(actual_names)}"
    )


def build_binding_contract(
    *,
    recipe: Any,
    canonical_tensors: Mapping[str, Any],
    runtime_only_tensor_names: Sequence[str],
    manifest_tensor_name: str,
    representation_contract_hash_factory: Any,
    manifest_bytes: bytes | None = None,
) -> LocalReadyBindingContract:
    realization_plan_proto = bytes(
        getattr(recipe, "realization_plan_proto", b"") or b""
    )
    realization_entry_count = tc_local_ready.compiled_recipe_realization_plan_count(
        recipe
    )
    if realization_entry_count <= 0:
        raise ArtifactRuntimeIntegrationError(
            "TensorCast local-ready binding contract requires a compiled "
            "recipe with a pre-lowered BindingRealizationPlan"
        )
    if not realization_plan_proto:
        raise ArtifactRuntimeIntegrationError(
            "TensorCast local-ready binding contract requires compiled "
            "recipe realization_plan_proto; regenerate the compiled recipe cache"
        )
    tc_tensor_schema.validate_tensor_schema_against_tensors(
        recipe.tensor_schema,
        canonical_tensors,
    )
    resolved_tensor_schema_hash = tensor_schema_hash(
        recipe=recipe,
        manifest_tensor_name=manifest_tensor_name,
        manifest_bytes=manifest_bytes,
    )
    return LocalReadyBindingContract(
        excluded_names=tuple(sorted(str(name) for name in runtime_only_tensor_names)),
        canonical_tensor_names=tuple(sorted(str(name) for name in canonical_tensors)),
        tensor_schema_hash=resolved_tensor_schema_hash,
        representation_contract_hash=representation_contract_hash_factory(
            resolved_tensor_schema_hash
        ),
        mapped_copy_plan=(),
        realization_plan_proto=realization_plan_proto,
        realization_entry_count=realization_entry_count,
        fallback_copy_plan=tuple(recipe.realization_fallback_plan),
    )


def recipe_summary_fields(recipe: Any) -> dict[str, int]:
    return RecipeBuildSession.recipe_summary_fields(recipe)


def materialization_identity(recipe: Any) -> LocalReadyMaterializationIdentity:
    return LocalReadyMaterializationIdentity(
        source_artifact_ref=str(recipe.source_artifact_ref),
        source_metadata_fingerprint=str(recipe.source_metadata_fingerprint),
    )


def requires_binding_finalize(recipe: Any) -> bool:
    runtime_facts = getattr(recipe, "runtime_facts", None)
    process_after_load_class = tc_readiness.coerce_finalize_class(
        getattr(runtime_facts, "process_after_load_class", None),
        default=FinalizeClass.RUNTIME_ONLY,
    )
    return process_after_load_class == FinalizeClass.REPRESENTATION_CHANGING


def validate_tensor_schema(
    *,
    recipe: Any,
    tensors: Mapping[str, Any],
) -> None:
    tc_tensor_schema.validate_tensor_schema_against_tensors(
        recipe.tensor_schema, tensors
    )


def freeze_binding(
    *,
    binding: Any,
    update_epoch: Any,
    source_artifact_ref: str,
) -> Any:
    return tc_local_ready.freeze_local_ready_binding(
        binding=binding,
        update_epoch=update_epoch,
        source_artifact_ref=source_artifact_ref,
    )


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def local_ready_manifest_from_bytes(manifest_bytes: bytes) -> Any:
    return tc.RuntimeArtifactManifest.from_bytes(manifest_bytes)


__all__ = [
    "assert_binding_tensor_set",
    "assert_finalize_admitted",
    "build_binding_contract",
    "build_manifest_carrier",
    "build_manifest_carrier_from_contract",
    "build_manifest_carrier_from_framework_context",
    "freeze_binding",
    "local_ready_manifest_from_bytes",
    "materialization_identity",
    "materialized_tensor_names",
    "prepare_manifest_carrier_from_framework_context",
    "recipe_summary_fields",
    "representation_contract_hash",
    "requires_binding_finalize",
    "semantic_validation_spec",
    "tensor_schema_hash",
    "validate_representation_contract_hash",
    "validate_tensor_schema",
]
