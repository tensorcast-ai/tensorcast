#  Copyright (c) 2026, TensorCast Team.
"""Internal request/result DTOs for artifact-runtime lifecycle orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from tensorcast.api.store.realization_kernel import (
    ArtifactRealizationSpec,
    ResolvedArtifactSelection,
)
from tensorcast.artifact_runtime.artifact.resolver import ResolvedRuntimeArtifact
from tensorcast.artifact_runtime.attachment import (
    RuntimeBindingState,
    RuntimeBindingView,
)
from tensorcast.artifact_runtime.binding.retained import RestoredRetainedBinding
from tensorcast.artifact_runtime.host import SourceSelector
from tensorcast.retained_realization_authority import (
    ParsedRetainedRealizationAuthority,
)


@dataclass(frozen=True)
class _DirectRuntimeLoad:
    artifact_locator: Any | None = None
    policy: Any | None = None
    materialization: Any | None = None
    configured_collective_policy: Any | None = None
    source_bound_contract_state: Any | None = None
    source_bound_contract_path: str | None = None
    execution_facts: Mapping[str, Any] | None = None
    operation_scope: str = "startup.direct_runtime_artifact.bind"
    require_materialization_options: bool = False
    framework_config: Any | None = None
    model_config: Any | None = None
    target_device: Any | None = None
    expected_member: Any | None = None
    timeout_s: float | None = 30.0
    artifact_ref: str | None = None
    source_selection: ResolvedArtifactSelection | None = None
    resolved_artifact: ResolvedRuntimeArtifact | None = None
    model: Any | None = None
    model_runtime_spec: ArtifactRealizationSpec | None = None


@dataclass(frozen=True)
class RuntimeLoadResult:
    model: Any | None = None
    runtime_state: RuntimeBindingState | None = None
    runtime_view: RuntimeBindingView | None = None
    resolved_artifact: ResolvedRuntimeArtifact | None = None
    binding_result: RuntimeBindingResult | None = None


@dataclass(frozen=True)
class _RuntimeReload:
    current_state: RuntimeBindingState | Any
    artifact_locator: Any | None = None
    policy: Any | None = None
    materialization: Any | None = None
    configured_collective_policy: Any | None = None
    source_bound_contract_state: Any | None = None
    source_bound_contract_path: str | None = None
    execution_facts: Mapping[str, Any] | None = None
    operation_scope: str = "runtime_binding.swap"
    contract_identity: str | None = None
    require_materialization_options: bool = False
    framework_config: Any | None = None
    model_config: Any | None = None
    target_device: Any | None = None
    artifact_ref: str | None = None
    resolved_artifact: ResolvedRuntimeArtifact | None = None
    model: Any | None = None


@dataclass(frozen=True)
class RuntimeReloadResult:
    runtime_state: RuntimeBindingState | None = None
    runtime_view: RuntimeBindingView | None = None
    resolved_artifact: ResolvedRuntimeArtifact | None = None
    binding_result: RuntimeBindingResult | None = None


@dataclass(frozen=True)
class _RetainedBindingAcquire:
    authority: ParsedRetainedRealizationAuthority | None = None
    framework_config: Any | None = None
    model_config: Any | None = None
    target_device: Any | None = None
    expected_member: Any | None = None
    runtime: Any | None = None
    client: Any | None = None
    restore_fn: Any | None = None
    timeout_s: float | None = 30.0
    model_runtime_spec: ArtifactRealizationSpec | None = None


@dataclass(frozen=True)
class RetainedBindingResult:
    model: Any | None = None
    runtime_state: RuntimeBindingState | None = None
    runtime_view: RuntimeBindingView | None = None
    restored: RestoredRetainedBinding | None = None


@dataclass(frozen=True)
class _LocalReadyBootstrap:
    """Internal lowering payload for ``LocalSourceBootstrap``."""

    source_selector: SourceSelector | Any | None = None
    bootstrap: Any | None = None
    configured_collective_policy: Any | None = None
    source_bound_contract_state: Any | None = None
    source_bound_contract_path: str | None = None
    execution_facts: Mapping[str, Any] | None = None
    operation_scope: str = "bootstrap.same_binding_fast_path.tensorcast_realize"
    contract_identity: str | None = None
    require_materialization_options: bool = False
    framework_config: Any | None = None
    model_config: Any | None = None
    target_device: Any | None = None
    source_subject_coordinator: Any | None = None
    recipe: Any | None = None
    source_catalog: Any | None = None
    source_catalog_config: Any | None = None
    cache_config: Any | None = None
    cache_config_factory: Any | None = None
    source_subject: Any | None = None
    placement: Any | None = None
    source_artifact_ref: str | None = None
    source_selection: ResolvedArtifactSelection | None = None
    serving_manifest_ref: str | None = None
    representation_contract_hash: str | None = None
    serving_build_digest: str | None = None
    model: Any | None = None
    manifest_tensor_name: str | None = None
    manifest_bytes: bytes | None = None
    build_recipe_from_framework_context: bool = False
    build_model_from_framework_context: bool = False
    build_manifest_carrier_from_framework_context: bool = False
    run_binding_finalize_hooks_when_required: bool = False
    options: Any | None = None
    binding_factory: Any | None = None
    family: str = ""
    tp_rank: int = 0
    tp_world_size: int = 1
    replace_meta_params: bool = True
    run_process_after_load: bool = False
    run_post_bind_finalize: bool = True
    run_semantic_validation: bool = False
    semantic_validation_spec: Any | None = None
    validate_representation_contract_hash: bool = False
    runtime_binding_schema_version: int | None = None
    serving_artifact_schema_version: int | None = None
    framework_name: str | None = None
    framework_version: str | None = None
    adapter_version: str | None = None
    serving_abi_version: str | None = None
    model_runtime_spec: ArtifactRealizationSpec | None = None


@dataclass(frozen=True)
class _LocalReadyFinalize:
    """Internal payload for local-ready attach/finalize state construction."""

    model: Any
    recipe: Any
    binding: Any
    update_epoch: Any
    source_artifact_ref: str
    serving_manifest_ref: str
    representation_contract_hash: str
    serving_build_digest: str
    manifest_tensor_name: str
    source_bound_contract_state: Any
    source_bound_contract_path: str
    target_device: Any
    source_selection: ResolvedArtifactSelection | None = None
    manifest_bytes: bytes | None = None
    framework_config: Any | None = None
    model_config: Any | None = None
    placement: Any | None = None
    family: str = ""
    tp_rank: int = 0
    tp_world_size: int = 1
    replace_meta_params: bool = True
    run_process_after_load: bool = False
    run_post_bind_finalize: bool = True
    run_semantic_validation: bool = False
    semantic_validation_spec: Any | None = None
    validate_representation_contract_hash: bool = False
    runtime_binding_schema_version: int | None = None
    serving_artifact_schema_version: int | None = None
    framework_name: str | None = None
    framework_version: str | None = None
    adapter_version: str | None = None
    serving_abi_version: str | None = None
    model_runtime_spec: ArtifactRealizationSpec | None = None


def _binding_tensors(binding: Any) -> Mapping[str, torch.Tensor]:
    tensors = getattr(binding, "tensors", {})
    if tensors is None:
        return {}
    return dict(tensors)


@dataclass(frozen=True)
class RuntimeBindingResult:
    """Attach-ready result from a runtime bind or swap operation."""

    binding: Any
    tensors: Mapping[str, torch.Tensor]
    binding_layout_id: str | None = None
    operation_result: Any | None = None
    execution_diagnostics: Any | None = None
    materialization_diagnostics: Any | None = None

    @classmethod
    def from_binding(
        cls,
        binding: Any,
        *,
        operation_result: Any | None = None,
    ) -> RuntimeBindingResult:
        return cls(
            binding=binding,
            tensors=_binding_tensors(binding),
            binding_layout_id=getattr(binding, "binding_layout_id", None),
            operation_result=operation_result,
            execution_diagnostics=getattr(binding, "last_execution_diagnostics", None),
            materialization_diagnostics=getattr(
                binding,
                "last_materialization_diagnostics",
                None,
            ),
        )


__all__ = [
    "RetainedBindingResult",
    "RuntimeBindingResult",
    "RuntimeLoadResult",
    "RuntimeReloadResult",
    "_DirectRuntimeLoad",
    "_LocalReadyBootstrap",
    "_LocalReadyFinalize",
    "_RetainedBindingAcquire",
    "_RuntimeReload",
]
