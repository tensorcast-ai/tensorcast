#  Copyright (c) 2026, TensorCast Team.
"""Artifact-runtime lifecycle implementation for framework integrations.

New framework integrations should prefer the artifact-runtime public modules
and runtime testing fixtures. This module owns lifecycle orchestration and keeps
low-level helpers out of the framework-facing host/runtime modules.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, NoReturn, cast

import torch

import tensorcast as tc
import tensorcast.artifact_runtime.binding.execution as tc_binding_runtime
import tensorcast.artifact_runtime.config as tc_runtime_config
import tensorcast.artifact_runtime.contract as tc_contract
import tensorcast.artifact_runtime.diagnostics as tc_diagnostics
import tensorcast.artifact_runtime.intent as tc_runtime_intent
import tensorcast.artifact_runtime.publication.replica as tc_replica_publication
import tensorcast.artifact_runtime.readiness as tc_readiness
import tensorcast.artifact_runtime.recipe.local_ready as tc_local_ready
import tensorcast.artifact_runtime.recipe.semantic_validation as tc_semantic_validation
import tensorcast.artifact_runtime.recipe.tensor_schema as tc_tensor_schema
import tensorcast.artifact_runtime.request_facts as tc_request_facts
import tensorcast.artifact_runtime.source as tc_source_catalog
from tensorcast.api._device import device_uuid_for
from tensorcast.api.store.common import canonical_index_to_bytes
from tensorcast.api.store.realization_kernel import (
    ArtifactRealizationHandle,
    ArtifactRealizationReport,
    ArtifactRealizationSpec,
    RealizationTargetPlan,
    ResolvedArtifactSelection,
    artifact_realization_report_to_dict,
    emit_artifact_realization_profile_event,
    envelope_for_runtime_attachment,
    model_runtime_report_for,
    report_for_runtime_attachment,
    resolve_artifact_selection,
)
from tensorcast.api.store.types import CanonicalIndex, CanonicalIndexEntry
from tensorcast.artifact_runtime.artifact.resolver import (
    ResolvedRuntimeArtifact,
    RuntimeArtifactResolver,
    canonical_index_from_descriptor,
    is_reserved_runtime_tensor_name,
)
from tensorcast.artifact_runtime.attachment import (
    RuntimeAttachment,
    RuntimeBindingState,
    RuntimeBindingView,
    RuntimeStateSeed,
)
from tensorcast.artifact_runtime.binding.retained import (
    RestoredRetainedBinding,
    restore_prepared_local_ready_binding,
    restore_retained_binding,
    runtime_restore_rejection_reason,
)
from tensorcast.artifact_runtime.dto import (
    FrameworkIntegrationContext,
    PreparedRuntimeArtifact,
    RuntimeBindingValue,
    RuntimePlacement,
)
from tensorcast.artifact_runtime.errors import (
    AdmissionRejectedError,
    ArtifactLocatorResolutionError,
    ArtifactRuntimeIntegrationError,
    ArtifactRuntimeNotImplementedError,
    AttachFinalizeError,
    AuthorityValidationError,
    CapabilityMissingError,
    ConfigConflictError,
    ManifestMismatchError,
    OwnershipTransferError,
    PlacementAdmissionError,
    RestoreBindingError,
    SchemaMismatchError,
    SourceProviderError,
    SourceSubjectError,
)
from tensorcast.artifact_runtime.errors import (
    capability_missing as _capability_missing,
)
from tensorcast.artifact_runtime.host import (
    AdmissionDecision,
    AdmissionRequest,
    DefaultAdmissionPolicy,
    FrameworkHost,
    FrameworkIdentity,
    IntegrationHost,
    MaterializationExecutionFacts,
    MaterializationPolicy,
    PlacementAdmissionFacts,
    PlacementIdentityFacts,
    PlacementMemberFacts,
    RecipeCachePolicy,
    RuntimeProfile,
    SourceCatalogRequest,
    SourceDownloadPolicy,
    SourceHost,
    SourceSelector,
    SourceSubjectCoordinator,
    TensorSurfaceHost,
    TorchTensorHost,
    runtime_placement_from_framework_facts,
)
from tensorcast.artifact_runtime.locator import (
    ArtifactLocator,
)
from tensorcast.artifact_runtime.policy import (
    RuntimePolicy,
)
from tensorcast.artifact_runtime.recipe.build import (
    RecipeBuildCacheConfig,
    RecipeBuildSession,
    RecipeBuildSessionRequest,
    RuntimeBindingPlan,
    recipe_build_cache_config_from_policy,
)
from tensorcast.artifact_runtime.recipe.build import (
    build_recipe_session as build_recipe_session_from_request,
)
from tensorcast.artifact_runtime.recipe.compiler import (
    TensorcastSemanticValidationSpec,
    TensorSchemaEntry,
)
from tensorcast.artifact_runtime.recipe.trace_ir import TracePlan
from tensorcast.artifact_runtime.source import (
    SourceSubject,
    is_public_disk_source_subject,
    resolve_source_subject,
    source_subject_broadcast_payload,
    source_subject_from_broadcast_payload,
)
from tensorcast.artifact_runtime.view import (
    RuntimeWorkerView,
    source_selection_projection_from_artifact_realization_report,
    source_selection_projection_from_execution_diagnostics,
    source_selection_projection_from_materialization_diagnostics,
)
from tensorcast.profile_utils import (
    emit_tensorcast_profile_event,
    tensorcast_profile_stage,
)
from tensorcast.retained_realization_authority import (
    ParsedRetainedRealizationAuthority,
)
from tensorcast.types import (
    BlobRef,
    CollectivePolicy,
    FinalizeClass,
    RealizationTarget,
    RuntimeBindingResolvedLayout,
    RuntimeBindingSourceRef,
    RuntimeBindingSourceReuseDecision,
    RuntimeRealizationSpecCacheEntry,
    RuntimeSupportLevel,
)

ArtifactError = tc.ArtifactError
BindingUpdateEpoch = tc.BindingUpdateEpoch
BindingReservationCapability = tc.BindingReservationCapability
BindingValueRef = tc.BindingValueRef
BuilderMode = tc.BuilderMode
DEFAULT_RUNTIME_PROFILE = tc_runtime_config.DEFAULT_RUNTIME_PROFILE
LOCAL_READY_BOOTSTRAP_BUILD_PIPELINE_VERSION = (
    tc_local_ready.LOCAL_READY_BOOTSTRAP_BUILD_PIPELINE_VERSION
)

_LOGGER = logging.getLogger(__name__)
PublishedModelVersion = tc.PublishedModelVersion
GroupRealizationAcquireRef = tc.GroupRealizationAcquireRef
SOURCE_BOUND_CONTRACT_PATH_COLLECTIVE_FIRST_V4 = (
    tc_contract.SOURCE_BOUND_CONTRACT_PATH_COLLECTIVE_FIRST_V4
)
SERVING_MANIFEST_TENSOR_NAME = tc.SERVING_MANIFEST_TENSOR_NAME
RuntimeArtifactManifest = tc.RuntimeArtifactManifest
TensorCastRuntimeConfig = tc_runtime_config.TensorCastRuntimeConfig
ReplicaPublicationPolicy = tc_runtime_config.ReplicaPublicationPolicy
RuntimeBindingMemberRef = tc.RuntimeBindingMemberRef
RuntimeArtifactPolicy = tc.RuntimeArtifactPolicy
SourceBoundContractState = tc_contract.SourceBoundContractState
source_bound_contract_profile_fields = tc_contract.source_bound_contract_profile_fields
SourceCatalog = tc_source_catalog.SourceCatalog
SOURCE_CATALOG_SCHEMA_VERSION = tc_source_catalog.SOURCE_CATALOG_SCHEMA_VERSION


ModelRuntimeRequestFactsError = tc_request_facts.ModelRuntimeRequestFactsError
resolve_model_runtime_request_facts = (
    tc_request_facts.resolve_model_runtime_request_facts
)

read_source_bound_contract_state = tc_contract.read_source_bound_contract_state
resolve_runtime_config_profile = tc_runtime_config.resolve_runtime_config_profile

binding_layout_debug_payload = tc_diagnostics.binding_layout_debug_payload
binding_layout_profile_fields = tc_diagnostics.binding_layout_profile_fields
binding_layout_tensor_count = tc_diagnostics.binding_layout_tensor_count
PLACEMENT_IDENTITY_FACTS_SCHEMA_VERSION = 1
PLACEMENT_ADMISSION_FACTS_SCHEMA_VERSION = 1
SOURCE_DOWNLOAD_POLICY_SCHEMA_VERSION = 1
RECIPE_CACHE_POLICY_SCHEMA_VERSION = 1
SOURCE_CATALOG_REQUEST_SCHEMA_VERSION = 1

__all__ = [
    "AdmissionDecision",
    "AdmissionRejectedError",
    "AdmissionRequest",
    "ArtifactLocatorResolutionError",
    "ArtifactRuntimeIntegration",
    "ArtifactRuntimeIntegrationError",
    "ArtifactRuntimeNotImplementedError",
    "ArtifactRuntimeSession",
    "AttachFinalizeError",
    "AuthorityValidationError",
    "BootstrapPolicy",
    "CapabilityMissingError",
    "ConfigConflictError",
    "DefaultAdmissionPolicy",
    "ExistingRuntimeArtifact",
    "FinalizeClass",
    "FrameworkIdentity",
    "IntegrationHost",
    "LocalReadyBindingContract",
    "LocalReadyManifestCarrierResult",
    "LocalReadyMaterializationIdentity",
    "LocalReadyPreparationResult",
    "LocalReadyRetainedTargetPlan",
    "LocalSourceBootstrap",
    "ManifestMismatchError",
    "MaterializationExecutionFacts",
    "OwnershipTransferError",
    "PLACEMENT_ADMISSION_FACTS_SCHEMA_VERSION",
    "PLACEMENT_IDENTITY_FACTS_SCHEMA_VERSION",
    "PlacementAdmissionError",
    "PlacementAdmissionFacts",
    "PreparedLocalReadyBundle",
    "PlacementIdentityFacts",
    "PlacementMemberFacts",
    "RECIPE_CACHE_POLICY_SCHEMA_VERSION",
    "SERVING_MANIFEST_TENSOR_NAME",
    "SOURCE_CATALOG_REQUEST_SCHEMA_VERSION",
    "SOURCE_CATALOG_SCHEMA_VERSION",
    "SOURCE_DOWNLOAD_POLICY_SCHEMA_VERSION",
    "RecipeBuildSessionRequest",
    "RecipeCachePolicy",
    "RequestContext",
    "RestoreBindingError",
    "RetainedBindingAcquire",
    "RuntimeAttachment",
    "RuntimeBindingMaterialization",
    "RuntimeBindingPlan",
    "RuntimeBindingResult",
    "RuntimeBindingState",
    "RuntimeBindingView",
    "RuntimeLoadResult",
    "RuntimePlacement",
    "RuntimeProfile",
    "RuntimeReloadResult",
    "RuntimeStateSeed",
    "RuntimeSupportLevel",
    "RuntimeWorkerView",
    "SchemaMismatchError",
    "SourceCatalogRequest",
    "SourceDownloadPolicy",
    "SourceHost",
    "SourceProviderError",
    "SourceSelector",
    "SourceSubject",
    "TensorSchemaEntry",
    "TorchTensorHost",
    "TensorcastSemanticValidationSpec",
    "_DirectRuntimeLoad",
    "_LocalReadyBootstrap",
    "_LocalReadyFinalize",
    "_RetainedBindingAcquire",
    "_RuntimeReload",
    "bind_runtime_artifact",
    "build_local_ready_prepared_artifact",
    "is_runtime_binding_swap_capable",
    "local_ready_current_value_summary_fields",
    "restore_prepared_local_ready_binding",
    "restore_retained_binding",
    "runtime_binding_state_from_runtime_view",
    "runtime_placement_from_framework_facts",
    "source_selection_projection_from_artifact_realization_report",
    "source_selection_projection_from_execution_diagnostics",
    "source_selection_projection_from_materialization_diagnostics",
    "source_subject_broadcast_payload",
    "source_subject_from_broadcast_payload",
    "swap_runtime_artifact",
]


BootstrapPolicy = tc_runtime_intent.BootstrapPolicy
RuntimeIntent = tc_runtime_intent.RuntimeIntent
ExistingRuntimeArtifact = tc_runtime_intent.ExistingRuntimeArtifact
LocalSourceBootstrap = tc_runtime_intent.LocalSourceBootstrap
RetainedBindingAcquire = tc_runtime_intent.RetainedBindingAcquire
RequestContext = tc_runtime_intent.RequestContext


@dataclass(frozen=True)
class RuntimeBindingMaterialization:
    """Core primitive for adapter-driven attach/finalize/state ownership."""

    host: IntegrationHost
    profile_sink: Any | None = None
    state_factory: Any = RuntimeBindingState

    def attach_and_finalize(
        self,
        *,
        model: object,
        tensors: Mapping[str, object],
        binding_handle: object,
        context: FrameworkIntegrationContext,
        state_seed: RuntimeStateSeed,
        replace_meta_params: bool,
        target_device: Any,
        model_config: object | None = None,
        run_process_after_load: bool = True,
        run_post_bind_finalize: bool = True,
        expected_tensor_schema_hash: str | None = None,
        semantic_validation_spec: Any | None = None,
        model_runtime_spec: ArtifactRealizationSpec | None = None,
    ) -> RuntimeBindingState:
        owner: Any = binding_handle
        transferred = False
        try:
            attach_start = time.perf_counter()
            self._emit("runtime_materialization.attach.start", state_seed)
            self._attach_bound_tensors(
                model,
                tensors,
                replace_meta_params=replace_meta_params,
            )
            attach_done = time.perf_counter()
            canonical = self._collect_runtime_tensors(
                model,
                remove_duplicate=False,
            )
            if expected_tensor_schema_hash is not None:
                actual_tensor_schema_hash = self._compute_tensor_schema_hash(
                    canonical,
                    remove_duplicate=False,
                )
                if actual_tensor_schema_hash != expected_tensor_schema_hash:
                    raise SchemaMismatchError(
                        "TensorCast runtime tensor schema hash mismatch: "
                        f"expected={expected_tensor_schema_hash}, "
                        f"actual={actual_tensor_schema_hash}"
                    )
            invariants = self._snapshot_tensor_invariants(canonical)
            self._allocate_runtime_only_tensors(
                model,
                torch.device(target_device),
            )
            if run_process_after_load:
                self._maybe_run_hook(
                    "run_process_after_load",
                    model,
                    model_config,
                    torch.device(target_device),
                )
            if run_post_bind_finalize:
                self._maybe_run_hook(
                    "run_runtime_only_post_bind",
                    model,
                    model_config,
                    torch.device(target_device),
                )
            if semantic_validation_spec is not None:
                self._run_semantic_validation(
                    semantic_validation_spec,
                    model,
                    model_config,
                )
            after = self._collect_runtime_tensors(
                model,
                remove_duplicate=False,
            )
            self._validate_tensor_invariants(invariants, after)
            transfer_to_runtime = getattr(binding_handle, "transfer_to_runtime", None)
            if callable(transfer_to_runtime):
                owner = transfer_to_runtime()
                transferred = True
            finalize_done = time.perf_counter()
            view = state_seed.runtime_view()
            realization_report = state_seed.realization_report
            if realization_report is not None:
                realization_report = replace(
                    realization_report,
                    runtime_attach_sec=max(0.0, attach_done - attach_start),
                    runtime_finalize_sec=max(0.0, finalize_done - attach_done),
                    total_sec=max(0.0, finalize_done - attach_start),
                )
            realization_handle = _runtime_attachment_realization_handle(
                report=realization_report,
                binding_handle=binding_handle,
                owner=owner,
            )
            model_runtime_ref: dict[str, RuntimeBindingState] = {}
            model_runtime_handle = _model_runtime_realization_handle_for_spec(
                spec=(
                    _model_runtime_spec_with_context_defaults(
                        spec=model_runtime_spec,
                        context=context,
                        target_device=target_device,
                    )
                    if model_runtime_spec is not None
                    else _model_runtime_spec_for_context(
                        context=context,
                        target_device=target_device,
                    )
                ),
                runtime_attachment_handle=realization_handle,
                attach_fn=lambda **_kwargs: model_runtime_ref["state"],
            )
            try:
                state = self.state_factory(
                    binding=binding_handle,
                    artifact_ref=state_seed.artifact_ref,
                    runtime_view=view,
                    ownership_handle=owner,
                    release_contract=None
                    if realization_handle is None
                    else realization_handle.release_contract,
                    realization_handle=realization_handle,
                    model_runtime_handle=model_runtime_handle,
                )
            except Exception as exc:
                self._close_quietly(realization_handle or owner)
                raise OwnershipTransferError(
                    "TensorCast runtime binding state construction failed"
                ) from exc
            model_runtime_ref["state"] = state
            self._emit("runtime_materialization.attach.done", state_seed)
            return state
        except OwnershipTransferError:
            raise
        except ModelRuntimeRequestFactsError:
            self._close_quietly(owner)
            raise
        except SchemaMismatchError:
            self._close_quietly(owner)
            raise
        except Exception as exc:
            self._close_quietly(owner)
            if transferred:
                raise OwnershipTransferError(
                    "TensorCast runtime binding ownership transfer failed"
                ) from exc
            raise AttachFinalizeError(
                "TensorCast runtime binding attach/finalize failed"
            ) from exc

    def _maybe_run_hook(
        self,
        name: str,
        model: object,
        model_config: object | None,
        target_device: torch.device,
    ) -> None:
        hook_host = self.host.framework
        hook = getattr(hook_host, name, None)
        if callable(hook):
            hook(model, model_config, target_device)
            return
        phase = {
            "run_process_after_load": "process_after_load",
            "run_runtime_only_post_bind": "runtime_only_post_bind",
        }.get(name)
        if phase is None:
            return
        run_hook = getattr(hook_host, "run_finalize_hook", None)
        if callable(run_hook):
            run_hook(phase, model, model_config, target_device)

    def _run_semantic_validation(
        self,
        spec: Any,
        model: object,
        model_config: object | None,
    ) -> Any:
        if getattr(spec, "kind", None) == "none":
            return tc_semantic_validation.evaluate_semantic_validation_spec(spec, None)
        hook_host = self.host.framework
        semantic_probes = getattr(hook_host, "semantic_probes", None)
        actual_payload = (
            semantic_probes(model, model_config) if callable(semantic_probes) else None
        )
        return tc_semantic_validation.evaluate_semantic_validation_spec(
            spec, actual_payload
        )

    def _surface(self) -> TensorSurfaceHost:
        if self.host.tensor_surface is None:
            raise _capability_missing(
                "IntegrationHost requires TensorSurfaceHost for runtime "
                "tensor attach/schema/invariant operations",
                level="level1-runtime",
                capability="tensor_surface",
                operation="runtime_tensor_surface",
                required_methods=(
                    "attach_bound_tensors",
                    "collect_runtime_tensors",
                    "compute_runtime_tensor_schema_hash",
                    "snapshot_tensor_invariants",
                    "validate_tensor_invariants",
                ),
                next_action=(
                    "Pass IntegrationHost(tensor_surface=...) or use "
                    "TorchTensorHost for PyTorch module carriers."
                ),
            )
        return self.host.tensor_surface

    def _attach_bound_tensors(
        self,
        model: object,
        tensors: Mapping[str, object],
        *,
        replace_meta_params: bool,
    ) -> object:
        return self._surface().attach_bound_tensors(
            model,
            tensors,
            replace_meta_params=replace_meta_params,
        )

    def _collect_runtime_tensors(
        self,
        model: object,
        *,
        remove_duplicate: bool,
    ) -> Mapping[str, object]:
        return self._surface().collect_runtime_tensors(
            model,
            remove_duplicate=remove_duplicate,
        )

    def _compute_tensor_schema_hash(
        self,
        tensors: Mapping[str, object],
        *,
        remove_duplicate: bool,
    ) -> str:
        return self._surface().compute_runtime_tensor_schema_hash(
            tensors,
            remove_duplicate=remove_duplicate,
        )

    def _allocate_runtime_only_tensors(
        self,
        model: object,
        target_device: object,
    ) -> Mapping[str, object]:
        return self._surface().allocate_runtime_only_tensors(model, target_device)

    def _snapshot_tensor_invariants(self, tensors: Mapping[str, object]) -> object:
        return self._surface().snapshot_tensor_invariants(tensors)

    def _validate_tensor_invariants(
        self,
        before: object,
        after: Mapping[str, object],
    ) -> None:
        self._surface().validate_tensor_invariants(before, after)

    def _emit(self, event: str, state_seed: RuntimeStateSeed) -> None:
        sink = self.profile_sink
        if callable(sink):
            sink(
                {
                    "event": event,
                    "artifact_ref": state_seed.artifact_ref,
                    "readiness": state_seed.readiness,
                }
            )

    @staticmethod
    def _close_quietly(handle: object) -> None:
        _close_quietly(handle)


@dataclass(frozen=True)
class _HostMaterializationRequest:
    configured_collective_policy: Any | None = None
    source_bound_contract_state: Any | None = None
    source_bound_contract_path: str | None = None
    execution_facts: Mapping[str, Any] | None = None
    operation_scope: str = ""
    require_materialization_options: bool = False


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
class _RuntimeArtifactPreflight:
    resolved_artifact: ResolvedRuntimeArtifact
    runtime_artifact_policy: Any | None


@dataclass(frozen=True)
class _RetainedBindingAcquire:
    authority: ParsedRetainedRealizationAuthority | None = None
    local_serving_ref: str | None = None
    framework_config: Any | None = None
    model_config: Any | None = None
    target_device: Any | None = None
    expected_member: Any | None = None
    expected_tensor_schema_hash: str | None = None
    expected_serving_build_digest: str | None = None
    expected_target_layout_hash: str | None = None
    expected_daemon_id: str | None = None
    expected_daemon_session_id: str | None = None
    serving_artifact_id: str | None = None
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
    """Internal lowering payload for ``LocalSourceBootstrap``.

    This is deliberately private: framework integrations enter through
    ``ArtifactRuntimeIntegration.start(LocalSourceBootstrap, context)`` and host facts.
    """

    source_selector: SourceSelector | Any | None = None
    bootstrap: Any | None = None
    materialization: Any | None = None
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


@dataclass(frozen=True)
class LocalReadyRuntimeResult:
    model: Any | None = None
    runtime_state: RuntimeBindingState | None = None
    runtime_view: RuntimeBindingView | None = None
    prepared: PreparedRuntimeArtifact | None = None
    binding_value: RuntimeBindingValue | None = None
    recipe: Any | None = None
    current_value: Any | None = None
    binding: Any | None = None
    update_epoch: Any | None = None
    layout: Any | None = None
    realization_entry_count: int | None = None
    realization: Any | None = None
    realization_report: ArtifactRealizationReport | None = None


@dataclass(frozen=True)
class LocalReadyPreparationResult:
    """Prepared local-ready recipe state without target materialization."""

    recipe: Any
    source_subject: Any
    source_artifact_ref: str
    source_catalog: Any | None = None
    cache_config: Any | None = None
    placement: Any | None = None
    family: str = ""
    tp_rank: int = 0
    tp_world_size: int = 1


@dataclass(frozen=True)
class LocalReadyRetainedTargetPlan:
    """Resolved local-ready retained binding target for artifact prefetch."""

    target: RealizationTarget
    materialization_options: Any | None
    manifest_tensor_name: str
    manifest_bytes: bytes
    tensor_schema_hash: str
    representation_contract_hash: str
    serving_build_digest: str
    target_layout_hash: str
    source_selection_digest: str


@dataclass(frozen=True)
class PreparedLocalReadyBundle:
    """Prepared local-ready artifact state and retained target plan."""

    prepared: LocalReadyPreparationResult
    retained_target_plan: LocalReadyRetainedTargetPlan


@dataclass(frozen=True)
class LocalReadyBindingContract:
    excluded_names: tuple[str, ...]
    canonical_tensor_names: tuple[str, ...]
    tensor_schema_hash: str
    representation_contract_hash: str
    mapped_copy_plan: tuple[Any, ...]
    realization_plan_proto: bytes
    realization_entry_count: int
    fallback_copy_plan: tuple[Any, ...]


@dataclass(frozen=True)
class LocalReadyMaterializationIdentity:
    source_artifact_ref: str
    source_metadata_fingerprint: str


@dataclass(frozen=True)
class LocalReadyManifestCarrierResult:
    representation_contract_hash: str
    manifest_bytes: bytes
    serving_manifest_ref: str
    serving_build_digest: str


def _binding_tensors(binding: Any) -> Mapping[str, torch.Tensor]:
    tensors = getattr(binding, "tensors", {})
    if tensors is None:
        return {}
    return dict(tensors)


def _canonical_index_bytes_from_tensors(
    tensors: Mapping[str, torch.Tensor],
) -> bytes:
    entries: list[tc.CanonicalIndexEntry] = []
    cursor = 0
    for key, tensor in sorted(tensors.items(), key=lambda item: str(item[0])):
        name = str(key)
        size_bytes = int(tensor.element_size()) * int(tensor.numel())
        entries.append(
            CanonicalIndexEntry(
                name=name,
                dtype=tensor.dtype,
                shape=tuple(int(dim) for dim in tensor.shape),
                stride=tuple(int(dim) for dim in tensor.stride()),
                storage_offset=int(tensor.storage_offset()),
                segment_offset=cursor,
                size_bytes=size_bytes,
            )
        )
        cursor += size_bytes
    return canonical_index_to_bytes(
        CanonicalIndex(entries=tuple(entries), total_size_bytes=cursor, avbs_hash="")
    )


def _canonical_index_bytes_for_runtime_selection(
    *,
    resolved: ResolvedRuntimeArtifact | Any | None,
    tensors: Mapping[str, torch.Tensor],
) -> bytes:
    descriptor = getattr(resolved, "descriptor", None)
    if descriptor is not None:
        try:
            return canonical_index_to_bytes(canonical_index_from_descriptor(descriptor))
        except (AttributeError, KeyError, TypeError, ValueError):
            _LOGGER.debug(
                "Failed to derive runtime canonical index from descriptor; using tensor metadata",
                exc_info=True,
            )
    return _canonical_index_bytes_from_tensors(tensors)


def _target_layout_digest_for_runtime_attachment(
    *,
    binding_layout_id: str | None,
    tensor_schema_hash: str,
) -> str:
    if binding_layout_id:
        return f"binding-layout:{binding_layout_id}"
    return f"runtime-schema:{tensor_schema_hash}"


def _runtime_attachment_report_for_resolved(
    *,
    resolved: ResolvedRuntimeArtifact | Any,
    tensors: Mapping[str, torch.Tensor],
    binding_handle: Any | None,
    target_device: Any,
    tensor_schema_hash: str,
    source_selection: ResolvedArtifactSelection | None = None,
    execution_diagnostics: Any | None = None,
    materialization_diagnostics: Any | None = None,
    options: Any | None = None,
) -> ArtifactRealizationReport:
    binding_layout_id = _optional_text(
        getattr(binding_handle, "binding_layout_id", None)
    )
    target_plan = RealizationTargetPlan(
        kind="runtime_attachment",
        device=target_device,
        target_layout_digest=_target_layout_digest_for_runtime_attachment(
            binding_layout_id=binding_layout_id,
            tensor_schema_hash=tensor_schema_hash,
        ),
        binding_layout_id=binding_layout_id,
    )
    envelope = envelope_for_runtime_attachment(tensors, retained=False)
    envelope.validate_for_target(target_plan)
    selection = source_selection or resolve_artifact_selection(
        artifact_id=str(getattr(resolved, "artifact_ref", "") or ""),
        canonical_index_bytes=_canonical_index_bytes_for_runtime_selection(
            resolved=resolved,
            tensors=tensors,
        ),
        tensor_names=tuple(str(name) for name in tensors),
        artifact_profile="runtime_artifact",
        authority_scope="daemon_mediated_runtime_attachment",
    )
    return report_for_runtime_attachment(
        selection=selection,
        target_plan=target_plan,
        envelope=envelope,
        binding_handle=binding_handle,
        materialization_diagnostics=materialization_diagnostics,
        execution_diagnostics=execution_diagnostics,
        risk_labels=("binding_lifecycle",),
        options=options,
    )


def _runtime_attachment_report_for_retained(
    *,
    authority: ParsedRetainedRealizationAuthority,
    tensors: Mapping[str, torch.Tensor],
    binding_handle: Any | None,
    target_device: Any,
    tensor_schema_hash: str,
    reservation_bytes: int,
    source_selection: ResolvedArtifactSelection | None = None,
) -> ArtifactRealizationReport:
    binding_layout_id = _optional_text(
        getattr(binding_handle, "binding_layout_id", None)
    )
    target_plan = RealizationTargetPlan(
        kind="runtime_attachment",
        device=target_device,
        target_layout_digest=_target_layout_digest_for_runtime_attachment(
            binding_layout_id=binding_layout_id,
            tensor_schema_hash=tensor_schema_hash,
        ),
        binding_layout_id=binding_layout_id,
        copy_plan_digest=authority.expected.resolved_spec_digest,
    )
    envelope = envelope_for_runtime_attachment(
        tensors,
        retained=True,
        reservation_bytes=reservation_bytes,
    )
    envelope.validate_for_target(target_plan)
    artifact_id = (
        authority.serving_artifact_id
        or authority.local_serving_ref
        or authority.binding_value_ref.binding_value_id
    )
    selection = source_selection or resolve_artifact_selection(
        artifact_id=str(artifact_id),
        canonical_index_bytes=_canonical_index_bytes_from_tensors(tensors),
        tensor_names=tuple(str(name) for name in tensors),
        artifact_profile="retained_binding",
        authority_scope="daemon_retained_runtime_attachment",
    )
    return report_for_runtime_attachment(
        selection=selection,
        target_plan=target_plan,
        envelope=envelope,
        binding_handle=binding_handle,
        retained_authority=authority,
        risk_labels=("retained_acquire",),
    )


def _runtime_attachment_report_for_artifact_id(
    *,
    artifact_id: str,
    tensors: Mapping[str, torch.Tensor],
    binding_handle: Any | None,
    target_device: Any,
    tensor_schema_hash: str,
    artifact_profile: str,
    authority_scope: str,
    source_selection: ResolvedArtifactSelection | None = None,
    retained: bool = False,
    reservation_bytes: int = 0,
    options: Any | None = None,
) -> ArtifactRealizationReport:
    binding_layout_id = _optional_text(
        getattr(binding_handle, "binding_layout_id", None)
    )
    target_plan = RealizationTargetPlan(
        kind="runtime_attachment",
        device=target_device,
        target_layout_digest=_target_layout_digest_for_runtime_attachment(
            binding_layout_id=binding_layout_id,
            tensor_schema_hash=tensor_schema_hash,
        ),
        binding_layout_id=binding_layout_id,
    )
    envelope = envelope_for_runtime_attachment(
        tensors,
        retained=retained,
        reservation_bytes=reservation_bytes,
    )
    envelope.validate_for_target(target_plan)
    selection = source_selection or resolve_artifact_selection(
        artifact_id=str(artifact_id),
        canonical_index_bytes=_canonical_index_bytes_from_tensors(tensors),
        tensor_names=tuple(str(name) for name in tensors),
        artifact_profile=artifact_profile,
        authority_scope=authority_scope,
    )
    return report_for_runtime_attachment(
        selection=selection,
        target_plan=target_plan,
        envelope=envelope,
        binding_handle=binding_handle,
        risk_labels=(artifact_profile,),
        options=options,
    )


def _close_quietly(handle: object) -> None:
    close = getattr(handle, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            _LOGGER.exception("Failed to close runtime attachment handle")


def _runtime_attachment_realization_handle(
    *,
    report: ArtifactRealizationReport | None,
    binding_handle: Any,
    owner: Any | None = None,
) -> ArtifactRealizationHandle | None:
    if report is None:
        return None
    owner_handle = owner if owner is not None else binding_handle
    close = getattr(owner_handle, "close", None)
    close_fn = None
    if callable(close):

        def close_owner_handle() -> None:
            close()

        close_fn = close_owner_handle
    handle = ArtifactRealizationHandle(
        target_kind="runtime_attachment",
        report=report,
        binding_value=binding_handle,
        close_fn=close_fn,
    )
    emit_artifact_realization_profile_event(report)
    return handle


def _model_runtime_spec_for_context(
    *,
    context: FrameworkIntegrationContext,
    target_device: Any,
) -> ArtifactRealizationSpec:
    placement = getattr(context, "placement", None)
    framework = str(getattr(context, "framework_name", "") or "unknown_framework")
    return ArtifactRealizationSpec.model_runtime(
        framework=framework,
        device=target_device,
        topology=getattr(placement, "topology", None),
        member=getattr(placement, "member", None),
        adapter_version=_optional_text(getattr(context, "adapter_version", None)),
        runtime_abi_version=_optional_text(
            getattr(context, "serving_abi_version", None)
        ),
    )


def _model_runtime_spec_with_context_defaults(
    *,
    spec: ArtifactRealizationSpec,
    context: FrameworkIntegrationContext,
    target_device: Any,
) -> ArtifactRealizationSpec:
    facts = resolve_model_runtime_request_facts(
        spec=spec,
        runtime_context=RequestContext(target_device=target_device),
        host_context=context,
        host_target_device=target_device,
    )
    return cast(ArtifactRealizationSpec, facts.spec)


def _model_runtime_realization_handle(
    *,
    context: FrameworkIntegrationContext,
    target_device: Any,
    runtime_attachment_handle: ArtifactRealizationHandle | None,
    attach_fn: Callable[..., RuntimeBindingState],
) -> ArtifactRealizationHandle | None:
    return _model_runtime_realization_handle_for_spec(
        spec=_model_runtime_spec_for_context(
            context=context,
            target_device=target_device,
        ),
        runtime_attachment_handle=runtime_attachment_handle,
        attach_fn=attach_fn,
    )


def _model_runtime_realization_handle_for_spec(
    *,
    spec: ArtifactRealizationSpec | None,
    runtime_attachment_handle: ArtifactRealizationHandle | None,
    attach_fn: Callable[..., RuntimeBindingState],
) -> ArtifactRealizationHandle | None:
    if runtime_attachment_handle is None:
        return None
    if spec is None:
        return None
    report = model_runtime_report_for(
        spec=spec,
        runtime_attachment_report=runtime_attachment_handle.report,
    )
    handle = ArtifactRealizationHandle(
        target_kind="model_runtime",
        report=report,
        attach_fn=attach_fn,
        release_contract=runtime_attachment_handle.release_contract,
    )
    emit_artifact_realization_profile_event(report)
    return handle


def _project_model_runtime_attachment(
    state: RuntimeBindingState,
    attachment: RuntimeAttachment,
) -> RuntimeAttachment:
    handle = state.model_runtime_handle
    if not isinstance(handle, ArtifactRealizationHandle):
        return attachment
    state.model_runtime_handle = ArtifactRealizationHandle(
        target_kind="model_runtime",
        report=handle.report,
        attachment_value=attachment,
        release_contract=handle.release_contract,
    )
    return attachment


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


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _sha256_hex_bytes(payload: bytes) -> str:
    return hashlib.sha256(bytes(payload)).hexdigest()


def _optional_text(value: Any) -> str | None:
    return _optional_str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _serving_realization_report(
    diagnostics: Mapping[str, object],
) -> Mapping[str, object] | None:
    value = diagnostics.get("serving_realization_report")
    if isinstance(value, Mapping):
        return value
    return None


def _nested_mapping(
    value: Mapping[str, object] | None,
    key: str,
) -> Mapping[str, object] | None:
    if value is None:
        return None
    nested = value.get(key)
    if isinstance(nested, Mapping):
        return nested
    return None


def _nested_value(
    value: Mapping[str, object] | None,
    *path: str,
) -> object | None:
    current: object | None = value
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _artifact_locator_kind(artifact_locator: object) -> str:
    if isinstance(artifact_locator, Mapping):
        return str(artifact_locator.get("kind") or "")
    return str(getattr(artifact_locator, "kind", "") or "")


def _collective_policy_value(policy: MaterializationPolicy) -> str:
    collective = str(policy.fields.get("collective", "auto") or "auto")
    return {
        "auto": "auto",
        "required": "require_collective",
        "disabled": "disable_collective",
        "collective_first": "collective_first",
        "require_collective": "require_collective",
        "disable_collective": "disable_collective",
    }.get(collective, collective)


def _execution_facts_payload(
    facts: MaterializationExecutionFacts,
) -> dict[str, object]:
    return {
        "tp_rank": facts.collective_rank,
        "tp_world_size": facts.collective_world_size,
        "same_node_tp": facts.same_node_tensor_parallel,
        "tp_ranks": tuple(int(rank) for rank in facts.tensor_parallel_ranks),
        "collective_world_size": facts.collective_world_size,
        "collective_rank": facts.collective_rank,
        "collective_context_unavailable": facts.collective_context_unavailable,
    }


def _framework_payload_mapping(payload: object | None) -> dict[str, object] | None:
    if not isinstance(payload, Mapping):
        return None
    return {str(key): value for key, value in payload.items()}


def source_subject_slice_count(recipe: Any, subject: Any) -> int:
    if is_public_disk_source_subject(subject):
        return 0
    return tc_local_ready.tensorcast_view_slice_count(recipe)


def runtime_binding_state_from_runtime_view(
    *,
    binding: Any,
    runtime_view: RuntimeBindingView,
    artifact_ref: str | None = None,
    ownership_handle: Any | None = None,
    artifact_realization_report: ArtifactRealizationReport | None = None,
    model_runtime_spec: ArtifactRealizationSpec | None = None,
) -> RuntimeBindingState:
    realization_handle = _runtime_attachment_realization_handle(
        report=artifact_realization_report,
        binding_handle=binding,
        owner=ownership_handle,
    )
    model_runtime_ref: dict[str, RuntimeBindingState] = {}
    model_runtime_handle = _model_runtime_realization_handle_for_spec(
        spec=model_runtime_spec,
        runtime_attachment_handle=realization_handle,
        attach_fn=lambda **_kwargs: model_runtime_ref["state"],
    )
    state = RuntimeBindingState(
        binding=binding,
        artifact_ref=artifact_ref or runtime_view.serving_artifact_ref,
        runtime_view=runtime_view,
        ownership_handle=ownership_handle,
        release_contract=None
        if realization_handle is None
        else realization_handle.release_contract,
        realization_handle=realization_handle,
        model_runtime_handle=model_runtime_handle,
    )
    model_runtime_ref["state"] = state
    return state


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def execution_diagnostics_summary_fields(
    diagnostics: Any | None,
    *,
    prefix: str,
) -> dict[str, Any]:
    if diagnostics is None:
        return {}
    fields = {
        "collective_requested": bool(
            getattr(diagnostics, "collective_requested", False)
        ),
        "collective_acknowledged": bool(
            getattr(diagnostics, "collective_acknowledged", False)
        ),
        "collective_used": bool(getattr(diagnostics, "collective_used", False)),
        "collective_policy": _enum_value(
            getattr(diagnostics, "collective_policy", None)
        ),
        "collective_failure_class": _enum_value(
            getattr(diagnostics, "collective_failure_class", None)
        ),
        "dominant_executor": getattr(diagnostics, "dominant_executor", None),
        "direct_write_supported": bool(
            getattr(diagnostics, "direct_write_supported", False)
        ),
        "fallback_bytes": int(getattr(diagnostics, "fallback_bytes", 0)),
        "residual_bytes": int(getattr(diagnostics, "residual_bytes", 0)),
        "actual_collective_committed_bytes": int(
            getattr(diagnostics, "actual_collective_committed_bytes", 0)
        ),
        "actual_local_typed_bytes": int(
            getattr(diagnostics, "actual_local_typed_bytes", 0)
        ),
        "actual_generic_backend_bytes": int(
            getattr(diagnostics, "actual_generic_backend_bytes", 0)
        ),
        "collective_skip_reason": getattr(diagnostics, "collective_skip_reason", None),
        "hash_rounds": int(getattr(diagnostics, "hash_rounds", 0)),
        "hash_backend": _enum_value(getattr(diagnostics, "hash_backend", None)),
        "hash_bytes": int(getattr(diagnostics, "hash_bytes", 0)),
        "hash_wall_time_ms": int(getattr(diagnostics, "hash_wall_time_ms", 0)),
        "hash_identity_forming": bool(
            getattr(diagnostics, "hash_identity_forming", False)
        ),
        "hash_location": _enum_value(getattr(diagnostics, "hash_location", None)),
        "identity_mint_strategy": _enum_value(
            getattr(diagnostics, "identity_mint_strategy", None)
        ),
    }
    return {f"{prefix}_{key}": value for key, value in fields.items()}


def source_bound_plan_diagnostics_summary_fields(
    diagnostics: Any | None,
    *,
    prefix: str,
) -> dict[str, Any]:
    if diagnostics is None:
        return {}
    fields = {
        "execution_plan_kind": getattr(diagnostics, "execution_plan_kind", None),
        "planned_collective_candidate_bytes": int(
            getattr(diagnostics, "planned_collective_candidate_bytes", 0)
        ),
        "planned_collective_admitted_bytes": int(
            getattr(diagnostics, "planned_collective_admitted_bytes", 0)
        ),
        "planned_local_typed_bytes": int(
            getattr(diagnostics, "planned_local_typed_bytes", 0)
        ),
        "planned_non_admitted_typed_bytes": int(
            getattr(diagnostics, "planned_non_admitted_typed_bytes", 0)
        ),
        "planned_generic_residual_bytes": int(
            getattr(diagnostics, "planned_generic_residual_bytes", 0)
        ),
        "collective_lowered_bytes": int(
            getattr(diagnostics, "collective_lowered_bytes", 0)
        ),
        "planner_reject_reason_buckets": dict(
            getattr(diagnostics, "planner_reject_reason_buckets", {})
        ),
        "planner_version": getattr(diagnostics, "planner_version", None),
        "plan_hash": getattr(diagnostics, "plan_hash", None),
        "estimated_collective_peak_temporary_bytes": int(
            getattr(diagnostics, "estimated_collective_peak_temporary_bytes", 0)
        ),
        "estimated_collective_batch_bytes": int(
            getattr(diagnostics, "estimated_collective_batch_bytes", 0)
        ),
        "estimated_collective_dedup_saving_bytes": int(
            getattr(diagnostics, "estimated_collective_dedup_saving_bytes", 0)
        ),
    }
    return {f"{prefix}_{key}": value for key, value in fields.items()}


def is_runtime_binding_swap_capable(binding: Any) -> bool:
    return bool(
        getattr(binding, "swap_capable", False)
        or callable(getattr(binding, "swap", None))
    )


def local_ready_current_value_summary_fields(
    current_value: Any,
    *,
    require_local_serving_ref: bool = False,
) -> dict[str, Any]:
    local_serving_ref = getattr(current_value, "local_serving_ref", None)
    if require_local_serving_ref and not local_serving_ref:
        raise ArtifactRuntimeIntegrationError(
            "TensorCast local-ready current value did not include local_serving_ref"
        )
    return {
        "binding_value_id": getattr(current_value, "binding_value_id", None),
        "verification_state": tc_local_ready.binding_value_verification_state_name(
            current_value
        ),
        "local_serving_ref": local_serving_ref,
    }


def binding_value_ref_from_current_value(current_value: Any) -> Any | None:
    to_ref = getattr(current_value, "to_binding_value_ref", None)
    if callable(to_ref):
        return to_ref()
    binding_id = str(getattr(current_value, "binding_id", "") or "")
    binding_layout_id = str(getattr(current_value, "binding_layout_id", "") or "")
    binding_value_id = str(getattr(current_value, "binding_value_id", "") or "")
    if not (binding_id and binding_layout_id and binding_value_id):
        return None
    return BindingValueRef(
        binding_id=binding_id,
        binding_layout_id=binding_layout_id,
        binding_value_id=binding_value_id,
        seal_generation=int(getattr(current_value, "seal_generation", 0) or 0),
    )


def build_local_ready_prepared_artifact(
    *,
    source_artifact_ref: str,
    serving_manifest_ref: str,
    representation_contract_hash: str,
    serving_build_digest: str,
    tensor_schema_hash: str,
    current_value: Any,
    binding: Any,
    family: str,
    tp_rank: int,
    tp_world_size: int,
    source_bound_contract_state: SourceBoundContractState,
    source_bound_contract_path: str,
    artifact_realization_report: ArtifactRealizationReport | None = None,
    model_runtime_spec: ArtifactRealizationSpec | None = None,
) -> LocalReadyRuntimeResult:
    current_value_fields = local_ready_current_value_summary_fields(
        current_value,
        require_local_serving_ref=True,
    )
    local_serving_ref = current_value_fields["local_serving_ref"]
    verification_state = str(
        current_value_fields["verification_state"] or "local_ready"
    )
    verification_job_id = getattr(current_value, "verification_job_id", None)
    binding_value_ref = binding_value_ref_from_current_value(current_value)
    binding_layout_id = getattr(binding, "binding_layout_id", None)
    execution_report = _strip_report_prefix(
        execution_diagnostics_summary_fields(
            getattr(binding, "last_execution_diagnostics", None),
            prefix="realize",
        ),
        prefix="realize",
    )
    plan_report = _strip_report_prefix(
        source_bound_plan_diagnostics_summary_fields(
            getattr(binding, "last_source_bound_plan_diagnostics", None),
            prefix="realize",
        ),
        prefix="realize",
    )
    realization_report = tc_diagnostics.RuntimeRealizationReport(
        source_artifact_ref=source_artifact_ref,
        serving_manifest_ref=serving_manifest_ref,
        representation_contract_hash=representation_contract_hash,
        serving_build_digest=serving_build_digest,
        tensor_schema_hash=tensor_schema_hash,
        family=family,
        tp_rank=int(tp_rank),
        tp_world_size=int(tp_world_size),
        source_bound_contract=tc_diagnostics.SourceContractReport(
            version=(source_bound_contract_state.source_bound_contract_version),
            capability_flags=tuple(
                source_bound_contract_state.source_bound_capability_names
            ),
            ready=source_bound_contract_state.source_bound_contract_ready,
            path=source_bound_contract_path,
        ),
        realization=tc_diagnostics.RealizationReport(
            binding_layout_id=binding_layout_id,
            binding_value=tc_diagnostics.BindingValueReport(
                verification_state=verification_state,
                verification_job_id=verification_job_id,
                local_serving_ref=local_serving_ref,
                binding_value_id=current_value_fields["binding_value_id"],
            ),
            execution=execution_report,
            plan=plan_report,
        ),
    )
    diagnostics = realization_report.to_runtime_diagnostics()
    if artifact_realization_report is not None:
        diagnostics["artifact_realization_report"] = (
            artifact_realization_report_to_dict(artifact_realization_report)
        )
    runtime_view = RuntimeBindingView(
        serving_artifact_ref=None,
        source_artifact_ref=source_artifact_ref,
        representation_contract_hash=representation_contract_hash,
        tensor_schema_hash=tensor_schema_hash,
        binding_value_ref=binding_value_ref,
        local_serving_ref=local_serving_ref,
        readiness="runtime_local_ready",
        diagnostics=diagnostics,
    )
    runtime_state = runtime_binding_state_from_runtime_view(
        binding=binding,
        runtime_view=runtime_view,
        artifact_ref=source_artifact_ref,
        artifact_realization_report=artifact_realization_report,
        model_runtime_spec=model_runtime_spec,
    )
    prepared = PreparedRuntimeArtifact(
        source_artifact_ref=source_artifact_ref,
        serving_artifact_ref=None,
        serving_manifest_ref=serving_manifest_ref,
        representation_contract_hash=representation_contract_hash,
        serving_build_digest=serving_build_digest,
        binding_value_ref=binding_value_ref,
        readiness="runtime_local_ready",
        family=family,
        tensor_schema_hash=tensor_schema_hash,
        binding_layout_id=binding_layout_id,
        local_serving_ref=local_serving_ref,
        verification_state=verification_state,
        verification_job_id=verification_job_id,
        tp_rank=int(tp_rank),
        tp_world_size=int(tp_world_size),
    )
    return LocalReadyRuntimeResult(
        runtime_state=runtime_state,
        runtime_view=runtime_view,
        prepared=prepared,
        binding_value=prepared.to_binding_value(),
        realization_report=artifact_realization_report,
    )


def _strip_report_prefix(fields: Mapping[str, Any], *, prefix: str) -> dict[str, Any]:
    prefix_text = f"{prefix}_"
    return {
        key.removeprefix(prefix_text): value
        for key, value in fields.items()
        if key.startswith(prefix_text)
    }


def build_collective_group_id(
    *,
    artifact_ref: str,
    operation_scope: str,
    tp_ranks: tuple[int, ...],
    contract_identity: str | None = None,
) -> str:
    payload_dict: dict[str, Any] = {
        "artifact_ref": str(artifact_ref),
        "operation_scope": operation_scope,
        "tp_ranks": [int(rank) for rank in tp_ranks],
    }
    if contract_identity:
        payload_dict["contract_identity"] = str(contract_identity)
    payload = json.dumps(payload_dict, sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"tensorcast-{digest}"


@dataclass(frozen=True)
class ArtifactRuntimeIntegration:
    """Small service object for framework-facing runtime lifecycle calls."""

    resolver: RuntimeArtifactResolver | None = None
    profile_sink: Any | None = None
    host: IntegrationHost | None = None

    @staticmethod
    def _lifecycle_not_implemented(method: str, phase: str) -> NoReturn:
        raise ArtifactRuntimeNotImplementedError(
            f"ArtifactRuntimeIntegration.{method} request DTO is available, but the "
            f"deep core-owned lifecycle is scheduled for {phase}"
        )

    def resolve(self, artifact_ref: str, **kwargs: Any) -> ResolvedRuntimeArtifact:
        return resolve_runtime_artifact(
            artifact_ref,
            resolver=self.resolver,
            **kwargs,
        )

    def read_manifest(
        self,
        artifact: Any,
        *,
        artifact_ref: str,
    ) -> ResolvedRuntimeArtifact:
        if self.resolver is None:
            raise ValueError(
                "ArtifactRuntimeIntegration.read_manifest requires resolver"
            )
        return read_runtime_artifact_manifest(
            artifact,
            artifact_ref=artifact_ref,
            resolver=self.resolver,
        )

    def cross_check(
        self,
        resolved_artifact: ResolvedRuntimeArtifact,
        **kwargs: Any,
    ) -> ResolvedRuntimeArtifact:
        if self.resolver is None:
            raise ValueError("ArtifactRuntimeIntegration.cross_check requires resolver")
        return cross_check_runtime_artifact(
            resolved_artifact,
            resolver=self.resolver,
            **kwargs,
        )

    def start(
        self,
        intent: RuntimeIntent,
        context: RequestContext,
    ) -> RuntimeAttachment:
        """Start runtime materialization from a public intent DTO."""

        decision = self._admit_intent(intent, context)
        if isinstance(intent, ExistingRuntimeArtifact):
            self._reject_source_selector_for_existing_artifact(intent.artifact_locator)
            materialization_request = self._host_materialization_request(
                context,
                operation_scope="startup.direct_runtime_artifact.bind",
            )
            load_result = self._load_existing_runtime_artifact(
                _DirectRuntimeLoad(
                    artifact_locator=intent.artifact_locator,
                    policy=intent.policy,
                    framework_config=context.framework_config,
                    model_config=context.model_config,
                    target_device=context.target_device,
                    timeout_s=context.timeout_s,
                    configured_collective_policy=(
                        materialization_request.configured_collective_policy
                    ),
                    source_bound_contract_state=(
                        materialization_request.source_bound_contract_state
                    ),
                    source_bound_contract_path=(
                        materialization_request.source_bound_contract_path
                    ),
                    execution_facts=materialization_request.execution_facts,
                    operation_scope=materialization_request.operation_scope,
                    require_materialization_options=(
                        materialization_request.require_materialization_options
                    ),
                )
            )
            if load_result.model is None or load_result.runtime_state is None:
                raise ArtifactRuntimeIntegrationError(
                    "ArtifactRuntimeIntegration.start returned no model/state for "
                    "ExistingRuntimeArtifact"
                )
            return self._attachment_from_load_result(load_result, decision)
        if isinstance(intent, RetainedBindingAcquire):
            authority = intent.authority
            expected_member = getattr(intent, "expected_member", None)
            if authority is not None:
                expected_member = authority.member
            if self.host is not None:
                placement = self._framework_context(
                    context.framework_config,
                    context.model_config,
                ).placement
                placement_member = None if placement is None else placement.member
                if (
                    authority is not None
                    and placement_member is not None
                    and placement_member != authority.member
                ):
                    raise AuthorityValidationError(
                        "ParsedRetainedRealizationAuthority.member does not match "
                        "runtime placement",
                        details={
                            "authority_member": repr(authority.member),
                            "placement_member": repr(placement_member),
                        },
                    )
                if (
                    authority is None
                    and expected_member is not None
                    and placement_member is not None
                    and expected_member != placement_member
                ):
                    raise AuthorityValidationError(
                        "RetainedBindingAcquire.expected_member does not match "
                        "runtime placement",
                        details={
                            "expected_member": repr(expected_member),
                            "placement_member": repr(placement_member),
                        },
                    )
                if placement_member is not None:
                    expected_member = placement_member
            if authority is None and expected_member is None:
                raise AuthorityValidationError(
                    "RetainedBindingAcquire(local_serving_ref=...) requires "
                    "expected_member or runtime placement"
                )
            retained_result = self._restore_retained_for_intent(
                _RetainedBindingAcquire(
                    authority=authority,
                    local_serving_ref=getattr(intent, "local_serving_ref", None),
                    framework_config=context.framework_config,
                    model_config=context.model_config,
                    target_device=context.target_device,
                    expected_member=expected_member,
                    expected_tensor_schema_hash=getattr(
                        intent, "expected_tensor_schema_hash", None
                    ),
                    expected_serving_build_digest=getattr(
                        intent, "expected_serving_build_digest", None
                    ),
                    expected_target_layout_hash=getattr(
                        intent, "expected_target_layout_hash", None
                    ),
                    expected_daemon_id=getattr(intent, "expected_daemon_id", None),
                    expected_daemon_session_id=getattr(
                        intent, "expected_daemon_session_id", None
                    ),
                    serving_artifact_id=getattr(intent, "serving_artifact_id", None),
                    timeout_s=context.timeout_s,
                )
            )
            if retained_result.model is None or retained_result.runtime_state is None:
                raise ArtifactRuntimeIntegrationError(
                    "ArtifactRuntimeIntegration.start returned no model/state for "
                    "RetainedBindingAcquire"
                )
            return self._attachment_from_retained_result(retained_result, decision)
        if isinstance(intent, LocalSourceBootstrap):
            local_ready_result = self._prepare_local_source_bootstrap(
                self._local_source_bootstrap_request(
                    intent,
                    context,
                    decision=decision,
                )
            )
            if (
                local_ready_result.model is None
                or local_ready_result.runtime_state is None
            ):
                raise ArtifactRuntimeIntegrationError(
                    "ArtifactRuntimeIntegration.start returned no model/state for "
                    "LocalSourceBootstrap"
                )
            self._run_local_ready_barrier(context)
            return self._attachment_from_local_ready_result(
                local_ready_result,
                decision,
            )
        raise ArtifactRuntimeIntegrationError(
            f"Unsupported TensorCast runtime intent: {type(intent).__name__}"
        )

    def prepare_local_ready_recipe(
        self,
        intent: LocalSourceBootstrap,
        context: RequestContext,
    ) -> LocalReadyPreparationResult:
        """Prepare local-ready recipe/source metadata without materializing target tensors."""

        if not isinstance(intent, LocalSourceBootstrap):
            raise ArtifactRuntimeIntegrationError(
                "ArtifactRuntimeIntegration.prepare_local_ready_recipe requires "
                "LocalSourceBootstrap"
            )
        decision = self._admit_intent(intent, context)
        request = self._local_source_bootstrap_request(
            intent,
            context,
            decision=decision,
        )
        if (
            request.recipe is None or request.source_subject is None
        ) and request.build_recipe_from_framework_context:
            request = self._local_ready_prepare_with_built_recipe(request)
        if request.recipe is None or request.source_subject is None:
            self._lifecycle_not_implemented("prepare_local_ready_recipe", "P5")
        self._validate_prepared_local_ready_request_identity(request)
        identity = self.local_ready_materialization_identity(request.recipe)
        source_subject = getattr(
            request.source_subject, "subject", request.source_subject
        )
        return LocalReadyPreparationResult(
            recipe=request.recipe,
            source_subject=source_subject,
            source_artifact_ref=identity.source_artifact_ref,
            source_catalog=request.source_catalog,
            cache_config=request.cache_config,
            placement=request.placement,
            family=request.family,
            tp_rank=int(request.tp_rank),
            tp_world_size=int(request.tp_world_size),
        )

    def prepare_local_ready_bundle(
        self,
        intent: LocalSourceBootstrap,
        context: RequestContext,
    ) -> PreparedLocalReadyBundle:
        """Prepare the complete reusable local-ready state for a runtime target.

        This is the artifact-centered entrypoint for callers that need the
        unified fast path: source/recipe metadata plus the retained local-ready
        prefetch target.
        """

        prepared = self.prepare_local_ready_recipe(intent, context)
        retained_target_plan = self.build_local_ready_retained_target_plan(
            intent,
            context,
            prepared=prepared,
        )
        return PreparedLocalReadyBundle(
            prepared=prepared,
            retained_target_plan=retained_target_plan,
        )

    def build_local_ready_retained_target_plan(
        self,
        intent: LocalSourceBootstrap,
        context: RequestContext,
        *,
        prepared: LocalReadyPreparationResult,
    ) -> LocalReadyRetainedTargetPlan:
        """Build a retained local-ready prefetch target from prepared recipe facts."""

        if not isinstance(intent, LocalSourceBootstrap):
            raise ArtifactRuntimeIntegrationError(
                "ArtifactRuntimeIntegration.build_local_ready_retained_target_plan "
                "requires LocalSourceBootstrap"
            )
        decision = self._admit_intent(intent, context)
        request = self._local_source_bootstrap_request(
            intent,
            context,
            decision=decision,
        )
        request = replace(
            request,
            recipe=prepared.recipe,
            source_subject=prepared.source_subject,
            source_artifact_ref=prepared.source_artifact_ref,
            source_catalog=prepared.source_catalog,
            cache_config=prepared.cache_config,
            placement=prepared.placement or request.placement,
            family=prepared.family,
            tp_rank=int(prepared.tp_rank),
            tp_world_size=int(prepared.tp_world_size),
            build_recipe_from_framework_context=False,
            build_model_from_framework_context=False,
            build_manifest_carrier_from_framework_context=True,
        )
        self._validate_prepared_local_ready_request_identity(request)
        return self._build_local_ready_retained_target_plan_from_request(request)

    def _build_local_ready_retained_target_plan_from_request(
        self,
        request: _LocalReadyBootstrap,
    ) -> LocalReadyRetainedTargetPlan:
        if self.host is None:
            raise ArtifactRuntimeIntegrationError(
                "ArtifactRuntimeIntegration.build_local_ready_retained_target_plan "
                "requires IntegrationHost"
            )
        if request.recipe is None:
            raise ArtifactRuntimeIntegrationError(
                "TensorCast local-ready retained target requires a prepared recipe"
            )
        if request.model_config is None:
            raise ArtifactRuntimeIntegrationError(
                "TensorCast local-ready retained target requires model_config"
            )
        if request.placement is None:
            raise ArtifactRuntimeIntegrationError(
                "TensorCast local-ready retained target requires runtime placement"
            )
        target_device = self._require_target_device(request.target_device)
        if target_device.index is None:
            raise ArtifactRuntimeIntegrationError(
                "TensorCast local-ready retained target requires an explicit CUDA device"
            )
        topology = getattr(request.placement, "topology", None)
        member = getattr(request.placement, "member", None)
        if topology is None or member is None:
            raise ArtifactRuntimeIntegrationError(
                "TensorCast local-ready retained target requires placement topology and member"
            )

        profile = self.host.runtime_profile or RuntimeProfile()
        framework_identity = self.host.framework.identity(request.model_config)
        runtime_binding_schema_version = (
            request.runtime_binding_schema_version
            if request.runtime_binding_schema_version is not None
            else self._runtime_binding_schema_version(profile)
        )
        serving_artifact_schema_version = (
            request.serving_artifact_schema_version
            if request.serving_artifact_schema_version is not None
            else self._serving_artifact_schema_version(profile)
        )
        manifest_tensor_name = str(
            request.manifest_tensor_name or profile.manifest_policy.manifest_tensor_name
        )
        carrier = self.prepare_local_ready_manifest_carrier_from_framework_context(
            recipe=request.recipe,
            manifest_tensor_name=manifest_tensor_name,
            model_config=request.model_config,
            placement=request.placement,
            runtime_binding_schema_version=int(runtime_binding_schema_version),
            serving_artifact_schema_version=int(serving_artifact_schema_version),
            framework_name=request.framework_name or framework_identity.framework_name,
            framework_version=(
                request.framework_version or framework_identity.framework_version
            ),
            adapter_version=request.adapter_version
            or framework_identity.adapter_version,
            serving_abi_version=(
                request.serving_abi_version or framework_identity.serving_abi_version
            ),
        )
        layout = tc_local_ready.build_binding_layout_for_recipe(
            request.recipe,
            target_device=target_device,
            manifest_tensor_name=manifest_tensor_name,
            manifest_bytes=carrier.manifest_bytes,
        )
        realization_plan = tc_local_ready.realization_plan_proto_with_manifest(
            bytes(getattr(request.recipe, "realization_plan_proto", b"") or b""),
            carrier.manifest_bytes,
            manifest_tensor_name=manifest_tensor_name,
        )
        realization_plan_bytes = realization_plan.SerializeToString(deterministic=True)
        source_identity = self.local_ready_materialization_identity(request.recipe)
        source_index_bytes = canonical_index_to_bytes(
            tc_local_ready.canonical_index_from_recipe(request.recipe)
        )
        selection = resolve_artifact_selection(
            artifact_id=source_identity.source_artifact_ref,
            canonical_index_bytes=source_index_bytes,
        )
        source = RuntimeBindingSourceRef(
            source_kind="checkpoint_artifact",
            artifact_selection_digest=selection.source_selection_digest,
            source_artifact_ref=source_identity.source_artifact_ref,
            source_schema_hash=source_identity.source_metadata_fingerprint,
            representation_contract_hash=carrier.representation_contract_hash,
        )
        source_reuse = RuntimeBindingSourceReuseDecision(
            mode="checkpoint_to_runtime",
            representation_contract_hash=carrier.representation_contract_hash,
        )
        target_layout_bytes = layout.target_layout.SerializeToString(deterministic=True)
        target_index_bytes = bytes(layout.target_index_bytes)
        target_layout_hash = _sha256_hex_bytes(target_layout_bytes)
        tensor_schema_hash = self.local_ready_tensor_schema_hash(
            recipe=request.recipe,
            manifest_tensor_name=manifest_tensor_name,
            manifest_bytes=carrier.manifest_bytes,
        )
        compute_hash = getattr(request.model_config, "compute_hash", None)
        model_config_digest = (
            str(compute_hash())
            if callable(compute_hash)
            else str(getattr(request.model_config, "model", "unknown"))
        )
        blob_refs = {
            "target_layout.bin": BlobRef(
                path="target_layout.bin",
                sha256=_sha256_hex_bytes(target_layout_bytes),
                size_bytes=len(target_layout_bytes),
            ),
            "target_index.json": BlobRef(
                path="target_index.json",
                sha256=_sha256_hex_bytes(target_index_bytes),
                size_bytes=len(target_index_bytes),
            ),
        }
        draft_entry = RuntimeRealizationSpecCacheEntry(
            schema_version=1,
            cache_key_digest="placeholder",
            spec_digest="placeholder",
            runtime=framework_identity.framework_name,
            source=source,
            source_reuse=source_reuse,
            topology=topology,
            member=member,
            source_schema_hash=source_identity.source_metadata_fingerprint,
            model_config_digest=model_config_digest,
            runtime_build_digest=carrier.serving_build_digest,
            binding_layout_id=layout.binding_layout_id,
            target_layout_hash=target_layout_hash,
            tensor_schema_hash=tensor_schema_hash,
            blob_refs=blob_refs,
        )
        keyed_entry = draft_entry.model_copy(
            update={"cache_key_digest": draft_entry.computed_cache_key_digest()}
        )
        spec_entry = keyed_entry.model_copy(
            update={"spec_digest": keyed_entry.computed_spec_digest()}
        )
        resolved_layout = RuntimeBindingResolvedLayout(
            binding_layout_id=layout.binding_layout_id,
            source=source,
            source_reuse=source_reuse,
            topology=topology,
            member=member,
            target_layout=target_layout_bytes,
            target_index_bytes=target_index_bytes,
            target_layout_hash=target_layout_hash,
            tensor_schema_hash=tensor_schema_hash,
            spec_digest=spec_entry.spec_digest,
            source_schema_hash=source_identity.source_metadata_fingerprint,
            realization_plan_bytes=realization_plan_bytes,
        )
        target = RealizationTarget(
            runtime=framework_identity.framework_name,
            device=str(target_device),
            device_uuid=device_uuid_for(int(target_device.index)),
            source=source,
            topology=topology,
            member=member,
            model_config_digest=model_config_digest,
            runtime_build_digest=carrier.serving_build_digest,
            resolved_layout=resolved_layout,
        )
        options = request.options
        if options is None:
            options = self._local_ready_materialization_options(request)
        return LocalReadyRetainedTargetPlan(
            target=target,
            materialization_options=options,
            manifest_tensor_name=manifest_tensor_name,
            manifest_bytes=carrier.manifest_bytes,
            tensor_schema_hash=tensor_schema_hash,
            representation_contract_hash=carrier.representation_contract_hash,
            serving_build_digest=carrier.serving_build_digest,
            target_layout_hash=target_layout_hash,
            source_selection_digest=selection.source_selection_digest,
        )

    def _retained_expected_member(
        self,
        authority: ParsedRetainedRealizationAuthority,
        context: RequestContext,
    ) -> Any:
        expected_member = authority.member
        if self.host is None:
            return expected_member
        placement = self._framework_context(
            context.framework_config,
            context.model_config,
        ).placement
        if (
            placement is not None
            and placement.member is not None
            and placement.member != authority.member
        ):
            raise AuthorityValidationError(
                "ParsedRetainedRealizationAuthority.member does not match "
                "runtime placement",
                details={
                    "authority_member": repr(authority.member),
                    "placement_member": repr(placement.member),
                },
            )
        if placement is not None and placement.member is not None:
            return placement.member
        return expected_member

    def realize_model_runtime(
        self,
        *,
        artifact_ref: str,
        spec: ArtifactRealizationSpec,
        context: RequestContext,
        source_selection: ResolvedArtifactSelection | None = None,
        runtime_artifact_policy: Any | None = None,
        materialization: Any | None = None,
    ) -> RuntimeAttachment:
        """Realize an artifact-rooted model runtime without a session."""

        if spec.target_kind != "model_runtime":
            raise ArtifactRuntimeIntegrationError(
                "ArtifactRuntimeIntegration.realize_model_runtime requires a model_runtime spec"
            )
        framework_context = self._framework_context(
            context.framework_config,
            context.model_config,
        )
        facts = resolve_model_runtime_request_facts(
            spec=spec,
            runtime_context=context,
            host_context=framework_context,
        )
        spec = cast(ArtifactRealizationSpec, facts.spec)
        context = cast(RequestContext, facts.context)
        intent = ExistingRuntimeArtifact(
            artifact_locator=str(artifact_ref), policy=runtime_artifact_policy
        )
        decision = self._admit_intent(intent, context)
        materialization_request = self._host_materialization_request(
            context,
            operation_scope="startup.direct_artifact_runtime.bind",
        )
        result = self._load_existing_runtime_artifact(
            _DirectRuntimeLoad(
                artifact_ref=str(artifact_ref),
                policy=runtime_artifact_policy,
                materialization=materialization,
                framework_config=context.framework_config,
                model_config=context.model_config,
                target_device=context.target_device,
                timeout_s=context.timeout_s,
                configured_collective_policy=(
                    materialization_request.configured_collective_policy
                ),
                source_selection=source_selection,
                source_bound_contract_state=(
                    materialization_request.source_bound_contract_state
                ),
                source_bound_contract_path=(
                    materialization_request.source_bound_contract_path
                ),
                execution_facts=materialization_request.execution_facts,
                operation_scope=materialization_request.operation_scope,
                require_materialization_options=(
                    materialization_request.require_materialization_options
                ),
                model_runtime_spec=spec,
            )
        )
        if result.model is None or result.runtime_state is None:
            raise ArtifactRuntimeIntegrationError(
                "ArtifactRuntimeIntegration.realize_model_runtime returned no model/state"
            )
        return self._attachment_from_load_result(result, decision)

    def realize_retained_model_runtime(
        self,
        *,
        authority: ParsedRetainedRealizationAuthority,
        spec: ArtifactRealizationSpec,
        context: RequestContext,
    ) -> RuntimeAttachment:
        """Realize a retained handoff for a model runtime without a session."""

        if spec.target_kind != "model_runtime":
            raise ArtifactRuntimeIntegrationError(
                "ArtifactRuntimeIntegration.realize_retained_model_runtime requires "
                "a model_runtime spec"
            )
        framework_context = self._framework_context(
            context.framework_config,
            context.model_config,
        )
        facts = resolve_model_runtime_request_facts(
            spec=spec,
            runtime_context=context,
            host_context=framework_context,
        )
        spec = cast(ArtifactRealizationSpec, facts.spec)
        context = cast(RequestContext, facts.context)
        intent = RetainedBindingAcquire(authority)
        decision = self._admit_intent(intent, context)
        retained_result = self._restore_retained_for_intent(
            _RetainedBindingAcquire(
                authority=authority,
                framework_config=context.framework_config,
                model_config=context.model_config,
                target_device=context.target_device,
                expected_member=self._retained_expected_member(authority, context),
                timeout_s=context.timeout_s,
                model_runtime_spec=spec,
            )
        )
        if retained_result.model is None or retained_result.runtime_state is None:
            raise ArtifactRuntimeIntegrationError(
                "ArtifactRuntimeIntegration.realize_retained_model_runtime returned no "
                "model/state"
            )
        return self._attachment_from_retained_result(retained_result, decision)

    def realize_mounted_source_model_runtime(
        self,
        *,
        artifact_ref: str,
        source_subject: Any,
        spec: ArtifactRealizationSpec,
        context: RequestContext,
        source_selection: ResolvedArtifactSelection | None = None,
        source_selector: SourceSelector | None = None,
        bootstrap_policy: Any | None = None,
        materialization: Any | None = None,
        prepared_local_ready: LocalReadyPreparationResult | None = None,
    ) -> RuntimeAttachment:
        """Realize a daemon-attested mounted source as a model runtime."""

        if spec.target_kind != "model_runtime":
            raise ArtifactRuntimeIntegrationError(
                "ArtifactRuntimeIntegration.realize_mounted_source_model_runtime "
                "requires a model_runtime spec"
            )
        framework_context = self._framework_context(
            context.framework_config,
            context.model_config,
        )
        facts = resolve_model_runtime_request_facts(
            spec=spec,
            runtime_context=context,
            host_context=framework_context,
        )
        spec = cast(ArtifactRealizationSpec, facts.spec)
        context = cast(RequestContext, facts.context)
        source_artifact_ref = tc_source_catalog.resolve_source_artifact_ref(
            str(artifact_ref)
        )
        if not source_artifact_ref.startswith("msa1:"):
            raise ArtifactRuntimeIntegrationError(
                "mounted-source model_runtime realization requires an msa1 "
                "mounted-source artifact"
            )
        subject = self._source_subject_for_mounted_source(
            source_artifact_ref=source_artifact_ref,
            source_subject=source_subject,
        )
        resolved_selector = source_selector or self._source_selector_for_subject(
            subject
        )
        intent = LocalSourceBootstrap(
            source_selector=resolved_selector,
            bootstrap_policy=bootstrap_policy or BootstrapPolicy(),
        )
        decision = self._admit_intent(intent, context)
        request = self._local_source_bootstrap_request(
            intent,
            context,
            decision=decision,
            model_runtime_spec=spec,
        )
        if prepared_local_ready is not None:
            request = replace(
                request,
                recipe=prepared_local_ready.recipe,
                source_catalog=prepared_local_ready.source_catalog,
                cache_config=prepared_local_ready.cache_config,
                source_subject=prepared_local_ready.source_subject,
                placement=prepared_local_ready.placement or request.placement,
                build_recipe_from_framework_context=False,
            )
        if materialization is not None:
            request = replace(request, options=materialization)
        local_ready_result = self._prepare_local_source_bootstrap(
            replace(
                request,
                source_subject=(
                    request.source_subject
                    if prepared_local_ready is not None
                    else subject
                ),
                source_artifact_ref=source_artifact_ref,
                source_selection=source_selection,
            )
        )
        if local_ready_result.model is None or local_ready_result.runtime_state is None:
            raise ArtifactRuntimeIntegrationError(
                "ArtifactRuntimeIntegration.realize_mounted_source_model_runtime "
                "returned no model/state"
            )
        self._run_local_ready_barrier(context)
        return self._attachment_from_local_ready_result(local_ready_result, decision)

    def reload(
        self,
        current_state: RuntimeBindingState | Any,
        intent: ExistingRuntimeArtifact,
        context: RequestContext,
        *,
        model: object | None = None,
        contract_identity: str | None = None,
    ) -> RuntimeAttachment:
        """Reload an existing runtime binding from a public runtime intent."""

        if not isinstance(intent, ExistingRuntimeArtifact):
            raise ArtifactRuntimeIntegrationError(
                "ArtifactRuntimeIntegration.reload currently accepts "
                "ExistingRuntimeArtifact intent only"
            )
        self._reject_source_selector_for_existing_artifact(intent.artifact_locator)
        decision = self._admit_intent(intent, context, reload=True)
        materialization_request = self._host_materialization_request(
            context,
            operation_scope="runtime_binding.swap",
        )
        result = self._reload_existing_runtime_artifact(
            _RuntimeReload(
                current_state=current_state,
                artifact_locator=intent.artifact_locator,
                policy=intent.policy,
                framework_config=context.framework_config,
                model_config=context.model_config,
                target_device=context.target_device,
                contract_identity=contract_identity,
                model=model,
                configured_collective_policy=(
                    materialization_request.configured_collective_policy
                ),
                source_bound_contract_state=(
                    materialization_request.source_bound_contract_state
                ),
                source_bound_contract_path=(
                    materialization_request.source_bound_contract_path
                ),
                execution_facts=materialization_request.execution_facts,
                operation_scope=materialization_request.operation_scope,
                require_materialization_options=(
                    materialization_request.require_materialization_options
                ),
            )
        )
        if result.runtime_state is None:
            raise ArtifactRuntimeIntegrationError(
                "ArtifactRuntimeIntegration.reload returned no runtime state"
            )
        runtime_model = (
            model if model is not None else getattr(current_state, "model", None)
        )
        view = self._worker_view_from_state(
            result.runtime_state,
            decision=decision,
            include_reload_response=True,
        )
        attachment = RuntimeAttachment(
            model=runtime_model,
            state=result.runtime_state,
            view=view,
        )
        return _project_model_runtime_attachment(result.runtime_state, attachment)

    def describe(self, state: RuntimeBindingState | Any) -> RuntimeWorkerView:
        """Return the typed endpoint/worker projection for core runtime state."""

        if isinstance(state, RuntimeWorkerView):
            return state
        return self._worker_view_from_state(state, decision=None)

    def _admit_intent(
        self,
        intent: RuntimeIntent,
        context: RequestContext,
        *,
        reload: bool = False,
    ) -> AdmissionDecision | None:
        if self.host is None:
            return None
        if context.model_config is None:
            raise ArtifactRuntimeIntegrationError(
                "ArtifactRuntimeIntegration host admission requires model_config"
            )
        framework_identity = self.host.framework.identity(context.model_config)
        placement_identity = self.host.placement.identity_facts(
            context.framework_config
        )
        placement_admission = self.host.placement.admission_facts(
            context.framework_config
        )
        request = AdmissionRequest(
            intent=intent,
            framework_identity=framework_identity,
            placement_identity=placement_identity,
            placement_admission=placement_admission,
            model_config=context.model_config,
            runtime_profile=self.host.runtime_profile or RuntimeProfile(),
        )
        policy = self.host.admission or DefaultAdmissionPolicy()
        decision = policy.admit(request)
        endpoint_fields = dict(decision.endpoint_fields)
        endpoint_fields.setdefault("family", decision.family)
        endpoint_fields.setdefault("tp_rank", placement_identity.tensor_parallel_rank)
        endpoint_fields.setdefault(
            "tp_world_size", placement_identity.tensor_parallel_size
        )
        decision = replace(decision, endpoint_fields=endpoint_fields)
        if reload:
            allowed = decision.reload_allowed
            action = "reload"
        elif isinstance(intent, LocalSourceBootstrap):
            allowed = decision.local_bootstrap_allowed
            action = "local bootstrap"
        else:
            allowed = decision.startup_allowed
            action = "startup"
        if not allowed:
            raise ArtifactRuntimeIntegrationError(
                "TensorCast admission rejected "
                f"{action}: family={decision.family!r}, "
                f"support_level={decision.support_level!r}"
            )
        return decision

    @staticmethod
    def _reject_source_selector_for_existing_artifact(
        artifact_locator: object,
    ) -> None:
        if isinstance(artifact_locator, SourceSelector):
            raise ArtifactRuntimeIntegrationError(
                "ExistingRuntimeArtifact requires a durable runtime artifact "
                "locator; local source selectors must use LocalSourceBootstrap"
            )
        if _artifact_locator_kind(artifact_locator) == "local_path":
            raise ArtifactRuntimeIntegrationError(
                "ExistingRuntimeArtifact rejects local_path artifact locators; use "
                "LocalSourceBootstrap for local source acquisition"
            )

    def _attachment_from_load_result(
        self,
        result: RuntimeLoadResult,
        decision: AdmissionDecision | None,
    ) -> RuntimeAttachment:
        state = result.runtime_state
        if state is None or result.model is None:
            raise ArtifactRuntimeIntegrationError(
                "RuntimeLoadResult is missing model or runtime_state"
            )
        attachment = RuntimeAttachment(
            model=result.model,
            state=state,
            view=self._worker_view_from_state(state, decision=decision),
        )
        return _project_model_runtime_attachment(state, attachment)

    def _attachment_from_retained_result(
        self,
        result: RetainedBindingResult,
        decision: AdmissionDecision | None,
    ) -> RuntimeAttachment:
        state = result.runtime_state
        if state is None or result.model is None:
            raise ArtifactRuntimeIntegrationError(
                "RetainedBindingResult is missing model or runtime_state"
            )
        attachment = RuntimeAttachment(
            model=result.model,
            state=state,
            view=self._worker_view_from_state(state, decision=decision),
        )
        return _project_model_runtime_attachment(state, attachment)

    def _attachment_from_local_ready_result(
        self,
        result: LocalReadyRuntimeResult,
        decision: AdmissionDecision | None,
    ) -> RuntimeAttachment:
        state = result.runtime_state
        if state is None or result.model is None:
            raise ArtifactRuntimeIntegrationError(
                "LocalReadyRuntimeResult is missing model or runtime_state"
            )
        attachment = RuntimeAttachment(
            model=result.model,
            state=state,
            view=self._worker_view_from_state(state, decision=decision),
            prepared=result.prepared,
            recipe=result.recipe,
        )
        return _project_model_runtime_attachment(state, attachment)

    def _local_source_bootstrap_request(
        self,
        intent: LocalSourceBootstrap,
        context: RequestContext,
        *,
        decision: AdmissionDecision | None,
        model_runtime_spec: ArtifactRealizationSpec | None = None,
    ) -> _LocalReadyBootstrap:
        if self.host is None:
            raise ArtifactRuntimeIntegrationError(
                "ArtifactRuntimeIntegration.start(LocalSourceBootstrap) requires "
                "IntegrationHost"
            )
        if context.model_config is None:
            raise ArtifactRuntimeIntegrationError(
                "ArtifactRuntimeIntegration.start(LocalSourceBootstrap) requires model_config"
            )
        profile = self.host.runtime_profile or RuntimeProfile()
        identity = self.host.framework.identity(context.model_config)
        placement_identity = self.host.placement.identity_facts(
            context.framework_config
        )
        placement = self._host_runtime_placement(context.framework_config)
        recipe = getattr(intent, "recipe", None)
        model = getattr(intent, "model", None)
        coordinator = getattr(intent, "coordinator", None)
        if coordinator is None:
            coordinator = self._host_source_subject_coordinator(
                context.framework_config
            )
        source_catalog_config = getattr(intent, "source_catalog_config", None)
        if source_catalog_config is None:
            source_catalog_config = self._host_source_catalog_config(
                context.framework_config,
                context.model_config,
            )
        cache_config = intent.cache_policy
        if cache_config is None:
            cache_config = self._host_recipe_cache_policy(
                context.framework_config,
                context.model_config,
            )
        materialization_request = self._host_materialization_request(
            context,
            operation_scope="bootstrap.same_binding_fast_path.tensorcast_realize",
        )
        return _LocalReadyBootstrap(
            source_selector=intent.source_selector,
            bootstrap=intent.bootstrap_policy,
            cache_config=cache_config,
            source_subject_coordinator=coordinator,
            recipe=recipe,
            source_subject=getattr(intent, "source_subject", None),
            source_artifact_ref=getattr(intent, "source_artifact_ref", None),
            source_catalog=getattr(intent, "source_catalog", None),
            source_catalog_config=source_catalog_config,
            cache_config_factory=getattr(intent, "cache_config_factory", None),
            framework_config=context.framework_config,
            model_config=context.model_config,
            target_device=context.target_device,
            model=model,
            manifest_tensor_name=profile.manifest_policy.manifest_tensor_name,
            placement=placement,
            family=(decision.family if decision is not None else ""),
            tp_rank=placement_identity.tensor_parallel_rank,
            tp_world_size=placement_identity.tensor_parallel_size,
            build_recipe_from_framework_context=recipe is None,
            build_model_from_framework_context=model is None,
            build_manifest_carrier_from_framework_context=True,
            run_binding_finalize_hooks_when_required=True,
            run_post_bind_finalize=True,
            validate_representation_contract_hash=True,
            runtime_binding_schema_version=self._runtime_binding_schema_version(
                profile
            ),
            serving_artifact_schema_version=self._serving_artifact_schema_version(
                profile
            ),
            framework_name=identity.framework_name,
            framework_version=identity.framework_version,
            adapter_version=identity.adapter_version,
            serving_abi_version=identity.serving_abi_version,
            binding_factory=getattr(intent, "binding_factory", None),
            configured_collective_policy=(
                materialization_request.configured_collective_policy
            ),
            source_bound_contract_state=(
                materialization_request.source_bound_contract_state
            ),
            source_bound_contract_path=(
                materialization_request.source_bound_contract_path
            ),
            execution_facts=materialization_request.execution_facts,
            operation_scope=materialization_request.operation_scope,
            require_materialization_options=(
                materialization_request.require_materialization_options
            ),
            model_runtime_spec=model_runtime_spec,
        )

    @staticmethod
    def _source_subject_for_mounted_source(
        *,
        source_artifact_ref: str,
        source_subject: Any,
    ) -> SourceSubject:
        if isinstance(source_subject, SourceSubject):
            subject_ref = tc_source_catalog.resolve_source_artifact_ref(
                source_subject.artifact_ref
            )
            if subject_ref != source_artifact_ref:
                raise ArtifactRuntimeIntegrationError(
                    "mounted-source subject artifact_ref does not match "
                    "realization artifact_ref"
                )
            return source_subject
        subject_artifact_ref = str(getattr(source_subject, "artifact_id", "") or "")
        if subject_artifact_ref and subject_artifact_ref != source_artifact_ref:
            raise ArtifactRuntimeIntegrationError(
                "mounted-source handle artifact_id does not match realization "
                "artifact_ref"
            )
        source_kind = (
            "public_disk" if is_public_disk_source_subject(source_subject) else "opaque"
        )
        return SourceSubject(
            artifact_ref=source_artifact_ref,
            subject=source_subject,
            source_kind=source_kind,
        )

    @staticmethod
    def _source_selector_for_subject(subject: SourceSubject) -> SourceSelector:
        source_path = getattr(subject.subject, "path", None)
        if source_path is None or not str(source_path).strip():
            raise ArtifactRuntimeIntegrationError(
                "mounted-source model_runtime realization requires a source "
                "selector or a source subject with a path"
            )
        return SourceSelector.local_path(str(source_path))

    def _host_source_subject_coordinator(
        self,
        framework_config: object | None,
    ) -> SourceSubjectCoordinator | None:
        if self.host is None or self.host.collective is None:
            return None
        return self.host.collective.source_subject_coordinator(framework_config)

    def _host_source_catalog_config(
        self,
        framework_config: Any | None,
        model_config: Any | None,
    ) -> Any | None:
        if self.host is None or self.host.source is None:
            return None
        return self.host.source.source_catalog_config(
            framework_config,
            model_config,
        )

    def _host_recipe_cache_policy(
        self,
        framework_config: Any | None,
        model_config: Any | None,
    ) -> RecipeCachePolicy | None:
        if self.host is None or self.host.source is None:
            return None
        policy = self.host.source.recipe_cache_policy(
            framework_config,
            model_config,
        )
        if policy is not None and not isinstance(policy, RecipeCachePolicy):
            raise ArtifactRuntimeIntegrationError(
                "IntegrationHost.source.recipe_cache_policy must return "
                "RecipeCachePolicy or None"
            )
        return policy

    def _run_local_ready_barrier(self, context: RequestContext) -> None:
        if self.host is None or self.host.collective is None:
            return
        self.host.collective.local_ready_barrier(
            context.framework_config,
            context.target_device,
        )

    @staticmethod
    def _worker_view_from_state(
        state: RuntimeBindingState | Any,
        *,
        decision: AdmissionDecision | None,
        include_reload_response: bool = False,
    ) -> RuntimeWorkerView:
        runtime_view = getattr(state, "runtime_view", None)
        if runtime_view is None:
            raise ArtifactRuntimeIntegrationError(
                "ArtifactRuntimeIntegration.describe requires state.runtime_view"
            )
        endpoint_fields = dict(decision.endpoint_fields) if decision else {}
        return RuntimeWorkerView.from_runtime_view(
            runtime_view,
            family=str(
                endpoint_fields.get(
                    "family",
                    decision.family if decision is not None else "",
                )
            ),
            tp_rank=_optional_int(endpoint_fields.get("tp_rank")),
            tp_world_size=_optional_int(endpoint_fields.get("tp_world_size")),
            include_reload_response=include_reload_response,
        )

    def _host_materialization_request(
        self,
        context: RequestContext,
        *,
        operation_scope: str,
    ) -> _HostMaterializationRequest:
        if self.host is None:
            return _HostMaterializationRequest(operation_scope=operation_scope)
        profile = self.host.runtime_profile or RuntimeProfile()
        return _HostMaterializationRequest(
            configured_collective_policy=_collective_policy_value(
                profile.materialization_policy
            ),
            source_bound_contract_state=read_source_bound_contract_state(),
            source_bound_contract_path=SOURCE_BOUND_CONTRACT_PATH_COLLECTIVE_FIRST_V4,
            execution_facts=_execution_facts_payload(
                self.host.placement.execution_facts(context.framework_config)
            ),
            operation_scope=operation_scope,
            require_materialization_options=True,
        )

    def _host_runtime_placement(
        self,
        framework_config: object | None,
    ) -> RuntimePlacement:
        if self.host is None:
            raise ArtifactRuntimeIntegrationError(
                "ArtifactRuntimeIntegration host placement requires IntegrationHost"
            )
        framework_payload = None
        framework_payload_fn = getattr(self.host.placement, "framework_payload", None)
        if callable(framework_payload_fn):
            framework_payload = _framework_payload_mapping(
                framework_payload_fn(framework_config)
            )
        identity_payload = None
        identity_payload_fn = getattr(self.host.placement, "identity_payload", None)
        if callable(identity_payload_fn):
            identity_payload = _framework_payload_mapping(
                identity_payload_fn(framework_config)
            )
        return runtime_placement_from_framework_facts(
            identity_facts=self.host.placement.identity_facts(framework_config),
            admission_facts=self.host.placement.admission_facts(framework_config),
            member_facts=self.host.placement.member_facts(framework_config),
            framework_payload=framework_payload,
            identity_payload=identity_payload,
        )

    @staticmethod
    def _runtime_binding_schema_version(profile: RuntimeProfile) -> int:
        value = profile.runtime_config.fields.get("runtime_binding_schema_version", 1)
        return _optional_int(value) or 1

    @staticmethod
    def _serving_artifact_schema_version(profile: RuntimeProfile) -> int:
        value = profile.manifest_policy.fields.get(
            "serving_artifact_schema_version", None
        )
        if value is None:
            model_fields = getattr(RuntimeArtifactManifest, "model_fields", {})
            schema_field = model_fields.get("schema_version")
            value = getattr(schema_field, "default", 1)
        return _optional_int(value) or 1

    def _load_existing_runtime_artifact(
        self, request: _DirectRuntimeLoad
    ) -> RuntimeLoadResult:
        target_device = self._require_target_device(request.target_device)
        context = self._framework_context(
            request.framework_config,
            request.model_config,
        )
        preflight = self._preflight_runtime_artifact(
            resolved_artifact=request.resolved_artifact,
            artifact_ref=request.artifact_ref,
            artifact_locator=request.artifact_locator,
            expected_tensor_schema_hash=None,
            policy=request.policy,
            placement=context.placement,
        )
        resolved = preflight.resolved_artifact
        policy = preflight.runtime_artifact_policy
        model = request.model
        if model is None:
            self._prepare_model_construction(
                request.framework_config,
                request.model_config,
            )
            model = self._build_meta_model(
                request.framework_config,
                request.model_config,
            )
        self._assert_model_ready_for_runtime_binding(
            model,
            context="TensorCast direct runtime artifact startup",
        )
        self._align_runtime_tensor_names(
            model,
            getattr(resolved, "tensor_names", ()),
        )
        current_tensors = self._collect_runtime_binding_tensors(
            model,
            remove_duplicate=False,
        )
        self._assert_tensor_names_match_expected(
            current_tensors,
            getattr(resolved, "tensor_names", ()),
        )
        tensor_schema_hash = self._compute_runtime_tensor_schema_hash(
            current_tensors,
            remove_duplicate=False,
        )
        preflight = self._preflight_runtime_artifact(
            resolved_artifact=resolved,
            artifact_ref=request.artifact_ref,
            artifact_locator=request.artifact_locator,
            expected_tensor_schema_hash=tensor_schema_hash,
            policy=policy,
            placement=context.placement,
        )
        resolved = preflight.resolved_artifact
        policy = preflight.runtime_artifact_policy
        manifest = getattr(resolved, "manifest", None)
        local_serving_ref = getattr(manifest, "local_serving_ref", None)
        if local_serving_ref:
            expected_member = request.expected_member
            if expected_member is None and context.placement is not None:
                expected_member = context.placement.member
            if expected_member is None:
                raise RestoreBindingError(
                    "ArtifactRuntimeIntegration._load_existing_runtime_artifact prepared "
                    "local-ready restore requires expected_member"
                )
            with restore_prepared_local_ready_binding(
                resolved_artifact=resolved,
                target_device=target_device,
                expected_member=expected_member,
                expected_tensor_schema_hash=tensor_schema_hash,
                expected_serving_build_digest=getattr(
                    manifest, "serving_build_digest", None
                ),
                caller_pid=os.getpid(),
                timeout_s=request.timeout_s,
            ) as restored:
                binding_result = RuntimeBindingResult.from_binding(restored)
                authority = getattr(restored, "authority", None)
                if authority is None:
                    artifact_report = _runtime_attachment_report_for_artifact_id(
                        artifact_id=str(getattr(resolved, "artifact_ref", "")),
                        tensors=binding_result.tensors,
                        binding_handle=restored,
                        target_device=target_device,
                        tensor_schema_hash=tensor_schema_hash,
                        artifact_profile="retained_binding",
                        authority_scope="daemon_retained_runtime_attachment",
                        source_selection=request.source_selection,
                        retained=True,
                        reservation_bytes=int(restored.reservation_bytes),
                    )
                else:
                    artifact_report = _runtime_attachment_report_for_retained(
                        authority=authority,
                        tensors=binding_result.tensors,
                        binding_handle=restored,
                        target_device=target_device,
                        tensor_schema_hash=tensor_schema_hash,
                        reservation_bytes=restored.reservation_bytes,
                        source_selection=request.source_selection,
                    )
                state_seed = self._state_seed(
                    resolved,
                    tensor_schema_hash=tensor_schema_hash,
                    execution_diagnostics=binding_result.execution_diagnostics,
                    materialization_diagnostics=(
                        binding_result.materialization_diagnostics
                    ),
                    binding_handle=restored,
                    artifact_realization_report=artifact_report,
                    readiness="runtime_local_ready",
                )
                runtime_state = self._materializer().attach_and_finalize(
                    model=model,
                    tensors=binding_result.tensors,
                    binding_handle=restored,
                    context=context,
                    state_seed=state_seed,
                    replace_meta_params=True,
                    target_device=target_device,
                    model_config=request.model_config,
                    model_runtime_spec=request.model_runtime_spec,
                )
        else:
            materialization = self._load_materialization_options(
                request,
                resolved,
            )
            binding_result = bind_runtime_artifact(
                resolved_artifact=resolved,
                tensor_names=tuple(current_tensors.keys()),
                device=target_device,
                runtime_artifact_policy=policy,
                options=materialization,
            )
            artifact_report = _runtime_attachment_report_for_resolved(
                resolved=resolved,
                tensors=binding_result.tensors,
                binding_handle=binding_result.binding,
                target_device=target_device,
                tensor_schema_hash=tensor_schema_hash,
                source_selection=request.source_selection,
                execution_diagnostics=binding_result.execution_diagnostics,
                materialization_diagnostics=binding_result.materialization_diagnostics,
                options=materialization,
            )
            state_seed = self._state_seed(
                resolved,
                tensor_schema_hash=tensor_schema_hash,
                execution_diagnostics=binding_result.execution_diagnostics,
                materialization_diagnostics=binding_result.materialization_diagnostics,
                binding_handle=binding_result.binding,
                artifact_realization_report=artifact_report,
            )
            runtime_state = self._materializer().attach_and_finalize(
                model=model,
                tensors=binding_result.tensors,
                binding_handle=binding_result.binding,
                context=context,
                state_seed=state_seed,
                replace_meta_params=True,
                target_device=target_device,
                model_config=request.model_config,
                model_runtime_spec=request.model_runtime_spec,
            )
        return RuntimeLoadResult(
            model=model,
            runtime_state=runtime_state,
            runtime_view=runtime_state.runtime_view,
            resolved_artifact=resolved,
            binding_result=binding_result,
        )

    def _reload_existing_runtime_artifact(
        self, request: _RuntimeReload
    ) -> RuntimeReloadResult:
        target_device = (
            torch.device(request.target_device)
            if request.target_device is not None
            else None
        )
        binding = getattr(request.current_state, "binding", None)
        if binding is None:
            raise ArtifactRuntimeIntegrationError(
                "ArtifactRuntimeIntegration._reload_existing_runtime_artifact requires current_state.binding"
            )
        if not is_runtime_binding_swap_capable(binding):
            raise ArtifactRuntimeIntegrationError(
                "ArtifactRuntimeIntegration._reload_existing_runtime_artifact requires a "
                "swap-capable runtime binding"
            )
        current_view = getattr(request.current_state, "runtime_view", None)
        expected_tensor_schema_hash = getattr(current_view, "tensor_schema_hash", None)
        runtime_tensors = None
        if request.model is not None:
            runtime_tensors = self._collect_runtime_binding_tensors(
                request.model,
                remove_duplicate=False,
            )
            expected_tensor_schema_hash = self._compute_runtime_tensor_schema_hash(
                runtime_tensors,
                remove_duplicate=False,
            )
            if target_device is None:
                for tensor in runtime_tensors.values():
                    target_device = torch.device(tensor.device)
                    break
        target_device = self._require_target_device(target_device)
        context = None
        if (
            request.model is not None
            or _artifact_locator_kind(request.artifact_locator) == "ranked_version_key"
        ):
            context = self._framework_context(
                request.framework_config,
                request.model_config,
            )
        placement = None if context is None else context.placement
        preflight = self._preflight_runtime_artifact(
            resolved_artifact=request.resolved_artifact,
            artifact_ref=request.artifact_ref,
            artifact_locator=request.artifact_locator,
            expected_tensor_schema_hash=expected_tensor_schema_hash,
            policy=request.policy,
            placement=placement,
        )
        resolved = preflight.resolved_artifact
        policy = preflight.runtime_artifact_policy
        materialization = self._reload_materialization_options(
            request,
            resolved,
        )
        binding_result = swap_runtime_artifact(
            binding=binding,
            resolved_artifact=resolved,
            tensor_names=(
                None if runtime_tensors is None else tuple(runtime_tensors.keys())
            ),
            runtime_artifact_policy=policy,
            options=materialization,
        )
        artifact_report = _runtime_attachment_report_for_resolved(
            resolved=resolved,
            tensors=binding_result.tensors,
            binding_handle=binding_result.binding,
            target_device=target_device,
            tensor_schema_hash=str(expected_tensor_schema_hash or ""),
            execution_diagnostics=binding_result.execution_diagnostics,
            materialization_diagnostics=binding_result.materialization_diagnostics,
            options=materialization,
        )
        state_seed = self._state_seed(
            resolved,
            tensor_schema_hash=str(expected_tensor_schema_hash or ""),
            execution_diagnostics=binding_result.execution_diagnostics,
            materialization_diagnostics=binding_result.materialization_diagnostics,
            binding_handle=binding_result.binding,
            artifact_realization_report=artifact_report,
        )
        if request.model is not None:
            context = context or self._framework_context(
                request.framework_config,
                request.model_config,
            )
            runtime_state = self._materializer().attach_and_finalize(
                model=request.model,
                tensors=binding_result.tensors,
                binding_handle=binding_result.binding,
                context=context,
                state_seed=state_seed,
                replace_meta_params=False,
                target_device=target_device,
                model_config=request.model_config,
                run_process_after_load=False,
            )
        else:
            realization_handle = _runtime_attachment_realization_handle(
                report=artifact_report,
                binding_handle=binding_result.binding,
                owner=(
                    getattr(request.current_state, "ownership_handle", None)
                    or binding_result.binding
                ),
            )
            runtime_state = RuntimeBindingState(
                binding=binding_result.binding,
                artifact_ref=state_seed.artifact_ref,
                runtime_view=state_seed.runtime_view(),
                ownership_handle=getattr(
                    request.current_state, "ownership_handle", None
                ),
                release_contract=None
                if realization_handle is None
                else realization_handle.release_contract,
                realization_handle=realization_handle,
            )
        return RuntimeReloadResult(
            runtime_state=runtime_state,
            runtime_view=runtime_state.runtime_view,
            resolved_artifact=resolved,
            binding_result=binding_result,
        )

    def _restore_retained_for_intent(
        self, request: _RetainedBindingAcquire
    ) -> RetainedBindingResult:
        target_device = self._require_target_device(request.target_device)
        authority = request.authority
        if authority is None and not request.local_serving_ref:
            raise RestoreBindingError(
                "ArtifactRuntimeIntegration._restore_retained_for_intent requires "
                "authority or local_serving_ref"
            )
        if authority is not None:
            rejection_reason = runtime_restore_rejection_reason(authority)
            if rejection_reason is not None:
                raise RestoreBindingError(rejection_reason)
        model = self._build_meta_model(
            request.framework_config,
            request.model_config,
        )
        try:
            with restore_retained_binding(
                authority=authority,
                local_serving_ref=request.local_serving_ref,
                target_device=target_device,
                expected_member=request.expected_member,
                expected_tensor_schema_hash=request.expected_tensor_schema_hash,
                expected_serving_build_digest=request.expected_serving_build_digest,
                expected_target_layout_hash=request.expected_target_layout_hash,
                expected_daemon_id=request.expected_daemon_id,
                expected_daemon_session_id=request.expected_daemon_session_id,
                serving_artifact_id=request.serving_artifact_id,
                caller_pid=os.getpid(),
                timeout_s=request.timeout_s,
                runtime=request.runtime,
                client=request.client,
                restore_fn=request.restore_fn,
            ) as restored:
                if authority is not None:
                    expected = getattr(authority, "expected", None)
                    expected_tensor_schema_hash = getattr(
                        expected, "tensor_schema_hash", None
                    )
                    artifact_report = _runtime_attachment_report_for_retained(
                        authority=authority,
                        tensors=restored.tensors,
                        binding_handle=restored,
                        target_device=target_device,
                        tensor_schema_hash=str(expected_tensor_schema_hash or ""),
                        reservation_bytes=restored.reservation_bytes,
                    )
                    artifact_ref = (
                        getattr(authority, "serving_artifact_id", None)
                        or getattr(authority, "local_serving_ref", None)
                        or ""
                    )
                    serving_artifact_ref = getattr(
                        authority, "serving_artifact_id", None
                    )
                    local_serving_ref = getattr(authority, "local_serving_ref", None)
                    readiness = str(
                        getattr(authority, "readiness", "") or "runtime_local_ready"
                    )
                    verification_state = str(
                        getattr(authority, "verification_state", "") or ""
                    )
                else:
                    expected_tensor_schema_hash = request.expected_tensor_schema_hash
                    artifact_ref = (
                        request.serving_artifact_id or request.local_serving_ref or ""
                    )
                    serving_artifact_ref = request.serving_artifact_id
                    local_serving_ref = request.local_serving_ref
                    readiness = "runtime_local_ready"
                    verification_state = ""
                    artifact_report = _runtime_attachment_report_for_artifact_id(
                        artifact_id=str(artifact_ref),
                        tensors=restored.tensors,
                        binding_handle=restored,
                        target_device=target_device,
                        tensor_schema_hash=str(expected_tensor_schema_hash or ""),
                        artifact_profile="retained_binding",
                        authority_scope="daemon_retained_runtime_attachment",
                        retained=True,
                        reservation_bytes=restored.reservation_bytes,
                    )
                expected_runtime_names = tuple(
                    str(name)
                    for name in restored.tensors
                    if not is_reserved_runtime_tensor_name(str(name))
                )
                self._align_runtime_tensor_names(model, expected_runtime_names)
                state_seed = RuntimeStateSeed(
                    artifact_ref=artifact_ref,
                    serving_artifact_ref=serving_artifact_ref,
                    tensor_schema_hash=str(expected_tensor_schema_hash or ""),
                    binding_value_ref=restored.binding_value_ref,
                    local_serving_ref=local_serving_ref,
                    readiness=readiness,
                    diagnostics={
                        "reservation_bytes": int(restored.reservation_bytes),
                        "verification_state": verification_state,
                        "artifact_realization_report": (
                            artifact_realization_report_to_dict(artifact_report)
                        ),
                    },
                    realization_report=artifact_report,
                )
                runtime_state = self._materializer().attach_and_finalize(
                    model=model,
                    tensors=restored.tensors,
                    binding_handle=restored,
                    context=self._framework_context(
                        request.framework_config,
                        request.model_config,
                    ),
                    state_seed=state_seed,
                    replace_meta_params=True,
                    target_device=target_device,
                    model_config=request.model_config,
                    run_process_after_load=False,
                    # Local-ready refs are admitted by daemon binding schema;
                    # authority restores still validate the full runtime model.
                    expected_tensor_schema_hash=(
                        expected_tensor_schema_hash if authority is not None else None
                    ),
                    model_runtime_spec=request.model_runtime_spec,
                )
                return RetainedBindingResult(
                    model=model,
                    runtime_state=runtime_state,
                    runtime_view=runtime_state.runtime_view,
                    restored=restored,
                )
        except (AttachFinalizeError, OwnershipTransferError, SchemaMismatchError):
            raise
        except Exception as exc:
            raise RestoreBindingError(
                "TensorCast retained binding acquire failed"
            ) from exc

    def _prepare_local_source_bootstrap(
        self, request: _LocalReadyBootstrap
    ) -> LocalReadyRuntimeResult:
        if (
            request.recipe is None or request.source_subject is None
        ) and request.build_recipe_from_framework_context:
            request = self._local_ready_prepare_with_built_recipe(request)
        if request.recipe is None or request.source_subject is None:
            self._lifecycle_not_implemented("_prepare_local_source_bootstrap", "P5")
        self._validate_prepared_local_ready_request_identity(request)
        if request.target_device is None:
            raise ArtifactRuntimeIntegrationError(
                "ArtifactRuntimeIntegration.start(LocalSourceBootstrap) requires target_device"
            )
        if not request.manifest_tensor_name:
            raise ArtifactRuntimeIntegrationError(
                "ArtifactRuntimeIntegration.start(LocalSourceBootstrap) requires manifest_tensor_name"
            )
        model = request.model
        if request.build_model_from_framework_context and model is None:
            if request.model_config is None:
                raise ArtifactRuntimeIntegrationError(
                    "ArtifactRuntimeIntegration.start(LocalSourceBootstrap) requires "
                    "model_config to build a framework model"
                )
            model = self._build_meta_model(
                request.framework_config, request.model_config
            )
        manifest_bytes = request.manifest_bytes
        serving_manifest_ref = request.serving_manifest_ref
        representation_contract_hash = request.representation_contract_hash
        serving_build_digest = request.serving_build_digest
        if request.build_manifest_carrier_from_framework_context and (
            manifest_bytes is None
            or not serving_manifest_ref
            or not representation_contract_hash
            or not serving_build_digest
        ):
            if request.model_config is None:
                raise ArtifactRuntimeIntegrationError(
                    "ArtifactRuntimeIntegration.start(LocalSourceBootstrap) requires "
                    "model_config to build a local-ready manifest carrier"
                )
            if request.placement is None:
                raise ArtifactRuntimeIntegrationError(
                    "ArtifactRuntimeIntegration.start(LocalSourceBootstrap) requires "
                    "placement to build a local-ready manifest carrier"
                )
            if request.runtime_binding_schema_version is None:
                raise ArtifactRuntimeIntegrationError(
                    "ArtifactRuntimeIntegration.start(LocalSourceBootstrap) requires "
                    "runtime_binding_schema_version to build a local-ready "
                    "manifest carrier"
                )
            if request.serving_artifact_schema_version is None:
                raise ArtifactRuntimeIntegrationError(
                    "ArtifactRuntimeIntegration.start(LocalSourceBootstrap) requires "
                    "serving_artifact_schema_version to build a local-ready "
                    "manifest carrier"
                )
            carrier = self.prepare_local_ready_manifest_carrier_from_framework_context(
                recipe=request.recipe,
                manifest_tensor_name=str(request.manifest_tensor_name),
                model_config=request.model_config,
                placement=request.placement,
                runtime_binding_schema_version=int(
                    request.runtime_binding_schema_version
                ),
                serving_artifact_schema_version=int(
                    request.serving_artifact_schema_version
                ),
                framework_name=request.framework_name,
                framework_version=request.framework_version,
                adapter_version=request.adapter_version,
                serving_abi_version=request.serving_abi_version,
            )
            manifest_bytes = carrier.manifest_bytes
            serving_manifest_ref = carrier.serving_manifest_ref
            representation_contract_hash = carrier.representation_contract_hash
            serving_build_digest = carrier.serving_build_digest
        options = request.options
        if model is not None:
            with tensorcast_profile_stage(
                "tensorcast",
                "local_ready.bootstrap.contract_preflight",
                logger=_LOGGER,
                device=request.target_device,
                extra={
                    "family": request.family,
                    "tp_rank": int(request.tp_rank),
                    "tp_world_size": int(request.tp_world_size),
                },
            ) as profile:
                self._assert_model_ready_for_runtime_binding(
                    model,
                    context="TensorCast local-ready binding realization",
                )
                self._align_runtime_tensor_names(
                    model,
                    self.local_ready_materialized_tensor_names(request.recipe),
                )
                canonical_tensors = self._collect_runtime_binding_tensors(
                    model,
                    remove_duplicate=False,
                )
                runtime_only_names = self.runtime_only_tensor_names(model)
                contract = self.build_local_ready_binding_contract(
                    recipe=request.recipe,
                    canonical_tensors=canonical_tensors,
                    runtime_only_tensor_names=runtime_only_names,
                    manifest_tensor_name=str(request.manifest_tensor_name),
                    manifest_bytes=manifest_bytes,
                    representation_contract_hash_factory=lambda tensor_schema_hash: "",
                )
                self._assert_local_ready_contract_realizable(
                    contract,
                    context="TensorCast local-ready binding realization",
                )
                if options is None:
                    options = self._local_ready_materialization_options(request)
                if profile is not None:
                    profile["canonical_tensor_count"] = len(canonical_tensors)
                    profile["runtime_only_tensor_count"] = len(runtime_only_names)
        with tensorcast_profile_stage(
            "tensorcast",
            "local_ready.bootstrap.realize_binding",
            logger=_LOGGER,
            device=request.target_device,
            extra={
                "family": request.family,
                "tp_rank": int(request.tp_rank),
                "tp_world_size": int(request.tp_world_size),
            },
        ):
            realization = tc_local_ready.realize_local_ready_binding_from_source(
                recipe=request.recipe,
                source_subject=request.source_subject,
                target_device=torch.device(request.target_device),
                manifest_tensor_name=str(request.manifest_tensor_name),
                manifest_bytes=manifest_bytes,
                options=options,
                binding_factory=request.binding_factory,
            )
        realized = LocalReadyRuntimeResult(
            recipe=request.recipe,
            binding=realization.binding,
            update_epoch=realization.update_epoch,
            layout=realization.layout,
            realization_entry_count=realization.realization_entry_count,
            realization=realization,
        )
        if self._local_ready_prepare_has_finalize_fields(
            request,
            model=model,
            serving_manifest_ref=serving_manifest_ref,
            representation_contract_hash=representation_contract_hash,
            serving_build_digest=serving_build_digest,
        ):
            run_process_after_load = request.run_process_after_load
            run_semantic_validation = request.run_semantic_validation
            if (
                request.run_binding_finalize_hooks_when_required
                and self.local_ready_requires_binding_finalize(request.recipe)
            ):
                run_process_after_load = True
                run_semantic_validation = True
            with tensorcast_profile_stage(
                "tensorcast",
                "local_ready.bootstrap.finalize_runtime",
                logger=_LOGGER,
                device=request.target_device,
                extra={
                    "family": request.family,
                    "tp_rank": int(request.tp_rank),
                    "tp_world_size": int(request.tp_world_size),
                },
            ):
                finalized = self._finalize_local_ready_runtime(
                    _LocalReadyFinalize(
                        model=model,
                        recipe=request.recipe,
                        binding=realization.binding,
                        update_epoch=realization.update_epoch,
                        source_artifact_ref=str(request.source_artifact_ref),
                        source_selection=request.source_selection,
                        serving_manifest_ref=str(serving_manifest_ref),
                        representation_contract_hash=str(representation_contract_hash),
                        serving_build_digest=str(serving_build_digest),
                        manifest_tensor_name=str(request.manifest_tensor_name),
                        source_bound_contract_state=request.source_bound_contract_state,
                        source_bound_contract_path=str(
                            request.source_bound_contract_path
                        ),
                        target_device=request.target_device,
                        manifest_bytes=manifest_bytes,
                        framework_config=request.framework_config,
                        model_config=request.model_config,
                        placement=request.placement,
                        family=request.family,
                        tp_rank=request.tp_rank,
                        tp_world_size=request.tp_world_size,
                        replace_meta_params=request.replace_meta_params,
                        run_process_after_load=run_process_after_load,
                        run_post_bind_finalize=request.run_post_bind_finalize,
                        run_semantic_validation=run_semantic_validation,
                        semantic_validation_spec=request.semantic_validation_spec,
                        validate_representation_contract_hash=request.validate_representation_contract_hash,
                        runtime_binding_schema_version=request.runtime_binding_schema_version,
                        serving_artifact_schema_version=request.serving_artifact_schema_version,
                        framework_name=request.framework_name,
                        framework_version=request.framework_version,
                        adapter_version=request.adapter_version,
                        serving_abi_version=request.serving_abi_version,
                        model_runtime_spec=request.model_runtime_spec,
                    )
                )
            return LocalReadyRuntimeResult(
                model=finalized.model,
                runtime_state=finalized.runtime_state,
                runtime_view=finalized.runtime_view,
                prepared=finalized.prepared,
                binding_value=finalized.binding_value,
                recipe=request.recipe,
                current_value=finalized.current_value,
                binding=finalized.binding,
                update_epoch=finalized.update_epoch,
                layout=realized.layout,
                realization_entry_count=realized.realization_entry_count,
                realization=realized.realization,
                realization_report=finalized.realization_report,
            )
        return realized

    def _local_ready_prepare_with_built_recipe(
        self,
        request: _LocalReadyBootstrap,
    ) -> _LocalReadyBootstrap:
        profile_extra = {
            "family": request.family,
            "target_device": str(request.target_device),
            "tp_rank": int(request.tp_rank),
            "tp_world_size": int(request.tp_world_size),
        }
        source_subject_record = request.source_subject
        if source_subject_record is None:
            with tensorcast_profile_stage(
                "tensorcast",
                "local_ready.prepare.resolve_source_subject",
                logger=_LOGGER,
                device=request.target_device,
                extra=profile_extra,
            ) as profile:
                source_subject_record = self._resolve_local_ready_source_subject(
                    request
                )
                if profile is not None:
                    profile["source_subject_type"] = type(
                        source_subject_record
                    ).__name__
        source_artifact_ref = request.source_artifact_ref or getattr(
            source_subject_record, "artifact_ref", None
        )
        if not source_artifact_ref:
            raise ArtifactRuntimeIntegrationError(
                "ArtifactRuntimeIntegration.start(LocalSourceBootstrap) could not "
                "derive source_artifact_ref from source subject"
            )
        with tensorcast_profile_stage(
            "tensorcast",
            "local_ready.prepare.resolve_source_artifact_ref",
            logger=_LOGGER,
            device=request.target_device,
            extra=profile_extra,
        ) as profile:
            try:
                source_artifact_ref = tc_source_catalog.resolve_source_artifact_ref(
                    source_artifact_ref
                )
            except ValueError as exc:
                raise ArtifactRuntimeIntegrationError(
                    "ArtifactRuntimeIntegration.start(LocalSourceBootstrap) requires "
                    "a real source artifact identity"
                ) from exc
            if profile is not None:
                profile["source_artifact_ref_kind"] = getattr(
                    source_artifact_ref, "kind", ""
                )
        source_realization_subject = getattr(
            source_subject_record, "subject", source_subject_record
        )
        placement = request.placement
        if placement is None and self.host is not None:
            with tensorcast_profile_stage(
                "tensorcast",
                "local_ready.prepare.framework_placement",
                logger=_LOGGER,
                device=request.target_device,
                extra=profile_extra,
            ):
                placement = self._framework_context(
                    request.framework_config,
                    request.model_config,
                ).placement
        with tensorcast_profile_stage(
            "tensorcast",
            "local_ready.prepare.build_source_catalog",
            logger=_LOGGER,
            device=request.target_device,
            extra=profile_extra,
        ) as profile:
            source_catalog = self._local_ready_source_catalog(
                request,
                source_subject=source_subject_record,
                source_artifact_ref=str(source_artifact_ref),
            )
            if profile is not None:
                ordered_names = getattr(source_catalog, "ordered_names", ())
                profile["source_tensor_count"] = len(ordered_names)
        with tensorcast_profile_stage(
            "tensorcast",
            "local_ready.prepare.recipe_cache_config",
            logger=_LOGGER,
            device=request.target_device,
            extra=profile_extra,
        ) as profile:
            cache_config = self._local_ready_recipe_cache_config(
                request,
                source_catalog=source_catalog,
            )
            if profile is not None:
                profile["cache_config_type"] = type(cache_config).__name__
        with tensorcast_profile_stage(
            "tensorcast",
            "local_ready.prepare.build_recipe",
            logger=_LOGGER,
            device=request.target_device,
            extra=profile_extra,
        ) as profile:
            recipe = self._build_local_ready_recipe_from_framework_context(
                request,
                source_subject=source_subject_record,
                source_artifact_ref=str(source_artifact_ref),
                source_catalog=source_catalog,
                cache_config=cache_config,
                placement=placement,
            )
            if profile is not None:
                profile["tensor_schema_count"] = len(
                    getattr(recipe, "tensor_schema", ())
                )
                profile["realization_fallback_count"] = len(
                    getattr(recipe, "realization_fallback_plan", ())
                )
        return replace(
            request,
            recipe=recipe,
            source_catalog=source_catalog,
            cache_config=cache_config,
            source_subject=source_realization_subject,
            source_artifact_ref=str(source_artifact_ref),
            placement=placement,
        )

    def _resolve_local_ready_source_subject(
        self,
        request: _LocalReadyBootstrap,
    ) -> SourceSubject:
        if request.source_selector is None:
            raise ArtifactRuntimeIntegrationError(
                "ArtifactRuntimeIntegration.start(LocalSourceBootstrap) requires "
                "source_selector when source_subject is not supplied"
            )
        verify_checksums = bool(
            getattr(request.bootstrap, "verify_source_checksums", False)
        )
        return self.resolve_source_subject(
            request.source_selector,
            verify_checksums=verify_checksums,
            coordinator=request.source_subject_coordinator,
        )

    def _local_ready_source_catalog(
        self,
        request: _LocalReadyBootstrap,
        *,
        source_subject: Any,
        source_artifact_ref: str,
    ) -> Any:
        try:
            expected_source_ref = tc_source_catalog.resolve_source_artifact_ref(
                source_artifact_ref
            )
        except ValueError as exc:
            raise ArtifactRuntimeIntegrationError(
                "ArtifactRuntimeIntegration.start(LocalSourceBootstrap) requires "
                "a real source artifact identity"
            ) from exc
        if request.source_catalog is not None:
            self._validate_source_catalog_artifact_ref(
                request.source_catalog,
                expected_source_artifact_ref=expected_source_ref,
            )
            return request.source_catalog
        if self.host is not None and self.host.source_catalog is not None:
            if not isinstance(request.source_selector, SourceSelector):
                raise ArtifactRuntimeIntegrationError(
                    "IntegrationHost.source_catalog requires a core SourceSelector"
                )
            if request.model_config is None:
                raise ArtifactRuntimeIntegrationError(
                    "IntegrationHost.source_catalog requires model_config"
                )
            source_catalog = self.host.source_catalog.build_catalog(
                SourceCatalogRequest(
                    source_subject=source_subject,
                    source_selector=request.source_selector,
                    source_artifact_ref=expected_source_ref,
                    framework_identity=self.host.framework.identity(
                        request.model_config
                    ),
                    framework_config=request.framework_config,
                    model_config=request.model_config,
                    download_policy=(
                        request.source_catalog_config
                        if isinstance(
                            request.source_catalog_config, SourceDownloadPolicy
                        )
                        else None
                    ),
                    cache_policy=(
                        request.cache_config
                        if isinstance(request.cache_config, RecipeCachePolicy)
                        else None
                    ),
                    source_catalog_config=request.source_catalog_config,
                )
            )
            self._validate_source_catalog_artifact_ref(
                source_catalog,
                expected_source_artifact_ref=expected_source_ref,
            )
            return source_catalog
        raise _capability_missing(
            "ArtifactRuntimeIntegration.start(LocalSourceBootstrap) requires "
            "IntegrationHost.source_catalog when recipe is not supplied",
            level="level2-local-bootstrap",
            capability="source_catalog",
            operation="local_bootstrap.source_catalog",
            required_methods=("build_catalog",),
            next_action=(
                "Add IntegrationHost(source_catalog=...) or provide a prepared "
                "recipe through the admin/offline bootstrap path."
            ),
        )

    @staticmethod
    def _validate_source_catalog_artifact_ref(
        source_catalog: Any,
        *,
        expected_source_artifact_ref: str,
    ) -> None:
        catalog_artifact_ref = getattr(source_catalog, "source_artifact_ref", None)
        if catalog_artifact_ref is None:
            raise ArtifactRuntimeIntegrationError(
                "SourceCatalogProvider returned a catalog without a real "
                "source_artifact_ref"
            )
        try:
            catalog_source_ref = tc_source_catalog.resolve_source_artifact_ref(
                str(catalog_artifact_ref)
            )
        except ValueError as exc:
            raise ArtifactRuntimeIntegrationError(
                "SourceCatalogProvider returned a catalog without a real "
                "source_artifact_ref"
            ) from exc
        if catalog_source_ref != expected_source_artifact_ref:
            raise ArtifactRuntimeIntegrationError(
                "SourceCatalogProvider returned source_artifact_ref "
                f"{catalog_source_ref!r}, expected {expected_source_artifact_ref!r}"
            )

    @staticmethod
    def _local_ready_recipe_cache_config(
        request: _LocalReadyBootstrap,
        *,
        source_catalog: Any,
    ) -> Any:
        cache_config_factory = request.cache_config_factory
        if callable(cache_config_factory):
            return cache_config_factory(source_catalog=source_catalog)
        if isinstance(request.cache_config, RecipeCachePolicy):
            return recipe_build_cache_config_from_policy(
                request.cache_config,
                source_catalog=source_catalog,
            )
        if request.cache_config is not None:
            return request.cache_config
        return RecipeBuildCacheConfig()

    def _build_local_ready_recipe_from_framework_context(
        self,
        request: _LocalReadyBootstrap,
        *,
        source_subject: Any,
        source_artifact_ref: str,
        source_catalog: Any,
        cache_config: Any,
        placement: Any | None,
    ) -> Any:
        if request.model_config is None:
            raise ArtifactRuntimeIntegrationError(
                "ArtifactRuntimeIntegration.start(LocalSourceBootstrap) requires "
                "model_config when recipe is not supplied"
            )
        adapter = self._recipe_framework_adapter(request.model_config)
        recipe_session = self.build_recipe_session(
            RecipeBuildSessionRequest(
                source_subject=source_subject,
                framework_config=request.framework_config,
                model_config=request.model_config,
                placement=placement,
                cache_config=cache_config,
            )
        )
        result = recipe_session.build_recipe(
            model_config=request.model_config,
            framework_config=request.framework_config,
            source_catalog=source_catalog,
            framework_adapter=adapter,
            build_meta_model=lambda: self._build_meta_model(
                request.framework_config,
                request.model_config,
            ),
            cache_config=cache_config,
            is_reserved_runtime_tensor_name=is_reserved_runtime_tensor_name,
            semantic_validation_spec=request.semantic_validation_spec,
            placement=placement,
            debug_extra={
                "source_artifact_ref": source_artifact_ref,
            },
            profile_sink=self._recipe_profile_sink(),
        )
        return result.recipe

    def _recipe_profile_sink(self) -> Any | None:
        if callable(self.profile_sink):
            return self.profile_sink

        def sink(payload: dict[str, Any]) -> None:
            stage = str(payload.get("stage") or "recipe.event")
            emit_tensorcast_profile_event(
                "tensorcast",
                stage,
                payload=payload,
                logger=_LOGGER,
            )

        return sink

    @staticmethod
    def _local_ready_prepare_has_finalize_fields(
        request: _LocalReadyBootstrap,
        *,
        model: Any | None = None,
        serving_manifest_ref: str | None = None,
        representation_contract_hash: str | None = None,
        serving_build_digest: str | None = None,
    ) -> bool:
        return all(
            (
                model is not None,
                request.source_artifact_ref,
                serving_manifest_ref,
                representation_contract_hash,
                serving_build_digest,
                request.source_bound_contract_state is not None,
                request.source_bound_contract_path,
            )
        )

    def _validate_prepared_local_ready_request_identity(
        self,
        request: _LocalReadyBootstrap,
    ) -> None:
        if request.recipe is None:
            return
        try:
            recipe_identity = self.local_ready_materialization_identity(request.recipe)
            recipe_source_ref = tc_source_catalog.resolve_source_artifact_ref(
                recipe_identity.source_artifact_ref
            )
        except (AttributeError, ValueError) as exc:
            raise ArtifactRuntimeIntegrationError(
                "TensorCast local-ready prepared recipe requires a real "
                "source artifact identity"
            ) from exc

        expected_source_ref = request.source_artifact_ref
        if expected_source_ref:
            try:
                expected_source_ref = tc_source_catalog.resolve_source_artifact_ref(
                    expected_source_ref
                )
            except ValueError as exc:
                raise ArtifactRuntimeIntegrationError(
                    "ArtifactRuntimeIntegration.start(LocalSourceBootstrap) "
                    "requires a real source artifact identity"
                ) from exc
            if recipe_source_ref != expected_source_ref:
                raise ArtifactRuntimeIntegrationError(
                    "TensorCast local-ready prepared recipe source_artifact_ref "
                    f"{recipe_source_ref!r} does not match request "
                    f"source_artifact_ref {expected_source_ref!r}"
                )

        source_subject_ref = getattr(request.source_subject, "artifact_ref", None)
        if source_subject_ref:
            try:
                source_subject_ref = tc_source_catalog.resolve_source_artifact_ref(
                    source_subject_ref
                )
            except ValueError as exc:
                raise ArtifactRuntimeIntegrationError(
                    "TensorCast local-ready source subject requires a real "
                    "source artifact identity"
                ) from exc
            if recipe_source_ref != source_subject_ref:
                raise ArtifactRuntimeIntegrationError(
                    "TensorCast local-ready prepared recipe source_artifact_ref "
                    f"{recipe_source_ref!r} does not match source subject "
                    f"artifact_ref {source_subject_ref!r}"
                )

        if request.source_catalog is not None:
            self._validate_source_catalog_artifact_ref(
                request.source_catalog,
                expected_source_artifact_ref=recipe_source_ref,
            )
            catalog_fingerprint = getattr(
                request.source_catalog, "metadata_fingerprint", None
            )
            if catalog_fingerprint is not None and str(catalog_fingerprint) != str(
                recipe_identity.source_metadata_fingerprint
            ):
                raise ArtifactRuntimeIntegrationError(
                    "TensorCast local-ready prepared recipe source metadata "
                    "fingerprint does not match prepared source catalog"
                )

    def _local_ready_materialization_options(
        self,
        request: _LocalReadyBootstrap,
    ) -> Any | None:
        execution_facts = self._request_execution_facts(request)
        if (
            request.configured_collective_policy is None
            or request.source_bound_contract_state is None
            or not request.source_bound_contract_path
            or execution_facts is None
        ):
            if request.require_materialization_options:
                raise ArtifactRuntimeIntegrationError(
                    "ArtifactRuntimeIntegration.start(LocalSourceBootstrap) requires "
                    "materialization execution context"
                )
            return None
        if request.require_materialization_options and not getattr(
            request.source_bound_contract_state,
            "source_bound_contract_ready",
            False,
        ):
            raise ArtifactRuntimeIntegrationError(
                "ArtifactRuntimeIntegration.start(LocalSourceBootstrap) requires "
                "ready source-bound contract state"
            )
        identity = self.local_ready_materialization_identity(request.recipe)
        options, _profile = self.build_materialization_options(
            artifact_ref=identity.source_artifact_ref,
            operation_scope=request.operation_scope,
            configured_policy=request.configured_collective_policy,
            source_bound_contract_state=request.source_bound_contract_state,
            source_bound_contract_path=request.source_bound_contract_path,
            execution_facts=execution_facts,
            contract_identity=(
                request.contract_identity or identity.source_metadata_fingerprint
            ),
        )
        return options

    def _request_execution_facts(self, request: Any) -> Mapping[str, Any] | None:
        execution_facts = getattr(request, "execution_facts", None)
        if execution_facts is not None:
            return execution_facts
        if self.host is None:
            return None
        return _execution_facts_payload(
            self.host.placement.execution_facts(
                getattr(request, "framework_config", None)
            )
        )

    @staticmethod
    def _assert_local_ready_contract_realizable(
        contract: LocalReadyBindingContract,
        *,
        context: str,
    ) -> None:
        if contract.realization_entry_count <= 0:
            raise ArtifactRuntimeIntegrationError(
                f"{context} requires a non-empty BindingRealizationPlan"
            )
        if not contract.fallback_copy_plan:
            return
        unsupported = ", ".join(
            f"{getattr(entry, 'op', '')}:{getattr(entry, 'dst_name', '')}"
            for entry in contract.fallback_copy_plan[:8]
        )
        if len(contract.fallback_copy_plan) > 8:
            unsupported = f"{unsupported}, ..." if unsupported else "..."
        raise ArtifactRuntimeIntegrationError(
            f"{context} requires a fully representable BindingRealizationPlan; "
            f"unsupported_entries={len(contract.fallback_copy_plan)} "
            f"[{unsupported}]"
        )

    def _finalize_local_ready_runtime(
        self, request: _LocalReadyFinalize
    ) -> LocalReadyRuntimeResult:
        target_device = self._require_target_device(request.target_device)
        if request.recipe is None:
            raise ArtifactRuntimeIntegrationError(
                "ArtifactRuntimeIntegration._finalize_local_ready_runtime requires recipe"
            )
        if request.model is None:
            raise ArtifactRuntimeIntegrationError(
                "ArtifactRuntimeIntegration._finalize_local_ready_runtime requires model"
            )
        if request.binding is None:
            raise ArtifactRuntimeIntegrationError(
                "ArtifactRuntimeIntegration._finalize_local_ready_runtime requires binding"
            )
        if request.update_epoch is None:
            raise ArtifactRuntimeIntegrationError(
                "ArtifactRuntimeIntegration._finalize_local_ready_runtime requires update_epoch"
            )
        if not request.manifest_tensor_name:
            raise ArtifactRuntimeIntegrationError(
                "ArtifactRuntimeIntegration._finalize_local_ready_runtime requires manifest_tensor_name"
            )
        try:
            with tensorcast_profile_stage(
                "tensorcast",
                "local_ready.finalize.preflight",
                logger=_LOGGER,
                device=target_device,
                extra={
                    "family": request.family,
                    "tp_rank": int(request.tp_rank),
                    "tp_world_size": int(request.tp_world_size),
                },
            ) as profile:
                framework_context = self._framework_context(
                    request.framework_config,
                    request.model_config,
                )
                self._assert_local_ready_binding_tensor_set(
                    recipe=request.recipe,
                    binding=request.binding,
                    manifest_tensor_name=str(request.manifest_tensor_name),
                )
                tensor_schema_hash = self.local_ready_tensor_schema_hash(
                    recipe=request.recipe,
                    manifest_tensor_name=str(request.manifest_tensor_name),
                    manifest_bytes=request.manifest_bytes,
                )
                self._validate_local_ready_representation_contract_hash(
                    request,
                    tensor_schema_hash=tensor_schema_hash,
                )
                semantic_validation_spec = self._local_ready_semantic_validation_spec(
                    request
                )
                self._assert_local_ready_finalize_admitted(
                    request,
                    semantic_validation_spec=semantic_validation_spec,
                )
                if profile is not None:
                    profile["tensor_schema_hash"] = tensor_schema_hash
            with tensorcast_profile_stage(
                "tensorcast",
                "local_ready.finalize.attach_and_finalize",
                logger=_LOGGER,
                device=target_device,
                extra={
                    "family": request.family,
                    "tp_rank": int(request.tp_rank),
                    "tp_world_size": int(request.tp_world_size),
                    "run_process_after_load": bool(request.run_process_after_load),
                    "run_post_bind_finalize": bool(request.run_post_bind_finalize),
                    "run_semantic_validation": bool(request.run_semantic_validation),
                },
            ):
                self._materializer().attach_and_finalize(
                    model=request.model,
                    tensors=_binding_tensors(request.binding),
                    binding_handle=request.binding,
                    context=framework_context,
                    state_seed=RuntimeStateSeed(
                        artifact_ref=str(request.source_artifact_ref),
                        source_artifact_ref=str(request.source_artifact_ref),
                        representation_contract_hash=str(
                            request.representation_contract_hash
                        ),
                        tensor_schema_hash=tensor_schema_hash,
                        readiness="runtime_local_ready",
                    ),
                    replace_meta_params=bool(request.replace_meta_params),
                    target_device=target_device,
                    model_config=request.model_config,
                    run_process_after_load=bool(request.run_process_after_load),
                    run_post_bind_finalize=bool(request.run_post_bind_finalize),
                    semantic_validation_spec=semantic_validation_spec,
                )
            with tensorcast_profile_stage(
                "tensorcast",
                "local_ready.finalize.validate_and_freeze",
                logger=_LOGGER,
                device=target_device,
                extra={
                    "family": request.family,
                    "tp_rank": int(request.tp_rank),
                    "tp_world_size": int(request.tp_world_size),
                },
            ) as profile:
                tensors = self._collect_runtime_binding_tensors(
                    request.model,
                    remove_duplicate=False,
                )
                self.validate_local_ready_tensor_schema(
                    recipe=request.recipe,
                    tensors=tensors,
                )
                current_value = self.freeze_local_ready(
                    binding=request.binding,
                    update_epoch=request.update_epoch,
                    source_artifact_ref=str(request.source_artifact_ref),
                )
                if profile is not None:
                    profile["runtime_tensor_count"] = len(tensors)
            source_ref = str(request.source_artifact_ref)
            artifact_profile = (
                "mounted_source"
                if source_ref.startswith("msa1:")
                else "local_ready_source_artifact"
            )
            authority_scope = (
                "daemon_local_mounted_source"
                if source_ref.startswith("msa1:")
                else "daemon_mediated_local_ready_runtime_attachment"
            )
            with tensorcast_profile_stage(
                "tensorcast",
                "local_ready.finalize.build_runtime_view",
                logger=_LOGGER,
                device=target_device,
                extra={
                    "family": request.family,
                    "tp_rank": int(request.tp_rank),
                    "tp_world_size": int(request.tp_world_size),
                    "artifact_profile": artifact_profile,
                },
            ):
                artifact_report = _runtime_attachment_report_for_artifact_id(
                    artifact_id=source_ref,
                    tensors=_binding_tensors(request.binding),
                    binding_handle=request.binding,
                    target_device=target_device,
                    tensor_schema_hash=tensor_schema_hash,
                    artifact_profile=artifact_profile,
                    authority_scope=authority_scope,
                    source_selection=request.source_selection,
                )
                prepared = build_local_ready_prepared_artifact(
                    source_artifact_ref=str(request.source_artifact_ref),
                    serving_manifest_ref=str(request.serving_manifest_ref),
                    representation_contract_hash=str(
                        request.representation_contract_hash
                    ),
                    serving_build_digest=str(request.serving_build_digest),
                    tensor_schema_hash=tensor_schema_hash,
                    current_value=current_value,
                    binding=request.binding,
                    family=str(request.family),
                    tp_rank=int(request.tp_rank),
                    tp_world_size=int(request.tp_world_size),
                    source_bound_contract_state=request.source_bound_contract_state,
                    source_bound_contract_path=str(request.source_bound_contract_path),
                    artifact_realization_report=artifact_report,
                    model_runtime_spec=(
                        _model_runtime_spec_with_context_defaults(
                            spec=request.model_runtime_spec,
                            context=framework_context,
                            target_device=target_device,
                        )
                        if request.model_runtime_spec is not None
                        else _model_runtime_spec_for_context(
                            context=framework_context,
                            target_device=target_device,
                        )
                    ),
                )
            return LocalReadyRuntimeResult(
                model=request.model,
                runtime_state=prepared.runtime_state,
                runtime_view=prepared.runtime_view,
                prepared=prepared.prepared,
                binding_value=prepared.binding_value,
                recipe=request.recipe,
                current_value=current_value,
                binding=request.binding,
                update_epoch=request.update_epoch,
                realization_report=artifact_report,
            )
        except Exception:
            _close_quietly(request.binding)
            raise

    def _assert_local_ready_finalize_admitted(
        self,
        request: _LocalReadyFinalize,
        *,
        semantic_validation_spec: Any | None,
    ) -> None:
        if not self.local_ready_requires_binding_finalize(request.recipe):
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

    @staticmethod
    def _local_ready_semantic_validation_spec(
        request: _LocalReadyFinalize,
    ) -> Any | None:
        if request.semantic_validation_spec is not None:
            return request.semantic_validation_spec
        if not request.run_semantic_validation:
            return None
        return getattr(request.recipe, "semantic_validation_spec", None)

    def _validate_local_ready_representation_contract_hash(
        self,
        request: _LocalReadyFinalize,
        *,
        tensor_schema_hash: str,
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
        actual = self.local_ready_representation_contract_hash(
            tensor_schema_hash=tensor_schema_hash,
            model_config=request.model_config,
            placement=request.placement,
            runtime_binding_schema_version=int(request.runtime_binding_schema_version),
            serving_artifact_schema_version=int(
                request.serving_artifact_schema_version
            ),
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

    def build_local_ready_manifest_carrier(
        self,
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

    def build_local_ready_manifest_carrier_from_contract(
        self,
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
        logical_topology_json_payload = (
            tc_local_ready.logical_topology_json_from_recipe(
                recipe,
                topology=topology,
                framework_payload=dict(framework_payload or {}),
            )
        )
        topology_admission_digest = _optional_text(
            getattr(topology, "schema_topology_digest", None)
        )
        return self.build_local_ready_manifest_carrier(
            recipe=recipe,
            manifest_tensor_name=manifest_tensor_name,
            representation_contract_hash=representation_contract_hash,
            logical_topology_json_payload=logical_topology_json_payload,
            topology_admission_digest=topology_admission_digest,
        )

    def local_ready_representation_contract_hash(
        self,
        *,
        tensor_schema_hash: str,
        model_config: Any,
        placement: Any,
        runtime_binding_schema_version: int,
        serving_artifact_schema_version: int,
        framework_name: str | None = None,
        framework_version: str | None = None,
        adapter_version: str | None = None,
        serving_abi_version: str | None = None,
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
            stable_identity_payload = getattr(
                placement, "stable_identity_payload", None
            )
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
            framework_name=framework_name
            or self._framework_identity(model_config).framework_name,
            framework_version=framework_version
            or self._framework_identity(model_config).framework_version,
            adapter_version=adapter_version
            or self._framework_identity(model_config).adapter_version,
            serving_abi_version=serving_abi_version
            or self._framework_identity(model_config).serving_abi_version,
            source_identity=source_identity,
        )

    def build_local_ready_manifest_carrier_from_framework_context(
        self,
        *,
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
    ) -> tuple[str, bytes]:
        return self.build_local_ready_manifest_carrier_from_contract(
            recipe=recipe,
            manifest_tensor_name=manifest_tensor_name,
            representation_contract_hash_factory=lambda tensor_schema_hash: (
                self.local_ready_representation_contract_hash(
                    tensor_schema_hash=tensor_schema_hash,
                    model_config=model_config,
                    placement=placement,
                    runtime_binding_schema_version=runtime_binding_schema_version,
                    serving_artifact_schema_version=serving_artifact_schema_version,
                    framework_name=framework_name,
                    framework_version=framework_version,
                    adapter_version=adapter_version,
                    serving_abi_version=serving_abi_version,
                )
            ),
            topology=getattr(placement, "topology", None),
            framework_payload=getattr(placement, "framework_payload", {}),
        )

    def prepare_local_ready_manifest_carrier_from_framework_context(
        self,
        *,
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
        representation_contract_hash, manifest_bytes = (
            self.build_local_ready_manifest_carrier_from_framework_context(
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
        )
        manifest = RuntimeArtifactManifest.from_bytes(manifest_bytes)
        return LocalReadyManifestCarrierResult(
            representation_contract_hash=representation_contract_hash,
            manifest_bytes=manifest_bytes,
            serving_manifest_ref=manifest.serving_manifest_ref,
            serving_build_digest=manifest.serving_build_digest,
        )

    def local_ready_tensor_schema_hash(
        self,
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

    def local_ready_materialized_tensor_names(
        self,
        recipe: Any,
    ) -> tuple[str, ...]:
        return tuple(
            str(entry.name)
            for entry in tc_local_ready.materialized_tensor_schema(recipe)
        )

    def _assert_local_ready_binding_tensor_set(
        self,
        *,
        recipe: Any,
        binding: Any,
        manifest_tensor_name: str,
    ) -> None:
        expected_names = tuple(
            sorted(self.local_ready_materialized_tensor_names(recipe))
        )
        actual_names = tuple(
            sorted(
                str(name)
                for name in _binding_tensors(binding)
                if str(name) != manifest_tensor_name
            )
        )
        if actual_names == expected_names:
            return
        raise SchemaMismatchError(
            "TensorCast local-ready binding tensor set does not match recipe "
            "schema: "
            f"expected={list(expected_names)}, actual={list(actual_names)}"
        )

    def build_local_ready_binding_contract(
        self,
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
        tensor_schema_hash = self.local_ready_tensor_schema_hash(
            recipe=recipe,
            manifest_tensor_name=manifest_tensor_name,
            manifest_bytes=manifest_bytes,
        )
        return LocalReadyBindingContract(
            excluded_names=tuple(
                sorted(str(name) for name in runtime_only_tensor_names)
            ),
            canonical_tensor_names=tuple(
                sorted(str(name) for name in canonical_tensors)
            ),
            tensor_schema_hash=tensor_schema_hash,
            representation_contract_hash=representation_contract_hash_factory(
                tensor_schema_hash
            ),
            mapped_copy_plan=(),
            realization_plan_proto=realization_plan_proto,
            realization_entry_count=realization_entry_count,
            fallback_copy_plan=tuple(recipe.realization_fallback_plan),
        )

    def local_ready_recipe_summary_fields(self, recipe: Any) -> dict[str, int]:
        return RecipeBuildSession.recipe_summary_fields(recipe)

    def local_ready_materialization_identity(
        self,
        recipe: Any,
    ) -> LocalReadyMaterializationIdentity:
        return LocalReadyMaterializationIdentity(
            source_artifact_ref=str(recipe.source_artifact_ref),
            source_metadata_fingerprint=str(recipe.source_metadata_fingerprint),
        )

    def local_ready_requires_binding_finalize(self, recipe: Any) -> bool:
        runtime_facts = getattr(recipe, "runtime_facts", None)
        process_after_load_class = tc_readiness.coerce_finalize_class(
            getattr(runtime_facts, "process_after_load_class", None),
            default=FinalizeClass.RUNTIME_ONLY,
        )
        return process_after_load_class == FinalizeClass.REPRESENTATION_CHANGING

    def validate_local_ready_tensor_schema(
        self,
        *,
        recipe: Any,
        tensors: Mapping[str, Any],
    ) -> None:
        tc_tensor_schema.validate_tensor_schema_against_tensors(
            recipe.tensor_schema, tensors
        )

    def freeze_local_ready(
        self,
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

    def build_materialization_options(
        self,
        *,
        artifact_ref: str,
        operation_scope: str,
        configured_policy: CollectivePolicy,
        source_bound_contract_state: SourceBoundContractState,
        source_bound_contract_path: str,
        execution_facts: Mapping[str, Any],
        contract_identity: str | None = None,
    ) -> tuple[Any, dict[str, object]]:
        facts = dict(execution_facts)
        return tc_binding_runtime.build_materialization_execution_context(
            artifact_ref=artifact_ref,
            operation_scope=operation_scope,
            configured_policy=configured_policy,
            tp_rank=int(facts.get("tp_rank", 0) or 0),
            tp_world_size=int(facts.get("tp_world_size", 1) or 1),
            same_node_tp=bool(facts.get("same_node_tp", False)),
            tp_ranks=tuple(int(rank) for rank in facts.get("tp_ranks", ()) or ()),
            collective_world_size=int(
                facts.get("collective_world_size", facts.get("tp_world_size", 1)) or 1
            ),
            collective_rank=int(
                facts.get("collective_rank", facts.get("tp_rank", 0)) or 0
            ),
            source_bound_contract_profile_fields=source_bound_contract_profile_fields(
                source_bound_contract_state,
                source_bound_contract_path,
            ),
            build_group_id=build_collective_group_id,
            contract_identity=contract_identity,
            collective_context_unavailable=bool(
                facts.get("collective_context_unavailable", False)
            ),
        )

    def build_recipe_session(
        self, request: RecipeBuildSessionRequest
    ) -> RecipeBuildSession:
        adapter = None
        placement = request.placement
        if request.identity is None:
            model_config = request.model_config
            if model_config is None:
                self._lifecycle_not_implemented("build_recipe_session", "P2")
            adapter = self._recipe_framework_adapter(model_config)
            if placement is None and self.host is not None:
                placement = self._framework_context(
                    request.framework_config,
                    model_config,
                ).placement
        return build_recipe_session_from_request(
            request,
            adapter=adapter,
            placement=placement,
        )

    def resolve_source_subject(
        self,
        path: str | SourceSelector,
        *,
        verify_checksums: bool,
        coordinator: Any | None = None,
    ) -> SourceSubject:
        if isinstance(path, SourceSelector):
            if path.kind != "local_path":
                raise SourceSubjectError(
                    f"Unsupported TensorCast source selector kind: {path.kind}"
                )
            path = str(path.value)
        if coordinator is not None:
            should_coordinate = getattr(coordinator, "should_coordinate", None)
            if not callable(should_coordinate) or bool(should_coordinate()):
                return self._resolve_source_subject_with_coordinator(
                    path,
                    verify_checksums=verify_checksums,
                    coordinator=coordinator,
                )
        return resolve_source_subject(path, verify_checksums=verify_checksums)

    def _resolve_source_subject_with_coordinator(
        self,
        path: str,
        *,
        verify_checksums: bool,
        coordinator: Any,
    ) -> SourceSubject:
        source_rank = int(getattr(coordinator, "source_rank", 0) or 0)
        is_source_rank = getattr(coordinator, "is_source_rank", None)
        resolve_locally = bool(is_source_rank()) if callable(is_source_rank) else True
        subject = (
            resolve_source_subject(path, verify_checksums=verify_checksums)
            if resolve_locally
            else None
        )
        payload = None if subject is None else source_subject_broadcast_payload(subject)
        broadcast = getattr(coordinator, "broadcast_object", None)
        if not callable(broadcast):
            raise SourceSubjectError(
                "TensorCast source subject coordinator must provide "
                "broadcast_object(payload, src)"
            )
        payload = broadcast(payload, src=source_rank)
        if payload is None:
            raise SourceSubjectError(
                "TensorCast source subject coordinator returned no payload"
            )
        if not isinstance(payload, Mapping):
            raise SourceSubjectError(
                "TensorCast source subject coordinator must broadcast a mapping payload"
            )
        return source_subject_from_broadcast_payload(payload)

    def source_subject_broadcast_payload(
        self, subject: SourceSubject
    ) -> dict[str, Any]:
        return source_subject_broadcast_payload(subject)

    def source_subject_from_broadcast_payload(
        self, payload: Mapping[str, Any]
    ) -> SourceSubject:
        return source_subject_from_broadcast_payload(payload)

    def _framework_host(self) -> FrameworkHost:
        if self.host is not None:
            return self.host.framework
        raise _capability_missing(
            "ArtifactRuntimeIntegration requires IntegrationHost.framework",
            level="level1-runtime",
            capability="framework",
            operation="framework_host",
            required_methods=(
                "identity",
                "build_runtime_model",
                "assert_model_ready_for_runtime_binding",
            ),
            next_action=(
                "Construct ArtifactRuntimeSession with IntegrationHost.framework."
            ),
        )

    def _framework_identity(
        self,
        model_config: Any | None,
    ) -> FrameworkIdentity:
        return self._framework_host().identity(model_config)

    def _build_meta_model(
        self,
        framework_config: Any | None,
        model_config: Any | None,
    ) -> object:
        return self._framework_host().build_meta_model(
            framework_config,
            model_config,
        )

    def _surface(self) -> TensorSurfaceHost:
        if self.host is not None:
            if self.host.tensor_surface is None:
                raise _capability_missing(
                    "IntegrationHost requires TensorSurfaceHost for runtime "
                    "tensor operations",
                    level="level1-runtime",
                    capability="tensor_surface",
                    operation="runtime_tensor_surface",
                    required_methods=(
                        "attach_bound_tensors",
                        "collect_runtime_tensors",
                        "compute_runtime_tensor_schema_hash",
                    ),
                    next_action=(
                        "Pass IntegrationHost(tensor_surface=...) or use "
                        "TorchTensorHost for PyTorch module carriers."
                    ),
                )
            return self.host.tensor_surface
        raise _capability_missing(
            "ArtifactRuntimeIntegration requires IntegrationHost.tensor_surface",
            level="level1-runtime",
            capability="tensor_surface",
            operation="runtime_tensor_surface",
            required_methods=(
                "attach_bound_tensors",
                "collect_runtime_tensors",
                "compute_runtime_tensor_schema_hash",
            ),
            next_action=(
                "Construct ArtifactRuntimeSession with IntegrationHost.tensor_surface."
            ),
        )

    @staticmethod
    def _require_target_device(target_device: Any | None) -> torch.device:
        if target_device is None:
            raise ArtifactRuntimeIntegrationError(
                "ArtifactRuntimeIntegration request requires target_device"
            )
        return torch.device(target_device)

    @staticmethod
    def _runtime_policy(policy: Any | None) -> Any | None:
        to_runtime_policy = getattr(policy, "to_runtime_policy", None)
        if callable(to_runtime_policy):
            return to_runtime_policy()
        return policy

    @staticmethod
    def _runtime_policy_with_placement(
        policy: Any | None, placement: Any | None
    ) -> Any | None:
        digest = _optional_text(
            getattr(
                getattr(placement, "topology", None), "schema_topology_digest", None
            )
        )
        if digest is None:
            return policy
        if policy is None:
            return RuntimeArtifactPolicy(
                require_manifest=True,
                expected_topology_admission_digest=digest,
            )
        model_copy = getattr(policy, "model_copy", None)
        if callable(model_copy):
            return model_copy(
                update={
                    "require_manifest": True,
                    "expected_topology_admission_digest": digest,
                }
            )
        return policy

    @classmethod
    def _runtime_policy_from_manifest(
        cls, policy: Any | None, resolved: Any, placement: Any | None = None
    ) -> Any | None:
        if policy is not None:
            return cls._runtime_policy_with_placement(policy, placement)
        manifest = getattr(resolved, "manifest", None)
        to_runtime_policy = getattr(manifest, "to_runtime_policy", None)
        if callable(to_runtime_policy):
            return cls._runtime_policy_with_placement(to_runtime_policy(), placement)
        return cls._runtime_policy_with_placement(None, placement)

    @staticmethod
    def _json_object_payload(value: Any, *, field_name: str) -> Any:
        try:
            payload = json.loads(str(value))
        except Exception as exc:
            raise ManifestMismatchError(
                f"TensorCast runtime artifact {field_name} is invalid JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise ManifestMismatchError(
                f"TensorCast runtime artifact {field_name} must be a JSON object"
            )
        return payload

    @classmethod
    def _validate_resolved_artifact_placement(
        cls,
        resolved_artifact: Any,
        *,
        placement: Any | None,
    ) -> None:
        manifest = getattr(resolved_artifact, "manifest", None)
        if manifest is None:
            return
        manifest_topology_digest = _optional_text(
            getattr(manifest, "topology_admission_digest", None)
        )
        placement_topology_digest = _optional_text(
            getattr(
                getattr(placement, "topology", None), "schema_topology_digest", None
            )
        )
        if manifest_topology_digest is not None:
            if placement_topology_digest is None:
                raise ManifestMismatchError(
                    "TensorCast runtime artifact topology admission digest "
                    "requires current framework placement"
                )
            if manifest_topology_digest != placement_topology_digest:
                raise ManifestMismatchError(
                    "TensorCast runtime artifact topology admission digest "
                    "mismatch: "
                    f"manifest={manifest_topology_digest}, "
                    f"current={placement_topology_digest}"
                )

        manifest_logical_topology = _optional_text(
            getattr(manifest, "logical_topology_json", None)
        )
        if manifest_logical_topology is None:
            return
        if placement is None:
            raise ManifestMismatchError(
                "TensorCast runtime artifact logical topology requires current "
                "framework placement"
            )
        try:
            current_logical_topology = tc_contract.logical_topology_json(
                placement.topology,
                framework_payload=dict(getattr(placement, "framework_payload", {})),
            )
        except Exception as exc:
            raise ManifestMismatchError(
                "TensorCast runtime artifact logical topology could not be "
                "computed from current framework placement"
            ) from exc
        if cls._json_object_payload(
            manifest_logical_topology, field_name="logical_topology_json"
        ) != cls._json_object_payload(
            current_logical_topology, field_name="current logical topology"
        ):
            raise ManifestMismatchError(
                "TensorCast runtime artifact logical topology mismatch"
            )

    def _prepare_model_construction(
        self,
        framework_config: Any | None,
        model_config: Any | None,
    ) -> None:
        host = self._framework_host()
        prepare = getattr(host, "prepare_model_construction", None)
        if callable(prepare):
            prepare(framework_config, model_config)

    def _assert_model_ready_for_runtime_binding(
        self,
        model: Any,
        *,
        context: str,
    ) -> None:
        host = self._framework_host()
        check = getattr(host, "assert_model_ready_for_runtime_binding", None)
        if callable(check):
            check(model, context=context)

    def _align_runtime_tensor_names(
        self,
        model: Any,
        expected_names: Sequence[str],
    ) -> int:
        return int(
            self._surface().align_runtime_tensor_names(model, expected_names) or 0
        )

    def _collect_runtime_binding_tensors(
        self,
        model: Any,
        *,
        remove_duplicate: bool,
    ) -> Mapping[str, Any]:
        return self._surface().collect_runtime_tensors(
            model,
            remove_duplicate=remove_duplicate,
        )

    def _compute_runtime_tensor_schema_hash(
        self,
        tensors: Mapping[str, Any],
        *,
        remove_duplicate: bool,
    ) -> str:
        return self._surface().compute_runtime_tensor_schema_hash(
            tensors,
            remove_duplicate=remove_duplicate,
        )

    def runtime_only_tensor_names(self, model: object) -> tuple[str, ...]:
        return self._surface().runtime_only_tensor_names(model)

    def support_level(
        self,
        model: object,
        model_config: object,
    ) -> RuntimeSupportLevel:
        host = self._framework_host()
        support_level = getattr(host, "support_level", None)
        if callable(support_level):
            return tc_readiness.coerce_runtime_support_level(
                support_level(model, model_config),
                default=RuntimeSupportLevel.BLOCKED,
            )
        return RuntimeSupportLevel.BLOCKED

    def process_after_load_class(
        self,
        model: object,
        model_config: object,
    ) -> FinalizeClass:
        host = self._framework_host()
        process_after_load = getattr(host, "process_after_load_class", None)
        if callable(process_after_load):
            return tc_readiness.coerce_finalize_class(
                process_after_load(model, model_config),
                default=FinalizeClass.UNKNOWN_BLOCKED,
            )
        finalize_policy = getattr(host, "finalize_policy", None)
        if callable(finalize_policy):
            finalize_policy(model, model_config)
            return FinalizeClass.RUNTIME_ONLY
        return FinalizeClass.UNKNOWN_BLOCKED

    def post_bind_finalize_class(
        self,
        model: object,
        model_config: object,
    ) -> FinalizeClass:
        host = self._framework_host()
        post_bind_finalize = getattr(host, "post_bind_finalize_class", None)
        if callable(post_bind_finalize):
            return tc_readiness.coerce_finalize_class(
                post_bind_finalize(model, model_config),
                default=FinalizeClass.RUNTIME_ONLY,
            )
        finalize_policy = getattr(host, "finalize_policy", None)
        if callable(finalize_policy):
            finalize_policy(model, model_config)
            return FinalizeClass.RUNTIME_ONLY
        return FinalizeClass.RUNTIME_ONLY

    def trace_model_load(
        self,
        model: object,
        ordered_names: Sequence[str],
        meta_by_name: Mapping[str, object],
        *,
        debug_dump_trace: bool = False,
    ) -> TracePlan:
        host = self._framework_host()
        trace = getattr(host, "trace_model_load", None)
        if not callable(trace):
            raise _capability_missing(
                "ArtifactRuntimeIntegration host requires RecipeTraceHost."
                "trace_model_load on recipe cache miss",
                level="level2-local-bootstrap",
                capability="recipe_trace",
                operation="local_bootstrap.trace_model_load",
                required_methods=("trace_model_load",),
                next_action=(
                    "Implement RecipeTraceHost.trace_model_load or precompute "
                    "a recipe through the admin/offline builder path."
                ),
            )
        return cast(
            TracePlan,
            trace(
                model,
                ordered_names,
                meta_by_name,
                debug_dump_trace=debug_dump_trace,
            ),
        )

    def cleanup_after_recipe_build(
        self,
        model: object,
        model_config: object,
        *,
        framework_config: object | None = None,
    ) -> None:
        host = self._framework_host()
        cleanup = getattr(host, "cleanup_after_recipe_build", None)
        if callable(cleanup):
            cleanup(
                model,
                model_config,
                framework_config=framework_config,
            )

    def semantic_probes(self, model: object, model_config: object) -> object:
        host = self._framework_host()
        semantic_probes = getattr(host, "semantic_probes", None)
        if callable(semantic_probes):
            return semantic_probes(model, model_config)
        return None

    def native_load_weights(self, model: object, weights: object) -> None:
        host = self._framework_host()
        native_load = getattr(host, "native_load_weights", None)
        if not callable(native_load):
            raise _capability_missing(
                "ArtifactRuntimeIntegration host requires NativeLoadHost for native "
                "checkpoint/source loading",
                level="level2-local-bootstrap",
                capability="native_load",
                operation="local_bootstrap.native_load_weights",
                required_methods=("native_load_weights",),
                next_action=(
                    "Implement NativeLoadHost.native_load_weights for source "
                    "bootstrap cache misses."
                ),
            )
        native_load(model, weights)

    def _recipe_framework_adapter(self, model_config: Any | None) -> Any:
        identity = self._framework_identity(model_config)
        return SimpleNamespace(
            framework_name=lambda: str(identity.framework_name),
            framework_version=lambda: str(identity.framework_version),
            adapter_version=lambda: str(identity.adapter_version),
            serving_abi_version=lambda _model_config=None: str(
                identity.serving_abi_version
            ),
            support_level=self.support_level,
            runtime_only_tensor_names=self.runtime_only_tensor_names,
            process_after_load_class=self.process_after_load_class,
            post_bind_finalize_class=self.post_bind_finalize_class,
            trace_model_load=self.trace_model_load,
            cleanup_after_recipe_build=self.cleanup_after_recipe_build,
            semantic_probes=self.semantic_probes,
            native_load_weights=self.native_load_weights,
        )

    @staticmethod
    def _assert_tensor_names_match_expected(
        tensors: Mapping[str, Any],
        expected_names: Sequence[str],
    ) -> None:
        expected = {str(name) for name in expected_names}
        if not expected:
            return
        actual = {str(name) for name in tensors}
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if not missing and not unexpected:
            return
        raise SchemaMismatchError(
            "TensorCast runtime tensor set does not match runtime artifact: "
            f"missing_count={len(missing)}, unexpected_count={len(unexpected)}"
        )

    def _load_materialization_options(
        self,
        request: _DirectRuntimeLoad,
        resolved: Any,
    ) -> Any | None:
        if request.materialization is not None:
            return request.materialization
        execution_facts = self._request_execution_facts(request)
        if (
            request.configured_collective_policy is None
            or request.source_bound_contract_state is None
            or not request.source_bound_contract_path
            or execution_facts is None
        ):
            if request.require_materialization_options:
                raise ArtifactRuntimeIntegrationError(
                    "ArtifactRuntimeIntegration._load_existing_runtime_artifact requires "
                    "materialization execution context for direct bind"
                )
            return None
        if request.require_materialization_options and not getattr(
            request.source_bound_contract_state,
            "source_bound_contract_ready",
            False,
        ):
            raise ArtifactRuntimeIntegrationError(
                "ArtifactRuntimeIntegration._load_existing_runtime_artifact requires ready "
                "source-bound contract state for direct bind"
            )
        manifest = getattr(resolved, "manifest", None)
        options, _profile = self.build_materialization_options(
            artifact_ref=str(getattr(resolved, "artifact_ref", "") or ""),
            operation_scope=request.operation_scope,
            configured_policy=request.configured_collective_policy,
            source_bound_contract_state=request.source_bound_contract_state,
            source_bound_contract_path=request.source_bound_contract_path,
            execution_facts=execution_facts,
            contract_identity=getattr(manifest, "representation_contract_hash", None),
        )
        return options

    def _reload_materialization_options(
        self,
        request: _RuntimeReload,
        resolved: Any,
    ) -> Any | None:
        if request.materialization is not None:
            return request.materialization
        execution_facts = self._request_execution_facts(request)
        if (
            request.configured_collective_policy is None
            or request.source_bound_contract_state is None
            or not request.source_bound_contract_path
            or execution_facts is None
        ):
            if request.require_materialization_options:
                raise ArtifactRuntimeIntegrationError(
                    "ArtifactRuntimeIntegration._reload_existing_runtime_artifact requires "
                    "materialization execution context for swap"
                )
            return None
        if request.require_materialization_options and not getattr(
            request.source_bound_contract_state,
            "source_bound_contract_ready",
            False,
        ):
            raise ArtifactRuntimeIntegrationError(
                "ArtifactRuntimeIntegration._reload_existing_runtime_artifact requires ready "
                "source-bound contract state for swap"
            )
        manifest = getattr(resolved, "manifest", None)
        options, _profile = self.build_materialization_options(
            artifact_ref=str(getattr(resolved, "artifact_ref", "") or ""),
            operation_scope=request.operation_scope,
            configured_policy=request.configured_collective_policy,
            source_bound_contract_state=request.source_bound_contract_state,
            source_bound_contract_path=request.source_bound_contract_path,
            execution_facts=execution_facts,
            contract_identity=(
                request.contract_identity
                or getattr(
                    manifest,
                    "representation_contract_hash",
                    None,
                )
            ),
        )
        return options

    def _resolved_artifact(
        self,
        *,
        resolved_artifact: ResolvedRuntimeArtifact | None,
        artifact_ref: str | None,
        artifact_locator: Any | None,
        expected_tensor_schema_hash: str | None,
        runtime_artifact_policy: Any | None,
        placement: RuntimePlacement | None = None,
    ) -> ResolvedRuntimeArtifact:
        if resolved_artifact is not None:
            if artifact_ref is not None and str(resolved_artifact.artifact_ref) != str(
                artifact_ref
            ):
                raise ManifestMismatchError(
                    "TensorCast resolved runtime artifact ref mismatch: "
                    f"resolved={resolved_artifact.artifact_ref}, "
                    f"requested={artifact_ref}"
                )
            self._validate_resolved_artifact_placement(
                resolved_artifact,
                placement=placement,
            )
            if self.resolver is not None and expected_tensor_schema_hash:
                return cross_check_runtime_artifact(
                    resolved_artifact,
                    resolver=self.resolver,
                    expected_tensor_schema_hash=expected_tensor_schema_hash,
                    runtime_artifact_policy=runtime_artifact_policy,
                )
            return resolved_artifact
        resolved_ref = artifact_ref
        if resolved_ref is None and artifact_locator is not None:
            resolve_artifact_ref = getattr(
                artifact_locator, "resolve_artifact_ref", None
            )
            if callable(resolve_artifact_ref):
                if _artifact_locator_kind(artifact_locator) == "ranked_version_key":
                    resolved_ref = resolve_artifact_ref(placement=placement)
                else:
                    resolved_ref = resolve_artifact_ref()
            else:
                resolved_ref = str(artifact_locator)
        if not resolved_ref:
            raise ArtifactRuntimeIntegrationError(
                "ArtifactRuntimeIntegration request requires resolved_artifact, "
                "artifact_ref, or artifact_locator"
            )
        resolved = resolve_runtime_artifact(
            str(resolved_ref),
            resolver=self.resolver,
            expected_tensor_schema_hash=expected_tensor_schema_hash,
            runtime_artifact_policy=runtime_artifact_policy,
        )
        self._validate_resolved_artifact_placement(
            resolved,
            placement=placement,
        )
        return resolved

    def _preflight_runtime_artifact(
        self,
        *,
        resolved_artifact: ResolvedRuntimeArtifact | None,
        artifact_ref: str | None,
        artifact_locator: Any | None,
        expected_tensor_schema_hash: str | None,
        policy: Any | None,
        placement: RuntimePlacement | None = None,
    ) -> _RuntimeArtifactPreflight:
        base_policy = self._runtime_policy(policy)
        resolved = self._resolved_artifact(
            resolved_artifact=resolved_artifact,
            artifact_ref=artifact_ref,
            artifact_locator=artifact_locator,
            expected_tensor_schema_hash=None,
            runtime_artifact_policy=None,
            placement=placement,
        )
        runtime_artifact_policy = self._runtime_policy_from_manifest(
            base_policy,
            resolved,
            placement=placement,
        )
        if expected_tensor_schema_hash is not None:
            resolved = self._resolved_artifact(
                resolved_artifact=resolved,
                artifact_ref=artifact_ref,
                artifact_locator=artifact_locator,
                expected_tensor_schema_hash=expected_tensor_schema_hash,
                runtime_artifact_policy=runtime_artifact_policy,
                placement=placement,
            )
        return _RuntimeArtifactPreflight(
            resolved_artifact=resolved,
            runtime_artifact_policy=runtime_artifact_policy,
        )

    def _framework_context(
        self,
        framework_config: Any | None,
        model_config: Any | None,
    ) -> FrameworkIntegrationContext:
        identity = self._framework_identity(model_config)
        placement = None
        if self.host is not None:
            try:
                placement = self._host_runtime_placement(framework_config)
            except Exception:
                placement = None
        return FrameworkIntegrationContext(
            framework_name=str(identity.framework_name),
            framework_version=str(identity.framework_version),
            adapter_version=str(identity.adapter_version),
            serving_abi_version=str(identity.serving_abi_version),
            placement=placement,
        )

    def _materializer(self) -> RuntimeBindingMaterialization:
        if self.host is None:
            raise _capability_missing(
                "ArtifactRuntimeIntegration runtime materialization requires IntegrationHost",
                level="level1-runtime",
                capability="integration_host",
                operation="runtime_materialization",
                required_methods=("framework", "placement", "tensor_surface"),
                next_action=(
                    "Construct ArtifactRuntimeSession with an IntegrationHost "
                    "instead of calling lifecycle helpers without host facts."
                ),
            )
        return RuntimeBindingMaterialization(
            host=self.host,
            profile_sink=self.profile_sink,
        )

    @staticmethod
    def _state_seed(
        resolved: ResolvedRuntimeArtifact,
        *,
        tensor_schema_hash: str,
        execution_diagnostics: Any | None,
        materialization_diagnostics: Any | None = None,
        binding_handle: Any | None = None,
        artifact_realization_report: ArtifactRealizationReport | None = None,
        readiness: str = "runtime_ready",
    ) -> RuntimeStateSeed:
        artifact_ref = str(getattr(resolved, "artifact_ref", "") or "")
        manifest = getattr(resolved, "manifest", None)
        representation_contract_hash = str(
            getattr(manifest, "representation_contract_hash", "") or ""
        )
        source_artifact_ref = getattr(manifest, "source_artifact_ref", None)
        serving_build_digest = getattr(manifest, "serving_build_digest", None)
        diagnostics = {}
        if materialization_diagnostics is not None:
            diagnostics["materialization"] = materialization_diagnostics
        if execution_diagnostics is not None:
            diagnostics["execution"] = execution_diagnostics
        if serving_build_digest:
            diagnostics["serving_build_digest"] = str(serving_build_digest)
        if artifact_realization_report is not None:
            diagnostics["artifact_realization_report"] = (
                artifact_realization_report_to_dict(artifact_realization_report)
            )
        binding_value_ref = getattr(binding_handle, "current_value", None)
        if binding_value_ref is None:
            binding_value_ref = getattr(binding_handle, "binding_value_ref", None)
        return RuntimeStateSeed(
            artifact_ref=artifact_ref,
            serving_artifact_ref=artifact_ref or None,
            source_artifact_ref=(
                str(source_artifact_ref) if source_artifact_ref else None
            ),
            representation_contract_hash=representation_contract_hash,
            tensor_schema_hash=str(tensor_schema_hash or ""),
            binding_value_ref=binding_value_ref,
            local_serving_ref=getattr(manifest, "local_serving_ref", None),
            readiness=readiness,
            diagnostics=diagnostics or None,
            realization_report=artifact_realization_report,
        )


def resolve_runtime_artifact(
    artifact_ref: str,
    *,
    resolver: RuntimeArtifactResolver | None = None,
    manifest_tensor_name: str | None = None,
    schema_version: int | None = None,
    expected_tensor_schema_hash: str | None = None,
    runtime_artifact_policy: Any | None = None,
) -> ResolvedRuntimeArtifact:
    """Resolve a runtime artifact and optionally cross-check runtime schema."""

    resolved_resolver = resolver or RuntimeArtifactResolver(
        manifest_tensor_name=manifest_tensor_name or tc.SERVING_MANIFEST_TENSOR_NAME,
        schema_version=(
            schema_version
            if schema_version is not None
            else int(tc.RuntimeArtifactManifest.model_fields["schema_version"].default)
        ),
    )
    resolved = resolved_resolver.resolve(str(artifact_ref))
    if expected_tensor_schema_hash is not None:
        resolved_resolver.cross_check(
            resolved,
            expected_tensor_schema_hash=expected_tensor_schema_hash,
            runtime_artifact_policy=runtime_artifact_policy,
        )
    return resolved


def read_runtime_artifact_manifest(
    artifact: Any,
    *,
    artifact_ref: str,
    resolver: RuntimeArtifactResolver,
) -> ResolvedRuntimeArtifact:
    """Read a runtime manifest from an already opened artifact handle."""

    return resolver.read_manifest(artifact, artifact_ref=str(artifact_ref))


def cross_check_runtime_artifact(
    resolved_artifact: ResolvedRuntimeArtifact,
    *,
    resolver: RuntimeArtifactResolver,
    expected_tensor_schema_hash: str,
    runtime_artifact_policy: Any | None = None,
) -> ResolvedRuntimeArtifact:
    """Validate manifest, descriptor schema, and runtime policy agreement."""

    return resolver.cross_check(
        resolved_artifact,
        expected_tensor_schema_hash=expected_tensor_schema_hash,
        runtime_artifact_policy=runtime_artifact_policy,
    )


@dataclass(frozen=True)
class ArtifactRuntimeSession:
    """Config-planned artifact runtime lifecycle entrypoint."""

    runtime_config: TensorCastRuntimeConfig
    host: IntegrationHost
    integration: ArtifactRuntimeIntegration
    profile_sink: Any | None = None

    @classmethod
    def from_config(
        cls,
        runtime_config: TensorCastRuntimeConfig | Mapping[str, Any],
        *,
        host: IntegrationHost,
        resolver: RuntimeArtifactResolver | None = None,
        profile_sink: Any | None = None,
    ) -> "ArtifactRuntimeSession":
        config = (
            runtime_config
            if isinstance(runtime_config, TensorCastRuntimeConfig)
            else TensorCastRuntimeConfig.from_mapping(runtime_config)
        )
        return cls(
            runtime_config=config,
            host=host,
            integration=ArtifactRuntimeIntegration(
                resolver=resolver,
                profile_sink=profile_sink,
                host=host,
            ),
            profile_sink=profile_sink,
        )

    def start(self, context: RequestContext) -> RuntimeAttachment:
        intent = self._plan_start_intent(context)
        return self._start_intent(intent, context)

    def publish_current_replica(
        self,
        *,
        current_attachment: RuntimeAttachment,
        context: RequestContext | None = None,
        policy: ReplicaPublicationPolicy | Mapping[str, Any] | None = None,
    ) -> RuntimeAttachment:
        """Publish the current artifact-backed runtime attachment as a replica."""

        del context
        return tc_replica_publication.publish_current_replica(
            current_attachment=current_attachment,
            policy=tc_replica_publication.replica_publication_policy(
                policy,
                default_policy=self.runtime_config.replica_publication,
            ),
            ensure_runtime_initialized=self._ensure_runtime_initialized,
            profile_sink=self.profile_sink,
        )

    def project_current_replica_publication_state(
        self,
        *,
        current_attachment: RuntimeAttachment,
        state: str,
        reason: str | None = None,
        operation_id: str | None = None,
    ) -> RuntimeAttachment:
        """Return an attachment with a non-authoritative publication projection.

        This helper is intentionally limited to observational states.  The
        authoritative ``published`` and ``retired`` states still come only from
        publish/retire lifecycle operations that touch the daemon-owned binding.
        """

        return tc_replica_publication.project_current_replica_publication_state(
            current_attachment=current_attachment,
            state=state,
            reason=reason,
            operation_id=operation_id,
        )

    def retire_current_replica(
        self,
        *,
        current_attachment: RuntimeAttachment,
        reason: str = "retire",
        drain_timeout_s: float | None = None,
        context: RequestContext | None = None,
    ) -> RuntimeAttachment:
        """Retire the published replica tied to a runtime attachment."""

        del context
        return tc_replica_publication.retire_current_replica(
            current_attachment=current_attachment,
            reason=reason,
            drain_timeout_s=drain_timeout_s,
            default_drain_timeout_s=(
                self.runtime_config.replica_publication.drain_timeout_s
            ),
            ensure_runtime_initialized=self._ensure_runtime_initialized,
            profile_sink=self.profile_sink,
        )

    def _start_intent(
        self,
        intent: RuntimeIntent,
        context: RequestContext,
    ) -> RuntimeAttachment:
        """Private/admin entrypoint for already lowered runtime intents."""

        self._ensure_runtime_initialized()
        return self.integration.start(intent, context)

    def reload(
        self,
        *,
        current_attachment: RuntimeAttachment | RuntimeBindingState | Any,
        artifact_locator: ArtifactLocator,
        policy: RuntimePolicy | None,
        context: RequestContext,
        model: object | None = None,
        contract_identity: str | None = None,
    ) -> RuntimeAttachment:
        self._reject_local_reload_artifact_locator(artifact_locator)
        if not isinstance(artifact_locator, ArtifactLocator):
            raise ConfigConflictError(
                "TensorCast runtime artifact reload requires an ArtifactLocator"
            )
        if policy is not None and not isinstance(policy, RuntimePolicy):
            raise ConfigConflictError(
                "TensorCast runtime artifact reload requires a RuntimePolicy or None"
            )
        if isinstance(current_attachment, RuntimeAttachment):
            self._reject_reload_with_active_publication(current_attachment)
        self._ensure_runtime_initialized()
        current_state = (
            current_attachment.state
            if isinstance(current_attachment, RuntimeAttachment)
            else current_attachment
        )
        runtime_model = (
            model if model is not None else getattr(current_attachment, "model", None)
        )
        return self.integration.reload(
            current_state,
            ExistingRuntimeArtifact(artifact_locator=artifact_locator, policy=policy),
            context,
            model=runtime_model,
            contract_identity=contract_identity,
        )

    def describe(
        self,
        attachment_or_state: RuntimeAttachment | RuntimeBindingState | Any,
    ) -> RuntimeWorkerView:
        if isinstance(attachment_or_state, RuntimeAttachment):
            return attachment_or_state.view
        return self.integration.describe(attachment_or_state)

    def _ensure_runtime_initialized(self) -> None:
        self.runtime_config.runtime.ensure_initialized()

    @staticmethod
    def _reject_reload_with_active_publication(
        current_attachment: RuntimeAttachment,
    ) -> None:
        tc_replica_publication.reject_reload_with_active_publication(current_attachment)

    def _plan_start_intent(self, context: RequestContext) -> RuntimeIntent:
        source_selector = self._source_selector_from_context(context)
        expected_member = None
        if (
            self.runtime_config.retained_binding_acquire.mode == "external"
            and self.host is not None
        ):
            placement = self.integration._framework_context(
                context.framework_config,
                context.model_config,
            ).placement
            if placement is not None:
                expected_member = placement.member
        try:
            plan = tc_runtime_config.plan_runtime_start(
                config=self.runtime_config,
                source_selector=source_selector,
                expected_member=expected_member,
            )
        except tc_runtime_config.RuntimeStartPlanError as exc:
            raise ConfigConflictError(str(exc)) from exc

        if isinstance(plan, tc_runtime_config.RuntimeRetainedRealizationStartPlan):
            return RetainedBindingAcquire(plan.authority)
        if isinstance(plan, tc_runtime_config.RuntimeArtifactBindStartPlan):
            return ExistingRuntimeArtifact(
                artifact_locator=plan.artifact_locator,
                policy=plan.policy,
            )
        if isinstance(plan, tc_runtime_config.RuntimeSourceBootstrapStartPlan):
            return LocalSourceBootstrap(
                source_selector=plan.source_selector,
                bootstrap_policy=plan.bootstrap_policy,
            )
        raise ConfigConflictError(
            f"TensorCast runtime planner returned unsupported plan: {plan!r}"
        )

    def _source_selector_from_context(
        self, context: RequestContext
    ) -> SourceSelector | None:
        if self.host.source is None:
            return None
        source_selector = getattr(self.host.source, "source_selector", None)
        if not callable(source_selector):
            return None
        selector = source_selector(context.framework_config, context.model_config)
        if selector is None:
            return None
        if not isinstance(selector, SourceSelector):
            raise ConfigConflictError(
                "IntegrationHost.source.source_selector must return "
                "SourceSelector or None"
            )
        return selector

    @staticmethod
    def _reject_local_reload_artifact_locator(artifact_locator: object) -> None:
        if (
            isinstance(artifact_locator, SourceSelector)
            or _artifact_locator_kind(artifact_locator) == "local_path"
        ):
            raise ConfigConflictError(
                "TensorCast runtime artifact reload requires a durable runtime "
                "artifact locator, not a local source selector"
            )


def bind_runtime_artifact(
    *,
    resolved_artifact: ResolvedRuntimeArtifact,
    tensor_names: Sequence[str],
    device: Any,
    runtime_artifact_policy: Any | None,
    options: Any | None,
) -> RuntimeBindingResult:
    """Bind a durable runtime artifact and return an attach-ready result."""

    binding = tc_binding_runtime.bind_runtime_artifact(
        resolved_artifact=resolved_artifact,
        tensor_names=tuple(tensor_names),
        device=device,
        runtime_artifact_policy=runtime_artifact_policy,
        options=options,
    )
    return RuntimeBindingResult.from_binding(binding)


def swap_runtime_artifact(
    *,
    binding: Any,
    resolved_artifact: ResolvedRuntimeArtifact,
    tensor_names: Sequence[str] | None = None,
    runtime_artifact_policy: Any | None,
    options: Any | None,
) -> RuntimeBindingResult:
    """Swap an existing runtime binding to another runtime artifact."""

    operation_result = tc_binding_runtime.swap_runtime_artifact(
        binding=binding,
        resolved_artifact=resolved_artifact,
        tensor_names=tensor_names,
        runtime_artifact_policy=runtime_artifact_policy,
        options=options,
    )
    result_binding = operation_result if operation_result is not None else binding
    if not hasattr(result_binding, "tensors"):
        result_binding = binding
    return RuntimeBindingResult.from_binding(
        result_binding,
        operation_result=operation_result,
    )
