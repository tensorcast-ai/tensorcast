#  Copyright (c) 2026, TensorCast Team.
"""Runtime binding lifecycle flows for durable and retained artifacts.

This module keeps attach-ready binding execution separate from
``ArtifactRuntimeIntegration``.  The integration object still owns orchestration
state and framework hooks; these helpers own the concrete durable bind, swap,
and retained-restore flows.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import torch

import tensorcast.artifact_runtime.binding.execution as tc_binding_runtime
import tensorcast.artifact_runtime.framework_bridge as tc_framework_bridge
from tensorcast.api.store.realization_kernel import (
    artifact_realization_report_to_dict,
)
from tensorcast.artifact_runtime.artifact.resolver import ResolvedRuntimeArtifact
from tensorcast.artifact_runtime.artifact_preflight import (
    artifact_locator_kind as _artifact_locator_kind,
)
from tensorcast.artifact_runtime.attachment import (
    RuntimeBindingState,
    RuntimeStateSeed,
)
from tensorcast.artifact_runtime.attachment_materialization import (
    runtime_attachment_realization_handle as _runtime_attachment_realization_handle,
)
from tensorcast.artifact_runtime.binding.retained import (
    restore_prepared_local_ready_binding,
    restore_retained_binding,
    runtime_restore_rejection_reason,
)
from tensorcast.artifact_runtime.errors import (
    ArtifactRuntimeIntegrationError,
    AttachFinalizeError,
    OwnershipTransferError,
    RestoreBindingError,
    SchemaMismatchError,
)
from tensorcast.artifact_runtime.lifecycle_requests import (
    RetainedBindingResult,
    RuntimeBindingResult,
    RuntimeLoadResult,
    RuntimeReloadResult,
    _DirectRuntimeLoad,
    _RetainedBindingAcquire,
    _RuntimeReload,
)
from tensorcast.artifact_runtime.realization_reports import (
    runtime_attachment_report_for_artifact_id as _runtime_attachment_report_for_artifact_id,
)
from tensorcast.artifact_runtime.realization_reports import (
    runtime_attachment_report_for_resolved as _runtime_attachment_report_for_resolved,
)
from tensorcast.artifact_runtime.realization_reports import (
    runtime_attachment_report_for_retained as _runtime_attachment_report_for_retained,
)


def is_runtime_binding_swap_capable(binding: Any) -> bool:
    return bool(
        getattr(binding, "swap_capable", False)
        or callable(getattr(binding, "swap", None))
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


def load_existing_runtime_artifact(
    integration: Any,
    request: _DirectRuntimeLoad,
    *,
    restore_prepared_fn: Any | None = None,
    bind_fn: Any | None = None,
) -> RuntimeLoadResult:
    restore_prepared_fn = restore_prepared_fn or restore_prepared_local_ready_binding
    bind_fn = bind_fn or bind_runtime_artifact
    target_device = tc_framework_bridge.require_target_device(request.target_device)
    context = integration._framework_context(
        request.framework_config,
        request.model_config,
    )
    preflight = integration._preflight_runtime_artifact(
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
        tc_framework_bridge.prepare_model_construction(
            integration.host,
            request.framework_config,
            request.model_config,
        )
        model = tc_framework_bridge.build_meta_model(
            integration.host,
            request.framework_config,
            request.model_config,
        )
    tc_framework_bridge.assert_model_ready_for_runtime_binding(
        integration.host,
        model,
        context="TensorCast direct runtime artifact startup",
    )
    tc_framework_bridge.align_runtime_tensor_names(
        integration.host,
        model,
        getattr(resolved, "tensor_names", ()),
    )
    current_tensors = tc_framework_bridge.collect_runtime_binding_tensors(
        integration.host,
        model,
        remove_duplicate=False,
    )
    tc_framework_bridge.assert_tensor_names_match_expected(
        current_tensors,
        getattr(resolved, "tensor_names", ()),
    )
    tensor_schema_hash = tc_framework_bridge.compute_runtime_tensor_schema_hash(
        integration.host,
        current_tensors,
        remove_duplicate=False,
    )
    preflight = integration._preflight_runtime_artifact(
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
        runtime_state, binding_result = _load_prepared_local_ready_runtime_artifact(
            integration,
            request,
            resolved=resolved,
            manifest=manifest,
            model=model,
            context=context,
            target_device=target_device,
            tensor_schema_hash=tensor_schema_hash,
            restore_prepared_fn=restore_prepared_fn,
        )
    else:
        runtime_state, binding_result = _bind_durable_runtime_artifact(
            integration,
            request,
            resolved=resolved,
            policy=policy,
            model=model,
            context=context,
            target_device=target_device,
            tensor_names=tuple(current_tensors.keys()),
            tensor_schema_hash=tensor_schema_hash,
            bind_fn=bind_fn,
        )
    return RuntimeLoadResult(
        model=model,
        runtime_state=runtime_state,
        runtime_view=runtime_state.runtime_view,
        resolved_artifact=resolved,
        binding_result=binding_result,
    )


def _load_prepared_local_ready_runtime_artifact(
    integration: Any,
    request: _DirectRuntimeLoad,
    *,
    resolved: Any,
    manifest: Any,
    model: Any,
    context: Any,
    target_device: torch.device,
    tensor_schema_hash: str,
    restore_prepared_fn: Any,
) -> tuple[RuntimeBindingState, RuntimeBindingResult]:
    expected_member = request.expected_member
    if expected_member is None and context.placement is not None:
        expected_member = context.placement.member
    if expected_member is None:
        raise RestoreBindingError(
            "ArtifactRuntimeIntegration._load_existing_runtime_artifact prepared "
            "local-ready restore requires expected_member"
        )
    with restore_prepared_fn(
        resolved_artifact=resolved,
        target_device=target_device,
        expected_member=expected_member,
        expected_tensor_schema_hash=tensor_schema_hash,
        expected_serving_build_digest=getattr(manifest, "serving_build_digest", None),
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
        state_seed = integration._state_seed(
            resolved,
            tensor_schema_hash=tensor_schema_hash,
            execution_diagnostics=binding_result.execution_diagnostics,
            materialization_diagnostics=binding_result.materialization_diagnostics,
            binding_handle=restored,
            artifact_realization_report=artifact_report,
            readiness="runtime_local_ready",
        )
        runtime_state = integration._materializer().attach_and_finalize(
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
        return runtime_state, binding_result


def _bind_durable_runtime_artifact(
    integration: Any,
    request: _DirectRuntimeLoad,
    *,
    resolved: Any,
    policy: Any,
    model: Any,
    context: Any,
    target_device: torch.device,
    tensor_names: Sequence[str],
    tensor_schema_hash: str,
    bind_fn: Any,
) -> tuple[RuntimeBindingState, RuntimeBindingResult]:
    materialization = integration._load_materialization_options(
        request,
        resolved,
    )
    binding_result = bind_fn(
        resolved_artifact=resolved,
        tensor_names=tuple(tensor_names),
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
    )
    state_seed = integration._state_seed(
        resolved,
        tensor_schema_hash=tensor_schema_hash,
        execution_diagnostics=binding_result.execution_diagnostics,
        materialization_diagnostics=binding_result.materialization_diagnostics,
        binding_handle=binding_result.binding,
        artifact_realization_report=artifact_report,
    )
    runtime_state = integration._materializer().attach_and_finalize(
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
    return runtime_state, binding_result


def reload_existing_runtime_artifact(
    integration: Any,
    request: _RuntimeReload,
    *,
    swap_capable_fn: Any | None = None,
    swap_fn: Any | None = None,
) -> RuntimeReloadResult:
    swap_capable_fn = swap_capable_fn or is_runtime_binding_swap_capable
    swap_fn = swap_fn or swap_runtime_artifact
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
    if not swap_capable_fn(binding):
        raise ArtifactRuntimeIntegrationError(
            "ArtifactRuntimeIntegration._reload_existing_runtime_artifact requires a "
            "swap-capable runtime binding"
        )
    current_view = getattr(request.current_state, "runtime_view", None)
    expected_tensor_schema_hash = getattr(current_view, "tensor_schema_hash", None)
    runtime_tensors = None
    if request.model is not None:
        runtime_tensors = tc_framework_bridge.collect_runtime_binding_tensors(
            integration.host,
            request.model,
            remove_duplicate=False,
        )
        expected_tensor_schema_hash = (
            tc_framework_bridge.compute_runtime_tensor_schema_hash(
                integration.host,
                runtime_tensors,
                remove_duplicate=False,
            )
        )
        if target_device is None:
            for tensor in runtime_tensors.values():
                target_device = torch.device(tensor.device)
                break
    target_device = tc_framework_bridge.require_target_device(target_device)
    context = None
    if (
        request.model is not None
        or _artifact_locator_kind(request.artifact_locator) == "ranked_version_key"
    ):
        context = integration._framework_context(
            request.framework_config,
            request.model_config,
        )
    placement = None if context is None else context.placement
    preflight = integration._preflight_runtime_artifact(
        resolved_artifact=request.resolved_artifact,
        artifact_ref=request.artifact_ref,
        artifact_locator=request.artifact_locator,
        expected_tensor_schema_hash=expected_tensor_schema_hash,
        policy=request.policy,
        placement=placement,
    )
    resolved = preflight.resolved_artifact
    policy = preflight.runtime_artifact_policy
    materialization = integration._reload_materialization_options(
        request,
        resolved,
    )
    binding_result = swap_fn(
        binding=binding,
        resolved_artifact=resolved,
        tensor_names=None if runtime_tensors is None else tuple(runtime_tensors.keys()),
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
    )
    state_seed = integration._state_seed(
        resolved,
        tensor_schema_hash=str(expected_tensor_schema_hash or ""),
        execution_diagnostics=binding_result.execution_diagnostics,
        materialization_diagnostics=binding_result.materialization_diagnostics,
        binding_handle=binding_result.binding,
        artifact_realization_report=artifact_report,
    )
    if request.model is not None:
        context = context or integration._framework_context(
            request.framework_config,
            request.model_config,
        )
        runtime_state = integration._materializer().attach_and_finalize(
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
            ownership_handle=getattr(request.current_state, "ownership_handle", None),
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


def restore_retained_for_intent(
    integration: Any,
    request: _RetainedBindingAcquire,
    *,
    restore_fn: Any | None = None,
    rejection_reason_fn: Any | None = None,
) -> RetainedBindingResult:
    restore_fn = restore_fn or restore_retained_binding
    rejection_reason_fn = rejection_reason_fn or runtime_restore_rejection_reason
    target_device = tc_framework_bridge.require_target_device(request.target_device)
    authority = request.authority
    if authority is None:
        raise RestoreBindingError(
            "ArtifactRuntimeIntegration._restore_retained_for_intent requires authority"
        )
    rejection_reason = rejection_reason_fn(authority)
    if rejection_reason is not None:
        raise RestoreBindingError(rejection_reason)
    model = tc_framework_bridge.build_meta_model(
        integration.host,
        request.framework_config,
        request.model_config,
    )
    try:
        with restore_fn(
            authority=authority,
            target_device=target_device,
            expected_member=request.expected_member,
            caller_pid=os.getpid(),
            timeout_s=request.timeout_s,
            runtime=request.runtime,
            client=request.client,
            restore_fn=request.restore_fn,
        ) as restored:
            expected = getattr(authority, "expected", None)
            expected_tensor_schema_hash = getattr(expected, "tensor_schema_hash", None)
            artifact_report = _runtime_attachment_report_for_retained(
                authority=authority,
                tensors=restored.tensors,
                binding_handle=restored,
                target_device=target_device,
                tensor_schema_hash=str(expected_tensor_schema_hash or ""),
                reservation_bytes=restored.reservation_bytes,
            )
            state_seed = RuntimeStateSeed(
                artifact_ref=(
                    getattr(authority, "serving_artifact_id", None)
                    or getattr(authority, "local_serving_ref", None)
                    or ""
                ),
                serving_artifact_ref=getattr(authority, "serving_artifact_id", None),
                tensor_schema_hash=str(expected_tensor_schema_hash or ""),
                binding_value_ref=restored.binding_value_ref,
                local_serving_ref=getattr(authority, "local_serving_ref", None),
                readiness=str(
                    getattr(authority, "readiness", "") or "runtime_local_ready"
                ),
                diagnostics={
                    "reservation_bytes": int(restored.reservation_bytes),
                    "verification_state": str(
                        getattr(authority, "verification_state", "") or ""
                    ),
                    "artifact_realization_report": (
                        artifact_realization_report_to_dict(artifact_report)
                    ),
                },
                realization_report=artifact_report,
            )
            runtime_state = integration._materializer().attach_and_finalize(
                model=model,
                tensors=restored.tensors,
                binding_handle=restored,
                context=integration._framework_context(
                    request.framework_config,
                    request.model_config,
                ),
                state_seed=state_seed,
                replace_meta_params=True,
                target_device=target_device,
                model_config=request.model_config,
                run_process_after_load=False,
                expected_tensor_schema_hash=expected_tensor_schema_hash,
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
        raise RestoreBindingError("TensorCast retained binding acquire failed") from exc


__all__ = [
    "bind_runtime_artifact",
    "is_runtime_binding_swap_capable",
    "load_existing_runtime_artifact",
    "reload_existing_runtime_artifact",
    "restore_retained_for_intent",
    "swap_runtime_artifact",
]
