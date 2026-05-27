#  Copyright (c) 2026, TensorCast Team.
"""Binding-backed runtime attachment materialization.

This module owns the process-local framework boundary: attach daemon-owned
binding tensors to a framework model, run framework finalize hooks, validate
runtime tensor invariants, and build the runtime/model realization handles.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, cast

import torch

import tensorcast.artifact_runtime.recipe.semantic_validation as tc_semantic_validation
from tensorcast.api.store.realization_kernel import (
    ArtifactRealizationHandle,
    ArtifactRealizationReport,
    ArtifactRealizationSpec,
    emit_artifact_realization_profile_event,
    model_runtime_report_for,
)
from tensorcast.artifact_runtime.attachment import (
    RuntimeAttachment,
    RuntimeBindingState,
    RuntimeBindingView,
    RuntimeStateSeed,
)
from tensorcast.artifact_runtime.dto import FrameworkIntegrationContext
from tensorcast.artifact_runtime.errors import (
    AttachFinalizeError,
    OwnershipTransferError,
    SchemaMismatchError,
)
from tensorcast.artifact_runtime.errors import capability_missing as _capability_missing
from tensorcast.artifact_runtime.host import IntegrationHost, TensorSurfaceHost
from tensorcast.artifact_runtime.intent import RequestContext
from tensorcast.artifact_runtime.request_facts import (
    ModelRuntimeRequestFactsError,
    resolve_model_runtime_request_facts,
)

_LOGGER = logging.getLogger(__name__)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def close_quietly(handle: object) -> None:
    close = getattr(handle, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            _LOGGER.exception("Failed to close runtime attachment handle")


def runtime_attachment_realization_handle(
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


def model_runtime_spec_for_context(
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


def model_runtime_spec_with_context_defaults(
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


def model_runtime_realization_handle_for_spec(
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


def project_model_runtime_attachment(
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


def runtime_binding_state_from_runtime_view(
    *,
    binding: Any,
    runtime_view: RuntimeBindingView,
    artifact_ref: str | None = None,
    ownership_handle: Any | None = None,
    artifact_realization_report: ArtifactRealizationReport | None = None,
    model_runtime_spec: ArtifactRealizationSpec | None = None,
) -> RuntimeBindingState:
    realization_handle = runtime_attachment_realization_handle(
        report=artifact_realization_report,
        binding_handle=binding,
        owner=ownership_handle,
    )
    model_runtime_ref: dict[str, RuntimeBindingState] = {}
    model_runtime_handle = model_runtime_realization_handle_for_spec(
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
            realization_handle = runtime_attachment_realization_handle(
                report=realization_report,
                binding_handle=binding_handle,
                owner=owner,
            )
            model_runtime_ref: dict[str, RuntimeBindingState] = {}
            model_runtime_handle = model_runtime_realization_handle_for_spec(
                spec=(
                    model_runtime_spec_with_context_defaults(
                        spec=model_runtime_spec,
                        context=context,
                        target_device=target_device,
                    )
                    if model_runtime_spec is not None
                    else model_runtime_spec_for_context(
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
                close_quietly(realization_handle or owner)
                raise OwnershipTransferError(
                    "TensorCast runtime binding state construction failed"
                ) from exc
            model_runtime_ref["state"] = state
            self._emit("runtime_materialization.attach.done", state_seed)
            return state
        except OwnershipTransferError:
            raise
        except ModelRuntimeRequestFactsError:
            close_quietly(owner)
            raise
        except SchemaMismatchError:
            close_quietly(owner)
            raise
        except Exception as exc:
            close_quietly(owner)
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
        close_quietly(handle)


__all__ = [
    "RuntimeBindingMaterialization",
    "close_quietly",
    "model_runtime_realization_handle_for_spec",
    "model_runtime_spec_for_context",
    "model_runtime_spec_with_context_defaults",
    "project_model_runtime_attachment",
    "runtime_attachment_realization_handle",
    "runtime_binding_state_from_runtime_view",
]
