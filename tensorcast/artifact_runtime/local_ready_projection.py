#  Copyright (c) 2026, TensorCast Team.
"""Local-ready runtime attachment projection helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import tensorcast as tc
import tensorcast.artifact_runtime.diagnostics as tc_diagnostics
import tensorcast.artifact_runtime.recipe.local_ready as tc_local_ready
from tensorcast.api.store.realization_kernel import (
    ArtifactRealizationReport,
    ArtifactRealizationSpec,
    artifact_realization_report_to_dict,
)
from tensorcast.artifact_runtime.attachment import (
    RuntimeBindingState,
    RuntimeBindingView,
)
from tensorcast.artifact_runtime.attachment_materialization import (
    runtime_binding_state_from_runtime_view,
)
from tensorcast.artifact_runtime.contract import SourceBoundContractState
from tensorcast.artifact_runtime.dto import (
    PreparedRuntimeArtifact,
    RuntimeBindingValue,
)
from tensorcast.artifact_runtime.errors import ArtifactRuntimeIntegrationError


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
    return tc.BindingValueRef(
        binding_id=binding_id,
        binding_layout_id=binding_layout_id,
        binding_value_id=binding_value_id,
        seal_generation=int(getattr(current_value, "seal_generation", 0) or 0),
    )


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


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _strip_report_prefix(fields: Mapping[str, Any], *, prefix: str) -> dict[str, Any]:
    prefix_text = f"{prefix}_"
    return {
        key.removeprefix(prefix_text): value
        for key, value in fields.items()
        if key.startswith(prefix_text)
    }


__all__ = [
    "LocalReadyBindingContract",
    "LocalReadyManifestCarrierResult",
    "LocalReadyMaterializationIdentity",
    "LocalReadyRuntimeResult",
    "binding_value_ref_from_current_value",
    "build_local_ready_prepared_artifact",
    "execution_diagnostics_summary_fields",
    "local_ready_current_value_summary_fields",
    "source_bound_plan_diagnostics_summary_fields",
]
