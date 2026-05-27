#  Copyright (c) 2026, TensorCast Team.
"""Runtime attachment realization report builders.

The artifact-runtime lifecycle layer should orchestrate runtime work, while the
shared realization model owns report, target-plan, and resource-envelope facts.
This module keeps runtime attachment report construction close to that model
without forcing framework lifecycle code to hand-build report details.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch

from tensorcast.api.store.common import canonical_index_to_bytes
from tensorcast.api.store.realization_kernel import (
    ArtifactRealizationReport,
    RealizationTargetPlan,
    ResolvedArtifactSelection,
    envelope_for_runtime_attachment,
    report_for_runtime_attachment,
    resolve_artifact_selection,
)
from tensorcast.api.store.types import CanonicalIndex, CanonicalIndexEntry
from tensorcast.artifact_runtime.artifact.resolver import (
    ResolvedRuntimeArtifact,
    canonical_index_from_descriptor,
)

_LOGGER = logging.getLogger(__name__)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _canonical_index_bytes_from_tensors(
    tensors: Mapping[str, torch.Tensor],
) -> bytes:
    entries: list[CanonicalIndexEntry] = []
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


def runtime_attachment_report_for_resolved(
    *,
    resolved: ResolvedRuntimeArtifact | Any,
    tensors: Mapping[str, torch.Tensor],
    binding_handle: Any | None,
    target_device: Any,
    tensor_schema_hash: str,
    source_selection: ResolvedArtifactSelection | None = None,
    execution_diagnostics: Any | None = None,
    materialization_diagnostics: Any | None = None,
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
    )


def runtime_attachment_report_for_retained(
    *,
    authority: Any,
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


def runtime_attachment_report_for_artifact_id(
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
    )


__all__ = [
    "runtime_attachment_report_for_artifact_id",
    "runtime_attachment_report_for_resolved",
    "runtime_attachment_report_for_retained",
]
