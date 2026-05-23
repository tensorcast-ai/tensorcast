#  Copyright (c) 2026, TensorCast Team.

"""Serving diagnostic helpers with no lifecycle authority."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SourceContractReport:
    version: int
    capability_flags: tuple[str, ...]
    ready: bool
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "capability_flags": list(self.capability_flags),
            "ready": bool(self.ready),
            "path": self.path,
        }


@dataclass(frozen=True)
class BindingValueReport:
    verification_state: str
    local_serving_ref: str | None = None
    binding_value_id: str | None = None
    verification_job_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "verification_state": self.verification_state,
            "local_serving_ref": self.local_serving_ref,
            "binding_value_id": self.binding_value_id,
            "verification_job_id": self.verification_job_id,
        }


@dataclass(frozen=True)
class RealizationReport:
    binding_layout_id: str | None
    binding_value: BindingValueReport
    execution: Mapping[str, Any] = field(default_factory=dict)
    plan: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "binding_layout_id": self.binding_layout_id,
            "binding_value": self.binding_value.to_dict(),
            "execution": dict(self.execution),
            "plan": dict(self.plan),
        }


@dataclass(frozen=True)
class ServingRealizationReport:
    source_artifact_ref: str
    serving_manifest_ref: str
    representation_contract_hash: str
    serving_build_digest: str
    tensor_schema_hash: str
    family: str
    tp_rank: int
    tp_world_size: int
    source_bound_contract: SourceContractReport
    realization: RealizationReport
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "source_artifact_ref": self.source_artifact_ref,
            "serving_manifest_ref": self.serving_manifest_ref,
            "representation_contract_hash": self.representation_contract_hash,
            "serving_build_digest": self.serving_build_digest,
            "tensor_schema_hash": self.tensor_schema_hash,
            "family": self.family,
            "tp_rank": int(self.tp_rank),
            "tp_world_size": int(self.tp_world_size),
            "source_bound_contract": self.source_bound_contract.to_dict(),
            "realization": self.realization.to_dict(),
        }

    def to_runtime_diagnostics(self) -> dict[str, Any]:
        return {"serving_realization_report": self.to_dict()}


def binding_layout_tensor_count(layout: Any) -> int:
    target_layout = getattr(layout, "target_layout", None)
    offsets = getattr(target_layout, "offsets", None)
    if offsets is None:
        return -1
    return len(offsets)


def binding_layout_profile_fields(layout: Any) -> dict[str, Any]:
    target_index_bytes = getattr(layout, "target_index_bytes", b"") or b""
    return {
        "target_index_bytes": len(target_index_bytes),
        "binding_tensor_count": binding_layout_tensor_count(layout),
    }


def binding_layout_debug_payload(
    layout: Any,
    *,
    target_device: Any,
    context: str,
    pid: int,
) -> dict[str, Any]:
    target_layout = layout.target_layout
    target_index_bytes = layout.target_index_bytes
    return {
        "context": str(context),
        "pid": int(pid),
        "target_device": str(target_device),
        "binding_layout_id": str(layout.binding_layout_id),
        "target_index_bytes_len": len(target_index_bytes),
        "target_index_sha256": hashlib.sha256(target_index_bytes).hexdigest(),
        "layout": {
            "layout_kind": int(target_layout.layout_kind),
            "index_kind": int(target_layout.index_kind),
            "tensor_spec_kind": int(target_layout.tensor_spec_kind),
            "logical_layout_hash": bytes(target_layout.logical_layout_hash).hex(),
            "view_id": str(target_layout.view_id),
            "storages": [
                {
                    "storage_id": str(storage.storage_id),
                    "device_id": int(storage.device_id),
                    "storage_length": int(storage.storage_length),
                    "mapping_base_offset": int(storage.mapping_base_offset),
                }
                for storage in target_layout.storages
            ],
            "offsets": [
                {
                    "name": str(offset.name),
                    "storage_id": str(offset.storage_id),
                    "storage_offset": int(offset.storage_offset),
                    "logical_length": int(offset.logical_length),
                }
                for offset in target_layout.offsets
            ],
        },
        "dst_specs": [
            {
                "name": str(spec.name),
                "dtype": str(spec.dtype),
                "shape": [int(v) for v in spec.shape],
                "stride": [int(v) for v in spec.stride],
                "storage_offset": int(spec.storage_offset),
                "logical_length": int(spec.logical_length),
            }
            for spec in layout.dst_specs
        ],
    }


__all__ = [
    "BindingValueReport",
    "RealizationReport",
    "ServingRealizationReport",
    "SourceContractReport",
    "binding_layout_debug_payload",
    "binding_layout_profile_fields",
    "binding_layout_tensor_count",
]
