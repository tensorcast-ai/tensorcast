#  Copyright (c) 2026, TensorCast Team.
"""Serving binding plan identity shared by trace and recipe compilation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from typing import Any

from tensorcast.types import ServingBindingMemberRef, ServingTopologyRef


@dataclass(frozen=True)
class ServingBindingPlan:
    """Cache and correctness identity for serving source bootstrap."""

    model_id: str
    model_revision: str | None
    dtype: str
    framework_name: str
    adapter_version: str
    serving_abi_version: str
    trace_cache_schema_version: int
    model_hash: str | None = None
    runtime_version: str | None = None
    framework_version: str | None = None
    tp_rank: int = 0
    tp_world_size: int = 1
    topology_ref: ServingTopologyRef | None = None
    member_ref: ServingBindingMemberRef | None = None
    placement: Any | None = None
    source_artifact_ref: str | None = None
    source_metadata_fingerprint: str | None = None
    source_schema_hash: str | None = None
    source_reuse_decision: Any | None = None
    selected_source_subject: Any | None = None
    model_config_digest: str | None = None
    load_config_digest: str | None = None
    serving_build_digest: str | None = None
    representation_contract_hash: str | None = None
    binding_layout_id: str | None = None
    target_layout_hash: str | None = None
    tensor_schema_hash: str | None = None
    resolved_spec_digest: str | None = None
    serving_facts: Any | None = None
    trace_plan: Any | None = None
    tensor_schema: tuple[Any, ...] = ()
    source_hull: tuple[Any, ...] = ()
    realization_plan: tuple[Any, ...] = ()
    realization_fallback_plan: tuple[Any, ...] = ()
    realization_plan_proto: bytes = b""
    realization_plan_digest: str | None = None
    realization_plan_count: int = 0
    realization_execution_policy: Any | None = None
    semantic_validation_spec: Any | None = None
    resolved_spec_cache_entry: Any | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def base_payload(self) -> dict[str, Any]:
        payload = {
            "model_hash": self.model_hash or self.model_id,
            "model": self.model_id,
            "revision": self.model_revision,
            "dtype": self.dtype,
            "version": self.runtime_version or self.framework_version,
            "trace_cache_schema_version": self.trace_cache_schema_version,
            "tp_rank": int(self.tp_rank),
            "tp_world_size": int(self.tp_world_size),
            "topology_ref": _jsonable(self.topology_ref),
            "member_ref": _jsonable(self.member_ref),
            "placement": _jsonable(self.placement),
            "identity_extra": _jsonable(self.extra),
        }
        payload.update(_optional_identity_payload(self))
        return payload

    def with_resolved_spec_cache_entry(
        self,
        resolved_spec_cache_entry: Any,
    ) -> "ServingBindingPlan":
        return replace(
            self,
            resolved_spec_cache_entry=resolved_spec_cache_entry,
            source_schema_hash=_optional_str(
                getattr(resolved_spec_cache_entry, "source_schema_hash", None)
            ),
            source_reuse_decision=getattr(
                resolved_spec_cache_entry, "source_reuse", None
            ),
            model_config_digest=_optional_str(
                getattr(resolved_spec_cache_entry, "model_config_digest", None)
            ),
            load_config_digest=_optional_str(
                getattr(resolved_spec_cache_entry, "load_config_digest", None)
            ),
            serving_build_digest=_optional_str(
                getattr(resolved_spec_cache_entry, "serving_build_digest", None)
            ),
            binding_layout_id=_optional_str(
                getattr(resolved_spec_cache_entry, "binding_layout_id", None)
            ),
            target_layout_hash=_optional_str(
                getattr(resolved_spec_cache_entry, "target_layout_hash", None)
            ),
            tensor_schema_hash=_optional_str(
                getattr(resolved_spec_cache_entry, "tensor_schema_hash", None)
            ),
            resolved_spec_digest=_optional_str(
                getattr(resolved_spec_cache_entry, "spec_digest", None)
            ),
            topology_ref=getattr(resolved_spec_cache_entry, "topology", None)
            or self.topology_ref,
            member_ref=getattr(resolved_spec_cache_entry, "member", None)
            or self.member_ref,
        )

    def source_payload(
        self,
        *,
        source_artifact_ref: str,
        metadata_fingerprint: str,
    ) -> dict[str, Any]:
        payload = self.base_payload()
        payload.update(
            {
                "source_artifact_ref": source_artifact_ref,
                "metadata_fingerprint": metadata_fingerprint,
            }
        )
        return payload

    def recipe_payload(self, *, metadata_fingerprint: str) -> dict[str, Any]:
        payload = self.base_payload()
        payload.update(
            {
                "metadata_fingerprint": metadata_fingerprint,
                "framework_name": self.framework_name,
                "framework_version": self.framework_version,
                "adapter_version": self.adapter_version,
                "serving_abi_version": self.serving_abi_version,
            }
        )
        return payload

    def trace_payload(self, *, metadata_fingerprint: str) -> dict[str, Any]:
        payload = self.base_payload()
        payload["metadata_fingerprint"] = metadata_fingerprint
        return payload

    def compile_payload(
        self,
        *,
        source_artifact_ref: str,
        source_metadata_fingerprint: str,
        serving_facts: Any,
        tensor_schema: Any,
        semantic_validation_spec: Any,
    ) -> dict[str, Any]:
        payload = self.source_payload(
            source_artifact_ref=source_artifact_ref,
            metadata_fingerprint=source_metadata_fingerprint,
        )
        payload.update(
            {
                "runtime_version": self.runtime_version,
                "framework_name": serving_facts.framework_name,
                "framework_version": serving_facts.framework_version,
                "adapter_version": serving_facts.adapter_version,
                "serving_abi_version": serving_facts.serving_abi_version,
                "identity_framework_name": self.framework_name,
                "identity_framework_version": self.framework_version,
                "identity_adapter_version": self.adapter_version,
                "identity_serving_abi_version": self.serving_abi_version,
                "support_level": str(serving_facts.support_level),
                "runtime_only_tensor_names": list(
                    serving_facts.runtime_only_tensor_names
                ),
                "process_after_load_class": str(serving_facts.process_after_load_class),
                "post_bind_finalize_class": str(serving_facts.post_bind_finalize_class),
                "tensor_schema": [
                    {
                        "name": item.name,
                        "dtype": item.dtype,
                        "shape": list(item.shape),
                        "stride": list(item.stride),
                    }
                    for item in tensor_schema
                ],
                "semantic_validation_spec": {
                    "kind": semantic_validation_spec.kind,
                    "payload": _jsonable(semantic_validation_spec.payload),
                },
                "version": self.framework_version,
            }
        )
        return payload

    def with_compiled_artifacts(
        self,
        *,
        source_artifact_ref: str,
        source_metadata_fingerprint: str,
        serving_facts: Any,
        trace_plan: Any,
        tensor_schema: tuple[Any, ...],
        source_hull: tuple[Any, ...],
        realization_plan: tuple[Any, ...],
        realization_fallback_plan: tuple[Any, ...],
        realization_plan_proto: bytes,
        realization_plan_count: int,
        semantic_validation_spec: Any,
        source_schema_hash: str | None = None,
        tensor_schema_hash: str | None = None,
        realization_plan_digest: str | None = None,
        resolved_spec_cache_entry: Any | None = None,
    ) -> "ServingBindingPlan":
        return replace(
            self,
            source_artifact_ref=str(source_artifact_ref),
            source_metadata_fingerprint=str(source_metadata_fingerprint),
            source_schema_hash=_optional_str(source_schema_hash),
            serving_facts=serving_facts,
            trace_plan=trace_plan,
            tensor_schema=tuple(tensor_schema),
            tensor_schema_hash=_optional_str(tensor_schema_hash),
            source_hull=tuple(source_hull),
            realization_plan=tuple(realization_plan),
            realization_fallback_plan=tuple(realization_fallback_plan),
            realization_plan_proto=bytes(realization_plan_proto or b""),
            realization_plan_digest=_optional_str(realization_plan_digest),
            realization_plan_count=int(realization_plan_count),
            semantic_validation_spec=semantic_validation_spec,
            resolved_spec_cache_entry=resolved_spec_cache_entry,
        )

    def compiled_artifact_payload(self) -> dict[str, Any]:
        return {
            "source_artifact_ref": self.source_artifact_ref,
            "source_metadata_fingerprint": self.source_metadata_fingerprint,
            "source_schema_hash": self.source_schema_hash,
            "source_reuse_decision": _jsonable(self.source_reuse_decision),
            "selected_source_subject": _jsonable(self.selected_source_subject),
            "model_config_digest": self.model_config_digest,
            "load_config_digest": self.load_config_digest,
            "serving_build_digest": self.serving_build_digest,
            "representation_contract_hash": self.representation_contract_hash,
            "binding_layout_id": self.binding_layout_id,
            "target_layout_hash": self.target_layout_hash,
            "tensor_schema_hash": self.tensor_schema_hash,
            "resolved_spec_digest": self.resolved_spec_digest,
            "serving_facts": _jsonable(self.serving_facts),
            "trace_plan": _jsonable(self.trace_plan),
            "tensor_schema": _jsonable(self.tensor_schema),
            "source_hull": _jsonable(self.source_hull),
            "realization_plan": _jsonable(self.realization_plan),
            "realization_fallback_plan": _jsonable(self.realization_fallback_plan),
            "realization_plan_proto_size": len(self.realization_plan_proto),
            "realization_plan_digest": self.realization_plan_digest,
            "realization_plan_count": int(self.realization_plan_count),
            "realization_execution_policy": _jsonable(
                self.realization_execution_policy
            ),
            "semantic_validation_spec": _jsonable(self.semantic_validation_spec),
            "resolved_spec_cache_entry": _jsonable(self.resolved_spec_cache_entry),
            "topology_ref": _jsonable(self.topology_ref),
            "member_ref": _jsonable(self.member_ref),
        }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _jsonable(value.model_dump(mode="python"))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_jsonable(item) for item in value]
    return repr(value)


def _optional_str(value: Any | None) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _optional_identity_payload(plan: ServingBindingPlan) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field_name in (
        "source_schema_hash",
        "source_reuse_decision",
        "selected_source_subject",
        "model_config_digest",
        "load_config_digest",
        "serving_build_digest",
        "representation_contract_hash",
        "binding_layout_id",
        "target_layout_hash",
        "tensor_schema_hash",
        "resolved_spec_digest",
        "realization_plan_digest",
        "realization_execution_policy",
    ):
        value = getattr(plan, field_name)
        if value is not None:
            payload[field_name] = _jsonable(value)
    if plan.resolved_spec_cache_entry is not None:
        payload["resolved_spec_cache_entry"] = _jsonable(plan.resolved_spec_cache_entry)
    return payload


__all__ = ["ServingBindingPlan"]
