#  Copyright (c) 2026, TensorCast Team.

"""Fail-closed request fact resolution for model-runtime realization."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch

from tensorcast.artifact_runtime.errors import ArtifactRuntimeIntegrationError
from tensorcast.artifact_runtime.intent import RequestContext


class ModelRuntimeRequestFactsError(ArtifactRuntimeIntegrationError):
    """Raised when model-runtime spec, request, and host facts disagree."""

    code = "invalid_argument"
    operation = "model_runtime_request"


@dataclass(frozen=True)
class ResolvedModelRuntimeRequestFacts:
    spec: Any
    context: Any


def resolve_model_runtime_request_facts(
    *,
    spec: Any,
    runtime_context: Any | None,
    host_context: Any | None = None,
    host_target_device: Any | None = None,
) -> ResolvedModelRuntimeRequestFacts:
    """Resolve request facts without silently preferring one authority."""

    context = runtime_context or RequestContext(
        target_device=getattr(spec, "device", None)
    )
    spec, context = _resolve_device_fact(
        spec=spec,
        context=context,
        host_target_device=host_target_device,
    )
    spec = _resolve_runtime_fact(
        spec=spec,
        context=context,
        host_context=host_context,
        field_name="topology",
        host_value=_placement_value(host_context, "topology"),
    )
    spec = _resolve_runtime_fact(
        spec=spec,
        context=context,
        host_context=host_context,
        field_name="member",
        host_value=_placement_value(host_context, "member"),
    )
    spec = _resolve_runtime_fact(
        spec=spec,
        context=context,
        host_context=host_context,
        field_name="adapter_version",
        host_value=_optional_text(getattr(host_context, "adapter_version", None)),
    )
    spec = _resolve_runtime_fact(
        spec=spec,
        context=context,
        host_context=host_context,
        field_name="runtime_abi_version",
        context_field_names=("runtime_abi_version", "serving_abi_version"),
        host_value=_optional_text(getattr(host_context, "serving_abi_version", None)),
    )
    return ResolvedModelRuntimeRequestFacts(spec=spec, context=context)


def _resolve_device_fact(
    *,
    spec: Any,
    context: Any,
    host_target_device: Any | None,
) -> tuple[Any, Any]:
    facts = (
        ("spec.device", getattr(spec, "device", None)),
        ("runtime_context.target_device", getattr(context, "target_device", None)),
        ("host.target_device", host_target_device),
    )
    resolved = _single_resolved_value(
        facts,
        normalize=_normalized_device,
        field_name="target_device",
    )
    if resolved is None:
        return spec, context
    if getattr(spec, "device", None) is None:
        spec = _replace_field(
            spec,
            field_name="device",
            new_value=resolved,
            subject="model_runtime spec",
        )
    if getattr(context, "target_device", None) is None:
        context = _replace_field(
            context,
            field_name="target_device",
            new_value=resolved,
            subject="model_runtime runtime_context",
        )
    return spec, context


def _resolve_runtime_fact(
    *,
    spec: Any,
    context: Any,
    host_context: Any | None,
    field_name: str,
    host_value: Any | None,
    context_field_names: tuple[str, ...] | None = None,
) -> Any:
    del host_context
    context_fields = context_field_names or (field_name,)
    context_value = _first_present_attr(context, context_fields)
    facts = (
        (f"spec.{field_name}", getattr(spec, field_name, None)),
        (f"runtime_context.{field_name}", context_value),
        (f"host.{field_name}", host_value),
    )
    resolved = _single_resolved_value(
        facts,
        normalize=lambda value: _normalized_fact(field_name, value),
        field_name=field_name,
    )
    if resolved is None or getattr(spec, field_name, None) is not None:
        return spec
    return _replace_field(
        spec,
        field_name=field_name,
        new_value=resolved,
        subject="model_runtime spec",
    )


def _single_resolved_value(
    facts: tuple[tuple[str, Any | None], ...],
    *,
    normalize: Any,
    field_name: str,
) -> Any | None:
    present: list[tuple[str, Any, Any]] = []
    for source, value in facts:
        if value is None:
            continue
        normalized = normalize(value)
        if normalized is None:
            continue
        present.append((source, value, normalized))
    if not present:
        return None
    expected = present[0][2]
    mismatches = [
        (source, normalized)
        for source, _value, normalized in present[1:]
        if normalized != expected
    ]
    if mismatches:
        details = {source: normalized for source, _value, normalized in present}
        raise ModelRuntimeRequestFactsError(
            f"model_runtime {field_name} facts disagree",
            details=details,
        )
    return present[0][1]


def _replace_field(
    obj: Any,
    *,
    field_name: str,
    new_value: Any,
    subject: str,
) -> Any:
    model_copy = getattr(obj, "model_copy", None)
    if callable(model_copy):
        return model_copy(update={field_name: new_value})
    try:
        return replace(obj, **{field_name: new_value})
    except TypeError as exc:
        raise ModelRuntimeRequestFactsError(
            f"{subject} must be dataclass-compatible when {field_name} is omitted",
            details={"field": field_name},
        ) from exc


def _normalized_device(value: Any) -> str:
    try:
        return str(torch.device(value))
    except Exception as exc:  # noqa: BLE001
        raise ModelRuntimeRequestFactsError(
            f"model_runtime target_device is invalid: {value!r}",
            details={"target_device": repr(value)},
        ) from exc


def _normalized_fact(field_name: str, value: Any) -> Any | None:
    if field_name == "topology":
        return _topology_identity(value)
    if field_name == "member":
        return _member_identity(value)
    return _optional_text(value)


def _topology_identity(value: Any) -> Any | None:
    digest = _optional_text(getattr(value, "schema_topology_digest", None))
    if digest is not None:
        return ("schema_topology_digest", digest)
    return _stable_value(value)


def _member_identity(value: Any) -> Any | None:
    member_id = _optional_text(getattr(value, "member_id", None))
    if member_id is not None:
        return (
            member_id,
            int(getattr(value, "member_index", 0)),
            int(getattr(value, "member_count", 1)),
            _optional_text(getattr(value, "group_id", None)),
        )
    return _stable_value(value)


def _stable_value(value: Any) -> Any | None:
    if value is None:
        return None
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return _stable_json(dump(mode="python"))
    if isinstance(value, Mapping):
        return _stable_json(value)
    return value


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _placement_value(host_context: Any | None, field_name: str) -> Any | None:
    placement = getattr(host_context, "placement", None)
    return getattr(placement, field_name, None)


def _first_present_attr(value: Any, names: tuple[str, ...]) -> Any | None:
    for name in names:
        attr = getattr(value, name, None)
        if attr is not None:
            return attr
    return None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "ModelRuntimeRequestFactsError",
    "ResolvedModelRuntimeRequestFacts",
    "resolve_model_runtime_request_facts",
]
