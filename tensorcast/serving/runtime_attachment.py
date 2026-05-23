#  Copyright (c) 2026, TensorCast Team.

"""Runtime attachment state carried by framework model objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tensorcast.serving.runtime_view import RuntimeWorkerView


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _nested_attr(value: object | None, *names: str) -> object | None:
    current = value
    for name in names:
        if current is None:
            return None
        current = getattr(current, name, None)
    return current


def _first_attr(value: object | None, *names: str) -> object | None:
    if value is None:
        return None
    for name in names:
        candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    return None


def _binding_published_lease_id(binding: object | None) -> str | None:
    return _optional_text(_first_attr(binding, "published_lease_id")) or _optional_text(
        _nested_attr(binding, "_slot", "published_lease_id")
    )


def _binding_published_replica_id(binding: object | None) -> str | None:
    return _optional_text(
        _first_attr(binding, "published_replica_id")
    ) or _optional_text(_nested_attr(binding, "_slot", "published_replica_id"))


def _binding_has_active_published_replica(binding: object | None) -> bool:
    return (
        _binding_published_lease_id(binding) is not None
        or _binding_published_replica_id(binding) is not None
    )


@dataclass(frozen=True)
class RuntimeBindingView:
    """Read-only framework-facing view of core-owned runtime binding state."""

    serving_artifact_ref: str | None = None
    source_artifact_ref: str | None = None
    representation_contract_hash: str = ""
    tensor_schema_hash: str = ""
    binding_value_ref: Any | None = None
    local_serving_ref: str | None = None
    readiness: str = ""
    diagnostics: Mapping[str, Any] | None = None


@dataclass
class RuntimeBindingState:
    """Core-owned runtime binding lifecycle state placeholder."""

    binding: Any | None = None
    artifact_ref: str | None = None
    runtime_view: RuntimeBindingView | None = None
    ownership_handle: Any | None = None

    def close(self) -> None:
        handle = self.ownership_handle or self.binding
        retire = getattr(handle, "retire", None)
        if callable(retire) and _binding_has_active_published_replica(handle):
            retire(drain_timeout_s=None)
        close = getattr(handle, "close", None)
        if callable(close):
            close()


@dataclass(frozen=True)
class RuntimeStateSeed:
    """Core state facts known before framework tensor materialization."""

    artifact_ref: str | None = None
    serving_artifact_ref: str | None = None
    source_artifact_ref: str | None = None
    representation_contract_hash: str = ""
    tensor_schema_hash: str = ""
    binding_value_ref: Any | None = None
    local_serving_ref: str | None = None
    readiness: str = "loaded"
    diagnostics: Mapping[str, Any] | None = None

    def runtime_view(self) -> RuntimeBindingView:
        return RuntimeBindingView(
            serving_artifact_ref=self.serving_artifact_ref,
            source_artifact_ref=self.source_artifact_ref,
            representation_contract_hash=self.representation_contract_hash,
            tensor_schema_hash=self.tensor_schema_hash,
            binding_value_ref=self.binding_value_ref,
            local_serving_ref=self.local_serving_ref,
            readiness=self.readiness,
            diagnostics=self.diagnostics,
        )


@dataclass(frozen=True)
class RuntimeAttachment:
    model: object
    state: RuntimeBindingState
    view: RuntimeWorkerView
    prepared: Any | None = None
    recipe: Any | None = None


__all__ = [
    "RuntimeAttachment",
    "RuntimeBindingState",
    "RuntimeBindingView",
    "RuntimeStateSeed",
]
