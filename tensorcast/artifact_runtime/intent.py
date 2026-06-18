#  Copyright (c) 2026, TensorCast Team.

"""Artifact runtime intent DTOs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tensorcast.artifact_runtime.errors import AuthorityValidationError
from tensorcast.artifact_runtime.locator import ArtifactLocator
from tensorcast.artifact_runtime.policy import RuntimePolicy
from tensorcast.retained_realization_authority import (
    ParsedRetainedRealizationAuthority,
)

if TYPE_CHECKING:
    from tensorcast.artifact_runtime.host import RecipeCachePolicy, SourceSelector


@dataclass(frozen=True)
class BootstrapPolicy:
    fields: Mapping[str, object] = field(default_factory=dict)


class RuntimeIntent:
    """Marker base class for artifact runtime lifecycle intent DTOs."""


@dataclass(frozen=True)
class ExistingRuntimeArtifact(RuntimeIntent):
    artifact_locator: ArtifactLocator | object
    policy: RuntimePolicy | object | None = None


@dataclass(frozen=True)
class LocalSourceBootstrap(RuntimeIntent):
    source_selector: SourceSelector
    bootstrap_policy: Any
    cache_policy: RecipeCachePolicy | None = None


@dataclass(frozen=True)
class RetainedBindingAcquire(RuntimeIntent):
    authority: ParsedRetainedRealizationAuthority | None = None
    local_serving_ref: str | None = None
    expected_member: Any | None = None
    expected_tensor_schema_hash: str | None = None
    expected_serving_build_digest: str | None = None
    expected_target_layout_hash: str | None = None
    expected_daemon_id: str | None = None
    expected_daemon_session_id: str | None = None
    serving_artifact_id: str | None = None

    def __post_init__(self) -> None:
        has_authority = self.authority is not None
        has_local_ref = bool(self.local_serving_ref)
        if has_authority == has_local_ref:
            raise AuthorityValidationError(
                "RetainedBindingAcquire requires exactly one of authority "
                "or local_serving_ref"
            )
        if has_authority and not isinstance(
            self.authority, ParsedRetainedRealizationAuthority
        ):
            raise AuthorityValidationError(
                "RetainedBindingAcquire.authority must be "
                "ParsedRetainedRealizationAuthority"
            )
        if has_local_ref and (
            not self.expected_tensor_schema_hash
            or not self.expected_serving_build_digest
        ):
            raise AuthorityValidationError(
                "RetainedBindingAcquire(local_serving_ref=...) requires "
                "expected_tensor_schema_hash and expected_serving_build_digest"
            )


@dataclass(frozen=True)
class RequestContext:
    framework_config: object | None = None
    model_config: object | None = None
    target_device: object | None = None
    timeout_s: float | None = 30.0


RuntimeRequestContext = RequestContext


__all__ = [
    "BootstrapPolicy",
    "ExistingRuntimeArtifact",
    "LocalSourceBootstrap",
    "RequestContext",
    "RetainedBindingAcquire",
    "RuntimeIntent",
    "RuntimeRequestContext",
]
