#  Copyright (c) 2026, TensorCast Team.

"""Serving runtime intent DTOs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from tensorcast.serving.errors import AuthorityValidationError
from tensorcast.serving.hosts import RecipeCachePolicy, SourceSelector
from tensorcast.serving.policy import ServingArtifactLocator, ServingPolicy
from tensorcast.serving.retained_binding import ParsedRetainedServingBindingAuthority


@dataclass(frozen=True)
class BootstrapPolicy:
    fields: Mapping[str, object] = field(default_factory=dict)


class ServingIntent:
    """Marker base class for serving lifecycle intent DTOs."""


@dataclass(frozen=True)
class ExistingServingArtifact(ServingIntent):
    artifact_locator: ServingArtifactLocator | object
    policy: ServingPolicy | object | None = None


@dataclass(frozen=True)
class LocalSourceBootstrap(ServingIntent):
    source_selector: SourceSelector
    bootstrap_policy: Any
    cache_policy: RecipeCachePolicy | None = None


@dataclass(frozen=True)
class RetainedBindingAcquire(ServingIntent):
    authority: ParsedRetainedServingBindingAuthority

    def __post_init__(self) -> None:
        if not isinstance(self.authority, ParsedRetainedServingBindingAuthority):
            raise AuthorityValidationError(
                "RetainedBindingAcquire.authority must be "
                "ParsedRetainedServingBindingAuthority"
            )


@dataclass(frozen=True)
class RequestContext:
    framework_config: object | None = None
    model_config: object | None = None
    target_device: object | None = None
    timeout_s: float | None = 30.0


__all__ = [
    "BootstrapPolicy",
    "ExistingServingArtifact",
    "LocalSourceBootstrap",
    "RequestContext",
    "RetainedBindingAcquire",
    "ServingIntent",
]
