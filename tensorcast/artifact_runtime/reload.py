#  Copyright (c) 2026, TensorCast Team.
"""Artifact runtime reload actions."""

from __future__ import annotations

from collections.abc import Callable

from tensorcast.artifact_runtime.artifact.resolver import RuntimeArtifactResolver
from tensorcast.artifact_runtime.attachment import (
    RuntimeAttachment,
    RuntimeBindingState,
)
from tensorcast.artifact_runtime.errors import ConfigConflictError
from tensorcast.artifact_runtime.host import RuntimeHostCapabilities
from tensorcast.artifact_runtime.intent import ExistingRuntimeArtifact, RequestContext
from tensorcast.artifact_runtime.lifecycle import ArtifactRuntimeIntegration
from tensorcast.artifact_runtime.locator import ArtifactLocator
from tensorcast.artifact_runtime.policy import (
    RuntimePolicy,
    merge_runtime_reload_extra_config,
    normalize_runtime_reload_request_payload,
)
from tensorcast.artifact_runtime.publication import replica as replica_publication


def _reject_local_reload_artifact_locator(artifact_locator: object) -> None:
    if getattr(artifact_locator, "kind", None) == "local_path":
        raise ConfigConflictError(
            "TensorCast runtime reload requires a durable artifact locator, "
            "not a local source selector"
        )


def reload_runtime_attachment(
    *,
    current_attachment: RuntimeAttachment | RuntimeBindingState,
    artifact_locator: object,
    policy: object | None,
    runtime_host: RuntimeHostCapabilities,
    runtime_context: RequestContext,
    ensure_runtime_initialized: Callable[[], None],
    model: object | None = None,
    contract_identity: str | None = None,
    runtime_resolver: RuntimeArtifactResolver | None = None,
    profile_sink: object | None = None,
) -> RuntimeAttachment:
    """Reload an existing artifact-backed runtime binding."""

    _reject_local_reload_artifact_locator(artifact_locator)
    if not isinstance(artifact_locator, ArtifactLocator):
        raise ConfigConflictError(
            "TensorCast runtime reload requires an ArtifactLocator"
        )
    if policy is not None and not isinstance(policy, RuntimePolicy):
        raise ConfigConflictError(
            "TensorCast runtime reload requires a RuntimePolicy or None"
        )
    if isinstance(current_attachment, RuntimeAttachment):
        replica_publication.reject_reload_with_active_publication(current_attachment)
    ensure_runtime_initialized()
    current_state = (
        current_attachment.state
        if isinstance(current_attachment, RuntimeAttachment)
        else current_attachment
    )
    runtime_model = (
        model if model is not None else getattr(current_attachment, "model", None)
    )
    return ArtifactRuntimeIntegration(
        resolver=runtime_resolver,
        profile_sink=profile_sink,
        host=runtime_host,
    ).reload(
        current_state,
        ExistingRuntimeArtifact(
            artifact_locator=artifact_locator,
            policy=policy,
        ),
        runtime_context,
        model=runtime_model,
        contract_identity=contract_identity,
    )


__all__ = [
    "merge_runtime_reload_extra_config",
    "normalize_runtime_reload_request_payload",
    "reload_runtime_attachment",
]
