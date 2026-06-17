#  Copyright (c) 2026, TensorCast Team.
"""Artifact runtime replica publication actions.

These helpers are the public artifact-runtime actions for publishing or
retiring the replica represented by a realized runtime attachment.  The current
implementation delegates to the serving runtime binding implementation while
keeping callers away from ``ArtifactRuntimeSession``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from tensorcast.artifact_runtime.attachment import RuntimeAttachment
from tensorcast.artifact_runtime.config import TensorCastRuntimeConfig
from tensorcast.artifact_runtime.publication import replica as replica_publication


@dataclass(frozen=True)
class RuntimeReplicaPublicationSettings:
    """Runtime replica publication settings parsed from loader configuration."""

    policy: object
    ensure_runtime_initialized: Callable[[], None]

    @property
    def drain_timeout_s(self) -> float:
        return float(getattr(self.policy, "drain_timeout_s", 30.0))


def runtime_replica_publication_settings(
    config: TensorCastRuntimeConfig | Mapping[str, Any] | None = None,
) -> RuntimeReplicaPublicationSettings:
    """Parse publication settings from runtime loader configuration."""

    parsed = (
        config
        if isinstance(config, TensorCastRuntimeConfig)
        else TensorCastRuntimeConfig.from_mapping(config or {})
    )
    return RuntimeReplicaPublicationSettings(
        policy=parsed.replica_publication,
        ensure_runtime_initialized=parsed.runtime.ensure_initialized,
    )


def publish_runtime_replica(
    *,
    current_attachment: RuntimeAttachment,
    policy: object,
    ensure_runtime_initialized: Callable[[], None],
    profile_sink: Callable[[Mapping[str, object]], object] | None = None,
) -> RuntimeAttachment:
    """Publish the current artifact-backed runtime attachment as a replica."""

    return replica_publication.publish_current_replica(
        current_attachment=current_attachment,
        policy=policy,
        ensure_runtime_initialized=ensure_runtime_initialized,
        profile_sink=profile_sink,
    )


def project_runtime_replica_publication_state(
    *,
    current_attachment: RuntimeAttachment,
    state: str,
    reason: str | None = None,
    operation_id: str | None = None,
) -> RuntimeAttachment:
    """Return an attachment with an observational publication projection."""

    return replica_publication.project_current_replica_publication_state(
        current_attachment=current_attachment,
        state=state,
        reason=reason,
        operation_id=operation_id,
    )


def retire_runtime_replica(
    *,
    current_attachment: RuntimeAttachment,
    reason: str = "retire",
    drain_timeout_s: float | None = None,
    default_drain_timeout_s: float | None = None,
    ensure_runtime_initialized: Callable[[], None],
    profile_sink: Callable[[Mapping[str, object]], object] | None = None,
) -> RuntimeAttachment:
    """Retire the published replica tied to a runtime attachment."""

    return replica_publication.retire_current_replica(
        current_attachment=current_attachment,
        reason=reason,
        drain_timeout_s=drain_timeout_s,
        default_drain_timeout_s=default_drain_timeout_s,
        ensure_runtime_initialized=ensure_runtime_initialized,
        profile_sink=profile_sink,
    )


__all__ = [
    "RuntimeReplicaPublicationSettings",
    "project_runtime_replica_publication_state",
    "publish_runtime_replica",
    "retire_runtime_replica",
    "runtime_replica_publication_settings",
]
