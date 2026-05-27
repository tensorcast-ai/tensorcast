#  Copyright (c) 2026, TensorCast Team.
"""Source resolution helpers for artifact-runtime lifecycle paths."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import tensorcast.artifact_runtime.source as tc_source_catalog
from tensorcast.artifact_runtime.errors import (
    ArtifactRuntimeIntegrationError,
    SourceSubjectError,
)
from tensorcast.artifact_runtime.errors import (
    capability_missing as _capability_missing,
)
from tensorcast.artifact_runtime.host import (
    IntegrationHost,
    RecipeCachePolicy,
    SourceCatalogRequest,
    SourceDownloadPolicy,
    SourceSelector,
    SourceSubjectCoordinator,
)
from tensorcast.artifact_runtime.recipe.build import (
    RecipeBuildCacheConfig,
    recipe_build_cache_config_from_policy,
)
from tensorcast.artifact_runtime.source import (
    SourceSubject,
    is_public_disk_source_subject,
    resolve_source_subject,
    source_subject_broadcast_payload,
    source_subject_from_broadcast_payload,
)


def source_subject_for_mounted_source(
    *,
    source_artifact_ref: str,
    source_subject: Any,
) -> SourceSubject:
    if isinstance(source_subject, SourceSubject):
        subject_ref = tc_source_catalog.resolve_source_artifact_ref(
            source_subject.artifact_ref
        )
        if subject_ref != source_artifact_ref:
            raise ArtifactRuntimeIntegrationError(
                "mounted-source subject artifact_ref does not match "
                "realization artifact_ref"
            )
        return source_subject
    subject_artifact_ref = str(getattr(source_subject, "artifact_id", "") or "")
    if subject_artifact_ref and subject_artifact_ref != source_artifact_ref:
        raise ArtifactRuntimeIntegrationError(
            "mounted-source handle artifact_id does not match realization artifact_ref"
        )
    source_kind = (
        "public_disk" if is_public_disk_source_subject(source_subject) else "opaque"
    )
    return SourceSubject(
        artifact_ref=source_artifact_ref,
        subject=source_subject,
        source_kind=source_kind,
    )


def source_selector_for_subject(subject: SourceSubject) -> SourceSelector:
    source_path = getattr(subject.subject, "path", None)
    if source_path is None or not str(source_path).strip():
        raise ArtifactRuntimeIntegrationError(
            "mounted-source model_runtime realization requires a source "
            "selector or a source subject with a path"
        )
    return SourceSelector.local_path(str(source_path))


def host_source_subject_coordinator(
    host: IntegrationHost | None,
    framework_config: object | None,
) -> SourceSubjectCoordinator | None:
    if host is None or host.collective is None:
        return None
    return host.collective.source_subject_coordinator(framework_config)


def host_source_catalog_config(
    host: IntegrationHost | None,
    framework_config: Any | None,
    model_config: Any | None,
) -> Any | None:
    if host is None or host.source is None:
        return None
    return host.source.source_catalog_config(
        framework_config,
        model_config,
    )


def host_recipe_cache_policy(
    host: IntegrationHost | None,
    framework_config: Any | None,
    model_config: Any | None,
) -> RecipeCachePolicy | None:
    if host is None or host.source is None:
        return None
    policy = host.source.recipe_cache_policy(
        framework_config,
        model_config,
    )
    if policy is not None and not isinstance(policy, RecipeCachePolicy):
        raise ArtifactRuntimeIntegrationError(
            "IntegrationHost.source.recipe_cache_policy must return "
            "RecipeCachePolicy or None"
        )
    return policy


def local_ready_source_catalog(
    request: Any,
    *,
    host: IntegrationHost | None,
    source_subject: Any,
    source_artifact_ref: str,
) -> Any:
    try:
        expected_source_ref = tc_source_catalog.resolve_source_artifact_ref(
            source_artifact_ref
        )
    except ValueError as exc:
        raise ArtifactRuntimeIntegrationError(
            "ArtifactRuntimeIntegration.start(LocalSourceBootstrap) requires "
            "a real source artifact identity"
        ) from exc
    if request.source_catalog is not None:
        validate_source_catalog_artifact_ref(
            request.source_catalog,
            expected_source_artifact_ref=expected_source_ref,
        )
        return request.source_catalog
    if host is not None and host.source_catalog is not None:
        if not isinstance(request.source_selector, SourceSelector):
            raise ArtifactRuntimeIntegrationError(
                "IntegrationHost.source_catalog requires a core SourceSelector"
            )
        if request.model_config is None:
            raise ArtifactRuntimeIntegrationError(
                "IntegrationHost.source_catalog requires model_config"
            )
        source_catalog = host.source_catalog.build_catalog(
            SourceCatalogRequest(
                source_subject=source_subject,
                source_selector=request.source_selector,
                source_artifact_ref=expected_source_ref,
                framework_identity=host.framework.identity(request.model_config),
                framework_config=request.framework_config,
                model_config=request.model_config,
                download_policy=(
                    request.source_catalog_config
                    if isinstance(request.source_catalog_config, SourceDownloadPolicy)
                    else None
                ),
                cache_policy=(
                    request.cache_config
                    if isinstance(request.cache_config, RecipeCachePolicy)
                    else None
                ),
                source_catalog_config=request.source_catalog_config,
            )
        )
        validate_source_catalog_artifact_ref(
            source_catalog,
            expected_source_artifact_ref=expected_source_ref,
        )
        return source_catalog
    raise _capability_missing(
        "ArtifactRuntimeIntegration.start(LocalSourceBootstrap) requires "
        "IntegrationHost.source_catalog when recipe is not supplied",
        level="level2-local-bootstrap",
        capability="source_catalog",
        operation="local_bootstrap.source_catalog",
        required_methods=("build_catalog",),
        next_action=(
            "Add IntegrationHost(source_catalog=...) or provide a prepared "
            "recipe through the admin/offline bootstrap path."
        ),
    )


def validate_source_catalog_artifact_ref(
    source_catalog: Any,
    *,
    expected_source_artifact_ref: str,
) -> None:
    catalog_artifact_ref = getattr(source_catalog, "source_artifact_ref", None)
    if catalog_artifact_ref is None:
        raise ArtifactRuntimeIntegrationError(
            "SourceCatalogProvider returned a catalog without a real "
            "source_artifact_ref"
        )
    try:
        catalog_source_ref = tc_source_catalog.resolve_source_artifact_ref(
            str(catalog_artifact_ref)
        )
    except ValueError as exc:
        raise ArtifactRuntimeIntegrationError(
            "SourceCatalogProvider returned a catalog without a real "
            "source_artifact_ref"
        ) from exc
    if catalog_source_ref != expected_source_artifact_ref:
        raise ArtifactRuntimeIntegrationError(
            "SourceCatalogProvider returned source_artifact_ref "
            f"{catalog_source_ref!r}, expected {expected_source_artifact_ref!r}"
        )


def local_ready_recipe_cache_config(
    request: Any,
    *,
    source_catalog: Any,
) -> Any:
    cache_config_factory = request.cache_config_factory
    if callable(cache_config_factory):
        return cache_config_factory(source_catalog=source_catalog)
    if isinstance(request.cache_config, RecipeCachePolicy):
        return recipe_build_cache_config_from_policy(
            request.cache_config,
            source_catalog=source_catalog,
        )
    if request.cache_config is not None:
        return request.cache_config
    return RecipeBuildCacheConfig()


def resolve_source_subject_request(
    path: str | SourceSelector,
    *,
    verify_checksums: bool,
    coordinator: Any | None = None,
    resolve_fn: Any = resolve_source_subject,
) -> SourceSubject:
    if isinstance(path, SourceSelector):
        if path.kind != "local_path":
            raise SourceSubjectError(
                f"Unsupported TensorCast source selector kind: {path.kind}"
            )
        path = str(path.value)
    if coordinator is not None:
        should_coordinate = getattr(coordinator, "should_coordinate", None)
        if not callable(should_coordinate) or bool(should_coordinate()):
            return resolve_source_subject_with_coordinator(
                path,
                verify_checksums=verify_checksums,
                coordinator=coordinator,
                resolve_fn=resolve_fn,
            )
    return resolve_fn(path, verify_checksums=verify_checksums)


def resolve_source_subject_with_coordinator(
    path: str,
    *,
    verify_checksums: bool,
    coordinator: Any,
    resolve_fn: Any = resolve_source_subject,
) -> SourceSubject:
    source_rank = int(getattr(coordinator, "source_rank", 0) or 0)
    is_source_rank = getattr(coordinator, "is_source_rank", None)
    resolve_locally = bool(is_source_rank()) if callable(is_source_rank) else True
    subject = (
        resolve_fn(path, verify_checksums=verify_checksums) if resolve_locally else None
    )
    payload = None if subject is None else source_subject_broadcast_payload(subject)
    broadcast = getattr(coordinator, "broadcast_object", None)
    if not callable(broadcast):
        raise SourceSubjectError(
            "TensorCast source subject coordinator must provide "
            "broadcast_object(payload, src)"
        )
    payload = broadcast(payload, src=source_rank)
    if payload is None:
        raise SourceSubjectError(
            "TensorCast source subject coordinator returned no payload"
        )
    if not isinstance(payload, Mapping):
        raise SourceSubjectError(
            "TensorCast source subject coordinator must broadcast a mapping payload"
        )
    return source_subject_from_broadcast_payload(payload)


__all__ = [
    "host_recipe_cache_policy",
    "host_source_catalog_config",
    "host_source_subject_coordinator",
    "local_ready_recipe_cache_config",
    "local_ready_source_catalog",
    "resolve_source_subject_request",
    "resolve_source_subject_with_coordinator",
    "source_selector_for_subject",
    "source_subject_broadcast_payload",
    "source_subject_from_broadcast_payload",
    "source_subject_for_mounted_source",
    "validate_source_catalog_artifact_ref",
]
