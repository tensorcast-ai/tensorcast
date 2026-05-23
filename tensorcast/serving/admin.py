#  Copyright (c) 2026, TensorCast Team.
"""Admin/offline serving helpers that are not runtime integration APIs."""

from dataclasses import dataclass

from tensorcast.serving._runtime_impl.lifecycle import (
    build_local_ready_prepared_artifact,
)
from tensorcast.serving.builder.publication import (
    RecipePublicationContext,
    build_binding_finalize_build_intent,
    build_pure_transform_build_intent,
)
from tensorcast.serving.hosts import SourceSubjectCoordinator
from tensorcast.serving.local_ready import (
    freeze_local_ready_binding,
    prepare_local_ready_serving,
    prepare_same_binding_manifest_carrier,
)
from tensorcast.serving.runtime import LocalSourceBootstrap


@dataclass(frozen=True)
class AdminLocalSourceBootstrap(LocalSourceBootstrap):
    """Admin local bootstrap request with prebuilt lifecycle inputs."""

    coordinator: SourceSubjectCoordinator | None = None
    source_catalog_config: object | None = None
    cache_config_factory: object | None = None
    recipe: object | None = None
    source_subject: object | None = None
    source_artifact_ref: str | None = None
    model: object | None = None
    binding_factory: object | None = None


__all__ = [
    "AdminLocalSourceBootstrap",
    "RecipePublicationContext",
    "build_binding_finalize_build_intent",
    "build_local_ready_prepared_artifact",
    "build_pure_transform_build_intent",
    "freeze_local_ready_binding",
    "prepare_local_ready_serving",
    "prepare_same_binding_manifest_carrier",
]
