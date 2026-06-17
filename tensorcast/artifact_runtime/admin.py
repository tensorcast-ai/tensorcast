#  Copyright (c) 2026, TensorCast Team.
"""Admin/offline runtime helpers that are not runtime integration APIs."""

from dataclasses import dataclass

from tensorcast.artifact_runtime.host import SourceSubjectCoordinator
from tensorcast.artifact_runtime.intent import LocalSourceBootstrap
from tensorcast.artifact_runtime.lifecycle import (
    build_local_ready_prepared_artifact,
)
from tensorcast.artifact_runtime.publication.context import (
    RecipePublicationContext,
    build_binding_finalize_build_intent,
    build_pure_transform_build_intent,
)
from tensorcast.artifact_runtime.recipe.local_ready import (
    freeze_local_ready_binding,
    prepare_same_binding_manifest_carrier,
    realize_local_ready_binding_from_source,
)


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
    "prepare_same_binding_manifest_carrier",
    "realize_local_ready_binding_from_source",
]
