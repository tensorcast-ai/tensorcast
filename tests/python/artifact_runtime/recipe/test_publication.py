#  Copyright (c) 2026, TensorCast Team.

import pytest

from tensorcast.artifact_runtime.publication.context import (
    RecipePublicationContext,
    build_pure_transform_build_intent,
)
from tensorcast.artifact_runtime.recipe.publication import (
    build_binding_finalize_publication_bundle_from_context,
    build_pure_transform_publication_spec_from_context,
)
from tensorcast.types import BindingValueRef, BuilderMode


def _context() -> RecipePublicationContext:
    return RecipePublicationContext(
        source_artifact_ref="mi2:test:source",
        framework_name="torch",
        adapter_version="adapter-v1",
        serving_abi_version="abi-v1",
        logical_topology_json='{"topology":"framework-owned"}',
    )


def test_publication_context_builds_serving_intent() -> None:
    intent = build_pure_transform_build_intent(
        _context(),
        build_pipeline_version="pipeline-v1",
        representation_contract_hash="repr-hash",
    )

    assert intent.builder_mode == BuilderMode.PURE_TRANSFORM
    assert intent.framework_name == "torch"
    assert intent.adapter_version == "adapter-v1"
    assert intent.serving_abi_version == "abi-v1"
    assert intent.source_artifact_ref == "mi2:test:source"
    assert intent.representation_contract_hash == "repr-hash"


def test_publication_context_passes_framework_topology_json() -> None:
    spec = build_pure_transform_publication_spec_from_context(
        _context(),
        build_pipeline_version="pipeline-v1",
        representation_contract_hash="repr-hash",
    )

    assert spec.logical_topology_json == '{"topology":"framework-owned"}'


def test_binding_finalize_publication_requires_explicit_admission_facts() -> None:
    with pytest.raises(ValueError, match="explicit admission_facts"):
        build_binding_finalize_publication_bundle_from_context(
            _context(),
            publication_subject=BindingValueRef(
                binding_id="binding-1",
                binding_layout_id="layout-1",
                binding_value_id="value-1",
                seal_generation=1,
            ),
            canonical_index=None,  # type: ignore[arg-type]
            build_pipeline_version="pipeline-v1",
            representation_contract_hash="repr-hash",
        )
