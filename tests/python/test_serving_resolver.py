#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

import tensorcast as tc
import tensorcast.serving.resolver as resolver_mod
from tensorcast.serving import PreparedServingArtifact, ServingArtifactResolver
from tensorcast.serving.artifact_manifest import ServingArtifactManifestHint


class _FakeArtifact:

    def __init__(self, descriptor: Any) -> None:
        self._descriptor = descriptor

    def describe(self) -> Any:
        return self._descriptor


def _descriptor(*, include_manifest: bool = True) -> Any:
    tensor_names = ["w"]
    if include_manifest:
        tensor_names.append(tc.SERVING_MANIFEST_TENSOR_NAME)
    return SimpleNamespace(
        tensor_names=tuple(tensor_names),
        tensor_metas={
            "w":
            SimpleNamespace(
                dtype=torch.float16,
                shape=(4, ),
                stride=(1, ),
                storage_offset=0,
                size_bytes=8,
            ),
        },
        total_bytes=8,
    )


def test_resolve_serving_artifact_cross_checks_manifest(monkeypatch) -> None:
    descriptor = _descriptor()
    artifact = _FakeArtifact(descriptor)
    resolver = ServingArtifactResolver(
        manifest_tensor_name=tc.SERVING_MANIFEST_TENSOR_NAME,
        schema_version=int(tc.ServingArtifactManifest.
                           model_fields["schema_version"].default),
    )
    tensor_schema_hash = resolver.compute_descriptor_tensor_schema_hash(
        descriptor)
    manifest = ServingArtifactManifestHint(
        serving_manifest_ref="tensor:manifest",
        representation_contract_hash="repr-hash",
        serving_build_digest="build-digest",
        tensor_schema_hash=tensor_schema_hash,
        canonical_tensor_count=1,
    )
    monkeypatch.setattr(resolver_mod.tc, "artifact", lambda ref: artifact)
    monkeypatch.setattr(
        resolver_mod.tc_artifact_manifest,
        "read_serving_artifact_manifest_tensor",
        lambda *_args, **_kwargs: manifest,
    )

    resolved = resolver.resolve("mi2:test:serving")

    assert resolved.artifact_ref == "mi2:test:serving"
    assert resolved.tensor_names == ("w", )
    assert resolver.cross_check(
        resolved,
        expected_tensor_schema_hash=tensor_schema_hash,
        serving_runtime_policy=manifest.to_runtime_policy(),
    ) is resolved
    with pytest.raises(RuntimeError, match="tensor schema hash mismatch"):
        resolver.cross_check(
            resolved,
            expected_tensor_schema_hash="wrong-schema",
        )


def test_resolve_serving_artifact_rejects_missing_manifest_tensor() -> None:
    resolver = ServingArtifactResolver(
        manifest_tensor_name=tc.SERVING_MANIFEST_TENSOR_NAME,
        schema_version=int(tc.ServingArtifactManifest.
                           model_fields["schema_version"].default),
    )

    with pytest.raises(RuntimeError, match="missing serving manifest tensor"):
        resolver.read_manifest(
            _FakeArtifact(_descriptor(include_manifest=False)),
            artifact_ref="mi2:test:serving",
        )


def test_resolve_prepared_rejects_local_ready_only_summary() -> None:
    resolver = ServingArtifactResolver(
        manifest_tensor_name=tc.SERVING_MANIFEST_TENSOR_NAME,
        schema_version=int(tc.ServingArtifactManifest.
                           model_fields["schema_version"].default),
    )
    prepared = PreparedServingArtifact(
        source_artifact_ref="disk:/model",
        serving_artifact_ref=None,
        serving_manifest_ref="tensor:manifest",
        representation_contract_hash="repr-hash",
        serving_build_digest="build-digest",
        readiness="serving_local_ready",
        family="demo",
        tensor_schema_hash="schema-hash",
    )

    with pytest.raises(RuntimeError, match="does not reference a durable"):
        resolver.resolve_prepared(prepared)
