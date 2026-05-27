#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

import tensorcast as tc
import tensorcast.artifact_runtime.artifact.resolver as resolver_mod
from tensorcast.artifact_runtime.artifact.resolver import RuntimeArtifactResolver
from tensorcast.artifact_runtime.dto import PreparedRuntimeArtifact


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
            "w": SimpleNamespace(
                dtype=torch.float16,
                shape=(4,),
                stride=(1,),
                storage_offset=0,
                size_bytes=8,
            ),
        },
        total_bytes=8,
    )


def _manifest(*, tensor_schema_hash: str) -> tc.RuntimeArtifactManifest:
    return tc.RuntimeArtifactManifest(
        framework_name="vllm",
        adapter_version="adapter-v1",
        serving_abi_version="abi-v1",
        representation_contract_hash="repr-hash",
        serving_build_digest="build-digest",
        tensor_schema_hash=tensor_schema_hash,
        canonical_tensor_count=1,
        serving_manifest_ref="tensor:manifest",
        builder_mode=tc.BuilderMode.PURE_TRANSFORM,
        build_pipeline_version="pipeline-v1",
    )


def test_resolve_runtime_artifact_cross_checks_manifest(monkeypatch) -> None:
    descriptor = _descriptor()
    artifact = _FakeArtifact(descriptor)
    resolver = RuntimeArtifactResolver(
        manifest_tensor_name=tc.SERVING_MANIFEST_TENSOR_NAME,
        schema_version=int(
            tc.RuntimeArtifactManifest.model_fields["schema_version"].default
        ),
        open_artifact_fn=lambda _ref: artifact,
    )
    tensor_schema_hash = resolver.compute_descriptor_tensor_schema_hash(descriptor)
    manifest = _manifest(tensor_schema_hash=tensor_schema_hash)
    monkeypatch.setattr(
        resolver_mod.tc_artifact_manifest,
        "read_runtime_artifact_manifest_tensor",
        lambda *_args, **_kwargs: manifest,
    )

    resolved = resolver.resolve("mi2:test:serving")

    assert resolved.artifact_ref == "mi2:test:serving"
    assert resolved.tensor_names == ("w",)
    assert (
        resolver.cross_check(
            resolved,
            expected_tensor_schema_hash=tensor_schema_hash,
            runtime_artifact_policy=manifest.to_runtime_policy(),
        )
        is resolved
    )
    with pytest.raises(RuntimeError, match="tensor schema hash mismatch"):
        resolver.cross_check(
            resolved,
            expected_tensor_schema_hash="wrong-schema",
        )


def test_resolve_runtime_artifact_rejects_missing_manifest_tensor() -> None:
    resolver = RuntimeArtifactResolver(
        manifest_tensor_name=tc.SERVING_MANIFEST_TENSOR_NAME,
        schema_version=int(
            tc.RuntimeArtifactManifest.model_fields["schema_version"].default
        ),
    )

    with pytest.raises(RuntimeError, match="missing runtime manifest tensor"):
        resolver.read_manifest(
            _FakeArtifact(_descriptor(include_manifest=False)),
            artifact_ref="mi2:test:serving",
        )


def test_resolve_prepared_rejects_local_ready_only_summary() -> None:
    resolver = RuntimeArtifactResolver(
        manifest_tensor_name=tc.SERVING_MANIFEST_TENSOR_NAME,
        schema_version=int(
            tc.RuntimeArtifactManifest.model_fields["schema_version"].default
        ),
    )
    prepared = PreparedRuntimeArtifact(
        source_artifact_ref="disk:/model",
        serving_artifact_ref=None,
        serving_manifest_ref="tensor:manifest",
        representation_contract_hash="repr-hash",
        serving_build_digest="build-digest",
        readiness="runtime_local_ready",
        family="demo",
        tensor_schema_hash="schema-hash",
    )

    with pytest.raises(RuntimeError, match="does not reference a durable"):
        resolver.resolve_prepared(prepared)


def test_resolve_prepared_reads_manifest_tensor(monkeypatch) -> None:
    descriptor = _descriptor()
    artifact = _FakeArtifact(descriptor)
    resolver = RuntimeArtifactResolver(
        manifest_tensor_name=tc.SERVING_MANIFEST_TENSOR_NAME,
        schema_version=int(
            tc.RuntimeArtifactManifest.model_fields["schema_version"].default
        ),
        open_artifact_fn=lambda _ref: artifact,
    )
    tensor_schema_hash = resolver.compute_descriptor_tensor_schema_hash(descriptor)
    manifest = _manifest(tensor_schema_hash=tensor_schema_hash)
    prepared = PreparedRuntimeArtifact(
        source_artifact_ref="mi2:test:source",
        serving_artifact_ref="mi2:test:serving",
        serving_manifest_ref="tensor:manifest",
        representation_contract_hash="repr-hash",
        serving_build_digest="build-digest",
        readiness="runtime_published_ready",
        family="demo",
        tensor_schema_hash=tensor_schema_hash,
    )
    calls = {"read_manifest": 0}

    def _read_manifest(*_args, **_kwargs):
        calls["read_manifest"] += 1
        return manifest

    monkeypatch.setattr(
        resolver_mod.tc_artifact_manifest,
        "read_runtime_artifact_manifest_tensor",
        _read_manifest,
    )

    resolved = resolver.resolve_prepared(prepared)

    assert calls["read_manifest"] == 1
    assert resolved.manifest == manifest


def test_resolve_prepared_rejects_manifest_summary_mismatch(monkeypatch) -> None:
    descriptor = _descriptor()
    artifact = _FakeArtifact(descriptor)
    resolver = RuntimeArtifactResolver(
        manifest_tensor_name=tc.SERVING_MANIFEST_TENSOR_NAME,
        schema_version=int(
            tc.RuntimeArtifactManifest.model_fields["schema_version"].default
        ),
        open_artifact_fn=lambda _ref: artifact,
    )
    tensor_schema_hash = resolver.compute_descriptor_tensor_schema_hash(descriptor)
    manifest = _manifest(tensor_schema_hash=tensor_schema_hash).model_copy(
        update={"serving_build_digest": "other-build"}
    )
    prepared = PreparedRuntimeArtifact(
        source_artifact_ref="mi2:test:source",
        serving_artifact_ref="mi2:test:serving",
        serving_manifest_ref="tensor:manifest",
        representation_contract_hash="repr-hash",
        serving_build_digest="build-digest",
        readiness="runtime_published_ready",
        family="demo",
        tensor_schema_hash=tensor_schema_hash,
    )
    monkeypatch.setattr(
        resolver_mod.tc_artifact_manifest,
        "read_runtime_artifact_manifest_tensor",
        lambda *_args, **_kwargs: manifest,
    )

    with pytest.raises(RuntimeError, match="summary field serving_build_digest"):
        resolver.resolve_prepared(prepared)
