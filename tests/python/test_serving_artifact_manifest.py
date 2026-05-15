#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import pytest
import torch

import tensorcast as tc
from tensorcast.serving.artifact_manifest import (
    SERVING_MANIFEST_TENSOR_NAME,
    ServingArtifactManifestHint,
    cross_check_serving_artifact_manifest,
    read_serving_artifact_manifest_tensor,
)


def _manifest(**overrides) -> tc.ServingArtifactManifest:
    values = {
        "framework_name": "vllm",
        "adapter_version": "adapter-v1",
        "serving_abi_version": "abi-v1",
        "representation_contract_hash": "repr-hash",
        "serving_build_digest": "build-digest",
        "tensor_schema_hash": "schema-hash",
        "canonical_tensor_count": 1,
        "builder_mode": tc.BuilderMode.PURE_TRANSFORM,
        "build_pipeline_version": "pipeline-v1",
    }
    values.update(overrides)
    return tc.ServingArtifactManifest(**values)


def test_serving_artifact_manifest_hint_builds_runtime_policy() -> None:
    hint = ServingArtifactManifestHint(
        serving_manifest_ref="tensor:manifest",
        representation_contract_hash="repr-hash",
        serving_build_digest="build-digest",
        tensor_schema_hash="schema-hash",
        canonical_tensor_count=1,
    )

    policy = hint.to_runtime_policy()

    assert policy.require_manifest is True
    assert policy.serving_manifest_ref == "tensor:manifest"
    assert policy.expected_representation_contract_hash == "repr-hash"
    assert policy.expected_serving_build_digest == "build-digest"


def test_cross_check_serving_artifact_manifest_accepts_matching_contract() -> None:
    manifest = _manifest()

    assert (
        cross_check_serving_artifact_manifest(
            manifest=manifest,
            descriptor_tensor_schema_hash="schema-hash",
            tensor_names=("w",),
            expected_tensor_schema_hash="schema-hash",
        )
        == manifest
    )


def test_cross_check_serving_artifact_manifest_rejects_mismatch() -> None:
    with pytest.raises(RuntimeError, match="tensor schema hash mismatch"):
        cross_check_serving_artifact_manifest(
            manifest=_manifest(tensor_schema_hash="other"),
            descriptor_tensor_schema_hash="schema-hash",
            tensor_names=("w",),
            expected_tensor_schema_hash="schema-hash",
        )


def test_read_serving_artifact_manifest_tensor_reads_uint8_payload() -> None:
    manifest = _manifest()

    class _Artifact:
        def subset(self, names):
            assert names == [SERVING_MANIFEST_TENSOR_NAME]
            return self

        def tensor_dict(self, *, device: str):
            assert device == "cpu"
            return {
                SERVING_MANIFEST_TENSOR_NAME: torch.tensor(
                    list(manifest.to_bytes()), dtype=torch.uint8
                )
            }

    assert (
        read_serving_artifact_manifest_tensor(
            _Artifact(),
            artifact_ref="mi2:serving",
        )
        == manifest
    )


def test_cross_check_serving_artifact_manifest_enforces_runtime_policy() -> None:
    manifest = _manifest()
    policy = manifest.to_runtime_policy()

    assert (
        cross_check_serving_artifact_manifest(
            manifest=manifest,
            descriptor_tensor_schema_hash="schema-hash",
            tensor_names=("w",),
            expected_tensor_schema_hash="schema-hash",
            serving_runtime_policy=policy,
        )
        == manifest
    )

    with pytest.raises(RuntimeError, match="manifest ref mismatch"):
        cross_check_serving_artifact_manifest(
            manifest=manifest.model_copy(
                update={"serving_manifest_ref": "tensor:other_manifest"}
            ),
            descriptor_tensor_schema_hash="schema-hash",
            tensor_names=("w",),
            expected_tensor_schema_hash="schema-hash",
            serving_runtime_policy=policy,
        )
