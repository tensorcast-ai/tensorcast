#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import pytest
import torch

import tensorcast as tc
from tensorcast.serving.artifact_manifest import (
    SERVING_MANIFEST_TENSOR_NAME,
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


def test_serving_artifact_manifest_builds_runtime_policy() -> None:
    manifest = _manifest(
        serving_manifest_ref="tensor:manifest",
    )

    policy = manifest.to_runtime_policy()

    assert policy.require_manifest is True
    assert policy.serving_manifest_ref == "tensor:manifest"
    assert policy.expected_representation_contract_hash == "repr-hash"
    assert policy.expected_serving_build_digest == "build-digest"


def test_serving_artifact_manifest_policy_round_trips_topology_digest() -> None:
    policy = _manifest(
        topology_admission_digest="topology-digest",
    ).to_runtime_policy()

    assert policy.expected_topology_admission_digest == "topology-digest"
    assert tc.ServingRuntimePolicy.from_proto(policy.to_proto()) == policy


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


def test_read_serving_artifact_manifest_tensor_falls_back_to_cuda_payload(
    monkeypatch,
) -> None:
    manifest = _manifest()
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)

    class _Artifact:
        def subset(self, names):
            assert names == [SERVING_MANIFEST_TENSOR_NAME]
            return self

        def tensor_dict(self, *, device: str):
            if device == "cpu":
                raise RuntimeError("CPU replica is not loaded")
            assert device == torch.device("cuda", 2)
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


def test_read_serving_artifact_manifest_tensor_reports_cpu_and_cuda_errors(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)

    class _Artifact:
        def subset(self, names):
            assert names == [SERVING_MANIFEST_TENSOR_NAME]
            return self

        def tensor_dict(self, *, device: str):
            if device == "cpu":
                raise RuntimeError("CPU replica is not loaded")
            assert device == torch.device("cuda", 3)
            raise RuntimeError("CUDA replica is not loaded")

    with pytest.raises(RuntimeError) as exc_info:
        read_serving_artifact_manifest_tensor(
            _Artifact(),
            artifact_ref="mi2:serving",
        )

    message = str(exc_info.value)
    assert "CPU replica is not loaded" in message
    assert "CUDA replica is not loaded" in message


def test_cross_check_serving_artifact_manifest_enforces_runtime_policy() -> None:
    manifest = _manifest(topology_admission_digest="topology-digest")
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


@pytest.mark.parametrize(
    ("manifest_update", "policy_update", "match"),
    [
        (
            {"representation_contract_hash": "other-repr"},
            {},
            "representation contract mismatch",
        ),
        (
            {"serving_build_digest": "other-build"},
            {},
            "build digest mismatch",
        ),
        (
            {"topology_admission_digest": "other-topology"},
            {},
            "topology admission digest mismatch",
        ),
        (
            {},
            {"expected_topology_admission_digest": "other-topology"},
            "topology admission digest mismatch",
        ),
    ],
)
def test_cross_check_serving_artifact_manifest_rejects_pinned_policy_mismatch(
    manifest_update,
    policy_update,
    match,
) -> None:
    manifest = _manifest(topology_admission_digest="topology-digest")
    policy = manifest.to_runtime_policy().model_copy(update=policy_update)

    with pytest.raises(RuntimeError, match=match):
        cross_check_serving_artifact_manifest(
            manifest=manifest.model_copy(update=manifest_update),
            descriptor_tensor_schema_hash="schema-hash",
            tensor_names=("w",),
            expected_tensor_schema_hash="schema-hash",
            serving_runtime_policy=policy,
        )
