#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import pytest
import torch

import tensorcast as tc
from tensorcast.artifact_runtime.artifact.manifest import (
    SERVING_MANIFEST_TENSOR_NAME,
    cross_check_runtime_artifact_manifest,
    read_runtime_artifact_manifest_tensor,
)


def _manifest(**overrides) -> tc.RuntimeArtifactManifest:
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
    return tc.RuntimeArtifactManifest(**values)


class _ManifestTensorResult:
    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
        releases: list[str],
        marker: object,
    ) -> None:
        self.tensors = tensors
        self._releases = releases
        self._marker = marker

    def release(self) -> None:
        self._releases.append(str(self._marker))


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
    assert tc.RuntimeArtifactPolicy.from_proto(policy.to_proto()) == policy


def test_cross_check_runtime_artifact_manifest_accepts_matching_contract() -> None:
    manifest = _manifest()

    assert (
        cross_check_runtime_artifact_manifest(
            manifest=manifest,
            descriptor_tensor_schema_hash="schema-hash",
            tensor_names=("w",),
            expected_tensor_schema_hash="schema-hash",
        )
        == manifest
    )


def test_cross_check_runtime_artifact_manifest_rejects_mismatch() -> None:
    with pytest.raises(RuntimeError, match="tensor schema hash mismatch"):
        cross_check_runtime_artifact_manifest(
            manifest=_manifest(tensor_schema_hash="other"),
            descriptor_tensor_schema_hash="schema-hash",
            tensor_names=("w",),
            expected_tensor_schema_hash="schema-hash",
        )


def test_read_runtime_artifact_manifest_tensor_reads_uint8_payload() -> None:
    manifest = _manifest()

    class _Artifact:
        def __init__(self) -> None:
            self.releases: list[str] = []

        def subset(self, names):
            assert names == [SERVING_MANIFEST_TENSOR_NAME]
            return self

        def tensor_dict_with_diagnostics(self, *, device: str):
            assert device == "cpu"
            return _ManifestTensorResult(
                {
                    SERVING_MANIFEST_TENSOR_NAME: torch.tensor(
                        list(manifest.to_bytes()), dtype=torch.uint8
                    )
                },
                self.releases,
                device,
            )

    artifact = _Artifact()
    assert (
        read_runtime_artifact_manifest_tensor(
            artifact,
            artifact_ref="mi2:serving",
        )
        == manifest
    )
    assert artifact.releases == ["cpu"]


def test_read_runtime_artifact_manifest_tensor_falls_back_to_cuda_payload(
    monkeypatch,
) -> None:
    manifest = _manifest()
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)

    class _Artifact:
        def __init__(self) -> None:
            self.releases: list[str] = []

        def subset(self, names):
            assert names == [SERVING_MANIFEST_TENSOR_NAME]
            return self

        def tensor_dict_with_diagnostics(self, *, device: str):
            if device == "cpu":
                raise RuntimeError("CPU replica is not loaded")
            assert device == torch.device("cuda", 2)
            return _ManifestTensorResult(
                {
                    SERVING_MANIFEST_TENSOR_NAME: torch.tensor(
                        list(manifest.to_bytes()), dtype=torch.uint8
                    )
                },
                self.releases,
                device,
            )

    artifact = _Artifact()
    assert (
        read_runtime_artifact_manifest_tensor(
            artifact,
            artifact_ref="mi2:serving",
        )
        == manifest
    )
    assert artifact.releases == ["cuda:2"]


def test_read_runtime_artifact_manifest_tensor_reports_cpu_and_cuda_errors(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)

    class _Artifact:
        def subset(self, names):
            assert names == [SERVING_MANIFEST_TENSOR_NAME]
            return self

        def tensor_dict_with_diagnostics(self, *, device: str):
            if device == "cpu":
                raise RuntimeError("CPU replica is not loaded")
            assert device == torch.device("cuda", 3)
            raise RuntimeError("CUDA replica is not loaded")

    with pytest.raises(RuntimeError) as exc_info:
        read_runtime_artifact_manifest_tensor(
            _Artifact(),
            artifact_ref="mi2:serving",
        )

    message = str(exc_info.value)
    assert "CPU replica is not loaded" in message
    assert "CUDA replica is not loaded" in message


def test_read_runtime_artifact_manifest_tensor_releases_invalid_payload() -> None:
    class _Artifact:
        def __init__(self) -> None:
            self.releases: list[str] = []

        def subset(self, names):
            assert names == [SERVING_MANIFEST_TENSOR_NAME]
            return self

        def tensor_dict_with_diagnostics(self, *, device: str):
            assert device == "cpu"
            return _ManifestTensorResult(
                {
                    SERVING_MANIFEST_TENSOR_NAME: torch.tensor(
                        [1.0],
                        dtype=torch.float32,
                    )
                },
                self.releases,
                device,
            )

    artifact = _Artifact()
    with pytest.raises(RuntimeError, match="1D torch.uint8"):
        read_runtime_artifact_manifest_tensor(
            artifact,
            artifact_ref="mi2:serving",
        )
    assert artifact.releases == ["cpu"]


def test_cross_check_runtime_artifact_manifest_enforces_runtime_policy() -> None:
    manifest = _manifest(topology_admission_digest="topology-digest")
    policy = manifest.to_runtime_policy()

    assert (
        cross_check_runtime_artifact_manifest(
            manifest=manifest,
            descriptor_tensor_schema_hash="schema-hash",
            tensor_names=("w",),
            expected_tensor_schema_hash="schema-hash",
            runtime_artifact_policy=policy,
        )
        == manifest
    )

    with pytest.raises(RuntimeError, match="manifest ref mismatch"):
        cross_check_runtime_artifact_manifest(
            manifest=manifest.model_copy(
                update={"serving_manifest_ref": "tensor:other_manifest"}
            ),
            descriptor_tensor_schema_hash="schema-hash",
            tensor_names=("w",),
            expected_tensor_schema_hash="schema-hash",
            runtime_artifact_policy=policy,
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
def test_cross_check_runtime_artifact_manifest_rejects_pinned_policy_mismatch(
    manifest_update,
    policy_update,
    match,
) -> None:
    manifest = _manifest(topology_admission_digest="topology-digest")
    policy = manifest.to_runtime_policy().model_copy(update=policy_update)

    with pytest.raises(RuntimeError, match=match):
        cross_check_runtime_artifact_manifest(
            manifest=manifest.model_copy(update=manifest_update),
            descriptor_tensor_schema_hash="schema-hash",
            tensor_names=("w",),
            expected_tensor_schema_hash="schema-hash",
            runtime_artifact_policy=policy,
        )
