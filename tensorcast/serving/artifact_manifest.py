#  Copyright (c) 2026, TensorCast Team.

"""Serving artifact manifest parse and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

import tensorcast as tc

SERVING_ARTIFACT_SCHEMA_VERSION = int(
    tc.ServingArtifactManifest.model_fields["schema_version"].default
)
SERVING_MANIFEST_TENSOR_NAME = tc.SERVING_MANIFEST_TENSOR_NAME


@dataclass(frozen=True)
class ServingArtifactManifestHint:
    serving_manifest_ref: str
    representation_contract_hash: str
    serving_build_digest: str
    tensor_schema_hash: str
    canonical_tensor_count: int
    local_serving_ref: str | None = None
    schema_version: int = SERVING_ARTIFACT_SCHEMA_VERSION
    artifact_kind: str = "serving"

    def to_runtime_policy(self) -> tc.ServingRuntimePolicy:
        return tc.ServingRuntimePolicy(
            require_manifest=True,
            serving_manifest_ref=self.serving_manifest_ref,
            expected_representation_contract_hash=(self.representation_contract_hash),
            expected_serving_build_digest=self.serving_build_digest,
        )


def serving_manifest_from_tensor_bytes(
    data: bytes | bytearray,
) -> tc.ServingArtifactManifest:
    return tc.ServingArtifactManifest.from_bytes(bytes(data))


def read_serving_artifact_manifest_tensor(
    artifact: Any,
    *,
    artifact_ref: str,
    manifest_tensor_name: str = SERVING_MANIFEST_TENSOR_NAME,
) -> tc.ServingArtifactManifest:
    try:
        manifest_tensor = artifact.subset([manifest_tensor_name]).tensor_dict(
            device="cpu"
        )[manifest_tensor_name]
    except Exception as exc:
        raise RuntimeError(
            f"Failed to materialize serving manifest from '{artifact_ref}'"
        ) from exc
    if manifest_tensor.dtype != torch.uint8 or manifest_tensor.dim() != 1:
        raise RuntimeError("TensorCast serving manifest tensor must be 1D torch.uint8")
    return serving_manifest_from_tensor_bytes(
        bytes(manifest_tensor.detach().cpu().tolist())
    )


def cross_check_serving_artifact_manifest(
    *,
    manifest: Any | None,
    descriptor_tensor_schema_hash: str,
    tensor_names: tuple[str, ...],
    expected_tensor_schema_hash: str,
    serving_runtime_policy: tc.ServingRuntimePolicy | None = None,
    expected_schema_version: int = SERVING_ARTIFACT_SCHEMA_VERSION,
) -> Any:
    if manifest is None:
        raise RuntimeError("TensorCast serving artifact manifest is missing")
    if manifest.schema_version != expected_schema_version:
        raise RuntimeError(
            "TensorCast serving artifact schema version mismatch: "
            f"{manifest.schema_version} != {expected_schema_version}"
        )
    if manifest.artifact_kind != "serving":
        raise RuntimeError(
            f"TensorCast artifact is not a serving artifact: {manifest.artifact_kind}"
        )
    if (
        serving_runtime_policy is not None
        and serving_runtime_policy.serving_manifest_ref is not None
        and manifest.serving_manifest_ref != serving_runtime_policy.serving_manifest_ref
    ):
        raise RuntimeError("TensorCast serving artifact manifest ref mismatch")
    if (
        serving_runtime_policy is not None
        and serving_runtime_policy.expected_representation_contract_hash is not None
        and manifest.representation_contract_hash
        != serving_runtime_policy.expected_representation_contract_hash
    ):
        raise RuntimeError(
            "TensorCast serving artifact representation contract mismatch"
        )
    if (
        serving_runtime_policy is not None
        and serving_runtime_policy.expected_serving_build_digest is not None
        and manifest.serving_build_digest
        != serving_runtime_policy.expected_serving_build_digest
    ):
        raise RuntimeError("TensorCast serving artifact build digest mismatch")
    if manifest.tensor_schema_hash != expected_tensor_schema_hash:
        raise RuntimeError(
            "TensorCast serving artifact tensor schema hash mismatch: "
            f"manifest={manifest.tensor_schema_hash}, "
            f"expected={expected_tensor_schema_hash}"
        )
    if descriptor_tensor_schema_hash != expected_tensor_schema_hash:
        raise RuntimeError(
            "TensorCast serving artifact descriptor schema hash mismatch: "
            f"descriptor={descriptor_tensor_schema_hash}, "
            f"expected={expected_tensor_schema_hash}"
        )
    if manifest.canonical_tensor_count != len(tensor_names):
        raise RuntimeError("TensorCast serving artifact tensor count mismatch")
    return manifest


__all__ = [
    "SERVING_ARTIFACT_SCHEMA_VERSION",
    "SERVING_MANIFEST_TENSOR_NAME",
    "ServingArtifactManifestHint",
    "cross_check_serving_artifact_manifest",
    "read_serving_artifact_manifest_tensor",
    "serving_manifest_from_tensor_bytes",
]
