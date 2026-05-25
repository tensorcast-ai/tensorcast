#  Copyright (c) 2026, TensorCast Team.
"""Runtime artifact manifest parse and validation helpers."""

from __future__ import annotations

from typing import Any

import torch

import tensorcast as tc

RUNTIME_ARTIFACT_SCHEMA_VERSION = int(
    tc.RuntimeArtifactManifest.model_fields["schema_version"].default
)
SERVING_MANIFEST_TENSOR_NAME = tc.SERVING_MANIFEST_TENSOR_NAME


class _InvalidRuntimeManifestTensor(RuntimeError):
    pass


def runtime_manifest_from_tensor_bytes(
    data: bytes | bytearray,
) -> tc.RuntimeArtifactManifest:
    return tc.RuntimeArtifactManifest.from_bytes(bytes(data))


def _runtime_manifest_bytes_from_device(
    subset: Any,
    *,
    device: torch.device | str,
    manifest_tensor_name: str,
) -> bytes:
    result = subset.tensor_dict_with_diagnostics(device=device)
    try:
        manifest_tensor = result.tensors[manifest_tensor_name]
        if manifest_tensor.dtype != torch.uint8 or manifest_tensor.dim() != 1:
            raise _InvalidRuntimeManifestTensor(
                "TensorCast runtime manifest tensor must be 1D torch.uint8"
            )
        return bytes(manifest_tensor.detach().cpu().tolist())
    finally:
        result.release()


def read_runtime_artifact_manifest_tensor(
    artifact: Any,
    *,
    artifact_ref: str,
    manifest_tensor_name: str = SERVING_MANIFEST_TENSOR_NAME,
) -> tc.RuntimeArtifactManifest:
    subset = artifact.subset([manifest_tensor_name])
    try:
        manifest_bytes = _runtime_manifest_bytes_from_device(
            subset,
            device="cpu",
            manifest_tensor_name=manifest_tensor_name,
        )
    except _InvalidRuntimeManifestTensor:
        raise
    except Exception as cpu_exc:
        try:
            cuda_device = torch.device("cuda", torch.cuda.current_device())
            manifest_bytes = _runtime_manifest_bytes_from_device(
                subset,
                device=cuda_device,
                manifest_tensor_name=manifest_tensor_name,
            )
        except _InvalidRuntimeManifestTensor:
            raise
        except Exception as cuda_exc:
            raise RuntimeError(
                f"Failed to materialize runtime manifest from '{artifact_ref}' "
                f"(cpu_error={cpu_exc!r}; cuda_error={cuda_exc!r})"
            ) from cuda_exc
    return runtime_manifest_from_tensor_bytes(manifest_bytes)


def cross_check_runtime_artifact_manifest(
    *,
    manifest: Any | None,
    descriptor_tensor_schema_hash: str,
    tensor_names: tuple[str, ...],
    expected_tensor_schema_hash: str,
    runtime_artifact_policy: tc.RuntimeArtifactPolicy | None = None,
    expected_schema_version: int = RUNTIME_ARTIFACT_SCHEMA_VERSION,
) -> Any:
    if manifest is None:
        raise RuntimeError("TensorCast runtime artifact manifest is missing")
    if manifest.schema_version != expected_schema_version:
        raise RuntimeError(
            "TensorCast runtime artifact schema version mismatch: "
            f"{manifest.schema_version} != {expected_schema_version}"
        )
    if manifest.artifact_kind != "serving":
        raise RuntimeError(
            "TensorCast runtime artifact has unsupported artifact_kind: "
            f"{manifest.artifact_kind}"
        )
    if (
        runtime_artifact_policy is not None
        and runtime_artifact_policy.serving_manifest_ref is not None
        and manifest.serving_manifest_ref
        != runtime_artifact_policy.serving_manifest_ref
    ):
        raise RuntimeError("TensorCast runtime artifact manifest ref mismatch")
    if (
        runtime_artifact_policy is not None
        and runtime_artifact_policy.expected_representation_contract_hash is not None
        and manifest.representation_contract_hash
        != runtime_artifact_policy.expected_representation_contract_hash
    ):
        raise RuntimeError(
            "TensorCast runtime artifact representation contract mismatch"
        )
    if (
        runtime_artifact_policy is not None
        and runtime_artifact_policy.expected_serving_build_digest is not None
        and manifest.serving_build_digest
        != runtime_artifact_policy.expected_serving_build_digest
    ):
        raise RuntimeError("TensorCast runtime artifact build digest mismatch")
    if (
        runtime_artifact_policy is not None
        and getattr(
            runtime_artifact_policy,
            "expected_topology_admission_digest",
            None,
        )
        is not None
        and getattr(manifest, "topology_admission_digest", None)
        != runtime_artifact_policy.expected_topology_admission_digest
    ):
        raise RuntimeError(
            "TensorCast runtime artifact topology admission digest mismatch"
        )
    if manifest.tensor_schema_hash != expected_tensor_schema_hash:
        raise RuntimeError(
            "TensorCast runtime artifact tensor schema hash mismatch: "
            f"manifest={manifest.tensor_schema_hash}, "
            f"expected={expected_tensor_schema_hash}"
        )
    if descriptor_tensor_schema_hash != expected_tensor_schema_hash:
        raise RuntimeError(
            "TensorCast runtime artifact descriptor schema hash mismatch: "
            f"descriptor={descriptor_tensor_schema_hash}, "
            f"expected={expected_tensor_schema_hash}"
        )
    if manifest.canonical_tensor_count != len(tensor_names):
        raise RuntimeError("TensorCast runtime artifact tensor count mismatch")
    return manifest


__all__ = [
    "RUNTIME_ARTIFACT_SCHEMA_VERSION",
    "SERVING_MANIFEST_TENSOR_NAME",
    "cross_check_runtime_artifact_manifest",
    "read_runtime_artifact_manifest_tensor",
    "runtime_manifest_from_tensor_bytes",
]
