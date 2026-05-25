#  Copyright (c) 2026, TensorCast Team.

"""Runtime artifact resolution facade for framework integrations."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import torch

import tensorcast as tc
import tensorcast.artifact_runtime.artifact.manifest as tc_artifact_manifest
import tensorcast.artifact_runtime.contract as tc_contract
import tensorcast.artifact_runtime.recipe.materialization as tc_core_materialization
from tensorcast.api.store.types import CanonicalIndexEntry

RuntimeArtifactManifest = tc.RuntimeArtifactManifest


def _default_open_artifact(artifact_ref: str) -> Any:
    store_api = importlib.import_module("tensorcast.api.store")
    open_fn = cast(Callable[[str], Any], store_api.artifact)
    return open_fn(artifact_ref)


@dataclass(frozen=True)
class ResolvedRuntimeArtifact:
    artifact: Any
    artifact_ref: str
    descriptor: Any
    manifest: Any | None
    tensor_names: tuple[str, ...]
    tensor_schema_hash: str


def is_reserved_runtime_tensor_name(name: str) -> bool:
    return name.startswith("__tensorcast_meta__.")


def contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    stride = [1] * len(shape)
    for idx in range(len(shape) - 2, -1, -1):
        stride[idx] = stride[idx + 1] * int(shape[idx + 1])
    return tuple(stride)


def model_tensor_names_from_descriptor(descriptor: Any) -> tuple[str, ...]:
    return tuple(
        str(name)
        for name in getattr(descriptor, "tensor_names", ())
        if not is_reserved_runtime_tensor_name(str(name))
    )


def canonical_index_from_descriptor(descriptor: Any) -> tc.CanonicalIndex:
    entries: list[CanonicalIndexEntry] = []
    metas = getattr(descriptor, "tensor_metas", {})
    cursor = 0
    for name in sorted(model_tensor_names_from_descriptor(descriptor)):
        meta = metas[name]
        shape = tuple(int(dim) for dim in meta.shape)
        dtype = getattr(meta, "dtype", torch.float32)
        if not isinstance(dtype, torch.dtype):
            dtype = tc_core_materialization.dtype_from_string(str(dtype))
        size_bytes = int(getattr(meta, "size_bytes", 0) or 0)
        if size_bytes <= 0:
            numel = 1
            for dim in shape:
                numel *= int(dim)
            size_bytes = int(numel * dtype.itemsize)
        entries.append(
            CanonicalIndexEntry(
                name=name,
                dtype=dtype,
                shape=shape,
                stride=tuple(
                    int(dim)
                    for dim in getattr(meta, "stride", contiguous_stride(shape))
                ),
                storage_offset=int(getattr(meta, "storage_offset", 0) or 0),
                segment_offset=cursor,
                size_bytes=size_bytes,
            )
        )
        cursor += size_bytes
    return tc.CanonicalIndex(
        entries=tuple(entries),
        total_size_bytes=int(getattr(descriptor, "total_bytes", cursor) or cursor),
        avbs_hash="",
    )


def compute_descriptor_tensor_schema_hash(
    descriptor: Any,
    *,
    manifest_tensor_name: str,
) -> str:
    return tc_contract.compute_canonical_runtime_tensor_schema_hash(
        canonical_index_from_descriptor(descriptor),
        manifest_tensor_name=manifest_tensor_name,
    )


def _prepared_summary_value(summary: Any, field_name: str) -> str | None:
    value = getattr(summary, field_name, None)
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _cross_check_prepared_manifest_summary(
    *,
    summary: Any,
    manifest: tc.RuntimeArtifactManifest,
) -> None:
    fields = (
        "serving_manifest_ref",
        "representation_contract_hash",
        "serving_build_digest",
        "tensor_schema_hash",
    )
    for field_name in fields:
        expected = _prepared_summary_value(summary, field_name)
        actual = _prepared_summary_value(manifest, field_name)
        if expected is not None and actual != expected:
            raise RuntimeError(
                "TensorCast prepared runtime artifact manifest does not match "
                f"summary field {field_name}: manifest={actual!r}, "
                f"summary={expected!r}"
            )


class RuntimeArtifactResolver:
    """Resolve runtime artifacts and enforce manifest/schema/policy checks."""

    def __init__(
        self,
        *,
        manifest_tensor_name: str,
        schema_version: int,
        open_artifact_fn: Callable[[str], Any] | None = None,
    ) -> None:
        self._manifest_tensor_name = manifest_tensor_name
        self._schema_version = schema_version
        self._open_artifact_fn = open_artifact_fn

    def open(self, artifact_ref: str) -> Any:
        open_fn = self._open_artifact_fn or _default_open_artifact
        artifact = open_fn(str(artifact_ref))
        artifact.describe()
        return artifact

    def compute_descriptor_tensor_schema_hash(self, descriptor: Any) -> str:
        return compute_descriptor_tensor_schema_hash(
            descriptor,
            manifest_tensor_name=self._manifest_tensor_name,
        )

    def read_manifest(
        self,
        artifact: Any,
        *,
        artifact_ref: str,
    ) -> ResolvedRuntimeArtifact:
        descriptor = artifact.describe()
        tensor_names = model_tensor_names_from_descriptor(descriptor)
        tensor_schema_hash = self.compute_descriptor_tensor_schema_hash(descriptor)
        if self._manifest_tensor_name not in getattr(descriptor, "tensor_names", ()):
            raise RuntimeError(
                f"TensorCast artifact '{artifact_ref}' is missing runtime "
                "manifest tensor"
            )
        manifest = tc_artifact_manifest.read_runtime_artifact_manifest_tensor(
            artifact,
            artifact_ref=artifact_ref,
            manifest_tensor_name=self._manifest_tensor_name,
        )
        return ResolvedRuntimeArtifact(
            artifact=artifact,
            artifact_ref=str(artifact_ref),
            descriptor=descriptor,
            manifest=manifest,
            tensor_names=tensor_names,
            tensor_schema_hash=tensor_schema_hash,
        )

    def resolve(self, artifact_ref: str) -> ResolvedRuntimeArtifact:
        return self.read_manifest(
            self.open(artifact_ref),
            artifact_ref=artifact_ref,
        )

    def resolve_prepared(
        self,
        summary: Any,
    ) -> ResolvedRuntimeArtifact:
        artifact_ref = getattr(summary, "serving_artifact_ref", None)
        if artifact_ref is None:
            raise RuntimeError(
                "TensorCast local-ready summary does not reference a durable "
                "runtime artifact"
            )
        artifact_ref = str(artifact_ref)
        artifact = self.open(artifact_ref)
        descriptor = artifact.describe()
        tensor_names = model_tensor_names_from_descriptor(descriptor)
        tensor_schema_hash = self.compute_descriptor_tensor_schema_hash(descriptor)
        if self._manifest_tensor_name not in getattr(descriptor, "tensor_names", ()):
            raise RuntimeError(
                f"TensorCast artifact '{artifact_ref}' is missing runtime "
                "manifest tensor"
            )
        manifest = tc_artifact_manifest.read_runtime_artifact_manifest_tensor(
            artifact,
            artifact_ref=artifact_ref,
            manifest_tensor_name=self._manifest_tensor_name,
        )
        _cross_check_prepared_manifest_summary(summary=summary, manifest=manifest)
        tc_artifact_manifest.cross_check_runtime_artifact_manifest(
            manifest=manifest,
            descriptor_tensor_schema_hash=tensor_schema_hash,
            tensor_names=tensor_names,
            expected_tensor_schema_hash=str(summary.tensor_schema_hash),
            expected_schema_version=self._schema_version,
        )
        return ResolvedRuntimeArtifact(
            artifact=artifact,
            artifact_ref=artifact_ref,
            descriptor=descriptor,
            manifest=manifest,
            tensor_names=tensor_names,
            tensor_schema_hash=tensor_schema_hash,
        )

    def cross_check(
        self,
        resolved: ResolvedRuntimeArtifact,
        *,
        expected_tensor_schema_hash: str,
        runtime_artifact_policy: Any | None = None,
    ) -> ResolvedRuntimeArtifact:
        tc_artifact_manifest.cross_check_runtime_artifact_manifest(
            manifest=resolved.manifest,
            descriptor_tensor_schema_hash=resolved.tensor_schema_hash,
            tensor_names=resolved.tensor_names,
            expected_tensor_schema_hash=expected_tensor_schema_hash,
            runtime_artifact_policy=runtime_artifact_policy,
            expected_schema_version=self._schema_version,
        )
        return resolved


def resolve_runtime_artifact(
    artifact_ref: str,
    *,
    manifest_tensor_name: str | None = None,
    schema_version: int | None = None,
    expected_tensor_schema_hash: str | None = None,
    runtime_artifact_policy: Any | None = None,
) -> ResolvedRuntimeArtifact:
    resolver = RuntimeArtifactResolver(
        manifest_tensor_name=manifest_tensor_name or tc.SERVING_MANIFEST_TENSOR_NAME,
        schema_version=(
            schema_version
            if schema_version is not None
            else int(tc.RuntimeArtifactManifest.model_fields["schema_version"].default)
        ),
    )
    resolved = resolver.resolve(artifact_ref)
    if expected_tensor_schema_hash is not None:
        resolver.cross_check(
            resolved,
            expected_tensor_schema_hash=expected_tensor_schema_hash,
            runtime_artifact_policy=runtime_artifact_policy,
        )
    return resolved


__all__ = [
    "ResolvedRuntimeArtifact",
    "RuntimeArtifactManifest",
    "RuntimeArtifactResolver",
    "canonical_index_from_descriptor",
    "compute_descriptor_tensor_schema_hash",
    "contiguous_stride",
    "is_reserved_runtime_tensor_name",
    "model_tensor_names_from_descriptor",
    "resolve_runtime_artifact",
]
