#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import logging
import time
from typing import Mapping, Sequence

from tensorcast.api._view_ops import (
    ResolvedViewInputs,
    SliceSpec,
    ViewSpecBuildResult,
    _coerce_slice_spec,
    build_view_spec,
)
from tensorcast.api.store.cache import ArtifactCacheEntry
from tensorcast.api.store.common import canonical_index_from_bytes
from tensorcast.api.store.runtime import StoreRuntimeContext
from tensorcast.api.store.types import ArtifactError, CanonicalIndex
from tensorcast.proto.daemon.v2 import store_daemon_pb2

logger = logging.getLogger(__name__)

TransformPlacement = store_daemon_pb2.TransformPlacement


class ViewOrchestrator:
    """Builds and resolves view specifications for Store pipelines."""

    def __init__(self, runtime: StoreRuntimeContext) -> None:
        self._runtime = runtime

    @staticmethod
    def _resolve_identifiers(
        artifact_id: str | None,
        key: str | None,
    ) -> tuple[str | None, str | None]:
        if artifact_id and key:
            raise ArtifactError(
                "Specify either artifact_id or key, not both",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if not artifact_id and not key:
            raise ArtifactError(
                "Either artifact_id or key is required",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        return artifact_id, key

    def _build_view_spec(
        self,
        *,
        canonical_index: CanonicalIndex,
        slices: Mapping[str, Sequence[object]] | None,
        transpose: Mapping[str, Sequence[tuple[int, int]]] | None,
    ) -> ViewSpecBuildResult:
        entry_shapes = {
            entry.name: tuple(entry.shape) for entry in canonical_index.entries
        }
        try:
            if slices is not None and not isinstance(slices, Mapping):
                raise ValueError("Slice spec must be a mapping of tensor name to slice")
            if transpose is not None and not isinstance(transpose, Mapping):
                raise ValueError(
                    "Transpose spec must be a mapping of tensor name to dim pairs"
                )
            typed_slices: Mapping[str, SliceSpec] | None = None
            if slices:
                typed_slices = {}
                for name, spec_seq in slices.items():
                    if not isinstance(spec_seq, Sequence) or not spec_seq:
                        raise ValueError(
                            f"Slice spec for '{name}' must be a non-empty sequence"
                        )
                    typed_slices[name] = _coerce_slice_spec(spec_seq)

            if transpose:
                for name, ops in transpose.items():
                    if not isinstance(ops, Sequence):
                        raise ValueError(
                            f"Transpose spec for '{name}' must be a non-empty sequence"
                        )
                    if not ops:
                        raise ValueError(
                            f"Transpose spec for '{name}' must be a non-empty sequence"
                        )

            return build_view_spec(
                entry_shapes=entry_shapes,
                slices=typed_slices,
                transpose=transpose,
            )
        except ValueError as exc:
            raise ArtifactError(
                str(exc),
                status_code="INVALID_ARGUMENT",
                retryable=False,
            ) from exc

    def resolve_view_inputs(
        self,
        *,
        artifact_id: str | None,
        key: str | None,
        slices: Mapping[str, Sequence[object]] | None,
        transpose: Mapping[str, Sequence[tuple[int, int]]] | None,
        view_id: str | None,
    ) -> ResolvedViewInputs:
        resolved_artifact_id, resolved_key = self._resolve_identifiers(artifact_id, key)
        cached_entry: ArtifactCacheEntry | None = None
        if resolved_artifact_id is None:
            assert resolved_key is not None
            resolved = self._runtime.resolve_key_mapping_cached(key=resolved_key)
            if isinstance(resolved, tuple):
                resolved_artifact_id = resolved[0]
            else:
                resolved_artifact_id = resolved
            if not resolved_artifact_id:
                raise ArtifactError(
                    f"Artifact key '{resolved_key}' is not mapped",
                    status_code="NOT_FOUND",
                    retryable=False,
                )
        if resolved_artifact_id:
            cached_entry = self._runtime.get_artifact_index_cached(resolved_artifact_id)

        if view_id is not None:
            if slices or transpose:
                raise ArtifactError(
                    "Provide either view_id or slices/transpose, not both",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            if view_id.strip() == "":
                raise ArtifactError(
                    "view_id must not be empty",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            return ResolvedViewInputs.from_view_id(
                artifact_id=resolved_artifact_id,
                view_id=view_id,
            )

        if not slices and not transpose:
            raise ArtifactError(
                "View retrieval requires slices/transpose or view_id",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )

        canonical_index_bytes = (
            cached_entry.canonical_index_bytes if cached_entry else None
        )
        canonical_index: CanonicalIndex | None = (
            cached_entry.parsed_index if cached_entry else None
        )
        if canonical_index_bytes is None:
            client = self._runtime.ensure_client()
            canonical_index_bytes = client.get_artifact_index_by_id(
                resolved_artifact_id
            )
        if canonical_index is None:
            canonical_index = canonical_index_from_bytes(canonical_index_bytes)
        if cached_entry is None:
            cache_entry = ArtifactCacheEntry(
                artifact_id=resolved_artifact_id,
                canonical_index_bytes=canonical_index_bytes,
                parsed_index=canonical_index,
                generation=None,
                expires_at=time.monotonic(),
            )
            self._runtime.cache_artifact_index(cache_entry)
        build_result = self._build_view_spec(
            canonical_index=canonical_index,
            slices=slices,
            transpose=transpose,
        )
        return ResolvedViewInputs.from_build_result(
            artifact_id=resolved_artifact_id,
            canonical_index_bytes=canonical_index_bytes,
            build_result=build_result,
        )

    @staticmethod
    def resolve_transform_placement(
        placement: str | None,
        *,
        has_transpose: bool,
        for_registration: bool = False,
    ) -> TransformPlacement:
        if placement is None:
            # Retrieval (get_view) always defaults to SERVER because client-side
            # transform engine doesn't exist yet (see tensorcast/api/README.md).
            # Registration defaults to CLIENT when transpose is present so the
            # client pre-applies the transform before upload.
            if for_registration and has_transpose:
                return TransformPlacement.TRANSFORM_PLACEMENT_CLIENT
            return TransformPlacement.TRANSFORM_PLACEMENT_SERVER
        normalized = placement.upper()
        if normalized == "SERVER":
            return TransformPlacement.TRANSFORM_PLACEMENT_SERVER
        if normalized == "CLIENT":
            return TransformPlacement.TRANSFORM_PLACEMENT_CLIENT
        raise ArtifactError(
            f"Unknown placement '{placement}', expected 'SERVER' or 'CLIENT'",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )


__all__ = ["TransformPlacement", "ViewOrchestrator"]
