#  Copyright (c) 2025, TensorCast Team.

# Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import hashlib
import json
import weakref
from dataclasses import dataclass
from typing import Mapping, Sequence

from tensorcast.api._view_ops import (
    NarrowOp,
    TensorViewOp,
    TransposeOp,
    ViewSpecBuildResult,
    build_view_spec,
)
from tensorcast.api.store.types import (
    ArtifactError,
    CanonicalIndex,
    CanonicalIndexEntry,
)


@dataclass(frozen=True, slots=True)
class ViewMetadataCache:
    view_id: str
    view_index_bytes: bytes
    view_data_hash: str
    tensor_names: tuple[str, ...]
    nbytes: int
    canonical_index: CanonicalIndex


def _serialize_index(index: CanonicalIndex) -> bytes:
    entries: dict[str, list[object]] = {}
    for entry in index.entries:
        entries[entry.name] = [
            int(entry.segment_offset),
            int(entry.size_bytes),
            list(entry.shape),
            list(entry.stride),
            str(entry.dtype),
            int(entry.storage_offset),
        ]
    return json.dumps(entries, separators=(",", ":")).encode("utf-8")


def _compose_narrow(
    *,
    base_shape: tuple[int, ...],
    parent: NarrowOp | None,
    child: NarrowOp | None,
    tensor_name: str,
) -> list[TensorViewOp]:
    ops: list[TensorViewOp] = []
    shape = tuple(int(dim) for dim in base_shape)

    def _validate(op: NarrowOp) -> None:
        if op.dim < 0 or op.dim >= len(shape):
            raise ArtifactError(
                f"Slice dim {op.dim} out of range for tensor '{tensor_name}'",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if op.start < 0 or op.length <= 0:
            raise ArtifactError(
                f"Slice for '{tensor_name}' must have positive start/length",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if op.start + op.length > shape[op.dim]:
            raise ArtifactError(
                f"Slice [{op.start}, {op.start + op.length}) exceeds dim {op.dim} for '{tensor_name}'",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )

    if parent is not None:
        _validate(parent)
    if child is None:
        if parent is None:
            return ops
        ops.append(parent)
        return ops

    _validate(child)
    if parent is None:
        if child.start == 0 and child.length == shape[child.dim]:
            return []
        ops.append(child)
        return ops

    if child.dim != parent.dim:
        raise ArtifactError(
            f"Multiple slice dimensions for tensor '{tensor_name}' are not supported",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )

    narrowed_length = parent.length
    if child.start + child.length > narrowed_length:
        raise ArtifactError(
            (
                f"Slice [{child.start}, {child.start + child.length}) exceeds "
                f"narrowed length {narrowed_length} for '{tensor_name}'"
            ),
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    composed_start = parent.start + child.start
    ops.append(NarrowOp(dim=parent.dim, start=composed_start, length=child.length))
    return ops


def _permutation_from_ops(ndim: int, ops: Sequence[TransposeOp]) -> list[int]:
    perm = list(range(ndim))
    for op in ops:
        if op.dim0 < 0 or op.dim1 < 0 or op.dim0 >= ndim or op.dim1 >= ndim:
            raise ArtifactError(
                "Transpose dims out of range",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        perm[op.dim0], perm[op.dim1] = perm[op.dim1], perm[op.dim0]
    return perm


def _ops_from_permutation(perm: list[int]) -> list[TensorViewOp]:
    ops: list[TensorViewOp] = []
    current = list(range(len(perm)))
    for target_idx, desired in enumerate(perm):
        if current[target_idx] == desired:
            continue
        try:
            source_idx = current.index(desired)
        except ValueError:
            continue
        if target_idx == source_idx:
            continue
        ops.append(TransposeOp(dim0=target_idx, dim1=source_idx))
        current[target_idx], current[source_idx] = (
            current[source_idx],
            current[target_idx],
        )
    return ops


def _apply_view_ops(
    entry: CanonicalIndexEntry, ops: Sequence[TensorViewOp]
) -> CanonicalIndexEntry:
    shape = list(entry.shape)
    stride = list(entry.stride)
    storage_offset = int(entry.storage_offset)

    for op in ops:
        if isinstance(op, NarrowOp):
            if op.dim < 0 or op.dim >= len(shape):
                raise ArtifactError(
                    f"Slice dim {op.dim} out of range for tensor '{entry.name}'",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            storage_offset += op.start * stride[op.dim]
            shape[op.dim] = op.length
        elif isinstance(op, TransposeOp):
            dim0, dim1 = op.dim0, op.dim1
            if dim0 < 0 or dim1 < 0 or dim0 >= len(shape) or dim1 >= len(shape):
                raise ArtifactError(
                    f"Transpose dims {dim0},{dim1} out of range for tensor '{entry.name}'",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
            stride[dim0], stride[dim1] = stride[dim1], stride[dim0]

    base_numel = 1
    for dim in entry.shape:
        base_numel *= dim
    base_numel = max(base_numel, 1)
    elem_size = float(entry.size_bytes) / float(base_numel)
    new_numel = 1
    for dim in shape:
        new_numel *= dim
    size_bytes = int(elem_size * float(new_numel)) if new_numel > 0 else 0

    return CanonicalIndexEntry(
        name=entry.name,
        dtype=entry.dtype,
        shape=tuple(int(v) for v in shape),
        stride=tuple(int(v) for v in stride),
        storage_offset=storage_offset,
        segment_offset=entry.segment_offset,
        size_bytes=size_bytes,
    )


class ViewSpecComposer:
    """Pure view composer that flattens parent/child specs without RPC calls."""

    def __init__(self, *, max_depth: int = 8) -> None:
        self._max_depth = max(1, int(max_depth))

    @staticmethod
    def hash_view_spec(
        view_spec: ViewSpecBuildResult | None, *, subset: Sequence[str] | None = None
    ) -> str:
        normalized_ops = view_spec.to_normalized_dict() if view_spec is not None else {}
        payload = {
            "ops": normalized_ops,
            "subset": sorted(subset) if subset else [],
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8"))
        return digest.hexdigest()

    def compose(
        self,
        *,
        canonical_index: CanonicalIndex,
        parent_spec: ViewSpecBuildResult | None,
        child_spec: ViewSpecBuildResult | None,
        parent_depth: int,
        subset_names: Sequence[str] | None = None,
    ) -> tuple[ViewSpecBuildResult | None, ViewMetadataCache | None, int]:
        depth = parent_depth
        base_spec = parent_spec if parent_spec and not parent_spec.is_identity else None
        child = child_spec if child_spec and not child_spec.is_identity else None
        composed_spec: ViewSpecBuildResult | None = None
        if base_spec is None and child is None:
            composed_spec = None
        elif base_spec is None:
            composed_spec = child
            depth += 1 if child is not None else 0
        elif child is None:
            composed_spec = base_spec
        else:
            composed_spec = self._compose_specs(
                canonical_index=canonical_index,
                parent=base_spec,
                child=child,
            )
            depth += 1

        if depth > self._max_depth:
            raise ArtifactError(
                f"View depth {depth} exceeds limit {self._max_depth}",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )

        view_index: CanonicalIndex | None = None
        tensor_names: tuple[str, ...] | None = None
        if composed_spec is not None or subset_names:
            view_index, tensor_names = self._build_view_index(
                canonical_index=canonical_index,
                view_spec=composed_spec,
                subset_names=subset_names,
            )

        view_cache: ViewMetadataCache | None = None
        if view_index is not None and tensor_names is not None:
            view_hash = self.hash_view_spec(composed_spec, subset=tensor_names)
            view_cache = ViewMetadataCache(
                view_id=view_hash,
                view_index_bytes=_serialize_index(view_index),
                view_data_hash=view_hash,
                tensor_names=tensor_names,
                nbytes=sum(entry.size_bytes for entry in view_index.entries),
                canonical_index=view_index,
            )

        return composed_spec, view_cache, depth

    def _compose_specs(
        self,
        *,
        canonical_index: CanonicalIndex,
        parent: ViewSpecBuildResult,
        child: ViewSpecBuildResult,
    ) -> ViewSpecBuildResult:
        tensor_ops: dict[str, list[TensorViewOp]] = {}
        entry_shapes = {
            entry.name: tuple(entry.shape) for entry in canonical_index.entries
        }
        tensor_names = set(parent.tensor_ops.keys()) | set(child.tensor_ops.keys())

        for name in sorted(tensor_names):
            shape = entry_shapes.get(name)
            if shape is None:
                raise ArtifactError(
                    f"View references unknown tensor '{name}'",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            parent_ops = list(parent.tensor_ops.get(name, ()))
            child_ops = list(child.tensor_ops.get(name, ()))
            has_parent_narrow = any(isinstance(op, NarrowOp) for op in parent_ops)
            has_child_narrow = any(isinstance(op, NarrowOp) for op in child_ops)
            has_parent_transpose = any(isinstance(op, TransposeOp) for op in parent_ops)
            has_child_transpose = any(isinstance(op, TransposeOp) for op in child_ops)

            if (has_parent_narrow or has_child_narrow) and (
                has_parent_transpose or has_child_transpose
            ):
                raise ArtifactError(
                    f"Cannot mix slice and transpose for tensor '{name}'",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )

            if has_parent_narrow or has_child_narrow:
                parent_narrow = next(
                    (op for op in parent_ops if isinstance(op, NarrowOp)), None
                )
                child_narrow = next(
                    (op for op in child_ops if isinstance(op, NarrowOp)), None
                )
                composed_narrows = _compose_narrow(
                    base_shape=shape,
                    parent=parent_narrow,
                    child=child_narrow,
                    tensor_name=name,
                )
                if composed_narrows:
                    tensor_ops[name] = composed_narrows
                continue

            if has_parent_transpose or has_child_transpose:
                parent_perm = _permutation_from_ops(
                    len(shape),
                    [op for op in parent_ops if isinstance(op, TransposeOp)],
                )
                child_perm = _permutation_from_ops(
                    len(shape),
                    [op for op in child_ops if isinstance(op, TransposeOp)],
                )
                # Apply child permutation on top of parent result.
                composed_perm = [parent_perm[idx] for idx in child_perm]
                composed_ops = _ops_from_permutation(composed_perm)
                if composed_ops:
                    tensor_ops[name] = composed_ops
                continue

        if not tensor_ops:
            return ViewSpecBuildResult.identity()

        view_spec_proto = build_view_spec(
            entry_shapes=entry_shapes,
            slices={
                name: (op.dim, slice(op.start, op.start + op.length))
                for name, ops in tensor_ops.items()
                for op in ops
                if isinstance(op, NarrowOp)
            },
            transpose={
                name: [(op.dim0, op.dim1) for op in ops if isinstance(op, TransposeOp)]
                for name, ops in tensor_ops.items()
                if any(isinstance(op, TransposeOp) for op in ops)
            },
        ).proto

        ordered_ops: dict[str, tuple[TensorViewOp, ...]] = {
            name: tuple(ops) for name, ops in tensor_ops.items()
        }
        return ViewSpecBuildResult(proto=view_spec_proto, tensor_ops=ordered_ops)

    def _build_view_index(
        self,
        *,
        canonical_index: CanonicalIndex,
        view_spec: ViewSpecBuildResult | None,
        subset_names: Sequence[str] | None,
    ) -> tuple[CanonicalIndex, tuple[str, ...]]:
        base_entries = {entry.name: entry for entry in canonical_index.entries}
        names = (
            tuple(subset_names)
            if subset_names is not None
            else tuple(entry.name for entry in canonical_index.entries)
        )
        unknown = [name for name in names if name not in base_entries]
        if unknown:
            raise ArtifactError(
                f"View references unknown tensor(s): {', '.join(sorted(unknown))}",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )

        view_entries: list[CanonicalIndexEntry] = []
        for name in names:
            entry = base_entries[name]
            ops: Sequence[TensorViewOp] = ()
            if view_spec is not None:
                ops = view_spec.tensor_ops.get(name, ())
            view_entries.append(_apply_view_ops(entry, ops))

        total = sum(entry.size_bytes for entry in view_entries)
        view_index = CanonicalIndex(
            entries=tuple(view_entries),
            total_size_bytes=total,
            avbs_hash=canonical_index.avbs_hash,
        )
        return view_index, tuple(names)


class ViewBuilder:
    """Fluent builder for composing multiple view operations before construction."""

    def __init__(
        self,
        *,
        artifact_ref: weakref.ReferenceType,
        composer: ViewSpecComposer,
    ) -> None:
        self._artifact_ref = artifact_ref
        self._composer = composer
        self._slices: dict[str, Sequence[object]] = {}
        self._transpose: dict[str, Sequence[tuple[int, int]]] = {}
        self._subset: list[str] | None = None

    def slice(self, tensor_slices: Mapping[str, Sequence[object]]) -> "ViewBuilder":
        self._slices.update(tensor_slices)
        return self

    def transpose(
        self, tensor_dims: Mapping[str, Sequence[tuple[int, int]]]
    ) -> "ViewBuilder":
        self._transpose.update(tensor_dims)
        return self

    def select(self, names: Sequence[str]) -> "ViewBuilder":
        self._subset = list(names)
        return self

    def build(self):
        artifact = self._artifact_ref()
        if artifact is None:
            raise ArtifactError(
                "Artifact no longer available",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        return artifact._derive_view(
            slices=self._slices or None,
            transpose=self._transpose or None,
            subset=self._subset,
            composer=self._composer,
        )


__all__ = ["ViewBuilder", "ViewMetadataCache", "ViewSpecComposer"]
