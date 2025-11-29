#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping, Sequence, Union

from tensorcast.proto.daemon.v1 import store_daemon_pb2

logger = logging.getLogger(__name__)

SliceSpec = slice | tuple[int, slice]
TensorViewOpsMapping = Mapping[str, Sequence["TensorViewOp"]]


@dataclass(frozen=True, slots=True)
class NarrowOp:
    """Single-dimension slice operation (torch.narrow semantics)."""

    dim: int
    start: int
    length: int


@dataclass(frozen=True, slots=True)
class TransposeOp:
    """Single dimension-pair swap (torch.transpose semantics)."""

    dim0: int
    dim1: int


TensorViewOp = Union[NarrowOp, TransposeOp]


@dataclass(frozen=True, slots=True)
class ViewSpecBuildResult:
    """Structured result from view spec construction."""

    proto: store_daemon_pb2.ViewSpec | None
    tensor_ops: Mapping[str, Sequence[TensorViewOp]]

    def __post_init__(self) -> None:
        if (self.proto is None) != (not self.tensor_ops):
            raise ValueError(
                "ViewSpecBuildResult must keep proto and tensor_ops in sync"
            )

    @property
    def has_transpose(self) -> bool:
        return any(
            isinstance(op, TransposeOp)
            for ops in self.tensor_ops.values()
            for op in ops
        )

    @property
    def is_identity(self) -> bool:
        return not self.tensor_ops

    def __repr__(self) -> str:
        op_count = sum(len(ops) for ops in self.tensor_ops.values())
        return (
            f"ViewSpecBuildResult(tensors={len(self.tensor_ops)}, "
            f"ops={op_count}, is_identity={self.is_identity})"
        )

    def to_normalized_dict(self) -> dict[str, list[dict[str, int | str]]]:
        result: dict[str, list[dict[str, int | str]]] = {}
        for name in sorted(self.tensor_ops.keys()):
            op_list: list[dict[str, int | str]] = []
            for op in self.tensor_ops[name]:
                if isinstance(op, NarrowOp):
                    op_list.append(
                        {
                            "type": "narrow",
                            "dim": op.dim,
                            "start": op.start,
                            "length": op.length,
                        }
                    )
                elif isinstance(op, TransposeOp):
                    op_list.append(
                        {
                            "type": "transpose",
                            "dim0": op.dim0,
                            "dim1": op.dim1,
                        }
                    )
            if op_list:
                result[name] = op_list
        return result

    @staticmethod
    def identity() -> ViewSpecBuildResult:
        return _get_identity_result()


@dataclass(frozen=True, slots=True)
class ResolvedViewInputs:
    """Structured result from view input resolution."""

    artifact_id: str
    canonical_index_bytes: bytes | None
    build_result: ViewSpecBuildResult | None
    disk_path_hint: str | None
    view_id: str | None

    def __post_init__(self) -> None:
        has_build = self.build_result is not None
        has_view_id = self.view_id is not None
        if has_build == has_view_id:
            raise ValueError(
                "ResolvedViewInputs requires exactly one of build_result or view_id"
            )

    @property
    def variant(self) -> Literal["build", "id"]:
        return "build" if self.build_result is not None else "id"

    @property
    def view_spec(self) -> store_daemon_pb2.ViewSpec | None:
        return self.build_result.proto if self.build_result else None

    @property
    def has_transpose(self) -> bool:
        return self.build_result.has_transpose if self.build_result else False

    @property
    def normalized_ops(self) -> dict[str, list[dict[str, int | str]]]:
        if self.build_result is None:
            return {}
        return self.build_result.to_normalized_dict()

    @classmethod
    def from_build_result(
        cls,
        *,
        artifact_id: str,
        canonical_index_bytes: bytes,
        build_result: ViewSpecBuildResult,
        disk_path_hint: str | None = None,
    ) -> "ResolvedViewInputs":
        return cls(
            artifact_id=artifact_id,
            canonical_index_bytes=canonical_index_bytes,
            build_result=build_result,
            disk_path_hint=disk_path_hint,
            view_id=None,
        )

    @classmethod
    def from_view_id(
        cls,
        *,
        artifact_id: str,
        view_id: str,
        disk_path_hint: str | None = None,
    ) -> "ResolvedViewInputs":
        return cls(
            artifact_id=artifact_id,
            canonical_index_bytes=None,
            build_result=None,
            disk_path_hint=disk_path_hint,
            view_id=view_id,
        )


_IDENTITY_RESULT: ViewSpecBuildResult | None = None


def _get_identity_result() -> ViewSpecBuildResult:
    global _IDENTITY_RESULT
    if _IDENTITY_RESULT is None:
        _IDENTITY_RESULT = ViewSpecBuildResult(
            proto=None, tensor_ops=MappingProxyType({})
        )
    return _IDENTITY_RESULT


def _coerce_slice_spec(seq: Sequence[object]) -> SliceSpec:
    if len(seq) != 1:
        raise ValueError(
            f"Only a single narrow operation is supported per tensor (got {len(seq)})"
        )
    item = seq[0]
    if isinstance(item, slice):
        return item
    if isinstance(item, tuple) and len(item) == 2:
        dim, slc = item
        if isinstance(dim, int) and isinstance(slc, slice):
            return (dim, slc)
    raise ValueError("Slice spec must be a slice object or (dim, slice) tuple")


def validate_narrow(
    tensor_name: str,
    shape: tuple[int, ...],
    spec: SliceSpec,
) -> NarrowOp | None:
    if isinstance(spec, slice):
        dim = 0
        slice_obj = spec
    elif isinstance(spec, tuple) and len(spec) == 2:
        dim, slice_obj = spec
        if not isinstance(dim, int) or not isinstance(slice_obj, slice):
            raise ValueError("Slice spec must be a slice object or (dim, slice) tuple")
    else:
        raise ValueError("Slice spec must be a slice object or (dim, slice) tuple")

    ndims = len(shape)
    if dim < 0 or dim >= ndims:
        raise ValueError(
            f"Slice dim {dim} out of range for tensor '{tensor_name}' (ndim={ndims})"
        )

    step = 1 if slice_obj.step is None else slice_obj.step
    if step != 1:
        raise ValueError(f"Slice step must be 1 for tensor '{tensor_name}'")

    dim_extent = shape[dim]
    start = 0 if slice_obj.start is None else int(slice_obj.start)
    stop = dim_extent if slice_obj.stop is None else int(slice_obj.stop)

    if start < 0:
        start += dim_extent
    if stop < 0:
        stop += dim_extent

    if start < 0 or start >= dim_extent:
        raise ValueError(f"Slice start {start} out of range for tensor '{tensor_name}'")
    if stop <= start:
        raise ValueError(
            f"Slice stop {stop} must be greater than start {start} for tensor '{tensor_name}'"
        )

    length = stop - start
    if length > dim_extent:
        raise ValueError(
            f"Slice length {length} exceeds dimension {dim_extent} for tensor '{tensor_name}'"
        )
    if start + length > dim_extent:
        raise ValueError(
            f"Slice range [{start}, {stop}) exceeds dimension {dim_extent} for tensor '{tensor_name}'"
        )
    if start == 0 and length == dim_extent:
        return None

    return NarrowOp(dim=int(dim), start=int(start), length=int(length))


def validate_transpose(
    tensor_name: str,
    shape: tuple[int, ...],
    ops: Sequence[tuple[int, int]],
) -> list[TransposeOp]:
    ndims = len(shape)
    if not ops:
        raise ValueError(
            f"Transpose spec for '{tensor_name}' must be a non-empty sequence"
        )

    permutation = list(range(ndims))
    for dim_pair in ops:
        if not isinstance(dim_pair, Sequence) or len(dim_pair) != 2:
            raise ValueError(
                f"Transpose spec for '{tensor_name}' must contain dim pairs"
            )
        dim0, dim1 = dim_pair
        if not isinstance(dim0, int) or not isinstance(dim1, int):
            raise ValueError(
                f"Transpose dims must be integers for tensor '{tensor_name}'"
            )
        a = int(dim0)
        b = int(dim1)
        if a == b:
            continue
        if a < 0 or a >= ndims or b < 0 or b >= ndims:
            raise ValueError(
                f"Transpose dims {a},{b} out of range for tensor '{tensor_name}'"
            )
        permutation[a], permutation[b] = permutation[b], permutation[a]

    if permutation == list(range(ndims)):
        return []

    canonical_ops: list[TransposeOp] = []
    current = list(range(ndims))
    for target_idx in range(ndims):
        desired = permutation[target_idx]
        if current[target_idx] == desired:
            continue
        source_idx = current.index(desired)
        if target_idx == source_idx:
            continue
        canonical_ops.append(TransposeOp(dim0=int(target_idx), dim1=int(source_idx)))
        current[target_idx], current[source_idx] = (
            current[source_idx],
            current[target_idx],
        )
    return canonical_ops


def build_view_spec(
    *,
    entry_shapes: Mapping[str, tuple[int, ...]],
    slices: Mapping[str, SliceSpec] | None,
    transpose: Mapping[str, Sequence[tuple[int, int]]] | None,
) -> ViewSpecBuildResult:
    if not slices and not transpose:
        return ViewSpecBuildResult.identity()

    entry_shapes_map = dict(entry_shapes)
    tensor_names_slices = set(slices.keys()) if slices else set()
    tensor_names_transpose = set(transpose.keys()) if transpose else set()
    overlap = tensor_names_slices & tensor_names_transpose
    if overlap:
        offending = ", ".join(sorted(overlap))
        raise ValueError(
            f"Cannot apply slices and transpose to the same tensor(s): {offending}"
        )

    tensor_ops: dict[str, list[TensorViewOp]] = {}

    if slices:
        for name, slice_spec in sorted(slices.items()):
            shape = entry_shapes_map.get(name)
            if shape is None:
                raise ValueError(f"View references unknown tensor '{name}'")
            narrow_op = validate_narrow(name, shape, slice_spec)
            if narrow_op is None:
                continue
            tensor_ops.setdefault(name, []).append(narrow_op)

    if transpose:
        for name, dim_pairs in sorted(transpose.items()):
            shape = entry_shapes_map.get(name)
            if shape is None:
                raise ValueError(f"View references unknown tensor '{name}'")
            normalized = validate_transpose(name, shape, dim_pairs)
            if normalized:
                tensor_ops.setdefault(name, []).extend(normalized)

    if not tensor_ops:
        return ViewSpecBuildResult.identity()

    ordered_ops = {name: tensor_ops[name] for name in sorted(tensor_ops.keys())}
    view_spec_proto = store_daemon_pb2.ViewSpec()
    for name, view_ops in ordered_ops.items():
        op_container = view_spec_proto.tensors[name]
        for view_op in view_ops:
            op_proto = op_container.ops.add()
            if isinstance(view_op, NarrowOp):
                op_proto.narrow.dim = int(view_op.dim)
                op_proto.narrow.start = int(view_op.start)
                op_proto.narrow.length = int(view_op.length)
            elif isinstance(view_op, TransposeOp):
                op_proto.transpose.dim0 = int(view_op.dim0)
                op_proto.transpose.dim1 = int(view_op.dim1)
    result = ViewSpecBuildResult(proto=view_spec_proto, tensor_ops=ordered_ops)
    logger.debug("build_view_spec result: %r", result)
    return result
