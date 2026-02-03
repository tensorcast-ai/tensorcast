#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from tensorcast.api._view_ops import NarrowOp, ViewSpecBuildResult
from tensorcast.api.store.types import ArtifactError, CanonicalIndex


@dataclass(frozen=True, slots=True)
class Range:
    dim: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return int(self.end) - int(self.start)


@dataclass(frozen=True, slots=True)
class CopyPlanEntry:
    ckpt_name: str
    ckpt_range: Range | None
    dst_name: str
    dst_range: Range | None


CopyPlan = Sequence[CopyPlanEntry]
TargetTensors = Mapping[str, torch.Tensor]


_COPY_PLAN_VERSION = 1


def copy_plan_to_json(plan: CopyPlan) -> str:
    payload = {
        "version": _COPY_PLAN_VERSION,
        "entries": [_entry_to_dict(entry) for entry in plan],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def copy_plan_from_json(data: str) -> tuple[CopyPlanEntry, ...]:
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:  # noqa: PERF203
        raise ArtifactError(
            f"Invalid copy plan JSON: {exc}",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        ) from exc
    if not isinstance(payload, dict):
        raise ArtifactError(
            "Copy plan JSON must be an object",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    version = int(payload.get("version", 0))
    if version != _COPY_PLAN_VERSION:
        raise ArtifactError(
            f"Unsupported copy plan version {version}",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ArtifactError(
            "Copy plan JSON requires a non-empty entries list",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    parsed: list[CopyPlanEntry] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ArtifactError(
                "Copy plan entries must be objects",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        parsed.append(_entry_from_dict(entry))
    return tuple(parsed)


def normalize_copy_plan(plan: Sequence[object]) -> tuple[CopyPlanEntry, ...]:
    if not isinstance(plan, Sequence) or not plan:
        raise ArtifactError(
            "mapping must be a non-empty sequence of CopyPlanEntry objects",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    entries: list[CopyPlanEntry] = []
    for entry in plan:
        if isinstance(entry, CopyPlanEntry):
            entries.append(entry)
            continue
        if isinstance(entry, Mapping):
            entries.append(_entry_from_dict(entry))
            continue
        if isinstance(entry, tuple) and len(entry) == 4:
            ckpt_name, ckpt_range, dst_name, dst_range = entry
            entries.append(
                CopyPlanEntry(
                    ckpt_name=str(ckpt_name),
                    ckpt_range=_coerce_range(ckpt_range),
                    dst_name=str(dst_name),
                    dst_range=_coerce_range(dst_range),
                )
            )
            continue
        raise ArtifactError(
            "mapping entries must be CopyPlanEntry objects or dicts",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    return tuple(entries)


def validate_copy_plan(
    *,
    plan: CopyPlan,
    canonical_index: CanonicalIndex,
    target_tensors: TargetTensors,
    view_narrows: Mapping[str, Range] | None,
    require_full_coverage: bool,
) -> None:
    if not plan:
        raise ArtifactError(
            "mapping must include at least one entry",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    dst_names = [entry.dst_name for entry in plan]
    if len(set(dst_names)) != len(dst_names):
        # Allow multiple entries per dst tensor, but disallow duplicate entries for identical ranges.
        pass

    target_names = {str(name) for name in target_tensors}
    if set(dst_names) != target_names:
        raise ArtifactError(
            "mapping dst_name set must match target_tensors",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )

    canonical_by_name = {entry.name: entry for entry in canonical_index.entries}
    dst_slices: dict[str, list[tuple[int, int, int]]] = {}
    dst_dims: dict[str, int | None] = {}
    for idx, entry in enumerate(plan):
        ckpt_name = str(entry.ckpt_name)
        dst_name = str(entry.dst_name)
        if not ckpt_name:
            raise ArtifactError(
                f"mapping entry {idx} missing ckpt_name",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if not dst_name:
            raise ArtifactError(
                f"mapping entry {idx} missing dst_name",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        src_meta = canonical_by_name.get(ckpt_name)
        if src_meta is None:
            raise ArtifactError(
                f"mapping entry {idx} references unknown source '{ckpt_name}'",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if not _is_contiguous(
            shape=tuple(int(v) for v in src_meta.shape),
            stride=tuple(int(v) for v in src_meta.stride),
        ):
            raise ArtifactError(
                f"mapping source '{ckpt_name}' must be contiguous",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        dst_tensor = target_tensors.get(dst_name)
        if dst_tensor is None:
            raise ArtifactError(
                f"mapping entry {idx} references unknown target '{dst_name}'",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if not dst_tensor.is_contiguous():
            raise ArtifactError(
                f"mapping target '{dst_name}' must be contiguous",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if int(dst_tensor.storage_offset()) != 0:
            raise ArtifactError(
                f"mapping target '{dst_name}' must have storage_offset=0",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if dst_tensor.dtype != src_meta.dtype:
            raise ArtifactError(
                (
                    f"mapping entry {idx} dtype mismatch: "
                    f"{ckpt_name}({src_meta.dtype}) -> {dst_name}({dst_tensor.dtype})"
                ),
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        _validate_range(
            idx=idx,
            name=ckpt_name,
            shape=src_meta.shape,
            rng=entry.ckpt_range,
            view_narrow=view_narrows.get(ckpt_name) if view_narrows else None,
            role="source",
        )
        _validate_range(
            idx=idx,
            name=dst_name,
            shape=tuple(int(v) for v in dst_tensor.shape),
            rng=entry.dst_range,
            view_narrow=None,
            role="target",
        )

        dim = _resolve_dim(entry.ckpt_range, entry.dst_range)
        if dim is not None:
            prev_dim = dst_dims.get(dst_name)
            if prev_dim is None:
                dst_dims[dst_name] = dim
            elif prev_dim != dim:
                raise ArtifactError(
                    f"mapping target '{dst_name}' mixes slice dims {prev_dim} and {dim}",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )

        dst_slice = _range_to_interval(
            dim=dim,
            shape=tuple(int(v) for v in dst_tensor.shape),
            rng=entry.dst_range,
        )
        dst_slices.setdefault(dst_name, []).append((dst_slice[0], dst_slice[1], idx))

        _validate_element_count(
            idx=idx,
            src_meta=src_meta,
            dst_tensor=dst_tensor,
            src_range=entry.ckpt_range,
            dst_range=entry.dst_range,
            view_narrow=view_narrows.get(ckpt_name) if view_narrows else None,
        )

    _validate_dst_coverage(
        dst_slices=dst_slices,
        dst_dims=dst_dims,
        target_tensors=target_tensors,
        require_full_coverage=require_full_coverage,
    )


def view_narrow_ranges(view_spec: ViewSpecBuildResult | None) -> dict[str, Range]:
    if view_spec is None or view_spec.is_identity:
        return {}
    if view_spec.has_transpose:
        raise ArtifactError(
            "mapped binding does not support transpose views",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    narrows: dict[str, Range] = {}
    for name, ops in view_spec.tensor_ops.items():
        narrow_ops = [op for op in ops if isinstance(op, NarrowOp)]
        if not narrow_ops:
            continue
        if len(narrow_ops) > 1:
            raise ArtifactError(
                f"mapped binding supports a single narrow op per tensor (got {len(narrow_ops)})",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        narrow = narrow_ops[0]
        narrows[str(name)] = Range(
            dim=int(narrow.dim),
            start=int(narrow.start),
            end=int(narrow.start + narrow.length),
        )
    return narrows


def _entry_to_dict(entry: CopyPlanEntry) -> dict[str, object]:
    return {
        "ckpt_name": entry.ckpt_name,
        "ckpt_range": _range_to_dict(entry.ckpt_range),
        "dst_name": entry.dst_name,
        "dst_range": _range_to_dict(entry.dst_range),
    }


def _entry_from_dict(entry: Mapping[str, object]) -> CopyPlanEntry:
    ckpt_name = str(entry.get("ckpt_name") or "")
    dst_name = str(entry.get("dst_name") or "")
    return CopyPlanEntry(
        ckpt_name=ckpt_name,
        ckpt_range=_range_from_dict(entry.get("ckpt_range")),
        dst_name=dst_name,
        dst_range=_range_from_dict(entry.get("dst_range")),
    )


def _range_from_dict(data: object) -> Range | None:
    if data is None:
        return None
    if isinstance(data, Range):
        return data
    if not isinstance(data, Mapping):
        raise ArtifactError(
            "Range must be a mapping with dim/start/end",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    return Range(
        dim=int(data.get("dim", 0)),
        start=int(data.get("start", 0)),
        end=int(data.get("end", 0)),
    )


def _range_to_dict(rng: Range | None) -> dict[str, int] | None:
    if rng is None:
        return None
    return {"dim": int(rng.dim), "start": int(rng.start), "end": int(rng.end)}


def _coerce_range(value: object) -> Range | None:
    if value is None or isinstance(value, Range):
        return value
    if isinstance(value, Mapping):
        return _range_from_dict(value)
    raise ArtifactError(
        "Range must be None or a mapping with dim/start/end",
        status_code="INVALID_ARGUMENT",
        retryable=False,
    )


def _resolve_dim(src_range: Range | None, dst_range: Range | None) -> int | None:
    if src_range is not None and dst_range is not None:
        if src_range.dim != dst_range.dim:
            raise ArtifactError(
                f"mapping range dims differ ({src_range.dim} vs {dst_range.dim})",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        return int(src_range.dim)
    if src_range is not None:
        return int(src_range.dim)
    if dst_range is not None:
        return int(dst_range.dim)
    return None


def _range_to_interval(
    *,
    dim: int | None,
    shape: tuple[int, ...],
    rng: Range | None,
) -> tuple[int, int]:
    if not shape:
        return (0, 1)
    if dim is None:
        dim = int(rng.dim) if rng is not None else 0
    if dim < 0 or dim >= len(shape):
        raise ArtifactError(
            f"Slice dim {dim} out of range for shape {shape}",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    if rng is None:
        return (0, int(shape[dim]))
    return (int(rng.start), int(rng.end))


def _validate_range(
    *,
    idx: int,
    name: str,
    shape: tuple[int, ...],
    rng: Range | None,
    view_narrow: Range | None,
    role: str,
) -> None:
    if not shape:
        if rng is not None:
            raise ArtifactError(
                f"mapping entry {idx} {role} range set for scalar '{name}'",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        return
    if rng is None:
        if view_narrow is not None:
            raise ArtifactError(
                (
                    f"mapping entry {idx} {role} range required for view-narrowed "
                    f"tensor '{name}'"
                ),
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        return
    dim = int(rng.dim)
    if dim not in (0, 1):
        raise ArtifactError(
            f"mapping entry {idx} {role} dim must be 0 or 1",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    if dim >= len(shape):
        raise ArtifactError(
            f"mapping entry {idx} {role} dim {dim} out of range for '{name}'",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    if rng.start < 0 or rng.end <= rng.start:
        raise ArtifactError(
            f"mapping entry {idx} {role} range must have start < end",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    if rng.end > shape[dim]:
        raise ArtifactError(
            f"mapping entry {idx} {role} range exceeds shape for '{name}'",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    if view_narrow is not None:
        if dim != view_narrow.dim:
            raise ArtifactError(
                (
                    f"mapping entry {idx} {role} dim {dim} does not match "
                    f"view narrow dim {view_narrow.dim} for '{name}'"
                ),
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if rng.start < view_narrow.start or rng.end > view_narrow.end:
            raise ArtifactError(
                (
                    f"mapping entry {idx} {role} range outside view bounds for '{name}' "
                    f"({view_narrow.start}:{view_narrow.end})"
                ),
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )


def _validate_element_count(
    *,
    idx: int,
    src_meta,
    dst_tensor: torch.Tensor,
    src_range: Range | None,
    dst_range: Range | None,
    view_narrow: Range | None,
) -> None:
    src_shape = tuple(int(v) for v in src_meta.shape)
    dst_shape = tuple(int(v) for v in dst_tensor.shape)
    dim = _resolve_dim(src_range, dst_range)
    if not src_shape:
        src_count = 1
    else:
        src_start, src_end = _range_to_interval(
            dim=dim,
            shape=src_shape,
            rng=src_range,
        )
        if view_narrow is not None:
            src_start -= int(view_narrow.start)
            src_end -= int(view_narrow.start)
        src_count = _slice_element_count(src_shape, dim, src_start, src_end)
    if not dst_shape:
        dst_count = 1
    else:
        dst_start, dst_end = _range_to_interval(
            dim=dim,
            shape=dst_shape,
            rng=dst_range,
        )
        dst_count = _slice_element_count(dst_shape, dim, dst_start, dst_end)
    if src_count != dst_count:
        raise ArtifactError(
            f"mapping entry {idx} element count mismatch ({src_count} != {dst_count})",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )


def _slice_element_count(
    shape: tuple[int, ...],
    dim: int | None,
    start: int,
    end: int,
) -> int:
    if not shape:
        return 1
    if dim is None:
        dim = 0
    count = int(end) - int(start)
    for idx, size in enumerate(shape):
        if idx == dim:
            continue
        count *= int(size)
    return int(count)


def _is_contiguous(*, shape: tuple[int, ...], stride: tuple[int, ...]) -> bool:
    if not shape:
        return not stride
    expected = _compact_stride(shape)
    return tuple(stride) == expected


def _compact_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    stride = [0] * len(shape)
    acc = 1
    for idx in range(len(shape) - 1, -1, -1):
        stride[idx] = acc
        acc *= int(shape[idx])
    return tuple(stride)


def _validate_dst_coverage(
    *,
    dst_slices: Mapping[str, Sequence[tuple[int, int, int]]],
    dst_dims: Mapping[str, int | None],
    target_tensors: TargetTensors,
    require_full_coverage: bool,
) -> None:
    for name, tensor in target_tensors.items():
        intervals = list(dst_slices.get(name, ()))
        if not intervals:
            raise ArtifactError(
                f"mapping missing target '{name}'",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        dim = dst_dims.get(name)
        shape = tuple(int(v) for v in tensor.shape)
        if not shape:
            if len(intervals) != 1:
                raise ArtifactError(
                    f"mapping target '{name}' must have a single entry for scalar",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            continue
        if dim is None:
            dim = 0
        if dim < 0 or dim >= len(shape):
            raise ArtifactError(
                f"mapping target '{name}' dim {dim} out of range",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        intervals.sort(key=lambda item: item[0])
        cursor = 0
        for start, end, idx in intervals:
            if start < cursor:
                raise ArtifactError(
                    f"mapping entry {idx} overlaps prior slice for '{name}'",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            if require_full_coverage and start != cursor:
                raise ArtifactError(
                    f"mapping target '{name}' has gaps before entry {idx}",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            cursor = end
        if require_full_coverage and cursor != shape[dim]:
            raise ArtifactError(
                f"mapping target '{name}' does not cover full dim {dim}",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )


__all__ = [
    "CopyPlan",
    "CopyPlanEntry",
    "Range",
    "TargetTensors",
    "copy_plan_from_json",
    "copy_plan_to_json",
    "normalize_copy_plan",
    "validate_copy_plan",
    "view_narrow_ranges",
]
