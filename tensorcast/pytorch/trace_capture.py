#  Copyright (c) 2026, TensorCast Team.

"""PyTorch trace capture primitives for serving recipe compilation."""

from __future__ import annotations

import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Optional, Protocol

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from tensorcast.serving.builder.trace_ir import (
    CopyPlanEntry,
    MultiRange,
    Range,
    RangeSpec,
    TracePlan,
)


class TensorMetaLike(Protocol):
    dtype: torch.dtype
    shape: tuple[int, ...]


@dataclass(frozen=True)
class TraceActivation:
    set_active: Callable[[], Any]
    reset_active: Callable[[Any], None]


class ScalarProxy(float):
    __slots__ = ("ckpt_name", "ckpt_range")
    ckpt_name: str
    ckpt_range: Optional[RangeSpec]

    def __new__(
        cls,
        value: float,
        ckpt_name: str,
        ckpt_range: Optional[RangeSpec],
    ):
        obj = float.__new__(cls, value)
        obj.ckpt_name = ckpt_name
        obj.ckpt_range = ckpt_range
        return obj


class TraceMode(TorchDispatchMode):
    _PASSTHROUGH = {
        "aten::view",
        "aten::reshape",
        "aten::_reshape_alias",
        "aten::view_as",
        "aten::reshape_as",
        "aten::alias",
        "aten::detach",
        "aten::contiguous",
        "aten::clone",
        "aten::flatten",
        "aten::to",
        "aten::_to_copy",
        "aten::type_as",
    }
    _SLICE_OPS = {
        "aten::slice",
        "aten::select",
        "aten::narrow",
        "aten::split",
        "aten::split_with_sizes",
        "aten::chunk",
        "aten::unbind",
    }

    def __init__(
        self,
        *,
        src_registry: dict[int, str],
        dst_registry: dict[int, str],
    ) -> None:
        super().__init__()
        self._src_registry = src_registry
        self._dst_registry = dst_registry
        self._view_meta: dict[
            int,
            tuple[
                weakref.ReferenceType[torch.Tensor],
                int,
                Optional[RangeSpec],
                Optional[tuple[int, ...]],
            ],
        ] = {}
        self.copy_plan: list[CopyPlanEntry] = []
        self.expected_src_names: set[str] = set()
        self.expected_dst_names: set[str] = set()
        self._prev_tensor_item: Optional[Callable[..., Any]] = None
        self._pending_scalar_proxy: Optional[ScalarProxy] = None

    def __enter__(self):
        super().__enter__()

        self._prev_tensor_item = torch.Tensor.item

        def _item_with_trace(tensor: torch.Tensor, *args, **kwargs):
            prev_tensor_item = self._prev_tensor_item
            if prev_tensor_item is None:
                raise RuntimeError("tensorcast trace mode is not initialized")
            value = prev_tensor_item(tensor, *args, **kwargs)
            if not isinstance(value, (int, float)):
                return value
            base_id, base_range = self._resolve_base_and_range(tensor)
            ckpt_name = self._src_registry.get(base_id)
            if ckpt_name is None:
                return value
            proxy = ScalarProxy(float(value), ckpt_name, base_range)
            self._pending_scalar_proxy = proxy
            return proxy

        torch.Tensor.item = _item_with_trace
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            if self._prev_tensor_item is not None:
                torch.Tensor.item = self._prev_tensor_item
                self._prev_tensor_item = None
            self._pending_scalar_proxy = None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = func._schema.name

        if name == "aten::copy_":
            dst = args[0]
            src = args[1]
            self._record_copy(dst, src)
            return dst

        if name == "aten::_local_scalar_dense":
            tensor = args[0]
            base_id, base_range = self._resolve_base_and_range(tensor)
            if base_id not in self._src_registry:
                raise RuntimeError(
                    "Trace encountered scalar from unknown source tensor"
                )
            ckpt_name = self._src_registry[base_id]
            proxy = ScalarProxy(0.0, ckpt_name, base_range)
            self._pending_scalar_proxy = proxy
            return proxy

        if name == "aten::fill_":
            dst = args[0]
            value = args[1]
            if isinstance(value, ScalarProxy):
                self._record_fill_from_ckpt(dst, value)
                self._pending_scalar_proxy = None
            elif isinstance(value, (int, float)):
                pending = self._pending_scalar_proxy
                if pending is not None and float(value) == float(pending):
                    self._record_fill_from_ckpt(dst, pending)
                else:
                    self._record_fill_const(dst, value)
                self._pending_scalar_proxy = None
            else:
                raise RuntimeError(
                    "Trace encountered fill_ with unsupported value type"
                )
            return dst

        if name in self._PASSTHROUGH:
            result = func(*args, **kwargs)
            return self._propagate_view(result, args[0])

        if name in self._SLICE_OPS:
            return self._handle_slice_op(name, func, args, kwargs)

        if self._has_tracked_tensor(args, kwargs):
            raise RuntimeError(f"Unsupported op in tensorcast trace: {name}")
        return func(*args, **kwargs)

    def _lookup_view_meta(
        self, tensor: torch.Tensor
    ) -> Optional[tuple[int, Optional[RangeSpec], Optional[tuple[int, ...]]]]:
        tid = id(tensor)
        item = self._view_meta.get(tid)
        if item is None:
            return None
        wr, base_id, base_range, dim_map = item
        if wr() is not tensor:
            self._view_meta.pop(tid, None)
            return None
        return base_id, base_range, dim_map

    def _has_tracked_tensor(self, args, kwargs) -> bool:
        for obj in list(args) + list(kwargs.values()):
            if isinstance(obj, torch.Tensor):
                if self._lookup_view_meta(obj) is not None:
                    return True
                if id(obj) in self._src_registry or id(obj) in self._dst_registry:
                    return True
        return False

    def _resolve_base_and_meta(
        self, tensor: torch.Tensor
    ) -> tuple[int, Optional[RangeSpec], Optional[tuple[int, ...]]]:
        meta = self._lookup_view_meta(tensor)
        if meta is not None:
            return meta
        tid = id(tensor)
        if tid in self._src_registry or tid in self._dst_registry:
            dim_map = tuple(range(tensor.dim()))
            return tid, None, dim_map
        raise RuntimeError("Trace encountered tensor that is not tracked")

    def _resolve_base_and_range(
        self,
        tensor: torch.Tensor,
    ) -> tuple[int, Optional[RangeSpec]]:
        base_id, base_range, _ = self._resolve_base_and_meta(tensor)
        return base_id, base_range

    def _propagate_view(
        self,
        result: Any,
        src: torch.Tensor,
        range_spec: Optional[RangeSpec] = None,
        dim_map: Optional[tuple[int, ...]] = None,
    ) -> Any:
        if not isinstance(result, torch.Tensor):
            return result
        base_id, base_range, base_dim_map = self._resolve_base_and_meta(src)
        propagated_range = (
            base_range
            if range_spec is None
            else self._compose_ranges(base_range, range_spec, base_dim_map)
        )
        propagated_dim_map = (
            base_dim_map
            if dim_map is None
            else tuple(base_dim_map[dim] for dim in dim_map)
        )
        self._view_meta[id(result)] = (
            weakref.ref(result),
            base_id,
            propagated_range,
            propagated_dim_map,
        )
        return result

    def _compose_ranges(
        self,
        parent: Optional[RangeSpec],
        child: RangeSpec,
        parent_dim_map: Optional[tuple[int, ...]],
    ) -> RangeSpec:
        if parent is None:
            return child
        if parent_dim_map is None:
            raise RuntimeError("Missing dim map for composed trace view")
        child_ranges = child.ranges if isinstance(child, MultiRange) else (child,)
        parent_ranges = parent.ranges if isinstance(parent, MultiRange) else (parent,)
        parent_by_dim = {rng.dim: rng for rng in parent_ranges}
        composed: list[Range] = [
            Range(dim=rng.dim, start=rng.start, end=rng.end) for rng in parent_ranges
        ]
        for rng in child_ranges:
            if rng.dim >= len(parent_dim_map):
                raise RuntimeError("Trace dim map is out of range")
            base_dim = parent_dim_map[rng.dim]
            parent_rng = parent_by_dim.get(base_dim)
            if parent_rng is None:
                composed.append(Range(dim=base_dim, start=rng.start, end=rng.end))
            else:
                composed = [
                    Range(
                        dim=item.dim,
                        start=(parent_rng.start + rng.start)
                        if item.dim == base_dim
                        else item.start,
                        end=(parent_rng.start + rng.end)
                        if item.dim == base_dim
                        else item.end,
                    )
                    for item in composed
                ]
        composed.sort(key=lambda item: item.dim)
        if len(composed) == 1:
            return composed[0]
        return MultiRange(ranges=tuple(composed))

    @staticmethod
    def _normalize_dim(dim: int, num_dims: int) -> int:
        if dim < 0:
            dim += num_dims
        if dim < 0 or dim >= num_dims:
            raise RuntimeError(f"Trace slice dim out of range: {dim}")
        return dim

    def _handle_slice_op(self, name: str, func, args, kwargs):
        src = args[0]
        result = func(*args, **kwargs)
        if not isinstance(result, torch.Tensor):
            return result
        if name == "aten::slice":
            dim = kwargs.get("dim", args[1] if len(args) > 1 else 0)
            start = kwargs.get("start", args[2] if len(args) > 2 else 0)
            end = kwargs.get("end", args[3] if len(args) > 3 else src.size(dim))
            dim = self._normalize_dim(int(dim), src.dim())
            size = src.size(dim)
            start = int(start)
            end = size if end is None else int(end)
            if start < 0:
                start += size
            if end < 0:
                end += size
            start = max(0, min(start, size))
            end = max(start, min(end, size))
            if start == 0 and end == size:
                return self._propagate_view(result, src)
            return self._propagate_view(
                result, src, range_spec=Range(dim=dim, start=start, end=end)
            )
        if name == "aten::select":
            dim = kwargs.get("dim", args[1])
            index = kwargs.get("index", args[2])
            dim = self._normalize_dim(int(dim), src.dim())
            size = src.size(dim)
            index = int(index)
            if index < 0:
                index += size
            if index < 0 or index >= size:
                raise RuntimeError("Trace select index out of range")
            dim_map = tuple(idx for idx in range(src.dim()) if idx != dim)
            return self._propagate_view(
                result,
                src,
                range_spec=Range(dim=dim, start=index, end=index + 1),
                dim_map=dim_map,
            )
        if name == "aten::narrow":
            dim = kwargs.get("dim", args[1])
            start = kwargs.get("start", args[2])
            length = kwargs.get("length", args[3])
            dim = self._normalize_dim(int(dim), src.dim())
            start = int(start)
            length = int(length)
            if start < 0:
                start += src.size(dim)
            end = start + length
            if start == 0 and end == src.size(dim):
                return self._propagate_view(result, src)
            return self._propagate_view(
                result, src, range_spec=Range(dim=dim, start=start, end=end)
            )
        if name == "aten::split":
            dim = kwargs.get("dim", args[2] if len(args) > 2 else 0)
            sections = kwargs.get("split_size", args[1])
            dim = self._normalize_dim(int(dim), src.dim())
            result_tensors = []
            start = 0
            size = src.size(dim)
            split_size = int(sections)
            while start < size:
                end = min(size, start + split_size)
                idx = len(result_tensors)
                result_tensors.append(
                    self._propagate_view(
                        result[idx],
                        src,
                        range_spec=Range(dim=dim, start=start, end=end),
                    )
                )
                start = end
            return type(result)(result_tensors)
        if name == "aten::split_with_sizes":
            dim = kwargs.get("dim", args[2] if len(args) > 2 else 0)
            split_sizes = kwargs.get("split_sizes", args[1])
            dim = self._normalize_dim(int(dim), src.dim())
            result_tensors = []
            start = 0
            for idx, part in enumerate(split_sizes):
                end = start + int(part)
                result_tensors.append(
                    self._propagate_view(
                        result[idx],
                        src,
                        range_spec=Range(dim=dim, start=start, end=end),
                    )
                )
                start = end
            return type(result)(result_tensors)
        if name == "aten::chunk":
            chunks = int(kwargs.get("chunks", args[1]))
            dim = kwargs.get("dim", args[2] if len(args) > 2 else 0)
            dim = self._normalize_dim(int(dim), src.dim())
            size = src.size(dim)
            base = size // chunks
            rem = size % chunks
            result_tensors = []
            start = 0
            for idx in range(len(result)):
                length = base + (1 if idx < rem else 0)
                end = start + length
                result_tensors.append(
                    self._propagate_view(
                        result[idx],
                        src,
                        range_spec=Range(dim=dim, start=start, end=end),
                    )
                )
                start = end
            return type(result)(result_tensors)
        if name == "aten::unbind":
            dim = kwargs.get("dim", args[1] if len(args) > 1 else 0)
            dim = self._normalize_dim(int(dim), src.dim())
            dim_map = tuple(idx for idx in range(src.dim()) if idx != dim)
            result_tensors = []
            for idx in range(len(result)):
                result_tensors.append(
                    self._propagate_view(
                        result[idx],
                        src,
                        range_spec=Range(dim=dim, start=idx, end=idx + 1),
                        dim_map=dim_map,
                    )
                )
            return type(result)(result_tensors)
        raise RuntimeError(f"Unsupported slice op in tensorcast trace: {name}")

    def _record_copy(self, dst: torch.Tensor, src: torch.Tensor) -> None:
        base_dst_id, dst_range = self._resolve_base_and_range(dst)
        base_src_id, src_range = self._resolve_base_and_range(src)

        dst_name = self._dst_registry.get(base_dst_id)
        if dst_name is None:
            raise RuntimeError("Trace encountered copy into unknown dst tensor")
        src_name = self._src_registry.get(base_src_id)
        if src_name is None:
            raise RuntimeError("Trace encountered copy from unknown source tensor")

        self.copy_plan.append(
            CopyPlanEntry(
                op="copy",
                ckpt_name=src_name,
                ckpt_range=src_range,
                dst_name=dst_name,
                dst_range=dst_range,
            )
        )
        self.expected_src_names.add(src_name)
        self.expected_dst_names.add(dst_name)

    def _record_fill_const(self, dst: torch.Tensor, value: float) -> None:
        base_dst_id, dst_range = self._resolve_base_and_range(dst)
        dst_name = self._dst_registry.get(base_dst_id)
        if dst_name is None:
            raise RuntimeError("Trace encountered fill into unknown dst tensor")
        self.copy_plan.append(
            CopyPlanEntry(
                op="fill",
                ckpt_name=None,
                ckpt_range=None,
                dst_name=dst_name,
                dst_range=dst_range,
                fill_value=float(value),
            )
        )
        self.expected_dst_names.add(dst_name)

    def _record_fill_from_ckpt(self, dst: torch.Tensor, value: ScalarProxy) -> None:
        base_dst_id, dst_range = self._resolve_base_and_range(dst)
        dst_name = self._dst_registry.get(base_dst_id)
        if dst_name is None:
            raise RuntimeError("Trace encountered fill into unknown dst tensor")
        self.copy_plan.append(
            CopyPlanEntry(
                op="fill",
                ckpt_name=value.ckpt_name,
                ckpt_range=value.ckpt_range,
                dst_name=dst_name,
                dst_range=dst_range,
            )
        )
        self.expected_src_names.add(value.ckpt_name)
        self.expected_dst_names.add(dst_name)


def meta_weights_iterator(
    ordered_names: list[str],
    meta_by_name: Mapping[str, TensorMetaLike],
    *,
    src_registry: dict[int, str],
    device: str = "meta",
    keepalive: Optional[list[torch.Tensor]] = None,
):
    for name in ordered_names:
        meta = meta_by_name[name]
        tensor = torch.empty(meta.shape, dtype=meta.dtype, device=device)
        src_registry[id(tensor)] = name
        if keepalive is not None:
            keepalive.append(tensor)
        yield name, tensor


def trace_model_load(
    model: nn.Module,
    ordered_names: list[str],
    meta_by_name: Mapping[str, TensorMetaLike],
    *,
    native_load_weights: Optional[Callable[[nn.Module, Any], None]] = None,
    activation: TraceActivation | None = None,
    debug_dump_trace: bool = False,
    logger: Any = None,
) -> TracePlan:
    src_registry: dict[int, str] = {}
    dst_registry: dict[int, str] = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        dst_registry.setdefault(id(param), name)
        dst_registry.setdefault(id(param.data), name)
    for name, buf in model.named_buffers(remove_duplicate=False):
        dst_registry.setdefault(id(buf), name)
        dst_registry.setdefault(id(buf.data), name)

    src_keepalive: list[torch.Tensor] = []
    weights_iterator = meta_weights_iterator(
        ordered_names,
        meta_by_name,
        src_registry=src_registry,
        keepalive=src_keepalive,
    )
    trace_mode = TraceMode(src_registry=src_registry, dst_registry=dst_registry)
    trace_token = activation.set_active() if activation is not None else None
    try:
        with trace_mode:
            if native_load_weights is None:
                model.load_weights(weights_iterator)
            else:
                native_load_weights(model, weights_iterator)
    finally:
        if activation is not None:
            activation.reset_active(trace_token)

    if not trace_mode.copy_plan:
        raise RuntimeError("tensorcast trace produced empty copy plan")
    if not trace_mode.expected_src_names:
        raise RuntimeError("tensorcast trace produced empty source set")
    if not trace_mode.expected_dst_names:
        raise RuntimeError("tensorcast trace produced empty dst set")

    tensorcast_slices, src_hull = build_tensorcast_slices(
        trace_mode.copy_plan, meta_by_name
    )

    if debug_dump_trace and logger is not None:
        logger.info(
            "tensorcast trace: %d copy entries, %d src tensors, %d dst tensors",
            len(trace_mode.copy_plan),
            len(trace_mode.expected_src_names),
            len(trace_mode.expected_dst_names),
        )

    return TracePlan(
        copy_plan=trace_mode.copy_plan,
        expected_src_names=trace_mode.expected_src_names,
        expected_dst_names=trace_mode.expected_dst_names,
        tensorcast_slices=tensorcast_slices,
        src_hull=src_hull,
    )


def build_tensorcast_slices(
    copy_plan: list[CopyPlanEntry],
    meta_by_name: Mapping[str, TensorMetaLike],
) -> tuple[dict[str, Range], dict[str, Range]]:
    full_reads: set[str] = set()
    ranges_by_name: dict[str, list[Range]] = {}
    for entry in copy_plan:
        if entry.ckpt_name is None:
            continue
        if entry.ckpt_range is None:
            full_reads.add(entry.ckpt_name)
            continue
        if isinstance(entry.ckpt_range, MultiRange):
            full_reads.add(entry.ckpt_name)
            continue
        ranges_by_name.setdefault(entry.ckpt_name, []).append(entry.ckpt_range)

    src_hull: dict[str, Range] = {}
    for name, ranges in ranges_by_name.items():
        dim = ranges[0].dim
        for rng in ranges[1:]:
            if rng.dim != dim:
                raise RuntimeError(f"tensorcast trace saw multi-dim slices for {name}")
        start = min(rng.start for rng in ranges)
        end = max(rng.end for rng in ranges)
        if start >= end:
            raise RuntimeError(f"tensorcast trace saw empty slice for {name}")
        src_hull[name] = Range(dim=dim, start=start, end=end)

    tensorcast_slices: dict[str, Range] = {}
    for name, hull in src_hull.items():
        meta = meta_by_name.get(name)
        if meta is None:
            raise RuntimeError(f"Missing safetensors metadata for {name}")
        if name in full_reads:
            continue
        if len(meta.shape) == 0:
            continue
        if hull.dim >= len(meta.shape):
            raise RuntimeError(f"Slice dim {hull.dim} out of range for {name}")
        extent = meta.shape[hull.dim]
        if hull.start < 0 or hull.end > extent:
            raise RuntimeError(
                f"Slice out of range for {name}: {hull.start}:{hull.end}"
            )
        if hull.start == 0 and hull.end == extent:
            continue
        tensorcast_slices[name] = hull
    return tensorcast_slices, src_hull


__all__ = [
    "ScalarProxy",
    "TensorMetaLike",
    "TraceActivation",
    "TraceMode",
    "build_tensorcast_slices",
    "meta_weights_iterator",
    "trace_model_load",
]
