#  Copyright (c) 2026, TensorCast Team.
"""Tensor parity diagnostics for TensorCast serving recipes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from tensorcast.api.store import BindingRealizationEntry
from tensorcast.api.store import Range as StoreRange
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.serving.builder.materialization import (
    narrow_by_range_spec,
    narrow_source_view,
)
from tensorcast.serving.builder.trace_ir import (
    CopyPlanEntry,
    MultiRange,
    Range,
    RangeSpec,
    TracePlan,
)


@dataclass(frozen=True)
class TensorParityProbe:
    op: str
    ckpt_name: str | None
    ckpt_range: RangeSpec | None
    dst_name: str
    dst_range: RangeSpec | None
    fill_value: Any | None = None

    @classmethod
    def from_copy_plan_entry(cls, entry: CopyPlanEntry) -> "TensorParityProbe":
        return cls(
            op=entry.op,
            ckpt_name=entry.ckpt_name,
            ckpt_range=entry.ckpt_range,
            dst_name=entry.dst_name,
            dst_range=entry.dst_range,
            fill_value=entry.fill_value,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "ckpt_name": self.ckpt_name,
            "dst_name": self.dst_name,
            "fill_value": self.fill_value,
        }


@dataclass(frozen=True)
class TensorParityMismatch:
    probe_index: int
    op: str
    ckpt_name: str | None
    dst_name: str
    reason: str
    total_elements: int
    sample_count: int
    mismatched_samples: int
    expected_shape: tuple[int, ...] | None = None
    actual_shape: tuple[int, ...] | None = None
    max_abs_diff: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe_index": self.probe_index,
            "op": self.op,
            "ckpt_name": self.ckpt_name,
            "dst_name": self.dst_name,
            "reason": self.reason,
            "expected_shape": self.expected_shape,
            "actual_shape": self.actual_shape,
            "total_elements": self.total_elements,
            "sample_count": self.sample_count,
            "mismatched_samples": self.mismatched_samples,
            "max_abs_diff": self.max_abs_diff,
        }


@dataclass(frozen=True)
class TensorParityReport:
    checked: int
    skipped: int
    mismatches: tuple[TensorParityMismatch, ...]

    @property
    def passed(self) -> bool:
        return not self.mismatches

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked": self.checked,
            "skipped": self.skipped,
            "passed": self.passed,
            "mismatches": [mismatch.to_dict() for mismatch in self.mismatches],
        }


def build_tensor_parity_probes_from_trace_plan(
    trace_plan: TracePlan,
    *,
    dst_names: Iterable[str] | None = None,
    max_entries: int | None = None,
    include_fill: bool = True,
) -> tuple[TensorParityProbe, ...]:
    selected_dst_names = None if dst_names is None else set(dst_names)
    probes: list[TensorParityProbe] = []
    for entry in trace_plan.copy_plan:
        if selected_dst_names is not None and entry.dst_name not in selected_dst_names:
            continue
        if entry.op == "fill" and not include_fill:
            continue
        if entry.op not in {"copy", "fill"}:
            continue
        probes.append(TensorParityProbe.from_copy_plan_entry(entry))
        if max_entries is not None and len(probes) >= int(max_entries):
            break
    return tuple(probes)


def build_tensor_parity_probes_from_realization_plan(
    realization_plan: Iterable[BindingRealizationEntry],
    *,
    dst_names: Iterable[str] | None = None,
    max_entries: int | None = None,
    include_fill: bool = True,
) -> tuple[TensorParityProbe, ...]:
    selected_dst_names = None if dst_names is None else set(dst_names)
    probes: list[TensorParityProbe] = []
    for entry in realization_plan:
        if selected_dst_names is not None and entry.dst_name not in selected_dst_names:
            continue
        if entry.op == "fill" and not include_fill:
            continue
        if entry.op not in {"copy", "fill"}:
            continue
        probes.append(
            TensorParityProbe(
                op=str(entry.op),
                ckpt_name=entry.source_name,
                ckpt_range=_range_spec_from_store_ranges(entry.source_ranges),
                dst_name=str(entry.dst_name),
                dst_range=_range_spec_from_store_ranges(entry.dst_ranges),
                fill_value=entry.fill_value,
            )
        )
        if max_entries is not None and len(probes) >= int(max_entries):
            break
    return tuple(probes)


def build_tensor_parity_probes_from_recipe(
    recipe: Any,
    *,
    dst_names: Iterable[str] | None = None,
    max_entries: int | None = None,
    include_fill: bool = True,
) -> tuple[TensorParityProbe, ...]:
    trace_plan = getattr(recipe, "trace_plan", None)
    copy_plan = tuple(getattr(trace_plan, "copy_plan", ()) or ())
    if copy_plan:
        return build_tensor_parity_probes_from_trace_plan(
            trace_plan,
            dst_names=dst_names,
            max_entries=max_entries,
            include_fill=include_fill,
        )
    realization_plan = tuple(getattr(recipe, "realization_plan", ()) or ())
    if realization_plan:
        return build_tensor_parity_probes_from_realization_plan(
            realization_plan,
            dst_names=dst_names,
            max_entries=max_entries,
            include_fill=include_fill,
        )
    realization_plan_proto = getattr(recipe, "realization_plan_proto", b"") or b""
    if realization_plan_proto:
        return build_tensor_parity_probes_from_realization_plan_proto(
            realization_plan_proto,
            dst_names=dst_names,
            max_entries=max_entries,
            include_fill=include_fill,
        )
    return ()


def build_tensor_parity_probes_from_realization_plan_proto(
    realization_plan_proto: bytes | store_daemon_pb2.BindingRealizationPlan,
    *,
    dst_names: Iterable[str] | None = None,
    max_entries: int | None = None,
    include_fill: bool = True,
) -> tuple[TensorParityProbe, ...]:
    proto = _coerce_realization_plan_proto(realization_plan_proto)
    selected_dst_names = None if dst_names is None else set(dst_names)
    probes: list[TensorParityProbe] = []
    for entry in proto.entries:
        if selected_dst_names is not None and entry.dst_name not in selected_dst_names:
            continue
        op = _op_from_proto(entry.op_kind)
        if op is None:
            continue
        if op == "fill" and not include_fill:
            continue
        probes.append(
            TensorParityProbe(
                op=op,
                ckpt_name=str(entry.source_name) if entry.source_name else None,
                ckpt_range=_range_spec_from_store_ranges(entry.source_ranges),
                dst_name=str(entry.dst_name),
                dst_range=_range_spec_from_store_ranges(entry.dst_ranges),
                fill_value=bytes(entry.fill_value) if entry.fill_value else None,
            )
        )
        if max_entries is not None and len(probes) >= int(max_entries):
            break
    return tuple(probes)


def evaluate_tensor_parity_probes(
    trace_plan: TracePlan,
    source_tensors: Mapping[str, torch.Tensor],
    target_tensors: Mapping[str, torch.Tensor],
    *,
    probes: Iterable[TensorParityProbe] | None = None,
    max_elements_per_probe: int = 4096,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> TensorParityReport:
    resolved_probes = (
        tuple(probes)
        if probes is not None
        else (build_tensor_parity_probes_from_trace_plan(trace_plan))
    )
    mismatches: list[TensorParityMismatch] = []
    checked = 0
    skipped = 0
    for probe_index, probe in enumerate(resolved_probes):
        mismatch = _evaluate_one_probe(
            probe_index,
            probe,
            trace_plan=trace_plan,
            source_tensors=source_tensors,
            target_tensors=target_tensors,
            max_elements_per_probe=max_elements_per_probe,
            atol=atol,
            rtol=rtol,
        )
        if mismatch is None:
            checked += 1
            continue
        if mismatch.reason == "empty_range":
            skipped += 1
            continue
        checked += 1
        mismatches.append(mismatch)
    return TensorParityReport(
        checked=checked,
        skipped=skipped,
        mismatches=tuple(mismatches),
    )


def evaluate_recipe_tensor_parity(
    recipe: Any,
    source_tensors: Mapping[str, torch.Tensor],
    target_tensors: Mapping[str, torch.Tensor],
    *,
    probes: Iterable[TensorParityProbe] | None = None,
    dst_names: Iterable[str] | None = None,
    max_entries: int | None = None,
    include_fill: bool = True,
    max_elements_per_probe: int = 4096,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> TensorParityReport:
    resolved_probes = (
        tuple(probes)
        if probes is not None
        else (
            build_tensor_parity_probes_from_recipe(
                recipe,
                dst_names=dst_names,
                max_entries=max_entries,
                include_fill=include_fill,
            )
        )
    )
    return evaluate_tensor_parity_probes(
        recipe.trace_plan,
        source_tensors,
        target_tensors,
        probes=resolved_probes,
        max_elements_per_probe=max_elements_per_probe,
        atol=atol,
        rtol=rtol,
    )


def _range_spec_from_store_ranges(
    ranges: Iterable[StoreRange],
) -> RangeSpec | None:
    converted = tuple(
        Range(
            dim=int(rng.dim),
            start=int(rng.start),
            end=int(rng.end),
        )
        for rng in ranges
    )
    if not converted:
        return None
    if len(converted) == 1:
        return converted[0]
    return MultiRange(ranges=converted)


def _coerce_realization_plan_proto(
    value: bytes | store_daemon_pb2.BindingRealizationPlan,
) -> store_daemon_pb2.BindingRealizationPlan:
    if isinstance(value, store_daemon_pb2.BindingRealizationPlan):
        return value
    proto = store_daemon_pb2.BindingRealizationPlan()
    proto.ParseFromString(bytes(value))
    return proto


def _op_from_proto(op_kind: int) -> str | None:
    if op_kind == store_daemon_pb2.BINDING_REALIZATION_OP_KIND_COPY:
        return "copy"
    if op_kind in {
        store_daemon_pb2.BINDING_REALIZATION_OP_KIND_CONST_FILL,
        store_daemon_pb2.BINDING_REALIZATION_OP_KIND_SCALAR_FILL,
    }:
        return "fill"
    return None


def _evaluate_one_probe(
    probe_index: int,
    probe: TensorParityProbe,
    *,
    trace_plan: TracePlan,
    source_tensors: Mapping[str, torch.Tensor],
    target_tensors: Mapping[str, torch.Tensor],
    max_elements_per_probe: int,
    atol: float,
    rtol: float,
) -> TensorParityMismatch | None:
    actual_base = target_tensors.get(probe.dst_name)
    if actual_base is None:
        return _mismatch(
            probe_index,
            probe,
            reason="missing_target_tensor",
            total_elements=0,
            sample_count=0,
            mismatched_samples=0,
        )
    actual_view = (
        actual_base
        if probe.dst_range is None
        else narrow_by_range_spec(actual_base, probe.dst_range)
    )
    total_elements = int(actual_view.numel())
    if total_elements == 0:
        return _mismatch(
            probe_index,
            probe,
            reason="empty_range",
            total_elements=0,
            sample_count=0,
            mismatched_samples=0,
            actual_shape=tuple(actual_view.shape),
        )

    expected_view = _expected_view_for_probe(
        probe_index,
        probe,
        trace_plan=trace_plan,
        source_tensors=source_tensors,
        actual_view=actual_view,
    )
    if isinstance(expected_view, TensorParityMismatch):
        return expected_view

    if tuple(expected_view.shape) != tuple(actual_view.shape):
        return _mismatch(
            probe_index,
            probe,
            reason="shape_mismatch",
            total_elements=total_elements,
            sample_count=0,
            mismatched_samples=0,
            expected_shape=tuple(expected_view.shape),
            actual_shape=tuple(actual_view.shape),
        )

    offsets = _sample_offsets(total_elements, max_elements_per_probe)
    expected_samples = _sample_tensor_values(expected_view, offsets)
    actual_samples = _sample_tensor_values(actual_view, offsets)
    equal_mask = _equal_samples(
        expected_samples,
        actual_samples,
        atol=atol,
        rtol=rtol,
    )
    mismatched_samples = int((~equal_mask).sum().item())
    if mismatched_samples == 0:
        return None
    return _mismatch(
        probe_index,
        probe,
        reason="value_mismatch",
        total_elements=total_elements,
        sample_count=len(offsets),
        mismatched_samples=mismatched_samples,
        expected_shape=tuple(expected_view.shape),
        actual_shape=tuple(actual_view.shape),
        max_abs_diff=_max_abs_diff(expected_samples, actual_samples),
    )


def _expected_view_for_probe(
    probe_index: int,
    probe: TensorParityProbe,
    *,
    trace_plan: TracePlan,
    source_tensors: Mapping[str, torch.Tensor],
    actual_view: torch.Tensor,
) -> torch.Tensor | TensorParityMismatch:
    if probe.op == "copy":
        if probe.ckpt_name is None:
            return _mismatch_for_probe(probe_index, probe, "missing_source_name")
        source_base = source_tensors.get(probe.ckpt_name)
        if source_base is None:
            return _mismatch_for_probe(probe_index, probe, "missing_source_tensor")
        source_view = narrow_source_view(
            source_base,
            probe.ckpt_name,
            probe.ckpt_range,
            trace_plan.src_hull,
        )
        if source_view.ndim == 0 and actual_view.numel() == 1:
            return source_view.reshape(actual_view.shape)
        return source_view

    if probe.op == "fill":
        fill_value = probe.fill_value
        if fill_value is None:
            if probe.ckpt_name is None:
                return _mismatch_for_probe(probe_index, probe, "missing_fill_value")
            source_base = source_tensors.get(str(probe.ckpt_name))
            if source_base is None:
                return _mismatch_for_probe(
                    probe_index, probe, "missing_fill_source_tensor"
                )
            source_view = narrow_source_view(
                source_base,
                str(probe.ckpt_name),
                probe.ckpt_range,
                trace_plan.src_hull,
            )
            if source_view.numel() != 1:
                return _mismatch_for_probe(probe_index, probe, "non_scalar_fill_source")
            fill_value = source_view.reshape(()).item()
        if isinstance(fill_value, bytes):
            return _mismatch_for_probe(probe_index, probe, "encoded_const_fill_value")
        return torch.full_like(actual_view, fill_value)

    return _mismatch_for_probe(probe_index, probe, f"unsupported_op:{probe.op}")


def _mismatch_for_probe(
    probe_index: int,
    probe: TensorParityProbe,
    reason: str,
) -> TensorParityMismatch:
    return TensorParityMismatch(
        probe_index=probe_index,
        op=probe.op,
        ckpt_name=probe.ckpt_name,
        dst_name=probe.dst_name,
        reason=reason,
        total_elements=0,
        sample_count=0,
        mismatched_samples=0,
    )


def _mismatch(
    probe_index: int,
    probe: TensorParityProbe,
    *,
    reason: str,
    total_elements: int,
    sample_count: int,
    mismatched_samples: int,
    expected_shape: tuple[int, ...] | None = None,
    actual_shape: tuple[int, ...] | None = None,
    max_abs_diff: float | None = None,
) -> TensorParityMismatch:
    return TensorParityMismatch(
        probe_index=probe_index,
        op=probe.op,
        ckpt_name=probe.ckpt_name,
        dst_name=probe.dst_name,
        reason=reason,
        expected_shape=expected_shape,
        actual_shape=actual_shape,
        total_elements=total_elements,
        sample_count=sample_count,
        mismatched_samples=mismatched_samples,
        max_abs_diff=max_abs_diff,
    )


def _sample_offsets(numel: int, max_elements: int) -> tuple[int, ...]:
    if numel <= 0 or max_elements <= 0:
        return ()
    if numel <= max_elements:
        return tuple(range(numel))
    if max_elements == 1:
        return (0,)
    last = numel - 1
    return tuple(
        dict.fromkeys(
            (index * last) // (max_elements - 1) for index in range(max_elements)
        )
    )


def _flat_offset_to_index(shape: tuple[int, ...], offset: int) -> tuple[int, ...]:
    if not shape:
        return ()
    indices = []
    remaining = int(offset)
    for dim_size in reversed(shape):
        indices.append(remaining % int(dim_size))
        remaining //= int(dim_size)
    return tuple(reversed(indices))


def _sample_tensor_values(
    tensor: torch.Tensor,
    offsets: tuple[int, ...],
) -> torch.Tensor:
    if not offsets:
        return torch.empty((0,), dtype=tensor.dtype, device="cpu")
    shape = tuple(int(dim) for dim in tensor.shape)
    values = [
        tensor[_flat_offset_to_index(shape, offset)].detach().cpu()
        for offset in offsets
    ]
    return torch.stack([value.reshape(()) for value in values])


def _equal_samples(
    expected: torch.Tensor,
    actual: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> torch.Tensor:
    if expected.dtype.is_floating_point or actual.dtype.is_floating_point:
        return torch.isclose(
            actual.to(torch.float32),
            expected.to(torch.float32),
            atol=atol,
            rtol=rtol,
            equal_nan=True,
        )
    if torch.is_complex(expected) or torch.is_complex(actual):
        return torch.isclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True)
    return actual == expected


def _max_abs_diff(expected: torch.Tensor, actual: torch.Tensor) -> float | None:
    if expected.numel() == 0:
        return None
    if expected.dtype == torch.bool or actual.dtype == torch.bool:
        return None
    if torch.is_complex(actual) or torch.is_complex(expected):
        diff = torch.abs(actual - expected)
    else:
        diff = actual.to(torch.float32) - expected.to(torch.float32)
        diff = diff.abs()
    return float(diff.max().item())


__all__ = [
    "TensorParityMismatch",
    "TensorParityProbe",
    "TensorParityReport",
    "build_tensor_parity_probes_from_realization_plan",
    "build_tensor_parity_probes_from_realization_plan_proto",
    "build_tensor_parity_probes_from_recipe",
    "build_tensor_parity_probes_from_trace_plan",
    "evaluate_recipe_tensor_parity",
    "evaluate_tensor_parity_probes",
]
