#  Copyright (c) 2026, TensorCast Team.
"""Generic PyTorch module tensor attach and collection helpers."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

import tensorcast.serving.contract as tc_contract

_RESERVED_TENSORCAST_PREFIX = "__tensorcast_meta__."


@dataclass(frozen=True)
class TensorInvariant:
    data_ptr: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True)
class AttachResult:
    attached: tuple[str, ...]
    skipped: tuple[str, ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]


def attach_tensors_to_module(
    model: nn.Module,
    tensors: Mapping[str, torch.Tensor],
    *,
    replace_meta_params: bool,
    skip_names: Collection[str] = (),
    skip_reserved_tensor_names: bool = True,
    preserve_aliases: bool = True,
    fail_on_missing: bool = True,
    fail_on_unexpected: bool = True,
) -> AttachResult:
    skip = {str(name) for name in skip_names}
    parameter_aliases, buffer_aliases = _collect_alias_registrations(model)
    known_names = _module_tensor_names(model)
    attached: set[str] = set()
    skipped: list[str] = []
    unexpected: list[str] = []

    for raw_name, tensor in tensors.items():
        full_name = str(raw_name)
        if full_name in skip or (
            skip_reserved_tensor_names and _is_reserved_tensorcast_name(full_name)
        ):
            skipped.append(full_name)
            continue
        if full_name not in known_names:
            unexpected.append(full_name)
            continue
        module, local_name = _lookup_module_and_name(model, full_name)
        param = module._parameters.get(local_name)
        if param is not None:
            aliases = (
                parameter_aliases.get(id(param), (full_name,))
                if preserve_aliases
                else (full_name,)
            )
            if replace_meta_params and param.is_meta:
                replacement = _materialize_parameter_like(param, tensor)
                for alias_name in aliases:
                    alias_module, alias_local_name = _lookup_module_and_name(
                        model, alias_name
                    )
                    alias_module._parameters[alias_local_name] = replacement
            else:
                param.data = tensor
            attached.update(aliases)
            continue
        buf = module._buffers.get(local_name)
        if buf is not None:
            replacement = _materialize_tensor_like(buf, tensor)
            aliases = (
                buffer_aliases.get(id(buf), (full_name,))
                if preserve_aliases
                else (full_name,)
            )
            for alias_name in aliases:
                alias_module, alias_local_name = _lookup_module_and_name(
                    model, alias_name
                )
                alias_module._buffers[alias_local_name] = replacement
            attached.update(aliases)
            continue
        unexpected.append(full_name)

    missing = tuple(sorted(known_names - attached - set(skip)))
    if skip_reserved_tensor_names:
        missing = tuple(
            name for name in missing if not _is_reserved_tensorcast_name(name)
        )
    result = AttachResult(
        attached=tuple(sorted(attached)),
        skipped=tuple(sorted(skipped)),
        missing=tuple(sorted(missing)),
        unexpected=tuple(sorted(unexpected)),
    )
    if fail_on_unexpected and result.unexpected:
        raise RuntimeError(
            "TensorCast attach received unexpected tensor names: "
            f"{list(result.unexpected)}"
        )
    if fail_on_missing and result.missing:
        raise RuntimeError(
            "TensorCast attach missing required module tensor names: "
            f"{list(result.missing)}"
        )
    return result


def collect_module_tensors(
    model: nn.Module,
    *,
    exclude_names: Collection[str] = (),
    reject_reserved_tensor_names: bool = True,
    remove_duplicate: bool = False,
) -> dict[str, torch.Tensor]:
    excluded = {str(name) for name in exclude_names}
    tensors: dict[str, torch.Tensor] = {}
    seen_keys: set[int] = set()
    for name, param in model.named_parameters(remove_duplicate=False):
        _maybe_collect_tensor(
            tensors,
            str(name),
            param.data,
            excluded=excluded,
            dedupe_key=id(param),
            seen_keys=seen_keys,
            reject_reserved_tensor_names=reject_reserved_tensor_names,
            remove_duplicate=remove_duplicate,
        )
    for name, buf in model.named_buffers(remove_duplicate=False):
        if buf is None:
            continue
        _maybe_collect_tensor(
            tensors,
            str(name),
            buf,
            excluded=excluded,
            dedupe_key=id(buf),
            seen_keys=seen_keys,
            reject_reserved_tensor_names=reject_reserved_tensor_names,
            remove_duplicate=remove_duplicate,
        )
    if not tensors:
        raise RuntimeError(
            "TensorCast runtime binding requires at least one model tensor"
        )
    return tensors


def collect_module_tensor_names(model: nn.Module) -> set[str]:
    names = {str(name) for name, _ in model.named_parameters(remove_duplicate=False)}
    names.update(
        str(name)
        for name, buf in model.named_buffers(remove_duplicate=False)
        if buf is not None
    )
    return names


def align_runtime_binding_exclude_names(
    model: nn.Module,
    canonical_names: Iterable[str],
    set_excluded_names: Any,
    *,
    fail_on_missing: bool,
) -> int:
    canonical = {str(name) for name in canonical_names}
    all_names = collect_module_tensor_names(model)
    missing = sorted(canonical - all_names)
    if fail_on_missing and missing:
        sample = ", ".join(missing[:8])
        if len(missing) > 8:
            sample = f"{sample}, ..."
        raise RuntimeError(
            "TensorCast serving artifact tensor names are missing from "
            f"the model: missing_count={len(missing)} [{sample}]"
        )
    extra_excluded = sorted(all_names - canonical)
    if extra_excluded:
        set_excluded_names(model, extra_excluded)
    return len(extra_excluded)


def assert_runtime_tensors_match_expected_names(
    tensors: Mapping[str, torch.Tensor],
    expected_names: Iterable[str],
) -> None:
    runtime_names = {str(name) for name in tensors}
    expected = {str(name) for name in expected_names}
    missing = sorted(expected - runtime_names)
    unexpected = sorted(runtime_names - expected)
    if not missing and not unexpected:
        return
    missing_sample = ", ".join(missing[:8])
    unexpected_sample = ", ".join(unexpected[:8])
    if len(missing) > 8:
        missing_sample = f"{missing_sample}, ..."
    if len(unexpected) > 8:
        unexpected_sample = f"{unexpected_sample}, ..."
    raise RuntimeError(
        "TensorCast serving artifact tensor set mismatch: "
        f"missing_count={len(missing)} [{missing_sample}], "
        f"unexpected_count={len(unexpected)} [{unexpected_sample}]"
    )


def assert_module_tensors_are_meta(
    model: nn.Module,
    *,
    context: str,
) -> None:
    """Fail if a module skeleton already has materialized tensors."""

    materialized: list[str] = []
    total_bytes = 0
    for name, param in model.named_parameters(remove_duplicate=False):
        tensor = param.data
        if tensor.is_meta:
            continue
        materialized.append(f"{name}:{tensor.device}:{tuple(tensor.shape)}")
        total_bytes += tensor.numel() * tensor.element_size()
    for name, buf in model.named_buffers(remove_duplicate=False):
        if buf is None or buf.is_meta:
            continue
        materialized.append(f"{name}:{buf.device}:{tuple(buf.shape)}")
        total_bytes += buf.numel() * buf.element_size()
    if not materialized:
        return
    sample = ", ".join(materialized[:8])
    if len(materialized) > 8:
        sample = f"{sample}, ..."
    raise RuntimeError(
        f"{context} requires a meta model skeleton before TensorCast "
        f"binding creation; found {len(materialized)} materialized "
        f"tensors totaling {total_bytes} bytes [{sample}]"
    )


def allocate_unbound_module_tensors(
    model: nn.Module,
    tensor_names: Iterable[str],
    *,
    target_device: torch.device,
) -> dict[str, torch.Tensor]:
    allocated: dict[str, torch.Tensor] = {}
    parameter_aliases, buffer_aliases = _collect_alias_registrations(model)
    for full_name in sorted({str(name) for name in tensor_names}):
        module, local_name = _lookup_module_and_name(model, full_name)
        param = module._parameters.get(local_name)
        if param is not None:
            if not param.is_meta:
                allocated[full_name] = param.data
                continue
            tensor = _allocate_like_runtime_tensor(param.data, target_device)
            replacement = _materialize_parameter_like(param, tensor)
            for alias_name in parameter_aliases.get(id(param), (full_name,)):
                alias_module, alias_local_name = _lookup_module_and_name(
                    model, alias_name
                )
                alias_module._parameters[alias_local_name] = replacement
            allocated[full_name] = replacement.data
            continue
        buf = module._buffers.get(local_name)
        if buf is not None:
            if not buf.is_meta:
                allocated[full_name] = buf
                continue
            tensor = _allocate_like_runtime_tensor(buf, target_device)
            replacement = _materialize_tensor_like(buf, tensor)
            for alias_name in buffer_aliases.get(id(buf), (full_name,)):
                alias_module, alias_local_name = _lookup_module_and_name(
                    model, alias_name
                )
                alias_module._buffers[alias_local_name] = replacement
            allocated[full_name] = replacement
            continue
        raise RuntimeError(
            f"TensorCast fallback attach target '{full_name}' is missing"
        )
    return allocated


def compute_runtime_tensor_schema_hash(
    tensors: Mapping[str, torch.Tensor],
    *,
    remove_duplicate: bool = False,
) -> str:
    """Compute TensorCast runtime schema hash for a PyTorch tensor mapping."""

    return tc_contract.compute_runtime_tensor_schema_hash(
        tc_contract.collect_runtime_tensor_schema(
            tensors,
            remove_duplicate=remove_duplicate,
        )
    )


def snapshot_tensor_invariants(
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, TensorInvariant]:
    return {
        str(name): TensorInvariant(
            data_ptr=int(tensor.data_ptr()),
            shape=tuple(int(dim) for dim in tensor.shape),
            stride=tuple(int(dim) for dim in tensor.stride()),
            dtype=tensor.dtype,
        )
        for name, tensor in tensors.items()
    }


def validate_tensor_invariants(
    before: Mapping[str, TensorInvariant],
    after: Mapping[str, torch.Tensor],
) -> None:
    if set(before) != set(after):
        raise RuntimeError("TensorCast canonical tensor set changed")
    for name, invariant in before.items():
        tensor = after[name]
        after_invariant = TensorInvariant(
            data_ptr=int(tensor.data_ptr()),
            shape=tuple(int(dim) for dim in tensor.shape),
            stride=tuple(int(dim) for dim in tensor.stride()),
            dtype=tensor.dtype,
        )
        if after_invariant != invariant:
            raise RuntimeError(
                "TensorCast canonical tensor invariant changed for "
                f"{name}: before={invariant}, after={after_invariant}"
            )


class TorchModuleAdapterMixin:
    """Default PyTorch tensor binding operations for framework adapters."""

    def runtime_only_tensor_names(self, model: nn.Module) -> tuple[str, ...]:
        del model
        return ()

    def collect_runtime_binding_tensors(
        self,
        model: nn.Module,
        *,
        remove_duplicate: bool = False,
    ) -> dict[str, torch.Tensor]:
        return collect_module_tensors(
            model,
            exclude_names=self.runtime_only_tensor_names(model),
            reject_reserved_tensor_names=True,
            remove_duplicate=remove_duplicate,
        )

    def compute_runtime_tensor_schema_hash(
        self,
        tensors: Mapping[str, torch.Tensor],
        *,
        remove_duplicate: bool = False,
    ) -> str:
        return compute_runtime_tensor_schema_hash(
            tensors,
            remove_duplicate=remove_duplicate,
        )

    def attach_bound_tensors(
        self,
        model: nn.Module,
        tensors: Mapping[str, torch.Tensor],
        *,
        replace_meta_params: bool = False,
    ) -> AttachResult:
        return attach_tensors_to_module(
            model,
            tensors,
            replace_meta_params=replace_meta_params,
            skip_reserved_tensor_names=True,
            fail_on_missing=False,
            fail_on_unexpected=True,
        )

    def allocate_runtime_only_tensors(
        self,
        model: nn.Module,
        target_device: torch.device,
    ) -> dict[str, torch.Tensor]:
        return allocate_unbound_module_tensors(
            model,
            self.runtime_only_tensor_names(model),
            target_device=target_device,
        )

    def snapshot_tensor_invariants(
        self,
        tensors: Mapping[str, torch.Tensor],
    ) -> dict[str, TensorInvariant]:
        return snapshot_tensor_invariants(tensors)

    def validate_tensor_invariants(
        self,
        before: Mapping[str, TensorInvariant],
        after: Mapping[str, torch.Tensor],
    ) -> None:
        validate_tensor_invariants(before, after)


def _maybe_collect_tensor(
    tensors: dict[str, torch.Tensor],
    name: str,
    tensor: torch.Tensor,
    *,
    excluded: set[str],
    dedupe_key: int,
    seen_keys: set[int],
    reject_reserved_tensor_names: bool,
    remove_duplicate: bool,
) -> None:
    if name in excluded:
        return
    if reject_reserved_tensor_names and _is_reserved_tensorcast_name(name):
        raise RuntimeError(
            f"Model tensor name '{name}' collides with TensorCast reserved names"
        )
    if remove_duplicate:
        if dedupe_key in seen_keys:
            return
        seen_keys.add(dedupe_key)
    tensors[name] = tensor


def _module_tensor_names(model: nn.Module) -> set[str]:
    names = {str(name) for name, _ in model.named_parameters(remove_duplicate=False)}
    names.update(
        str(name)
        for name, buf in model.named_buffers(remove_duplicate=False)
        if buf is not None
    )
    return names


def _lookup_module_and_name(
    root: nn.Module, qualified_name: str
) -> tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    module = root
    for part in parts[:-1]:
        module = getattr(module, part)
    return module, parts[-1]


def _make_parameter_like(param: nn.Parameter, tensor: torch.Tensor) -> nn.Parameter:
    new_param = torch.Tensor._make_subclass(
        type(param), tensor.detach(), param.requires_grad
    )
    if hasattr(param, "__dict__"):
        new_param.__dict__.update(param.__dict__)
    return new_param


def _swap_tensor_in_place(dst: torch.Tensor, src: torch.Tensor) -> bool:
    swap_tensors = getattr(torch.utils, "swap_tensors", None)
    if not callable(swap_tensors):
        return False
    swap_tensors(dst, src)
    return True


def _materialize_parameter_like(
    param: nn.Parameter, tensor: torch.Tensor
) -> nn.Parameter:
    replacement = _make_parameter_like(param, tensor)
    if param.is_meta and _swap_tensor_in_place(param, replacement):
        return param
    return replacement


def _materialize_tensor_like(
    old_tensor: torch.Tensor, new_tensor: torch.Tensor
) -> torch.Tensor:
    if old_tensor.is_meta:
        try:
            if _swap_tensor_in_place(old_tensor, new_tensor.detach()):
                return old_tensor
        except RuntimeError:
            return new_tensor
    return new_tensor


def _collect_alias_registrations(
    model: nn.Module,
) -> tuple[dict[int, tuple[str, ...]], dict[int, tuple[str, ...]]]:
    parameter_aliases: dict[int, list[str]] = {}
    buffer_aliases: dict[int, list[str]] = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        parameter_aliases.setdefault(id(param), []).append(str(name))
    for name, buf in model.named_buffers(remove_duplicate=False):
        if buf is not None:
            buffer_aliases.setdefault(id(buf), []).append(str(name))
    return (
        {key: tuple(names) for key, names in parameter_aliases.items()},
        {key: tuple(names) for key, names in buffer_aliases.items()},
    )


def _allocate_like_runtime_tensor(
    tensor: torch.Tensor, device: torch.device
) -> torch.Tensor:
    return torch.empty_strided(
        size=tuple(int(dim) for dim in tensor.shape),
        stride=tuple(int(dim) for dim in tensor.stride()),
        dtype=tensor.dtype,
        device=device,
    )


def _is_reserved_tensorcast_name(name: str) -> bool:
    return str(name).startswith(_RESERVED_TENSORCAST_PREFIX)


__all__ = [
    "AttachResult",
    "TensorInvariant",
    "TorchModuleAdapterMixin",
    "allocate_unbound_module_tensors",
    "align_runtime_binding_exclude_names",
    "assert_module_tensors_are_meta",
    "assert_runtime_tensors_match_expected_names",
    "attach_tensors_to_module",
    "collect_module_tensor_names",
    "collect_module_tensors",
    "compute_runtime_tensor_schema_hash",
    "snapshot_tensor_invariants",
    "validate_tensor_invariants",
]
