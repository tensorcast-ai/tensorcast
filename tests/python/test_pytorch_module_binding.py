#  Copyright (c) 2026, TensorCast Team.

from collections.abc import Mapping

import pytest
import torch
from torch import nn

from tensorcast.pytorch.module_binding import (
    TorchModuleAdapterMixin,
    align_runtime_binding_exclude_names,
    allocate_unbound_module_tensors,
    assert_module_tensors_are_meta,
    assert_runtime_tensors_match_expected_names,
    attach_tensors_to_module,
    collect_module_tensor_names,
    collect_module_tensors,
    compute_runtime_tensor_schema_hash,
    snapshot_tensor_invariants,
    validate_tensor_invariants,
)
from tensorcast.artifact_runtime.host import TorchTensorHost


class _TaggedParameter(nn.Parameter):
    pass


def test_attach_tensors_materializes_meta_parameter_aliases_and_subclass() -> None:
    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = _TaggedParameter(
                torch.empty((2,), device="meta", dtype=torch.float32),
                requires_grad=False,
            )
            self.alias = self.w
            self.captured = [self.w]

    model = _Model()
    original = model.w
    bound = torch.tensor([1.0, 2.0], dtype=torch.float32)

    result = attach_tensors_to_module(
        model,
        {"w": bound},
        replace_meta_params=True,
    )

    assert result.attached == ("alias", "w")
    assert result.missing == ()
    assert model.w is model.alias
    assert model.w is original
    assert model.captured[0] is original
    assert isinstance(model.w, _TaggedParameter)
    assert not model.w.is_meta
    assert model.w.data_ptr() == bound.data_ptr()
    assert torch.equal(model.w, bound)


def test_attach_tensors_materializes_meta_buffer_aliases() -> None:
    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer(
                "b", torch.empty((2,), device="meta", dtype=torch.float32)
            )
            self.register_buffer("alias_b", self.b)
            self.captured = [self.b]

    model = _Model()
    original = model.b
    bound = torch.tensor([3.0, 4.0], dtype=torch.float32)

    result = attach_tensors_to_module(
        model,
        {"b": bound},
        replace_meta_params=True,
    )

    assert result.attached == ("alias_b", "b")
    assert result.missing == ()
    assert model.b is model.alias_b
    assert model.b is original
    assert model.captured[0] is original
    assert not model.b.is_meta
    assert model.b.data_ptr() == bound.data_ptr()
    assert torch.equal(model.b, bound)


def test_attach_tensors_skips_reserved_tensorcast_names() -> None:
    model = nn.Module()
    model.register_parameter("w", nn.Parameter(torch.zeros((1,))))

    result = attach_tensors_to_module(
        model,
        {
            "w": torch.ones((1,)),
            "__tensorcast_meta__.manifest_json": torch.ones((4,), dtype=torch.uint8),
        },
        replace_meta_params=False,
        fail_on_missing=False,
    )

    assert result.attached == ("w",)
    assert result.skipped == ("__tensorcast_meta__.manifest_json",)
    assert torch.equal(model.w, torch.ones((1,)))


def test_attach_tensors_fail_closed_on_missing_and_unexpected_names() -> None:
    model = nn.Module()
    model.register_parameter("w", nn.Parameter(torch.zeros((1,))))
    model.register_buffer("b", torch.zeros((1,)))

    with pytest.raises(RuntimeError, match="missing required"):
        attach_tensors_to_module(
            model,
            {"w": torch.ones((1,))},
            replace_meta_params=False,
        )

    with pytest.raises(RuntimeError, match="unexpected tensor names"):
        attach_tensors_to_module(
            model,
            {"unexpected": torch.ones((1,))},
            replace_meta_params=False,
            fail_on_missing=False,
        )


def test_collect_module_tensors_handles_excludes_reserved_and_duplicates() -> None:
    model = nn.Module()
    model.register_parameter("w", nn.Parameter(torch.ones((1,), dtype=torch.float32)))
    model.register_parameter("alias", model.w)
    model.register_parameter(
        "other_meta",
        nn.Parameter(torch.empty((1,), device="meta", dtype=torch.float32)),
    )
    reserved = nn.Module()
    reserved.register_buffer("manifest_json", torch.ones((1,)))
    model.add_module("__tensorcast_meta__", reserved)

    with pytest.raises(RuntimeError, match="reserved names"):
        collect_module_tensors(model)

    tensors = collect_module_tensors(
        model,
        exclude_names={"__tensorcast_meta__.manifest_json"},
        remove_duplicate=False,
    )
    assert tuple(tensors) == ("w", "alias", "other_meta")

    deduped = collect_module_tensors(
        model,
        exclude_names={"__tensorcast_meta__.manifest_json"},
        remove_duplicate=True,
    )
    assert tuple(deduped) == ("w", "other_meta")


def test_allocate_unbound_module_tensors_materializes_aliases() -> None:
    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(
                torch.empty_strided((2, 3), (3, 1), device="meta", dtype=torch.float16),
                requires_grad=False,
            )
            self.alias = self.w
            self.register_buffer(
                "b",
                torch.empty_strided((4,), (1,), device="meta", dtype=torch.float32),
            )
            self.register_buffer("alias_b", self.b)

    model = _Model()

    allocated = allocate_unbound_module_tensors(
        model,
        ["w", "b"],
        target_device=torch.device("cpu"),
    )

    assert set(allocated) == {"w", "b"}
    assert model.w is model.alias
    assert model.b is model.alias_b
    assert not model.w.is_meta
    assert not model.b.is_meta
    assert model.w.shape == (2, 3)
    assert model.w.stride() == (3, 1)
    assert model.w.dtype == torch.float16
    assert model.b.shape == (4,)
    assert model.b.dtype == torch.float32
    assert allocated["w"].data_ptr() == model.w.data.data_ptr()
    assert allocated["b"].data_ptr() == model.b.data_ptr()


def test_align_and_assert_runtime_tensor_names() -> None:
    model = nn.Module()
    model.register_parameter("w", nn.Parameter(torch.ones((1,))))
    model.register_buffer("runtime_only", torch.ones((1,)))
    captured: list[tuple[str, ...]] = []

    count = align_runtime_binding_exclude_names(
        model,
        {"w"},
        lambda _model, names: captured.append(tuple(names)),
        fail_on_missing=True,
    )

    assert count == 1
    assert captured == [("runtime_only",)]
    assert collect_module_tensor_names(model) == {"w", "runtime_only"}
    assert_runtime_tensors_match_expected_names({"w": model.w}, {"w"})
    with pytest.raises(RuntimeError, match="tensor set mismatch"):
        assert_runtime_tensors_match_expected_names({"w": model.w}, {"missing"})


def test_assert_module_tensors_are_meta_reports_materialized_tensors() -> None:
    meta_model = nn.Module()
    meta_model.register_parameter(
        "w",
        nn.Parameter(torch.empty((1,), device="meta")),
    )
    assert_module_tensors_are_meta(meta_model, context="test context")

    materialized = nn.Module()
    materialized.register_parameter("w", nn.Parameter(torch.ones((2,))))
    with pytest.raises(RuntimeError, match="test context"):
        assert_module_tensors_are_meta(materialized, context="test context")


def test_runtime_tensor_schema_hash_and_invariants() -> None:
    tensors = {"w": torch.ones((2,), dtype=torch.float32)}

    schema_hash = compute_runtime_tensor_schema_hash(tensors)
    before = snapshot_tensor_invariants(tensors)
    validate_tensor_invariants(before, tensors)

    assert schema_hash
    changed = {"w": torch.ones((3,), dtype=torch.float32)}
    with pytest.raises(RuntimeError, match="invariant changed"):
        validate_tensor_invariants(before, changed)


def test_torch_module_adapter_mixin_provides_default_binding_ops() -> None:
    class _Adapter(TorchModuleAdapterMixin):
        def runtime_only_tensor_names(self, model: nn.Module) -> tuple[str, ...]:
            del model
            return ("runtime_only",)

    model = nn.Module()
    model.register_parameter(
        "w",
        nn.Parameter(torch.ones((1,), dtype=torch.float32), requires_grad=False),
    )
    model.register_buffer(
        "runtime_only",
        torch.empty((1,), device="meta", dtype=torch.float32),
    )

    adapter = _Adapter()
    tensors = adapter.collect_runtime_binding_tensors(model)
    assert tuple(tensors) == ("w",)
    assert adapter.compute_runtime_tensor_schema_hash(tensors)

    bound = torch.tensor([2.0], dtype=torch.float32)
    adapter.attach_bound_tensors(model, {"w": bound})
    allocated = adapter.allocate_runtime_only_tensors(
        model,
        torch.device("cpu"),
    )

    assert torch.equal(model.w, bound)
    assert set(allocated) == {"runtime_only"}
    invariants = adapter.snapshot_tensor_invariants({"w": model.w})
    adapter.validate_tensor_invariants(invariants, {"w": model.w})


def test_torch_module_adapter_mixin_rehydrates_runtime_only_tensors() -> None:
    class _Adapter(TorchModuleAdapterMixin):
        def runtime_only_tensor_names(self, model: nn.Module) -> tuple[str, ...]:
            del model
            return ("runtime_only",)

        def rehydrate_runtime_only_tensors(
            self,
            model: nn.Module,
            allocated: Mapping[str, torch.Tensor],
            target_device: torch.device,
        ) -> Mapping[str, torch.Tensor]:
            assert set(allocated) == {"runtime_only"}
            tensor = torch.full((2,), 7.0, device=target_device)
            model._buffers["runtime_only"] = tensor
            return {"runtime_only": tensor}

    model = nn.Module()
    model.register_buffer(
        "runtime_only",
        torch.empty((2,), device="meta", dtype=torch.float32),
    )

    allocated = _Adapter().allocate_runtime_only_tensors(
        model,
        torch.device("cpu"),
    )

    assert torch.equal(model.runtime_only, torch.full((2,), 7.0))
    assert torch.equal(allocated["runtime_only"], torch.full((2,), 7.0))


def test_torch_tensor_host_rehydrates_runtime_only_tensors() -> None:
    class _Surface(TorchTensorHost):
        def runtime_only_tensor_names(self, model: object) -> tuple[str, ...]:
            del model
            return ("runtime_only",)

        def rehydrate_runtime_only_tensors(
            self,
            model: object,
            allocated: Mapping[str, object],
            target_device: object,
        ) -> Mapping[str, object]:
            del allocated
            tensor = torch.full((2,), 11.0, device=target_device)
            model._buffers["runtime_only"] = tensor
            return {"runtime_only": tensor}

    model = nn.Module()
    model.register_buffer(
        "runtime_only",
        torch.empty((2,), device="meta", dtype=torch.float32),
    )

    allocated = _Surface().allocate_runtime_only_tensors(
        model,
        torch.device("cpu"),
    )

    assert torch.equal(model.runtime_only, torch.full((2,), 11.0))
    assert torch.equal(allocated["runtime_only"], torch.full((2,), 11.0))
