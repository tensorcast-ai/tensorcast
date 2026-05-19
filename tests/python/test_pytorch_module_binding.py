#  Copyright (c) 2026, TensorCast Team.

import pytest
import torch
from torch import nn

from tensorcast.pytorch.module_binding import (
    allocate_unbound_module_tensors,
    attach_tensors_to_module,
    collect_module_tensors,
)


class _TaggedParameter(nn.Parameter):
    pass


def test_attach_tensors_materializes_meta_parameter_aliases_and_subclass(
) -> None:

    class _Model(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.w = _TaggedParameter(
                torch.empty((2, ), device="meta", dtype=torch.float32),
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
                "b", torch.empty((2, ), device="meta", dtype=torch.float32))
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
    model.register_parameter("w", nn.Parameter(torch.zeros((1, ))))

    result = attach_tensors_to_module(
        model,
        {
            "w": torch.ones((1, )),
            "__tensorcast_meta__.manifest_json": torch.ones((4, ),
                                                           dtype=torch.uint8),
        },
        replace_meta_params=False,
        fail_on_missing=False,
    )

    assert result.attached == ("w", )
    assert result.skipped == ("__tensorcast_meta__.manifest_json", )
    assert torch.equal(model.w, torch.ones((1, )))


def test_attach_tensors_fail_closed_on_missing_and_unexpected_names() -> None:
    model = nn.Module()
    model.register_parameter("w", nn.Parameter(torch.zeros((1, ))))
    model.register_buffer("b", torch.zeros((1, )))

    with pytest.raises(RuntimeError, match="missing required"):
        attach_tensors_to_module(
            model,
            {"w": torch.ones((1, ))},
            replace_meta_params=False,
        )

    with pytest.raises(RuntimeError, match="unexpected tensor names"):
        attach_tensors_to_module(
            model,
            {"unexpected": torch.ones((1, ))},
            replace_meta_params=False,
            fail_on_missing=False,
        )


def test_collect_module_tensors_handles_excludes_reserved_and_duplicates(
) -> None:
    model = nn.Module()
    model.register_parameter(
        "w", nn.Parameter(torch.ones((1, ), dtype=torch.float32)))
    model.register_parameter("alias", model.w)
    model.register_parameter(
        "other_meta",
        nn.Parameter(torch.empty((1, ), device="meta", dtype=torch.float32)),
    )
    reserved = nn.Module()
    reserved.register_buffer("manifest_json", torch.ones((1, )))
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
                torch.empty_strided((2, 3), (3, 1),
                                    device="meta",
                                    dtype=torch.float16),
                requires_grad=False,
            )
            self.alias = self.w
            self.register_buffer(
                "b",
                torch.empty_strided((4, ), (1, ),
                                    device="meta",
                                    dtype=torch.float32),
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
    assert model.b.shape == (4, )
    assert model.b.dtype == torch.float32
    assert allocated["w"].data_ptr() == model.w.data.data_ptr()
    assert allocated["b"].data_ptr() == model.b.data_ptr()
