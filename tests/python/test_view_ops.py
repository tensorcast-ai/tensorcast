#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import pytest

from tensorcast.api._view_ops import (
    NarrowOp,
    ResolvedViewInputs,
    TransposeOp,
    ViewSpecBuildResult,
    build_view_spec,
    validate_narrow,
    validate_transpose,
    _coerce_slice_spec,
)


def test_validate_narrow_normalizes_and_folds_identity() -> None:
    op = validate_narrow("weights", (10, 20), (1, slice(-5, None)))
    assert op == NarrowOp(dim=1, start=15, length=5)
    assert validate_narrow("weights", (8,), slice(None)) is None


def test_validate_narrow_rejects_invalid_params() -> None:
    with pytest.raises(ValueError, match="Slice step must be 1"):
        validate_narrow("weights", (8,), slice(None, None, 2))
    with pytest.raises(ValueError, match="out of range"):
        validate_narrow("weights", (4,), (1, slice(0, 2)))


def test_validate_transpose_canonicalizes_and_cancels() -> None:
    ops = validate_transpose("tensor", (2, 3, 4), [(0, 1), (0, 2)])
    assert [(op.dim0, op.dim1) for op in ops] == [(0, 2), (1, 2)]

    cancelled = validate_transpose("tensor", (2, 3), [(0, 1), (0, 1)])
    assert cancelled == []

    with pytest.raises(ValueError, match="out of range"):
        validate_transpose("tensor", (2, 3), [(0, 3)])


def test_build_view_spec_conflict_and_identity() -> None:
    entry_shapes = {"weights": (4,)}
    with pytest.raises(ValueError, match="Cannot apply slices and transpose"):
        build_view_spec(
            entry_shapes=entry_shapes,
            slices={"weights": slice(0, 1)},
            transpose={"weights": [(0, 0)]},
        )

    identity = build_view_spec(
        entry_shapes=entry_shapes,
        slices={"weights": slice(None)},
        transpose=None,
    )
    assert identity is ViewSpecBuildResult.identity()
    assert identity.proto is None
    assert identity.tensor_ops == {}


def test_build_view_spec_orders_tensor_ops() -> None:
    entry_shapes = {"b": (4,), "a": (2, 2)}
    result = build_view_spec(
        entry_shapes=entry_shapes,
        slices={"b": slice(0, 2)},
        transpose={"a": [(0, 1)]},
    )
    assert list(result.tensor_ops.keys()) == ["a", "b"]
    assert result.proto is not None
    assert result.tensor_ops["a"][0] == TransposeOp(dim0=0, dim1=1)
    assert result.tensor_ops["b"][0] == NarrowOp(dim=0, start=0, length=2)


def test_identity_result_is_read_only() -> None:
    identity = ViewSpecBuildResult.identity()
    with pytest.raises(TypeError):
        identity.tensor_ops["weights"] = []  # type: ignore[index]


def test_resolved_view_inputs_variants() -> None:
    build_result = build_view_spec(
        entry_shapes={"weights": (4,)},
        slices={"weights": slice(1, 3)},
        transpose=None,
    )
    resolved_build = ResolvedViewInputs.from_build_result(
        artifact_id="artifact",
        canonical_index_bytes=b"{}",
        build_result=build_result,
    )
    assert resolved_build.variant == "build"
    assert resolved_build.has_transpose is False
    assert resolved_build.view_spec is build_result.proto
    assert resolved_build.normalized_ops["weights"][0]["type"] == "narrow"

    resolved_id = ResolvedViewInputs.from_view_id(
        artifact_id="artifact",
        view_id="view-123",
    )
    assert resolved_id.variant == "id"
    assert resolved_id.has_transpose is False
    assert resolved_id.view_spec is None
    assert resolved_id.normalized_ops == {}

    with pytest.raises(ValueError):
        ResolvedViewInputs(
            artifact_id="artifact",
            canonical_index_bytes=None,
            build_result=None,
            disk_path_hint=None,
            view_id=None,
        )


def test_coerce_slice_spec_validates() -> None:
    assert _coerce_slice_spec((slice(0, 1),)) == slice(0, 1)
    assert _coerce_slice_spec(((1, slice(0, 1)),)) == (1, slice(0, 1))
    with pytest.raises(ValueError):
        _coerce_slice_spec(())
    with pytest.raises(ValueError):
        _coerce_slice_spec((object(),))
