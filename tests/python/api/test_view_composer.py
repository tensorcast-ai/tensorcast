#  Copyright (c) 2025-2026, TensorCast Team.

import torch

from tensorcast.api._view_ops import NarrowOp, build_view_spec
from tensorcast.api.store.common import (
    canonical_index_from_bytes,
    canonical_index_to_bytes,
)
from tensorcast.api.store.types import CanonicalIndex, CanonicalIndexEntry
from tensorcast.api.store.view_composer import (
    ViewSpecComposer,
    _apply_view_ops,
    _compose_narrow,
)


def _index() -> CanonicalIndex:
    return CanonicalIndex(
        entries=(
            CanonicalIndexEntry(
                name="a",
                dtype=torch.float32,
                shape=(4, 4),
                stride=(4, 1),
                storage_offset=0,
                segment_offset=0,
                size_bytes=64,
            ),
            CanonicalIndexEntry(
                name="b",
                dtype=torch.float32,
                shape=(2, 2),
                stride=(2, 1),
                storage_offset=0,
                segment_offset=64,
                size_bytes=16,
            ),
        ),
        total_size_bytes=80,
        avbs_hash="",
    )


def test_view_composer_composes_narrow():
    composer = ViewSpecComposer()
    base_index = _index()

    parent = composer.compose(
        canonical_index=base_index,
        parent_spec=None,
        child_spec=composer.compose(
            canonical_index=base_index,
            parent_spec=None,
            child_spec=None,
            parent_depth=0,
            subset_names=None,
        )[0],
        parent_depth=0,
        subset_names=None,
    )[0]

    child_spec, cache, depth = composer.compose(
        canonical_index=base_index,
        parent_spec=parent,
        child_spec=composer.compose(
            canonical_index=base_index,
            parent_spec=None,
            child_spec=None,
            parent_depth=0,
            subset_names=None,
        )[0],
        parent_depth=0,
        subset_names=["a"],
    )

    assert depth <= 8
    assert cache is not None
    assert cache.tensor_names == ("a",)
    assert child_spec is None or child_spec.is_identity


def test_view_composer_transpose_hash_stable():
    composer = ViewSpecComposer()
    base_index = _index()
    spec, cache, _ = composer.compose(
        canonical_index=base_index,
        parent_spec=None,
        child_spec=None,
        parent_depth=0,
        subset_names=["a", "b"],
    )
    h1 = composer.hash_view_spec(spec, subset=("a", "b"))
    h2 = composer.hash_view_spec(spec, subset=("a", "b"))
    assert h1 == h2


def test_compose_narrow_collapses_parent_and_child():
    parent = NarrowOp(dim=1, start=1, length=2)
    child = NarrowOp(dim=1, start=1, length=1)
    ops = _compose_narrow(
        base_shape=(4, 4),
        parent=parent,
        child=child,
        tensor_name="a",
    )
    assert len(ops) == 1
    fused = ops[0]
    assert isinstance(fused, NarrowOp)
    assert fused.start == 2
    assert fused.length == 1


def test_compose_narrow_storage_offset_applies_once():
    base_entry = _index().entries[0]
    parent = NarrowOp(dim=0, start=2, length=2)
    child = NarrowOp(dim=0, start=1, length=1)
    ops = _compose_narrow(
        base_shape=base_entry.shape,
        parent=parent,
        child=child,
        tensor_name="a",
    )
    view_entry = _apply_view_ops(base_entry, ops)
    expected_offset = (parent.start + child.start) * base_entry.stride[0]
    assert view_entry.storage_offset == expected_offset


def test_compose_view_cache_uses_selected_index_bytes_semantics() -> None:
    composer = ViewSpecComposer()
    base_index = CanonicalIndex(
        entries=(
            CanonicalIndexEntry(
                name="w",
                dtype=torch.float32,
                shape=(4, 8),
                stride=(8, 1),
                storage_offset=0,
                segment_offset=0,
                size_bytes=128,
            ),
        ),
        total_size_bytes=128,
        avbs_hash="",
    )
    child_spec = build_view_spec(
        entry_shapes={"w": (4, 8)},
        slices={"w": (1, slice(0, 4))},
        transpose=None,
    )
    _, cache, _ = composer.compose(
        canonical_index=base_index,
        identity_index_bytes=canonical_index_to_bytes(base_index),
        parent_spec=None,
        child_spec=child_spec,
        parent_depth=0,
        subset_names=None,
    )

    assert cache is not None
    assert cache.selected_index is not None
    selected_index = canonical_index_from_bytes(cache.view_index_bytes)
    assert cache.selected_index == selected_index
    selected_entry = selected_index.entries[0]
    assert selected_entry.name == "w"
    assert selected_entry.shape == (4, 4)
    assert selected_entry.stride == (4, 1)
    assert selected_entry.storage_offset == 0
