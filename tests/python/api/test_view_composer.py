#  Copyright (c) 2025, TensorCast Team.

import torch

from tensorcast.api.store.types import CanonicalIndex, CanonicalIndexEntry
from tensorcast.api.store.view_composer import ViewSpecComposer


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
