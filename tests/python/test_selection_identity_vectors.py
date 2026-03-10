#  Copyright (c) 2026, TensorCast Team.

from tensorcast.api.store.view_composer import compute_view_id
from tensorcast.common.selection_identity import (
    compute_selection_hash,
    compute_view_subset_hash,
)
from tensorcast.proto.common.v1 import common_pb2


def test_view_subset_hash_golden_vector() -> None:
    subset_hash = compute_view_subset_hash(["b", "a", "a"])
    assert subset_hash.hex() == "0473ef2dc0d324ab659d3580c1134e9d812035905c4781fdd6d529b0c6860e13"


def test_selection_hash_normalizes_empty_subset_to_none() -> None:
    hash_none = compute_selection_hash(view_id="", view_subset_hash=None)
    hash_empty = compute_selection_hash(view_id="", view_subset_hash=b"")
    assert hash_none.hex() == "9aa60740c2337b335b9279865963ff5385937c856dca7f7ebdfe49d1a27c8795"
    assert hash_none == hash_empty


def test_selection_hash_golden_vectors() -> None:
    subset_hash = compute_view_subset_hash(["b", "a", "a"])
    assert compute_selection_hash(view_id="", view_subset_hash=subset_hash).hex() == (
        "5edac2095f23abba792f8e98422b0e233e49a93ad736c7d64e23f573e43f9202"
    )
    assert compute_selection_hash(view_id="view-123", view_subset_hash=None).hex() == (
        "b9a1dff059db74a724ad7df7a72d5b7f524d5cc3725f057dddac302784d9fb8c"
    )
    assert compute_selection_hash(view_id="view-123", view_subset_hash=subset_hash).hex() == (
        "09c927caf75fcb752fa6efdf1eeaa3bb3167b6e1d5d2517923c860c2db9d98dd"
    )


def test_view_id_golden_vector() -> None:
    canonical_index_bytes = (
        b'{"a":[0,16,[2,2],[2,1],"torch.float32",0],"b":[16,16,[2,2],[2,1],"torch.float32",0]}'
    )
    view_spec = common_pb2.ViewSpec()
    view_spec.tensors["b"].ops.add().narrow.dim = 1
    view_spec.tensors["b"].ops[0].narrow.start = 0
    view_spec.tensors["b"].ops[0].narrow.length = 1
    view_spec.tensors["a"].ops.add().transpose.dim0 = 0
    view_spec.tensors["a"].ops[0].transpose.dim1 = 1
    view_id = compute_view_id(view_spec, canonical_index_bytes)
    assert view_id == "bciqd3js2h75e5gw6cagg5kp3d6xc7laeqca263wrnmirudisjlwuwwa"
