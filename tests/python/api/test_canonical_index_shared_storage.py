#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import json

from tensorcast.api.store.common import canonical_index_from_bytes
from tensorcast.api.store.owned_binding_layout import build_owned_layout
from tensorcast.common.selection_contract import compute_selected_index_bytes
from tensorcast.proto.daemon.v2 import store_daemon_pb2


def _shared_storage_index_bytes() -> bytes:
    return json.dumps(
        {
            "alpha": [0, 1024, [2], [1], "torch.float32", 0],
            "beta": [0, 1024, [4], [1], "torch.uint8", 16],
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def test_canonical_index_total_uses_storage_extent_not_entry_sum() -> None:
    index = canonical_index_from_bytes(_shared_storage_index_bytes())

    assert index.total_size_bytes == 1024


def test_owned_layout_compacts_shared_storage_entries_by_tensor_span() -> None:
    index = canonical_index_from_bytes(_shared_storage_index_bytes())

    layout = build_owned_layout(
        entries=index.entries,
        device_id=0,
        index_kind=store_daemon_pb2.TargetLayout.INDEX_KIND_CANONICAL_UNSPECIFIED,
        logical_layout_hash=None,
    )

    storage = layout.target_layout.storages[0]
    offsets = {offset.name: offset for offset in layout.target_layout.offsets}
    assert storage.storage_length == 12
    assert offsets["alpha"].storage_offset == 0
    assert offsets["alpha"].logical_length == 8
    assert offsets["beta"].storage_offset == 8
    assert offsets["beta"].logical_length == 4


def test_subset_view_index_compacts_shared_storage_tensor_span() -> None:
    selected = compute_selected_index_bytes(
        canonical_index_bytes=_shared_storage_index_bytes(),
        view_spec=None,
        tensor_names=["beta"],
    )

    index = canonical_index_from_bytes(selected)
    assert index.total_size_bytes == 4
    [entry] = index.entries
    assert entry.name == "beta"
    assert entry.segment_offset == 0
    assert entry.size_bytes == 4
    assert entry.storage_offset == 0
