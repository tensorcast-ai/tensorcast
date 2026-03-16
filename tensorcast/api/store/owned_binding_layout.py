#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from typing import Sequence

from tensorcast.api.store.common import canonical_index_to_bytes
from tensorcast.api.store.types import CanonicalIndex, CanonicalIndexEntry
from tensorcast.common.selection_identity import compute_logical_layout_hash
from tensorcast.proto.daemon.v2 import store_daemon_pb2


@dataclass(frozen=True, slots=True)
class BindingLayout:
    binding_layout_id: str
    target_layout: store_daemon_pb2.TargetLayout
    target_index_bytes: bytes
    dst_specs: tuple[store_daemon_pb2.MappedTensorSpec, ...] = ()


OwnedBindingLayout = BindingLayout


def _multibase_multihash_sha256(digest: bytes) -> str:
    if len(digest) != 32:
        raise ValueError("SHA256 digest must be 32 bytes")
    mh = b"\x12\x20" + digest
    encoded = base64.b32encode(mh).decode("ascii").lower().rstrip("=")
    return f"b{encoded}"


def compute_binding_layout_id(
    *,
    target_layout: store_daemon_pb2.TargetLayout,
    target_index_bytes: bytes,
    dst_specs: Sequence[store_daemon_pb2.MappedTensorSpec] | None = None,
) -> str:
    digest = hashlib.sha256()
    digest.update(bytes(target_index_bytes))
    digest.update(target_layout.SerializeToString(deterministic=True))
    for spec in dst_specs or ():
        digest.update(spec.SerializeToString(deterministic=True))
    return f"bl1:{_multibase_multihash_sha256(digest.digest())}"


def build_binding_layout(
    *,
    target_layout: store_daemon_pb2.TargetLayout,
    target_index_bytes: bytes,
    dst_specs: Sequence[store_daemon_pb2.MappedTensorSpec] | None = None,
) -> BindingLayout:
    resolved_dst_specs = tuple(dst_specs or ())
    return BindingLayout(
        binding_layout_id=compute_binding_layout_id(
            target_layout=target_layout,
            target_index_bytes=target_index_bytes,
            dst_specs=resolved_dst_specs,
        ),
        target_layout=target_layout,
        target_index_bytes=bytes(target_index_bytes),
        dst_specs=resolved_dst_specs,
    )


def build_owned_layout(
    *,
    entries: Sequence[CanonicalIndexEntry],
    device_id: int,
    index_kind: store_daemon_pb2.TargetLayout.IndexKind,
    logical_layout_hash: bytes | None,
    view_id: str | None = None,
    ordered_names: Sequence[str] | None = None,
    dst_specs: Sequence[store_daemon_pb2.MappedTensorSpec] | None = None,
) -> BindingLayout:
    packed_entries = _pack_entries(entries, ordered_names=ordered_names)
    packed_index = CanonicalIndex(
        entries=tuple(packed_entries),
        total_size_bytes=sum(int(entry.size_bytes) for entry in packed_entries),
        avbs_hash="",
    )
    target_index_bytes = canonical_index_to_bytes(packed_index)
    target_layout = store_daemon_pb2.TargetLayout(
        layout_kind=store_daemon_pb2.TargetLayout.LAYOUT_KIND_COALESCED_UNSPECIFIED,
        index_kind=index_kind,
        tensor_spec_kind=store_daemon_pb2.TargetLayout.TENSOR_SPEC_KIND_OFFSETS,
        logical_layout_hash=bytes(
            logical_layout_hash
            if logical_layout_hash is not None
            else compute_logical_layout_hash(
                index_bytes=target_index_bytes,
                needs_view_index=(
                    index_kind == store_daemon_pb2.TargetLayout.INDEX_KIND_VIEW
                ),
            )
        ),
    )
    resolved_view_id = str(view_id or "").strip()
    if resolved_view_id:
        target_layout.view_id = resolved_view_id
    storage_id = "binding:coalesced:0"
    target_layout.storages.add(
        storage_id=storage_id,
        device_id=int(device_id),
        storage_length=int(packed_index.total_size_bytes),
        mapping_base_offset=0,
    )
    for entry in packed_entries:
        target_layout.offsets.add(
            name=entry.name,
            storage_id=storage_id,
            storage_offset=int(entry.segment_offset),
            logical_length=int(entry.size_bytes),
        )
    return build_binding_layout(
        target_layout=target_layout,
        target_index_bytes=target_index_bytes,
        dst_specs=tuple(dst_specs or ()),
    )


def build_mapped_tensor_spec(
    *,
    name: str,
    shape: Sequence[int],
    stride: Sequence[int],
    dtype: str,
    logical_length: int,
) -> store_daemon_pb2.MappedTensorSpec:
    spec = store_daemon_pb2.MappedTensorSpec(
        name=str(name),
        dtype=str(dtype),
        storage_offset=0,
        logical_length=int(logical_length),
    )
    spec.shape.extend(int(v) for v in shape)
    spec.stride.extend(int(v) for v in stride)
    return spec


def _pack_entries(
    entries: Sequence[CanonicalIndexEntry],
    ordered_names: Sequence[str] | None = None,
) -> tuple[CanonicalIndexEntry, ...]:
    if ordered_names is None:
        source_entries = list(entries)
    else:
        entry_by_name = {str(entry.name): entry for entry in entries}
        source_entries = [entry_by_name[str(name)] for name in ordered_names]
    packed: list[CanonicalIndexEntry] = []
    cursor = 0
    for entry in source_entries:
        packed.append(
            CanonicalIndexEntry(
                name=str(entry.name),
                dtype=entry.dtype,
                shape=tuple(int(v) for v in entry.shape),
                stride=tuple(int(v) for v in entry.stride),
                storage_offset=0,
                segment_offset=int(cursor),
                size_bytes=int(entry.size_bytes),
            )
        )
        cursor += int(entry.size_bytes)
    return tuple(packed)


__all__ = [
    "BindingLayout",
    "OwnedBindingLayout",
    "build_mapped_tensor_spec",
    "build_binding_layout",
    "build_owned_layout",
    "compute_binding_layout_id",
]
