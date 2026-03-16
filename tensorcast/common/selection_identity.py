#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from tensorcast.proto.common.v1 import common_pb2

_BYTE_ARTIFACT_LAYOUT_PROFILE: bytes = b"tensorcast.byte_artifact.layout.v1\n"
_BYTE_ARTIFACT_SELECTION_PROFILE: bytes = b"tensorcast.byte_artifact.selection.v1\n"
_BYTE_ARTIFACT_LOGICAL_LAYOUT_HASH: bytes = hashlib.sha256(
    _BYTE_ARTIFACT_LAYOUT_PROFILE
).digest()
_BYTE_ARTIFACT_SELECTION_HASH: bytes = hashlib.sha256(
    _BYTE_ARTIFACT_SELECTION_PROFILE
).digest()


class SelectionIdentity(BaseModel):
    model_config = ConfigDict(frozen=True)

    artifact_id: str
    logical_layout_hash: bytes
    selection_hash: bytes


def compute_logical_layout_hash(*, index_bytes: bytes, needs_view_index: bool) -> bytes:
    digest = hashlib.sha256()
    digest.update(index_bytes)
    digest.update(b"|view" if needs_view_index else b"|canonical")
    return digest.digest()


def compute_view_subset_hash(tensor_names: Sequence[str]) -> bytes:
    names = sorted({str(name) for name in tensor_names})
    if not names:
        return b""
    payload = json.dumps(names, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).digest()


def compute_selection_hash(*, view_id: str, view_subset_hash: bytes | None) -> bytes:
    if view_subset_hash == b"":
        view_subset_hash = None
    digest = hashlib.sha256()
    digest.update(view_id.encode("utf-8"))
    if view_subset_hash is not None:
        digest.update(view_subset_hash)
    else:
        digest.update(b"|all")
    digest.update(b"|v1")
    return digest.digest()


def compute_byte_artifact_logical_layout_hash() -> bytes:
    return _BYTE_ARTIFACT_LOGICAL_LAYOUT_HASH


def compute_byte_artifact_selection_hash() -> bytes:
    return _BYTE_ARTIFACT_SELECTION_HASH


def build_selection_identity(
    selection: common_pb2.ArtifactSelection,
) -> SelectionIdentity:
    if not selection.artifact_id:
        raise ValueError("selection.artifact_id is required")
    if not selection.logical_layout_hash:
        raise ValueError("selection.logical_layout_hash is required")
    if not selection.selection_hash:
        raise ValueError("selection.selection_hash is required")
    return SelectionIdentity(
        artifact_id=str(selection.artifact_id),
        logical_layout_hash=bytes(selection.logical_layout_hash),
        selection_hash=bytes(selection.selection_hash),
    )


__all__ = [
    "SelectionIdentity",
    "build_selection_identity",
    "compute_byte_artifact_logical_layout_hash",
    "compute_byte_artifact_selection_hash",
    "compute_logical_layout_hash",
    "compute_selection_hash",
    "compute_view_subset_hash",
]
