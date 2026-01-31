#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence


def compute_logical_layout_hash(*, index_bytes: bytes, needs_view_index: bool) -> bytes:
    digest = hashlib.sha256()
    digest.update(index_bytes)
    digest.update(b"|view" if needs_view_index else b"|canonical")
    return digest.digest()


def compute_view_subset_hash(tensor_names: Sequence[str]) -> bytes:
    names = sorted({str(name) for name in tensor_names})
    payload = json.dumps(names, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).digest()


def compute_selection_hash(*, view_id: str, view_subset_hash: bytes | None) -> bytes:
    digest = hashlib.sha256()
    digest.update(view_id.encode("utf-8"))
    if view_subset_hash is not None:
        digest.update(view_subset_hash)
    else:
        digest.update(b"|all")
    digest.update(b"|v1")
    return digest.digest()


__all__ = [
    "compute_logical_layout_hash",
    "compute_selection_hash",
    "compute_view_subset_hash",
]
