#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

from collections.abc import Mapping

import torch


def collect_storage_bases(target: Mapping[str, torch.Tensor]) -> dict[int, int]:
    """Return mapping of storage base pointer to storage nbytes.

    The base pointer is computed as data_ptr - storage_offset_bytes so it matches
    the underlying storage allocation for contiguous tensors.
    """
    bases: dict[int, int] = {}
    for tensor in target.values():
        storage = (
            tensor.untyped_storage()
            if hasattr(tensor, "untyped_storage")
            else tensor.storage()
        )
        storage_nbytes = int(storage.nbytes())
        storage_offset_bytes = int(tensor.storage_offset()) * int(tensor.element_size())
        base_ptr = int(tensor.data_ptr()) - storage_offset_bytes
        existing = bases.get(base_ptr)
        if existing is None or storage_nbytes > existing:
            bases[base_ptr] = storage_nbytes
    return bases


__all__ = ["collect_storage_bases"]
