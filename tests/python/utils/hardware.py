#  Copyright (c) 2025, TensorCast Team.

"""Hardware capability helpers for tests."""

from __future__ import annotations

import torch


def has_cuda_or_fake() -> bool:
    """Return True if either real CUDA is available or the fake backend is built."""
    try:
        from tensorcast._C import is_fake_cuda
    except Exception:  # noqa: BLE001
        is_fake = False
    else:
        is_fake = bool(is_fake_cuda())
    return bool(torch.cuda.is_available() or is_fake)
