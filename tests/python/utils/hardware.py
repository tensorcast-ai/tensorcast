#  Copyright (c) 2025-2026, TensorCast Team.

"""Hardware capability helpers for tests."""

from __future__ import annotations

import os

import torch


def _prepare_fake_cuda_env() -> None:
    if os.environ.get("TENSORCAST_CUDA_BACKEND") == "fake":
        os.environ.setdefault("PYTEST_CURRENT_TEST", "tensorcast_pytest_session")
        os.environ.setdefault("TEST_TMPDIR", "/tmp/tensorcast_pytest")


def is_fake_cuda_backend() -> bool:
    _prepare_fake_cuda_env()
    from tensorcast._C import is_fake_cuda

    return bool(is_fake_cuda())


def has_cuda_or_fake() -> bool:
    """Return True if either real CUDA is available or the fake backend is selected.

    The fake backend is enabled in tests via `TENSORCAST_CUDA_BACKEND=fake`.
    This helper is also responsible for ensuring the native extension is
    importable. Some environments report `torch.cuda.is_available() == True`
    but lack runtime CUDA libraries required by `tensorcast._C`, in which case
    CUDA-dependent tests must be skipped.
    """
    if is_fake_cuda_backend():
        return True
    return bool(torch.cuda.is_available())


def synchronize_cuda(device: int | str | torch.device | None = None) -> None:
    if is_fake_cuda_backend():
        return
    if not torch.cuda.is_available():
        return
    if device is None:
        torch.cuda.synchronize()
        return
    torch.cuda.synchronize(device)
