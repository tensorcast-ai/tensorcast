#  Copyright (c) 2025-2026, TensorCast Team.

"""Build configuration validation for tensorcast.

This module provides utilities to detect and report build configuration
mismatches between the tensorcast._C extension and the runtime environment.
"""

from __future__ import annotations

import os
import warnings


class BuildConfigMismatchError(RuntimeError):
    """Raised when the build configuration doesn't match the runtime environment."""

    pass


def validate_cuda_backend_consistency() -> None:
    """Validate that the CUDA backend configuration is consistent.

    This function checks if the runtime is configured to use the fake CUDA
    backend but PyTorch has real CUDA available. This mismatch causes
    subtle runtime failures (e.g., "Pointer not found in allocations") because
    the fake backend doesn't track allocations made by PyTorch's real CUDA allocator.

    Raises:
        BuildConfigMismatchError: If fake CUDA extension is used with real PyTorch CUDA.

    Environment Variables:
        TENSORCAST_SKIP_CUDA_VALIDATION: Set to "1" to skip this validation.
    """
    # Allow users to skip validation if they know what they're doing
    if os.environ.get("TENSORCAST_SKIP_CUDA_VALIDATION") == "1":
        return

    try:
        import torch

        from tensorcast._C import is_fake_cuda
    except ImportError:
        # If we can't import, skip validation
        return

    extension_uses_fake_cuda = is_fake_cuda()
    pytorch_has_real_cuda = torch.cuda.is_available()

    if extension_uses_fake_cuda and pytorch_has_real_cuda:
        error_msg = (
            "\n"
            "=" * 72 + "\n"
            "TENSORCAST BUILD CONFIGURATION MISMATCH DETECTED\n"
            "=" * 72 + "\n"
            "\n"
            "The runtime is configured to use the Fake CUDA backend,\n"
            "but PyTorch has real CUDA available.\n"
            "\n"
            "This will cause runtime failures when using CUDA tensors because:\n"
            "  - The fake CUDA backend only tracks its own allocations\n"
            "  - PyTorch CUDA tensors are allocated by the real CUDA runtime\n"
            "  - IPC handle operations will fail with 'Pointer not found'\n"
            "\n"
            "To fix this, unset the fake backend override and run with real CUDA:\n"
            "\n"
            "    unset TENSORCAST_CUDA_BACKEND\n"
            "\n"
            "Or, if you intentionally want to use fake CUDA (CPU-only testing),\n"
            "set TENSORCAST_SKIP_CUDA_VALIDATION=1 to suppress this error.\n"
            "=" * 72
        )
        raise BuildConfigMismatchError(error_msg)

    # Also warn if using fake CUDA without real CUDA (informational)
    if extension_uses_fake_cuda and not pytorch_has_real_cuda:
        warnings.warn(
            "tensorcast._C is using the fake CUDA backend. "
            "GPU operations will be simulated on CPU.",
            UserWarning,
            stacklevel=2,
        )
