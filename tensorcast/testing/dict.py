#  Copyright (c) 2025, TensorCast Team.

import torch


def assert_state_dict_equal(
    d1: dict[str, torch.Tensor], d2: dict[str, torch.Tensor]
) -> None:
    # Ensure key sets match exactly
    if d1.keys() != d2.keys():
        diff = d1.keys() ^ d2.keys()
        raise AssertionError(f"Key sets differ: {diff}")

    # Ensure tensors match exactly in shape, dtype, and values
    for name in d1:
        t1 = d1[name]
        t2 = d2[name]

        if t1.shape != t2.shape:
            raise AssertionError(
                f"Shape mismatch for '{name}': {t1.shape} vs {t2.shape}"
            )
        if t1.dtype != t2.dtype:
            # Align dtype to destination for strict equality check
            t1 = t1.to(dtype=t2.dtype)

        # Compare on the destination device to avoid host/device mismatch
        device = t2.device if t2.is_cuda else torch.device("cpu")
        t1c = t1.to(device=device).contiguous()
        t2c = t2.to(device=device).contiguous()

        if not torch.equal(t1c, t2c):
            raise AssertionError(f"Tensor '{name}' differs")
