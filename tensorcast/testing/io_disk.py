#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from tensorcast.api._io_disk import load_dict_from_disk as _load_dict_from_disk
from tensorcast.api._io_disk import save_dict as _save_dict


def save_dict(
    state_dict: dict[str, torch.Tensor],
    disk_path: str,
    streaming_config: Mapping[str, Any] | None = None,
) -> dict:
    """Test-only helper: persist a state_dict to a local disk directory.

    This API is not supported for production flows; it exists for regression
    tests and offline fixtures.
    """

    return _save_dict(
        state_dict,
        disk_path,
        streaming_config=streaming_config,
        _unsafe_allow_local_disk_io=True,
    )


def load_dict_from_disk(
    disk_path: str,
    *,
    device_id: int | torch.device = 0,
) -> dict[str, torch.Tensor]:
    """Test-only helper: restore a state_dict from a local disk directory."""

    return _load_dict_from_disk(
        disk_path,
        device_id=device_id,
        _unsafe_allow_local_disk_io=True,
    )
