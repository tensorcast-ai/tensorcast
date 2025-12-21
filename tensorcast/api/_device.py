#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import torch

from tensorcast._c_ext import get_device_uuid_map

from ._errors import DeviceMismatch


def resolve_device(device: int | torch.device) -> int:
    if device is None:
        raise DeviceMismatch("device is required")
    if isinstance(device, torch.device):
        if device.type == "cpu":
            raise DeviceMismatch("CPU device is not supported for this operation")
        return device.index if device.index is not None else 0
    return int(device)


def device_uuid_for(device_id: int) -> str:
    device_uuid_map = get_device_uuid_map()
    try:
        return device_uuid_map[device_id]
    except KeyError as e:  # pragma: no cover - defensive guard
        raise DeviceMismatch(
            f"Device ordinal {device_id} not found in daemon device map; available={list(device_uuid_map.keys())}"
        ) from e
