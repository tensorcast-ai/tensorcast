#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import os

import torch

from tensorcast._c_ext import get_device_uuid_map

from ._errors import DeviceMismatch

CPU_DEVICE_ID = -1


def resolve_device(device: int | torch.device, *, allow_cpu: bool = False) -> int:
    if device is None:
        raise DeviceMismatch("device is required")
    if isinstance(device, torch.device):
        if device.type == "cpu":
            if allow_cpu:
                return CPU_DEVICE_ID
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


def protocol_device_id_for(device_id: int) -> int:
    """Map a process-local CUDA ordinal to the physical ordinal used on RPCs.

    TensorCast local processes may run with `CUDA_VISIBLE_DEVICES` set, in
    which case torch-visible ordinals are remapped (for example visible
    `cuda:0` may correspond to physical GPU1). Daemon RPCs validate target
    layouts against the daemon's full-device registry, so those protocol
    surfaces must use physical ordinals.
    """

    if device_id < 0:
        return CPU_DEVICE_ID

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or visible in {"-1", "none", "None"}:
        return int(device_id)

    tokens = [token.strip() for token in visible.split(",") if token.strip()]
    if device_id >= len(tokens):
        raise DeviceMismatch(
            "Visible CUDA device ordinal "
            f"{device_id} is out of range for CUDA_VISIBLE_DEVICES={visible!r}"
        )

    token = tokens[device_id]
    if token.isdigit():
        return int(token)
    return int(device_id)
