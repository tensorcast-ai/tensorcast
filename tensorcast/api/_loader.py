#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

from pathlib import Path
from typing import Callable, cast

import torch

from tensorcast.api._config import DEFAULT_PINNED_TIMEOUT_MS
from tensorcast.api._device import device_uuid_for, resolve_device
from tensorcast.api._errors import DaemonUnavailable
from tensorcast.api._io_disk import load_dict_from_disk
from tensorcast.api._load_engine import (
    DaemonLoader,
    LoadHandle,
    prepare_artifact_layout,
)
from tensorcast.api._runtime import (
    apply_client_load_defaults_if_present,
    require_runtime,
)
from tensorcast.api._utils import new_uuid


def load_dict_sync(
    *,
    disk_path: str | Path | None = None,
    device_id: int | torch.device = 0,
    storage_path: str | Path | None = None,
    enable_verification: bool = True,
    pinned_allocation_timeout_ms: int = DEFAULT_PINNED_TIMEOUT_MS,
) -> dict[str, torch.Tensor]:
    runtime = require_runtime()

    device_id_int = resolve_device(device_id)
    if disk_path is None:
        raise ValueError("disk_path must be provided")

    artifact_dir = Path(str(disk_path))
    tensor_meta_index, tensor_device_offsets = prepare_artifact_layout(
        artifact_dir, device_id=device_id_int
    )

    if not torch.cuda.is_available():
        return load_dict_from_disk(artifact_dir, device_id=device_id_int)

    (
        pinned_allocation_timeout_ms,
        enable_verification,
        _wait_for_completion,
    ) = apply_client_load_defaults_if_present(
        pinned_allocation_timeout_ms,
        enable_verification,
        True,
        runtime_address=runtime.address,
    )

    loader = DaemonLoader(
        runtime=runtime,
        disk_path=str(disk_path),
        replica_uuid=new_uuid(),
        device_uuid=device_uuid_for(device_id_int),
        device_id=device_id_int,
        artifact_dir=artifact_dir,
        tensor_meta_index=tensor_meta_index,
        tensor_device_offsets=tensor_device_offsets,
        pinned_allocation_timeout_ms=pinned_allocation_timeout_ms,
        wait_for_completion=True,
        enable_verification=enable_verification,
    )

    try:
        result = loader.load()
    except RuntimeError as exc:
        if "Local StoreDaemon" in str(exc) or "not available" in str(exc):
            return load_dict_from_disk(artifact_dir, device_id=device_id_int)
        raise DaemonUnavailable(str(exc)) from exc

    if isinstance(result, tuple):
        state, _ = result
        return state
    return result


def load_dict_async(
    disk_path: str | Path | None = None,
    device_id: int | torch.device = 0,
    storage_path: str | Path | None = None,
    enable_verification: bool = True,
    pinned_allocation_timeout_ms: int = DEFAULT_PINNED_TIMEOUT_MS,
) -> LoadHandle:
    runtime = require_runtime()

    device_id_int = resolve_device(device_id)
    if disk_path is None:
        raise ValueError("disk_path must be provided")

    artifact_dir = Path(str(disk_path))
    tensor_meta_index, tensor_device_offsets = prepare_artifact_layout(
        artifact_dir, device_id=device_id_int
    )

    if not torch.cuda.is_available():
        state = load_dict_from_disk(artifact_dir, device_id=device_id_int)

        def _confirm() -> bool:
            return True

        return LoadHandle(state, _confirm)

    (
        pinned_allocation_timeout_ms,
        enable_verification,
        _wait_for_completion,
    ) = apply_client_load_defaults_if_present(
        pinned_allocation_timeout_ms,
        enable_verification,
        False,
        runtime_address=runtime.address,
    )

    loader = DaemonLoader(
        runtime=runtime,
        disk_path=str(disk_path),
        replica_uuid=new_uuid(),
        device_uuid=device_uuid_for(device_id_int),
        device_id=device_id_int,
        artifact_dir=artifact_dir,
        tensor_meta_index=tensor_meta_index,
        tensor_device_offsets=tensor_device_offsets,
        pinned_allocation_timeout_ms=pinned_allocation_timeout_ms,
        wait_for_completion=False,
        enable_verification=enable_verification,
    )

    state, confirm = cast(
        tuple[dict[str, torch.Tensor], Callable[[], bool]],
        loader.load(),
    )
    return LoadHandle(state, confirm)


__all__ = ["LoadHandle", "load_dict_async", "load_dict_sync"]
