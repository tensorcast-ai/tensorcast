#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

from ._config import (
    GetArtifactOptions,
    PlanType,
    RegisterArtifactOptions,
    get_daemon_address,
    set_daemon_address,
)
from ._indices import build_indices_from_safetensors, calculate_tensor_device_offsets
from ._io_disk import load_dict_from_disk, save_dict
from ._loader import (
    get_artifact_async,
    get_artifact_sync,
    load_dict_async,
    load_dict_sync,
)
from ._register import (
    RegisteredArtifact,
    RegisteredLease,
    begin_register_artifact_sdk,
    register_artifact,
)

__all__ = [
    # Stable public API
    "load_dict_sync",
    "load_dict_async",
    "get_artifact_sync",
    "get_artifact_async",
    "save_dict",
    "load_dict_from_disk",
    "register_artifact",
    "begin_register_artifact_sdk",
    "RegisteredArtifact",
    "RegisteredLease",
    "set_daemon_address",
    "get_daemon_address",
    # Config/Options
    "PlanType",
    "RegisterArtifactOptions",
    "GetArtifactOptions",
    # Low-level helpers (used by tests/examples)
    "calculate_tensor_device_offsets",
    "build_indices_from_safetensors",
]
