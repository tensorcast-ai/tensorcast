#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import json
import uuid
from pathlib import Path

from tensorcast._c_ext import inspect_or_generate_descriptor

from ._errors import DiskIndexMismatch, IndexParseError


def new_uuid() -> str:
    return str(uuid.uuid4())


def compose_artifact_dir(raw_disk_path: Path, storage_path: str | None) -> Path:
    if storage_path is None:
        return raw_disk_path
    s = str(storage_path)
    if s == "":
        return raw_disk_path
    return Path(s) / raw_disk_path


def ensure_artifact_descriptor_safe(artifact_dir: Path) -> None:
    try:
        _ = inspect_or_generate_descriptor(str(artifact_dir))
    except Exception as e:  # noqa: BLE001
        # Avoid raising up the stack; only warn via print to keep dependency light here
        print(f"[tensorcast.api] warn: ensure artifact_descriptor.json failed: {e}")


def validate_disk_index_matches(index_bytes: bytes, disk_path: str) -> None:
    p = Path(disk_path)
    index_file = p / "tensor_index.json"
    if not index_file.exists():
        raise DiskIndexMismatch(f"Disk index not found at {index_file}")
    try:
        disk_raw = index_file.read_bytes()
        disk_obj = json.loads(disk_raw)
        disk_canon = json.dumps(disk_obj, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    except Exception as e:  # noqa: BLE001
        raise IndexParseError(f"Failed to parse disk index at {index_file}: {e}") from e
    if disk_canon != index_bytes:
        raise DiskIndexMismatch(
            "Disk index does not match registration index (canonical JSON mismatch)"
        )
