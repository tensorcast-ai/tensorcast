#  Copyright (c) 2025, StepCast Team.

from __future__ import annotations

from pathlib import Path
import json

__all__ = ["create_dummy_artifact", "create_dummy_safetensors"]


def create_dummy_artifact(storage_root: Path, disk_path: str, size_bytes: int = 1 * 1024 * 1024) -> None:
    """Create a minimal on-disk representation of a artifact for unit-tests.

    The native C++ ``StoreEngine`` expects each artifact to live in its own
    directory containing one or more *tensor.data_* shards.  For Python tests
    we only need a single, tiny shard so that the directory is considered a
    valid artifact location by the underlying implementation.

    Parameters
    ----------
    storage_root:
        Root directory configured for the local ``StoreEngine`` instance.
    disk_path:
        Relative on-disk path used for legacy/disk-based loading (sub-directory name).
    size_bytes:
        Size of the dummy tensor file, defaults to 1 MiB – ample for allocation
        logic without impacting test runtime.
    """

    artifact_dir = storage_root / disk_path
    artifact_dir.mkdir(parents=True, exist_ok=True)

    tensor_file = artifact_dir / "tensor.data_0"
    # Generate deterministic, non-trivial content (repeating alphabet pattern).
    if not tensor_file.exists():
        alphabet = b"abcdefghijklmnopqrstuvwxyz"
        remaining = size_bytes
        with tensor_file.open("wb") as fp:
            while remaining > 0:
                chunk = alphabet[: min(len(alphabet), remaining)]
                fp.write(chunk)
                remaining -= len(chunk)

    # Write a minimal canonical tensor index required by the C++ DiskLoader
    index_path = artifact_dir / "tensor_index.json"
    if not index_path.exists():
        index_obj = {
            "__dummy__": [
                0,  # offset
                int(size_bytes),  # size
                [],  # shape
                [],  # stride
                "torch.uint8",  # dtype
                0,  # storage_offset
            ]
        }
        with index_path.open("w") as fp:
            json.dump(index_obj, fp)

    # Write a minimal artifact descriptor with required fields only
    descriptor_path = artifact_dir / "artifact_descriptor.json"
    # Always write/overwrite descriptor to ensure compliance with current loader rules
    # and avoid stale descriptors from previous runs
    descriptor = {
        # Per RFC-0007 and DiskLoader validation, artifact_id must be content-addressed (mi2:...)
        # Use fixed dummy multihashes suitable for tests.
        "artifact_id": "mi2:dummy_index:dummy_data",
        "index_multihash": "mh:dummy_index",
        "data_multihash": "mh:dummy_data",
        # Helpful extras for completeness
        "schema_version": "1",
        "encoding": "raw",
        "total_size": int(size_bytes),
    }
    with descriptor_path.open("w") as fp:
        json.dump(descriptor, fp)


def create_dummy_safetensors(storage_root: Path, disk_path: str) -> None:
    """Create a minimal .safetensors-based artifact directory for tests using the safetensors library."""
    from safetensors.torch import save_file
    import torch

    artifact_dir = storage_root / disk_path
    artifact_dir.mkdir(parents=True, exist_ok=True)

    path = artifact_dir / "weights.safetensors"
    if path.exists():
        return

    tensor = torch.arange(256, dtype=torch.uint8)
    save_file({"t": tensor}, str(path))

    # Provide a minimal tensor_index.json so pure-local loader can compute device offsets
    index_path = artifact_dir / "tensor_index.json"
    if not index_path.exists():
        import json
        entry = {
            # (offset, size, shape, stride, dtype, storage_offset)
            "t": [0, int(tensor.element_size() * tensor.numel()), [256], [1], str(tensor.dtype), 0],
        }
        with index_path.open("w") as fp:
            json.dump(entry, fp)