#  Copyright (c) 2025, StepCast Team.

from __future__ import annotations

from pathlib import Path

__all__ = ["create_dummy_model"]


def create_dummy_model(storage_root: Path, model_name: str, size_bytes: int = 1 * 1024 * 1024) -> None:
    """Create a minimal on-disk representation of a model for unit-tests.

    The native C++ ``CheckpointStore`` expects each model to live in its own
    directory containing one or more *tensor.data_* shards.  For Python tests
    we only need a single, tiny shard so that the directory is considered a
    valid model location by the underlying implementation.

    Parameters
    ----------
    storage_root:
        Root directory configured for the local ``CheckpointStore`` instance.
    model_name:
        The logical model identifier (sub-directory name).
    size_bytes:
        Size of the dummy tensor file, defaults to 1 MiB – ample for allocation
        logic without impacting test runtime.
    """

    model_dir = storage_root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    tensor_file = model_dir / "tensor.data_0"
    # Keep the fixture idempotent – skip recreation if the file already exists.
    if tensor_file.exists():
        return

    # Generate deterministic, non-trivial content (repeating alphabet pattern).
    alphabet = b"abcdefghijklmnopqrstuvwxyz"
    remaining = size_bytes
    with tensor_file.open("wb") as fp:
        while remaining > 0:
            chunk = alphabet[: min(len(alphabet), remaining)]
            fp.write(chunk)
            remaining -= len(chunk)