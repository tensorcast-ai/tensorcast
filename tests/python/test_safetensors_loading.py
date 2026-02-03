#  Copyright (c) 2025-2026, TensorCast Team.

from pathlib import Path

import torch
from safetensors.torch import save_file as st_save

from tensorcast.api._indices import build_indices_from_safetensors
from tests.python.utils.artifact_utils import create_dummy_safetensors


def test_build_indices_from_safetensors_single_and_multi(tmp_path: Path):
    storage_root = tmp_path / "models"
    storage_root.mkdir(parents=True, exist_ok=True)

    # Artifact directory with a single weights.safetensors
    artifact_id = "st_simple"
    create_dummy_safetensors(storage_root, artifact_id)

    # Force safetensors header path by removing tensor_index.json if present
    index_file = storage_root / artifact_id / "tensor_index.json"
    if index_file.exists():
        index_file.unlink()

    # Validate our header parser produces indices without raising (single file)
    meta_idx, data_idx = build_indices_from_safetensors(storage_root / artifact_id)
    assert set(meta_idx.keys()) == {"t"}
    assert set(data_idx.keys()) == {"t"}
    shape, stride, dtype, storage_offset = meta_idx["t"]
    assert shape == [256]
    assert stride == [1]
    assert dtype == "torch.uint8"
    assert storage_offset == 0
    off, length = data_idx["t"]
    assert off == 0 and length == 256

    # Create a second safetensors to validate multi-file concatenation
    extra = torch.arange(16, dtype=torch.uint8)
    st_save({"u": extra}, str(storage_root / artifact_id / "extra.safetensors"))

    meta2, data2 = build_indices_from_safetensors(storage_root / artifact_id)
    assert set(meta2.keys()) == {"t", "u"}
    assert set(data2.keys()) == {"t", "u"}
    # Canonical offsets are coalesced by sorted tensor names (t then u), not file order.
    assert data2["t"] == (0, 256)
    assert data2["u"] == (256, 16)
