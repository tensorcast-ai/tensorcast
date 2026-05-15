#  Copyright (c) 2026, TensorCast Team.

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tensorcast.api.store.types import CanonicalIndex, CanonicalIndexEntry
from tensorcast.serving.builder.source_catalog import (
    SourceManifest,
    source_catalog_from_all_safetensors_dir,
    source_catalog_from_canonical_index,
    source_catalog_from_manifest,
    source_catalog_from_selected_safetensors,
)


def test_source_catalog_from_selected_safetensors_ignores_unselected_files(
        tmp_path: Path) -> None:
    save_file({"a": torch.arange(4, dtype=torch.float32)},
              str(tmp_path / "a.safetensors"))
    save_file({"z": torch.arange(4, dtype=torch.float32)},
              str(tmp_path / "z.safetensors"))

    catalog = source_catalog_from_selected_safetensors(
        tmp_path,
        selected_files=["a.safetensors"],
        source_artifact_ref="mi2:test:source",
    )

    assert catalog.ordered_names == ("a", )
    assert tuple(catalog.meta_by_name) == ("a", )
    assert catalog.meta_by_name["a"].dtype == torch.float32
    assert catalog.meta_by_name["a"].shape == (4, )
    assert catalog.meta_by_name["a"].stride == (1, )
    assert catalog.meta_by_name["a"].storage_offset == 0
    assert catalog.selected_files[0].logical_name == "a.safetensors"
    assert catalog.source_artifact_ref == "mi2:test:source"


def test_source_catalog_requires_selected_files_and_rejects_duplicates(
        tmp_path: Path) -> None:
    save_file({"a": torch.arange(1)}, str(tmp_path / "a.safetensors"))

    with pytest.raises(ValueError, match="requires selected_files"):
        source_catalog_from_selected_safetensors(
            tmp_path,
            selected_files=[],
            source_artifact_ref="mi2:test:source",
        )

    with pytest.raises(ValueError, match="Duplicate selected source file"):
        source_catalog_from_selected_safetensors(
            tmp_path,
            selected_files=["a.safetensors", "a.safetensors"],
            source_artifact_ref="mi2:test:source",
        )


def test_source_catalog_preserves_selected_file_order(tmp_path: Path) -> None:
    save_file({"b": torch.arange(1)}, str(tmp_path / "b.safetensors"))
    save_file({"a": torch.arange(1)}, str(tmp_path / "a.safetensors"))

    catalog = source_catalog_from_selected_safetensors(
        tmp_path,
        selected_files=["b.safetensors", "a.safetensors"],
        source_artifact_ref="mi2:test:source",
    )

    assert tuple(entry.logical_name for entry in catalog.selected_files) == (
        "b.safetensors",
        "a.safetensors",
    )
    assert catalog.ordered_names == ("a", "b")


def test_source_catalog_manifest_round_trips_canonical_identity(
        tmp_path: Path) -> None:
    save_file({"a": torch.arange(4, dtype=torch.float16)},
              str(tmp_path / "a.safetensors"))
    catalog = source_catalog_from_selected_safetensors(
        tmp_path,
        selected_files=["a.safetensors"],
        source_artifact_ref="mi2:test:source",
    )

    roundtrip = source_catalog_from_manifest(
        SourceManifest(
            canonical_index_bytes=catalog.canonical_index_bytes,
            selected_files=catalog.selected_files,
        ),
        source_artifact_ref="mi2:test:source",
    )

    assert roundtrip.ordered_names == catalog.ordered_names
    assert roundtrip.canonical_index_hash == catalog.canonical_index_hash
    assert roundtrip.metadata_fingerprint == catalog.metadata_fingerprint


def test_source_catalog_from_canonical_index_preserves_stride_storage_offset(
) -> None:
    index = CanonicalIndex(
        entries=(CanonicalIndexEntry(
            name="view",
            dtype=torch.float32,
            shape=(2, 2),
            stride=(4, 1),
            storage_offset=3,
            segment_offset=0,
            size_bytes=16,
        ), ),
        total_size_bytes=16,
        avbs_hash="",
    )

    catalog = source_catalog_from_canonical_index(
        index,
        source_artifact_ref="mi2:test:source",
    )

    assert catalog.ordered_names == ("view", )
    assert catalog.meta_by_name["view"].stride == (4, 1)
    assert catalog.meta_by_name["view"].storage_offset == 3


def test_source_catalog_all_safetensors_dir_is_offline_helper(
        tmp_path: Path) -> None:
    save_file({"a": torch.arange(1)}, str(tmp_path / "a.safetensors"))
    save_file({"b": torch.arange(1)}, str(tmp_path / "b.safetensors"))

    catalog = source_catalog_from_all_safetensors_dir(
        tmp_path,
        source_artifact_ref="mi2:test:source",
    )

    assert catalog.ordered_names == ("a", "b")


def test_source_catalog_rejects_synthetic_source_identity(tmp_path: Path) -> None:
    save_file({"a": torch.arange(1)}, str(tmp_path / "a.safetensors"))

    with pytest.raises(ValueError, match="real source artifact identity"):
        source_catalog_from_selected_safetensors(
            tmp_path,
            selected_files=["a.safetensors"],
            source_artifact_ref="disk:/tmp/model",
        )
